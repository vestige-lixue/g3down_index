"""
Recursive Whisper transcription with resume support.

Walks <audioRoot>, transcribes every .wav file with faster-whisper, and writes a JSON
transcript per file under <jsonRoot> mirroring the input directory structure. The model
is loaded once for the whole run.

Usage:
    uv run transcribe_audio.py <audio_root> <json_root> [options]

Output JSON schema (camelCase, version 1):
    {
      "version": 1,
      "model": "large-v3",
      "language": "zh",
      "languageProbability": 0.99,
      "audioDurationSeconds": 1234.5,
      "transcriptionSeconds": 145.2,
      "audioRelativePath": "2024-08/japan/day3.wav",
      "segments": [
        {
          "id": 0, "start": 0.0, "end": 5.2,
          "text": " ...",
          "avgLogprob": -0.3, "noSpeechProb": 0.01, "compressionRatio": 1.5,
          "words": [
            {"start": 0.0, "end": 0.4, "word": " 京都", "probability": 0.95}, ...
          ]
        }, ...
      ]
    }
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, cast

# --- Tunables (CLI flags override) -------------------------------------------------------

# large-v3 is the most accurate; switch to "large-v3-turbo" for ~8x speed with small loss.
DEFAULT_MODEL: str = "large-v3"

# "zh" forces Chinese decoding. Use "auto" if your vlogs mix languages frequently.
# Forcing the right language is generally more reliable than auto-detection on long files.
DEFAULT_LANGUAGE: str = "zh"

DEFAULT_DEVICE: str = "cuda"

# int8_float16 fits comfortably in 8GB VRAM (4060 Laptop). Pure float16 needs ~6-8GB.
# If you hit OOM, drop to "int8" (CPU-quality on GPU; faster but slight accuracy loss).
DEFAULT_COMPUTE_TYPE: str = "int8_float16"

# Beam size 5 is faster-whisper's default. Drop to 1 if VRAM tight; raise rarely helps.
DEFAULT_BEAM_SIZE: int = 5

# VAD threshold: 0.5 is faster-whisper's default; we lower to 0.3 because quiet speech
# can be misclassified as silence. Raise back toward 0.5 if you see noise being transcribed
# as words. Speech is kept when probability of speech >= threshold.
DEFAULT_VAD_THRESHOLD: float = 0.3
DEFAULT_VAD_MIN_SILENCE_MS: int = 500

# Bump when JSON schema changes; downstream indexer should refuse mismatched versions.
JSON_SCHEMA_VERSION: int = 1

# Print progress every N seconds of audio processed (keeps long files visible).
PROGRESS_INTERVAL_SECONDS: float = 60.0


class TranscriptionWord(Protocol):
    start: float
    end: float
    word: str
    probability: float


class TranscriptionSegment(Protocol):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: list[TranscriptionWord] | None


class TranscriptionInfo(Protocol):
    language: str
    language_probability: float
    duration: float


class TranscriptionModel(Protocol):
    def transcribe(
        self,
        audio: str,
        *,
        language: str | None,
        word_timestamps: bool,
        vad_filter: bool,
        vad_parameters: dict[str, int | float],
        condition_on_previous_text: bool,
        beam_size: int,
    ) -> tuple[Iterable[TranscriptionSegment], TranscriptionInfo]: ...


class TranscriptionModelConstructor(Protocol):
    def __call__(
        self,
        modelName: str,
        *,
        device: str,
        compute_type: str,
    ) -> TranscriptionModel: ...


class TranscriptionWordJson(TypedDict):
    start: float
    end: float
    word: str
    probability: float


class TranscriptionSegmentJson(TypedDict):
    id: int
    start: float
    end: float
    text: str
    avgLogprob: float
    noSpeechProb: float
    compressionRatio: float
    words: list[TranscriptionWordJson]


class TranscriptionResult(TypedDict):
    version: int
    model: str
    language: str
    languageProbability: float
    audioDurationSeconds: float
    transcriptionSeconds: float
    audioRelativePath: str
    segments: list[TranscriptionSegmentJson]


@dataclass(frozen=True)
class TranscriptionTask:
    inputAudio: Path
    outputJson: Path


def collectTasks(audioRoot: Path, jsonRoot: Path) -> list[TranscriptionTask]:
    """Walk audioRoot, build a task per .wav, skip files with existing non-empty JSON."""
    tasks: list[TranscriptionTask] = []
    for audioPath in audioRoot.rglob("*.wav"):
        if not audioPath.is_file():
            continue
        # Defense in depth: skip any temp files an interrupted extraction may have left.
        if audioPath.name.endswith(".tmp.wav"):
            continue

        relativePath = audioPath.relative_to(audioRoot)
        jsonPath = (jsonRoot / relativePath).with_suffix(".json")

        # Resume: skip if transcript exists and is non-empty. (Empty/half-written means
        # a previous run died mid-write; we'll redo it.)
        if jsonPath.exists() and jsonPath.stat().st_size > 0:
            continue

        tasks.append(TranscriptionTask(inputAudio=audioPath, outputJson=jsonPath))

    # Sort for reproducibility and so progress feels predictable (alphabetical by path).
    tasks.sort(key=lambda task: task.inputAudio)
    return tasks


def setupCudaDllPathsWindows() -> None:
    """On Windows, point Python at the cuBLAS / cuDNN DLLs shipped by the nvidia-* pip
    packages. ctranslate2 4.x dlopens these by name and fails if they're not in the DLL
    search path. We walk the nvidia namespace packages, find any .dll files, and register
    each containing directory via os.add_dll_directory (Python 3.8+, Windows-only).

    Then we *preload* cuBLAS and cuDNN with ctypes.WinDLL. This is the critical step:
    ctranslate2 lazily LoadLibrary's cublas64_12.dll during the first GEMM call, and on
    Windows that lazy LoadLibrary does not always honour os.add_dll_directory paths set
    earlier on a different thread. Symptom: model loads fine, language detection works,
    but decoding fails with "Library cublas64_12.dll is not found or cannot be loaded".
    Preloading puts the DLLs in the process's loaded-modules table, so subsequent
    LoadLibrary calls return the existing handle and skip path search entirely.

    Also prepends each directory to PATH as a belt-and-suspenders fallback for any
    transitive DLL the preload doesn't cover.

    No-op on non-Windows platforms or if the packages aren't installed (then the original
    ctranslate2 error surfaces, which is the correct outcome — user should pip-install them).
    """
    if sys.platform != "win32":
        return

    print("setting up Windows CUDA DLL paths ...")

    # nvidia-cuda-runtime-cu12 provides cudart64_12.dll, which cublas64_12.dll depends on.
    # Including it defensively even though faster-whisper README doesn't mention it; some
    # CTranslate2 wheels need it explicitly.
    candidateModules = ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime", "nvidia.cuda_nvrtc")

    registeredDirectories: set[str] = set()
    discoveredDlls: list[Path] = []
    for moduleName in candidateModules:
        try:
            module = __import__(moduleName, fromlist=[""])
        except ImportError:
            print(f"  [skip] {moduleName} not installed")
            continue

        searchPaths = cast(Iterable[str], getattr(module, "__path__", None) or [])
        dllsFound: list[Path] = []
        for basePath in searchPaths:
            dllsFound.extend(Path(basePath).rglob("*.dll"))

        if not dllsFound:
            print(f"  [warn] {moduleName} imported but contains no DLLs")
            continue

        for dllPath in dllsFound:
            parentDirectory = str(dllPath.parent)
            if parentDirectory not in registeredDirectories:
                registeredDirectories.add(parentDirectory)
                os.add_dll_directory(parentDirectory)
            discoveredDlls.append(dllPath)
        print(f"  [ok]   {moduleName}: {len(dllsFound)} DLL(s)")

    if not registeredDirectories:
        print("  WARNING: no DLL directories registered; ctranslate2 will likely fail.")
        print("  Install: uv add nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'")
        return

    # PATH fallback: helps transitive deps that aren't preloaded below.
    os.environ["PATH"] = os.pathsep.join(sorted(registeredDirectories)) + os.pathsep + os.environ.get("PATH", "")

    # Preload order matters: dependencies first (cudart -> cublas -> cublasLt -> cudnn).
    # cublas64_12.dll depends on cudart64_12.dll; cuDNN ops depend on cuBLAS.
    import ctypes
    preloadOrder = (
        "cudart64_12.dll",
        "cublasLt64_12.dll",
        "cublas64_12.dll",
        "cudnn_graph64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn64_9.dll",
    )
    discoveredByName = {p.name: p for p in discoveredDlls}
    preloadedCount = 0
    for dllName in preloadOrder:
        dllPath = discoveredByName.get(dllName)
        if dllPath is None:
            continue
        try:
            ctypes.WinDLL(str(dllPath))
            preloadedCount += 1
        except OSError as error:
            print(f"  [warn] preload {dllName} failed: {error}")
    print(f"  preloaded {preloadedCount} CUDA DLL(s) into process.")


def loadModel(modelName: str, device: str, computeType: str) -> TranscriptionModel:
    """Lazy import so --help doesn't trigger CUDA init. Returns the WhisperModel instance."""
    # Must happen BEFORE importing faster_whisper / ctranslate2 on Windows so the loader
    # finds cublas64_12.dll and the cudnn DLLs at import time.
    setupCudaDllPathsWindows()
    fasterWhisper = importlib.import_module("faster_whisper")  # heavy import, kept local
    modelConstructor = cast(
        TranscriptionModelConstructor,
        getattr(fasterWhisper, "WhisperModel"),
    )

    print(f"loading model: {modelName} on {device} ({computeType}) ...")
    startSeconds = time.monotonic()
    model = modelConstructor(modelName, device=device, compute_type=computeType)
    print(f"  loaded in {time.monotonic() - startSeconds:.1f}s.")
    return model


def formatHms(seconds: float) -> str:
    """Format a duration as H:MM:SS, or M:SS when under an hour."""
    totalSeconds = int(seconds)
    hours, remainder = divmod(totalSeconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def transcribeOne(
    model: TranscriptionModel,
    task: TranscriptionTask,
    modelName: str,
    language: str | None,
    beamSize: int,
    vadThreshold: float,
    audioRoot: Path,
    showSegments: bool,
) -> TranscriptionResult | None:
    """Transcribe one file. Returns the JSON-ready dict, or None on failure."""
    task.outputJson.parent.mkdir(parents=True, exist_ok=True)
    startSeconds = time.monotonic()

    try:
        segmentIterator, info = model.transcribe(
            str(task.inputAudio),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": DEFAULT_VAD_MIN_SILENCE_MS,
                "threshold": vadThreshold,
            },
            # Prevents hallucination loops in long quiet stretches at the cost of cross-
            # segment coherence. For search/indexing we don't need the coherence.
            condition_on_previous_text=False,
            beam_size=beamSize,
        )
    except Exception as error:
        print(f"  ERROR: transcribe init failed: {error}")
        return None

    audioDurationSeconds = float(info.duration)
    print(f"  audio: {formatHms(audioDurationSeconds)}  "
          f"detected language: {info.language} (p={info.language_probability:.2f})")

    segmentDicts: list[TranscriptionSegmentJson] = []
    totalWords = 0
    nextProgressAt = PROGRESS_INTERVAL_SECONDS

    try:
        for segment in segmentIterator:
            words: list[TranscriptionWordJson] = []
            # word_timestamps=True populates segment.words; defensive None check anyway.
            if segment.words is not None:
                for word in segment.words:
                    words.append({
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    })
            totalWords += len(words)

            segmentDicts.append({
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "avgLogprob": segment.avg_logprob,
                "noSpeechProb": segment.no_speech_prob,
                "compressionRatio": segment.compression_ratio,
                "words": words,
            })

            # Live progress: print when we cross each PROGRESS_INTERVAL_SECONDS marker.
            if segment.end >= nextProgressAt:
                elapsedSeconds = time.monotonic() - startSeconds
                percent = (100.0 * segment.end / audioDurationSeconds
                           if audioDurationSeconds > 0 else 0.0)
                print(f"  [{formatHms(segment.end)} / {formatHms(audioDurationSeconds)}] "
                      f"{percent:5.1f}%  elapsed={elapsedSeconds:.0f}s  "
                      f"segments={len(segmentDicts)}")
                # Skip forward past the current segment's end so we don't print twice for
                # one segment that crossed multiple markers.
                while nextProgressAt <= segment.end:
                    nextProgressAt += PROGRESS_INTERVAL_SECONDS

            if showSegments:
                preview = segment.text.strip().replace("\n", " ")[:60]
                print(f"    [{segment.start:7.2f}-{segment.end:7.2f}] {preview}")
    except Exception as error:
        print(f"  ERROR: failed during decoding: {error}")
        return None

    elapsedSeconds = time.monotonic() - startSeconds
    speedRatio = (audioDurationSeconds / elapsedSeconds
                  if elapsedSeconds > 0 else 0.0)

    result: TranscriptionResult = {
        "version": JSON_SCHEMA_VERSION,
        "model": modelName,
        "language": info.language,
        "languageProbability": info.language_probability,
        "audioDurationSeconds": audioDurationSeconds,
        "transcriptionSeconds": elapsedSeconds,
        "audioRelativePath": str(task.inputAudio.relative_to(audioRoot)),
        "segments": segmentDicts,
    }

    print(f"  done: {len(segmentDicts)} segments, {totalWords} words, "
          f"{elapsedSeconds:.0f}s ({speedRatio:.1f}x realtime)")
    return result


def writeJsonAtomic(data: TranscriptionResult, outputPath: Path) -> None:
    """Write JSON to a .tmp sibling then rename, so resume never sees a partial file."""
    tempPath = outputPath.with_suffix(outputPath.suffix + ".tmp")
    # ensure_ascii=False keeps CJK characters readable in the file.
    # indent=2 costs ~25% size but the files are tiny relative to the audio.
    with tempPath.open("w", encoding="utf-8") as fileHandle:
        json.dump(data, fileHandle, ensure_ascii=False, indent=2)
    tempPath.rename(outputPath)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Whisper transcription with resume.")
    parser.add_argument("audioRoot", type=Path, help="Root directory of .wav files.")
    parser.add_argument("jsonRoot", type=Path, help="Root directory for JSON transcripts.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name or path (default {DEFAULT_MODEL}).")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        help=f"Language code, or 'auto' to detect (default {DEFAULT_LANGUAGE}).")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help=f"Device for inference (default {DEFAULT_DEVICE}).")
    parser.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE,
                        help=f"CTranslate2 compute type (default {DEFAULT_COMPUTE_TYPE}).")
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE,
                        help=f"Beam search width (default {DEFAULT_BEAM_SIZE}).")
    parser.add_argument("--vad-threshold", type=float, default=DEFAULT_VAD_THRESHOLD,
                        help=f"VAD speech-probability threshold (default {DEFAULT_VAD_THRESHOLD}).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N pending files (0 = no limit).")
    parser.add_argument("--show-segments", action="store_true",
                        help="Print every segment's text as it's produced (verbose).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List pending files and exit without loading the model.")
    args = parser.parse_args()

    audioRoot: Path = args.audioRoot.resolve()
    jsonRoot: Path = args.jsonRoot.resolve()
    if not audioRoot.is_dir():
        sys.exit(f"error: audio root does not exist: {audioRoot}")
    jsonRoot.mkdir(parents=True, exist_ok=True)

    print(f"scanning {audioRoot} ...")
    tasks = collectTasks(audioRoot, jsonRoot)
    if not tasks:
        print("nothing to do (all transcripts exist or no .wav files found).")
        return 0
    print(f"{len(tasks)} files pending.")

    if args.dry_run:
        for task in tasks:
            print(f"  {task.inputAudio}  ->  {task.outputJson}")
        return 0

    if args.limit > 0:
        tasks = tasks[:args.limit]
        print(f"limited to first {len(tasks)} file(s) due to --limit.")

    # "auto" sentinel becomes None for faster-whisper, which then auto-detects per file.
    language: str | None = None if args.language == "auto" else args.language

    try:
        model = loadModel(args.model, args.device, args.compute_type)
    except Exception as error:
        sys.exit(f"failed to load model: {error}")

    successCount = 0
    failureCount = 0
    failures: list[Path] = []
    totalAudioSeconds = 0.0
    totalTranscribeSeconds = 0.0
    overallStart = time.monotonic()

    try:
        for index, task in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {task.inputAudio.relative_to(audioRoot)}")
            result = transcribeOne(
                model, task,
                modelName=args.model,
                language=language,
                beamSize=args.beam_size,
                vadThreshold=args.vad_threshold,
                audioRoot=audioRoot,
                showSegments=args.show_segments,
            )
            if result is None:
                failureCount += 1
                failures.append(task.inputAudio)
                continue
            try:
                writeJsonAtomic(result, task.outputJson)
            except OSError as error:
                print(f"  ERROR: write failed: {error}")
                failureCount += 1
                failures.append(task.inputAudio)
                continue

            successCount += 1
            totalAudioSeconds += result["audioDurationSeconds"]
            totalTranscribeSeconds += result["transcriptionSeconds"]
    except KeyboardInterrupt:
        print("\ninterrupted. progress preserved up to last completed file.")
        return 130

    overallElapsed = time.monotonic() - overallStart
    print()
    print(f"done. success={successCount}  failure={failureCount}")
    if totalAudioSeconds > 0 and totalTranscribeSeconds > 0:
        ratio = totalAudioSeconds / totalTranscribeSeconds
        print(f"transcribed {formatHms(totalAudioSeconds)} of audio in "
              f"{formatHms(overallElapsed)} wall time "
              f"(model time {formatHms(totalTranscribeSeconds)}, {ratio:.1f}x realtime)")
    if failures:
        print("failed files:")
        for path in failures:
            print(f"  {path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
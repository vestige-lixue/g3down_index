"""
Recursively extract audio from video files with dynamic gain normalization.

Usage:
    python extract_audio.py <input_dir> <output_dir> [--workers N] [--dry-run]

Output mirrors input directory structure. Already-processed files are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# ffmpeg dynaudnorm filter parameters; tuned for vlog with variable speaking volume.
# f=200ms frame, g=15 smoothing, p=0.95 peak, m=15 max gain factor, s=12 compression.
AUDIO_FILTER: str = "dynaudnorm=f=200:g=15:p=0.95:m=15:s=12"

# Output audio format: 16kHz mono PCM s16le, what faster-whisper expects.
SAMPLE_RATE: int = 16000

# Sentinel filename suffix recording "this video has no audio track". Created alongside
# where the .wav would have been, so resume runs skip the video without re-probing.
NO_AUDIO_MARKER_SUFFIX: str = ".noaudio"

# dynaudnorm requires roughly (f * g / 1000) seconds of audio to initialize its smoothing
# window. With f=200, g=15 that's 3.0s. We use 4.0s as a safety margin: shorter clips skip
# the filter entirely (raw extraction). These are typically transitions / reaction shots
# where dynamic gain matters little, and Whisper handles brief unfiltered audio fine.
MIN_DURATION_FOR_FILTER_SECONDS: float = 4.0


@dataclass(frozen=True)
class ExtractionTask:
    inputVideo: Path
    outputAudio: Path


@dataclass(frozen=True)
class ExtractionResult:
    inputVideo: Path
    outputAudio: Path
    success: bool
    message: str
    # True if the video had no audio track. Still counted as success; no .wav is produced,
    # but a marker file is written so resume runs skip the video.
    noAudio: bool = False


def findFfmpeg() -> str:
    """Return path to ffmpeg executable, or exit if not found."""
    ffmpegPath = shutil.which("ffmpeg")
    if ffmpegPath is None:
        sys.exit("error: ffmpeg not found in PATH. Install it first.")
    return ffmpegPath


def findFfprobe() -> str | None:
    """Return path to ffprobe, or None. Missing ffprobe degrades gracefully (no duration probe)."""
    return shutil.which("ffprobe")


@dataclass(frozen=True)
class VideoProbe:
    hasAudio: bool
    durationSeconds: float | None


def probeVideo(videoPath: Path, ffprobePath: str) -> VideoProbe | None:
    """Probe stream types and duration in one call. Returns None on probe failure."""
    command = [
        ffprobePath,
        "-v", "error",
        "-show_entries", "stream=codec_type:format=duration",
        "-of", "json",
        str(videoPath),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        data = cast(dict[str, Any], json.loads(completed.stdout))
    except json.JSONDecodeError:
        return None

    streams = cast(list[dict[str, Any]], data.get("streams", []) or [])
    hasAudio = any(stream.get("codec_type") == "audio" for stream in streams)

    durationSeconds: float | None
    try:
        formatData = cast(dict[str, Any], data.get("format", {}))
        durationSeconds = float(formatData.get("duration", ""))
    except (TypeError, ValueError):
        durationSeconds = None

    return VideoProbe(hasAudio=hasAudio, durationSeconds=durationSeconds)


def noAudioMarkerFor(audioPath: Path) -> Path:
    """Return the sibling marker path for a would-be audio file."""
    return audioPath.parent / (audioPath.name + NO_AUDIO_MARKER_SUFFIX)


def collectTasks(inputRoot: Path, outputRoot: Path) -> list[ExtractionTask]:
    """Walk inputRoot, build a task per video, skip already-extracted ones."""
    tasks: list[ExtractionTask] = []
    for videoPath in inputRoot.rglob("*"):
        if not videoPath.is_file():
            continue
        if videoPath.suffix.lower() != ".mp4":
            continue

        # Mirror directory structure under outputRoot, replace extension with .wav.
        relativePath = videoPath.relative_to(inputRoot)
        audioPath = (outputRoot / relativePath).with_suffix(".wav")

        # Resume: skip if the output already exists and is non-empty.
        # A zero-byte file means a previous run was interrupted mid-write; redo it.
        if audioPath.exists() and audioPath.stat().st_size > 0:
            continue
        # Resume: also skip if we previously confirmed this video has no audio track.
        if noAudioMarkerFor(audioPath).exists():
            continue

        tasks.append(ExtractionTask(inputVideo=videoPath, outputAudio=audioPath))
    return tasks


def _runFfmpegExtract(
    ffmpegPath: str,
    inputVideo: Path,
    outputPath: Path,
    audioFilter: str | None,
) -> tuple[bool, str]:
    """Single ffmpeg invocation. Returns (success, message). Caller handles temp/rename."""
    command = [
        ffmpegPath,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(inputVideo),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
    ]
    if audioFilter is not None:
        command += ["-af", audioFilter]
    command += ["-c:a", "pcm_s16le", str(outputPath)]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        return False, f"failed to invoke ffmpeg: {error}"

    if completed.returncode != 0:
        stderr = completed.stderr.strip().splitlines()
        lastLine = stderr[-1] if stderr else "unknown ffmpeg error"
        return False, f"ffmpeg exit {completed.returncode}: {lastLine}"
    return True, "ok"


def extractOne(
    task: ExtractionTask, ffmpegPath: str, ffprobePath: str | None,
) -> ExtractionResult:
    """Run ffmpeg for a single video. Writes to a .tmp.wav first, then renames on success."""
    task.outputAudio.parent.mkdir(parents=True, exist_ok=True)
    markerPath = noAudioMarkerFor(task.outputAudio)

    # Probe video metadata once. We use it for two decisions: (1) is there an audio track at
    # all? (2) is the audio long enough for dynaudnorm? Both are best-effort; if ffprobe is
    # missing or fails, we fall back to attempting extraction and detecting empty output.
    probe: VideoProbe | None = None
    if ffprobePath is not None:
        probe = probeVideo(task.inputVideo, ffprobePath)

    # Fast path: probe confirmed no audio track. Drop a marker and return success.
    if probe is not None and not probe.hasAudio:
        markerPath.touch()
        return ExtractionResult(
            inputVideo=task.inputVideo,
            outputAudio=task.outputAudio,
            success=True,
            message="no audio track",
            noAudio=True,
        )

    # Atomic-ish write: ffmpeg writes to .tmp.wav; we rename only on success.
    # Prevents resume from skipping a half-written file.
    tempOutput = task.outputAudio.with_suffix(".tmp.wav")

    # Decide whether to apply dynaudnorm based on clip duration. Short clips break the
    # filter's initialization window; we extract them raw instead.
    durationSeconds: float | None = probe.durationSeconds if probe is not None else None
    # If probe failed/missing (None), optimistically try the filter; we retry unfiltered on failure.
    useFilter: bool = durationSeconds is None or durationSeconds >= MIN_DURATION_FOR_FILTER_SECONDS

    success, message = _runFfmpegExtract(
        ffmpegPath, task.inputVideo, tempOutput,
        audioFilter=AUDIO_FILTER if useFilter else None,
    )

    # Retry without filter if filtered attempt failed. Catches probe misreads and any other
    # filter-incompatible content. Successful retry is reported so it's visible in the log.
    retriedUnfiltered = False
    if not success and useFilter:
        tempOutput.unlink(missing_ok=True)
        success, message = _runFfmpegExtract(
            ffmpegPath, task.inputVideo, tempOutput, audioFilter=None,
        )
        retriedUnfiltered = success

    if not success:
        tempOutput.unlink(missing_ok=True)
        return ExtractionResult(
            inputVideo=task.inputVideo,
            outputAudio=task.outputAudio,
            success=False,
            message=message,
        )

    # ffmpeg returned 0 but produced no audio: video has no audio track and probe didn't
    # catch it (e.g., ffprobe was unavailable). Treat as silent success with marker.
    if not tempOutput.exists() or tempOutput.stat().st_size == 0:
        tempOutput.unlink(missing_ok=True)
        markerPath.touch()
        return ExtractionResult(
            inputVideo=task.inputVideo,
            outputAudio=task.outputAudio,
            success=True,
            message="no audio track (detected post-extraction)",
            noAudio=True,
        )

    tempOutput.rename(task.outputAudio)

    if retriedUnfiltered:
        note = "ok (filter failed, extracted raw)"
    elif not useFilter:
        durationText = f"{durationSeconds:.2f}s" if durationSeconds is not None else "unknown"
        note = f"ok (no filter, short clip {durationText})"
    else:
        note = "ok"
    return ExtractionResult(
        inputVideo=task.inputVideo,
        outputAudio=task.outputAudio,
        success=True,
        message=note,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch extract normalized audio from videos.")
    parser.add_argument("inputDir", type=Path, help="Root directory to scan for videos.")
    parser.add_argument("outputDir", type=Path, help="Root directory to write .wav files into.")
    parser.add_argument(
        "--workers", "-w", type=int, default=2,
        help="Parallel ffmpeg processes (default 2; ffmpeg is largely I/O bound here).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be processed and exit.",
    )
    args = parser.parse_args()

    inputRoot: Path = args.inputDir.resolve()
    outputRoot: Path = args.outputDir.resolve()

    if not inputRoot.is_dir():
        sys.exit(f"error: input directory does not exist: {inputRoot}")

    ffmpegPath = findFfmpeg()
    ffprobePath = findFfprobe()
    if ffprobePath is None:
        print("warning: ffprobe not found; cannot pre-check clip duration. Filter failures "
              "will retry unfiltered, so output is still correct, just slower for tiny clips.")
    outputRoot.mkdir(parents=True, exist_ok=True)

    print(f"scanning {inputRoot} ...")
    tasks = collectTasks(inputRoot, outputRoot)

    if not tasks:
        print("nothing to do (all outputs already exist or no videos found).")
        return 0

    print(f"found {len(tasks)} videos to process.")
    if args.dry_run:
        for task in tasks:
            print(f"  {task.inputVideo}  ->  {task.outputAudio}")
        return 0

    successCount = 0
    noAudioCount = 0
    failureCount = 0
    failures: list[ExtractionResult] = []

    # ProcessPoolExecutor so Ctrl-C is clean and ffmpeg processes are isolated.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futureToTask = {
            executor.submit(extractOne, task, ffmpegPath, ffprobePath): task for task in tasks
        }
        for index, future in enumerate(as_completed(futureToTask), start=1):
            result = future.result()
            # Tag distinguishes: SKIP (no audio, silent success), OK (audio extracted), FAIL.
            if result.noAudio:
                tag = "SKIP"
            elif result.success:
                tag = "OK  "
            else:
                tag = "FAIL"
            # Show the message for failures and for non-trivial OKs (filter skipped / retried).
            suffix = f"  [{result.message}]" if result.message != "ok" else ""
            print(f"[{index:>4}/{len(tasks)}] {tag} {result.inputVideo.name}{suffix}")
            if result.success:
                if result.noAudio:
                    noAudioCount += 1
                else:
                    successCount += 1
            else:
                failureCount += 1
                failures.append(result)

    print()
    print(f"done. extracted={successCount}  no_audio={noAudioCount}  failure={failureCount}")
    if failures:
        print("failed files:")
        for result in failures:
            print(f"  {result.inputVideo}: {result.message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
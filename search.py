"""
Search Whisper transcripts in the SQLite FTS5 index and jump to the moment in mpv.

Usage:
    # One-shot: print results and exit
    uv run search.py [options] <query>

    # REPL: interactive search loop
    uv run search.py [options]

Options:
    --db DB                SQLite DB (default: transcripts.db in current directory)
    --video-root DIR       Where the original videos live; needed to open in mpv
    --limit N              Max results per query (default 20)
    --context N            Show N segments before/after each match (default 0)
    --no-filter            Disable the hallucination filter

REPL commands:
    <query>     run a search; multiple words = all must appear (AND)
    "<phrase>"  quoted phrase = exact substring match including spaces
    <number>    open result #N in mpv (after a search)
    /noflt      toggle hallucination filter on/off
    /ctx N      set context segment radius to N
    q           quit
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Must match build_index.py's DB_SCHEMA_VERSION.
EXPECTED_DB_SCHEMA_VERSION: int = 1

# Used to map audio_relative_path (.wav) back to the original video file.
VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts",
)

# Hallucination filter thresholds. These match the values faster-whisper uses internally
# to decide whether a segment was a decoding failure (and retry with higher temperature).
# Segments exceeding any of these are very likely garbage repetition or silence-as-text.
JUNK_COMPRESSION_RATIO_THRESHOLD: float = 2.4   # > this = repetitive output
JUNK_AVG_LOGPROB_THRESHOLD: float = -1.0        # < this = low confidence
JUNK_NO_SPEECH_PROB_THRESHOLD: float = 0.6      # > this = probably silence

DEFAULT_LIMIT: int = 20


@dataclass(frozen=True)
class SearchResult:
    fileRelativePath: str    # JSON path stored in files.relative_path
    audioRelativePath: str   # what's in files.audio_relative_path; .wav-extensioned
    segmentId: int           # the per-file segment id (0, 1, 2, ...)
    startSeconds: float
    endSeconds: float
    text: str


# --- DB ---------------------------------------------------------------------------------

def openDatabaseReadOnly(dbPath: Path) -> sqlite3.Connection:
    """Open the DB read-only. Multiple search.py instances can coexist this way."""
    if not dbPath.exists():
        sys.exit(f"error: database does not exist: {dbPath}")
    # uri=True lets us use the file: URI for ro mode.
    connection = sqlite3.connect(f"file:{dbPath}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    row = connection.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    actualVersion = row["value"] if row is not None else "(missing)"
    if row is None or int(row["value"]) != EXPECTED_DB_SCHEMA_VERSION:
        sys.exit(
            f"error: DB schema version {actualVersion} != expected "
            f"{EXPECTED_DB_SCHEMA_VERSION}. Run build_index.py --rebuild."
        )
    return connection


# --- Query construction -----------------------------------------------------------------

def escapeLikePattern(term: str) -> str:
    """Escape LIKE wildcards so the user's input is treated literally. Pair with ESCAPE '\\'."""
    return (term
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_"))


def parseQueryTerms(query: str) -> list[str]:
    """Split a query into terms. shlex respects "quoted phrases" but won't crash on input
    with unbalanced quotes — falls back to whitespace split."""
    try:
        terms = shlex.split(query)
    except ValueError:
        terms = query.split()
    return [t for t in terms if t]


def runSearch(
    connection: sqlite3.Connection,
    queryTerms: list[str],
    limit: int,
    useFilter: bool,
) -> list[SearchResult]:
    """Multi-term AND search via LIKE on the trigram FTS table. Empty terms → no results."""
    if not queryTerms:
        return []

    # One LIKE clause per term; trigram-indexed for 3+ char terms, scanned for shorter.
    likeClauses = ["segments_fts.text LIKE ? ESCAPE '\\'" for _ in queryTerms]
    likeParams = [f"%{escapeLikePattern(term)}%" for term in queryTerms]

    if useFilter:
        # NULL-safe comparisons: if a stat is missing we keep the segment (innocent until
        # proven guilty). faster-whisper always populates these, but defensive anyway.
        filterClause = (
            " AND (s.compression_ratio IS NULL OR s.compression_ratio < ?)"
            " AND (s.avg_logprob       IS NULL OR s.avg_logprob       > ?)"
            " AND (s.no_speech_prob    IS NULL OR s.no_speech_prob    < ?)"
        )
        filterParams = [
            JUNK_COMPRESSION_RATIO_THRESHOLD,
            JUNK_AVG_LOGPROB_THRESHOLD,
            JUNK_NO_SPEECH_PROB_THRESHOLD,
        ]
    else:
        filterClause = ""
        filterParams = []

    sql = f"""
        SELECT
            f.relative_path        AS file_relative_path,
            f.audio_relative_path  AS audio_relative_path,
            s.segment_id           AS segment_id,
            s.start_seconds        AS start_seconds,
            s.end_seconds          AS end_seconds,
            s.text                 AS text
        FROM segments_fts
        JOIN segments s ON segments_fts.rowid = s.id
        JOIN files    f ON s.file_id = f.id
        WHERE {' AND '.join(likeClauses)}
        {filterClause}
        ORDER BY f.relative_path, s.start_seconds
        LIMIT ?
    """
    rows = connection.execute(sql, [*likeParams, *filterParams, limit]).fetchall()
    return [
        SearchResult(
            fileRelativePath=row["file_relative_path"],
            audioRelativePath=row["audio_relative_path"] or "",
            segmentId=int(row["segment_id"]),
            startSeconds=float(row["start_seconds"]),
            endSeconds=float(row["end_seconds"]),
            text=str(row["text"]),
        )
        for row in rows
    ]


def loadContextSegments(
    connection: sqlite3.Connection,
    fileRelativePath: str,
    centerSegmentId: int,
    radius: int,
) -> list[tuple[int, str]]:
    """Fetch segments within ±radius of a hit, excluding the hit itself."""
    if radius <= 0:
        return []
    fileRow = connection.execute(
        "SELECT id FROM files WHERE relative_path = ?", (fileRelativePath,)
    ).fetchone()
    if fileRow is None:
        return []
    rows = connection.execute(
        """
        SELECT segment_id, text FROM segments
        WHERE file_id = ?
          AND segment_id BETWEEN ? AND ?
          AND segment_id != ?
        ORDER BY segment_id
        """,
        (fileRow["id"],
         centerSegmentId - radius, centerSegmentId + radius,
         centerSegmentId),
    ).fetchall()
    return [(int(row["segment_id"]), str(row["text"])) for row in rows]


# --- Presentation -----------------------------------------------------------------------

def formatTimestamp(seconds: float) -> str:
    """HH:MM:SS.t — what mpv --start= accepts and what humans read."""
    totalSeconds = max(0.0, seconds)
    hours, remainder = divmod(totalSeconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:04.1f}"


def normalizePath(rawPath: str) -> str:
    """Forward slashes for display, regardless of how the DB stored them."""
    return rawPath.replace("\\", "/")


def resolveVideoPath(audioRelativePath: str, videoRoot: Path | None) -> Path | None:
    """Find the original video corresponding to a (stored) audio relative path. Tries each
    known video extension. Returns None if videoRoot isn't set or no candidate found."""
    if videoRoot is None or not audioRelativePath:
        return None
    audioPath = Path(audioRelativePath.replace("\\", "/"))
    videoDir = videoRoot / audioPath.parent
    stem = audioPath.stem
    for extension in VIDEO_EXTENSIONS:
        candidate = videoDir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def displayResults(
    results: list[SearchResult],
    connection: sqlite3.Connection,
    contextRadius: int,
    videoRoot: Path | None,
) -> None:
    """Print numbered, human-readable results with optional surrounding context."""
    if not results:
        print("no matches.")
        return

    for index, result in enumerate(results, start=1):
        timestamp = formatTimestamp(result.startSeconds)
        durationSeconds = result.endSeconds - result.startSeconds
        videoPath = resolveVideoPath(result.audioRelativePath, videoRoot)
        # Prefer showing the actual video path if resolvable; otherwise the JSON path.
        locationLabel = str(videoPath) if videoPath is not None \
            else normalizePath(result.fileRelativePath)

        print(f"[{index:>2}] {locationLabel}")
        print(f"     @ {timestamp}  ({durationSeconds:.1f}s)")

        if contextRadius > 0:
            neighbours = loadContextSegments(
                connection, result.fileRelativePath, result.segmentId, contextRadius
            )
            for segmentId, text in neighbours:
                if segmentId < result.segmentId:
                    print(f"     -  {text.strip()}")
            print(f"     >  {result.text.strip()}")
            for segmentId, text in neighbours:
                if segmentId > result.segmentId:
                    print(f"     -  {text.strip()}")
        else:
            print(f"     {result.text.strip()}")
        print()


# --- mpv --------------------------------------------------------------------------------

def openInMpv(videoPath: Path, startSeconds: float, mpvBinary: str) -> bool:
    """Launch mpv in the background at the given timestamp. Fire-and-forget."""
    try:
        subprocess.Popen(
            [mpvBinary, f"--start={startSeconds:.2f}", str(videoPath)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except OSError as error:
        print(f"  failed to launch mpv: {error}")
        return False


# --- REPL -------------------------------------------------------------------------------

def runRepl(
    connection: sqlite3.Connection,
    videoRoot: Path | None,
    initialLimit: int,
    initialContext: int,
    initialFilter: bool,
    mpvBinary: str | None,
) -> None:
    """Interactive search loop."""
    print("type a query to search, a number to open that result, q to quit.")
    print("commands: /noflt (toggle filter)  /ctx N (context radius)")
    print()

    limit = initialLimit
    contextRadius = initialContext
    useFilter = initialFilter
    lastResults: list[SearchResult] = []

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            break

        if line == "/noflt":
            useFilter = not useFilter
            print(f"  filter is now {'ON' if useFilter else 'OFF'}")
            continue

        if line.startswith("/ctx"):
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                contextRadius = max(0, int(parts[1]))
                print(f"  context radius = {contextRadius}")
            else:
                print("  usage: /ctx <integer>")
            continue

        # Pure-digit input picks a result number from the last search.
        if line.isdigit():
            chosen = int(line)
            if not (1 <= chosen <= len(lastResults)):
                print(f"  no result #{chosen} (have {len(lastResults)})")
                continue
            result = lastResults[chosen - 1]
            videoPath = resolveVideoPath(result.audioRelativePath, videoRoot)
            if videoPath is None:
                print(f"  no video found for {result.audioRelativePath}"
                      f" (check --video-root)")
                continue
            if mpvBinary is None:
                print(f"  mpv not in PATH. Open manually:")
                print(f"    {videoPath}  @  {formatTimestamp(result.startSeconds)}")
                continue
            print(f"  opening {videoPath.name} @ {formatTimestamp(result.startSeconds)}")
            openInMpv(videoPath, result.startSeconds, mpvBinary)
            continue

        # Otherwise: it's a search query.
        terms = parseQueryTerms(line)
        if not terms:
            continue
        lastResults = runSearch(connection, terms, limit, useFilter)
        displayResults(lastResults, connection, contextRadius, videoRoot)
        if lastResults:
            print(f"  {len(lastResults)} result(s). Enter a number to open in mpv.")
            if len(lastResults) >= limit:
                print(f"  (showing first {limit}; refine query to narrow down)")


# --- entry point ------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search Whisper transcripts; open hits in mpv at the timestamp.",
    )
    parser.add_argument("query", nargs="*",
                        help="Query terms. Omit to enter the interactive REPL.")
    parser.add_argument("--db", type=Path, default=Path("transcripts.db"),
                        help="SQLite DB path (default: transcripts.db).")
    parser.add_argument("--video-root", type=Path, default=None,
                        help="Root directory of original video files (required for mpv jump).")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Max results per query (default {DEFAULT_LIMIT}).")
    parser.add_argument("--context", type=int, default=0,
                        help="Show N segments before/after each match (default 0).")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable hallucination filter (keep low-confidence segments).")
    args = parser.parse_args()

    dbPath: Path = args.db.resolve()
    connection = openDatabaseReadOnly(dbPath)

    videoRoot: Path | None = args.video_root.resolve() if args.video_root else None
    if videoRoot is not None and not videoRoot.is_dir():
        print(f"warning: --video-root does not exist: {videoRoot}", file=sys.stderr)
        videoRoot = None

    mpvBinary = shutil.which("mpv")
    if videoRoot is not None and mpvBinary is None:
        print("warning: mpv not found in PATH; can't auto-open videos.", file=sys.stderr)
        print("         install with: winget install mpv", file=sys.stderr)

    useFilter = not args.no_filter

    if args.query:
        # One-shot mode: print results, exit.
        terms = list(args.query)
        results = runSearch(connection, terms, args.limit, useFilter)
        displayResults(results, connection, args.context, videoRoot)
    else:
        runRepl(connection, videoRoot, args.limit, args.context, useFilter, mpvBinary)

    connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
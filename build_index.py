"""
Build / refresh a SQLite FTS5 index from Whisper JSON transcripts.

Walks <jsonRoot>, ingests every .json transcript into a SQLite database with an FTS5
trigram index over segment text. Resume-friendly: files whose mtime hasn't changed since
the last run are skipped.

Usage:
    uv run build_index.py <json_root> [--db DB] [--rebuild] [--dry-run]

Schema (DB version 1):
    meta(key, value)                            -- schema_version, json_schema_version
    files(id, relative_path, json_mtime, ...)   -- one row per JSON transcript
    segments(id, file_id, segment_id, start, end, text, avg_logprob, ...)
    segments_fts(text)                          -- FTS5 trigram, external-content over segments
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Bump this if the DB schema changes incompatibly; the script will refuse to open an
# existing DB at a different version (use --rebuild to recreate).
DB_SCHEMA_VERSION: int = 1

# JSON schema we expect from transcribe_audio.py. Must match its JSON_SCHEMA_VERSION.
EXPECTED_JSON_SCHEMA_VERSION: int = 1

# Trigram tokenizer requires SQLite 3.34+. Python 3.10's bundled sqlite is newer than this,
# but we check explicitly because the failure mode is otherwise opaque.
MIN_SQLITE_VERSION: tuple[int, int, int] = (3, 34, 0)

# Tolerance when comparing stored mtime to current mtime (filesystem precision varies).
MTIME_EPSILON_SECONDS: float = 0.001


@dataclass(frozen=True)
class IndexCounters:
    new: int = 0
    reindex: int = 0
    skip: int = 0
    error: int = 0
    segments: int = 0


def ensureFts5TrigramAvailable(connection: sqlite3.Connection) -> None:
    """Fail fast if the SQLite build doesn't include FTS5 trigram. Better than a cryptic
    error mid-ingest."""
    actualVersion = sqlite3.sqlite_version_info
    if actualVersion < MIN_SQLITE_VERSION:
        sys.exit(
            f"error: SQLite {'.'.join(map(str, actualVersion))} is too old; "
            f"need {'.'.join(map(str, MIN_SQLITE_VERSION))}+ for the trigram tokenizer."
        )
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE temp._fts5_probe USING fts5(x, tokenize='trigram')"
        )
        connection.execute("DROP TABLE temp._fts5_probe")
    except sqlite3.OperationalError as error:
        sys.exit(f"error: FTS5 trigram tokenizer unavailable: {error}")


def initializeSchema(connection: sqlite3.Connection) -> None:
    """Create tables, indexes, FTS table, and sync triggers if missing. Verifies schema
    version on existing DBs and exits with a clear message on mismatch."""
    # If meta table already exists with a schema_version row, validate it before touching
    # anything else.
    cursor = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
    )
    if cursor.fetchone() is not None:
        cursor = connection.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cursor.fetchone()
        if row is not None:
            existingVersion = int(row[0])
            if existingVersion != DB_SCHEMA_VERSION:
                sys.exit(
                    f"error: existing DB schema version {existingVersion}, "
                    f"script expects {DB_SCHEMA_VERSION}. Run with --rebuild to recreate."
                )

    # Schema. External-content FTS5 means we don't duplicate the text — the FTS table just
    # maintains the inverted index, pulling content from `segments` via rowid.
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            id                      INTEGER PRIMARY KEY,
            relative_path           TEXT    NOT NULL UNIQUE,
            json_mtime              REAL    NOT NULL,
            audio_duration_seconds  REAL,
            transcription_seconds   REAL,
            language                TEXT,
            language_probability    REAL,
            model                   TEXT,
            audio_relative_path     TEXT,
            indexed_at              TEXT
        );

        CREATE TABLE IF NOT EXISTS segments (
            id                  INTEGER PRIMARY KEY,
            file_id             INTEGER NOT NULL,
            segment_id          INTEGER NOT NULL,
            start_seconds       REAL    NOT NULL,
            end_seconds         REAL    NOT NULL,
            text                TEXT    NOT NULL,
            avg_logprob         REAL,
            no_speech_prob      REAL,
            compression_ratio   REAL,
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_segments_file  ON segments(file_id);
        CREATE INDEX IF NOT EXISTS idx_segments_start ON segments(start_seconds);

        CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
            text,
            content='segments',
            content_rowid='id',
            tokenize='trigram'
        );

        -- Keep FTS in sync with segments. AFTER DELETE handles both direct deletes and
        -- CASCADE-from-files deletes.
        CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
            INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
            INSERT INTO segments_fts(segments_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
            INSERT INTO segments_fts(segments_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
            INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
        END;
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(DB_SCHEMA_VERSION),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('json_schema_version', ?)",
        (str(EXPECTED_JSON_SCHEMA_VERSION),),
    )
    connection.commit()


def openDatabase(dbPath: Path) -> sqlite3.Connection:
    """Open SQLite DB with sane pragmas, verify trigram support, initialize schema."""
    connection = sqlite3.connect(dbPath)
    connection.row_factory = sqlite3.Row
    # WAL: better concurrency, faster commits. synchronous=NORMAL is safe with WAL.
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    # Required for ON DELETE CASCADE to fire.
    connection.execute("PRAGMA foreign_keys = ON")
    ensureFts5TrigramAvailable(connection)
    initializeSchema(connection)
    return connection


def collectJsonFiles(jsonRoot: Path) -> list[Path]:
    """Find all .json files under jsonRoot, sorted for reproducibility."""
    paths: list[Path] = []
    for jsonPath in jsonRoot.rglob("*.json"):
        if not jsonPath.is_file():
            continue
        if jsonPath.name.endswith(".tmp"):
            continue  # half-written from interrupted transcription
        paths.append(jsonPath)
    paths.sort()
    return paths


def loadTranscriptJson(jsonPath: Path) -> dict | None:
    """Load a transcript JSON, validating schema version. Returns None on any failure."""
    try:
        with jsonPath.open("r", encoding="utf-8") as fileHandle:
            data = json.load(fileHandle)
    except (OSError, json.JSONDecodeError) as error:
        print(f"  [error] failed to read: {error}")
        return None

    version = data.get("version")
    if version != EXPECTED_JSON_SCHEMA_VERSION:
        print(f"  [error] unexpected JSON schema version: {version}")
        return None
    if not isinstance(data.get("segments"), list):
        print(f"  [error] missing or invalid 'segments' field")
        return None
    return data


def indexFile(
    connection: sqlite3.Connection,
    jsonPath: Path,
    relativePath: str,
    data: dict,
) -> tuple[str, int]:
    """Index one JSON file inside a single transaction. Returns (action, segmentCount)
    where action is one of 'new', 'reindex', or 'skip'."""
    cursor = connection.cursor()
    currentMtime = jsonPath.stat().st_mtime

    cursor.execute(
        "SELECT id, json_mtime FROM files WHERE relative_path = ?", (relativePath,)
    )
    existing = cursor.fetchone()

    if existing is not None:
        if abs(existing["json_mtime"] - currentMtime) < MTIME_EPSILON_SECONDS:
            return "skip", 0
        # mtime changed: drop old file row; CASCADE removes its segments and (via the
        # AFTER DELETE trigger) the corresponding FTS entries.
        cursor.execute("DELETE FROM files WHERE id = ?", (existing["id"],))
        action = "reindex"
    else:
        action = "new"

    cursor.execute(
        """
        INSERT INTO files (
            relative_path, json_mtime, audio_duration_seconds, transcription_seconds,
            language, language_probability, model, audio_relative_path, indexed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            relativePath,
            currentMtime,
            data.get("audioDurationSeconds"),
            data.get("transcriptionSeconds"),
            data.get("language"),
            data.get("languageProbability"),
            data.get("model"),
            data.get("audioRelativePath"),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    fileRowId = cursor.lastrowid

    # Build rows in a list first so executemany can batch the insert.
    segmentRows: list[tuple] = []
    for segment in data["segments"]:
        text = (segment.get("text") or "").strip()
        if not text:
            continue  # empty text is useless and would pollute FTS
        segmentRows.append((
            fileRowId,
            segment.get("id"),
            segment.get("start"),
            segment.get("end"),
            text,
            segment.get("avgLogprob"),
            segment.get("noSpeechProb"),
            segment.get("compressionRatio"),
        ))

    if segmentRows:
        cursor.executemany(
            """
            INSERT INTO segments (
                file_id, segment_id, start_seconds, end_seconds, text,
                avg_logprob, no_speech_prob, compression_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            segmentRows,
        )

    return action, len(segmentRows)


def normalizeRelativePath(jsonPath: Path, jsonRoot: Path) -> str:
    """Path relative to jsonRoot, with forward slashes (cross-platform DB portability)."""
    return str(jsonPath.relative_to(jsonRoot)).replace("\\", "/")


def printSummary(connection: sqlite3.Connection, counters: IndexCounters,
                 elapsedSeconds: float) -> None:
    print()
    print(f"done in {elapsedSeconds:.1f}s.  "
          f"new={counters.new}  reindex={counters.reindex}  "
          f"skip={counters.skip}  error={counters.error}")
    print(f"segments indexed this run: {counters.segments}")

    fileTotal = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    segmentTotal = connection.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    audioTotal = connection.execute(
        "SELECT COALESCE(SUM(audio_duration_seconds), 0) FROM files"
    ).fetchone()[0]
    audioHours = audioTotal / 3600.0
    print(f"db totals: files={fileTotal}  segments={segmentTotal}  "
          f"audio={audioHours:.1f}h")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build / refresh SQLite FTS5 index from Whisper transcripts."
    )
    parser.add_argument("jsonRoot", type=Path, help="Root directory of .json transcripts.")
    parser.add_argument("--db", type=Path, default=None,
                        help="SQLite DB path (default: <jsonRoot>/../transcripts.db).")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing DB before indexing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files found and exit without writing.")
    args = parser.parse_args()

    jsonRoot: Path = args.jsonRoot.resolve()
    if not jsonRoot.is_dir():
        sys.exit(f"error: JSON root does not exist: {jsonRoot}")

    if args.db is not None:
        dbPath = args.db.resolve()
    else:
        dbPath = (jsonRoot.parent / "transcripts.db").resolve()
    dbPath.parent.mkdir(parents=True, exist_ok=True)

    if args.rebuild and dbPath.exists():
        print(f"removing existing db: {dbPath}")
        dbPath.unlink()
        # WAL/SHM sidecar files outlive the main db on hard kills; clean them up too.
        for suffix in ("-wal", "-shm"):
            sidecar = dbPath.with_name(dbPath.name + suffix)
            if sidecar.exists():
                sidecar.unlink()

    print(f"scanning {jsonRoot} ...")
    jsonPaths = collectJsonFiles(jsonRoot)
    print(f"found {len(jsonPaths)} .json file(s)")
    if not jsonPaths:
        return 0

    if args.dry_run:
        for path in jsonPaths[:30]:
            print(f"  {path.relative_to(jsonRoot)}")
        if len(jsonPaths) > 30:
            print(f"  ... and {len(jsonPaths) - 30} more")
        return 0

    print(f"db: {dbPath}")
    connection = openDatabase(dbPath)

    counters = IndexCounters()
    # We rebuild the IndexCounters dataclass via dict; frozen dataclasses can't be mutated
    # but updating local ints is cleaner anyway.
    newCount = reindexCount = skipCount = errorCount = segmentCount = 0
    startTime = time.monotonic()

    try:
        for index, jsonPath in enumerate(jsonPaths, start=1):
            relativePath = normalizeRelativePath(jsonPath, jsonRoot)
            print(f"[{index:>4}/{len(jsonPaths)}] {relativePath}", end="", flush=True)

            data = loadTranscriptJson(jsonPath)
            if data is None:
                errorCount += 1
                continue

            try:
                action, segmentsAdded = indexFile(connection, jsonPath, relativePath, data)
                connection.commit()
            except sqlite3.Error as error:
                connection.rollback()
                print(f"  [error] {error}")
                errorCount += 1
                continue

            if action == "skip":
                skipCount += 1
                # Replace the in-progress line silently.
                print("  [skip]")
                continue
            if action == "new":
                newCount += 1
            elif action == "reindex":
                reindexCount += 1
            segmentCount += segmentsAdded
            print(f"  [{action}] {segmentsAdded} segments")
    except KeyboardInterrupt:
        connection.commit()
        print("\ninterrupted. progress preserved up to last committed file.")
        return 130

    counters = IndexCounters(
        new=newCount, reindex=reindexCount, skip=skipCount,
        error=errorCount, segments=segmentCount,
    )
    printSummary(connection, counters, time.monotonic() - startTime)
    connection.close()
    return 0 if errorCount == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
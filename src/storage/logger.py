
"""
Store and retrieve authenticity-checker analysis history using SQLite.

The database stores analysis metadata and results, not the original image.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "analysis_history.db"


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the database and safely upgrade older database schemas."""
    path = Path(db_path)

    if str(db_path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL,
                confidence_type TEXT,
                manipulation_score REAL,
                evidence_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )

        # Upgrade databases created before detailed analysis was supported.
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(analyses)")
        }
        if "details_json" not in columns:
            connection.execute(
                "ALTER TABLE analyses "
                "ADD COLUMN details_json TEXT NOT NULL DEFAULT '{}'"
            )

        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyses_created_at "
            "ON analyses(created_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyses_verdict "
            "ON analyses(verdict)"
        )
        connection.commit()
    finally:
        connection.close()


def log_analysis(
    source_filename: str,
    result: dict[str, Any],
    db_path: str | Path = DEFAULT_DB_PATH,
    analysis_details: dict[str, Any] | None = None,
) -> int:
    """Save a verdict and optional detailed evidence; return its database ID."""
    if not isinstance(source_filename, str) or not source_filename.strip():
        raise ValueError("source_filename must be a non-empty string")

    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary")

    verdict = result.get("verdict")
    if not isinstance(verdict, str) or not verdict.strip():
        raise ValueError("result must contain a non-empty 'verdict'")

    evidence = result.get("evidence", [])
    warnings = result.get("warnings", [])

    if not isinstance(evidence, list):
        raise TypeError("result['evidence'] must be a list")
    if not isinstance(warnings, list):
        raise TypeError("result['warnings'] must be a list")
    if analysis_details is None:
        analysis_details = {}
    if not isinstance(analysis_details, dict):
        raise TypeError("analysis_details must be a dictionary or None")

    # Serialize before opening the connection so invalid JSON data fails early.
    evidence_json = json.dumps(evidence)
    warnings_json = json.dumps(warnings)
    details_json = json.dumps(analysis_details)
    created_at = datetime.now(timezone.utc).isoformat()

    initialize_database(db_path)
    connection = sqlite3.connect(str(db_path))

    try:
        cursor = connection.execute(
            """
            INSERT INTO analyses (
                created_at,
                source_filename,
                verdict,
                confidence,
                confidence_type,
                manipulation_score,
                evidence_json,
                warnings_json,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                source_filename.strip(),
                verdict,
                result.get("confidence"),
                result.get("confidence_type"),
                result.get("manipulation_score"),
                evidence_json,
                warnings_json,
                details_json,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert a database row into a readable Python dictionary."""
    return {
        "id": row[0],
        "created_at": row[1],
        "source_filename": row[2],
        "verdict": row[3],
        "confidence": row[4],
        "confidence_type": row[5],
        "manipulation_score": row[6],
        "evidence": json.loads(row[7]),
        "warnings": json.loads(row[8]),
        "details": json.loads(row[9]),
    }


def get_analysis(
    analysis_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Retrieve one analysis by ID, or None if it does not exist."""
    if isinstance(analysis_id, bool) or not isinstance(analysis_id, int):
        raise TypeError("analysis_id must be an integer")

    initialize_database(db_path)
    connection = sqlite3.connect(str(db_path))

    try:
        cursor = connection.execute(
            """
            SELECT id, created_at, source_filename, verdict, confidence,
                   confidence_type, manipulation_score, evidence_json,
                   warnings_json, details_json
            FROM analyses
            WHERE id = ?
            """,
            (analysis_id,),
        )
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        connection.close()


def get_recent_analyses(
    limit: int = 20,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return the newest analyses first."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    initialize_database(db_path)
    connection = sqlite3.connect(str(db_path))

    try:
        cursor = connection.execute(
            """
            SELECT id, created_at, source_filename, verdict, confidence,
                   confidence_type, manipulation_score, evidence_json,
                   warnings_json, details_json
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [_row_to_dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()

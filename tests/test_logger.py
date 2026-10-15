
import json

import pytest

from src.storage.logger import (
    get_analysis,
    get_recent_analyses,
    initialize_database,
    log_analysis,
)


@pytest.fixture
def db_path(tmp_path):
    """Give each test its own temporary database file."""
    return tmp_path / "test_history.db"


@pytest.fixture
def sample_result():
    return {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.0,
        "confidence_type": "heuristic_evidence_strength_not_calibrated_probability",
        "manipulation_score": 0.2,
        "evidence": ["EXIF metadata is absent (weak clue only)."],
        "warnings": ["Missing metadata does not prove manipulation."],
    }


def test_initialize_database_creates_database(db_path):
    initialize_database(db_path)

    assert db_path.exists()


def test_log_analysis_returns_id_and_saves_record(db_path, sample_result):
    analysis_id = log_analysis("sample.jpg", sample_result, db_path)

    assert isinstance(analysis_id, int)
    assert analysis_id > 0

    saved = get_analysis(analysis_id, db_path)

    assert saved is not None
    assert saved["source_filename"] == "sample.jpg"
    assert saved["verdict"] == "UNVERIFIABLE"
    assert saved["confidence"] == 0.0
    assert saved["evidence"] == sample_result["evidence"]
    assert saved["warnings"] == sample_result["warnings"]
    assert saved["created_at"]


def test_get_analysis_returns_none_for_unknown_id(db_path):
    assert get_analysis(9999, db_path) is None


def test_recent_analyses_are_newest_first(db_path, sample_result):
    first_id = log_analysis("first.jpg", sample_result, db_path)

    second_result = {
        **sample_result,
        "verdict": "MANIPULATED",
        "confidence": 0.75,
    }
    second_id = log_analysis("second.jpg", second_result, db_path)

    recent = get_recent_analyses(limit=2, db_path=db_path)

    assert [item["id"] for item in recent] == [second_id, first_id]
    assert recent[0]["source_filename"] == "second.jpg"


def test_recent_analyses_respects_limit(db_path, sample_result):
    for number in range(3):
        log_analysis(f"image_{number}.jpg", sample_result, db_path)

    recent = get_recent_analyses(limit=2, db_path=db_path)

    assert len(recent) == 2


def test_empty_filename_is_rejected(db_path, sample_result):
    with pytest.raises(ValueError):
        log_analysis("  ", sample_result, db_path)


def test_non_dictionary_result_is_rejected(db_path):
    with pytest.raises(TypeError):
        log_analysis("sample.jpg", "not a result", db_path)


def test_missing_verdict_is_rejected(db_path):
    with pytest.raises(ValueError):
        log_analysis("sample.jpg", {}, db_path)


def test_invalid_evidence_is_rejected(db_path):
    with pytest.raises(TypeError):
        log_analysis(
            "sample.jpg",
            {"verdict": "UNVERIFIABLE", "evidence": "not a list"},
            db_path,
        )


def test_invalid_limit_is_rejected(db_path):
    with pytest.raises(ValueError):
        get_recent_analyses(limit=0, db_path=db_path)


def test_stored_json_is_valid(db_path, sample_result):
    analysis_id = log_analysis("sample.jpg", sample_result, db_path)
    saved = get_analysis(analysis_id, db_path)

    # Evidence and warnings are restored as Python lists from stored JSON.
    assert isinstance(saved["evidence"], list)
    assert isinstance(saved["warnings"], list)
    assert json.dumps(saved["evidence"])

def test_logger_saves_and_retrieves_analysis_details(tmp_path):
    from src.storage.logger import log_analysis, get_analysis

    db_path = tmp_path / "details.db"
    details = {
        "source_sha256": "abc123",
        "metadata": {"width": 640, "height": 480},
        "perceptual_hash": {"hash_hex": "ff00"},
    }

    analysis_id = log_analysis(
        source_filename="sample.jpg",
        result={
            "verdict": "UNVERIFIABLE",
            "confidence": 0.25,
            "evidence": ["No trusted match found"],
            "warnings": ["Limited evidence"],
        },
        db_path=db_path,
        analysis_details=details,
    )

    saved = get_analysis(analysis_id, db_path)

    assert saved is not None
    assert saved["details"]["source_sha256"] == "abc123"
    assert saved["details"]["metadata"]["width"] == 640
    assert saved["details"]["perceptual_hash"]["hash_hex"] == "ff00"


def test_logger_migrates_existing_database(tmp_path):
    import sqlite3
    from src.storage.logger import initialize_database, log_analysis, get_analysis

    db_path = tmp_path / "old_schema.db"

    # Simulate a database created by the earlier logger version.
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL,
                confidence_type TEXT,
                manipulation_score REAL,
                evidence_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO analyses (
                created_at, source_filename, verdict, confidence,
                confidence_type, manipulation_score, evidence_json,
                warnings_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-01-01T00:00:00+00:00",
                "old.jpg",
                "UNVERIFIABLE",
                0.1,
                "heuristic",
                0.0,
                "[]",
                "[]",
            ),
        )

    initialize_database(db_path)

    # The old record should still be readable after migration.
    old_record = get_analysis(1, db_path)
    assert old_record is not None
    assert old_record["source_filename"] == "old.jpg"
    assert old_record["details"] == {}

    # The upgraded database should accept new detailed records too.
    new_id = log_analysis(
        source_filename="new.jpg",
        result={"verdict": "UNVERIFIABLE", "evidence": [], "warnings": []},
        db_path=db_path,
        analysis_details={"source_sha256": "newhash"},
    )
    new_record = get_analysis(new_id, db_path)
    assert new_record["details"]["source_sha256"] == "newhash"
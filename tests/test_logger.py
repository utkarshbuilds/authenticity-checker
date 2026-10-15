
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

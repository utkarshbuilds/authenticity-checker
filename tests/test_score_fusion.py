
import pytest

from src.fusion.score_fusion import VERDICTS, fuse_scores


def test_no_evidence_is_unverifiable():
    result = fuse_scores()

    assert result["verdict"] == "UNVERIFIABLE"
    assert result["confidence"] == 0.0


def test_exact_trusted_match_can_support_authentic():
    result = fuse_scores(exact_trusted_reference_match=True)

    assert result["verdict"] == "AUTHENTIC"
    assert result["confidence"] == 1.0


def test_high_ela_and_missing_exif_raise_manipulation_score():
    result = fuse_scores(
        metadata={"has_exif": False, "software_tag_present": False},
        forensic={"mean_difference": 20.0},
    )

    assert result["verdict"] == "MANIPULATED"
    assert result["manipulation_score"] == 0.7


def test_ela_alone_does_not_cross_default_threshold():
    result = fuse_scores(forensic={"mean_difference": 20.0})

    assert result["manipulation_score"] == 0.6
    assert result["verdict"] == "UNVERIFIABLE"


def test_synthetic_detector_can_trigger_synthetic_verdict():
    result = fuse_scores(synthetic_probability=0.92)

    assert result["verdict"] == "SYNTHETIC"
    assert result["confidence"] == 0.92


def test_weak_synthetic_score_does_not_trigger_synthetic_verdict():
    result = fuse_scores(synthetic_probability=0.65)

    assert result["verdict"] == "UNVERIFIABLE"


def test_synthetic_verdict_takes_priority_over_weak_clues():
    result = fuse_scores(
        forensic={"mean_difference": 20.0},
        synthetic_probability=0.90,
    )

    assert result["verdict"] == "SYNTHETIC"


def test_hash_distance_is_reported_but_does_not_prove_authenticity():
    result = fuse_scores(perceptual_hash_distance=2)

    assert result["verdict"] == "UNVERIFIABLE"
    assert any("Hamming distance" in item for item in result["evidence"])


def test_text_and_claim_signals_are_explained():
    result = fuse_scores(
        text={
            "has_sensational_language": True,
            "detected_sensational_phrases": ["breaking"],
        },
        claim_verification={"status": "references_found"},
    )

    assert result["verdict"] == "UNVERIFIABLE"
    assert any("sensational" in item.lower() for item in result["evidence"])
    assert any("claim-reference" in item.lower() for item in result["evidence"])


def test_custom_manipulation_threshold():
    result = fuse_scores(
        forensic={"mean_difference": 20.0},
        manipulation_threshold=0.5,
    )

    assert result["verdict"] == "MANIPULATED"


def test_invalid_synthetic_probability():
    with pytest.raises(ValueError):
        fuse_scores(synthetic_probability=1.5)


def test_invalid_metadata_type():
    with pytest.raises(TypeError):
        fuse_scores(metadata="missing exif")


def test_all_verdict_names_are_defined():
    assert VERDICTS == {
        "AUTHENTIC",
        "MANIPULATED",
        "SYNTHETIC",
        "UNVERIFIABLE",
    }

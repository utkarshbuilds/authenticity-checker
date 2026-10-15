
import pytest

from src.analyzers.claim_verifier import verify_claim


REFERENCES = {
    "ref_flood": "Heavy rainfall caused flooding in the city.",
    "ref_weather": "Heavy rainfall is expected during the storm.",
    "ref_sports": "The local team won the football match.",
}


def test_exact_match_ranks_reference_first():
    result = verify_claim(
        "Heavy rainfall caused flooding in the city.",
        REFERENCES,
    )

    assert result["status"] == "references_found"
    assert result["matched_references"][0]["reference_id"] == "ref_flood"
    assert result["matched_references"][0]["similarity_score"] == 1.0


def test_partial_match_returns_shared_keywords():
    result = verify_claim("Heavy rainfall caused flooding", REFERENCES)

    top_match = result["matched_references"][0]
    assert top_match["reference_id"] == "ref_flood"
    assert "rainfall" in top_match["matched_keywords"]
    assert top_match["similarity_score"] > 0


def test_results_are_ranked_by_similarity():
    result = verify_claim("Heavy rainfall", REFERENCES)

    scores = [
        item["similarity_score"]
        for item in result["matched_references"]
    ]
    assert scores == sorted(scores, reverse=True)


def test_unrelated_claim_has_no_close_match():
    result = verify_claim(
        "An astronaut discovered a new planet",
        REFERENCES,
    )

    assert result["status"] == "no_close_match"
    assert result["matched_references"] == []


def test_empty_claim_is_unverifiable():
    result = verify_claim("   ", REFERENCES)

    assert result["status"] == "unverifiable"
    assert result["matched_references"] == []


def test_empty_reference_set():
    result = verify_claim("A flood happened", {})

    assert result["status"] == "no_close_match"
    assert result["matched_references"] == []


def test_top_k_limits_results():
    result = verify_claim("Heavy rainfall", REFERENCES, top_k=1)

    assert len(result["matched_references"]) == 1


def test_invalid_claim_type():
    with pytest.raises(TypeError):
        verify_claim(123, REFERENCES)


def test_invalid_similarity_threshold():
    with pytest.raises(ValueError):
        verify_claim("A flood happened", REFERENCES, min_similarity=1.5)


def test_invalid_top_k():
    with pytest.raises(ValueError):
        verify_claim("A flood happened", REFERENCES, top_k=0)

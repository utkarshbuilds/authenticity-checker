
"""
Compare a user's claim with a small set of trusted reference entries.

This module finds relevant references; it does NOT prove a claim is true.
"""

from src.dsa.inverted_index import InvertedIndex


def verify_claim(
    claim: str,
    reference_entries: dict[str, str],
    min_similarity: float = 0.2,
    top_k: int = 5,
) -> dict:
    """
    Find and rank reference entries that share words with a claim.

    Args:
        claim: Text supplied by the user.
        reference_entries: Mapping of reference IDs to trusted text.
        min_similarity: Minimum Jaccard score for a close match (0 to 1).
        top_k: Maximum number of references to return.

    Returns:
        A dictionary containing the status, scores, and matched keywords.
    """
    if not isinstance(claim, str):
        raise TypeError("claim must be a string")

    if not isinstance(reference_entries, dict):
        raise TypeError("reference_entries must be a dictionary")

    if not isinstance(min_similarity, (int, float)) or isinstance(
        min_similarity, bool
    ):
        raise TypeError("min_similarity must be a number")

    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    # Build the index and validate every reference entry.
    index = InvertedIndex()

    for reference_id, text in reference_entries.items():
        if not isinstance(reference_id, str) or not reference_id.strip():
            raise ValueError("reference IDs must be non-empty strings")

        if not isinstance(text, str):
            raise TypeError("every reference text must be a string")

        index.add_document(reference_id, text)

    claim_words = set(index.tokenize(claim))

    # Without searchable words, there is nothing to compare.
    if not claim_words:
        return {
            "claim": claim,
            "status": "unverifiable",
            "message": "The claim contains no searchable words.",
            "matched_references": [],
        }

    # Use the inverted index to collect references matching ANY claim word.
    candidate_ids = set()
    for word in claim_words:
        candidate_ids.update(index.search(word))

    results = []

    for reference_id in candidate_ids:
        reference_text = index.documents[reference_id]
        reference_words = set(index.tokenize(reference_text))

        shared_words = claim_words & reference_words
        all_words = claim_words | reference_words

        # Jaccard similarity = shared words / all distinct words.
        score = len(shared_words) / len(all_words) if all_words else 0.0

        results.append(
            {
                "reference_id": reference_id,
                "reference_text": reference_text,
                "similarity_score": round(score, 4),
                "matched_keywords": sorted(shared_words),
            }
        )

    # Highest score first; reference ID breaks ties consistently.
    results.sort(
        key=lambda item: (-item["similarity_score"], item["reference_id"])
    )
    results = results[:top_k]

    close_match_found = bool(results) and (
        results[0]["similarity_score"] >= min_similarity
    )

    return {
        "claim": claim,
        "status": "references_found" if close_match_found else "no_close_match",
        "message": (
            "Relevant reference text was found; this does not prove the claim."
            if close_match_found
            else "No sufficiently similar reference was found."
        ),
        "matched_references": results,
    }

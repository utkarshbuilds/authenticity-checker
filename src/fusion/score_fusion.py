
"""
Combine authenticity signals into an explainable verdict.

Important:
- These are transparent heuristics, not scientifically calibrated probabilities.
- ELA and metadata clues cannot prove an image was manipulated.
- A perceptual-hash match indicates visual similarity, not authenticity.
"""

from typing import Any


VERDICTS = {
    "AUTHENTIC",
    "MANIPULATED",
    "SYNTHETIC",
    "UNVERIFIABLE",
}


def fuse_scores(
    metadata: dict[str, Any] | None = None,
    forensic: dict[str, Any] | None = None,
    perceptual_hash_distance: int | None = None,
    text: dict[str, Any] | None = None,
    claim_verification: dict[str, Any] | None = None,
    exact_trusted_reference_match: bool = False,
    synthetic_probability: float | None = None,
    manipulation_threshold: float = 0.65,
) -> dict[str, Any]:
    """
    Combine available evidence into a verdict and explanation.

    synthetic_probability must come from a separate synthetic-image
    detector. The current project analyzers do not produce this value.

    exact_trusted_reference_match must mean an independently confirmed
    exact match, not merely a perceptual-hash match.
    """
    for name, value in (
        ("metadata", metadata),
        ("forensic", forensic),
        ("text", text),
        ("claim_verification", claim_verification),
    ):
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"{name} must be a dictionary or None")

    if not isinstance(exact_trusted_reference_match, bool):
        raise TypeError("exact_trusted_reference_match must be a boolean")

    if synthetic_probability is not None:
        if (
            isinstance(synthetic_probability, bool)
            or not isinstance(synthetic_probability, (int, float))
        ):
            raise TypeError("synthetic_probability must be a number or None")
        if not 0 <= synthetic_probability <= 1:
            raise ValueError("synthetic_probability must be between 0 and 1")

    if (
        isinstance(manipulation_threshold, bool)
        or not isinstance(manipulation_threshold, (int, float))
    ):
        raise TypeError("manipulation_threshold must be a number")
    if not 0 <= manipulation_threshold <= 1:
        raise ValueError("manipulation_threshold must be between 0 and 1")

    if perceptual_hash_distance is not None:
        if (
            isinstance(perceptual_hash_distance, bool)
            or not isinstance(perceptual_hash_distance, int)
            or perceptual_hash_distance < 0
        ):
            raise ValueError("perceptual_hash_distance must be a non-negative integer")

    metadata = metadata or {}
    forensic = forensic or {}
    text = text or {}
    claim_verification = claim_verification or {}

    score = 0.0
    evidence = []
    warnings = []

    # ELA differences are clues, not proof. Cap their contribution at 0.60.
    mean_difference = forensic.get("mean_difference")
    if mean_difference is not None:
        if (
            isinstance(mean_difference, bool)
            or not isinstance(mean_difference, (int, float))
            or mean_difference < 0
        ):
            raise ValueError("mean_difference must be a non-negative number")

        ela_contribution = 0.60 * min(mean_difference / 20.0, 1.0)
        score += ela_contribution
        evidence.append(
            f"ELA mean difference: {mean_difference} "
            f"(contribution {ela_contribution:.2f})"
        )
        warnings.append("ELA differences can also occur in ordinary JPEG images.")

    # Missing EXIF metadata is common, so its contribution is small.
    if metadata and not metadata.get("has_exif", False):
        score += 0.10
        evidence.append("EXIF metadata is absent (weak clue only).")

    # Editing software can be used for harmless edits too.
    if metadata.get("software_tag_present", False):
        score += 0.20
        evidence.append("An editing-software EXIF tag is present (not proof).")

    score = round(min(score, 1.0), 4)

    # A pHash distance is useful for similarity search, but cannot by itself
    # tell whether an image is authentic or manipulated.
    if perceptual_hash_distance is not None:
        evidence.append(
            f"Perceptual-hash Hamming distance to a reference: "
            f"{perceptual_hash_distance} (similarity clue only)."
        )

    # Text signals help explain the caption but do not determine image origin.
    if text.get("has_sensational_language", False):
        phrases = text.get("detected_sensational_phrases", [])
        evidence.append(f"Potentially sensational wording detected: {phrases}.")
        warnings.append("Sensational wording does not establish whether a claim is false.")

    claim_status = claim_verification.get("status")
    if claim_status:
        evidence.append(f"Claim-reference search status: {claim_status}.")
        warnings.append("Keyword similarity does not prove a claim is true.")

    # Prefer a strong, explicit synthetic-detector result over weak heuristics.
    if synthetic_probability is not None and synthetic_probability >= 0.80:
        verdict = "SYNTHETIC"
        confidence = round(float(synthetic_probability), 4)
        evidence.append(
            f"Supplied synthetic-detector score: {synthetic_probability:.2f}."
        )
        warnings.append(
            "This score is only as reliable as the external detector and its calibration."
        )

    elif score >= manipulation_threshold:
        verdict = "MANIPULATED"
        confidence = score
        evidence.append(
            f"Combined heuristic manipulation score reached the threshold "
            f"of {manipulation_threshold:.2f}."
        )
        warnings.append("The heuristic score is not a probability of manipulation.")

    elif exact_trusted_reference_match:
        verdict = "AUTHENTIC"
        confidence = round(1.0 - score, 4)
        evidence.append("An exact trusted-reference match was reported.")
        warnings.append(
            "Authenticity depends on the reference being trustworthy and the "
            "match being independently confirmed."
        )

    else:
        verdict = "UNVERIFIABLE"
        confidence = 0.0
        evidence.append(
            "Available signals do not establish authenticity, manipulation, "
            "or synthetic origin."
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "confidence_type": "heuristic_evidence_strength_not_calibrated_probability",
        "manipulation_score": score,
        "evidence": evidence,
        "warnings": warnings,
    }

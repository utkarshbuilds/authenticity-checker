
"""Run the authenticity-checking modules as one pipeline."""

import hashlib
from pathlib import Path
from typing import Any

from src.analyzers.claim_verifier import verify_claim
from src.analyzers.forensic_analyzer import analyze_ela
from src.analyzers.metadata_analyzer import analyze_metadata
from src.analyzers.perceptual_hash_analyzer import analyze_perceptual_hash
from src.analyzers.text_analyzer import analyze_text
from src.fusion.score_fusion import fuse_scores
from src.storage.logger import DEFAULT_DB_PATH, log_analysis


def _sha256_file(image_path: Path) -> str:
    """Calculate a cryptographic fingerprint of the original file."""
    digest = hashlib.sha256()

    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _hamming_distance(first: str, second: str) -> int:
    """Count different bits in two equal-length binary hashes."""
    if len(first) != len(second):
        raise ValueError("Reference hashes must have equal lengths")
    if set(first + second) - {"0", "1"}:
        raise ValueError("Reference hashes must contain only 0 and 1")

    return sum(bit_a != bit_b for bit_a, bit_b in zip(first, second))


def analyze_content(
    image_path: str | Path,
    caption: str = "",
    claim: str = "",
    reference_entries: dict[str, str] | None = None,
    reference_phashes: dict[str, str] | None = None,
    trusted_sha256_hashes: set[str] | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    ela_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Analyze an image and optional text, then save the result.

    reference_entries:
        Mapping of reference IDs to trusted text used for claim search.

    reference_phashes:
        Mapping of reference IDs to binary perceptual hashes.

    trusted_sha256_hashes:
        SHA-256 fingerprints of independently verified trusted files.
        An exact match supports a reference match, not universal proof
        that the image is authentic.
    """
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not isinstance(caption, str) or not isinstance(claim, str):
        raise TypeError("caption and claim must be strings")

    reference_entries = reference_entries or {}
    reference_phashes = reference_phashes or {}
    trusted_sha256_hashes = trusted_sha256_hashes or set()

    # 1. Collect image metadata and perceptual hash.
    metadata = analyze_metadata(path)
    perceptual_hash = analyze_perceptual_hash(path)

    # 2. Run ELA only for JPEG files. ELA is not supported for other formats.
    forensic = None
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        forensic = analyze_ela(path, output_path=ela_output_path)

    # 3. Analyze optional caption and claim.
    text_result = analyze_text(caption) if caption.strip() else None
    claim_result = (
        verify_claim(claim, reference_entries)
        if claim.strip()
        else None
    )

    # 4. Compare the perceptual hash with available reference hashes.
    binary_hash = perceptual_hash["binary_hash"]
    distances = []

    for reference_id, reference_hash in reference_phashes.items():
        distance = _hamming_distance(binary_hash, reference_hash)
        distances.append(
            {
                "reference_id": reference_id,
                "hamming_distance": distance,
            }
        )

    # Sorting is simple and adequate for our small reference collection.
    distances.sort(key=lambda item: item["hamming_distance"])
    closest_matches = distances[:5]
    closest_distance = (
        closest_matches[0]["hamming_distance"] if closest_matches else None
    )

    # A cryptographic hash match means the file bytes match a trusted file.
    file_sha256 = _sha256_file(path)
    exact_trusted_match = file_sha256 in trusted_sha256_hashes

    # 5. Combine signals into a verdict with an explanation.
    verdict_result = fuse_scores(
        metadata=metadata,
        forensic=forensic,
        perceptual_hash_distance=closest_distance,
        text=text_result,
        claim_verification=claim_result,
        exact_trusted_reference_match=exact_trusted_match,
    )

    # Keep the source fingerprint and individual signals for later review.
    analysis_record = {
        **verdict_result,
        "source_filename": path.name,
        "source_sha256": file_sha256,
        "metadata": metadata,
        "forensic": forensic,
        "perceptual_hash": perceptual_hash,
        "perceptual_hash_matches": closest_matches,
        "text_analysis": text_result,
        "claim_verification": claim_result,
        "exact_trusted_reference_match": exact_trusted_match,
    }

    # The logger stores verdict fields, evidence, warnings, and confidence.
    # Additional analyzer details are returned here for callers to inspect.
    analysis_id = log_analysis(
        source_filename=path.name,
        result=verdict_result,
        db_path=db_path,
    )
    analysis_record["analysis_id"] = analysis_id

    return analysis_record


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Analyze an image for digital-content authenticity clues."
    )
    parser.add_argument("image", help="Path to an image")
    parser.add_argument("--caption", default="", help="Optional image caption")
    parser.add_argument("--claim", default="", help="Optional claim to check")
    args = parser.parse_args()

    result = analyze_content(
        image_path=args.image,
        caption=args.caption,
        claim=args.claim,
    )

    print(json.dumps(result, indent=2, default=str))

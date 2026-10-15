
"""Generate perceptual hashes for image similarity checks."""

from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError


def analyze_perceptual_hash(
    image_path: str | Path,
    hash_size: int = 8,
) -> dict:
    """
    Generate a perceptual hash for an image.

    Args:
        image_path: Path to the image file.
        hash_size: Hash dimension. The default 8 produces 64 bits.

    Returns:
        A dictionary containing image information and its hash.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image is invalid or hash_size is invalid.
    """
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    if (
        not isinstance(hash_size, int)
        or isinstance(hash_size, bool)
        or hash_size < 1
    ):
        raise ValueError("hash_size must be a positive integer.")

    try:
        with Image.open(path) as image:
            original_format = image.format
            width, height = image.size
            image = image.convert("RGB")

            perceptual_hash = imagehash.phash(
            image,
            hash_size=hash_size,
            )

            # The hexadecimal hash is compact and easy to store.
            hash_hex = str(perceptual_hash)

            # Our BK-tree expects binary strings, so convert each
            # Boolean hash bit into "1" or "0".
            binary_hash = "".join(
                "1" if bit else "0"
                for bit in perceptual_hash.hash.flatten()
            )

            return {
                "filename": path.name,
                "format": image.format,
                "width": width,
                "height": height,
                "hash_size": hash_size,
                "hash_hex": hash_hex,
                "binary_hash": binary_hash,
                "hash_bits": len(binary_hash),
            }

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"Cannot read a valid image from: {path}"
        ) from error


"""Perform a basic Error Level Analysis (ELA) on JPEG images."""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageStat, UnidentifiedImageError


def analyze_ela(
    image_path: str | Path,
    output_path: str | Path | None = None,
    quality: int = 90,
    scale: float = 10.0,
) -> dict:
    """
    Compare a JPEG image with a recompressed copy.

    Args:
        image_path: Path to the original JPEG image.
        output_path: Optional path for saving the amplified ELA image.
        quality: JPEG quality used for recompression, from 1 to 95.
        scale: Brightness multiplier used for the saved ELA image.

    Returns:
        A dictionary containing ELA measurements and image information.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the input is invalid or parameters are out of range.
    """
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not isinstance(quality, int) or isinstance(quality, bool):
        raise ValueError("Quality must be an integer from 1 to 95.")

    if not 1 <= quality <= 95:
        raise ValueError("Quality must be between 1 and 95.")

    if (
        not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or scale <= 0
    ):
        raise ValueError("Scale must be a positive number.")

    try:
        with Image.open(path) as image:
            # ELA is intended for JPEG compression analysis.
            if image.format != "JPEG":
                raise ValueError(
                    "ELA requires a JPEG image. PNG and other formats "
                    "are not supported by this analyzer."
                )

            original = image.convert("RGB")
            original.load()

        # Save a recompressed copy in memory instead of creating a
        # temporary file on disk.
        buffer = BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        with Image.open(buffer) as recompressed_image:
            recompressed = recompressed_image.convert("RGB")
            recompressed.load()

        # Calculate the pixel-by-pixel difference between both images.
        difference = ImageChops.difference(original, recompressed)

        # Measure the average difference before amplifying brightness.
        statistics = ImageStat.Stat(difference)
        mean_difference = sum(statistics.mean) / len(statistics.mean)

        # Find the largest channel difference for an additional signal.
        extrema = difference.getextrema()
        max_difference = max(channel_max for _, channel_max in extrema)

        result = {
            "filename": path.name,
            "format": "JPEG",
            "width": original.width,
            "height": original.height,
            "quality": quality,
            "mean_difference": round(mean_difference, 4),
            "max_difference": max_difference,
            "ela_image_saved": False,
            "output_path": None,
        }

        # Optionally save a brighter version so differences are easier
        # to inspect visually. This does not change the reported metric.
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)

            amplified = ImageEnhance.Brightness(difference).enhance(scale)
            amplified.save(destination, format="PNG")

            result["ela_image_saved"] = True
            result["output_path"] = str(destination)

        return result

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"Cannot read or analyze a valid JPEG image: {path}"
        ) from error

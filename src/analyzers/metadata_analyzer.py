
"""Extract basic image metadata and EXIF information using Pillow."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError


# EXIF tag IDs mapped to readable names.
# These IDs help us identify useful metadata stored inside images.
INTERESTING_TAGS = {
    271: "camera_make",
    272: "camera_model",
    306: "datetime",
    305: "software",
}


def analyze_metadata(image_path: str | Path) -> dict:
    """
    Analyze an image and return its basic metadata and selected EXIF tags.

    Args:
        image_path: Path to the image file.

    Returns:
        A dictionary containing image properties and selected EXIF data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a valid or readable image.
    """
    path = Path(image_path)

    # Check that the file exists before trying to open it.
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        # Open the image safely. The with statement closes it automatically.
        with Image.open(path) as image:
            result = {
                "filename": path.name,
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "exif": {},
                "has_exif": False,
                "software_tag_present": False,
            }

            # Read the image's EXIF metadata.
            exif_data = image.getexif()

            # Keep only the EXIF tags that are useful to our project.
            for tag_id, value in exif_data.items():
                if tag_id in INTERESTING_TAGS:
                    tag_name = INTERESTING_TAGS[tag_id]
                    result["exif"][tag_name] = str(value)

            # Record whether any EXIF metadata exists.
            result["has_exif"] = bool(exif_data)

            # Record whether an editing-software tag was found.
            result["software_tag_present"] = (
                "software" in result["exif"]
            )

            return result

    except (UnidentifiedImageError, OSError) as error:
        # Convert Pillow's image-reading errors into a clear message.
        raise ValueError(
            f"Cannot read a valid image from: {path}"
        ) from error

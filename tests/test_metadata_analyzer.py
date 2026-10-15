
import pytest
from PIL import Image

from src.analyzers.metadata_analyzer import analyze_metadata


def test_analyze_basic_image_metadata(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 60), color="white").save(image_path)

    result = analyze_metadata(image_path)

    assert result["filename"] == "sample.png"
    assert result["format"] == "PNG"
    assert result["width"] == 100
    assert result["height"] == 60
    assert result["mode"] == "RGB"
    assert result["has_exif"] is False
    assert result["exif"] == {}


def test_missing_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        analyze_metadata(missing_path)


def test_invalid_image_file_raises_error(tmp_path):
    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_text("This is not an image", encoding="utf-8")

    with pytest.raises(ValueError):
        analyze_metadata(invalid_path)


def test_exif_camera_and_software_tags_are_extracted(tmp_path):
    image_path = tmp_path / "with_exif.jpg"

    image = Image.new("RGB", (80, 50), color="blue")
    exif = Image.Exif()
    exif[271] = "ExampleCamera"
    exif[272] = "ModelX"
    exif[305] = "ExampleEditor"
    image.save(image_path, exif=exif)

    result = analyze_metadata(image_path)

    assert result["has_exif"] is True
    assert result["exif"]["camera_make"] == "ExampleCamera"
    assert result["exif"]["camera_model"] == "ModelX"
    assert result["exif"]["software"] == "ExampleEditor"
    assert result["software_tag_present"] is True

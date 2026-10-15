
import pytest
from PIL import Image

from src.analyzers.forensic_analyzer import analyze_ela


def create_jpeg(path, size=(100, 80), color=(120, 80, 160)):
    """Create a small JPEG image for testing."""
    Image.new("RGB", size, color=color).save(
        path, format="JPEG", quality=95
    )
    return path


def test_analyze_jpeg_returns_expected_information(tmp_path):
    image_path = create_jpeg(tmp_path / "sample.jpg")

    result = analyze_ela(image_path)

    assert result["filename"] == "sample.jpg"
    assert result["format"] == "JPEG"
    assert result["width"] == 100
    assert result["height"] == 80
    assert result["quality"] == 90
    assert isinstance(result["mean_difference"], float)
    assert result["mean_difference"] >= 0
    assert 0 <= result["max_difference"] <= 255
    assert result["ela_image_saved"] is False


def test_ela_image_can_be_saved(tmp_path):
    image_path = create_jpeg(tmp_path / "sample.jpg")
    output_path = tmp_path / "results" / "ela.png"

    result = analyze_ela(image_path, output_path=output_path)

    assert result["ela_image_saved"] is True
    assert output_path.is_file()
    assert result["output_path"] == str(output_path)

    with Image.open(output_path) as ela_image:
        assert ela_image.format == "PNG"
        assert ela_image.size == (100, 80)


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_ela(tmp_path / "missing.jpg")


def test_invalid_image_file_raises_error(tmp_path):
    invalid_path = tmp_path / "invalid.jpg"
    invalid_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError):
        analyze_ela(invalid_path)


def test_non_jpeg_image_is_rejected(tmp_path):
    png_path = tmp_path / "sample.png"
    Image.new("RGB", (40, 30), color="white").save(png_path)

    with pytest.raises(ValueError, match="requires a JPEG"):
        analyze_ela(png_path)


@pytest.mark.parametrize("quality", [0, 96, -1, 100])
def test_invalid_quality_is_rejected(tmp_path, quality):
    image_path = create_jpeg(tmp_path / "sample.jpg")

    with pytest.raises(ValueError, match="Quality"):
        analyze_ela(image_path, quality=quality)


def test_zero_scale_is_rejected(tmp_path):
    image_path = create_jpeg(tmp_path / "sample.jpg")

    with pytest.raises(ValueError, match="Scale"):
        analyze_ela(image_path, scale=0)

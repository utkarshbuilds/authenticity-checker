
import pytest
from PIL import Image

from src.analyzers.perceptual_hash_analyzer import (
    analyze_perceptual_hash,
)


def test_generates_64_bit_hash_by_default(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 80), color="white").save(image_path)

    result = analyze_perceptual_hash(image_path)

    assert result["filename"] == "sample.png"
    assert result["width"] == 100
    assert result["height"] == 80
    assert result["hash_size"] == 8
    assert result["hash_bits"] == 64
    assert len(result["hash_hex"]) == 16
    assert len(result["binary_hash"]) == 64
    assert set(result["binary_hash"]) <= {"0", "1"}


def test_same_image_produces_same_hash(tmp_path):
    image_path = tmp_path / "same.png"
    Image.new("RGB", (100, 100), color="blue").save(image_path)

    first = analyze_perceptual_hash(image_path)
    second = analyze_perceptual_hash(image_path)

    assert first["hash_hex"] == second["hash_hex"]
    assert first["binary_hash"] == second["binary_hash"]


def test_custom_hash_size(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (50, 50), color="red").save(image_path)

    result = analyze_perceptual_hash(image_path, hash_size=4)

    assert result["hash_size"] == 4
    assert result["hash_bits"] == 16
    assert len(result["binary_hash"]) == 16


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_perceptual_hash(tmp_path / "missing.png")


def test_invalid_image_raises_error(tmp_path):
    image_path = tmp_path / "invalid.png"
    image_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot read"):
        analyze_perceptual_hash(image_path)


@pytest.mark.parametrize("hash_size", [0, -1, 1.5, True])
def test_invalid_hash_size_raises_error(tmp_path, hash_size):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 20), color="white").save(image_path)

    with pytest.raises(ValueError, match="hash_size"):
        analyze_perceptual_hash(image_path, hash_size=hash_size)

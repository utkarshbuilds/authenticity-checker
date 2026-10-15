
import hashlib

import pytest
from PIL import Image

from src.main import analyze_content
from src.storage.logger import get_analysis


@pytest.fixture
def sample_jpeg(tmp_path):
    """Create a tiny JPEG image for pipeline tests."""
    image_path = tmp_path / "sample.jpg"
    image = Image.new("RGB", (64, 64), color=(120, 160, 200))
    image.save(image_path, format="JPEG")
    return image_path


@pytest.fixture
def sample_png(tmp_path):
    """Create a PNG to verify that ELA is skipped for non-JPEG files."""
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (64, 64), color=(40, 100, 160))
    image.save(image_path, format="PNG")
    return image_path


def test_pipeline_analyzes_jpeg_and_logs_result(sample_jpeg, tmp_path):
    db_path = tmp_path / "pipeline.db"

    result = analyze_content(
        image_path=sample_jpeg,
        caption="A person standing near a building.",
        claim="A person is standing near a building.",
        reference_entries={
            "reference-1": "A person is standing near a building."
        },
        db_path=db_path,
    )

    assert result["source_filename"] == "sample.jpg"
    assert result["metadata"]["width"] == 64
    assert result["forensic"] is not None
    assert result["perceptual_hash"]["binary_hash"]
    assert result["text_analysis"] is not None
    assert result["claim_verification"] is not None
    assert result["verdict"] in {
        "AUTHENTIC", "MANIPULATED", "SYNTHETIC", "UNVERIFIABLE"
    }

    saved = get_analysis(result["analysis_id"], db_path)
    assert saved is not None
    assert saved["source_filename"] == "sample.jpg"
    assert saved["verdict"] == result["verdict"]


def test_pipeline_skips_ela_for_png(sample_png, tmp_path):
    result = analyze_content(
        image_path=sample_png,
        db_path=tmp_path / "png_pipeline.db",
    )

    assert result["forensic"] is None
    assert result["metadata"]["width"] == 64
    assert result["verdict"] == "UNVERIFIABLE"


def test_pipeline_records_sha256_fingerprint(sample_jpeg, tmp_path):
    expected_hash = hashlib.sha256(sample_jpeg.read_bytes()).hexdigest()

    result = analyze_content(
        image_path=sample_jpeg,
        db_path=tmp_path / "hash_pipeline.db",
    )

    assert result["source_sha256"] == expected_hash


def test_pipeline_rejects_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_content(
            image_path=tmp_path / "does_not_exist.jpg",
            db_path=tmp_path / "missing_pipeline.db",
        )


def test_pipeline_rejects_non_string_caption(sample_jpeg, tmp_path):
    with pytest.raises(TypeError):
        analyze_content(
            image_path=sample_jpeg,
            caption=123,
            db_path=tmp_path / "invalid_pipeline.db",
        )


"""Tests for the image evaluation module."""

import pandas as pd
import pytest

from evaluation import evaluate as evaluator


def create_manifest(tmp_path, rows):
    """Create dummy image files and a CSV manifest for testing."""
    project_root = tmp_path / "project"
    project_root.mkdir()

    for row in rows:
        image_path = project_root / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"dummy image bytes")

    manifest_path = project_root / "labels.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    return project_root, manifest_path


def test_evaluate_dataset_calculates_metrics(tmp_path, monkeypatch):
    rows = [
        {"image_path": "images/a.jpg", "true_label": "AUTHENTIC"},
        {"image_path": "images/b.jpg", "true_label": "MANIPULATED"},
        {"image_path": "images/c.jpg", "true_label": "SYNTHETIC"},
        {"image_path": "images/d.jpg", "true_label": "UNVERIFIABLE"},
    ]
    project_root, manifest_path = create_manifest(tmp_path, rows)

    # These predictions are intentionally mocked, not real detector results.
    fake_predictions = {
        "a.jpg": "AUTHENTIC",
        "b.jpg": "AUTHENTIC",
        "c.jpg": "SYNTHETIC",
        "d.jpg": "MANIPULATED",
    }

    def fake_analyze_content(image_path, db_path):
        return {
            "verdict": fake_predictions[image_path.name],
            "confidence": 0.75,
            "analysis_id": 1,
        }

    monkeypatch.setattr(evaluator, "analyze_content", fake_analyze_content)

    report = evaluator.evaluate_dataset(
        manifest_path=manifest_path,
        project_root=project_root,
        db_path=tmp_path / "test_history.db",
    )

    assert report["dataset_size"] == 4
    assert report["accuracy"] == pytest.approx(0.5)
    assert 0.0 <= report["macro_precision"] <= 1.0
    assert 0.0 <= report["macro_recall"] <= 1.0
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert len(report["confusion_matrix"]) == 4
    assert all(len(row) == 4 for row in report["confusion_matrix"])
    assert len(report["predictions"]) == 4


def test_evaluate_dataset_rejects_unknown_label(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "sample.jpg").write_bytes(b"dummy image bytes")

    manifest_path = project_root / "labels.csv"
    pd.DataFrame(
        [{"image_path": "sample.jpg", "true_label": "MAYBE"}]
    ).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="Unknown label"):
        evaluator.evaluate_dataset(
            manifest_path=manifest_path,
            project_root=project_root,
            db_path=tmp_path / "history.db",
        )


def test_evaluate_dataset_rejects_missing_image(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()

    manifest_path = project_root / "labels.csv"
    pd.DataFrame(
        [{"image_path": "missing.jpg", "true_label": "AUTHENTIC"}]
    ).to_csv(manifest_path, index=False)

    with pytest.raises(FileNotFoundError, match="was not found"):
        evaluator.evaluate_dataset(
            manifest_path=manifest_path,
            project_root=project_root,
            db_path=tmp_path / "history.db",
        )


def test_evaluate_dataset_rejects_missing_csv_column(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    manifest_path = project_root / "labels.csv"

    pd.DataFrame([{"image_path": "sample.jpg"}]).to_csv(
        manifest_path, index=False
    )

    with pytest.raises(ValueError, match="missing required column"):
        evaluator.evaluate_dataset(
            manifest_path=manifest_path,
            project_root=project_root,
            db_path=tmp_path / "history.db",
        )

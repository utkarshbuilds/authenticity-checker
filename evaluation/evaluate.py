
"""Evaluate image-authenticity predictions against labelled images.

Run from the project root:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --manifest data/raw/image_labels.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.main import analyze_content


# Keep this order fixed so confusion-matrix rows and columns are predictable.
LABELS = ["AUTHENTIC", "MANIPULATED", "SYNTHETIC", "UNVERIFIABLE"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "raw" / "image_labels.csv"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "evaluation_history.db"


def evaluate_dataset(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    project_root: str | Path = PROJECT_ROOT,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Run the pipeline on each labelled image and calculate evaluation metrics.

    Image paths in the CSV are resolved relative to project_root.
    Raises ValueError or FileNotFoundError when the manifest is invalid.
    """
    manifest_path = Path(manifest_path)
    project_root = Path(project_root)
    db_path = Path(db_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {manifest_path}")

    # Read labels and paths from the CSV.
    dataframe = pd.read_csv(manifest_path)
    required_columns = {"image_path", "true_label"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "CSV is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    if dataframe.empty:
        raise ValueError("The dataset CSV contains no image rows.")

    if dataframe["image_path"].isna().any():
        raise ValueError("The image_path column contains an empty value.")

    if dataframe["true_label"].isna().any():
        raise ValueError("The true_label column contains an empty value.")

    # Normalize labels to uppercase and reject unexpected labels.
    dataframe["true_label"] = (
        dataframe["true_label"].astype(str).str.strip().str.upper()
    )
    invalid_labels = sorted(set(dataframe["true_label"]) - set(LABELS))
    if invalid_labels:
        raise ValueError(
            f"Unknown label(s): {invalid_labels}. Allowed labels: {LABELS}"
        )

    predictions = []

    for row_number, row in dataframe.iterrows():
        relative_path = Path(str(row["image_path"]).strip())

        # Absolute paths are accepted too; relative paths use project_root.
        image_path = (
            relative_path
            if relative_path.is_absolute()
            else project_root / relative_path
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image on CSV row {row_number + 2} was not found: {image_path}"
            )

        # We do not provide a caption or claim for image-only evaluation.
        result = analyze_content(
            image_path=image_path,
            db_path=db_path,
        )

        predicted_label = str(result.get("verdict", "")).strip().upper()
        if predicted_label not in LABELS:
            raise ValueError(
                f"Pipeline returned an unsupported verdict for {image_path}: "
                f"{predicted_label!r}"
            )

        predictions.append(
            {
                "image_path": str(relative_path),
                "true_label": row["true_label"],
                "predicted_label": predicted_label,
                "confidence": result.get("confidence"),
                "analysis_id": result.get("analysis_id"),
            }
        )

    y_true = [item["true_label"] for item in predictions]
    y_pred = [item["predicted_label"] for item in predictions]

    # Macro averages calculate each class's score first, then average classes
    # equally. This is useful when some classes have fewer examples.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    report = {
        "dataset_size": len(predictions),
        "labels": LABELS,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "predictions": predictions,
        "limitations": [
            "Metrics describe only the images in this labelled dataset.",
            "The pipeline uses heuristic signals, not a validated general-purpose "
            "AI-image detector.",
            "A passing test suite does not establish real-world detection accuracy.",
        ],
    }

    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate the image-authenticity pipeline on labelled images."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV with image_path and true_label columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the report as JSON.",
    )
    args = parser.parse_args()

    report = evaluate_dataset(args.manifest)
    output_text = json.dumps(report, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Evaluation report saved to: {args.output}")

    print(output_text)


if __name__ == "__main__":
    main()

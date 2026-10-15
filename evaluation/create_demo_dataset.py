
"""Create tiny demo images and a CSV manifest for testing the evaluator.

These are synthetic test fixtures, NOT a valid real-world benchmark.
Run from the project root:
    python -m evaluation.create_demo_dataset
"""

import csv
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "demo"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "image_labels.csv"


def create_demo_dataset() -> None:
    """Create simple images and a labelled CSV manifest."""
    examples = [
        ("authentic", "AUTHENTIC", "A demo labelled authentic"),
        ("manipulated", "MANIPULATED", "A demo labelled manipulated"),
        ("synthetic", "SYNTHETIC", "A demo labelled synthetic"),
    ]

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for folder, label, caption in examples:
        image_dir = DATASET_DIR / folder
        image_dir.mkdir(parents=True, exist_ok=True)

        image_path = image_dir / f"{folder}.jpg"

        # Create a simple, deterministic picture with a label.
        image = Image.new("RGB", (256, 160), color="white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 236, 140), outline="black", width=3)
        draw.text((35, 65), caption, fill="black")
        image.save(image_path, format="JPEG", quality=95)

        rows.append(
            {
                "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "true_label": label,
                "source_dataset": "demo_fixture_not_benchmark",
            }
        )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_path", "true_label", "source_dataset"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {len(rows)} demo images.")
    print(f"Manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print("Reminder: demo labels are placeholders, not verified ground truth.")


if __name__ == "__main__":
    create_demo_dataset()

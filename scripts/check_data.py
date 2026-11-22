"""Read-only audit of the UC Food Waste dataset layout."""

from __future__ import annotations

import os
import csv
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UC_FOODWASTE_DATA_ROOT", PROJECT_ROOT / "data")
).expanduser()
NPZ_DIR = Path(
    os.environ.get("UC_FOODWASTE_NPZ_DIR", DATA_ROOT / "data_processed_multimodal" / "images")
).expanduser()
SPLIT_CSV = Path(
    os.environ.get("UC_FOODWASTE_SPLIT_CSV", PROJECT_ROOT / "metadata" / "data_split.csv")
).expanduser()
RGB_DIR = DATA_ROOT / "data_processed" / "images"
MASK_DIR = DATA_ROOT / "data_processed" / "masks"


def report_pairs(label: str, folder: Path, suffix: str, expected: set[str]) -> None:
    found = {path.stem for path in folder.glob(f"*{suffix}")} if folder.is_dir() else set()
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    print(f"{label}: {len(found)} files; missing={len(missing)}, extra={len(extra)}")
    if missing:
        print("  Missing examples:", ", ".join(missing[:5]))
    if extra:
        print("  Extra examples:", ", ".join(extra[:5]))


def main() -> int:
    print(f"Project: {PROJECT_ROOT}")
    print(f"Data root: {DATA_ROOT}")
    print(f"NPZ folder: {NPZ_DIR}")

    if not SPLIT_CSV.is_file():
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV}")

    with SPLIT_CSV.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"filename", "split"}
    absent = required - set(rows[0]) if rows else required
    if absent:
        raise ValueError(f"CSV is missing required columns: {sorted(absent)}")

    expected = {str(row["filename"]) for row in rows}
    split_counts = Counter(row["split"] for row in rows)
    print(f"Split rows: {len(rows)}; unique filenames: {len(expected)}")
    print("Split counts:", dict(sorted(split_counts.items())))
    if len(expected) != len(rows):
        print("WARNING: duplicate filenames found in split CSV")

    report_pairs("RGB JPG", RGB_DIR, ".jpg", expected)
    report_pairs("Mask PNG", MASK_DIR, ".png", expected)

    if NPZ_DIR.is_dir():
        report_pairs("Multimodal NPZ", NPZ_DIR, ".npz", expected)
        samples = sorted(NPZ_DIR.glob("*.npz"))
        if samples:
            with np.load(samples[0]) as archive:
                print(f"NPZ sample {samples[0].name}: keys={archive.files}")
                if "image" not in archive:
                    raise KeyError("NPZ archive has no 'image' array")
                image = archive["image"]
            print(
                f"  shape={image.shape}, dtype={image.dtype}, "
                f"all_finite={bool(np.isfinite(image).all())}"
            )
            if image.shape != (512, 768, 19):
                print("WARNING: expected NPZ shape is (512, 768, 19)")
    else:
        print("Multimodal NPZ folder is not connected; RGB/Mask/CSV checks still ran.")
        print("Set UC_FOODWASTE_NPZ_DIR to the shared Drive images folder when using Colab.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit the fixed split and stage a separate semantic dataset."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

EXPECTED = {"train": (224, 90, 30, 30), "val": (50, 20, 4, 10), "test": (49, 19, 6, 10)}
SPLIT_LF_SHA256 = "859dcb499be4519087815814d1abd1cdcb00377c9b551a056883047f6a0059ba"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(canonical).hexdigest() != SPLIT_LF_SHA256:
        raise ValueError("Split CSV contents differ from the fixed collaborator split")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [r["filename"] for r in rows]
    if len(set(names)) != len(rows) or any(not n.isascii() or not n.replace("_", "").isalnum() for n in names):
        raise ValueError("Invalid or duplicate filename in split")
    counts = Counter((r["split"], r["group"]) for r in rows)
    if counts != {(s, g): n for s, sizes in EXPECTED.items() for g, n in zip("ABCD", sizes)}:
        raise ValueError("Split/group counts differ from the fixed protocol")
    return rows


def label(mask, connectivity):
    return ndimage.label(mask, structure=ndimage.generate_binary_structure(2, connectivity // 4))


def overlay(image, positive):
    rgb = np.array(image)
    rgb[positive] = (0.45 * rgb[positive] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
    return Image.fromarray(rgb)


def rgb_path(root, name):
    matches = [root / "images" / f"{name}{suffix}" for suffix in (".jpg", ".png")]
    present = [path for path in matches if path.is_file()]
    if len(present) != 1:
        raise ValueError(f"Expected exactly one RGB JPG or PNG for {name} in {root}")
    return present[0]


def audit(root, split_csv, report, config, stage=None, include_test=False):
    all_rows = read_rows(split_csv)
    rows = [row for row in all_rows if include_test or row["split"] != "test"]
    names = {r["filename"] for r in rows}
    known = {r["filename"] for r in all_rows}
    splits = tuple(k for k in EXPECTED if include_test or k != "test")
    for folder, suffixes in (("images", {".jpg", ".png"}), ("masks", {".png"})):
        actual = {p.stem for p in (root / folder).glob("*") if p.suffix in suffixes}
        if names - actual or actual - known:
            raise ValueError(f"{folder}: missing={sorted(names-actual)}, extra={sorted(actual-known)}")
    views = report / "visualizations" / "ground_truth"
    views.mkdir(parents=True, exist_ok=True)
    if stage:
        stage.mkdir(parents=True)
        for split in splits:
            for folder in ("images", "masks"):
                (stage / folder / split).mkdir(parents=True)
    seen, examples, values = {}, Counter(), set()
    low, high = 255, 0
    details, regions = [], []
    size = config["width"], config["height"]
    for row in rows:
        name, split, group = (row[k] for k in ("filename", "split", "group"))
        image_path = rgb_path(root, name)
        mask_path = root / "masks" / f"{name}.png"
        with Image.open(image_path) as src:
            if src.format not in {"JPEG", "PNG"} or src.mode != "RGB" or src.size != (1536, 1024) or src.getexif().get(274, 1) != 1:
                raise ValueError(f"Unexpected RGB image metadata: {name}")
            pixels = np.asarray(src)
            image = src.resize(size, Image.Resampling.BILINEAR)
        low, high = min(low, int(pixels.min())), max(high, int(pixels.max()))
        key = hashlib.sha256(pixels.tobytes()).hexdigest()
        if key in seen and seen[key][0] != split:
            raise ValueError(f"Duplicate RGB pixels across splits: {name}, {seen[key][1]}")
        seen[key] = split, name
        with Image.open(mask_path) as src:
            if src.format != "PNG":
                raise ValueError(f"Expected PNG: {name}")
            raw = np.asarray(src)
        pixel_values = set(np.unique(raw).tolist())
        if raw.shape != (1024, 1536) or not (pixel_values <= {0, 1} or pixel_values <= {0, 255}):
            raise ValueError(f"Unexpected mask shape or class values: {name}: {raw.shape}, {pixel_values}")
        values.update(pixel_values)
        mask = Image.fromarray((raw != 0).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
        foreground = np.asarray(mask)
        labels, count = label(foreground, config["connectivity"])
        regions.extend({**row, "area_pixels": int(a)} for a in np.bincount(labels.ravel())[1:])
        details.append({**row, "contaminant_fraction": float(foreground.mean()), "regions": count,
                        "rgb_sha256": key, "mask_sha256": digest(mask_path)})
        if split != "test" and examples[split, group, bool(count)] < 1:
            overlay(image, foreground.astype(bool)).save(views / f"{split}_{group}_{name}.png")
            examples[split, group, bool(count)] += 1
        if stage:
            image.save(stage / "images" / split / f"{name}.png")
            mask.save(stage / "masks" / split / f"{name}.png")
    summary = {"split_sha256": digest(split_csv), "original_hw": [1024, 1536], "prepared_hw": list(size[::-1]),
               "source_pairs_sha256": hashlib.sha256(
                   json.dumps(details, sort_keys=True).encode("utf-8")).hexdigest(),
               "rgb_dtype": "uint8", "rgb_range": [low, high], "source_mask_values": sorted(values),
               "prepared_mask_values": [0, 1], "provisional_eda_connectivity": config["connectivity"],
               "split_counts": {k: sum(EXPECTED[k]) for k in EXPECTED}, "splits": {}}
    for split in splits:
        subset = [r for r in details if r["split"] == split]
        areas = [r["area_pixels"] for r in regions if r["split"] == split]
        summary["splits"][split] = {
            "images": len(subset), "groups": dict(Counter(r["group"] for r in subset)),
            "clean_images": sum(r["regions"] == 0 for r in subset),
            "clean_fraction": float(np.mean([r["regions"] == 0 for r in subset])),
            "mean_contaminant_fraction": float(np.mean([r["contaminant_fraction"] for r in subset])),
            "regions": len(areas), "area_quantiles_pixels": np.quantile(areas, [0,.25,.5,.75,1]).tolist() if areas else [],
        }
    write_csv(report / "dataset_images.csv", details)
    write_csv(report / "dataset_regions.csv", regions)
    write_json(report / "dataset_summary.json", summary)
    plot_eda(report, details, regions)
    if stage:
        (stage / "dataset.yaml").write_text(yaml.safe_dump({
            "train": "images/train", "val": "images/val",
            "masks_dir": "masks", "names": config["names"], "channels": 3,
        }), encoding="utf-8")
        write_json(stage / "dataset_summary.json", summary)
    return summary


def plot_eda(report, details, regions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = report / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "val"):
        axes[0].hist([r["contaminant_fraction"] for r in details if r["split"] == split],
                     bins=np.linspace(0, 1, 26), histtype="step", label=split)
        axes[1].hist([r["area_pixels"] for r in regions if r["split"] == split],
                     bins=np.geomspace(1, 512 * 768, 30), histtype="step", label=split)
    axes[0].set(xlabel="Contaminant pixel fraction", ylabel="Images")
    axes[1].set(xlabel="Region area (pixels)", ylabel="Regions", xscale="log")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "dataset_distribution.png", dpi=150)
    plt.close(fig)

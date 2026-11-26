"""Read frozen Validation results and write independent error-analysis artifacts."""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from yolo_sem.data import digest, rgb_path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def box(bounds):
    y, x = bounds
    return dict(xmin=x.start, ymin=y.start, xmax_exclusive=x.stop, ymax_exclusive=y.stop,
                width=x.stop - x.start, height=y.stop - y.start)


def quantiles(values):
    if not len(values):
        return dict.fromkeys(("mean", "p10", "p50", "p90", "p95", "max"))
    q = np.quantile(values, [.1, .5, .9, .95, 1])
    return dict(zip(("mean", "p10", "p50", "p90", "p95", "max"), [float(values.mean()), *map(float, q)]))


def dataset_item(name):
    probability = np.load(RUN / "probability_maps" / "val" / f"{name}.npy", allow_pickle=False)
    with Image.open(SOURCE / "masks" / f"{name}.png") as src:
        gt = np.asarray(src.resize((768, 512), Image.Resampling.NEAREST)) > 0
    with Image.open(RUN / "predictions" / "val" / f"{name}.png") as src:
        pred = np.asarray(src) == 255
    assert gt.shape == probability.shape == pred.shape == (512, 768)
    assert probability.dtype == np.float32 and np.isfinite(probability).all()
    assert np.array_equal(pred, probability >= np.float32(THRESHOLD))
    return gt, probability, pred


def visual(name, group, components, path, focus=None):
    gt, probability, pred = dataset_item(name)
    with Image.open(rgb_path(SOURCE, name)) as image:
        rgb = np.asarray(image.resize((768, 512), Image.Resampling.BILINEAR))
    overlay = rgb.copy()
    focus_fn = gt & ~pred
    if components:
        labels, _ = ndimage.label(gt, structure=STRUCTURE)
        focus_fn &= np.isin(labels, components)
    for mask, color in ((gt & pred, (75, 220, 105)), (pred & ~gt, (255, 172, 40)),
                        (focus_fn, (247, 45, 95))):
        overlay[mask] = (rgb[mask] * .32 + np.array(color) * .68).astype(np.uint8)
    crop = np.s_[:, :]
    if focus:
        x, y, x2, y2 = (focus[k] for k in ("xmin", "ymin", "xmax_exclusive", "ymax_exclusive"))
        xpad, ypad = max(28, (x2-x)//3), max(28, (y2-y)//3)
        crop = np.s_[max(y-ypad,0):min(y2+ypad,512), max(x-xpad,0):min(x2+xpad,768)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), layout="constrained")
    title = f"{name}  |  Group {group}  |  threshold {THRESHOLD:g}"
    if focus:
        title += f"  |  missed GT #{focus['region_id']}  |  coverage {focus['overlap_fraction']:.1%}"
    fig.suptitle(title, x=.02, ha="left")
    for ax, label, array in zip(axes, ["Source RGB", "Ground truth", "Contaminant probability", "Prediction",
                                      "TP green · FP amber · FN pink"], [rgb, gt, probability, pred, overlay]):
        ax.set_title(label, fontsize=9)
        ax.set_axis_off()
        if label == "Contaminant probability":
            heat = ax.imshow(array[crop], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            fig.colorbar(heat, ax=ax, fraction=.04, shrink=.7)
        else:
            ax.imshow(array[crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    fig.savefig(path, dpi=145, facecolor="white")
    plt.close(fig)


def main():
    global RUN, OUT, SOURCE, THRESHOLD, STRUCTURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--data-root", type=Path, default=os.getenv("UC_FOODWASTE_DATA_ROOT", ROOT / "data"))
    parser.add_argument("--split-csv", type=Path, default=ROOT / "metadata/data_split.csv")
    args = parser.parse_args()
    RUN, OUT = args.run, args.out or args.run / "error_analysis"
    SOURCE = args.data_root / "data_processed"
    rows = [row for row in read_csv(args.split_csv) if row["split"] == "val"]
    recorded = {r["filename"]: r for r in read_csv(RUN / "val_results.csv")}
    assert len(rows) == len(recorded) == 84 and {r["filename"] for r in rows} == recorded.keys()
    threshold = json.loads((RUN / "threshold.json").read_text(encoding="utf-8"))
    identity = threshold["identity"]
    config = identity["evaluation_config"]
    if (threshold["selected_on"] != "val" or digest(args.split_csv) != identity["split_sha256"]
            or (config["height"], config["width"]) != (512, 768)):
        raise ValueError("Expected frozen Validation results at H512×W768 on the specified split")
    THRESHOLD = threshold["threshold"]
    STRUCTURE = ndimage.generate_binary_structure(2, config["connectivity"] // 4)
    OUT.mkdir(parents=True)
    regions, images, blobs = [], [], []
    fp_values = defaultdict(list)
    for row in rows:
        name, group = row["filename"], row["group"]
        identity = dict(image_id=int(row["image_id"]), filename=name, group=group)
        gt, probability, pred = dataset_item(name)
        fp = pred & ~gt
        if gt.any() and fp.any():
            distances = ndimage.distance_transform_edt(~gt)[fp]
            within = {f"fp_within_{d}px": int(np.count_nonzero(distances <= d)) for d in (2, 8, 16, 32)}
        else:
            within = {f"fp_within_{d}px": 0 if gt.any() else None for d in (2, 8, 16, 32)}
        pred_labels, _ = ndimage.label(pred, structure=STRUCTURE)
        linked = np.unique(pred_labels[pred & gt])
        attached = int(np.count_nonzero(np.isin(pred_labels[fp], linked)))
        fp_labels, n_fp_blobs = ndimage.label(fp, structure=STRUCTURE)
        fp_prob = probability[fp]
        fp_values[group].append(fp_prob)
        largest = 0
        for i, bounds in enumerate(ndimage.find_objects(fp_labels), 1):
            component = fp_labels[bounds] == i
            values = probability[bounds][component]
            largest = max(largest, len(values))
            blobs.append({**identity, "fp_component_id": i, "area_px": len(values), **box(bounds),
                          **{f"prob_{key}": value for key, value in quantiles(values).items()}})
        labels, n_regions = ndimage.label(gt, structure=STRUCTURE)
        detected = 0
        for i, bounds in enumerate(ndimage.find_objects(labels), 1):
            component = labels[bounds] == i
            values = probability[bounds][component]
            hit = int(np.count_nonzero(pred[bounds][component]))
            area = len(values)
            overlap = hit / area
            detected += overlap >= config["region_overlap"]
            regions.append({**identity, "region_id": i, "area_px": area, **box(bounds),
                            "mean_probability": float(values.mean()), "max_probability": float(values.max()),
                            "overlap_px": hit, "overlap_fraction": overlap,
                            "detected": int(overlap >= config["region_overlap"])})
        tp, fn, tn = (int(mask.sum()) for mask in (pred & gt, ~pred & gt, ~pred & ~gt))
        counts = dict(tp=tp, fp=len(fp_prob), fn=fn, tn=tn, regions=n_regions, regions_detected=detected)
        assert all(recorded[name][k] == str(v) for k, v in counts.items()), (name, counts)
        images.append({**identity, **counts, "pixel_fpr": len(fp_prob)/(len(fp_prob)+tn),
                       "fp_image_fraction": len(fp_prob)/gt.size,
                       "fp_component_count": n_fp_blobs, "largest_fp_component_px": largest,
                       "fp_attached_to_gt_overlapping_prediction_px": attached, **within,
                       **{f"fp_probability_{key}": value for key, value in quantiles(fp_prob).items()}})
    missed = sorted((r for r in regions if not r["detected"]), key=lambda r: (r["area_px"], r["filename"], r["region_id"]))
    assert len(regions) == sum(int(r["regions"]) for r in recorded.values())
    save_csv(OUT / "all_regions.csv", regions)
    save_csv(OUT / "missed_regions.csv", missed)
    save_csv(OUT / "image_false_positives.csv", sorted(images, key=lambda r: (r["group"], -r["fp"])))
    save_csv(OUT / "fp_components.csv", sorted(blobs, key=lambda r: (r["group"], -r["area_px"])))

    recorded_groups = {row["group"]: row for row in read_csv(RUN / "val_groups.csv")}
    group_rows = []
    for group in "ABCD":
        subset = [r for r in images if r["group"] == group]
        g = recorded_groups[group]
        assert all(sum(r[k] for r in subset) == int(g[k]) for k in ("tp","fp","fn","tn","regions","regions_detected"))
        group_rows.append({**g, "fp_images": sum(r["fp"] > 0 for r in subset),
                           "fp_share_of_all": int(g["fp"])/sum(r["fp"] for r in images),
                           "fp_attached_to_gt_overlapping_prediction_px":
                           sum(r["fp_attached_to_gt_overlapping_prediction_px"] for r in subset),
                           **{f"fp_probability_{k}": v for k, v in quantiles(np.concatenate(fp_values[group])).items()}})
    save_csv(OUT / "group_comparison.csv", group_rows)

    # Explicit pixel-area bins fixed for this descriptive report.
    bins = [(0,1000), (1000,5000), (5000,10000), (10000,25000), (25000,None)]
    area_rows = []
    for group in ("All", "A", "C", "D"):
        for low, high in bins:
            subset = [r for r in regions if r["area_px"] >= low and (high is None or r["area_px"] < high)
                      and (group == "All" or r["group"] == group)]
            n_missed = sum(not r["detected"] for r in subset)
            area_rows.append(dict(group=group, min_area_px=low, max_area_px_exclusive=high, regions=len(subset),
                                  missed=n_missed, miss_rate=n_missed/len(subset) if subset else None))
    save_csv(OUT / "size_analysis.csv", area_rows)
    output = dict(threshold=THRESHOLD, connectivity=config["connectivity"], overlap_min=config["region_overlap"],
                  val_images=len(rows), gt_regions=len(regions),
                  missed_regions=len(missed), missed_images=len({r["filename"] for r in missed}),
                  gt_area_median_missed=median_area(missed),
                  gt_area_median_detected=median_area([r for r in regions if r["detected"]]),
                  missed_zero_overlap=sum(r["overlap_px"] == 0 for r in missed),
                  missed_max_below_threshold=sum(r["max_probability"] < float(np.float32(THRESHOLD)) for r in missed))
    (OUT / "analysis_summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    for folder in ("missed_images", "missed_region_crops", "fp_examples"):
        (OUT / folder).mkdir()
    missed_by_image = defaultdict(list)
    for r in missed:
        missed_by_image[r["filename"]].append(r)
        visual(r["filename"], r["group"], [r["region_id"]],
               OUT / "missed_region_crops" / f"{r['filename']}_r{r['region_id']}.png", r)
    for name, subset in missed_by_image.items():
        visual(name, subset[0]["group"], [r["region_id"] for r in subset], OUT / "missed_images" / f"{name}.png")
    top_a = sorted((r for r in images if r["group"] == "A"), key=lambda r: -r["fp"])[:6]
    fp_b = [r for r in images if r["group"] == "B" and r["fp"]]
    for r in top_a + fp_b:
        name = r["filename"]
        visual(name, r["group"], [], OUT / "fp_examples" / f"{r['group']}_{name}.png")
    print(json.dumps(output, indent=2))


def median_area(rows):
    return float(np.median([r["area_px"] for r in rows])) if rows else None


if __name__ == "__main__":
    main()

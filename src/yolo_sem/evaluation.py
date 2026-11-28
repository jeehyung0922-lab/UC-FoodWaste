"""Generate probability maps and evaluate with an explicitly frozen threshold."""
import csv
import json
import time

import numpy as np
from PIL import Image

from .data import digest, overlay, read_rows, write_csv, write_json
from .metrics import area_under_pro, select_threshold, summarize, sweep


def prediction_identity(args, config, output):
    checkpoint = output / "best.pt"
    summary = json.loads((args.prepared / "dataset_summary.json").read_text(encoding="utf-8"))
    if digest(args.split_csv) != summary["split_sha256"]:
        raise ValueError("Prepared dataset was built from a different split CSV")
    return {"checkpoint_sha256": digest(checkpoint), "split_sha256": summary["split_sha256"],
            "prepared_manifest_sha256": digest(args.prepared / "dataset_summary.json"),
            "evaluation_config": config}


def checked_threshold(output, identity):
    path = output / "threshold.json"
    if not path.is_file():
        raise ValueError("Validation threshold must be frozen before Test prediction/evaluation")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen["selected_on"] != "val" or frozen["identity"] != identity:
        raise ValueError("The frozen validation threshold belongs to a different model or evaluation configuration")
    return frozen["threshold"]


def write_common_input_metadata(destination, prediction):
    """Declare the probability export contract for the common Validation evaluator."""
    classes = prediction["identity"]["evaluation_config"]["names"]
    if (prediction["split"] != "val" or prediction["images"] != 84
            or prediction["tensor_shape"] != [1, 3, 512, 768]
            or classes != ["Background", "Contaminant"]):
        return
    metadata = {
        "positive_class": "Contaminant",
        "grid": "ucfw-val-512x768-aligned",
        "quantity": "probability",
        "thresholded": False,
        "probability_method": (
            "YOLO semantic logits with Background=0 and Contaminant=1. Prepared RGB and GT "
            "share the fixed H512 x W768 Validation grid (bilinear RGB resize and nearest-neighbor mask resize). "
            "Inference uses prepared RGB directly, scaled by 1/255, with no letterbox or crop. "
            "Float32 logits are resized to the input grid by bilinear interpolation "
            "(align_corners=False), followed by channel-wise softmax. Channel 1 is saved "
            "as an unthresholded float32 .npy array using the Validation image ID."
        ),
        "parameters": prediction["parameters"],
        "inference_ms": prediction["mean_inference_ms"],
    }
    with (destination / "input_metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")


def predict(args, config, output, split):
    import torch
    from ultralytics import YOLO

    from .model import probability, require_cuda

    device = require_cuda(args.device)
    identity = prediction_identity(args, config, output)
    if split == "test":
        checked_threshold(output, identity)
    destination = output / "probability_maps" / split
    destination.mkdir(parents=True, exist_ok=False)
    wrapper = YOLO(output / "best.pt")
    if wrapper.task != "semantic" or list(wrapper.names.values()) != config["names"]:
        raise ValueError("Checkpoint does not use the configured semantic classes")
    model = wrapper.model.to(device).float().eval()
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    rows = [row for row in read_rows(args.split_csv) if row["split"] == split]
    with torch.inference_mode():
        for row in rows:
            name = row["filename"]
            with Image.open(args.prepared / "images" / split / f"{name}.png") as image:
                rgb = np.array(image)
            if rgb.shape != (config["height"], config["width"], 3):
                raise ValueError(f"Unexpected prepared RGB shape: {name}")
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(device).float() / 255
            torch.cuda.synchronize(device)
            if not timings:
                probability(model, tensor)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = probability(model, tensor)
            torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
            np.save(destination / f"{name}.npy", result[0].cpu().numpy(), allow_pickle=False)
    manifest = {"identity": identity, "split": split, "images": len(rows),
               "parameters": sum(p.numel() for p in model.parameters()), "tensor_shape": list(tensor.shape),
               "mean_inference_ms": float(np.mean(timings)), "median_inference_ms": float(np.median(timings)),
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "timing_scope": "CUDA forward, logits interpolation and softmax, batch=1 fp32, one warm-up"}
    write_json(destination / "manifest.json", manifest)
    write_common_input_metadata(destination, manifest)


def evaluate(args, config, output, split, calibrate=False):
    identity = prediction_identity(args, config, output)
    metrics_path = output / f"{split}_metrics.json"
    if calibrate and (split != "val" or (output / "threshold.json").exists()):
        raise ValueError("Calibration requires Validation and a new threshold file")
    if metrics_path.exists():
        raise FileExistsError(f"{split} evaluation already exists")
    frozen = None if calibrate else checked_threshold(output, identity)
    source = output / "probability_maps" / split
    prediction = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    rows = [r for r in read_rows(args.split_csv) if r["split"] == split]
    if prediction["identity"] != identity or prediction["split"] != split or prediction["images"] != len(rows):
        raise ValueError("Probability map manifest is inconsistent with this evaluation")
    limit = config["fpr_limit"]
    grid = np.r_[np.linspace(0, 1, config["threshold_steps"] + 1),
                 float(np.nextafter(np.float32(1), np.float32(2)))]
    if frozen is not None:
        grid = np.unique(np.r_[grid, frozen])
    counts = np.zeros((len(grid), 6), dtype=np.int64)
    pro = np.zeros(len(grid))
    per_image = []
    for row in rows:
        name = row["filename"]
        with Image.open(args.prepared / "masks" / split / f"{name}.png") as image:
            mask = np.array(image)
        if mask.shape != (config["height"], config["width"]):
            raise ValueError(f"Unexpected prepared mask shape: {name}")
        probabilities = np.load(source / f"{name}.npy", allow_pickle=False)
        one, one_pro = sweep(mask, probabilities, grid, config["connectivity"], config["region_overlap"])
        counts += one
        pro += one_pro
        per_image.append((row, one, one_pro))
    index = select_threshold(grid, counts, limit) if calibrate else int(np.searchsorted(grid, frozen))
    threshold = float(grid[index])
    if calibrate:
        curve = [{"threshold": float(t), **summarize(c), "provisional_spro": float(p / c[5]) if c[5] else None}
                 for t, c, p in zip(grid, counts, pro)]
        write_csv(output / "threshold_search.csv", curve)
        plot_curve(output, curve, threshold, limit)
        result = {"threshold": threshold, "selected_on": "val", "identity": identity,
                  "tie_break": "max region recall, min FPR, highest threshold", "provisional": True}
        write_json(output / "threshold.json", result)
        return result
    views = output / "visualizations" / split
    masks = output / "predictions" / split
    views.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    image_rows, group_rows = [], []
    for row, one, one_pro in per_image:
        name = row["filename"]
        predicted = np.load(source / f"{name}.npy", allow_pickle=False) >= np.float32(threshold)
        image_rows.append({**row, **summarize(one[index]), "provisional_spro":
                           float(one_pro[index] / one[index, 5]) if one[index, 5] else None,
                           "predicted_area_fraction": float(predicted.mean())})
        Image.fromarray(predicted.astype(np.uint8) * 255).save(masks / f"{name}.png")
        with Image.open(args.prepared / "images" / split / f"{name}.png") as image:
            overlay(image, predicted).save(views / f"{name}.png")
    for group in "ABCD":
        members = [(one, one_pro) for row, one, one_pro in per_image if row["group"] == group]
        group_counts = sum(one for one, _ in members)
        group_pro = sum(one_pro for _, one_pro in members)
        n_regions = group_counts[index, 5]
        group_rows.append({"split": split, "group": group, "images": len(members),
                           **summarize(group_counts[index]),
                           "provisional_spro": float(group_pro[index] / n_regions) if n_regions else None,
                           "provisional_auspro": area_under_pro(group_counts, group_pro, limit)})
    clean = [row for row in image_rows if row["group"] == "B"]
    summary = {**summarize(counts[index]), "split": split, "images": len(rows), "threshold": threshold,
               "region_protocol": "provisional; team evaluator unavailable",
               "provisional_spro": float(pro[index] / counts[index, 5]) if counts[index, 5] else None,
               "provisional_auspro": area_under_pro(counts, pro, limit),
               "clean_group_b": {"images": len(clean),
                                 "false_alarm_image_fraction": float(np.mean([r["fp"] > 0 for r in clean])),
                                 "mean_predicted_area_fraction": float(np.mean([r["predicted_area_fraction"] for r in clean]))},
               "inference": prediction}
    write_csv(output / f"{split}_results.csv", image_rows)
    write_csv(output / f"{split}_groups.csv", group_rows)
    all_groups = []
    for name in ("val", "test"):
        path = output / f"{name}_groups.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                all_groups.extend(csv.DictReader(stream))
    write_csv(output / "group_metrics.csv", all_groups)
    write_json(metrics_path, summary)
    return summary


def plot_curve(output, curve, threshold, limit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    relevant = [row for row in curve if row["threshold"] <= 1]
    x = [row["threshold"] for row in relevant]
    axes[0].plot(x, [row["region_recall"] for row in relevant], label="Region recall")
    axes[0].plot(x, [row["pixel_fpr"] for row in relevant], label="Pixel FPR")
    axes[0].axhline(limit, color="gray", linestyle=":")
    if threshold <= 1:
        axes[0].axvline(threshold, color="black", linestyle="--")
    axes[0].set(xlabel="Validation threshold", ylabel="Fraction")
    axes[0].legend()
    axes[1].plot([row["pixel_fpr"] for row in curve[::-1]],
                 [row["provisional_spro"] for row in curve[::-1]])
    axes[1].axvline(limit, color="gray", linestyle=":")
    axes[1].set(xlabel="Pixel FPR", ylabel="Mean region overlap (provisional)")
    fig.tight_layout()
    fig.savefig(plots / "validation_threshold.png", dpi=150)
    plt.close(fig)

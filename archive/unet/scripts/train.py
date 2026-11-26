"""Train and evaluate a UC Food Waste binary segmentation model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ARCHIVE_ROOT.parents[1]
sys.path.insert(0, str(ARCHIVE_ROOT / "src"))

from ucfoodwaste.data import (  # noqa: E402
    CHANNELS_BY_MODE,
    ChannelStats,
    build_dataloaders,
    compute_channel_stats,
    read_split_csv,
)
from ucfoodwaste.engine import binary_metrics, evaluate, fit  # noqa: E402
from ucfoodwaste.model import BCEDiceLoss, build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    data_root = Path(os.environ.get("UC_FOODWASTE_DATA_ROOT", PROJECT_ROOT / "data"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=CHANNELS_BY_MODE, default="rgb")
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=Path(os.environ.get("UC_FOODWASTE_NPZ_DIR", data_root / "data_processed_multimodal" / "images")),
    )
    parser.add_argument("--mask-dir", type=Path, default=data_root / "data_processed" / "masks")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path(os.environ.get("UC_FOODWASTE_SPLIT_CSV", PROJECT_ROOT / "metadata" / "data_split.csv")),
    )
    parser.add_argument("--output-dir", type=Path, default=ARCHIVE_ROOT / "runs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize_test_metrics(results: pd.DataFrame) -> dict:
    overall = binary_metrics(int(results.tp.sum()), int(results.fp.sum()), int(results.fn.sum()))
    print("\nTest overall:", " | ".join(f"{k}={v:.4f}" for k, v in overall.items()))
    summary: dict = {"overall": overall, "groups": {}}
    if "group" in results:
        print("Test by group:")
        for group, subset in results.groupby("group", sort=True):
            values = binary_metrics(int(subset.tp.sum()), int(subset.fp.sum()), int(subset.fn.sum()))
            summary["groups"][str(group)] = {"count": len(subset), **values}
            print(f"  {group} ({len(subset)}): " + " | ".join(f"{k}={v:.4f}" for k, v in values.items()))
        clean = results.loc[results["group"] == "B"]
        if not clean.empty:
            summary["clean_group_b"] = {
                "count": len(clean),
                "false_alarm_image_pct": float((clean.fp > 0).mean() * 100),
                "mean_false_area_pct": float(clean.pred_area_pct.mean()),
            }
            print(
                f"  B false-alarm images={(clean.fp > 0).mean() * 100:.2f}% | "
                f"mean false area={clean.pred_area_pct.mean():.4f}%"
            )
    return summary


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    channels = CHANNELS_BY_MODE[args.mode]
    run_dir = args.output_dir / args.mode
    stats_path = run_dir / "channel_stats.json"
    if args.recompute_stats or not stats_path.is_file():
        print(f"Computing train-only channel statistics for {args.mode}...")
        stats = compute_channel_stats(args.npz_dir, args.split_csv, channels)
        stats.save(stats_path)
    else:
        stats = ChannelStats.load(stats_path)
    print(f"Channel stats: {stats_path}")
    if args.stats_only:
        print(json.dumps({"channels": channels, "mean": stats.mean, "std": stats.std}, indent=2))
        return 0

    loaders = build_dataloaders(
        args.npz_dir,
        args.mask_dir,
        args.split_csv,
        args.mode,
        stats,
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    sample_images, sample_masks, _ = next(iter(loaders["train"]))
    print(f"Train batch: images={tuple(sample_images.shape)}, masks={tuple(sample_masks.shape)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run_config = {
        "mode": args.mode,
        "channels": channels,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "threshold": args.threshold,
        "seed": args.seed,
        "encoder": "resnet18",
        "encoder_weights": None if args.no_pretrained else "imagenet",
        "height": 512,
        "width": 768,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    model = build_model(
        len(channels), encoder_weights=None if args.no_pretrained else "imagenet"
    ).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path, _ = fit(
        model,
        loaders,
        criterion,
        optimizer,
        device,
        run_dir,
        checkpoint_metadata={
            **run_config,
            "mean": stats.mean,
            "std": stats.std,
        },
        epochs=args.epochs,
        patience=args.patience,
        threshold=args.threshold,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_results = evaluate(
        model, loaders["test"], criterion, device, device.type == "cuda", args.threshold
    )
    metadata = read_split_csv(args.split_csv)
    detailed_results = test_results.merge(
        metadata, on="filename", how="left", validate="one_to_one"
    )
    summary = summarize_test_metrics(detailed_results)
    summary["loss"] = test_metrics["loss"]
    detailed_results.to_csv(run_dir / "test_results.csv", index=False, encoding="utf-8-sig")
    (run_dir / "test_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Artifacts saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

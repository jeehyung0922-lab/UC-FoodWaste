"""Exp01 stages. Run from PowerShell or a Colab GPU runtime."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "yolo_settings"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / "matplotlib"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import yaml

from yolo_sem.data import audit, digest
from yolo_sem.evaluation import evaluate, predict


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["environment", "import-source", "weights", "audit", "prepare", "smoke", "train",
                                         "predict", "threshold", "evaluate", "pipeline"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/yolo/exp01_rgb.yaml")
    parser.add_argument("--data-root", type=Path, default=os.getenv("UC_FOODWASTE_DATA_ROOT", ROOT / "data"))
    parser.add_argument("--split-csv", type=Path, default=ROOT / "metadata/data_split.csv")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/yolo_exp01")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--weights", type=Path, help="Local official yolo26s-sem.pt asset")
    parser.add_argument("--source-archive", type=Path, help="Original ZIP with images/*.jpg and masks/*.png")
    parser.add_argument("--source-images", type=Path, help="Folder of original .npy/.npz images")
    parser.add_argument("--source-masks", type=Path, help="Folder of original .npy/.npz masks")
    parser.add_argument("--mask-key", default="mask", help="Array key for NPZ masks")
    parser.add_argument("--prepare-test", action="store_true", help="Explicitly include Test data in preparation")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--include-test", action="store_true", help="Explicitly execute frozen-threshold test in pipeline")
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or args.device < 0:
        parser.error("batch must be positive; workers and device must be nonnegative")
    if args.resume and args.stage not in ("train", "smoke"):
        parser.error("--resume applies to train or smoke")
    if args.include_test and args.stage != "pipeline":
        parser.error("--include-test applies to pipeline")
    if args.split == "test" and args.stage not in ("predict", "evaluate"):
        parser.error("--split test applies to predict or evaluate")
    if args.weights:
        args.weights = args.weights.resolve()
        if args.weights.name != "yolo26s-sem.pt" or (args.stage != "weights" and not args.weights.is_file()):
            parser.error("--weights must point to yolo26s-sem.pt (existing except for weights stage)")
    has_arrays = bool(args.source_images and args.source_masks)
    any_arrays = bool(args.source_images or args.source_masks)
    if args.stage == "import-source":
        if (bool(args.source_archive) == has_arrays) or (any_arrays and not has_arrays):
            parser.error("Use --source-archive or both --source-images and --source-masks")
    elif args.source_archive or any_arrays:
        parser.error("Source import options require import-source")
    if args.prepare_test and args.stage not in ("import-source", "audit", "prepare", "pipeline"):
        parser.error("--prepare-test applies to data preparation")
    return args


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    fixed = dict(experiment="Exp01_RGB_Baseline", weights="yolo26s-sem.pt", height=512, width=768,
                 epochs=20, seed=42, pretrained=True, amp=True, names=["Background", "Contaminant"],
                 fpr_limit=0.02, region_overlap=0.10)
    if (not isinstance(config, dict) or any(config.get(k) != v for k, v in fixed.items()) or
            config.get("connectivity") not in (4, 8) or type(config.get("threshold_steps")) is not int or
            config["threshold_steps"] < 1):
        raise ValueError("Configuration violates the fixed Exp01 protocol")
    return config


def main():
    args = parse()
    config = load_config(args.config)
    output = args.run_root / config["experiment"]
    smoke = args.run_root / (config["experiment"] + "_smoke")
    report = args.run_root / "yolo_audit"
    source = args.data_root / "data_processed"
    if args.stage == "import-source":
        from yolo_sem.assets import materialize_source
        print(f"Imported {materialize_source(args.data_root, args.split_csv, args.source_archive, args.source_images, args.source_masks, args.mask_key, args.prepare_test)} pairs into {source}")
        return
    if args.stage == "weights":
        from yolo_sem.assets import official_weights
        print(official_weights(args.weights or args.data_root / "yolo_weights" / config["weights"]))
        return
    if args.stage == "environment":
        from yolo_sem.model import environment
        print(json.dumps(environment(), indent=2))
        return
    if args.stage in ("audit", "prepare"):
        stage = args.prepared if args.stage == "prepare" else None
        print(json.dumps(audit(source, args.split_csv, report, config, stage, args.prepare_test), indent=2))
        return
    if args.stage in ("smoke", "train", "pipeline"):
        from yolo_sem.model import environment, require_cuda, train
        require_cuda(args.device)
        if args.stage == "pipeline":
            print(json.dumps(environment(), indent=2))
            include_test = args.prepare_test or args.include_test
            audit(source, args.split_csv, report, config, None if args.prepared.exists() else args.prepared, include_test)
        prepared = json.loads((args.prepared / "dataset_summary.json").read_text(encoding="utf-8"))
        if prepared["split_sha256"] != digest(args.split_csv):
            raise ValueError("Prepared dataset does not match the configured split")
        if args.stage == "pipeline":
            audited = json.loads((report / "dataset_summary.json").read_text(encoding="utf-8"))
            if audited["splits"] != {s: prepared["splits"].get(s) for s in audited["splits"]}:
                raise ValueError("Prepared split summaries differ from the current source audit")
            if audited["splits"].keys() == prepared["splits"].keys() and audited["source_pairs_sha256"] != prepared.get("source_pairs_sha256"):
                raise ValueError("Prepared source fingerprint differs from the current source audit")
        if args.stage in ("smoke", "pipeline"):
            train(args, config, smoke, smoke=True)
            predict(args, config, smoke, "val")
            evaluate(args, config, smoke, "val", calibrate=True)
            evaluate(args, config, smoke, "val")
        if args.stage in ("train", "pipeline"):
            train(args, config, output, smoke=False)
    if args.stage in ("predict", "pipeline"):
        predict(args, config, output, args.split)
    if args.stage in ("threshold", "pipeline"):
        result = evaluate(args, config, output, "val", calibrate=True)
        print(json.dumps(result, indent=2))
    if args.stage in ("evaluate", "pipeline"):
        result = evaluate(args, config, output, args.split)
        print(json.dumps(result, indent=2))
    if args.stage == "pipeline" and args.include_test:
        predict(args, config, output, "test")
        print(json.dumps(evaluate(args, config, output, "test"), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

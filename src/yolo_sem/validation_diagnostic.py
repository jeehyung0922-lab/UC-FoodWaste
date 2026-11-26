"""Isolate Validation loss precision from prediction precision on a frozen model."""
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.loss import SemanticSegmentationLoss
from ultralytics.utils.metrics import SemanticMetrics

from .data import digest, read_rows
from .model import RectangularSemanticDataset, require_cuda, validate_batch


def fp32_loss(criterion, logits, batch):
    return criterion(logits.float(), batch)[1]


def describe(losses):
    result = {}
    for name, column in zip(("ce", "dice", "aux"), zip(*losses)):
        finite = [v for v in column if math.isfinite(v)]
        result[name] = dict(nonfinite_batches=len(column)-len(finite),
                            mean=sum(finite)/len(finite) if len(finite)==len(column) else None)
    return result


@torch.inference_mode()
def diagnose(args, config):
    if args.out.exists():
        raise FileExistsError(args.out)
    device = require_cuda(args.device)
    summary = json.loads((args.prepared / "dataset_summary.json").read_text(encoding="utf-8"))
    if summary["split_sha256"] != digest(args.split_csv):
        raise ValueError("Prepared dataset does not match the fixed split")
    expected = {r["filename"] for r in read_rows(args.split_csv) if r["split"] == "val"}
    path = (args.prepared / "dataset.yaml").resolve()
    data = check_det_dataset(str(path), autodownload=False)
    dataset = RectangularSemanticDataset(
        img_path=data["val"], imgsz=config["width"], batch_size=args.batch, augment=False,
        hyp=get_cfg(), rect=True, stride=32, pad=0.0, task="semantic", data=data,
    )
    if len(dataset) != len(expected) or {Path(p).stem for p in dataset.im_files} != expected:
        raise ValueError("Diagnostic loader does not match the fixed Validation IDs")
    loader = build_dataloader(dataset, batch=args.batch, workers=0, shuffle=False, rank=-1)
    checkpoint = YOLO(args.checkpoint)
    if checkpoint.task != "semantic" or list(checkpoint.names.values()) != config["names"]:
        raise ValueError("Expected a trained two-class segmentation checkpoint")
    full = checkpoint.model.to(device).float().eval()
    half = copy.deepcopy(full).half().eval()
    half_criterion, full_criterion = SemanticSegmentationLoss(half), SemanticSegmentationLoss(full)
    losses = {"native_fp16": [], "fp32_loss_on_same_fp16_logits": [], "full_fp32": []}
    metrics = {name: SemanticMetrics(checkpoint.names) for name in ("native_fp16", "full_fp32")}
    disagreements = 0
    for batch in loader:
        validate_batch(batch)
        rgb = batch["img"].to(device)
        target = batch["semantic_mask"].to(device)
        gt = {"semantic_mask": target}
        half_logits = half(rgb.half() / 255)
        full_logits = full(rgb.float() / 255)
        losses["native_fp16"].append(half_criterion(half_logits, gt)[1].tolist())
        losses["fp32_loss_on_same_fp16_logits"].append(fp32_loss(full_criterion, half_logits, gt).tolist())
        losses["full_fp32"].append(fp32_loss(full_criterion, full_logits, gt).tolist())
        half_mask = F.interpolate(half_logits, target.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        full_mask = F.interpolate(full_logits, target.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        metrics["native_fp16"].update_stats(half_mask, target)
        metrics["full_fp32"].update_stats(full_mask, target)
        disagreements += int((half_mask != full_mask).sum())
    for measure in metrics.values():
        measure.process()
    result = dict(checkpoint_sha256=digest(args.checkpoint), split="val", images=len(dataset),
                  batch=args.batch, batches=len(loader), losses={k: describe(v) for k, v in losses.items()},
                  miou_native_fp16=metrics["native_fp16"].miou,
                  miou_fp32_loss_on_same_fp16_logits=metrics["native_fp16"].miou,
                  miou_full_fp32=metrics["full_fp32"].miou, prediction_disagreement_pixels=disagreements)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))

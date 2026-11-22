"""Training and evaluation loops."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    epsilon = 1e-7
    return {
        "iou": tp / (tp + fp + fn + epsilon),
        "dice": 2 * tp / (2 * tp + fp + fn + epsilon),
        "recall": tp / (tp + fn + epsilon),
        "precision": tp / (tp + fp + epsilon),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> float:
    model.train()
    loss_sum = 0.0
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            loss = criterion(model(images), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * images.shape[0]
    return loss_sum / len(loader.dataset)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    threshold: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    loss_sum = 0.0
    records: list[dict[str, int | float | str]] = []
    for images, masks, names in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
            loss = criterion(logits, masks)
        loss_sum += loss.item() * images.shape[0]
        predictions = torch.sigmoid(logits.float()) >= threshold
        targets = masks >= 0.5
        for index, name in enumerate(names):
            prediction, target = predictions[index], targets[index]
            tp = int((prediction & target).sum().item())
            fp = int((prediction & ~target).sum().item())
            fn = int((~prediction & target).sum().item())
            records.append(
                {
                    "filename": name,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_area_pct": prediction.float().mean().item() * 100,
                    "true_area_pct": target.float().mean().item() * 100,
                }
            )
    frame = pd.DataFrame(records)
    metrics = binary_metrics(int(frame.tp.sum()), int(frame.fp.sum()), int(frame.fn.sum()))
    metrics["loss"] = loss_sum / len(loader.dataset)
    return metrics, frame


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_dir: Path,
    checkpoint_metadata: dict,
    epochs: int = 50,
    patience: int = 8,
    threshold: float = 0.5,
) -> tuple[Path, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    best_iou = -1.0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, amp
        )
        val_metrics, _ = evaluate(
            model, loaders["val"], criterion, device, amp, threshold
        )
        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(history_path, index=False)
        print(
            f"Epoch {epoch:02d}/{epochs} | train {train_loss:.4f} | "
            f"val {val_metrics['loss']:.4f} | IoU {val_metrics['iou']:.4f} | "
            f"Dice {val_metrics['dice']:.4f} | Recall {val_metrics['recall']:.4f}"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_iou": best_iou,
                    **checkpoint_metadata,
                },
                checkpoint_path,
            )
            print(f"  saved best checkpoint: {checkpoint_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    return checkpoint_path, pd.DataFrame(history)

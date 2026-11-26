"""NPZ-backed datasets and dataloaders for binary segmentation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


CHANNELS_BY_MODE: dict[str, tuple[int, ...]] = {
    "rgb": (2, 1, 0),
    "rgb_thermal": (2, 1, 0, 18),
    "all19": (2, 1, 0, *range(3, 19)),
}


@dataclass(frozen=True)
class ChannelStats:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    channels: tuple[int, ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"mean": self.mean, "std": self.std, "channels": self.channels},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ChannelStats":
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mean=tuple(values["mean"]),
            std=tuple(values["std"]),
            channels=tuple(values["channels"]),
        )


def read_split_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"filename", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split CSV is missing columns: {sorted(missing)}")
    if frame["filename"].duplicated().any():
        raise ValueError("Split CSV contains duplicate filenames")
    frame["split"] = frame["split"].astype(str).str.lower()
    unknown = set(frame["split"]) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"Unknown split values: {sorted(unknown)}")
    return frame


def compute_channel_stats(
    npz_dir: Path,
    split_csv: Path,
    channels: Sequence[int],
) -> ChannelStats:
    """Calculate exact per-channel mean/std using train images only."""
    train_names = read_split_csv(split_csv).query("split == 'train'")["filename"]
    total = np.zeros(len(channels), dtype=np.float64)
    total_sq = np.zeros(len(channels), dtype=np.float64)
    pixel_count = 0

    for name in train_names:
        path = npz_dir / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Training NPZ not found: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if "image" not in archive:
                raise KeyError(f"{path} has no 'image' key")
            image = archive["image"]
        if image.ndim != 3 or image.shape[-1] != 19:
            raise ValueError(f"Expected HWC array with 19 channels, got {image.shape}: {path}")
        values = image[..., list(channels)].astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite value found in {path}")
        total += values.sum(axis=(0, 1))
        total_sq += np.square(values).sum(axis=(0, 1))
        pixel_count += image.shape[0] * image.shape[1]

    if pixel_count == 0:
        raise ValueError("Train split is empty")
    mean = total / pixel_count
    variance = np.maximum(total_sq / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std < 1e-12):
        bad = [int(channels[i]) for i in np.flatnonzero(std < 1e-12)]
        raise ValueError(f"Near-constant training channels cannot be normalized: {bad}")
    return ChannelStats(tuple(mean.tolist()), tuple(std.tolist()), tuple(channels))


class FoodWasteDataset(Dataset):
    """Load selected HWC NPZ channels and a matching binary PNG mask."""

    def __init__(
        self,
        npz_dir: Path,
        mask_dir: Path,
        split_csv: Path,
        split: str,
        channels: Sequence[int],
        stats: ChannelStats,
        augment: bool = False,
        image_size: tuple[int, int] = (512, 768),
    ) -> None:
        self.npz_dir = Path(npz_dir)
        self.mask_dir = Path(mask_dir)
        self.split = split.lower()
        self.channels = tuple(channels)
        self.augment = augment
        self.height, self.width = image_size
        if stats.channels != self.channels:
            raise ValueError("Normalization-stat channels do not match dataset channels")
        self.mean = np.asarray(stats.mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.asarray(stats.std, dtype=np.float32).reshape(1, 1, -1)
        rows = read_split_csv(Path(split_csv))
        self.rows = rows.loc[rows["split"] == self.split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split={self.split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        name = str(self.rows.iloc[index]["filename"])
        npz_path = self.npz_dir / f"{name}.npz"
        mask_path = self.mask_dir / f"{name}.png"
        with np.load(npz_path, allow_pickle=False) as archive:
            image = archive["image"]
        if image.shape != (self.height, self.width, 19):
            raise ValueError(
                f"Expected {(self.height, self.width, 19)}, got {image.shape}: {npz_path}"
            )
        image = image[..., list(self.channels)].astype(np.float32)

        with Image.open(mask_path) as opened:
            mask_image = opened.convert("L")
            if mask_image.size != (self.width, self.height):
                mask_image = mask_image.resize(
                    (self.width, self.height), Image.Resampling.NEAREST
                )
            mask = np.asarray(mask_image, dtype=np.uint8) > 0

        if self.augment and random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        image = np.ascontiguousarray((image - self.mean) / self.std)
        mask = np.ascontiguousarray(mask.astype(np.float32)[None, ...])
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask), name


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(
    npz_dir: Path,
    mask_dir: Path,
    split_csv: Path,
    mode: str,
    stats: ChannelStats,
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 42,
) -> dict[str, DataLoader]:
    channels = CHANNELS_BY_MODE[mode]
    generator = torch.Generator().manual_seed(seed)
    common = dict(
        npz_dir=npz_dir,
        mask_dir=mask_dir,
        split_csv=split_csv,
        channels=channels,
        stats=stats,
    )
    datasets = {
        split: FoodWasteDataset(**common, split=split, augment=(split == "train"))
        for split in ("train", "val", "test")
    }
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        for split, dataset in datasets.items()
    }

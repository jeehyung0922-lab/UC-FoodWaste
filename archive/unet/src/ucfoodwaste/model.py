"""Model and loss definitions matching the reference baseline."""

from __future__ import annotations

import torch
from torch import nn


def build_model(in_channels: int, encoder_weights: str | None = "imagenet") -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
        activation=None,
    )


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = (probabilities + targets).sum(dim=(1, 2, 3))
        dice = 1.0 - (
            (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        ).mean()
        return bce + dice

"""Exercise unmodified Ultralytics 8.4.52 Dice on CPU. No training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "yolo_settings"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / "matplotlib"))

import ultralytics
from ultralytics import YOLO
from ultralytics.utils.loss import SemanticSegmentationLoss


class Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.model = [SimpleNamespace(nc=2)]
        self.args = SimpleNamespace(data="dataset.yaml")


def serial(values):
    return [float(value) if torch.isfinite(value) else str(float(value)) for value in values]


def one(criterion, mask, dtype):
    logits = torch.zeros((len(mask), 2, *mask.shape[1:]), dtype=dtype)
    with torch.no_grad():
        dice = criterion._dice_loss(logits, mask)
        probs = logits.softmax(1).movedim(1, -1).reshape(-1, 2)
        targets = mask.reshape(-1).long()
        valid = targets != 255
        targets = targets[valid]
        probs = probs[valid]
        intersection = torch.zeros(2, dtype=dtype).scatter_add_(0, targets, probs.gather(1, targets[:, None])[:, 0])
        target_counts = torch.bincount(targets, minlength=2)
        prediction_mass = probs.sum(0)
        card = prediction_mass + target_counts.to(dtype)
        return dict(dtype=str(dtype), shape=list(mask.shape), target_counts=target_counts.tolist(),
                    intersection=serial(intersection), twice_intersection=serial(2 * intersection),
                    prediction_mass=serial(prediction_mass), cardinality=serial(card), dice_loss=serial([dice])[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Optionally check the best/last checkpoint tensors")
    parser.add_argument("--out", type=Path, required=True, help="New JSON file for the results")
    args = parser.parse_args()
    torch.set_num_threads(2)
    assert ultralytics.__version__ == "8.4.52"
    mix = torch.zeros(1, 512, 768, dtype=torch.int32)
    mix[:, 128:384, 192:576] = 1
    examples = {"small_empty": torch.zeros(1,64,96,dtype=torch.int32),
                "full_empty": torch.zeros_like(mix), "full_mixed": mix,
                "full_all_ignore": torch.full_like(mix,255)}
    criterion = SemanticSegmentationLoss(Stub())
    cases = {name: [one(criterion, mask, dtype) for dtype in (torch.float16, torch.float32)]
             for name, mask in examples.items()}
    assert cases["small_empty"][0]["dice_loss"] != "nan"
    assert all(cases[k][0]["dice_loss"] == "nan" for k in ("full_empty","full_mixed"))
    assert all(isinstance(rows[1]["dice_loss"], float) for rows in cases.values())
    assert cases["full_all_ignore"][0]["dice_loss"] == 0.
    for rows in cases.values():
        assert rows[1]["dice_loss"] >= 0.
    checkpoints = {}
    for name in ("best.pt","last.pt") if args.run else ():
        model = YOLO(args.run / name)
        tensors = list(model.model.state_dict().values())
        nonfinite = sum(not bool(torch.isfinite(tensor).all()) for tensor in tensors if tensor.is_floating_point())
        assert nonfinite == 0
        checkpoints[name] = dict(sha256=hashlib.sha256((args.run / name).read_bytes()).hexdigest(),
                                 tensors=len(tensors), nonfinite_float_tensors=nonfinite)
    result = dict(torch=str(torch.__version__), ultralytics=ultralytics.__version__,
                  fp16_max=torch.finfo(torch.float16).max, cases=cases, checkpoints=checkpoints)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for name, rows in cases.items():
        print(name, [(row["dtype"], row["dice_loss"], row["cardinality"]) for row in rows])
    print(checkpoints)


if __name__ == "__main__":
    main()

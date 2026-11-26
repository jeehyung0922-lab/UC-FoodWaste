"""Ultralytics 8.4.52 integration for fixed rectangular semantic inputs."""
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import SemanticDataset
from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import unwrap_model

from .data import digest, write_json
from .assets import official_weights, verify_weights

HW = (512, 768)


def environment():
    return dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                ultralytics=ultralytics.__version__, cuda_build=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), cuda_arches=torch.cuda.get_arch_list(),
                dependencies={name: version(name) for name in
                              ("torchvision", "numpy", "opencv-python", "scipy", "Pillow", "PyYAML")},
                devices=[dict(name=torch.cuda.get_device_name(i), capability=torch.cuda.get_device_capability(i))
                         for i in range(torch.cuda.device_count())])


def require_cuda(index):
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError("CUDA GPU unavailable. Enable a Colab GPU or install a CUDA PyTorch wheel and NVIDIA driver.")
    device = torch.device(f"cuda:{index}")
    try:
        (torch.ones(1, device=device) + 1).item()
    except RuntimeError as exc:
        raise RuntimeError("CUDA kernel check failed. For RTX 5090 use a Blackwell-capable PyTorch CUDA wheel.") from exc
    return device


class RectangularSemanticDataset(SemanticDataset):
    def load_image(self, i, rect_mode=True):
        # The source is already H512 x W768. Avoid the native short-side upscale on train.
        return BaseDataset.load_image(self, i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            # Native affine uses (width, height); preserve the original default affine parameters.
            transforms[0][-1].size = HW[::-1]
        return transforms


class RectangularTrainer(SemanticSegmentationTrainer):
    first_train_batch = None

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        dataset = RectangularSemanticDataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=mode == "train",
            hyp=self.args, rect=True, cache=self.args.cache or None, stride=stride, pad=0.0,
            task="semantic", data=self.data,
        )
        if not np.all(dataset.batch_shapes == HW):
            raise ValueError(f"{mode} dataset does not have the expected rectangular batch shape")
        if len(dataset) != {"train": 374, "val": 84}[mode]:
            raise ValueError(f"{mode} dataset did not load the full fixed split")
        return dataset

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.first_train_batch is None:
            self.first_train_batch = validate_batch(batch)
        return batch


def validate_batch(batch):
    rgb, mask = batch["img"], batch["semantic_mask"]
    if tuple(rgb.shape[1:]) != (3, *HW) or tuple(mask.shape) != (len(rgb), *HW):
        raise ValueError(f"Unexpected input shapes: {tuple(rgb.shape)}, {tuple(mask.shape)}")
    return {"input": list(rgb.shape), "mask": list(mask.shape),
            "rgb_dtype": str(rgb.dtype), "mask_dtype": str(mask.dtype),
            "mask_values": sorted(mask.unique().tolist())}


def probability(model, tensor):
    logits = model(tensor)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("Expected two-channel semantic logits")
    logits = torch.nn.functional.interpolate(logits.float(), size=tensor.shape[-2:],
                                             mode="bilinear", align_corners=False)
    return logits.softmax(dim=1)[:, 1]


def git_provenance(root):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}


def train(args, config, output, smoke):
    require_cuda(args.device)
    dataset = args.prepared / "dataset.yaml"
    if not dataset.is_file():
        raise FileNotFoundError("Run prepare first")
    if args.resume:
        checkpoint = output / "weights" / "last.pt"
        _, state = load_checkpoint(checkpoint)
        saved = state["train_args"]
        if (state.get("optimizer") is None or saved["epochs"] != (1 if smoke else config["epochs"])
                or Path(saved["data"]).resolve() != dataset.resolve()):
            raise ValueError("Checkpoint must be an interrupted run for this experiment and dataset")
    else:
        if output.exists():
            raise FileExistsError(f"Run already exists: {output}. Choose another run root or use --resume.")
        checkpoint = args.weights or official_weights(args.data_root / "yolo_weights" / config["weights"])
        verify_weights(checkpoint)
    model = YOLO(checkpoint)
    if model.task != "semantic":
        raise ValueError(f"Expected a semantic checkpoint, received {model.task}")
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        write_json(output / "source_revision.json", git_provenance(Path(__file__).resolve().parents[2]))
    if not args.resume:
        asset = Path(model.ckpt_path)
        write_json(output / "initial_checkpoint.json", {"filename": asset.name, "sha256": digest(asset),
                   "source_classes": len(model.names)})

    observed_steps = []

    def record(trainer):
        status = dict(train=trainer.first_train_batch,
                      val=validate_batch(next(iter(trainer.test_loader))))
        env = environment()
        write_json(output / "environment.json", env)
        write_json(output / "run_config.json", {
            "experiment": config, "arguments": vars(trainer.args),
            "optimizer": type(trainer.optimizer).__name__,
            "param_groups": [{k: v for k, v in group.items() if k != "params"}
                             for group in trainer.optimizer.param_groups],
            "scheduler": type(trainer.scheduler).__name__,
            "loss": "SemanticSegmentationLoss (CE + Dice + auxiliary)",
            "amp_active": bool(trainer.amp), "batch_shapes": status,
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
            "train_transforms": repr(trainer.train_loader.dataset.transforms),
        })

    def track_optimizer(trainer):
        param = next(p for p in trainer.model.parameters() if p.requires_grad)
        before = param.detach().clone()

        def after_step(optimizer, step_args, step_kwargs):
            if not observed_steps and not torch.equal(param.detach(), before):
                observed_steps.append(True)

        trainer.optimizer.register_step_post_hook(after_step)

    if smoke:
        model.add_callback("on_train_start", track_optimizer)
    model.add_callback("on_train_end", record)
    torch.cuda.reset_peak_memory_stats(args.device)
    model.train(trainer=RectangularTrainer, data=str(dataset.resolve()), epochs=1 if smoke else config["epochs"],
                imgsz=config["width"], rect=True, seed=config["seed"], amp=config["amp"],
                batch=args.batch, workers=args.workers, device=args.device, pretrained=config["pretrained"],
                project=str(output.parent.resolve()), name=output.name, exist_ok=True,
                resume=str(checkpoint.resolve()) if args.resume else False)
    from shutil import copyfile
    for name in ("best.pt", "last.pt"):
        copyfile(output / "weights" / name, output / name)
    copyfile(output / "results.csv", output / "history.csv")
    copyfile(args.prepared / "dataset_summary.json", output / "dataset_summary.json")
    write_json(output / "train_resources.json", {"peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device)})
    if smoke:
        write_json(output / "smoke_checks.json", {"observed_parameter_update": bool(observed_steps)})
        if not observed_steps:
            raise RuntimeError("Smoke test did not observe an update to the sampled parameter")

"""Check Exp02 initialization, native loss, and architecture equivalence on CPU."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from exp01 import ROOT, load_config, main as cli_main

import torch
from ultralytics import YOLO, __version__
from ultralytics.nn.tasks import SemanticSegmentationModel
from ultralytics.utils.loss import SemanticSegmentationLoss

from yolo_sem.data import digest, write_json
from yolo_sem.initialization import MODEL_YAML, initialize, signature, state_hash
from yolo_sem.model import Exp02Trainer, RectangularTrainer, git_provenance, train
from yolo_sem.validation_diagnostic import fp32_loss


def trainer_model(config, weights=None):
    trainer = object.__new__(Exp02Trainer)
    trainer.resume = False
    trainer.data = {"nc": 2, "channels": 3}
    trainer.args = SimpleNamespace(**config)
    cfg = weights.yaml if weights is not None else MODEL_YAML
    return trainer.get_model(cfg, weights, verbose=False), trainer.initialization_report


def check_train_metadata(data_root, config, expected):
    """Run production setup through the start callbacks without starting any epochs."""
    class StopAfterMetadata(Exception):
        pass

    def start_only(wrapper, *, trainer, pretrained, batch, **kwargs):
        assert pretrained is config["pretrained"]
        assert bool(wrapper.ckpt_path) is pretrained
        instance = object.__new__(trainer)
        instance.args = SimpleNamespace(seed=config["seed"], pretrained=pretrained, lr0=0.01,
                                        momentum=0.937, warmup_bias_lr=0.1)
        instance.data = {"nc": 2, "channels": 3}
        instance.resume = False
        weights = wrapper.model if pretrained else None
        instance.model = instance.get_model(wrapper.model.yaml, weights, verbose=False)
        assert state_hash(instance.model) == expected["initial_state_sha256"]
        instance.optimizer = instance.build_optimizer(instance.model, name="auto", decay=0.0005, iterations=120)
        instance.scheduler = torch.optim.lr_scheduler.LambdaLR(instance.optimizer, lambda _: 1.0)
        instance.accumulate = 32
        instance.batch_size = batch
        instance.amp = True
        for callback in wrapper.callbacks["on_train_start"]:
            # Only run callbacks registered by our train() wrapper.
            if callback.__module__ == "yolo_sem.model":
                callback(instance)
        raise StopAfterMetadata

    with TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        (scratch / "dataset.yaml").touch()
        output = scratch / "smoke"
        args = SimpleNamespace(device=0, prepared=scratch, resume=False, data_root=data_root,
                               weights=None, save_period=None, batch=2, workers=0)
        with (patch("yolo_sem.model.require_cuda"), patch.object(torch.cuda, "reset_peak_memory_stats"),
              patch.object(YOLO, "train", start_only)):
            try:
                train(args, config, output, smoke=True)
            except StopAfterMetadata:
                pass
            else:
                raise AssertionError("Expected to reach the start callbacks")
        metadata = json.loads((output / "initial_checkpoint.json").read_text(encoding="utf-8"))
        assert all(metadata[key] == expected[key] for key in expected)
        asset = data_root / "yolo_weights" / config["weights"] if config["pretrained"] else None
        assert metadata["filename"] == (asset.name if asset else None)
        assert metadata["sha256"] == (digest(asset) if asset else None)
        revision = json.loads((output / "source_revision.json").read_text(encoding="utf-8"))
        assert revision == git_provenance(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--exp01-best", type=Path, help="Optional baseline checkpoint for exact architecture comparison")
    args = parser.parse_args()
    assert __version__ == "8.4.52"
    torch.set_num_threads(2)
    config_dir = ROOT / "configs/yolo"
    city_cfg, ade_cfg, random_cfg = [
        load_config(config_dir / name) for name in ("exp01_rgb.yaml", "exp02_a_ade20k.yaml", "exp02_b_random.yaml")
    ]
    assert Exp02Trainer.build_dataset is RectangularTrainer.build_dataset
    with TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        prepared = scratch / "prepared"
        prepared.mkdir()
        (prepared / "dataset.yaml").touch()
        write_json(prepared / "dataset_summary.json", {"split_sha256": digest(ROOT / "metadata/data_split.csv")})
        expected_runs = set()
        with (patch("yolo_sem.model.require_cuda"), patch("yolo_sem.model.train") as dry_train,
              patch("exp01.predict"), patch("exp01.evaluate")):
            for filename, config in zip(("exp02_a_ade20k.yaml", "exp02_b_random.yaml"), (ade_cfg, random_cfg)):
                for stage, suffix in (("train", ""), ("smoke", "_smoke")):
                    expected_runs.add(scratch / (config["experiment"] + suffix))
                    argv = ["exp01.py", stage, "--config", str(config_dir / filename), "--batch", "16",
                            "--run-root", str(scratch), "--prepared", str(prepared)]
                    with patch("sys.argv", argv):
                        cli_main()
            assert {call.args[2] for call in dry_train.call_args_list} == expected_runs
        occupied = scratch / ade_cfg["experiment"]
        occupied.mkdir()
        with patch("yolo_sem.model.require_cuda"), patch("yolo_sem.model.YOLO") as forbidden:
            try:
                train(SimpleNamespace(device=0, prepared=prepared, resume=False), ade_cfg, occupied, smoke=False)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Existing run was accepted")
            forbidden.assert_not_called()
    from yolo_sem.initialization import resolve_initial_source

    city_source, ade_source = [
        YOLO(resolve_initial_source(SimpleNamespace(weights=None, data_root=args.data_root), c)).model
        for c in (city_cfg, ade_cfg)
    ]
    assert city_source.yaml["nc"] == 19 and ade_source.yaml["nc"] == 150
    city, city_report = initialize(city_source.yaml, city_source, city_cfg["seed"])
    ade, ade_report = trainer_model(ade_cfg, ade_source)
    with patch.object(SemanticSegmentationModel, "load", side_effect=AssertionError("Loaded weights on random path")):
        random, random_report = trainer_model(random_cfg)
        random_repeat, _ = trainer_model(random_cfg)
    assert state_hash(random) == state_hash(random_repeat)
    assert len({state_hash(m) for m in (city, ade, random)}) == 3
    assert len({signature(m) for m in (city, ade, random)}) == 1
    if args.exp01_best:
        assert signature(YOLO(args.exp01_best).model) == signature(ade)
    assert city_report["transferred_items"] == ade_report["transferred_items"] == 360
    assert random_report["transferred_items"] == 0
    assert len(ade_report["not_transferred"]) == 4
    check_train_metadata(args.data_root, ade_cfg, ade_report)
    check_train_metadata(args.data_root, random_cfg, random_report)
    for name in ade_report["not_transferred"]:
        assert torch.equal(ade.state_dict()[name], random.state_dict()[name])
    optimizer_check = object.__new__(Exp02Trainer)
    optimizer_check.data = {"nc": 2}
    optimizer_check.args = SimpleNamespace(warmup_bias_lr=0.1, lr0=0.01, momentum=0.937)
    optimizer = optimizer_check.build_optimizer(random, name="auto", lr=0.01, momentum=0.937,
                                                decay=0.0005, iterations=120)
    assert type(optimizer).__name__ == "AdamW"
    assert optimizer_check.args.warmup_bias_lr == 0.0
    assert all(g["lr"] == 0.001667 and g["betas"] == (0.9, 0.999) for g in optimizer.param_groups)
    assert {g["param_group"]: g["weight_decay"] for g in optimizer.param_groups} == {
        "weight": 0.0005, "bn": 0.0, "bias": 0.0,
    }
    image = torch.zeros(1, 3, 512, 768)
    for model in (ade, random):
        with torch.inference_mode():
            logits = model.eval()(image)
            assert logits.shape == (1, 2, 64, 96) and torch.isfinite(logits).all()
        model.train()
        model.args = SimpleNamespace(data="dataset.yaml")
        criterion = SemanticSegmentationLoss(model)
        gt = torch.zeros((2, 64, 96), dtype=torch.int32)
        gt[:, 16:48, 24:72] = 1
        gt[:, 0, :] = 255
        prediction = model(torch.rand(2, 3, 64, 96))
        loss, items = criterion(prediction, {"semantic_mask": gt})
        loss.backward()
        assert torch.isfinite(items).all()
        assert any(p.grad is not None for p in model.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    big_gt = {"semantic_mask": torch.zeros(1, 512, 768, dtype=torch.int32)}
    half = torch.zeros(1, 2, 64, 96, dtype=torch.float16)
    assert torch.isfinite(fp32_loss(criterion, half, big_gt)).all()
    print("CPU checks passed: same 6,498,756 parameters and [1,2,64,96] logits; native loss/backward finite.")
    print("ADE20K transferred", ade_report["transferred_items"], "of", ade_report["target_items"])
    print("ADE20K skipped", ade_report["not_transferred"])
    print("Random transferred", random_report["transferred_items"], "items; deterministic seed 42.")


if __name__ == "__main__":
    main()

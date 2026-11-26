"""CPU checks for the native loader, logits and provisional metric arithmetic."""
import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image

from exp01 import ROOT, load_config
from yolo_sem.assets import array, mask_array, rgb_array, verify_weights
from yolo_sem.data import digest, read_rows, write_json
from yolo_sem.evaluation import evaluate, prediction_identity
from yolo_sem.metrics import area_under_pro, select_threshold, summarize, sweep


class EvaluationTests(unittest.TestCase):
    def test_source_array_channels_and_binary_union(self):
        rgb = np.zeros((1024, 1536, 19), dtype=np.uint8)
        rgb[0, 0, :3] = [10, 20, 30]
        self.assertEqual(rgb_array(rgb, "dummy")[0, 0].tolist(), [30, 20, 10])
        mask = np.zeros((5, 1024, 1536), dtype=np.uint8)
        mask[4, 0, 0] = 1
        mask[3, 0, 1] = 255
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.npz"
            np.savez_compressed(path, custom=mask)
            np.testing.assert_array_equal(mask_array(array(path, "custom"), "dummy")[0, :2], [0, 255])
            with self.assertRaisesRegex(ValueError, "requires array key"):
                array(path, "missing")

    def test_pixel_counts_and_threshold(self):
        mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        probability = np.array([[.9, .4], [.8, .2]], dtype=np.float32)
        grid = np.array([0, .4, .8, .9, 1])
        counts, pro = sweep(mask, probability, grid, 8, .1)
        i = select_threshold(grid, counts, .02)
        self.assertEqual(grid[i], .9)
        self.assertEqual(summarize(counts[i])["iou"], .5)
        self.assertEqual(summarize(counts[i])["region_recall"], 1)
        self.assertEqual(summarize(counts[i])["pixel_fpr"], 0)
        self.assertEqual(pro[i], .5)

    def test_perfect_region_curve(self):
        counts, pro = sweep(np.array([[1, 0]]), np.array([[.9, .1]]), [0, .5, 1], 8, .1)
        self.assertAlmostEqual(area_under_pro(counts, pro, .02), 1)

    def test_connectivity_and_undefined_region_recall(self):
        mask = np.eye(2, dtype=np.uint8)
        counts, _ = sweep(mask, mask.astype(float), [.5], 4, .1)
        self.assertEqual(counts[0, 5], 2)
        counts, _ = sweep(mask, mask.astype(float), [.5], 8, .1)
        self.assertEqual(counts[0, 5], 1)
        counts, _ = sweep(mask * 0, mask.astype(float), [.5], 8, .1)
        self.assertIsNone(summarize(counts[0])["region_recall"])

    def test_fixed_split_and_config(self):
        config = load_config(ROOT / "configs/yolo/exp01_rgb.yaml")
        self.assertEqual(config["epochs"], 20)
        fixed = ROOT / "metadata/data_split.csv"
        self.assertEqual(len(read_rows(fixed)), 542)
        with tempfile.TemporaryDirectory() as temporary:
            lf = Path(temporary) / "split.csv"
            lf.write_bytes(fixed.read_bytes().replace(b"\r\n", b"\n"))
            self.assertEqual(read_rows(lf), read_rows(fixed))
            lf.write_bytes(lf.read_bytes().replace(b"image000000", b"image099999"))
            with self.assertRaisesRegex(ValueError, "contents differ"):
                read_rows(lf)

    def test_evaluation_freezes_validation_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(prepared=root / "prepared", split_csv=ROOT / "metadata/data_split.csv")
            args.prepared.mkdir()
            config = dict(load_config(ROOT / "configs/yolo/exp01_rgb.yaml"), height=2, width=2)
            write_json(args.prepared / "dataset_summary.json", {"split_sha256": digest(args.split_csv)})
            (root / "best.pt").write_bytes(b"synthetic checkpoint identity")
            identity = prediction_identity(args, config, root)
            rows = []
            for split in ("val", "test"):
                probability_dir = root / "probability_maps" / split
                probability_dir.mkdir(parents=True)
                write_json(probability_dir / "manifest.json", {"identity": identity, "split": split, "images": 4})
                for folder in ("images", "masks"):
                    (args.prepared / folder / split).mkdir(parents=True)
                for group in "ABCD":
                    name = split + group
                    rows.append({"filename": name, "split": split, "group": group})
                    mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
                    Image.fromarray(mask).save(args.prepared / "masks" / split / f"{name}.png")
                    Image.new("RGB", (2, 2)).save(args.prepared / "images" / split / f"{name}.png")
                    probability = np.array([[.9, .2 if split == "val" else 1.], [.2, .2]], dtype=np.float32)
                    np.save(probability_dir / f"{name}.npy", probability)
            with patch("yolo_sem.evaluation.read_rows", return_value=rows):
                with self.assertRaisesRegex(ValueError, "Validation threshold"):
                    evaluate(args, config, root, "test")
                threshold = evaluate(args, config, root, "val", calibrate=True)["threshold"]
                self.assertEqual(threshold, .9)
                self.assertEqual(evaluate(args, config, root, "val")["pixel_fpr"], 0)
                test = evaluate(args, config, root, "test")
                self.assertEqual(test["threshold"], threshold)
                self.assertAlmostEqual(test["pixel_fpr"], 1 / 3)
                self.assertTrue((root / "group_metrics.csv").is_file())


def check_archive():
    root = ROOT / "data"
    archive = root / "archives/data_processed.zip"
    with ZipFile(archive) as source:
        checked = 0
        for member in source.namelist():
            path = Path(member)
            if path.parent.name not in ("images", "masks") or path.suffix not in (".png", ".jpg"):
                continue
            local = root / "data_processed" / path.parent.name / path.name
            if hashlib.sha256(source.read(member)).hexdigest() != digest(local):
                raise ValueError(f"Local source differs from archive: {local.name}")
            checked += 1
    if checked != 1084:
        raise ValueError(f"Expected 1084 archive entries, found {checked}")
    print(f"Archive byte equality: {checked} files")


def check_model(weights, prepared):
    verify_weights(weights)
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import SemanticSegmentationModel

    from yolo_sem.model import HW, RectangularSemanticDataset, probability, validate_batch

    torch.set_num_threads(2)
    official = YOLO(weights)
    assert official.task == "semantic" and len(official.names) == 19
    native = SemanticSegmentationModel(official.model.yaml, nc=2, ch=3, verbose=False)
    native.load(official.model, verbose=False)
    data = check_det_dataset(str((prepared / "dataset.yaml").resolve()), autodownload=False)
    for split, length in (("train", 374), ("val", 84)):
        dataset = RectangularSemanticDataset(img_path=data[split], imgsz=768, batch_size=2,
                    augment=split == "train", hyp=get_cfg(), rect=True, stride=32, pad=0.0,
                    task="semantic", data=data)
        assert len(dataset) == length
        assert np.all(dataset.batch_shapes == HW)
        batch = next(iter(build_dataloader(dataset, batch=2, workers=0, rank=-1)))
        info = validate_batch(batch)
        assert set(info["mask_values"]) <= ({0, 1, 255} if split == "train" else {0, 1})
        if split == "val":
            with Image.open(batch["im_file"][0]) as image:
                assert np.array_equal(np.asarray(image), batch["img"][0].permute(1, 2, 0).numpy())
        print(split, info)
    with torch.inference_mode():
        native.eval()
        logits = native(torch.zeros(1, 3, *HW))
        logits = logits[0] if isinstance(logits, (tuple, list)) else logits
        output = probability(native, batch["img"][:1].float() / 255)
    assert logits.shape[1] == 2 and output.shape == (1, *HW)
    assert output.isfinite().all() and (0 <= output).all() and (output <= 1).all()
    print("Official asset semantic classes:", len(official.names), "binary logits:", tuple(logits.shape),
          "binary probability:", tuple(output.shape), "sha256:", digest(weights))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/yolo_exp01")
    parser.add_argument("--check-archive", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.check_archive:
        check_archive()
    if args.weights:
        check_model(args.weights, args.prepared)

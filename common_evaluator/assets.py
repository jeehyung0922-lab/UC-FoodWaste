"""Validation-only ground truth and source polygon bundle."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

VERSION = "0.1.0"
HW = (512, 768)
SPLIT_SHA256 = "859dcb499be4519087815814d1abd1cdcb00377c9b551a056883047f6a0059ba"
REGION_MANIFEST_SHA256 = "08a9c70733fa4f7f520bcd92cbf9efa88f244d4d81ab3bb0253f770f155f6291"
PIXEL_CONTENT_SHA256 = "57b77de9bcff4b1d5025aad15bd79ae09fea595bb78a976a8fd13e8c809c9e66"
LABELS = {"soft_plastic", "hard_plastic", "glass", "metal"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalized_split(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def val_rows(path):
    if normalized_split(path) != SPLIT_SHA256:
        raise ValueError("Split differs from the approved UC-FoodWaste split")
    result = {int(r["image_id"]): r for r in rows(path) if r["split"] == "val"}
    if len(result) != 84 or len({r["filename"] for r in result.values()}) != 84:
        raise ValueError("Expected 84 unique Validation IDs")
    return result


def rasterize(points, context):
    xy = np.asarray(points, dtype=np.float64)
    if xy.ndim != 1 or len(xy) < 6 or len(xy) % 2 or not np.isfinite(xy).all():
        raise ValueError(f"{context}: Expected finite flat x,y pairs with at least 3 vertices")
    xy = xy.reshape(-1,2)
    x, y = xy.T
    if (x < 0).any() or (x > 1536).any() or (y < 0).any() or (y > 1024).any():
        raise ValueError(f"{context}: Polygon outside original 1536x1024 extent")
    if abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))) <= 1e-9:
        raise ValueError(f"{context}: zero-area polygon")
    canvas = Image.new("L", (1536,1024))
    ImageDraw.Draw(canvas).polygon([tuple(pair) for pair in xy], fill=1)
    grid = np.asarray(canvas.resize(HW[::-1], Image.Resampling.NEAREST))
    pixels = np.flatnonzero(grid).astype(np.int32)
    if not pixels.size:
        raise ValueError(f"{context}: polygon has zero rasterized area")
    return pixels


def read_val_annotations(path, val):
    annotations = {}
    for row in rows(path):
        identifier = int(row["Image ID"])
        if identifier not in val:
            continue
        name = val[identifier]["filename"]
        if (Path(row["Image Name"]).stem.casefold()!=name or
                (int(row["Height"]),int(row["Width"])) != (1024,1536) or name in annotations):
            raise ValueError(f"Inconsistent or duplicate source annotation for {name}")
        polygons = ast.literal_eval(row["Polygons"])
        if not isinstance(polygons,list):
            raise ValueError(f"{name}: expected a list of polygons")
        annotations[name] = polygons
    if set(annotations) != {r["filename"] for r in val.values()}:
        raise ValueError("Polygon CSV is missing Validation rows; no Connected Component fallback")
    return annotations


def build_bundle(split, masks, annotations, out):
    val = val_rows(split)
    polygons = read_val_annotations(annotations, val)
    arrays, images, regions, differences = {}, [], [], []
    for identifier, row in sorted(val.items()):
        name = row["filename"]
        with Image.open(masks / f"{name}.png") as image:
            mask = np.asarray(image)
            if mask.shape == (1024,1536):
                mask = np.asarray(image.resize(HW[::-1], Image.Resampling.NEAREST))
        if mask.shape != HW or not np.isin(mask, [0,1,255]).all():
            raise ValueError(f"{name}: invalid binary GT mask")
        gt = (mask != 0).ravel()
        union = np.zeros_like(gt)
        arrays[f"g_{name}"] = np.flatnonzero(gt).astype(np.int32)
        images.append(dict(id=identifier, filename=name, group=row["group"]))
        if bool(gt.any()) != (row["has_contaminant"]=="1"):
            raise ValueError(f"{name}: GT foreground disagrees with split metadata")
        for i, obj in enumerate(polygons[name],1):
            region_id = f"{name}:p{i:03d}"
            if obj["Label"] not in LABELS:
                raise ValueError(f"{region_id}: unknown polygon label {obj['Label']}")
            pixels = rasterize(obj["Points"], region_id)
            if not gt[pixels].any():
                raise ValueError(f"{region_id}: polygon and trusted GT have no overlap")
            # Each annotation independently keeps its area, including overlap with other polygons.
            union[pixels] = True
            key = f"r_{len(regions)}"
            arrays[key] = pixels
            regions.append(dict(id=region_id, filename=name, label=obj["Label"], area=len(pixels), key=key))
        if bool(gt.any()) != bool(union.any()):
            raise ValueError(f"{name}: GT/Polygon emptiness disagrees")
        common = int((gt & union).sum())
        either = int((gt | union).sum())
        differences.append(dict(filename=name, mask_only=int((gt & ~union).sum()),
                                polygon_only=int((union & ~gt).sum()), intersection=common, union=either,
                                iou=common/either if either else None))
    out.mkdir(parents=True)
    manifest = out/"region_manifest.json"
    write(manifest,dict(images=images, regions=regions, polygon_gt_audit=differences))
    archive = out/"validation_pixels.npz"
    np.savez_compressed(archive, **arrays)
    write(out/"asset_manifest.json",dict(version=VERSION, split="val", split_sha256=SPLIT_SHA256,
          rasterization="Pillow full-resolution polygon fill, then nearest resize",
          polygon_csv_sha256=sha(annotations), region_manifest_sha256=sha(manifest), pixel_archive_sha256=sha(archive)))
    return len(images), len(regions)


def load_bundle(folder):
    manifest = read(folder/"asset_manifest.json")
    pixels, regions = folder/"validation_pixels.npz", folder/"region_manifest.json"
    if (manifest.get("version"), manifest.get("split"), manifest.get("split_sha256")) != (VERSION,"val",SPLIT_SHA256):
        raise ValueError("Incompatible Validation asset pack")
    if sha(pixels)!=manifest["pixel_archive_sha256"] or sha(regions)!=manifest["region_manifest_sha256"]:
        raise ValueError("Ground truth/region bundle hash mismatch")
    if manifest["region_manifest_sha256"] != REGION_MANIFEST_SHA256:
        raise ValueError("Region manifest differs from the v0.1 Validation reference")
    data = read(regions)
    if len(data["images"]) != 84 or len({r["filename"] for r in data["images"]}) != 84:
        raise ValueError("Asset pack must contain 84 distinct Validation images")
    digest = hashlib.sha256()
    with np.load(pixels,allow_pickle=False) as archive:
        for key in sorted(archive.files):
            indices = archive[key]
            if indices.ndim != 1 or indices.dtype.kind != "i" or indices.dtype.itemsize != 4:
                raise ValueError("Validation GT pixel indices must be one-dimensional int32 arrays")
            digest.update(key.encode("ascii")+b"\0")
            digest.update(indices.astype("<i4",copy=False).tobytes())
            digest.update(b"\xff")
    if digest.hexdigest() != PIXEL_CONTENT_SHA256:
        raise ValueError("GT or polygon pixels differ from the v0.1 Validation reference")
    return manifest, data, pixels


def match_maps(folder, images):
    expected = {r["filename"] for r in images}
    aliases = {str(r["id"]):r["filename"] for r in images}
    found = {}
    for file in folder.iterdir():
        if file.suffix.lower() != ".npy":
            continue
        name = file.stem.casefold()
        if re.fullmatch(r"[0-9]{1,8}", name):
            name = aliases.get(str(int(name)),name)
        if name not in expected:
            raise ValueError(f"{file.name}: outside the fixed Validation IDs")
        if name in found:
            raise ValueError(f"Duplicate image ID: {found[name].name} and {file.name}")
        found[name] = file
    if expected - found.keys():
        raise ValueError(f"Missing probability maps: {', '.join(sorted(expected-found.keys()))}")
    return found


def load_probability(path):
    try:
        value = np.load(path,allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"{path.name}: unreadable .npy array ({exc})") from exc
    if not isinstance(value,np.ndarray):
        value.close()
        raise ValueError(f"{path.name}: expected a .npy array, not an NPZ archive")
    if value.shape != HW or value.dtype != np.float32:
        raise ValueError(f"{path.name}: expected float32 {HW}; got {value.dtype}, {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{path.name}: contains NaN or Inf")
    if value.min() < 0 or value.max() > 1:
        raise ValueError(f"{path.name}: probability outside [0,1]")
    return value.ravel()

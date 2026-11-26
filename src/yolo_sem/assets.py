"""Fetch the exact Exp01 initialization and materialize local source data."""
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image

from .data import digest, read_rows

WEIGHT_NAME = "yolo26s-sem.pt"
WEIGHT_SHA256 = "bc3e2152329831303de83e1af91d4b57d547e8794de8bb6edb1f2a622d1d5435"
WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/" + WEIGHT_NAME


def official_weights(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temp:
            downloaded = Path(temp) / WEIGHT_NAME
            with urlopen(WEIGHT_URL, timeout=90) as remote, downloaded.open("xb") as local:
                shutil.copyfileobj(remote, local, length=1024 * 1024)
            verify_weights(downloaded)
            # A hard link publishes a verified file without replacing a concurrent download.
            try:
                path.hardlink_to(downloaded)
            except FileExistsError:
                pass
    verify_weights(path)
    return path.resolve()


def verify_weights(path):
    if digest(path) != WEIGHT_SHA256:
        raise ValueError(f"Unexpected SHA-256 for {path}. Use the pinned official {WEIGHT_NAME}.")


def import_archive(archive, destination, rows):
    expected = {(kind, row["filename"] + suffix) for row in rows
                for kind, suffix in (("images", ".jpg"), ("masks", ".png"))}
    with ZipFile(archive) as source:
        entries = {}
        for item in source.infolist():
            path = PurePosixPath(item.filename)
            key = (path.parent.name, path.name)
            if key in expected:
                if key in entries:
                    raise ValueError(f"Duplicate archive input for {key}")
                entries[key] = item
        if entries.keys() != expected:
            raise ValueError(f"Missing RGB/mask pairs in archive: {sorted(expected - entries.keys())[:10]}")
        for (kind, name), item in entries.items():
            with source.open(item) as stream, (destination / kind / name).open("xb") as output:
                shutil.copyfileobj(stream, output)


def array(path, key):
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            if key not in data.files:
                raise ValueError(f"{path} requires array key {key!r}; found {data.files}")
            return data[key]
    return data


def array_paths(folder, rows):
    result = {}
    for row in rows:
        name = row["filename"]
        candidates = [folder / (name + ext) for ext in (".npy", ".npz")]
        found = [candidate for candidate in candidates if candidate.is_file()]
        if len(found) != 1:
            raise ValueError(f"Expected one .npy or .npz for {name} under {folder}")
        result[name] = found[0]
    return result


def rgb_array(image, name):
    if image.shape != (1024, 1536, 19):
        raise ValueError(f"{name}: expected original H1024×W1536×19 image")
    rgb = image[..., [2, 1, 0]]
    if rgb.dtype == np.uint8:
        return rgb
    if np.issubdtype(rgb.dtype, np.floating) and np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1:
        return np.rint(rgb * 255).astype(np.uint8)
    raise ValueError(f"{name}: BGR must be uint8 or finite float in [0,1]")


def mask_array(mask, name):
    if mask.shape == (5, 1024, 1536):
        mask = np.moveaxis(mask, 0, -1)
    if mask.shape not in ((1024, 1536), (1024, 1536, 5)) or not np.isin(mask, (0, 1, 255)).all():
        raise ValueError(f"{name}: expected binary HW or five binary class planes")
    foreground = (mask[..., :4] != 0).any(axis=-1) if mask.ndim == 3 else mask != 0
    return foreground.astype(np.uint8) * 255


def import_arrays(images, masks, destination, rows, mask_key):
    sources, targets = array_paths(images, rows), array_paths(masks, rows)
    for name in sources:
        rgb = rgb_array(array(sources[name], "image"), name)
        mask = mask_array(array(targets[name], mask_key), name)
        Image.fromarray(rgb).save(destination / "images" / f"{name}.png")
        Image.fromarray(mask).save(destination / "masks" / f"{name}.png")


def materialize_source(data_root, split_csv, archive=None, images=None, masks=None, mask_key="mask",
                       include_test=False):
    rows = [row for row in read_rows(split_csv) if include_test or row["split"] != "test"]
    destination = data_root / "data_processed"
    if destination.exists():
        raise FileExistsError(f"Choose a new data root; already exists: {destination}")
    data_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=data_root) as temp:
        stage = Path(temp) / "data_processed"
        for kind in ("images", "masks"):
            (stage / kind).mkdir(parents=True)
        if archive:
            import_archive(archive, stage, rows)
        else:
            import_arrays(images, masks, stage, rows, mask_key)
        stage.rename(destination)
    return len(rows)

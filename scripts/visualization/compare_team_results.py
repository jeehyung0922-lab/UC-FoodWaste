"""Compare Common Evaluator Validation exports without loading model predictions."""
import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from html import escape
from itertools import combinations
import json
import math
from pathlib import Path
import stat
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common_evaluator"))
from assets import load_bundle, read, write
from engine import csv_out, verify_metadata
from metrics import auspro, pixel_metrics, region_metric, select
from report import document


METRICS = ("region_recall", "pixel_recall", "pixel_precision", "iou", "pixel_fpr", "spro", "auspro")
IDENTITY = ("version", "split", "split_sha256", "rasterization", "region_manifest_sha256", "pixel_archive_sha256")
COUNTS = ("tp", "fp", "fn", "tn")
SIZES = ((0, 1000), (1000, 5000), (5000, 10000), (10000, 25000), (25000, math.inf))
COLORS = ("#007B82", "#B54723", "#6554AC")


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def rows(folder, name):
    with (folder / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def unique(records, key):
    result = {row[key]: row for row in records}
    ensure(len(result) == len(records), f"Duplicate {key}")
    return result


def equal_number(a, b):
    if a in (None, "") or b in (None, ""):
        return a in (None, "") and b in (None, "")
    return math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-12)


def unpack(path, out):
    with ZipFile(path) as archive:
        members = archive.infolist()
        files = [info for info in members if not info.is_dir()]
        ensure(len(files) <= 100 and sum(info.file_size for info in files) <= 20_000_000, "Unexpected export size")
        names = set()
        for info in members:
            name = Path(info.filename)
            ensure(not name.is_absolute() and ".." not in name.parts and "\\" not in info.filename
                   and ":" not in info.filename and not stat.S_ISLNK(info.external_attr >> 16),
                   f"Unsafe ZIP entry: {info.filename}")
            normalized = name.as_posix().casefold()
            ensure(normalized not in names, f"Duplicate ZIP entry: {info.filename}")
            names.add(normalized)
        summary = [Path(info.filename).parent for info in files
                   if Path(info.filename).name == "evaluation_summary.json"]
        ensure(len(summary) == 1, "Expected exactly one result set per ZIP")
        archive.extractall(out)
    return out / summary[0]


def validate(folder, expected, asset, reference, protocol):
    summary = read(folder / "evaluation_summary.json")
    manifest = read(folder / "evaluation_manifest.json")
    verify_metadata(manifest["input_metadata"])
    ensure(summary["provisional"] is True and summary["protocol"] == protocol["id"], "Protocol label mismatch")
    ensure(all(manifest["asset"][key] == asset[key] for key in IDENTITY),
           "Effective GT, regions, split or rasterization mismatch")
    ensure(manifest["protocol_sha256"] == reference["protocol_sha256"]
           and manifest["evaluator"]["version"] == reference["evaluator"]["version"]
           and manifest["evaluator"]["source_sha256"] == reference["evaluator"]["source_sha256"],
           "Protocol or evaluator source mismatch")
    images = unique(rows(folder, "per_image_metrics.csv"), "filename")
    region = unique(rows(folder, "per_region_metrics.csv"), "id")
    image_reference = unique(expected["images"], "filename")
    ensure(images.keys() == image_reference.keys() == manifest["probability_maps"].keys(), "Validation image IDs differ")
    ensure(summary["images"] == len(images), "Image count mismatch")
    ensure(region.keys() == unique(expected["regions"], "id").keys(), "Validation Region IDs differ")
    for name, row in images.items():
        ensure(row["id"] == str(image_reference[name]["id"]) and row["group"] == image_reference[name]["group"],
               f"Image identity mismatch: {name}")
        ensure(sum(int(row[key]) for key in COUNTS) == 512 * 768, f"Invalid pixel counts: {name}")
    gt_audit = unique(rows(folder, "polygon_gt_audit.csv"), "filename")
    ensure(gt_audit.keys() == images.keys(), "GT audit image IDs differ")
    for expected_audit in expected["polygon_gt_audit"]:
        name = expected_audit["filename"]
        ensure(all(equal_number(gt_audit[name][k], expected_audit[k]) for k in
                   ("intersection", "union", "mask_only", "polygon_only", "iou")), f"GT audit mismatch: {name}")

    for truth in expected["regions"]:
        row = region[truth["id"]]
        ensure(all(str(truth[k]) == row[k] for k in ("filename", "label", "area", "key")), f"Region differs: {row['id']}")
        ensure(row["group"] == images[row["filename"]]["group"], f"Region group differs: {row['id']}")
        a, p = int(row["area"]), int(row["overlap_px"])
        ensure(0 <= p <= a and equal_number(row["overlap"], p/a)
               and row["detected"] == str(p*10 >= a)
               and 0 <= float(row["mean_probability"]) <= float(row["max_probability"]) + 1e-7
               and 0 <= float(row["max_probability"]) <= 1, f"Invalid region metrics: {row['id']}")

    sweep = rows(folder, "threshold_curve.csv")
    thresholds = np.array([float(r["threshold"]) for r in sweep])
    grid = np.r_[np.arange(1001) / 1000, np.nextafter(np.float32(1), np.float32(2))]
    ensure(np.array_equal(thresholds, grid), "Threshold grid mismatch")
    counts = np.array([[int(r[k]) for k in COUNTS] for r in sweep])
    fpr = np.array([float(r["pixel_fpr"]) for r in sweep])
    spro = np.array([float(r["spro"]) for r in sweep])
    detected = np.rint([float(r["region_recall"]) * len(region) for r in sweep]).astype(int)
    selected = select(thresholds, counts, detected, spro, protocol["fpr_cap"])
    ensure([i for i, r in enumerate(sweep) if r["selected"] == "True"] == [selected], "Selected sweep row differs")
    threshold = read(folder / "selected_threshold.json")
    ensure(threshold["threshold"] == summary["selected_threshold"] == thresholds[selected]
           and threshold["selection_order"] == protocol["threshold_order"], "Reported threshold differs")

    def check_aggregates(target, ims, regs):
        pix = pixel_metrics(*(sum(int(row[k]) for row in ims) for k in COUNTS))
        n, recall, sr = region_metric(np.array([int(r["overlap_px"]) for r in regs]),
                                      np.array([int(r["area"]) for r in regs]))
        checks = dict(**pix, regions=len(regs), detected_regions=n, missed_regions=len(regs)-n,
                      region_recall=recall, spro=sr)
        ensure(all(equal_number(target[key], value) for key, value in checks.items()), "Aggregate values differ")

    check_aggregates(summary, list(images.values()), list(region.values()))
    ensure(all(int(summary[k]) == counts[selected, i] for i, k in enumerate(COUNTS))
           and detected[selected] == summary["detected_regions"]
           and equal_number(fpr[selected], summary["pixel_fpr"])
           and equal_number(spro[selected], summary["spro"])
           and equal_number(auspro(fpr, spro, protocol["fpr_cap"]), summary["auspro"]), "Sweep summary differs")
    groups = unique(rows(folder, "group_metrics.csv"), "group")
    ensure(groups.keys() == set("ABCD"), "Missing group rows")
    for group, row in groups.items():
        ims = [r for r in images.values() if r["group"] == group]
        check_aggregates(row, ims, [r for r in region.values() if r["group"] == group])
        fp_images = sum(int(img["fp"]) > 0 for img in ims)
        ensure(int(row["images"]) == len(ims) and int(row["fp_images"]) == fp_images
               and equal_number(row["fp_image_fraction"], fp_images/len(ims)), f"Group count differs: {group}")
    ensure(equal_number(summary["group_b_false_alarm_image_fraction"], groups["B"]["fp_image_fraction"]),
           "Group B false alarm fraction differs")
    size_rows = rows(folder, "size_group_metrics.csv")
    size_keys = {(r["group"], int(r["min_area_px"])) for r in size_rows}
    ensure(len(size_rows) == len(size_keys) and size_keys == {(g, lo) for g in ("All", *"ABCD") for lo, _ in SIZES},
           "Size strata differ")
    for row in size_rows:
        lo = int(row["min_area_px"])
        hi = dict(SIZES)[lo]
        ensure(row["max_area_exclusive_px"] == (str(hi) if math.isfinite(hi) else ""), "Size boundary differs")
        subset = [r for r in region.values() if (row["group"] == "All" or r["group"] == row["group"])
                  and lo <= int(r["area"]) < hi]
        hits = sum(r["detected"] == "True" for r in subset)
        ensure(int(row["regions"]) == len(subset) and int(row["detected_regions"]) == hits
               and int(row["missed_regions"]) == len(subset) - hits
               and equal_number(row["region_recall"], hits/len(subset) if subset else None), "Size metrics differ")
    return summary, manifest, region, groups


def size_label(area):
    for lo, hi in SIZES:
        if lo <= area < hi:
            return f"{lo:,} to <{hi:,}" if math.isfinite(hi) else f"{lo:,}+"


def graph(path, title, labels, series, scale=1):
    height = 104 + len(labels)*(24 + 21*len(series))
    elements = []
    for j, (name, _) in enumerate(series.items()):
        elements.append(f'<text x="{38+j*250}" y="63" fill="{COLORS[j%len(COLORS)]}" font-size="14">{escape(name)}</text>')
    for i, (key, label) in enumerate(labels.items()):
        y = 95+i*(24+21*len(series))
        elements.append(f'<text x="38" y="{y}" font-size="14">{escape(label)}</text>')
        for j, values in enumerate(series.values()):
            v = values[key]
            top = y-12+j*21
            if v is not None:
                elements.append(f'<rect x="283" y="{top}" width="{475*v/scale:.2f}" height="13" fill="{COLORS[j%len(COLORS)]}"/>')
            elements.append(f'<text x="768" y="{top+11}" font-size="13">{v:.2%}</text>' if v is not None
                            else f'<text x="768" y="{top+11}" font-size="13">N/A</text>')
    document(path, title, elements, height)


def case_image(path, row, asset_pixels, rgb, model_names):
    name = row["filename"]
    with Image.open(rgb / f"{name}.png") as source:
        image = np.asarray(source.convert("RGB")).copy()
    ensure(image.shape == (512, 768, 3), f"Unexpected RGB shape: {name}")
    gt = np.zeros(512*768, dtype=bool)
    gt[asset_pixels[f"g_{name}"]] = True
    polygon = np.zeros_like(gt)
    indices = asset_pixels[row["key"]]
    polygon[indices] = True
    ys, xs = np.divmod(indices, 768)
    box = (max(0, int(xs.min())-40), max(0, int(ys.min())-40),
           min(768, int(xs.max())+41), min(512, int(ys.max())+41))
    overlay = image.copy().reshape(-1, 3)
    overlay[gt] = (.55*overlay[gt] + .45*np.array([0, 160, 220])).astype("uint8")
    overlay[polygon] = (.38*overlay[polygon] + .62*np.array([255, 190, 0])).astype("uint8")
    panels = [Image.fromarray(x).crop(box) for x in (image, overlay.reshape(image.shape))]
    canvas = Image.new("RGB", (820, 405), "white")
    draw = ImageDraw.Draw(canvas)
    for j, panel in enumerate(panels):
        panel.thumbnail((394, 290))
        canvas.paste(panel, (12+j*404, 50))
    draw.text((12, 12), f'{row["id"]} | {row["label"]} | {row["area"]:,} px | Group {row["group"]}', fill="black")
    draw.text((12, 30), "RGB", fill="black")
    draw.text((416, 30), "GT in blue | evaluated Polygon in amber", fill="black")
    for j, model in enumerate(model_names):
        status = "detected" if row[f"{model}_detected"] else "missed"
        draw.text((12,355+j*17), f'{model}: {status} | overlap {row[f"{model}_overlap"]:.2%}', fill="black")
    canvas.save(path)


def run(args):
    paths = {}
    for item in args.model:
        label, path = item.split("=", 1)
        ensure(label not in paths and label.strip(), "Duplicate or empty model label")
        paths[label] = Path(path)
    ensure(2 <= len(paths) <= 3, "Supply two or three model exports")
    asset, expected, pixels = load_bundle(args.assets)
    reference = read(args.reference / "evaluation_manifest.json")
    protocol_path = Path(__file__).resolve().parents[2] / "common_evaluator/protocol.json"
    protocol = read(protocol_path)
    ensure(hashlib.sha256(protocol_path.read_bytes()).hexdigest() == reference["protocol_sha256"], "Local protocol differs")
    args.out.mkdir(parents=True)
    results = {}
    for j, (label, path) in enumerate(paths.items()):
        folder = unpack(path, args.out / "extracted" / str(j+1)) if path.suffix.lower() == ".zip" else path
        results[label] = validate(folder, expected, asset, reference, protocol)
    summary_rows = [dict(model=label, **{key:s[key] for key in (*METRICS, "selected_threshold", "detected_regions",
                    "missed_regions", "images", "group_b_false_alarm_image_fraction")})
                    for label, (s, _, _, _) in results.items()]
    groups = [dict(model=label, **g) for label, (_, _, _, data) in results.items() for g in data.values()]
    pairs, strata, outcomes, missed = [], [], [], []
    for truth in expected["regions"]:
        rid = truth["id"]
        row = {k: truth[k] for k in ("id", "filename", "label", "area", "key")}
        row["size"] = size_label(row["area"])
        for model, (_, _, regions, _) in results.items():
            r = regions[rid]
            row["group"] = r["group"]
            row[f"{model}_detected"] = r["detected"] == "True"
            for field in ("overlap", "mean_probability", "max_probability"):
                row[f"{model}_{field}"] = float(r[field])
        for first, second in combinations(results, 2):
            a,b = row[f"{first}_detected"],row[f"{second}_detected"]
            row[f"{first} vs {second}"] = {(True,True):"both_detected", (False,False):"both_missed",
                                        (True,False):"first_only", (False,True):"second_only"}[a,b]
        pairs.append(row)
    for dim in ("size", "label", "group"):
        bins = defaultdict(list)
        for row in pairs:
            bins[row[dim]].append(row)
        for value, records in bins.items():
            for model in results:
                hits = sum(r[f"{model}_detected"] for r in records)
                strata.append(dict(dimension=dim, category=value, model=model, regions=len(records),
                                   detected=hits, missed=len(records)-hits, region_recall=hits/len(records)))
    for first,second in combinations(results, 2):
        pair_key = f"{first} vs {second}"
        subsets = {"All":pairs}
        for dim in ("size", "label", "group"):
            for row in pairs:
                subsets.setdefault(f"{dim}:{row[dim]}", []).append(row)
        for key, subset in subsets.items():
            tally = Counter(r[pair_key] for r in subset)
            outcomes.append(dict(pair=pair_key, stratum=key, total=len(subset),
                            **{k:tally[k] for k in ("both_detected","both_missed","first_only","second_only")}))
    for model in results:
        for r in pairs:
            if not r[f"{model}_detected"]:
                missed.append(dict(model=model, id=r["id"], label=r["label"], group=r["group"],
                                   area=r["area"], size=r["size"], overlap=r[f"{model}_overlap"],
                                   max_probability=r[f"{model}_max_probability"]))
    for name, records in (("model_comparison", summary_rows), ("group_metrics", groups), ("region_recall_strata", strata),
                          ("paired_regions", pairs), ("paired_outcomes", outcomes), ("missed_regions", missed)):
        if records:
            csv_out(args.out / f"{name}.csv", records)
    provenance = {}
    for label, (_, manifest, _, _) in results.items():
        path = paths[label]
        provenance[label] = dict(input=path.name, zip_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                                 if path.is_file() else None, manifest=manifest,
                                 wrapper_hash_matches_reference=manifest["asset_manifest_sha256"] == reference["asset_manifest_sha256"],
                                 source_annotation_hash_matches_reference=manifest["asset"]["polygon_csv_sha256"]
                                 == reference["asset"]["polygon_csv_sha256"])
    write(args.out / "comparison.json", dict(provisional=True, comparable=True, models=summary_rows, group_metrics=groups,
                                            strata=strata, paired_outcomes=outcomes, provenance=provenance))
    names = {key:key.replace("_"," ").upper() for key in METRICS if key != "pixel_fpr"}
    graph(args.out/"model_comparison.svg", "Common Evaluator v0.1 | Provisional Validation",
          names, {r["model"]:r for r in summary_rows})
    graph(args.out/"pixel_fpr.svg", "Pixel FPR | common cap 2%", {"pixel_fpr":"Selected Pixel FPR"},
          {r["model"]:r for r in summary_rows}, scale=protocol["fpr_cap"])
    for dim, filename in (("size","size_recall"),("label","material_recall")):
        subset=[r for r in strata if r["dimension"]==dim]
        order=[size_label(x) for x,_ in SIZES] if dim=="size" else sorted({r["category"] for r in subset})
        counts={r["category"]:r["regions"] for r in subset}
        labels={k:f"{k} | n={counts[k]}" for k in order if k in counts}
        graph(args.out/f"{filename}.svg", "Polygon Region Recall | Provisional Validation", labels,
              {model:{r["category"]:r["region_recall"] for r in subset if r["model"]==model} for model in results})
    if args.rgb:
        first, second = list(results)[:2]
        key = f"{first} vs {second}"
        with np.load(pixels,allow_pickle=False) as archive:
            for status in ("both_detected", "both_missed", "first_only", "second_only"):
                candidates=[r for r in pairs if r[key]==status]
                if candidates:
                    row=sorted(candidates,key=lambda r:r["area"])[len(candidates)//2]
                    case_image(args.out/f"example_{status}.png", row, archive, args.rgb, (first,second))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=ZIP_OR_DIR")
    for flag in ("assets", "reference", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, help="Optional directory of aligned Validation RGB PNGs")
    run(parser.parse_args())

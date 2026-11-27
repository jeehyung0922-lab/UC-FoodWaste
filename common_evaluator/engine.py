"""Validation-only evaluator orchestration, shared by CLI and Tkinter."""
import csv
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np

from assets import HW, VERSION, load_bundle, load_probability, match_maps, read, sha, write
from metrics import auspro, pixel_metrics, ratio, region_metric, select, sweep, sweep_regions
from report import bars, curve

BASE = Path(__file__).resolve().parent
# The distributed protocol cannot be altered through GUI or CLI.
PROTOCOL = {
    "id":"ucfw-common-v0.1-provisional", "split":"val", "shape":[512,768],
    "dtype":"float32", "region_definition":"source_polygon", "region_overlap":0.1,
    "fpr_cap":0.02, "threshold_steps":1000, "threshold_operator":">=",
    "threshold_order":["region_recall","spro","highest_threshold"],
    "spro":"mean(min(overlap/0.1,1))",
    "auspro":"max sPRO at duplicate FPR; linear boundary; normalized trapezoid",
    "size_edges_px":[0,1000,5000,10000,25000],
}


def csv_out(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


METADATA_DECLARATION = dict(positive_class="Contaminant", grid="ucfw-val-512x768-aligned",
                            quantity="probability", thresholded=False)


def verify_metadata(data_or_path):
    data = read(data_or_path) if isinstance(data_or_path, Path) else data_or_path
    if not isinstance(data, dict):
        raise ValueError("Input metadata must be a JSON object")
    for key, value in METADATA_DECLARATION.items():
        if data.get(key)!=value or type(data[key]) is not type(value):
            raise ValueError(f"Input metadata requires {key}={value!r}")
    if not isinstance(data.get("probability_method"), str) or not data["probability_method"].strip():
        raise ValueError("Describe sigmoid/softmax and geometric alignment in probability_method")
    for key in ("inference_ms","parameters"):
        if key in data and (type(data[key]) not in (int,float) or not np.isfinite(data[key]) or data[key]<0
                            or (key=="parameters" and type(data[key])!=int)):
            raise ValueError(f"Invalid optional {key}")
    return data


def versions():
    build = BASE/"build_info.json"
    if build.exists():
        return read(build)
    names = ("assets.py","metrics.py","report.py","engine.py","main.py","gui.py","protocol.json")
    return {"version":VERSION, "python":platform.python_version(), "numpy":version("numpy"),
            "pillow":version("Pillow"), "source_sha256":{name:sha(BASE/name) for name in names}}


def evaluate(model, maps, assets, out, metadata_path=None, progress=lambda done,total:None,
             *, declared_metadata=None):
    if not model.strip():
        raise ValueError("Enter a model name")
    if out.exists():
        raise FileExistsError(f"Evaluation output already exists: {out}")
    protocol_path=BASE/"protocol.json"
    if read(protocol_path)!=PROTOCOL:
        raise ValueError("Protocol file differs from the locked v0.1 evaluator")
    metadata_path=metadata_path or maps/"input_metadata.json"
    try:
        metadata=verify_metadata(metadata_path if declared_metadata is None else declared_metadata)
    except FileNotFoundError as exc:
        raise ValueError("Probability folder requires input_metadata.json; see the guide") from exc
    asset_info,data,pixels=load_bundle(assets)
    images, regions = data["images"], data["regions"]
    found=match_maps(maps,images)
    grid=np.r_[np.linspace(0,1,1001,dtype=np.float32),np.nextafter(np.float32(1),np.float32(2))]
    inputs={}
    per_image=[]
    hit_matrix=[]
    per_region=[]
    by_image={r["filename"]:[] for r in images}
    for region in regions:
        by_image[region["filename"]].append(region)
    with np.load(pixels,allow_pickle=False) as archive:
        for i,image in enumerate(images,1):
            name=image["filename"]
            probability=load_probability(found[name])
            inputs[name]=dict(file=found[name].name,sha256=sha(found[name]))
            region_rows=by_image[name]
            areas=[archive[r["key"]] for r in region_rows]
            counts,hits=sweep(probability,archive["g_"+name],areas,grid)
            per_image.append((image,counts))
            hit_matrix.extend(hits)
            per_region.extend((r,float(probability[area].mean()),float(probability[area].max()))
                              for r,area in zip(region_rows,areas))
            progress(i,len(images)+1)
    matrix=np.asarray(hit_matrix,dtype=np.int64).reshape(len(per_region),len(grid))
    areas=np.array([r["area"] for r,_,_ in per_region],dtype=np.int64)
    totals=np.sum([counts for _,counts in per_image],axis=0)
    detected,spro=sweep_regions(matrix,areas,grid)
    i=select(grid,totals,detected,spro,PROTOCOL["fpr_cap"])
    threshold=float(grid[i]) if i==len(grid)-1 else i/1000
    out.mkdir(parents=True)
    names_to_groups={r["filename"]:r["group"] for r in images}

    def stats(counts, hits, area):
        n, recall, sr=region_metric(hits[:,i],area)
        negatives=counts[:,1]+counts[:,3]
        fpr=np.divide(counts[:,1],negatives,out=np.full(len(grid),np.nan),where=negatives>0)
        _, sp=sweep_regions(hits,area,grid)
        return {**pixel_metrics(*counts[i]), "regions":len(area),"detected_regions":n,
                "missed_regions":len(area)-n, "region_recall":recall,
                "spro":sr,"auspro":auspro(fpr,sp,PROTOCOL["fpr_cap"])}

    group_rows=[]
    for group in "ABCD":
        ims=[c for img,c in per_image if img["group"]==group]
        mask=np.array([names_to_groups[r["filename"]]==group for r,_,_ in per_region],dtype=bool)
        group_rows.append(dict(group=group,images=len(ims),**stats(np.sum(ims,axis=0),matrix[mask],areas[mask]),
                               fp_images=sum(c[i,1]>0 for c in ims), fp_image_fraction=float(np.mean([c[i,1]>0 for c in ims]))))
    overall=stats(totals,matrix,areas)
    summary=dict(model=model,protocol=PROTOCOL["id"],provisional=True,images=len(images),selected_threshold=threshold,
                 **overall,group_b_false_alarm_image_fraction=group_rows[1]["fp_image_fraction"],
                 external_model_metadata={k:metadata[k] for k in ("inference_ms","parameters") if k in metadata})
    threshold_rows=[]
    for j,t in enumerate(grid):
        threshold_rows.append(dict(threshold=float(t) if j==len(grid)-1 else j/1000,
                                   **pixel_metrics(*totals[j]),region_recall=float(detected[j]/len(areas)),
                                   spro=float(spro[j]),selected=j==i))
    region_rows=[]
    for row, (region,mean,maximum), area in zip(matrix,per_region,areas):
        region_rows.append(dict(**region,group=names_to_groups[region["filename"]],overlap_px=int(row[i]),
                                overlap=float(row[i]/area),detected=bool(row[i]*10>=area),
                                max_probability=maximum,mean_probability=mean))
    image_rows=[]
    for img,counts in per_image:
        rs=[r for r in region_rows if r["filename"]==img["filename"]]
        found_n=sum(r["detected"] for r in rs)
        image_rows.append(dict(**img,**pixel_metrics(*counts[i]),regions=len(rs),
                               detected_regions=found_n,missed_regions=len(rs)-found_n,
                               region_recall=ratio(found_n,len(rs)),
                               spro=float(np.mean([min(r["overlap"]*10,1) for r in rs])) if rs else None))
    size_rows=[]
    edges=PROTOCOL["size_edges_px"]
    for group in ("All","A","B","C","D"):
        for lo,hi in zip(edges,[*edges[1:],None]):
            subset=[r for r in region_rows if (group=="All" or r["group"]==group) and
                    r["area"]>=lo and (hi is None or r["area"]<hi)]
            hits=sum(r["detected"] for r in subset)
            size_rows.append(dict(group=group,min_area_px=lo,max_area_exclusive_px=hi,regions=len(subset),
                                  detected_regions=hits,missed_regions=len(subset)-hits,region_recall=ratio(hits,len(subset))))
    for filename,records in (("threshold_curve.csv",threshold_rows),("per_image_metrics.csv",image_rows),
                             ("per_region_metrics.csv",region_rows),("size_group_metrics.csv",size_rows),
                             ("group_metrics.csv",group_rows),("polygon_gt_audit.csv",data["polygon_gt_audit"])):
        csv_out(out/filename,records)
    write(out/"selected_threshold.json",dict(threshold=threshold,no_positive_candidate=i==len(grid)-1,
          detected_regions=overall["detected_regions"],spro=overall["spro"],fpr=overall["pixel_fpr"],
          selection_order=PROTOCOL["threshold_order"]))
    write(out/"evaluation_summary.json",summary)
    if declared_metadata is not None:
        metadata_path=out/"input_metadata.json"
        write(metadata_path,metadata)
    write(out/"evaluation_manifest.json",dict(asset=asset_info, asset_manifest_sha256=sha(assets/"asset_manifest.json"),
          protocol_sha256=sha(protocol_path), probability_maps=inputs,
          input_metadata_sha256=sha(metadata_path), input_metadata=metadata, evaluator=versions()))
    title=f"{model} | provisional Validation"
    for key in ("region_recall","spro"):
        curve(out/f"fpr_{key}.svg",title,key,[(r["pixel_fpr"],r[key]) for r in threshold_rows],
              (overall["pixel_fpr"],overall[key]))
    bars(out/"metric_comparison.svg",title,[("Pixel recall",overall["pixel_recall"]),
         ("IoU",overall["iou"]),("Region recall",overall["region_recall"]),("sPRO",overall["spro"])])
    bars(out/"region_size_recall.svg",title,[(f"{r['min_area_px']} to {r['max_area_exclusive_px'] or 'inf'} px",r["region_recall"])
         for r in size_rows if r["group"]=="All"])
    progress(len(images)+1,len(images)+1)
    return summary


def compare(results, out):
    manifests=[read(path/"evaluation_manifest.json") for path in results]
    first=manifests[0]
    keys=("asset_manifest_sha256","protocol_sha256","evaluator")
    if any(m[k]!=first[k] for m in manifests[1:] for k in keys):
        raise ValueError("Cannot compare different Validation assets, protocols, or evaluator builds")
    summaries=[read(path/"evaluation_summary.json") for path in results]
    if len({r["model"] for r in summaries})!=len(summaries):
        raise ValueError("Use a distinct name for each model")
    out.mkdir(parents=True)
    columns=("model","selected_threshold","region_recall","pixel_recall","pixel_precision","iou","dice",
             "pixel_accuracy","pixel_fpr","spro","auspro")
    csv_out(out/"comparison.csv",[{k:r[k] for k in columns} for r in summaries])
    bars(out/"model_comparison.svg","Provisional Validation metrics",
         [(r["model"]+" | "+key,r[key]) for r in summaries for key in ("region_recall","iou","auspro")])

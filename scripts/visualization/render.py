"""Render and compare recorded Validation runs on CPU."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"runs/matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT/"runs/yolo_settings"))

from plots import COLORS, diagnostics, learning, read, save, threshold, values, write
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path,data):
    with path.open("x",encoding="utf-8") as file:
        json.dump(data,file,ensure_ascii=False,indent=2,allow_nan=False)
        file.write("\n")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def profile(path,h,w,trace):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    import ultralytics
    import thop

    torch.set_num_threads(min(4,torch.get_num_threads()))
    model=YOLO(path).model.cpu().float().eval()
    image=torch.zeros(1,3,h,w)
    rows=[]

    def shape(x):
        return list(x.shape) if isinstance(x,torch.Tensor) else [shape(v) for v in x]

    def callback(layer, inputs, output):
        rows.append(dict(layer=layer.i,from_layers=json.dumps(layer.f),module=type(layer).__name__,
                         input_shape=json.dumps(shape(inputs[0])),output_shape=json.dumps(shape(output)),
                         parameters=sum(p.numel() for p in layer.parameters())))

    hooks=[layer.register_forward_hook(callback) for layer in model.model]
    with torch.inference_mode():
        model(image)
    for hook in hooks:
        hook.remove()
    write(trace,rows)
    with torch.inference_mode():
        macs,_=thop.profile(copy.deepcopy(model),inputs=(image,),verbose=False)
    return dict(parameters=sum(p.numel() for p in model.parameters()),gflops=2*macs/1e9,
                macs=macs,tool=f"THOP {thop.__version__}; 2 FLOPs/MAC; unfused eval forward at exact input shape",
                ultralytics=ultralytics.__version__,input_shape=list(image.shape),checkpoint_sha256=sha(path),
                ultralytics_scaled_gflops=float(get_flops(model,[h,w])))


def run_figures(args):
    run=args.run.resolve()
    out=(args.out or run/"figures").resolve()
    if out==run:
        raise ValueError("Use a new output directory")
    metric=load(run/"val_metrics.json")
    frozen=load(run/"threshold.json")
    recorded_inference=load(run/"probability_maps/val/manifest.json")
    identity=frozen["identity"]
    cfg=identity["evaluation_config"]
    if (metric["split"]!="val" or frozen["selected_on"]!="val" or metric["threshold"]!=frozen["threshold"]
        or metric["inference"]!=recorded_inference or recorded_inference["identity"]!=identity
        or sha(args.split_csv)!=identity["split_sha256"]):
        raise ValueError("Validation artifact identity mismatch")
    chosen=metric["threshold"]
    scan=read(run/"threshold_search.csv")
    point=next(row for row in scan if float(row["threshold"])==chosen)
    keys=("region_recall","pixel_recall","iou","pixel_fpr")
    if any(not np.isclose(float(point[k]),metric[k]) for k in keys):
        raise ValueError("Selected threshold row differs from Validation metrics")
    if frozen.get("provisional"):
        from yolo_sem.metrics import select_threshold
        count=np.array([[int(r[k]) for k in ("tp","fp","fn","tn","regions_detected","regions")] for r in scan])
        grid=values(scan,"threshold")
        if grid[select_threshold(grid,count,cfg["fpr_limit"])]!=chosen:
            raise ValueError("Saved threshold disagrees with provisional selection rule")
    history=read(run/"history.csv")
    best=int(history[np.nanargmax(values(history,"metrics/mIoU"))]["epoch"])
    out.mkdir(parents=True)
    title=cfg["experiment"]
    learning(run,out,title,best)
    threshold(scan,out,title,chosen,cfg["fpr_limit"],frozen.get("provisional",False))
    val=[row for row in read(args.split_csv) if row["split"]=="val"]
    small,n,gt_hash=diagnostics(run,out,args.data_root/"data_processed",val,cfg,chosen,title)
    if len(val)!=metric["images"]:
        raise ValueError("Validation sample count does not match")
    env=load(run/"environment.json")
    train=load(run/"train_resources.json")
    saved_profile=None
    if args.profile_checkpoint:
        checkpoint=run/"best.pt"
        if sha(checkpoint)!=identity["checkpoint_sha256"]:
            raise ValueError("Best checkpoint hash mismatch")
        saved_profile=profile(checkpoint,cfg["height"],cfg["width"],out/"feature_maps.csv")
        if saved_profile["parameters"]!=recorded_inference["parameters"]:
            raise ValueError("Parameters disagree with recorded inference")
        dump(out/"model_profile.json",saved_profile)
    devices=[d["name"] for d in env.get("devices",[])]
    settings=load(run/"run_config.json")["arguments"]
    device_index=str(settings.get("device",""))
    device=devices[int(device_index)] if device_index.isdecimal() and int(device_index)<len(devices) else None
    efficiency=dict(parameters=recorded_inference["parameters"],
                    gflops=saved_profile["gflops"] if saved_profile else None,
                    training_seconds=float(history[-1]["time"]),
                    train_peak_allocated_gib=train["peak_gpu_allocated_bytes"]/2**30,
                    inference_peak_allocated_gib=recorded_inference["peak_gpu_allocated_bytes"]/2**30,
                    inference_mean_ms=recorded_inference["mean_inference_ms"],
                    inference_median_ms=recorded_inference["median_inference_ms"])
    units={"parameters":"count","gflops":"GFLOPs","training_seconds":"s","train_peak_allocated_gib":"GiB",
           "inference_peak_allocated_gib":"GiB","inference_mean_ms":"ms","inference_median_ms":"ms"}
    write(out/"model_efficiency.csv",[dict(metric=k,value=v,unit=units[k]) for k,v in efficiency.items()])
    fig,ax=plt.subplots(figsize=(10,4),layout="constrained")
    ax.axis("off")
    table=[(k.replace("_"," ").capitalize(),f"{v:,.4g} {units[k]}" if v is not None else "N/A")
           for k,v in efficiency.items()]
    ax.table(cellText=table,colLabels=["Metric","Recorded value"],loc="center",cellLoc="left").scale(1,1.6)
    ax.set_title(f"{title} · efficiency · {device or 'GPU unspecified'}",loc="left")
    save(fig,out,"model_efficiency")
    protocol=dict(split="val",split_sha256=identity["split_sha256"],provisional=frozen.get("provisional"),
                  validation_gt_sha256=gt_hash,
                  recorded_region_protocol=metric.get("region_protocol"),threshold_rule=frozen["tie_break"],
                  height=cfg["height"],width=cfg["width"],connectivity=cfg["connectivity"],
                  min_overlap=cfg["region_overlap"],fpr_limit=cfg["fpr_limit"],
                  threshold_steps=cfg["threshold_steps"],small_region_max_exclusive_px=5000,
                  metric_protocol_id=args.protocol_id)
    metadata=load(args.ablation_metadata) if args.ablation_metadata else {}
    dump(out/"comparison_input.json",dict(experiment=title,protocol=protocol,best_epoch=best,threshold=chosen,
         small_region_recall=small,small_region_count=n,**{k:metric[k] for k in keys},**efficiency,
         latency_protocol=dict(device=device,torch=env.get("torch"),cuda=env.get("cuda_build"),
                               timing_scope=recorded_inference["timing_scope"],
                               shape=recorded_inference["tensor_shape"]),
         flops_method=saved_profile["tool"] if saved_profile else None,ablation=metadata))
    print(out)


def compare(args):
    rows=[load(path) for path in args.inputs]
    if len({r["experiment"] for r in rows})!=len(rows):
        raise ValueError("Duplicate experiment IDs")
    if any(r["protocol"]!=rows[0]["protocol"] for r in rows[1:]):
        raise ValueError("Incompatible evaluation protocols. Generate separate comparison outputs.")
    out=args.out
    out.mkdir(parents=True)
    keys=["region_recall","small_region_recall","pixel_recall","iou","pixel_fpr"]
    latency_ok=len(rows)==1 or (rows[0]["latency_protocol"]["device"] and
               all(r["latency_protocol"]==rows[0]["latency_protocol"] for r in rows))
    methods={r["flops_method"] for r in rows if r["gflops"] is not None}
    flops_ok=len(methods)<=1
    flat=[dict(experiment=r["experiment"],**{k:r[k] for k in keys+["small_region_count","parameters",
          "inference_mean_ms","gflops"]},latency_comparable=bool(latency_ok),gflops_comparable=flops_ok) for r in rows]
    write(out/"experiment_comparison.csv",flat)
    by_id={r["experiment"]:r for r in rows}
    ablations=[]
    for r in rows:
        if r["ablation"]:
            metadata=r["ablation"]
            baseline=by_id.get(metadata.get("baseline"))
            entry=dict(experiment=r["experiment"],baseline=metadata.get("baseline"),
                       baseline_in_comparison=baseline is not None,change=metadata.get("change"),
                       controls=json.dumps(metadata.get("controls",[]),ensure_ascii=False),**{k:r[k] for k in keys})
            for k in keys:
                entry[f"{k}_delta_pp"]=(100*(r[k]-baseline[k]) if baseline and r[k] is not None
                                       and baseline[k] is not None else None)
            ablations.append(entry)
    write(out/"ablation_study.csv",ablations)
    dump(out/"comparison_metadata.json",dict(protocol=rows[0]["protocol"],latency_comparable=bool(latency_ok),
         gflops_comparable=flops_ok,ablation_metadata_supplied=bool(ablations)))
    names=[r["experiment"] for r in rows]
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout="constrained")
    for ax,k in zip(axes.flat,keys):
        for i,r in enumerate(rows):
            if r[k] is not None:
                ax.bar(i,r[k]*100,color=COLORS[i%len(COLORS)])
        ax.set(xticks=range(len(rows)),xticklabels=names,title=k.replace("_"," ").title(),ylabel="%",
               ylim=(0,max(5,1.1*max(r[k] or 0 for r in rows)*100)) if k=="pixel_fpr" else (0,100))
        ax.tick_params(axis="x",rotation=15)
        if k=="pixel_fpr":
            ax.axhline(rows[0]["protocol"]["fpr_limit"]*100,ls="--",color="#4c5660")
    ax=axes.flat[-1]
    if latency_ok:
        for r in rows:
            ax.scatter(r["inference_mean_ms"],100*r["iou"],label=r["experiment"])
        ax.set(xlabel="Mean inference (ms)",ylabel="Contaminant IoU (%)",ylim=(0,100))
        ax.legend(frameon=False)
    else:
        ax.text(.5,.5,"Latency scopes differ",ha="center")
        ax.axis("off")
    tag="provisional" if rows[0]["protocol"]["provisional"] else "recorded protocol"
    fig.suptitle(f"Validation experiment comparison · {tag}",x=.02,ha="left")
    save(fig,out,"experiment_comparison")
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,key,label,scale in zip(axes,("parameters","gflops"),("Parameters (million)","THOP GFLOPs"),(1e6,1)):
        if key=="gflops" and not flops_ok:
            ax.text(.5,.5,"Profiling methods differ",ha="center",transform=ax.transAxes)
            ax.axis("off")
            continue
        for i,r in enumerate(rows):
            if r[key] is not None:
                ax.bar(i,r[key]/scale,color=COLORS[i%len(COLORS)])
            else:
                ax.text(i,.05,"N/A",ha="center",transform=ax.get_xaxis_transform())
        ax.set(xticks=range(len(rows)),xticklabels=names,ylabel=label)
        ax.tick_params(axis="x",rotation=15)
    fig.suptitle("Model cost at the recorded evaluation shape",x=.02,ha="left")
    save(fig,out,"model_cost_comparison")
    print(out)


def dataset_figures(args):
    out=args.out
    out.mkdir(parents=True)
    rows=read(args.split_csv)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    groups="ABCD"
    x=np.arange(len(groups))
    for index,(split,color) in enumerate(zip(("train","val","test"),COLORS)):
        count=np.array([sum(r["group"]==g and r["split"]==split for r in rows) for g in groups])
        bars=axes[0].bar(x+(index-1)*.25,count,width=.25,label=split,color=color)
        axes[0].bar_label(bars,padding=3,fontsize=9)
    axes[0].set(xticks=x,xticklabels=list(groups),xlabel="Group in fixed split CSV",ylabel="Images")
    axes[0].margins(y=.13)
    axes[0].legend(frameon=False)
    region_rows=read(args.regions)
    val_ids={r["filename"] for r in rows if r["split"]=="val"}
    if any(r["filename"] not in val_ids for r in region_rows):
        raise ValueError("Only Validation region records are accepted")
    sizes=values(region_rows,"area_px")
    axes[1].hist(sizes,bins=np.geomspace(1,512*768,24),color=COLORS[0])
    axes[1].set(xscale="log",xlabel="Validation GT region area (px)",ylabel="Regions")
    fig.suptitle("UC-FoodWaste · fixed split metadata and Validation GT · H512 × W768",x=.02,ha="left")
    save(fig,out,"dataset_distribution")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(required=True)
    render=sub.add_parser("run")
    render.add_argument("run",type=Path)
    render.add_argument("--out",type=Path)
    render.add_argument("--split-csv",type=Path,default=ROOT/"metadata/data_split.csv")
    render.add_argument("--data-root",type=Path,default=ROOT/"data")
    render.add_argument("--profile-checkpoint",action="store_true")
    render.add_argument("--protocol-id",default="ucfw-exp01-provisional-v1",help="Explicit ID for common evaluator conventions")
    render.add_argument("--ablation-metadata",type=Path)
    render.set_defaults(action=run_figures)
    comp=sub.add_parser("compare")
    comp.add_argument("inputs",type=Path,nargs="+")
    comp.add_argument("--out",type=Path,required=True)
    comp.set_defaults(action=compare)
    dataset=sub.add_parser("dataset")
    dataset.add_argument("--split-csv",type=Path,default=ROOT/"metadata/data_split.csv")
    dataset.add_argument("--regions",type=Path,required=True)
    dataset.add_argument("--out",type=Path,required=True)
    dataset.set_defaults(action=dataset_figures)
    args=parser.parse_args()
    try:
        args.action(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2,f"Error: {exc}\n")


if __name__=="__main__":
    main()

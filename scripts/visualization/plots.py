"""Plot recorded metrics and derive Validation-only diagnostics."""
import csv
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
from scipy import ndimage

COLORS = ["#286381", "#b55542", "#327962", "#725486"]
BINS = [0, 1000, 5000, 10000, 25000, np.inf]
BIN_LABELS = ["<1k", "1k–<5k", "5k–<10k", "10k–<25k", "≥25k"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.prop_cycle": plt.cycler(color=COLORS), "svg.fonttype": "none"})


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if rows:
        with path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save(fig, out, name, raster_only=False):
    for ext in (["png"] if raster_only else ["svg", "png"]):
        fig.savefig(out / f"{name}.{ext}", dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def values(rows, key):
    return np.array([float(row[key]) if row.get(key) not in ("", None) else np.nan for row in rows])


def learning(run, out, title, best):
    rows = read(run / "history.csv")
    epochs = values(rows, "epoch")
    panels = [("Train loss", ["train/ce_loss", "train/dice_loss", "train/aux_loss"]),
              ("Validation loss", ["val/ce_loss", "val/dice_loss", "val/aux_loss"]),
              ("Validation score (%)", ["metrics/mIoU", "metrics/pixel_acc"]),
              ("Learning rate", [key for key in rows[0] if key.startswith("lr/")])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, (ylabel, columns) in zip(axes.flat, panels):
        missing = []
        for key in columns:
            y = values(rows, key)
            y[~np.isfinite(y)] = np.nan
            if np.isfinite(y).any():
                ax.plot(epochs, y * (100 if key.startswith("metrics/") else 1), label=key.split("/")[-1], lw=2)
            if np.isnan(y).any():
                missing.append(f"{key.split('/')[-1]}: {np.isnan(y).sum()} non-finite / missing")
        if missing:
            if ylabel == "Validation loss" and (run/"error_analysis/dice_reproduction.json").is_file():
                missing.append("FP16 Dice overflow reproduced on CPU")
            ax.text(.98, .96, "\n".join(missing), transform=ax.transAxes, va="top", ha="right", fontsize=9,
                    bbox={"facecolor":"white","edgecolor":"none","alpha":.9,"pad":3})
        ax.axvline(best, color="#657382", linestyle="--")
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if ylabel.endswith("(%)"):
            ax.set_ylim(0, 100)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, loc="lower left" if missing else "best")
        ax.grid(alpha=.2)
    fig.suptitle(f"{title} · best epoch {best}", x=.02, ha="left")
    save(fig,out,"learning_curves")


def threshold(rows, out, title, chosen, limit, provisional):
    t, fpr, region = (values(rows, k) for k in ("threshold","pixel_fpr","region_recall"))
    pick = int(np.argmin(abs(t-chosen)))
    fig, ax = plt.subplots(2,2,figsize=(12,7),layout="constrained")
    for key,label in (("region_recall","Region Recall"),("pixel_recall","Pixel Recall"),("iou","IoU")):
        ax[0,0].plot(t,values(rows,key)*100,label=label)
    ax[0,0].legend(frameon=False)
    ax[0,0].set_ylabel("Score (%)")
    ax[0,1].plot(t,fpr*100)
    ax[0,1].axhline(limit*100,color=COLORS[1],ls="--")
    ax[0,1].set_ylabel("Pixel FPR (%)")
    for panel in ax[0]:
        panel.axvline(chosen,color="#4c5660",ls="--")
        panel.set(xlabel="Probability threshold",xlim=(0,1),ylim=(0,100))
    ax[1,0].plot(fpr[::-1]*100,region[::-1]*100)
    ax[1,0].scatter(fpr[pick]*100,region[pick]*100,color=COLORS[1],zorder=4)
    xx=np.r_[0.,fpr[::-1]*100]
    pro_key="provisional_spro" if "provisional_spro" in rows[0] else "spro"
    yy=np.r_[0.,values(rows,pro_key)[::-1]*100]
    ax[1,1].plot(xx,yy)
    ax[1,1].fill_between(xx,0,yy,where=xx<=limit*100,interpolate=True,alpha=.22)
    ax[1,0].set_ylabel("Region Recall (%)")
    ax[1,1].set_ylabel("Per-region overlap (%)")
    for panel in ax[1]:
        panel.axvline(limit*100,color=COLORS[1],ls="--")
        panel.set(xlabel="Pixel FPR (%)",xlim=(0,max(5,limit*200)),ylim=(0,100))
    for panel in ax.flat:
        panel.grid(alpha=.2)
    fig.suptitle(f"{title} · Validation threshold {chosen:g} · FPR cap {limit:.1%}"+
                 (" · provisional" if provisional else ""),x=.02,ha="left")
    save(fig,out,"threshold_analysis")


def montage(out, name, title, rgb, gt, prob, pred, crop=None):
    overlay = rgb.copy()
    prediction_overlay = rgb.copy()
    prediction_overlay[pred] = (.35*rgb[pred]+.65*np.array((249,167,44))).astype(np.uint8)
    for mask, color in ((gt&pred,(67,190,120)),(~gt&pred,(249,167,44)),(gt&~pred,(228,55,103))):
        overlay[mask]=(.35*rgb[mask]+.65*np.array(color)).astype(np.uint8)
    crop=crop or np.s_[:,:]
    fig, axes=plt.subplots(1,6,figsize=(16,3.4),layout="constrained")
    for ax,a,label in zip(axes,(rgb,gt,prob,pred,prediction_overlay,overlay),
            ("Source RGB","GT mask","Probability","Predicted mask","Prediction overlay","TP green · FP amber · FN pink")):
        im=ax.imshow(a[crop],cmap="magma" if label=="Probability" else "gray",vmin=0,vmax=1,interpolation="nearest")
        if label=="Probability":
            fig.colorbar(im,ax=ax,shrink=.65)
        ax.set_title(label,fontsize=10)
        ax.axis("off")
    fig.suptitle(title,x=.02,ha="left")
    save(fig,out,name,raster_only=True)


def diagnostics(run,out,source,split,config,chosen,title):
    from yolo_sem.data import label, rgb_path
    probability_dir=run/"probability_maps/val"
    precomputed={r["filename"]:r for r in read(run/"val_results.csv")}
    if {r["filename"] for r in split}!={p.stem for p in probability_dir.glob("*.npy")} or set(precomputed)!={r["filename"] for r in split}:
        raise ValueError("Validation split, maps and metric IDs differ")
    h,w=config["height"],config["width"]
    regions,examples,distances=[],{},{}
    ground_truth_hash=hashlib.sha256()
    views=out/"examples"
    views.mkdir()
    for row in sorted(split,key=lambda row:row["filename"]):
        name,g=row["filename"],row["group"]
        with Image.open(source/"masks"/f"{name}.png") as image:
            gt=np.asarray(image.resize((w,h),Image.Resampling.NEAREST))!=0
        ground_truth_hash.update(name.encode("utf-8")+b"\0"+gt.tobytes())
        p=np.load(probability_dir/f"{name}.npy",allow_pickle=False)
        if p.shape!=gt.shape or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
            raise ValueError(f"Invalid probability map: {name}")
        pred=p>=np.asarray(chosen,dtype=p.dtype)
        fp=pred&~gt
        labels,n=label(gt,config["connectivity"])
        hits=0
        crop=None
        for i,box in enumerate(ndimage.find_objects(labels),1):
            component=labels[box]==i
            scores=p[box][component]
            detected=float(pred[box][component].mean())>=config["region_overlap"]
            hits+=detected
            regions.append(dict(filename=name,group=g,region=i,area_px=len(scores),detected=int(detected),
                                mean_probability=float(scores.mean()),max_probability=float(scores.max())))
            if not detected and len(scores)<5000 and crop is None:
                y,x=box
                crop=np.s_[max(0,y.start-24):min(h,y.stop+24),max(0,x.start-24):min(w,x.stop+24)]
        recorded=precomputed[name]
        counts={"fp":fp.sum(),"tp":(gt&pred).sum(),"fn":(gt&~pred).sum(),"tn":(~gt&~pred).sum(),
                "regions":n,"regions_detected":hits}
        if any(int(recorded[k])!=v for k,v in counts.items()):
            raise ValueError(f"Saved evaluation counts differ: {name}")
        if fp.any() and gt.any():
            d=ndimage.distance_transform_edt(~gt)[fp]
            distances[g]=distances.get(g,np.zeros(7,dtype=int))+np.histogram(d,[0,2,4,8,16,32,64,np.inf])[0]
        if (crop is not None and g not in examples) or (g=="B" and fp.any()):
            with Image.open(rgb_path(source, name)) as image:
                rgb=np.asarray(image.convert("RGB").resize((w,h),Image.Resampling.BILINEAR))
            montage(views,f"{g}_{name}",f"{title} · {name} · Group {g} · Validation · t={chosen:g}",rgb,gt,p,pred,crop)
            examples[g]=name
    write(out/"region_profile.csv",regions)
    write(out/"fp_distance_bins.csv",[dict(group=g,lower_px=lo,upper_exclusive_px=hi,count=int(n))
          for g,v in distances.items() for lo,hi,n in zip([0,2,4,8,16,32,64],[2,4,8,16,32,64,None],v)])
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout="constrained")
    ax[0,0].hist([[r["area_px"] for r in regions if r["detected"]==d] for d in (1,0)],
                 bins=np.geomspace(1,h*w,23),label=["Detected","Missed"],color=COLORS[:2])
    ax[0,0].set(xscale="log",xlabel="GT region area (px)",ylabel="GT regions")
    ax[0,0].legend(frameon=False)
    recall=[]
    for lo,hi in zip(BINS,BINS[1:]):
        a=[r for r in regions if lo<=r["area_px"]<hi]
        recall.append(100*sum(r["detected"] for r in a)/len(a) if a else np.nan)
    ax[0,1].bar(BIN_LABELS,recall,color=COLORS[2])
    ax[0,1].set(xlabel="GT region area (px)",ylabel="Region Recall (%)",ylim=(0,100))
    for k in ("mean_probability","max_probability"):
        ax[1,0].hist([r[k] for r in regions if not r["detected"]],bins=np.linspace(0,1,16),
                    histtype="step",lw=2,label=k.split("_")[0])
    ax[1,0].axvline(chosen,color="#4c5660",ls="--")
    ax[1,0].set(xlabel="Probability within missed GT",ylabel="Missed regions")
    ax[1,0].legend(frameon=False)
    for g,n in sorted(distances.items()):
        ax[1,1].plot(["<2","2–<4","4–<8","8–<16","16–<32","32–<64","≥64"],100*n/n.sum(),marker="o",label=g)
    ax[1,1].set(xlabel="FP distance to nearest GT pixel (px)",ylabel="FP within group (%)")
    if distances:
        ax[1,1].legend(frameon=False)
    fig.suptitle(f"{title} · Validation errors · threshold {chosen:g} · 8/4-connectivity per run",x=.02,ha="left")
    save(fig,out,"error_analysis")
    groups=read(run/"val_groups.csv")
    fig,axes=plt.subplots(1,4,figsize=(12,3.4),layout="constrained")
    for ax,k in zip(axes,("region_recall","pixel_recall","iou","pixel_fpr")):
        vals=100*values(groups,k)
        for i,value in enumerate(vals):
            if np.isfinite(value):
                ax.bar(i,value,color=COLORS[i%len(COLORS)])
            else:
                ax.text(i,.04,"N/A",ha="center",transform=ax.get_xaxis_transform())
        ax.set(xticks=range(len(groups)),xticklabels=[f"{r['group']}\n(n={r['images']})" for r in groups],
               title=k.replace("_"," ").title(),ylabel="%",ylim=(0,100) if k!="pixel_fpr" else (0,max(5,np.nanmax(vals)*1.05)))
    fig.suptitle(f"{title} · Validation groups · threshold {chosen:g}",x=.02,ha="left")
    save(fig,out,"group_metrics")
    small=[r for r in regions if r["area_px"]<5000]
    return ((sum(r["detected"] for r in small)/len(small) if small else None),len(small),
            ground_truth_hash.hexdigest())

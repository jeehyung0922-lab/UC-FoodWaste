"""Model-independent pixel and polygon metrics."""
import numpy as np


def ratio(a,b):
    return float(a/b) if b else None


def pixel_metrics(tp,fp,fn,tn):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                pixel_recall=ratio(tp,tp+fn), pixel_precision=ratio(tp,tp+fp),
                iou=ratio(tp,tp+fp+fn), dice=ratio(2*tp,2*tp+fp+fn),
                pixel_accuracy=ratio(tp+tn,tp+fp+fn+tn), pixel_fpr=ratio(fp,fp+tn))


def sweep(prob, gt, regions, grid):
    pos = np.zeros(prob.size, dtype=bool)
    pos[gt] = True
    positive, negative = np.sort(prob[pos]), np.sort(prob[~pos])
    tp = len(positive)-np.searchsorted(positive,grid)
    fp = len(negative)-np.searchsorted(negative,grid)
    return np.stack([tp,fp,len(positive)-tp,len(negative)-fp],axis=1), [
        len(region)-np.searchsorted(np.sort(prob[region]),grid) for region in regions]


def region_metric(hits, areas):
    if not len(areas):
        return 0, None, None
    # Integer comparison ensures exactly 10% is detected.
    detected = int(np.count_nonzero(hits*10 >= areas))
    return detected, ratio(detected,len(areas)), float(np.minimum(hits*10/areas,1).mean())


def sweep_regions(region_hits, areas, grid):
    if not len(areas):
        return np.zeros(len(grid),dtype=np.int64), np.full(len(grid),np.nan)
    fractions = region_hits*10/areas[:,None]
    return (region_hits*10>=areas[:,None]).sum(axis=0), np.minimum(fractions,1).mean(axis=0)


def select(grid, counts, detected, spro, cap):
    negatives = counts[:,1]+counts[:,3]
    if not np.all(negatives>0) or not np.isfinite(spro).any():
        raise ValueError("Threshold selection requires background pixels and foreground regions")
    eligible = np.flatnonzero((counts[:,1]<=cap*negatives)&np.isfinite(spro))
    if not len(eligible):
        raise ValueError("No candidate meets the FPR cap")
    best = eligible[detected[eligible]==detected[eligible].max()]
    best = best[spro[best] >= spro[best].max()-1e-12]
    return int(max(best,key=lambda i:grid[i]))


def auspro(x,y,cap):
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    unique = np.unique(x)
    if unique[0] != 0 or unique[-1] < cap:
        return None
    upper = np.array([y[x==value].max() for value in unique])
    keep = unique<cap
    xx = np.r_[unique[keep],cap]
    yy = np.r_[upper[keep],np.interp(cap,unique,upper)]
    return float(np.sum(np.diff(xx)*(yy[:-1]+yy[1:])*.5)/cap)

"""Binary pixel metrics and a provisional connected-region protocol."""
import numpy as np

from .data import label


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(counts):
    tp, fp, fn, tn, found, total = map(int, counts)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, regions_detected=found, regions=total,
                region_recall=ratio(found, total), pixel_recall=ratio(tp, tp + fn),
                iou=ratio(tp, tp + fp + fn), pixel_fpr=ratio(fp, fp + tn),
                pixel_precision=ratio(tp, tp + fp), dice=ratio(2 * tp, 2 * tp + fp + fn))


def sweep(mask, probabilities, thresholds, connectivity, overlap):
    if (mask.shape != probabilities.shape or mask.ndim != 2 or
            not np.isin(mask, [0, 1]).all() or not np.isfinite(probabilities).all() or
            probabilities.min() < 0 or probabilities.max() > 1):
        raise ValueError("Expected aligned binary mask and finite 2D probabilities in [0,1]")
    mask = mask.astype(bool)
    # Compare in the model map dtype consistently for sweeps and final masks.
    thresholds = np.asarray(thresholds, dtype=probabilities.dtype)
    positive, negative = np.sort(probabilities[mask]), np.sort(probabilities[~mask])
    tp = positive.size - np.searchsorted(positive, thresholds)
    fp = negative.size - np.searchsorted(negative, thresholds)
    labels, num_regions = label(mask, connectivity)
    found = np.zeros(len(thresholds), dtype=np.int64)
    pro_sum = np.zeros(len(thresholds))
    if num_regions:
        ids = labels[mask]
        grouped = probabilities[mask][np.argsort(ids, kind="stable")]
        ends = np.cumsum(np.bincount(ids)[1:])
        for region in np.split(grouped, ends[:-1]):
            fractions = (region.size - np.searchsorted(np.sort(region), thresholds)) / region.size
            found += fractions >= overlap
            pro_sum += fractions
    counts = np.stack([tp, fp, positive.size - tp, negative.size - fp, found,
                       np.full_like(tp, num_regions)], axis=1)
    return counts, pro_sum


def select_threshold(thresholds, counts, limit):
    positive_regions = counts[:, 5] > 0
    negative_pixels = counts[:, 1] + counts[:, 3]
    eligible = np.flatnonzero(positive_regions & (negative_pixels > 0) &
                              (counts[:, 1] <= limit * negative_pixels))
    if not eligible.size:
        raise ValueError("No validation threshold satisfies the FPR constraint with defined metrics")
    # On the same validation set, the denominators are constant.
    return max(eligible, key=lambda i: (counts[i, 4], -counts[i, 1], thresholds[i]))


def area_under_pro(counts, pro_sum, limit):
    """Provisional normalized area of region overlap versus FPR up to limit."""
    if not counts[0, 5] or not counts[0, 1] + counts[0, 3]:
        return None
    # Reversing descending operating strictness preserves vertical ROC segments.
    fpr = (counts[:, 1] / (counts[:, 1] + counts[:, 3]))[::-1]
    pro = (pro_sum / counts[:, 5])[::-1]
    x = np.r_[0.0, fpr]
    y = np.r_[0.0, pro]
    idx = np.searchsorted(x, limit, side="right")
    if idx == len(x):
        return None
    endpoint = y[idx-1] + (y[idx] - y[idx-1]) * (limit - x[idx-1]) / (x[idx] - x[idx-1])
    x, y = np.r_[x[:idx], limit], np.r_[y[:idx], endpoint]
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2) / limit)

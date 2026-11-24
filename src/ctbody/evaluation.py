"""Segmentation evaluation: Dice and 95th-percentile Hausdorff distance.

Used to validate a model's masks against reference (ground-truth) masks,
e.g. TotalSegmentator predictions vs. the manually reviewed labels in the
public TotalSegmentator dataset.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    """Dice similarity coefficient. Both empty -> 1.0 (correct negative)."""
    p, r = pred.astype(bool), ref.astype(bool)
    denom = p.sum() + r.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, r).sum() / denom)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def hd95(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, ...]) -> float:
    """95th-percentile symmetric Hausdorff distance in mm.

    Max of the two directed 95th percentiles (same convention as MONAI).
    Both empty -> 0.0; exactly one empty -> inf.
    """
    p, r = pred.astype(bool), ref.astype(bool)
    if not p.any() and not r.any():
        return 0.0
    if not p.any() or not r.any():
        return float("inf")
    sp, sr = _surface(p), _surface(r)
    dt_r = ndimage.distance_transform_edt(~sr, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d_pr = dt_r[sp]
    d_rp = dt_p[sr]
    return float(max(np.percentile(d_pr, 95), np.percentile(d_rp, 95)))

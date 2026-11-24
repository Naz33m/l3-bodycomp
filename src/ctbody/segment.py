"""Segmentation via TotalSegmentator (https://github.com/wasserth/TotalSegmentator).

Two paths:
* `tissue_types` task -> full skeletal muscle, subcutaneous and visceral fat.
  Needs a (free, non-commercial) TotalSegmentator licence key.
* Fallback: open `total` task -> psoas (iliopsoas) + paraspinal (autochthon)
  muscles only. Reported as a partial muscle area, NOT as full SMA.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .io import load_mask

log = logging.getLogger(__name__)

TOTAL_ROIS = [
    "vertebrae_L3",
    "iliopsoas_left",
    "iliopsoas_right",
    "autochthon_left",
    "autochthon_right",
]
PARTIAL_MUSCLES = TOTAL_ROIS[1:]


@dataclass
class Masks:
    l3: np.ndarray
    muscle: np.ndarray
    muscle_source: str                     # "tissue_types" | "psoas+paraspinal"
    sat: Optional[np.ndarray] = None       # subcutaneous fat
    vat: Optional[np.ndarray] = None       # visceral ("torso") fat
    files: dict[str, str] = field(default_factory=dict)


def _ts():
    from totalsegmentator.python_api import totalsegmentator

    return totalsegmentator


def segment(ct_nifti: Path, out_dir: Path, fast: bool = True, device: str = "gpu",
            use_tissue_types: bool = True) -> Masks:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_dir = out_dir / "total"
    _ts()(str(ct_nifti), str(total_dir), task="total", roi_subset=TOTAL_ROIS,
          fast=fast, device=device)
    l3 = load_mask(total_dir / "vertebrae_L3.nii.gz")

    if use_tissue_types:
        tissue_dir = out_dir / "tissue"
        try:
            _ts()(str(ct_nifti), str(tissue_dir), task="tissue_types", device=device)
            return Masks(
                l3=l3,
                muscle=load_mask(tissue_dir / "skeletal_muscle.nii.gz"),
                muscle_source="tissue_types",
                sat=load_mask(tissue_dir / "subcutaneous_fat.nii.gz"),
                vat=load_mask(tissue_dir / "torso_fat.nii.gz"),
            )
        except Exception as exc:  # licence missing, OOM, etc.
            log.warning("tissue_types unavailable (%s); falling back to psoas+paraspinal", exc)

    muscle = np.zeros_like(l3)
    for name in PARTIAL_MUSCLES:
        muscle |= load_mask(total_dir / f"{name}.nii.gz")
    return Masks(l3=l3, muscle=muscle, muscle_source="psoas+paraspinal")

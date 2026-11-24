"""Pure-numpy body-composition metrics at the L3 vertebral level.

Everything here works on plain arrays so it can be unit-tested with
synthetic phantoms, independent of any segmentation model.

Conventions
-----------
* Volumes are indexed (x, y, z) in RAS+ orientation (nibabel canonical),
  so z increases toward the head.
* `spacing` is (sx, sy, sz) in millimetres.
* CT values are Hounsfield units (HU).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import numpy as np

# Standard HU range for skeletal muscle (Mitsiopoulos et al., 1998).
MUSCLE_HU = (-29.0, 150.0)
# Standard HU range for adipose tissue.
FAT_HU = (-190.0, -30.0)

Sex = Literal["male", "female"]


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
def window(hu: np.ndarray, level: float = 40.0, width: float = 400.0) -> np.ndarray:
    """Map HU to [0, 1] using a display window (default: soft tissue W400/L40)."""
    lo = level - width / 2.0
    return np.clip((hu.astype(np.float32) - lo) / width, 0.0, 1.0)


# --------------------------------------------------------------------------
# Slice selection
# --------------------------------------------------------------------------
def l3_slice_index(l3_mask: np.ndarray) -> int:
    """Axial slice through the L3 vertebra's centre of mass along z.

    Raises ValueError if the mask is empty (L3 not in the field of view),
    which is a common real-world failure: chest-only or pelvis-only scans.
    """
    if l3_mask.ndim != 3:
        raise ValueError(f"expected a 3D mask, got shape {l3_mask.shape}")
    counts = l3_mask.astype(bool).sum(axis=(0, 1))
    if counts.sum() == 0:
        raise ValueError("L3 vertebra not found in scan (outside field of view?)")
    z = np.arange(counts.size)
    return int(round(float((counts * z).sum() / counts.sum())))


# --------------------------------------------------------------------------
# Areas
# --------------------------------------------------------------------------
def area_cm2(mask2d: np.ndarray, spacing_xy: tuple[float, float]) -> float:
    """Area of a 2D mask in cm² (pixel count × pixel area in mm² / 100)."""
    return float(mask2d.astype(bool).sum() * spacing_xy[0] * spacing_xy[1] / 100.0)


@dataclass
class TissueStats:
    area_cm2: float
    mean_hu: Optional[float]
    pixels: int


def tissue_stats(
    hu2d: np.ndarray,
    mask2d: np.ndarray,
    spacing_xy: tuple[float, float],
    hu_range: Optional[tuple[float, float]] = None,
) -> TissueStats:
    """Area and mean attenuation of a tissue on one slice.

    If `hu_range` is given, the mask is intersected with that HU window,
    which is the standard way to clean up segmentation edges
    (e.g. muscle restricted to -29..150 HU).
    """
    m = mask2d.astype(bool)
    if hu_range is not None:
        m &= (hu2d >= hu_range[0]) & (hu2d <= hu_range[1])
    n = int(m.sum())
    mean = float(hu2d[m].mean()) if n else None
    return TissueStats(area_cm2=area_cm2(m, spacing_xy), mean_hu=mean, pixels=n)


# --------------------------------------------------------------------------
# Indices and clinical cut-offs
# --------------------------------------------------------------------------
def skeletal_muscle_index(sma_cm2: float, height_cm: float) -> float:
    """SMI = skeletal muscle area / height² (cm²/m²)."""
    if height_cm <= 0:
        raise ValueError("height must be positive")
    h = height_cm / 100.0
    return sma_cm2 / (h * h)


def bmi(weight_kg: float, height_cm: float) -> float:
    h = height_cm / 100.0
    return weight_kg / (h * h)


@dataclass
class Classification:
    flag: bool
    threshold: float
    criteria: str


def sarcopenia(smi: float, sex: Sex, bmi_value: Optional[float] = None) -> Classification:
    """Classify L3 sarcopenia.

    With BMI: Martin et al., J Clin Oncol 2013 (BMI-dependent cut-offs).
    Without BMI: Prado et al., Lancet Oncol 2008.
    These are research cut-offs from oncology cohorts, not a diagnosis.
    """
    if bmi_value is not None:
        if sex == "male":
            t = 43.0 if bmi_value < 25 else 53.0
        else:
            t = 41.0
        return Classification(smi < t, t, "Martin 2013")
    t = 52.4 if sex == "male" else 38.5
    return Classification(smi < t, t, "Prado 2008")


def myosteatosis(mean_muscle_hu: float, bmi_value: float) -> Classification:
    """Low muscle attenuation (fatty infiltration), Martin et al. 2013."""
    t = 41.0 if bmi_value < 25 else 33.0
    return Classification(mean_muscle_hu < t, t, "Martin 2013")


def as_dict(obj) -> dict:
    return asdict(obj)

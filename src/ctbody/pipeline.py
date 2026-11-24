"""End-to-end: CT -> masks -> L3 slice -> body-composition report."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import metrics as M
from .io import load_volume, to_nifti
from .report import render
from .segment import Masks, segment

DISCLAIMER = "Research use only. Not a medical device; not for diagnostic use."


def compute(hu: np.ndarray, spacing, masks: Masks, height_cm: float, sex: M.Sex,
            weight_kg: Optional[float] = None) -> dict:
    """Pure computation from arrays + masks (no model), so it is testable."""
    z = M.l3_slice_index(masks.l3)
    hu2d = hu[:, :, z]
    sxy = (spacing[0], spacing[1])

    muscle = M.tissue_stats(hu2d, masks.muscle[:, :, z], sxy, M.MUSCLE_HU)
    result: dict = {
        "l3_slice_index": z,
        "spacing_mm": list(spacing),
        "muscle_source": masks.muscle_source,
        "muscle": M.as_dict(muscle),
        "warnings": [],
    }
    if masks.sat is not None:
        result["sat"] = M.as_dict(M.tissue_stats(hu2d, masks.sat[:, :, z], sxy, M.FAT_HU))
    if masks.vat is not None:
        result["vat"] = M.as_dict(M.tissue_stats(hu2d, masks.vat[:, :, z], sxy, M.FAT_HU))

    if spacing[2] > 5.0:
        result["warnings"].append(f"thick slices ({spacing[2]:.1f} mm): L3 level less precise")

    smi = M.skeletal_muscle_index(muscle.area_cm2, height_cm)
    bmi_value = M.bmi(weight_kg, height_cm) if weight_kg else None
    result["bmi"] = bmi_value

    if masks.muscle_source == "tissue_types":
        result["smi_cm2_m2"] = smi
        result["sarcopenia"] = M.as_dict(M.sarcopenia(smi, sex, bmi_value))
        if bmi_value is not None and muscle.mean_hu is not None:
            result["myosteatosis"] = M.as_dict(M.myosteatosis(muscle.mean_hu, bmi_value))
    else:
        # Cut-offs are defined for total SMA only; don't misapply them.
        result["psoas_paraspinal_index_cm2_m2"] = smi
        result["warnings"].append(
            "Partial muscle (psoas + paraspinal) only: sarcopenia cut-offs not applied."
        )
    return result


def analyze(ct_path: str | Path, out_dir: str | Path, height_cm: float, sex: M.Sex,
            weight_kg: Optional[float] = None, fast: bool = True, device: str = "gpu",
            use_tissue_types: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    nifti = to_nifti(ct_path, out)
    vol = load_volume(nifti)
    t_load = time.perf_counter()

    masks = segment(nifti, out / "seg", fast=fast, device=device,
                    use_tissue_types=use_tissue_types)
    t_seg = time.perf_counter()

    result = compute(vol.hu, vol.spacing, masks, height_cm, sex, weight_kg)
    result["input"] = {"height_cm": height_cm, "sex": sex, "weight_kg": weight_kg}
    result["timing_s"] = {
        "load": round(t_load - t0, 2),
        "segmentation": round(t_seg - t_load, 2),
        "total": round(time.perf_counter() - t0, 2),
    }
    result["disclaimer"] = DISCLAIMER

    render(vol.hu, masks, result, out)
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

"""Loading CT scans (NIfTI or DICOM series) into a canonical RAS+ volume."""
from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Volume:
    hu: np.ndarray                      # (x, y, z) float32, Hounsfield units
    spacing: tuple[float, float, float]  # mm
    affine: np.ndarray
    nifti_path: Path                     # path fed to the segmentation model


def to_nifti(path: str | Path, workdir: Path) -> Path:
    """Return a NIfTI path for a .nii/.nii.gz file, a DICOM folder, or a zip of DICOMs."""
    p = Path(path)
    if p.suffix in {".nii", ".gz"}:
        return p
    if p.suffix == ".zip":
        extract = Path(tempfile.mkdtemp(dir=workdir))
        with zipfile.ZipFile(p) as zf:
            zf.extractall(extract)
        p = extract
    if p.is_dir():
        import dicom2nifti  # lazy: only needed for DICOM input

        out = workdir / "ct.nii.gz"
        dicom2nifti.dicom_series_to_nifti(str(p), str(out), reorient_nifti=True)
        return out
    raise ValueError(f"unsupported input: {p} (expected .nii.gz, DICOM folder or .zip)")


def load_volume(nifti_path: Path) -> Volume:
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(nifti_path)))
    hu = np.asanyarray(img.dataobj).astype(np.float32)
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    return Volume(hu=hu, spacing=(sx, sy, sz), affine=img.affine, nifti_path=nifti_path)


def load_mask(path: Path) -> np.ndarray:
    """Load a binary mask in the same canonical orientation as load_volume."""
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asanyarray(img.dataobj) > 0

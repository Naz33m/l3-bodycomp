"""Command line: `ctbody analyze ...` and `ctbody evaluate ...`."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _analyze(a: argparse.Namespace) -> None:
    from .pipeline import analyze

    r = analyze(a.ct, a.out, a.height, a.sex, a.weight, fast=not a.full_res,
                device=a.device, use_tissue_types=not a.no_tissue)
    print(json.dumps(r, indent=2))
    print(f"\nReport: {Path(a.out) / 'report.html'}", file=sys.stderr)


def _evaluate(a: argparse.Namespace) -> None:
    """Compare predicted masks to reference masks over a set of cases.

    Layout (matches the public TotalSegmentator dataset):
        <pred_root>/<case>/<structure>.nii.gz
        <ref_root>/<case>/segmentations/<structure>.nii.gz
    """
    from .evaluation import dice, hd95
    from .io import load_mask

    import nibabel as nib

    rows = []
    for case in sorted(p for p in Path(a.pred).iterdir() if p.is_dir()):
        ref = Path(a.ref) / case.name / "segmentations" / f"{a.structure}.nii.gz"
        pred = case / f"{a.structure}.nii.gz"
        if not (ref.exists() and pred.exists()):
            continue
        spacing = nib.as_closest_canonical(nib.load(str(ref))).header.get_zooms()[:3]
        p, r = load_mask(pred), load_mask(ref)
        rows.append({"case": case.name, "dice": dice(p, r), "hd95_mm": hd95(p, r, spacing),
                     "ref_empty": not r.any()})

    if not rows:
        sys.exit("no matching cases found")
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    d = np.array([x["dice"] for x in rows if not x["ref_empty"]])
    h = np.array([x["hd95_mm"] for x in rows if not x["ref_empty"] and np.isfinite(x["hd95_mm"])])
    print(f"{a.structure}: n={len(d)}  Dice {d.mean():.3f} ± {d.std():.3f} "
          f"(median {np.median(d):.3f})  HD95 {np.median(h):.1f} mm median")
    worst = sorted((x for x in rows if not x["ref_empty"]), key=lambda x: x["dice"])[:5]
    print("worst cases:", ", ".join(f"{x['case']} ({x['dice']:.2f})" for x in worst))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ctbody")
    sub = ap.add_subparsers(required=True)

    an = sub.add_parser("analyze", help="L3 body composition from a CT")
    an.add_argument("ct", help=".nii.gz, DICOM folder, or .zip of DICOMs")
    an.add_argument("--out", default="out")
    an.add_argument("--height", type=float, required=True, help="cm")
    an.add_argument("--sex", choices=["male", "female"], required=True)
    an.add_argument("--weight", type=float, help="kg (enables BMI-based cut-offs)")
    an.add_argument("--device", default="gpu", help="gpu | cpu | gpu:1")
    an.add_argument("--full-res", action="store_true", help="1.5 mm model instead of fast 3 mm")
    an.add_argument("--no-tissue", action="store_true", help="skip tissue_types (licensed) task")
    an.set_defaults(func=_analyze)

    ev = sub.add_parser("evaluate", help="Dice / HD95 against reference masks")
    ev.add_argument("--pred", required=True)
    ev.add_argument("--ref", required=True)
    ev.add_argument("--structure", default="vertebrae_L3")
    ev.add_argument("--csv", default="eval.csv")
    ev.set_defaults(func=_evaluate)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()

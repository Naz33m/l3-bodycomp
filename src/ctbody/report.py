"""Overlay image + self-contained HTML report."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from .metrics import window


def _orient(a2d: np.ndarray) -> np.ndarray:
    """RAS (x, y) slice -> radiological display: anterior up, patient right on image left."""
    return a2d.T[::-1, ::-1]


def _overlay_png(hu2d, layers, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = _orient(window(hu2d))
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(img, cmap="gray")
    for mask, color in layers:
        rgba = np.zeros((*img.shape, 4))
        rgba[_orient(mask.astype(bool))] = (*color, 0.45)
        ax.imshow(rgba)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path)
    plt.close(fig)


def render(hu, masks, result: dict, out: Path) -> None:
    z = result["l3_slice_index"]
    layers = [(masks.muscle[:, :, z], (0.85, 0.2, 0.2))]
    if masks.sat is not None:
        layers.append((masks.sat[:, :, z], (0.95, 0.8, 0.2)))
    if masks.vat is not None:
        layers.append((masks.vat[:, :, z], (0.2, 0.7, 0.9)))
    _overlay_png(hu[:, :, z], layers, out / "l3_overlay.png")

    rows = []
    m = result["muscle"]
    label = "Skeletal muscle" if result["muscle_source"] == "tissue_types" else "Psoas + paraspinal"
    rows.append((f"{label} area", f"{m['area_cm2']:.1f} cm²"))
    if m["mean_hu"] is not None:
        rows.append(("Muscle attenuation", f"{m['mean_hu']:.1f} HU"))
    if "smi_cm2_m2" in result:
        rows.append(("Skeletal muscle index", f"{result['smi_cm2_m2']:.1f} cm²/m²"))
        s = result["sarcopenia"]
        rows.append(("Sarcopenia", f"{'YES' if s['flag'] else 'no'} (< {s['threshold']}, {s['criteria']})"))
    if "psoas_paraspinal_index_cm2_m2" in result:
        rows.append(("Psoas+paraspinal index", f"{result['psoas_paraspinal_index_cm2_m2']:.1f} cm²/m²"))
    if "myosteatosis" in result:
        my = result["myosteatosis"]
        rows.append(("Myosteatosis", f"{'YES' if my['flag'] else 'no'} (< {my['threshold']} HU)"))
    for key, name in (("sat", "Subcutaneous fat"), ("vat", "Visceral fat")):
        if key in result:
            rows.append((f"{name} area", f"{result[key]['area_cm2']:.1f} cm²"))

    table = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows)
    warns = "".join(f"<li>{html.escape(w)}</li>" for w in result.get("warnings", []))
    timing = result.get("timing_s", {})
    (out / "report.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>L3 body composition</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1d1d1f}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:2rem;align-items:start}}
img{{width:100%;border-radius:8px}} table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:.45rem .3rem;border-bottom:1px solid #e5e5e5}} th{{font-weight:500;color:#555}}
.warn{{color:#a15c00}} .disc{{font-size:.85rem;color:#777;margin-top:2rem}}
</style></head><body>
<h1>L3 body composition</h1>
<div class="grid"><img src="l3_overlay.png" alt="L3 axial slice with tissue overlay">
<div><table>{table}</table>
<p>L3 slice index {z} · segmentation {timing.get('segmentation', '?')} s · total {timing.get('total', '?')} s</p>
{f'<ul class="warn">{warns}</ul>' if warns else ''}</div></div>
<p class="disc">{html.escape(result.get('disclaimer', ''))}</p>
</body></html>""", encoding="utf-8")

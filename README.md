# ctbody: L3 body composition from abdominal CT

Automated **sarcopenia / body-composition** analysis from a routine abdominal CT:

```
CT (DICOM / NIfTI)
  └─▶ canonical RAS volume, Hounsfield units
       └─▶ TotalSegmentator ──▶ L3 vertebra + skeletal muscle / SAT / VAT masks
            └─▶ L3 axial slice (centre of mass of the vertebra)
                 └─▶ muscle area (−29…150 HU), mean muscle HU, SAT/VAT area
                      └─▶ SMI = SMA / height²  ──▶ sarcopenia & myosteatosis flags
                           └─▶ JSON + overlay PNG + HTML report  (CLI or FastAPI)
```

> **Research use only.** Not a medical device and not for diagnostic use.
> Use public, de-identified data only.

## Why this exists

Oncologists measure L3 skeletal muscle by hand on CT to estimate sarcopenia, which
predicts chemotherapy toxicity and survival. It takes minutes per scan and varies
between readers. This project automates it end to end and, just as importantly,
**measures how well it works** (see *Evaluation*).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# GPU: install the CUDA build of torch first, per pytorch.org

pytest                                                  # phantom tests, no model needed

ctbody analyze path/to/ct.nii.gz --height 172 --sex male --weight 70 --out out/case1
# open out/case1/report.html

uvicorn ctbody.api:app --port 8000                     # web UI at http://localhost:8000
```

### Full skeletal muscle needs a TotalSegmentator licence

The `tissue_types` task (whole skeletal muscle, subcutaneous and visceral fat) needs a
free non-commercial licence key from the TotalSegmentator project:
`totalseg_set_license -l <key>`. Without it the pipeline falls back to the open
`total` task, reports **psoas + paraspinal** muscle area only, and deliberately
does **not** apply sarcopenia cut-offs, because those are defined for total SMA.

## Data

* **TotalSegmentator dataset** (Zenodo, CC BY 4.0): 1,200+ CTs with reference masks.
  Use it for both demo inputs and evaluation.
* **The Cancer Imaging Archive (TCIA)**: oncology CT collections (DICOM). Choose a
  series that covers the abdomen, since L3 must be in the field of view.

## Evaluation

```bash
# 1. run the model over N cases → preds/<case>/vertebrae_L3.nii.gz
# 2. score against reference masks
ctbody evaluate --pred preds --ref totalseg_dataset --structure vertebrae_L3 --csv eval.csv
```

Reports mean ± SD and median Dice, median HD95 (mm) and the worst 5 cases. Look at
the worst cases: they usually come from the scan (no L3 in view, metal, contrast
phase, thick slices), not from the model.

| Structure | n | Dice (mean ± SD) | HD95 median | Notes |
|---|---|---|---|---|
| vertebrae_L3 | _fill in_ | _fill in_ | _fill in_ | |

## Design decisions

* **Pure metrics module.** `metrics.py` and `evaluation.py` are numpy-only and tested
  on synthetic phantoms with known geometry, independent of any model.
* **HU gating inside the mask** (−29…150 HU for muscle, −190…−30 HU for fat) is
  standard practice and makes area robust to boundary errors (see the tests).
* **Single L3 slice at the vertebra's centre of mass**, as in the clinical studies.
  Averaging several slices is a natural next step.
* **Explicit failure modes:** L3 not in view gives HTTP 422; thick slices (>5 mm)
  produce a warning; partial-muscle mode never claims sarcopenia.
* **Cut-offs:** Martin et al. 2013 (BMI-dependent) when weight is given, otherwise
  Prado et al. 2008. Myosteatosis follows Martin 2013.

## Limitations / next steps

* Validate SMA against manual L3 annotations (Bland–Altman, ICC), not only masks.
* L3 slice choice versus a radiologist's pick: measure the distance in slices.
* Queue and GPU worker instead of synchronous inference; DICOM SEG output; a
  PACS/DICOMweb integration path.
* Pre-processing QA: detect contrast phase, metal artefacts and truncated FOV.
* Regulatory: versioned models, audit trail of inputs → outputs, IEC 62304-style
  traceability from requirement to test.

## Layout

```
src/ctbody/
  io.py          DICOM/NIfTI → canonical volume
  segment.py     TotalSegmentator wrapper (+ licence fallback)
  metrics.py     L3 slice, areas, SMI, cut-offs (pure numpy)
  evaluation.py  Dice, HD95
  pipeline.py    end to end, timing, JSON
  report.py      overlay PNG + HTML
  api.py         FastAPI upload → report
  cli.py         analyze / evaluate
tests/           synthetic phantom tests
```

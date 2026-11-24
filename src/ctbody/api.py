"""FastAPI service: upload a CT, get a body-composition report.

Run:  uvicorn ctbody.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .pipeline import analyze

DATA = Path(os.environ.get("CTBODY_DATA", "cases"))
DEVICE = os.environ.get("CTBODY_DEVICE", "gpu")
DATA.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ctbody: L3 body composition (research use only)")

UPLOAD_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ctbody</title>
<style>body{font-family:system-ui;max-width:520px;margin:3rem auto}label{display:block;margin:.8rem 0}</style>
</head><body><h1>L3 body composition</h1>
<form action="/analyze" method="post" enctype="multipart/form-data">
<label>CT (.nii.gz or .zip of DICOM) <input type="file" name="ct" required></label>
<label>Height (cm) <input type="number" step="0.1" name="height_cm" required></label>
<label>Weight (kg, optional) <input type="number" step="0.1" name="weight_kg"></label>
<label>Sex <select name="sex"><option>male</option><option>female</option></select></label>
<input type="hidden" name="redirect" value="1">
<button>Analyze</button></form>
<p><small>Research use only. Do not upload identifiable patient data.</small></p></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UPLOAD_PAGE


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/analyze")
def analyze_endpoint(
    ct: UploadFile = File(...),
    height_cm: float = Form(...),
    sex: Literal["male", "female"] = Form(...),
    weight_kg: Optional[float] = Form(None),
    redirect: Optional[str] = Form(None),
):
    name = ct.filename or ""
    if not (name.endswith(".nii.gz") or name.endswith(".nii") or name.endswith(".zip")):
        raise HTTPException(400, "upload a .nii.gz, .nii or .zip of a DICOM series")

    case_id = uuid.uuid4().hex[:12]
    case_dir = DATA / case_id
    case_dir.mkdir()
    src = case_dir / ("input.zip" if name.endswith(".zip") else
                      "input.nii.gz" if name.endswith(".gz") else "input.nii")
    with src.open("wb") as f:
        shutil.copyfileobj(ct.file, f)

    # Sync def endpoint -> FastAPI runs it in a threadpool. Fine for a demo;
    # production would queue jobs (one GPU worker) and poll for status.
    try:
        result = analyze(src, case_dir, height_cm, sex, weight_kg, device=DEVICE)
    except ValueError as exc:  # e.g. L3 not in field of view
        raise HTTPException(422, str(exc))

    if redirect:
        return HTMLResponse(f'<meta http-equiv="refresh" content="0;url=/cases/{case_id}/report.html">')
    return {"case_id": case_id, "report": f"/cases/{case_id}/report.html", "result": result}


@app.get("/cases/{case_id}/{filename}")
def case_file(case_id: str, filename: str):
    if filename not in {"report.html", "l3_overlay.png", "result.json"} or not case_id.isalnum():
        raise HTTPException(404)
    path = DATA / case_id / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import pixel_baseline_predict, toy_predict, vlm_predict_placeholder
from src.guardrails import apply_safety_guardrails

app = FastAPI(title="RX Analyzer")

_DIR    = Path(__file__).parent
_STATIC = _DIR / "static"

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/icon.png")
def icon():
    p = _DIR / "icon.png"
    return FileResponse(p) if p.exists() else FileResponse(_STATIC / "icon.png")


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("pixel_baseline"),):
    suffix = Path(file.filename or "img.png").suffix or ".png"
    data   = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        if model == "pixel_baseline":
            result = apply_safety_guardrails(pixel_baseline_predict(tmp_path))

        elif model == "vlm": # call model MedGemma
            result = apply_safety_guardrails(vlm_predict_placeholder(tmp_path, mode="baseline"))
        elif model == "vlm_improved":
            result = apply_safety_guardrails(vlm_predict_placeholder(tmp_path, mode="improved"))
        else:
            result = apply_safety_guardrails(toy_predict(tmp_path, mode=model))
    except Exception as e:
        result = {
            "predicted_class": "error",
            "confidence": 0.0,
            "visual_evidence": ["prediction failed"],
            "justification": str(e),
            "limitations": ["model error"],
            "warning": "Prediction failed. Check model setup.",
        }
    finally:
        tmp_path.unlink(missing_ok=True)

    return result


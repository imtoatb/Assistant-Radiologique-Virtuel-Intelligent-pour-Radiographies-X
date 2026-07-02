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
from src.image_validator import validate_image
from src.anonymizer import anonymize_to_path

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
    model: str = Form("pixel_baseline"),
    prompt_version: int = Form(0),
):
    suffix = Path(file.filename or "img.png").suffix or ".png"
    data   = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    # ── Validation : est-ce bien une radio ? ────────────────────────────────
    validation = validate_image(tmp_path)
    if not validation["is_xray"]:
        tmp_path.unlink(missing_ok=True)
        return {
            "predicted_class": "rejected",
            "confidence": 0.0,
            "visual_evidence": [],
            "justification": validation["reason"],
            "limitations": ["image rejetée avant analyse"],
            "warning": "L'image soumise ne semble pas être une radiographie thoracique.",
            "is_xray": False,
        }

    # ── Anonymisation : suppression des métadonnées ──────────────────────────
    clean_path = anonymize_to_path(tmp_path)
    tmp_path.unlink(missing_ok=True)

    try:
        if model == "pixel_baseline":
            result = apply_safety_guardrails(pixel_baseline_predict(clean_path))
        elif model == "vlm":
            result = apply_safety_guardrails(vlm_predict_placeholder(clean_path, mode="baseline", version=prompt_version))
        elif model == "vlm_improved":
            result = apply_safety_guardrails(vlm_predict_placeholder(clean_path, mode="improved", version=prompt_version))
        else:
            result = apply_safety_guardrails(toy_predict(clean_path, mode=model))

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
        clean_path.unlink(missing_ok=True)

    return result
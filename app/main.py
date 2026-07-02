from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import pixel_baseline_predict, toy_predict, vlm_predict_placeholder
from src.guardrails import apply_safety_guardrails
from src.image_validator import validate_image
from src.anonymizer import anonymize_to_path
from src.database import DEFAULT_DB, get_runs, insert_case, insert_prompt, insert_run

app = FastAPI(title="Pulmonar")

_DIR    = Path(__file__).parent
_STATIC = _DIR / "static"
_DATA   = ROOT / "data"
_UPLOADS = _DATA / "uploads"
_UPLOADS.mkdir(parents=True, exist_ok=True)

# même fichier que scripts/run_evaluation.py, pour retrouver toutes les évaluations (CLI + web) au même endroit
_LOGS_DIR = _DATA / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(_LOGS_DIR / "run_evaluation.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


def _log_run(image_path: Path, model: str, prompt_version: int, result: dict) -> None:
    """Persiste l'image (anonymisée), logge le run en base pour l'historique web, et trace l'évaluation dans run_evaluation.log."""
    try:
        dest = _UPLOADS / f"{uuid.uuid4().hex}{image_path.suffix}"
        shutil.copy(image_path, dest)
        case_id = insert_case(DEFAULT_DB, image_path=str(dest), source="upload", split="upload")
        prompt_id = insert_prompt(
            DEFAULT_DB,
            prompt_name=model,
            prompt_version=str(result.get("prompt_version", prompt_version)),
            prompt_text="",
        )
        insert_run(DEFAULT_DB, case_id, str(dest), result, prompt_id)
        logging.info(
            f"[web] {case_id} — {result.get('predicted_class')} "
            f"({result.get('confidence', 0.0):.2f}) latency={result.get('latency_ms', 0)}ms"
        )
    except Exception as e:
        print(f"[history] failed to log run: {e}")


def _resolve_run_image(image_path: str) -> Path | None:
    """Retrouve le fichier image d'un run sur disque, même si le chemin stocké
    en base vient d'une autre machine (on retombe sur le dossier data/ du repo)."""
    if not image_path:
        return None
    raw = Path(image_path.replace("\\", "/"))
    candidates = [raw, ROOT / raw]
    if "data" in raw.parts:
        candidates.append(ROOT.joinpath(*raw.parts[raw.parts.index("data"):]))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved.is_relative_to(_DATA):
            return resolved
    return None


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

    #Validation
    validation = validate_image(tmp_path)
    if not validation["is_xray"]:
        result = {
            "predicted_class": "rejected",
            "model_name": "validation",
            "confidence": 0.0,
            "visual_evidence": [],
            "justification": validation["reason"],
            "limitations": ["image rejetée avant analyse"],
            "warning": "L'image soumise ne semble pas être une radiographie thoracique.",
            "is_xray": False,
        }
        # on anonymise avant de logger, même une image rejetée peut contenir des métadonnées sensibles
        try:
            rejected_path = anonymize_to_path(tmp_path)
        except Exception:
            rejected_path = tmp_path
        _log_run(rejected_path, model, prompt_version, result)
        if rejected_path != tmp_path:
            rejected_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)
        return result

    #suppression des métadonnées
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

        _log_run(clean_path, model, prompt_version, result)

    except Exception as e:
        result = {
            "predicted_class": "error",
            "model_name": model,
            "confidence": 0.0,
            "visual_evidence": ["prediction failed"],
            "justification": str(e),
            "limitations": ["model error"],
            "warning": "Prediction failed. Check model setup.",
        }
        _log_run(clean_path, model, prompt_version, result)
    finally:
        clean_path.unlink(missing_ok=True)

    return result


# Endpoint pour lister les runs déjà loggés en base (historique)
@app.get("/runs")
def list_runs():
    runs = []
    for r in get_runs():
        try:
            prediction = json.loads(r["prediction_json"]) if r["prediction_json"] else {}
        except json.JSONDecodeError:
            prediction = {}
        runs.append({
            "id": r["id"],
            "case_id": r["case_id"],
            "model_name": r["model_name"],
            "predicted_class": r["predicted_class"],
            "confidence": r["confidence"],
            "created_at": r["created_at"],
            "prediction": prediction,
            "has_image": _resolve_run_image(r["image_path"] or "") is not None,
        })
    return runs


# Endpoint pour récupérer l'image associée à un run (rejoue l'affichage à l'identique)
@app.get("/runs/{run_id}/image")
def run_image(run_id: int):
    row = next((r for r in get_runs() if r["id"] == run_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    resolved = _resolve_run_image(row["image_path"] or "")
    if resolved is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(resolved)

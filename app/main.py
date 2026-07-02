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
from src.guardrails import WARNING_TEXT, apply_safety_guardrails

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

# Endpoint pour recevoir une image et renvoyer la prédiction du modèle
@app.post("/predict")
async def predict(
    file: UploadFile = File(...), # le fichier image choisi par l'utilisateur
    model: str = Form("pixel_baseline"), # le modèle choisi par l'utilisateur (pixel_baseline, vlm, vlm_improved, toy)
    prompt_version: int = Form(0),): 
    suffix = Path(file.filename or "img.png").suffix or ".png" # le suffixe du fichier (extension) pour créer un fichier temporaire
    data   = await file.read() # lit le contenu du fichier image envoyé par l'utilisateur

    # Crée un fichier temporaire pour stocker l'image et le passer au modèle
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        # appel le modèle pixel_baseline (modèle de référence)
        if model == "pixel_baseline": 
            result = apply_safety_guardrails(pixel_baseline_predict(tmp_path))

        # appel le modèle MedGemma avec le prompt de base (baseline)
        elif model == "vlm":
            result = apply_safety_guardrails(vlm_predict_placeholder(tmp_path, mode="baseline", version=prompt_version))

        # appel le modèle MedGemma avec le prompt amélioré (improved)
        elif model == "vlm_improved":
            result = apply_safety_guardrails(vlm_predict_placeholder(tmp_path, mode="improved", version=prompt_version))
               
        # appel le modèle de test toy_predict
        else: 
            result = apply_safety_guardrails(toy_predict(tmp_path, mode=model))

    except Exception as e:
        result = apply_safety_guardrails({
            "image_quality": "unknown",
            "predicted_class": "uncertain",
            "confidence": 0.0,
            "visual_evidence": ["prediction failed"],
            "justification": str(e),
            "limitations": ["model error"],
            "warning": WARNING_TEXT,
        })
    finally:
        tmp_path.unlink(missing_ok=True) # supprime le fichier temporaire après l'inférence

    return result


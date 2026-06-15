from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any

from PIL import Image, ImageStat

from .pixel_model import extract_features, load_model, model_exists
from .preprocessing import basic_quality_flag

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."


def _filename_signal(image_path: str | Path) -> str:
    name = Path(image_path).name.lower()
    tokens = set(re.split(r"[^a-z0-9]+", name))
    if "suspected_opacity" in name or {"suspected", "opacity"} <= tokens:
        return "suspected_opacity"
    if "normal" in tokens:
        return "normal"
    return "uncertain"


def _image_signal_strength(image_path: str | Path) -> float:
    try:
        image = Image.open(image_path).convert("L").resize((128, 128))
    except OSError:
        return 0.5

    stat = ImageStat.Stat(image)
    mean = stat.mean[0] / 255.0
    contrast = stat.stddev[0] / 128.0
    return max(0.0, min(1.0, (mean * 0.35) + (contrast * 0.65)))


def _confidence(signal: str, image_path: str | Path, mode: str) -> float:
    strength = _image_signal_strength(image_path)
    mode_penalty = 0.04 if mode == "improved" else 0.0
    if signal == "suspected_opacity":
        return 0.58 + (0.32 * strength) - mode_penalty
    if signal == "normal":
        return 0.56 + (0.28 * (1.0 - strength)) - mode_penalty
    return 0.45 + (0.12 * strength)


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Deterministic toy predictor used to validate the repo pipeline.

    It reads synthetic labels from filenames. This is not medical inference.
    """
    start = time.perf_counter()
    signal = _filename_signal(image_path)
    quality = basic_quality_flag(image_path)

    if signal == "suspected_opacity":
        pred = "suspected_opacity"
        conf = _confidence(signal, image_path, mode)
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = "The synthetic image contains a localized brighter region compatible with the toy opacity class. This is a pipeline validation result, not a medical interpretation."
    elif signal == "normal":
        pred = "normal"
        conf = _confidence(signal, image_path, mode)
        evidence = ["no synthetic opacity marker detected"]
        justification = "The synthetic image does not contain the opacity marker used by the toy generator. This conclusion is limited to the synthetic validation setting."
    else:
        pred = "uncertain"
        conf = _confidence(signal, image_path, mode)
        evidence = ["limited synthetic image quality"]
        justification = "The image is treated as limited quality in the toy catalog. The safe output is uncertainty rather than a forced class."

    # Improved mode is more conservative.
    if mode == "improved" and quality != "good":
        pred = "uncertain"
        conf = min(conf, 0.55)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": pred,
        "confidence": round(float(conf), 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["synthetic toy image", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
        "model_name": f"toy-rule-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def pixel_baseline_predict(image_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    quality = basic_quality_flag(image_path)

    if not model_exists():
        pred = "uncertain"
        conf = 0.0
        evidence = ["pixel baseline model not trained"]
        justification = "Run scripts/train_pixel_baseline.py before using the pixel baseline."
    else:
        model = load_model()
        features = extract_features(image_path).reshape(1, -1)
        pred = str(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        conf = float(max(probabilities))
        evidence = ["prediction based on image intensity, contrast, histogram and edge features"]
        justification = "A lightweight logistic-regression baseline predicted from image pixels, not from the filename."

        if conf < 0.6:
            pred = "uncertain"

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": pred,
        "confidence": round(conf, 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["simple pixel baseline", "small dataset", "not a validated medical model"],
        "warning": WARNING,
        "model_name": "pixel-logistic-baseline",
        "prompt_version": "pixel_baseline_v1",
        "latency_ms": latency_ms,
    }


def vlm_predict_placeholder(image_path: str | Path, prompt: str) -> dict[str, Any]:
    """Placeholder for a Hugging Face / MedGemma / Gemma 4 VLM call.

    Students should keep the same output schema as toy_predict.
    """
    return toy_predict(image_path, mode="baseline")

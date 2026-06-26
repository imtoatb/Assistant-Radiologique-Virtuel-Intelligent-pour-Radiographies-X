from __future__ import annotations

from pathlib import Path
import re
import time
import json
from typing import Any
from functools import lru_cache

from PIL import Image, ImageStat

from .pixel_model import extract_features, load_model, model_exists
from .preprocessing import basic_quality_flag, load_image

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch

import gc


WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

@lru_cache(maxsize=1)
def _load_vlm():
    """ Load the MedGemma-4b-it model and processor from local files only, with 4-bit quantization to limit memory usage."""

    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True) # avoid downloading from Hugging Face Hub
    quantization_config = BitsAndBytesConfig(load_in_4bit=True) # limit memory usage for the model MedGemma-4b-it
    
    # Load the model with device_map="cuda:0" to use GPU if available, and limit max_memory to 20GB
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda:0",
        local_files_only=True,
        max_memory={0: "4GB"}
    )    
    # check if the model is on GPU or CPU
    print(f"Modèle chargé sur : {next(model.parameters()).device}")

    return processor, model

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
    start = time.perf_counter()
    signal = _filename_signal(image_path)
    quality = basic_quality_flag(image_path)

    if signal == "suspected_opacity":
        pred = "suspected_opacity"
        conf = _confidence(signal, image_path, mode)
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = "Pipeline validation result, not a medical interpretation."
    elif signal == "normal":
        pred = "normal"
        conf = _confidence(signal, image_path, mode)
        evidence = ["no synthetic opacity marker detected"]
        justification = "The synthetic image does not contain the opacity marker."
    else:
        pred = "uncertain"
        conf = _confidence(signal, image_path, mode)
        evidence = ["limited synthetic image quality"]
        justification = "Safe fallback to uncertainty."

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
        pred, conf = "uncertain", 0.0
        evidence = ["pixel baseline model not trained"]
        justification = "Run scripts/train_pixel_baseline.py before using the pixel baseline."
    else:
        model = load_model()
        features = extract_features(image_path).reshape(1, -1)
        pred = str(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        conf = float(max(probabilities))
        evidence = ["prediction based on image intensity, contrast, histogram and edge features"]
        justification = "Lightweight logistic-regression baseline from image pixels."
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


def vlm_predict_placeholder(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    start = time.perf_counter()
    quality = basic_quality_flag(image_path)
    # print(f"=== IMAGE QUALITY === {quality} for {image_path}")
    image = load_image(image_path)

    # Lecture du prompt depuis le fichier
    prompt_file = Path(__file__).resolve().parents[1] / "prompts" / f"{mode}_prompt.txt"
    system_prompt = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else ""

    processor, model = _load_vlm()

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": system_prompt}
        ]}
    ]

    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        images=image,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=300)
    response = processor.decode(outputs[0], skip_special_tokens=True)

    # to clear GPU memory after generation 
    del inputs, outputs # Supprimer la référence à inputs et outputs pour libérer la mémoire
    gc.collect()
    torch.cuda.empty_cache()

    # print("=== RESPONSE BRUTE ===")
    # print(response)
    # print("=== FIN RESPONSE ===")

    # Extraire la partie après "model" et nettoyer les balises markdown
    response_after_model = response.split("model\n")[-1] if "model\n" in response else response
    response_clean = re.sub(r'```json\s*|\s*```', '', response_after_model).strip()
    match = re.search(r'\{.*\}', response_clean, re.DOTALL)

    try:
        parsed = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        parsed = {}

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": parsed.get("predicted_class", "uncertain"),
        "confidence": float(parsed.get("confidence", 0.0)),
        "visual_evidence": parsed.get("visual_evidence", []),
        "justification": parsed.get("justification", ""),
        "limitations": parsed.get("limitations", []),
        "warning": WARNING,
        "model_name": "medgemma-4b-it",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }
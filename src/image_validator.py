"""
image_validator.py
==================
Validation des images entrantes en deux niveaux en cascade :

Niveau 1 — Heuristiques (rapide, CPU, sans modèle)
    - Format supporté
    - Taille minimale
    - Image en niveaux de gris (critère principal pour une radio)
    - Ratio largeur/hauteur cohérent avec une radio thoracique
    - Pas trop lumineuse ni trop sombre (contraste minimal)

Niveau 2 — Classifier ML léger (si les heuristiques ne sont pas concluantes)
    - Logistic regression sur features pixel (même logique que pixel_baseline)
    - Entraîné sur des exemples radio vs non-radio
    - Si le modèle n'est pas encore entraîné, on reste sur le résultat heuristique

Usage depuis l'API :
    from src.image_validator import validate_image
    result = validate_image("path/to/image.jpg")
    if not result["is_xray"]:
        return {"error": result["reason"]}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageStat, ImageFilter

# ── Constantes ──────────────────────────────────────────────────────────────

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

# Taille minimale acceptable (en pixels)
MIN_WIDTH  = 128
MIN_HEIGHT = 128

# Une radio thoracique a un ratio proche de 1 (légèrement plus large que haute)
# On accepte entre 0.6 et 2.0 pour être tolérant
RATIO_MIN = 0.6
RATIO_MAX = 2.0

# Seuils de niveaux de gris :
# Une radio est quasi exclusivement en gris → la saturation moyenne doit être très faible
# On travaille en RGB et on calcule l'écart max entre canaux R, G, B par pixel
# Si l'image est vraiment en gris, cet écart est proche de 0
GRAYSCALE_THRESHOLD = 15.0   # écart max moyen entre canaux (sur 255)

# Contraste minimal : une radio n'est pas toute noire ni toute blanche
CONTRAST_MIN = 20.0    # stddev minimale sur les niveaux de gris (sur 255)
BRIGHTNESS_MIN = 10.0  # moyenne minimale (pas trop sombre)
BRIGHTNESS_MAX = 245.0 # moyenne maximale (pas surexposée)

# Chemin du modèle ML de validation (entraîné séparément)
VALIDATOR_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xray_validator.joblib"


# ── Niveau 1 : Heuristiques ──────────────────────────────────────────────────

def _check_format(path: Path) -> tuple[bool, str]:
    """Vérifie que l'extension est supportée."""
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return False, f"Format non supporté : {path.suffix}. Formats acceptés : {', '.join(ALLOWED_SUFFIXES)}"
    return True, ""


def _check_size(image: Image.Image) -> tuple[bool, str]:
    """Vérifie que l'image est assez grande pour être analysée."""
    w, h = image.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False, f"Image trop petite ({w}x{h}px). Minimum requis : {MIN_WIDTH}x{MIN_HEIGHT}px"
    return True, ""


def _check_ratio(image: Image.Image) -> tuple[bool, str]:
    """Vérifie que le ratio largeur/hauteur est cohérent avec une radio thoracique."""
    w, h = image.size
    ratio = w / h
    if not (RATIO_MIN <= ratio <= RATIO_MAX):
        return False, f"Ratio image inhabituel ({ratio:.2f}). Une radio thoracique a un ratio entre {RATIO_MIN} et {RATIO_MAX}"
    return True, ""


def _check_grayscale(image: Image.Image) -> tuple[bool, str, float]:
    """
    Vérifie que l'image est en niveaux de gris.

    On convertit en RGB et on mesure l'écart moyen entre les canaux R, G, B.
    Une vraie image en gris a R≈G≈B sur chaque pixel → écart proche de 0.
    Une photo couleur a des écarts importants.

    Retourne aussi le score pour les heuristiques composites.
    """
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Écart max entre les 3 canaux, moyenné sur tous les pixels
    max_channel_diff = np.mean(np.max(np.stack([
        np.abs(r - g),
        np.abs(r - b),
        np.abs(g - b)
    ], axis=0), axis=0))

    is_gray = max_channel_diff < GRAYSCALE_THRESHOLD
    if not is_gray:
        reason = (
            f"L'image semble être en couleur (écart inter-canaux moyen : {max_channel_diff:.1f}/255). "
            "Une radiographie thoracique est en niveaux de gris."
        )
        return False, reason, max_channel_diff
    return True, "", max_channel_diff


def _check_contrast(image: Image.Image) -> tuple[bool, str]:
    """
    Vérifie que l'image a un contraste minimal et une luminosité raisonnable.
    Une image toute noire ou toute blanche n'est pas une radio valide.
    """
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    mean   = stat.mean[0]
    stddev = stat.stddev[0]

    if mean < BRIGHTNESS_MIN:
        return False, f"Image trop sombre (luminosité moyenne : {mean:.0f}/255). L'image pourrait être corrompue."
    if mean > BRIGHTNESS_MAX:
        return False, f"Image trop lumineuse (luminosité moyenne : {mean:.0f}/255). L'image pourrait être surexposée."
    if stddev < CONTRAST_MIN:
        return False, f"Contraste insuffisant (stddev : {stddev:.0f}/255). L'image pourrait être vide ou corrompue."
    return True, ""


def _heuristic_score(image: Image.Image) -> dict[str, Any]:
    """
    Calcule un score heuristique composite sur [0, 1] représentant la probabilité
    que l'image soit une radio thoracique.

    Utilisé pour décider si on passe au niveau ML ou si on est suffisamment sûr.
    """
    gray = image.convert("L").resize((256, 256))
    arr  = np.array(gray, dtype=np.float32) / 255.0

    stat   = ImageStat.Stat(gray)
    mean   = stat.mean[0] / 255.0
    stddev = stat.stddev[0] / 255.0

    # Feature 1 : contraste (une radio a une stddev typiquement entre 0.15 et 0.45)
    contrast_score = 1.0 if 0.10 <= stddev <= 0.50 else max(0.0, 1.0 - abs(stddev - 0.30) * 4)

    # Feature 2 : luminosité (une radio a une moyenne typiquement entre 0.15 et 0.60)
    brightness_score = 1.0 if 0.10 <= mean <= 0.65 else max(0.0, 1.0 - abs(mean - 0.35) * 3)

    # Feature 3 : homogénéité des bords (une radio a des structures douces, pas des bords nets partout)
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    edge_density = float(edges.mean())
    edge_score = 1.0 if edge_density < 0.15 else max(0.0, 1.0 - (edge_density - 0.10) * 5)

    # Feature 4 : distribution des intensités — une radio a un histogramme étalé
    hist, _ = np.histogram(arr, bins=16, range=(0.0, 1.0), density=True)
    hist_nonzero = np.sum(hist > 0.5)  # nombre de bins significatifs
    hist_score = min(1.0, hist_nonzero / 10.0)

    # Score composite pondéré
    score = (
        0.30 * contrast_score +
        0.25 * brightness_score +
        0.25 * edge_score +
        0.20 * hist_score
    )

    return {
        "score": round(float(score), 3),
        "contrast_score": round(contrast_score, 3),
        "brightness_score": round(brightness_score, 3),
        "edge_score": round(edge_score, 3),
        "hist_score": round(hist_score, 3),
    }


# ── Niveau 2 : Classifier ML ─────────────────────────────────────────────────

def _ml_validator_exists() -> bool:
    return VALIDATOR_MODEL_PATH.exists()


def _extract_validator_features(image: Image.Image) -> np.ndarray:
    """
    Features pour le classifier de validation (radio vs non-radio).
    Similaire à pixel_baseline mais orienté détection de type d'image.
    """
    gray = image.convert("L").resize((128, 128))
    arr  = np.array(gray, dtype=np.float32) / 255.0
    stat = ImageStat.Stat(gray)

    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=32, range=(0.0, 1.0), density=True)

    # Stats de base
    features = [
        float(arr.mean()),
        float(arr.std()),
        float(np.percentile(arr, 5)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
        float(np.percentile(arr, 95)),
        float(edges.mean()),
        float(edges.std()),
        float(stat.extrema[0][0] / 255.0),
        float(stat.extrema[0][1] / 255.0),
    ]

    # Pour les images RGB : mesure de la saturation (couleur vs gris)
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    channel_diff = float(np.mean(np.max(np.stack([
        np.abs(r - g), np.abs(r - b), np.abs(g - b)
    ], axis=0), axis=0)) / 255.0)
    features.append(channel_diff)

    return np.asarray([*features, *hist.tolist()], dtype=np.float32)


_cached_validator_model = None

def _ml_predict(image: Image.Image) -> tuple[bool, float]:
    global _cached_validator_model
    if not _ml_validator_exists():
        return None, 0.0
    try:
        import joblib
        if _cached_validator_model is None:
            _cached_validator_model = joblib.load(VALIDATOR_MODEL_PATH)
        model = _cached_validator_model
        features = _extract_validator_features(image).reshape(1, -1)
        pred = str(model.predict(features)[0])
        proba = float(max(model.predict_proba(features)[0]))
        return pred == "xray", proba
    except Exception:
        return None, 0.0

# ── Interface principale ──────────────────────────────────────────────────────

def validate_image(path: str | Path) -> dict[str, Any]:
    """
    Valide qu'une image est bien une radiographie thoracique.

    Pipeline en cascade :
    1. Vérifications de base (format, taille, ratio)          → rejet immédiat si invalide
    2. Vérification niveaux de gris                           → rejet immédiat si couleur franche
    3. Vérification contraste/luminosité                      → rejet si image corrompue
    4. Score heuristique composite                            → si score > 0.7 : accepté
                                                              → si score < 0.3 : rejeté
                                                              → sinon : on passe au ML
    5. Classifier ML léger (si heuristiques inconcluses)      → décision finale

    Retourne un dict avec :
        is_xray (bool)       : True si l'image est validée comme radio
        confidence (float)   : niveau de confiance [0, 1]
        reason (str)         : explication lisible si rejeté (vide si accepté)
        method (str)         : "heuristic" ou "ml" selon ce qui a tranché
        details (dict)       : scores intermédiaires pour debug/log
    """
    path = Path(path)

    #Format
    ok, reason = _check_format(path)
    if not ok:
        return _reject(reason, method="heuristic", details={"step": "format"})

    #Chargement image
    try:
        image = Image.open(path)
        image.verify()                                      # détecte les fichiers corrompus
        image = Image.open(path)                            # rouvrir après verify() (verify() consomme le flux)
    except Exception as e:
        return _reject(f"Impossible d'ouvrir l'image : {e}", method = "heuristic", details = {"step": "load"})

    #Taille
    ok, reason = _check_size(image)
    if not ok:
        return _reject(reason, method = "heuristic", details = {"step": "size"})

    #Ratio
    ok, reason = _check_ratio(image)
    if not ok:
        return _reject(reason, method="heuristic", details = {"step": "ratio", "size": image.size})

    #Niveaux de gris
    ok, reason, channel_diff = _check_grayscale(image)
    if not ok:
        return _reject(reason, method = "heuristic", details = {"step": "grayscale", "channel_diff": round(channel_diff, 2)})

    #Contraste /luminosité
    ok, reason = _check_contrast(image)
    if not ok:
        return _reject(reason, method = "heuristic", details = {"step": "contrast"})

    #Score heuristique composite
    scores = _heuristic_score(image)
    h_score = scores["score"]

    if h_score >= 0.55:
        # Score suffisamment élevé = on accepte sans ML
        return {
            "is_xray": True,
            "confidence": h_score,
            "reason": "",
            "method": "heuristic",
            "details": scores,
        }

    if h_score < 0.30:
        # Score trop bas et pas de ML disponible ou score très faible
        if not _ml_validator_exists():
            return _reject(
                f"Pas une radiographie thoracique (score heuristique : {h_score:.2f}).",
                method="heuristic",
                details=scores,
            )

    #Classifier ML (zone d'incertitude ou score faible)
    ml_result, ml_confidence = _ml_predict(image)

    if ml_result is None:
        # Modèle ML indisponible = on se fie aux heuristiques
        if h_score >= 0.50:
            return {
                "is_xray": True,
                "confidence": h_score,
                "reason": "",
                "method": "heuristic_fallback",
                "details": scores,
            }
        return _reject(
            f"L'image ne ressemble pas à une radiographie thoracique (score : {h_score:.2f}). "
            "Le modèle de validation ML n'est pas encore disponible.",
            method="heuristic_fallback",
            details=scores,
        )

    if ml_result:
        return {
            "is_xray": True,
            "confidence": round(ml_confidence, 3),
            "reason": "",
            "method": "ml",
            "details": {**scores, "ml_confidence": ml_confidence},
        }
    else:
        return _reject(
            f"L'image ne semble pas être une radiographie thoracique (confiance ML : {ml_confidence:.0%}).",
            method="ml",
            details={**scores, "ml_confidence": ml_confidence},
        )


def _reject(reason: str, method: str, details: dict) -> dict[str, Any]:
    return {
        "is_xray": False,
        "confidence": 0.0,
        "reason": reason,
        "method": method,
        "details": details,
    }
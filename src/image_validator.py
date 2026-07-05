"""
Validation Heuristiques
    -Format supporté
    -Taille minimale
    -Image en niveaux de gris (critère principal pour une radio)
    -Ratio largeur/hauteur cohérent avec une radio thoracique
    -Pas trop lumineuse ni trop sombre (contraste minimal)

Sinon Classifier ML léger
    -Logistic regression sur features pixel (même logique que pixel_baseline)
    -Entraîné sur des exemples radio vs non-radio
    -Si le modèle n'est pas encore entraîné, on reste sur le résultat heuristique
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageStat, ImageFilter



ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

#Taille minimale acceptable
MIN_WIDTH  = 128
MIN_HEIGHT = 128

#Une radio thoracique a un ratio proche de 1 (légèrement plus large que haute)
#On accepte entre 0.6 et 2.0 pour être tolérant
RATIO_MIN = 0.6
RATIO_MAX = 2.0

#Seuil de niveaux de gris : la saturation moyenne doit être très faible
#On travaille en RGB et on calcule l'écart max entre canaux R, G, B par pixel
#Si l'image est vraiment en gris, cet écart est proche de 0
GRAYSCALE_THRESHOLD = 10.0                                                  # écart max moyen entre canaux (sur 255)

#une radio n'est pas toute noire ni toute blanche
CONTRAST_MIN = 20.0                                                         #stddev minimale sur les niveaux de gris (sur 255)
BRIGHTNESS_MIN = 10.0                                                       #moyenne minimale (pas trop sombre)
BRIGHTNESS_MAX = 245.0                                                      #moyenne maximale (pas surexposée)

# Chemin du modèle ML de validation (entraîné séparément)
VALIDATOR_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xray_validator.joblib"

REFERENCE_THORAX_PATH = Path(__file__).resolve().parents[1] / "data" / "reference_thorax" / "image.png"




#Heuristiques

def _check_format(path: Path) -> tuple[bool, str]:                          #doit être bmp, png, jpeg ou jpg
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return False, f"Format non supporté : {path.suffix}. Formats acceptés : {', '.join(ALLOWED_SUFFIXES)}"
    return True, ""


def _check_size(image: Image.Image) -> tuple[bool, str]:                    #Taille de l'image
    w, h = image.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False, f"Image trop petite ({w}x{h}px). Minimum requis : {MIN_WIDTH}x{MIN_HEIGHT}px"
    return True, ""


def _check_ratio(image: Image.Image) -> tuple[bool, str]:                   #Ratio hauteur largeur
    w, h = image.size
    ratio = w / h
    if not (RATIO_MIN <= ratio <= RATIO_MAX):
        return False, f"Ratio image inhabituel ({ratio:.2f}). Une radio thoracique a un ratio entre {RATIO_MIN} et {RATIO_MAX}"
    return True, ""


def _check_grayscale(image: Image.Image) -> tuple[bool, str, float]:        #Niveau de gris
    """
    On convertit en RGB et on mesure l'écart moyen entre les canaux R, G, B.
    Une vraie image en gris a R≈G≈B sur chaque pixel → écart proche de 0.
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
            f"Couleur (écart inter-canaux moyen : {max_channel_diff:.1f}/255)"
            "Pas une radio"
        )
        return False, reason, max_channel_diff
    return True, "", max_channel_diff

 
def _check_contrast(image: Image.Image) -> tuple[bool, str]:        #Contraste et Luminosité (pas nore ni blanche)
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    mean   = stat.mean[0]
    stddev = stat.stddev[0]

    if mean < BRIGHTNESS_MIN:
        return False, f"Image trop sombre (luminosité moyenne : {mean:.0f}/255)"
    if mean > BRIGHTNESS_MAX:
        return False, f"Image trop lumineuse (luminosité moyenne : {mean:.0f}/255)"
    if stddev < CONTRAST_MIN:
        return False, f"Contraste insuffisant (stddev : {stddev:.0f}/255)"
    return True, ""


def _heuristic_score(image: Image.Image) -> dict[str, Any]:
    """
    Calcule un score heuristique composite sur [0, 1] représentant la probabilité
    que l'image soit une radio thoracique.
    """
    gray = image.convert("L").resize((256, 256))
    arr  = np.array(gray, dtype=np.float32) / 255.0

    stat   = ImageStat.Stat(gray)
    mean   = stat.mean[0] / 255.0
    stddev = stat.stddev[0] / 255.0

    #contraste (une radio a une stddev typiquement entre 0.15 et 0.45)
    contrast_score = 1.0 if 0.10 <= stddev <= 0.50 else max(0.0, 1.0 - abs(stddev - 0.30) * 4)

    #luminosité (une radio a une moyenne typiquement entre 0.15 et 0.60)
    brightness_score = 1.0 if 0.10 <= mean <= 0.65 else max(0.0, 1.0 - abs(mean - 0.35) * 3)

    # homogénéité des bords (une radio a des structures douces, pas des bords nets partout)
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    edge_density = float(edges.mean())
    edge_score = 1.0 if edge_density < 0.15 else max(0.0, 1.0 - (edge_density - 0.10) * 5)

    #distribution des intensités — une radio a un histogramme étalé
    hist, _ = np.histogram(arr, bins=16, range=(0.0, 1.0), density=True)
    hist_nonzero = np.sum(hist > 0.5)  # nombre de bins significatifs
    hist_score = min(1.0, hist_nonzero / 10.0)

    #Score composite pondéré
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


#Classifier ML

def _ml_validator_exists() -> bool:
    return VALIDATOR_MODEL_PATH.exists()


def _extract_validator_features(image: Image.Image) -> np.ndarray:
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

    # Pour les images RGB, mesure de la saturation (couleur vs gris)
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
    


_reference_hist: np.ndarray | None = None


def _get_reference_hist() -> np.ndarray | None:                                 #met en cache la rado de référence
    global _reference_hist
    if _reference_hist is not None:
        return _reference_hist
    if not REFERENCE_THORAX_PATH.exists():
        return None
    try:
        ref = Image.open(REFERENCE_THORAX_PATH).convert("L").resize((128, 128))
        arr = np.array(ref, dtype=np.float32) / 255.0
        hist, _ = np.histogram(arr, bins=64, range=(0.0, 1.0), density=True)
        _reference_hist = hist
        return _reference_hist
    except Exception:
        return None


def _check_thorax_similarity(image: Image.Image) -> tuple[bool, str, float]:
    """
    Compare l'histogramme de l'image avec celui de la référence thoracique.
    """
    ref_hist = _get_reference_hist()
    if ref_hist is None:
        return True, "", 1.0  # pas de référence = on laisse passer

    gray = image.convert("L").resize((128, 128))
    arr = np.array(gray, dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=64, range=(0.0, 1.0), density=True)

    # Corrélation de Pearson entre les deux histogrammes
    ref_mean = ref_hist.mean()
    img_mean = hist.mean()
    numerator = np.sum((ref_hist - ref_mean) * (hist - img_mean))
    denominator = np.sqrt(np.sum((ref_hist - ref_mean) ** 2) * np.sum((hist - img_mean) ** 2))
    score = float(numerator / denominator) if denominator > 1e-8 else 0.0
    # Normalise de [-1, 1] vers [0, 1]
    score_normalized = (score + 1.0) / 2.0

    if score_normalized < 0.50:
        return False, (
            f"L'image n'est pas une radiographie thoracique "
            f"(similarité : {score_normalized:.0%}). "
        ), score_normalized
    return True, "", score_normalized

#Interface principale

def validate_image(path: str | Path) -> dict[str, Any]:
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
    
    ok, reason, sim_score = _check_thorax_similarity(image)
    if not ok:
        return _reject(reason, method="heuristic", details={"step": "thorax_similarity", "similarity": round(sim_score, 3)})

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
            f"L'image n'estpas une radiographie thoracique (score : {h_score:.2f}) ",
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
            f"Pas une radiographie thoracique (confiance ML : {ml_confidence:.0%}).",
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
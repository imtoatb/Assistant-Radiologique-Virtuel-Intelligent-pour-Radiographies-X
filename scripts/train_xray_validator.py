from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

#Constantes
XRAY_DIR      = ROOT / "data" / "validator_training" / "xray"
NOT_XRAY_DIR  = ROOT / "data" / "validator_training" / "not_xray"
MODEL_PATH    = ROOT / "models" / "xray_validator.joblib"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


#Import des features depuis image_validator
from src.image_validator import _extract_validator_features
from PIL import Image


def load_examples(directory: Path, label: str, max_n: int | None = None) -> tuple[list, list]:
    """Charge les images d'un dossier et extrait leurs features."""
    images = [p for p in directory.iterdir() if p.suffix.lower() in ALLOWED_SUFFIXES]
    if max_n:
        images = images[:max_n]

    X, y = [], []
    for i, img_path in enumerate(images, 1):
        try:
            image = Image.open(img_path)
            features = _extract_validator_features(image)
            X.append(features)
            y.append(label)
            if i % 50 == 0:
                print(f"  {label}: {i}/{len(images)} images chargées")
        except Exception as e:
            print(f"  Ignoré {img_path.name} : {e}")

    return X, y


def generate_synthetic_negatives(n: int = 100) -> tuple[list, list]:
    """
    Génère des images synthétiques colorées comme exemples négatifs.
    """
    from PIL import ImageDraw
    import math

    X, y = [], []
    rng = random.Random(42)

    for i in range(n):
        # Image RGB colorée aléatoire (simulate une photo)
        w, h = rng.randint(200, 800), rng.randint(200, 800)
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)

        # Remplir avec des formes colorées
        for _ in range(rng.randint(5, 20)):
            color = (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))
            x0 = rng.randint(0, w)
            y0 = rng.randint(0, h)
            x1 = rng.randint(x0, min(x0 + 200, w))
            y1 = rng.randint(y0, min(y0 + 200, h))
            draw.rectangle([x0, y0, x1, y1], fill=color)

        features = _extract_validator_features(img)
        X.append(features)
        y.append("not_xray")

    print(f"  {n} images synthétiques générées comme exemples négatifs")
    return X, y


def train(X: np.ndarray, y: list[str]) -> Any:
    """Entraîne et retourne le classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter = 1000, C = 1.0, random_state = 42)),
    ])
    model.fit(X, y)
    return model


def evaluate(model: Any, X: np.ndarray, y: list[str]) -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X)

    print(classification_report(y, y_pred))
    print("Matrice de confusion :")
    print(confusion_matrix(y, y_pred, labels=["xray", "not_xray"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-negative", action="store_true",
                        help="Générer des images synthétiques si pas de données not_xray")
    parser.add_argument("--max-positives", type=int, default=None,
                        help="Nombre max d'images radio à utiliser")
    parser.add_argument("--evaluate", action="store_true",
                        help="Évaluer le modèle après entraînement")
    parser.add_argument("--n-synthetic", type=int, default=200,
                        help="Nombre d'images synthétiques à générer (--auto-negative)")
    args = parser.parse_args()


    #Exemples positifs (radios)
    if not XRAY_DIR.exists() or not any(XRAY_DIR.iterdir()):
        # Utiliser les images du dataset principal comme exemples positifs
        fallback_dirs = [
            ROOT / "data" / "brutes",
            ROOT / "data" / "abnormal_lung_images",
            ROOT / "data" / "lung_images",
        ]
        xray_source = next((d for d in fallback_dirs if d.exists()), None)
        if xray_source is None:
            print("Aucun dossier d'images radio trouvé. Créez data/validator_training/xray/")
            sys.exit(1)
        print(f"Utilisation de {xray_source} comme source d'exemples positifs")
    else:
        xray_source = XRAY_DIR

    print(f"Chargement des exemples positifs depuis {xray_source}...")
    X_pos, y_pos = load_examples(xray_source, "xray", max_n=args.max_positives)
    print(f"  → {len(X_pos)} exemples positifs chargés")

    if not X_pos:
        print("Aucun exemple positif trouvé. Vérifiez le dossier.")
        sys.exit(1)

    #Exemples négatifs
    if NOT_XRAY_DIR.exists() and any(NOT_XRAY_DIR.iterdir()):
        print(f"Chargement des exemples négatifs depuis {NOT_XRAY_DIR}...")
        X_neg, y_neg = load_examples(NOT_XRAY_DIR, "not_xray")
        print(f"  → {len(X_neg)} exemples négatifs chargés")
    elif args.auto_negative:
        print("Génération d'exemples négatifs synthétiques...")
        X_neg, y_neg = generate_synthetic_negatives(n=args.n_synthetic)
    else:
        print(
            "Pas d'exemples négatifs. Utilisez --auto-negative ou créez data/validator_training/not_xray/"
        )
        sys.exit(1)

    #Entraînement
    X = np.array(X_pos + X_neg)
    y = y_pos + y_neg

    print(f"\nEntraînement sur {len(X)} exemples ({len(X_pos)} radios / {len(X_neg)} non-radios)...")
    model = train(X, y)

    #Sauvegarde
    import joblib
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé → {MODEL_PATH}")

    #Évaluation
    if args.evaluate:
        evaluate(model, X, y)
        print("\nNote : ces métriques sont sur les données d'entraînement (pas de test set séparé).")


if __name__ == "__main__":
    main()
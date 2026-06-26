"""
generate_cases_csv.py
Parcourt data/brutes/, extrait le label depuis le nom de fichier,
et génère data/cases.csv pour alimenter la table radios via seed_radios().

Usage : python scripts/generate_cases_csv.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Constants
BRUTES_DIR = Path(__file__).resolve().parents[1] / "data" / "brutes"
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "cases.csv"
SOURCE = "fkarimovv/abnormal-lung"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


# extrait le label depuis le nom de fichier
def extract_label(filename: str) -> str:
    name = filename.lower()
    if "suspected_opacity" in name:
        return "suspected_opacity"
    if "normal" in name:
        return "normal"
    return "uncertain"


# Génère le CSV à partir des images dans data/brutes/
def main() -> None:
    # Récupère toutes les images dans le dossier BRUTES_DIR avec les suffixes autorisés
    images = sorted(
        p for p in BRUTES_DIR.iterdir()
        if p.suffix.lower() in ALLOWED_SUFFIXES
    )

    if not images:
        print(f"Aucune image trouvée dans {BRUTES_DIR}")
        return

    rows = []
    # Parcourt les images et extrait le label pour chaque image
    for i, img in enumerate(images, start=1):
        label = extract_label(img.name)
        case_id = i
        rows.append({
            "case_id": case_id,
            "image_path": f"data/brutes/{img.name}",
            "source": SOURCE,
            "label": label,
            "split": "external",
            "notes": f"Importé depuis {img.name}",
        })
    
    # écrit le CSV avec les colonnes : case_id, image_path, source, label, split, notes
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "image_path", "source", "label", "split", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    # compte le nombre d'images par label pour l'affichage
    normal = sum(1 for r in rows if r["label"] == "normal")
    suspected = sum(1 for r in rows if r["label"] == "suspected_opacity")
    uncertain = sum(1 for r in rows if r["label"] == "uncertain")

    print(f"CSV généré : {OUTPUT_CSV}")
    print(f"Total : {len(rows)} images")
    print(f"normal : {normal}")
    print(f"suspected_opacity : {suspected}")
    print(f"uncertain : {uncertain}")


if __name__ == "__main__":
    main()
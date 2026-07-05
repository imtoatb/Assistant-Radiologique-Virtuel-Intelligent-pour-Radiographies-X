"""
generate_cases_csv.py
Parcourt un dossier d'images, extrait le label depuis le nom de fichier,
et génère un CSV pour alimenter la table cases via seed_cases().

Usage : python scripts/generate_cases_csv.py --brutes-dir data/brutes_kaggle --source fkarimovv/abnormal-lung --output-csv data/cases_kaggle_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


# extrait le label depuis le nom de fichier (fonctionne pour Kaggle et RSNA,
# les deux embarquent le label en toutes lettres dans le nom de fichier)
def extract_label(filename: str) -> str:
    name = filename.lower()
    if "suspected_opacity" in name:
        return "suspected_opacity"
    if "normal" in name:
        return "normal"
    return "uncertain"


# Génère le CSV à partir des images dans brutes_dir
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brutes-dir", type=Path, default=ROOT / "data" / "brutes_kaggle")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "data" / "cases_kaggle_dataset.csv")
    parser.add_argument("--source", default="fkarimovv/abnormal-lung")
    args = parser.parse_args()

    images = sorted(
        p for p in args.brutes_dir.iterdir()
        if p.suffix.lower() in ALLOWED_SUFFIXES
    )

    if not images:
        print(f"Aucune image trouvée dans {args.brutes_dir}")
        return

    rows = []
    for i, img in enumerate(images, start=1):
        label = extract_label(img.name)
        rows.append({
            "case_id": i,
            "image_path": img.relative_to(ROOT).as_posix(),
            "source": args.source,
            "label": label,
            "split": "external",
            "notes": f"Importé depuis {img.name}",
        })

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "image_path", "source", "label", "split", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    normal = sum(1 for r in rows if r["label"] == "normal")
    suspected = sum(1 for r in rows if r["label"] == "suspected_opacity")
    uncertain = sum(1 for r in rows if r["label"] == "uncertain")

    print(f"CSV généré : {args.output_csv}")
    print(f"Total : {len(rows)} images")
    print(f"normal : {normal}")
    print(f"suspected_opacity : {suspected}")
    print(f"uncertain : {uncertain}")


if __name__ == "__main__":
    main()
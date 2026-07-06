from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pixel_model import MODEL_PATH, extract_features


def read_cases(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-path", type=Path, default=ROOT / "data" / "lung_cases.csv")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=ROOT / "eval" / "pixel_baseline_report.json")
    args = parser.parse_args()

    rows = [row for row in read_cases(args.cases_path) if row["label"] in {"normal", "suspected_opacity"}]
    if len(rows) < 10:
        raise RuntimeError("Not enough labeled rows to train")

    x = np.vstack([extract_features(ROOT / row["image_path"]) for row in rows])
    y = np.asarray([row["label"] for row in rows])
    case_ids = np.asarray([int(row["case_id"]) for row in rows])

    x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(
        x,
        y,
        case_ids,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)

    # trace des case_id vus a l'entrainement et jamais vus, pour eviter les fuites de donnees
    # quand ce modele sert ensuite d'avis auxiliaire pour evaluer MedGemma sur un echantillon
    stem = args.model_path.stem
    (args.model_path.parent / f"{stem}_train_case_ids.txt").write_text(
        "\n".join(str(i) for i in sorted(ids_train)), encoding="utf-8"
    )
    (args.model_path.parent / f"{stem}_test_case_ids.txt").write_text(
        "\n".join(str(i) for i in sorted(ids_test)), encoding="utf-8"
    )

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"model_path": str(args.model_path), "report_path": str(args.report_path), "accuracy": report["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()

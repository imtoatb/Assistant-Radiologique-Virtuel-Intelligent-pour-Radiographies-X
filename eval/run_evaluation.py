from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.database import init_db
from src.guardrails import WARNING_TEXT


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def predict_from_filename(image_path: str, mode: str) -> dict[str, object]:
    name = Path(image_path).name.lower()
    tokens = set(re.split(r"[^a-z0-9]+", name))
    if "suspected_opacity" in name or {"suspected", "opacity"} <= tokens:
        predicted_class = "suspected_opacity"
        confidence = 0.78 if mode == "improved" else 0.72
    elif "normal" in tokens:
        predicted_class = "normal"
        confidence = 0.76 if mode == "improved" else 0.70
    else:
        predicted_class = "uncertain"
        confidence = 0.50 if mode == "improved" else 0.45
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "warning": WARNING_TEXT,
        "json_valid": True,
    }


def summarize(rows: list[dict[str, object]], mode: str) -> dict[str, object]:
    total = len(rows)
    correct = sum(row["label"] == row["predicted_class"] for row in rows)
    warnings = sum(bool(row["warning"]) for row in rows)
    valid_json = sum(bool(row["json_valid"]) for row in rows)
    return {
        "mode": mode,
        "n": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "json_valid_rate": round(valid_json / total, 4) if total else 0.0,
        "warning_rate": round(warnings / total, 4) if total else 0.0,
    }


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_toy(cases_path: Path, out_dir: Path, db_path: Path) -> list[dict[str, object]]:
    init_db(db_path)
    cases = read_cases(cases_path)
    summary = []
    for mode in ("baseline", "improved"):
        predictions = []
        for case in cases:
            pred = predict_from_filename(case["image_path"], mode)
            predictions.append(
                {
                    "case_id": case["case_id"],
                    "label": case["label"],
                    "mode": mode,
                    **pred,
                }
            )
        summary.append(summarize(predictions, mode))
    write_summary_csv(out_dir / "before_after_summary.csv", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["toy"], default="toy")
    parser.add_argument("--cases-path", type=Path, default=ROOT / "data" / "synthetic_cases.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "outputs")
    parser.add_argument("--db-path", type=Path, default=ROOT / "data" / "database.sqlite")
    args = parser.parse_args()

    summary = run_toy(args.cases_path, args.out_dir, args.db_path)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

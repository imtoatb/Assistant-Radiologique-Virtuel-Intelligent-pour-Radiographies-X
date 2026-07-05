from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails
from src.metrics import summarize_metrics
from src.database import init_db, seed_cases, insert_prompt, insert_run, insert_evaluation


def read_cases(cases_path: Path) -> list[dict]:
    with cases_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# execute le predicteur toy sur tous les cas, dans un mode donne (baseline ou improved)
# insere chaque run et son evaluation en base, puis calcule les metriques agregees
def run_mode(mode: str, cases: list[dict], db_path: Path) -> tuple[dict, list[dict]]:
    prompt_id = insert_prompt(db_path, prompt_name=mode, prompt_version="toy_smoke", prompt_text="")

    rows = []
    for case in cases:
        image_path = ROOT / case["image_path"]
        pred = apply_safety_guardrails(toy_predict(image_path, mode=mode))

        run_id = insert_run(db_path, int(case["case_id"]), str(image_path), pred, prompt_id=prompt_id)
        insert_evaluation(db_path, run_id, case["label"], pred["predicted_class"])

        rows.append({
            "label": case["label"],
            "predicted_class": pred["predicted_class"],
            "confidence": pred["confidence"],
            "json_valid": True,
            "warning": pred["warning"],
            "latency_ms": pred["latency_ms"],
        })

    metrics = summarize_metrics(rows)
    metrics["mode"] = mode
    return metrics, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["toy"], default="toy")
    parser.add_argument("--db-path", type=Path, default=ROOT / "data" / "database.sqlite")
    parser.add_argument("--cases-path", type=Path, default=ROOT / "data" / "synthetic_cases.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "outputs")
    args = parser.parse_args()

    init_db(args.db_path)
    seed_cases(args.db_path, args.cases_path)
    cases = read_cases(args.cases_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for mode in ("baseline", "improved"):
        metrics, rows = run_mode(mode, cases, args.db_path)
        summary.append(metrics)
        write_csv(args.out_dir / f"{mode}_predictions.csv", rows)
        (args.out_dir / f"{mode}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    write_csv(args.out_dir / "before_after_summary.csv", summary)

    print(json.dumps(summary))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import toy_predict, vlm_predict_placeholder
from src.guardrails import apply_safety_guardrails, validate_prediction
from src.metrics import summarize_metrics
from src.database import insert_evaluation, insert_run, init_db

import logging

logging.basicConfig(
    filename=str(ROOT / 'data' / 'logs' / 'run_evaluation.log'),    
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def read_cases(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

# permet d'exécuter le modèle sur un ensemble de cas et de stocker les résultats dans la base de données
def run(mode: str, db_path: Path, cases_path: Path, max_cases: int | None = None) -> tuple[list[dict], dict]:
    cases = read_cases(cases_path)

    if max_cases is not None:
        cases = cases[:max_cases]

    rows = []
    for case in cases:

        image_path = ROOT / case['image_path']
        
        # on choisit la fonction de prédiction selon le mode : toy, baseline ou improved
        predict_fct = vlm_predict_placeholder if mode in ['baseline', 'improved'] else toy_predict
        pred = apply_safety_guardrails(predict_fct(image_path, mode=mode))

        valid, errors = validate_prediction(pred)
        row = {
            'case_id': case['case_id'],
            'label': case['label'],
            'predicted_class': pred['predicted_class'],
            'confidence': pred['confidence'],
            'json_valid': valid,
            'warning': pred.get('warning', ''),
            'latency_ms': pred.get('latency_ms', 0),
            'guardrail_errors': ';'.join(errors),
        }
        rows.append(row)

        # récupère l'id du run inséré dans la table runs pour l'utiliser ensuite dans insert_evaluation
        run_id = insert_run(db_path, int(case['case_id']), str(image_path), pred)        
        insert_evaluation(db_path, run_id, case['label'], pred['predicted_class'])
        logging.info(f"{case['case_id']} — {pred['predicted_class']} ({pred['confidence']:.2f}) latency={pred['latency_ms']}ms")

    # calcule les métriques globales sur l'ensemble des cas
    metrics = summarize_metrics(rows)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['toy', 'baseline', 'improved'], default='toy')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'eval' / 'outputs')

    # la database est stockée dans data/medical_ai_evidence.sqlite
    parser.add_argument('--db-path', type=Path, default=ROOT / 'data' / 'medical_ai_evidence.sqlite')

    parser.add_argument('--cases-path', type=Path, default=ROOT / 'data' / 'synthetic_cases.csv')

    # permet de limiter le nombre de cas à traiter (optionel) 
    parser.add_argument('--max-cases', type=int, default=None)

    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ['baseline', 'improved'] if args.mode == 'toy' else [args.mode]
    summary = []
    for mode in modes:
        rows, metrics = run(mode, args.db_path, args.cases_path, max_cases=args.max_cases)
        write_csv(out_dir / f'{mode}_predictions.csv', rows)
        (out_dir / f'{mode}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        summary.append({'mode': mode, **metrics})
    write_csv(out_dir / 'before_after_summary.csv', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

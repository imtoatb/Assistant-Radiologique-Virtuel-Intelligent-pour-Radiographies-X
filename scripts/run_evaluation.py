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
from src.database import insert_evaluation, insert_run, get_evaluations

from src.database import insert_prompt

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

# permet de calculer les métriques globales sur l'ensemble des évaluations stockées en base
def compute_metrics(db_path):
    rows = get_evaluations(db_path)
    if not rows:
        print("Aucune évaluation en base.")
        return
    summary = summarize_metrics([{
        'label': r['ground_truth_label'],
        'predicted_class': r['predicted_class'],
        'confidence': 0,
        'json_valid': True,
        'warning': '',
        'latency_ms': 0,
        'guardrail_errors': ''
    } for r in rows])
    print(json.dumps(summary, indent=2))

# permet d'exécuter le modèle sur un ensemble de cas et de stocker les résultats dans la base de données
def run(mode, db_path, cases_path, max_cases=None, prompt_version=0, case_id=None):
    cases = read_cases(cases_path)

    if case_id is not None:
        cases = [c for c in cases if int(c['case_id']) == case_id]
    elif max_cases is not None:
        cases = cases[:max_cases]

    prompt_text = (ROOT / 'prompts' / f'{mode}_prompt_{prompt_version}.txt').read_text(encoding='utf-8')
    prompt_id = insert_prompt(db_path, prompt_name=mode, prompt_version=f'v{prompt_version}', prompt_text=prompt_text)

    rows = []
    for case in cases:

        image_path = ROOT / case['image_path']
        
        # on choisit la fonction de prédiction selon le mode : toy, baseline ou improved
        predict_fct = vlm_predict_placeholder if mode in ['baseline', 'improved'] else toy_predict
        pred = apply_safety_guardrails(predict_fct(image_path, mode=mode, version=prompt_version))
    
        # récupère l'id du run inséré dans la table runs pour l'utiliser ensuite dans insert_evaluation
        run_id = insert_run(db_path, int(case['case_id']), str(image_path), pred, prompt_id=prompt_id)
        insert_evaluation(db_path, run_id, case['label'], pred['predicted_class'])
        logging.info(f"{case['case_id']} — {pred['predicted_class']} ({pred['confidence']:.2f}) latency={pred['latency_ms']}ms")
        print(f"case_id={case['case_id']} | pred={pred['predicted_class']} | conf={pred['confidence']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    # to choose the model
    parser.add_argument('--mode', choices=['toy', 'baseline', 'improved'], default='toy')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'data' / 'medical_ai_evidence.sqlite')
    parser.add_argument('--cases-path', type=Path, default=ROOT / 'data' / 'synthetic_cases.csv')
    
    # to indicte a maximum number of case
    parser.add_argument('--max-cases', type=int, default=None)

    # choose the version of the prompt
    parser.add_argument('--prompt-version', type=int, default=0)

    # choose the case id to run the evaluation on a specific case
    parser.add_argument('--case-id', type=int, default=None)

    # to compute metrics on the existing evaluations in the database
    parser.add_argument('--compute-metrics', action='store_true')

    args = parser.parse_args()

    if args.compute_metrics:
        compute_metrics(args.db_path)
    else:
        modes = ['baseline', 'improved'] if args.mode == 'toy' else [args.mode]
        for mode in modes:
            run(mode, args.db_path, args.cases_path,
                max_cases=args.max_cases,
                prompt_version=args.prompt_version,
                case_id=args.case_id)



if __name__ == '__main__':
    main()

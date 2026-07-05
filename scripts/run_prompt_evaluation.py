from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import toy_predict, vlm_predict_placeholder
from src.guardrails import apply_safety_guardrails, validate_prediction
from src.metrics import summarize_metrics
from src.database import insert_evaluation, insert_run, get_evaluations, insert_prompt, get_case_ids_with_prompt

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

# sélectionne un échantillon aléatoire équilibré entre normal et suspected_opacity
# seed permet de reproduire le même échantillon pour comparer baseline vs improved sur les mêmes images
def balanced_sample(cases: list[dict], n: int = 100, seed: int = 42) -> list[dict]:
    random.seed(seed)
    normal = [c for c in cases if c['label'] == 'normal']
    abnormal = [c for c in cases if c['label'] == 'suspected_opacity']
    n_each = n // 2
    selected = random.sample(normal, min(n_each, len(normal))) + random.sample(abnormal, min(n_each, len(abnormal)))
    random.shuffle(selected)
    print(f"Échantillon : {len([c for c in selected if c['label'] == 'normal'])} normal, {len([c for c in selected if c['label'] == 'suspected_opacity'])} suspected_opacity")
    return selected

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
# mode : 'toy' pour la fausse IA de test, 'baseline' ou 'improved' pour MedGemma
# prompt_version : 0 pour baseline, 1/2/3... pour les versions du prompt improved
def run(mode, db_path, cases_path, max_cases=None, prompt_version=0, case_id=None, case_ids=None, sample_n=None, sample_seed=42):
    cases = read_cases(cases_path)

    if case_ids is not None:
        cases = [c for c in cases if int(c['case_id']) in case_ids] # liste de cas

    elif case_id is not None:
        cases = [c for c in cases if int(c['case_id']) == case_id] 

    elif sample_n is not None:
        cases = balanced_sample(cases, n=sample_n, seed=sample_seed)

    elif max_cases is not None:
        cases = cases[:max_cases]

    # le nom du prompt correspond au mode pour baseline/improved, et à "toy" pour le mode de test
    prompt_name = mode if mode in ["baseline", "improved"] else "toy"
    prompt_path = ROOT / "prompts" / f"{prompt_name}_prompt_{prompt_version}.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_id = insert_prompt(db_path, prompt_name=prompt_name, prompt_version=f"v{prompt_version}", prompt_text=prompt_text)

    # cas deja testes avec ce prompt exact (meme nom et meme version) : on ne relance pas l'inference dessus
    already_done = get_case_ids_with_prompt(db_path, prompt_id)

    for i, case in enumerate(cases, 1):
        case_id = int(case['case_id'])
        if case_id in already_done:
            print(f"[{i}/{len(cases)}] case_id={case_id} deja evalue avec {prompt_name} v{prompt_version}, ignore")
            continue

        print(f"[{i}/{len(cases)}]", "**"*100)
        image_path = ROOT / case['image_path']

        # fausse IA basée sur le nom de fichier, utilisée uniquement pour tester le pipeline
        if mode == "toy":
            pred = apply_safety_guardrails(toy_predict(image_path, mode="baseline", version=0))
        # vraie IA MedGemma avec le prompt baseline ou improved
        else:
            pred = apply_safety_guardrails(vlm_predict_placeholder(image_path, mode=mode, version=prompt_version))

        # récupère l'id du run inséré dans la table runs pour l'utiliser ensuite dans insert_evaluation
        run_id = insert_run(db_path, case_id, str(image_path), pred, prompt_id=prompt_id)
        insert_evaluation(db_path, run_id, case['label'], pred['predicted_class'])
        logging.info(f"{case_id} — {pred['predicted_class']} ({pred['confidence']:.2f}) latency={pred['latency_ms']}ms")
        print(f"case_id={case_id} | pred={pred['predicted_class']} | conf={pred['confidence']:.2f}")

# lance automatiquement baseline puis improved sur le même échantillon équilibré de 100 images
def run_full_evaluation(db_path, cases_path, improved_version=1, sample_n=100, sample_seed=42):
    print("=== BASELINE ===")
    run("baseline", db_path, cases_path, prompt_version=0, sample_n=sample_n, sample_seed=sample_seed)
    print("=== IMPROVED ===")
    run("improved", db_path, cases_path, prompt_version=improved_version, sample_n=sample_n, sample_seed=sample_seed)

def main() -> None:
    parser = argparse.ArgumentParser()
    # choisir entre la fausse IA de test (toy) ou la vraie IA MedGemma (baseline / improved)
    parser.add_argument('--mode', choices=['toy', 'baseline', 'improved', 'full'], default='toy')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'data' / 'database.sqlite')
    parser.add_argument('--cases-path', type=Path, default=ROOT / 'data' / 'cases.csv')

    # limiter le nombre de cas à traiter (optionnel, ignoré si --sample-n est utilisé)
    parser.add_argument('--max-cases', type=int, default=None)

    # version du prompt : 0 pour baseline, 1/2/3... pour les versions du prompt improved
    parser.add_argument('--prompt-version', type=int, default=0)

    # choisir un cas spécifique avec son ID (optionnel)
    parser.add_argument('--case-id', type=int, default=None)

    # échantillon aléatoire équilibré de N images (par défaut 100)
    parser.add_argument('--sample-n', type=int, default=None)

    # seed pour reproduire le même échantillon
    parser.add_argument('--sample-seed', type=int, default=42)

    # calculer les métriques sur l'ensemble des évaluations stockées en base
    parser.add_argument('--compute-metrics', action='store_true')

    # choisir une liste de cas
    parser.add_argument('--case-ids', type=int, nargs='+', default=None)

    args = parser.parse_args()

    if args.compute_metrics:
        compute_metrics(args.db_path)
    # lance baseline + improved automatiquement sur le même échantillon équilibré
    elif args.mode == 'full':
        run_full_evaluation(args.db_path, args.cases_path, improved_version=args.prompt_version, sample_n=args.sample_n or 100, sample_seed=args.sample_seed)
    else:
        run(
            args.mode,
            args.db_path,
            args.cases_path,
            max_cases=args.max_cases, # limite le nombre de cas (optionnel)
            prompt_version=args.prompt_version, # version du prompt (0=baseline, 1/2/3=improved)
            case_id=args.case_id, # un seul cas par son ID (optionnel)
            case_ids=args.case_ids, # liste de cas par leurs IDs (optionnel)
            sample_n=args.sample_n, # échantillon aléatoire équilibré de N images
            sample_seed=args.sample_seed # seed pour reproduire le même échantillon
            )

if __name__ == '__main__':
    main()
"""
Script batch pour comparer toy_predict vs vlm_predict_placeholder sur un échantillon d'images.
Le CSV est initialisé avec toutes les images dès le départ.
Les colonnes résultats sont remplies au fur et à mesure.
Reprend automatiquement là où il s'est arrêté si interrompu.

Usage :
    python scripts/run_batch.py
    python scripts/run_batch.py --sample 5       # 5 images par dossier
    python scripts/run_batch.py --mode baseline  # un seul mode VLM
    python scripts/run_batch.py --delay 10       # 10 secondes entre chaque image
    python scripts/run_batch.py --init           # réinitialise le CSV
"""

from __future__ import annotations

import argparse
import gc
import random
import time
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.inference import toy_predict, vlm_predict_placeholder
from src.guardrails import apply_safety_guardrails

FOLDERS = {
    'normal_lung_probe': 'normal',
    'abnormal_lung_images': 'suspected_opacity',
}

RESULTS_PATH = ROOT / 'results' / 'batch_results.csv'

def build_sample(data_root: Path, n: int, seed: int = 42) -> list[dict]:
    # Construire un échantillon aléatoire d'images à partir des dossiers spécifiés.
    random.seed(seed)
    sample = []
    for folder, ground_truth in FOLDERS.items():
        all_images = sorted((data_root / folder).glob('*.jpg'))
        picked = random.sample(all_images, min(n, len(all_images)))
        for img_path in picked:
            sample.append({
                'file': img_path.name,
                'path': str(img_path.relative_to(ROOT)),
                'folder': folder,
                'ground_truth': ground_truth,
            })
    return sample

# initialisation d'un csv avec les images à traiter pour palier les interruptions (pb de gpu qui surchauffe par example)
def init_csv(sample: list[dict]):
    """Initialise le CSV avec toutes les images et colonnes résultats vides."""
    df = pd.DataFrame(sample)
    df['toy_class'] = None
    df['toy_confidence'] = None
    df['vlm_baseline_class'] = None
    df['vlm_baseline_confidence'] = None
    df['vlm_improved_class'] = None
    df['vlm_improved_confidence'] = None
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"CSV initialisé avec {len(df)} images : {RESULTS_PATH}")
    return df


def run(sample_size: int = 10, vlm_mode: str = 'both', delay: float = 2.0, reinit: bool = False):
    data_root = ROOT / 'data'
    sample = build_sample(data_root, sample_size)

    # Initialisation ou chargement du CSV
    if reinit or not RESULTS_PATH.exists():
        df = init_csv(sample)
    else:
        df = pd.read_csv(RESULTS_PATH)
        print(f"CSV existant chargé : {len(df)} images")

        # Forcer les colonnes de classe en string
        for col in ['toy_class', 'vlm_baseline_class', 'vlm_improved_class']:
            if col in df.columns:
                df[col] = df[col].where(df[col].notna(), None).astype(object)

    # Identifier les lignes non traitées selon le mode
    if vlm_mode == 'baseline':
        remaining = df[df['vlm_baseline_class'].isna()]
    elif vlm_mode == 'improved':
        remaining = df[df['vlm_improved_class'].isna()]
    else:
        remaining = df[df['vlm_baseline_class'].isna() | df['vlm_improved_class'].isna()]

    print(f"Traités : {len(df) - len(remaining)}/{len(df)}")
    print(f"Restants : {len(remaining)}")

    for i, (idx, r) in enumerate(remaining.iterrows()):
        img_path = ROOT / r['path']
        print(f"\n[{i+1}/{len(remaining)}] {r['file']}")

        # Toy predict (si pas encore fait)
        if pd.isna(r.get('toy_class')):
            try:
                pred = apply_safety_guardrails(toy_predict(img_path, mode='baseline'))
                df.at[idx, 'toy_class'] = pred['predicted_class']
                df.at[idx, 'toy_confidence'] = pred['confidence']
            except Exception as e:
                df.at[idx, 'toy_class'] = 'error'
                print(f"  toy error: {e}")

        # VLM baseline
        if vlm_mode in ('baseline', 'both') and pd.isna(r.get('vlm_baseline_class')):
            try:
                pred = apply_safety_guardrails(vlm_predict_placeholder(img_path, mode='baseline'))
                df.at[idx, 'vlm_baseline_class'] = pred['predicted_class']
                df.at[idx, 'vlm_baseline_confidence'] = pred['confidence']
                print(f"  VLM baseline : {pred['predicted_class']} ({pred['confidence']:.2f})")
            except Exception as e:
                df.at[idx, 'vlm_baseline_class'] = 'error'
                print(f"  VLM baseline error: {type(e).__name__}: {e}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            time.sleep(delay)

        # VLM improved
        if vlm_mode in ('improved', 'both') and pd.isna(r.get('vlm_improved_class')):
            try:
                pred = apply_safety_guardrails(vlm_predict_placeholder(img_path, mode='improved'))
                df.at[idx, 'vlm_improved_class'] = pred['predicted_class']
                df.at[idx, 'vlm_improved_confidence'] = pred['confidence']
                print(f"  VLM improved : {pred['predicted_class']} ({pred['confidence']:.2f})")
            except Exception as e:
                df.at[idx, 'vlm_improved_class'] = 'error'
                print(f"  VLM improved error: {type(e).__name__}: {e}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            time.sleep(delay)

        # Sauvegarde après chaque image
        df.to_csv(RESULTS_PATH, index=False)
        print(f"  OK — sauvegardé")

    # Résultats finaux
    print("\n=== RESULTATS FINAUX ===")
    df_final = pd.read_csv(RESULTS_PATH)
    for col in ['toy_class', 'vlm_baseline_class', 'vlm_improved_class']:
        if col in df_final.columns:
            valid = df_final[df_final[col] != 'error']
            if len(valid) > 0:
                acc = (valid[col] == valid['ground_truth']).mean()
                print(f"{col} accuracy: {acc:.1%} ({len(valid)} images)")

    print(df_final.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=10, help='Images par dossier')
    parser.add_argument('--mode', choices=['baseline', 'improved', 'both'], default='both')
    parser.add_argument('--delay', type=float, default=5.0, help='Secondes entre chaque image')
    parser.add_argument('--init', action='store_true', help='Réinitialise le CSV')
    args = parser.parse_args()

    run(sample_size=args.sample, vlm_mode=args.mode, delay=args.delay, reinit=args.init)
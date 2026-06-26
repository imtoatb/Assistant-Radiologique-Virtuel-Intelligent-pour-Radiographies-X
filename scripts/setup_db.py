"""
Script d'initialisation de la base de données.
Usage :
    python scripts/setup_db.py --all # tout initialiser
    python scripts/setup_db.py --init # crée les tables uniquement
    python scripts/setup_db.py --cases  # charge les cas uniquement
    python scripts/setup_db.py --prompts # charge les prompts uniquement
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.database import init_db, seed_cases, insert_prompt

DB_PATH = ROOT / 'data' / 'medical_ai_evidence.sqlite'
CASES_CSV = ROOT / 'data' / 'cases.csv'
PROMPTS_DIR = ROOT / 'prompts'


def fill_prompts(db_path: Path, prompts_dir: Path) -> None:
    """Insère les prompts du dossier prompts/ dans la table prompts
    Ignore les prompts déjà présents
    """
    for prompt_file in sorted(prompts_dir.glob('*.txt')):
        parts = prompt_file.stem.split('_')
        prompt_name = parts[0]
        prompt_version = f"v{parts[-1]}"
        prompt_id = insert_prompt(db_path, prompt_name, prompt_version, prompt_file.read_text(encoding='utf-8'))
        print(f"  {prompt_file.name} -> id={prompt_id} ({prompt_name} {prompt_version})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', action='store_true', help='Crée les tables')
    parser.add_argument('--cases', action='store_true', help='Charge les cas')
    parser.add_argument('--prompts', action='store_true', help='Charge les prompts')
    parser.add_argument('--all', action='store_true', help='Tout initialiser')
    args = parser.parse_args()

    if not any([args.init, args.cases, args.prompts, args.all]):
        parser.print_help()
        return

    if args.all or args.init:
        print("Initialisation de la base de données")
        init_db(DB_PATH)
        print("  -> OK")

    if args.all or args.cases:
        print("Chargement des cas depuis le CSV")
        inserted = seed_cases(DB_PATH, CASES_CSV)
        print(f"  -> {inserted} cas insérés")

    if args.all or args.prompts:
        print("Chargement des prompts depuis le dossier prompts/")
        fill_prompts(DB_PATH, PROMPTS_DIR)
        print("  -> OK")


if __name__ == '__main__':
    main()
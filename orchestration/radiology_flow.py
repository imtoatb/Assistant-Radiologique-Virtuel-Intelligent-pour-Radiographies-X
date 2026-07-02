from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFECT_HOME = ROOT / ".prefect_home"
os.environ.setdefault("PREFECT_HOME", str(DEFAULT_PREFECT_HOME))
DEFAULT_PREFECT_HOME.mkdir(parents=True, exist_ok=True)

from prefect import flow, get_run_logger, task


def _run_command(args: list[str]) -> None:
    logger = get_run_logger()
    logger.info("Running: %s", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def _run_optional_command(args: list[str], continue_on_error: bool) -> None:
    logger = get_run_logger()
    try:
        _run_command(args)
    except subprocess.CalledProcessError:
        if not continue_on_error:
            raise
        logger.warning("Optional command failed; continuing pipeline: %s", " ".join(args))


@task
def import_abnormal_lung_dataset(clean: bool, continue_on_error: bool) -> None:
    args = [sys.executable, "scripts/import_abnormal_lung_dataset.py"]
    if clean:
        args.append("--clean")
    _run_optional_command(args, continue_on_error=continue_on_error)


@task
def import_rsna_pneumonia(max_cases: int, clean: bool, continue_on_error: bool) -> None:
    args = [
        sys.executable,
        "scripts/import_rsna_pneumonia.py",
        "--max-cases",
        str(max_cases),
    ]
    if clean:
        args.append("--clean")
    _run_optional_command(args, continue_on_error=continue_on_error)


@task
def generate_cases_csv() -> None:
    _run_command([sys.executable, "scripts/generate_cases_csv.py"])


@task
def setup_database() -> None:
    _run_command([sys.executable, "scripts/setup_db.py", "--all"])


@task
def train_pixel_baseline() -> None:
    _run_command(
        [
            sys.executable,
            "scripts/train_pixel_baseline.py",
            "--cases-path",
            "data/cases.csv",
            "--report-path",
            "eval/pixel_baseline_report.json",
        ]
    )


@task
def run_evaluation(
    mode: str,
    max_cases: int | None,
    prompt_version: int,
    sample_n: int | None,
    sample_seed: int,
) -> None:
    args = [
        sys.executable,
        "scripts/run_evaluation.py",
        "--mode",
        mode,
        "--db-path",
        "data/database.sqlite",
        "--cases-path",
        "data/cases.csv",
        "--prompt-version",
        str(prompt_version),
        "--sample-seed",
        str(sample_seed),
    ]
    if sample_n is not None:
        args.extend(["--sample-n", str(sample_n)])
    elif max_cases is not None:
        args.extend(["--max-cases", str(max_cases)])
    _run_command(args)


@task
def compute_metrics() -> None:
    _run_command(
        [
            sys.executable,
            "scripts/run_evaluation.py",
            "--db-path",
            "data/database.sqlite",
            "--compute-metrics",
        ]
    )


@task
def verify_outputs() -> None:
    expected_paths = [
        ROOT / "data" / "cases.csv",
        ROOT / "data" / "database.sqlite",
        ROOT / "models" / "pixel_baseline.joblib",
        ROOT / "eval" / "pixel_baseline_report.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected outputs: {missing}")


@flow(name="radiology-pipeline")
def radiology_pipeline(
    max_cases: int | None = 20,
    with_abnormal_import: bool = False,
    with_rsna: bool = False,
    clean_imports: bool = False,
    rsna_max_cases: int = 150,
    train: bool = True,
    eval_mode: str = "improved",
    prompt_version: int = 1,
    sample_n: int | None = None,
    sample_seed: int = 42,
    compute_eval_metrics: bool = True,
    continue_on_import_error: bool = False,
) -> None:
    if with_abnormal_import:
        import_abnormal_lung_dataset(
            clean=clean_imports,
            continue_on_error=continue_on_import_error,
        )
    if with_rsna:
        import_rsna_pneumonia(
            max_cases=rsna_max_cases,
            clean=clean_imports,
            continue_on_error=continue_on_import_error,
        )

    generate_cases_csv()
    setup_database()
    if train:
        train_pixel_baseline()
    if eval_mode != "none":
        run_evaluation(
            mode=eval_mode,
            max_cases=max_cases,
            prompt_version=prompt_version,
            sample_n=sample_n,
            sample_seed=sample_seed,
        )
    if compute_eval_metrics:
        compute_metrics()
    verify_outputs()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--with-abnormal-import", action="store_true")
    parser.add_argument("--with-rsna", action="store_true")
    parser.add_argument("--clean-imports", action="store_true")
    parser.add_argument("--rsna-max-cases", type=int, default=150)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--eval-mode",
        choices=["none", "toy", "baseline", "improved", "full"],
        default="improved",
    )
    parser.add_argument("--prompt-version", type=int, default=1)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument(
        "--continue-on-import-error",
        action="store_true",
        help="Continue the flow if optional imports fail, for example when Kaggle auth is missing.",
    )
    args = parser.parse_args()
    max_cases = None if args.max_cases <= 0 else args.max_cases
    radiology_pipeline(
        max_cases=max_cases,
        with_abnormal_import=args.with_abnormal_import,
        with_rsna=args.with_rsna,
        clean_imports=args.clean_imports,
        rsna_max_cases=args.rsna_max_cases,
        train=not args.skip_training,
        eval_mode=args.eval_mode,
        prompt_version=args.prompt_version,
        sample_n=args.sample_n,
        sample_seed=args.sample_seed,
        compute_eval_metrics=not args.skip_metrics,
        continue_on_import_error=args.continue_on_import_error,
    )


if __name__ == "__main__":
    main()

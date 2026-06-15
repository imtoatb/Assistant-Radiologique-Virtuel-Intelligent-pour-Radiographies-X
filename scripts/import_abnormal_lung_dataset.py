from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import kagglehub


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "image"


def infer_label(path: Path, dataset: str) -> str:
    text = " ".join(part.lower() for part in path.parts)
    dataset_text = dataset.lower()
    tokens = set(re.split(r"[^a-z0-9]+", f"{text} {dataset_text}"))
    if "abnormal" in tokens:
        return "suspected_opacity"
    if "normal" in tokens:
        return "normal"
    if any(token in tokens for token in ("opacity", "pneumonia", "disease", "infected")):
        return "suspected_opacity"
    return "uncertain"


def case_prefix(dataset: str) -> str:
    return slugify(dataset.split("/", maxsplit=1)[-1]).upper()


def iter_images(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def cached_dataset_dir(dataset: str) -> Path | None:
    owner, name = dataset.split("/", maxsplit=1)
    versions_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / owner / name / "versions"
    if not versions_dir.exists():
        return None
    versions = sorted(path for path in versions_dir.iterdir() if path.is_dir())
    return versions[-1] if versions else None


def resolve_dataset_dir(dataset: str) -> Path:
    cached = cached_dataset_dir(dataset)
    if cached is not None:
        print(f"Using cached dataset: {cached}")
        return cached
    try:
        return Path(kagglehub.dataset_download(dataset))
    except Exception as exc:
        raise RuntimeError(f"Could not download {dataset} and no local cache was found") from exc


def clean_images_dir(images_dir: Path) -> None:
    if not images_dir.exists():
        return
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            path.unlink()


def import_dataset(dataset: str, images_dir: Path, start_index: int = 1) -> list[dict]:
    images_dir = images_dir.resolve()
    dataset_dir = resolve_dataset_dir(dataset)
    images = iter_images(dataset_dir)
    if not images:
        raise RuntimeError(f"No image files found in downloaded dataset: {dataset_dir}")

    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    prefix = case_prefix(dataset)

    for index, source_path in enumerate(images, start=start_index):
        label = infer_label(source_path.relative_to(dataset_dir), dataset)
        case_id = f"{prefix}_{index:05d}"
        target_name = f"{case_id}_{label}_{slugify(source_path.stem)}{source_path.suffix.lower()}"
        target_path = images_dir / target_name
        shutil.copy2(source_path, target_path)
        rows.append(
            {
                "case_id": case_id,
                "image_path": target_path.relative_to(ROOT).as_posix(),
                "source": dataset,
                "label": label,
                "split": "external",
                "quality": "unknown",
                "notes": f"Imported from {source_path.relative_to(dataset_dir).as_posix()}",
            }
        )

    print(f"Imported {len(rows)} images from {dataset}")
    return rows


def write_cases(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case_id", "image_path", "source", "label", "split", "quality", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Kaggle dataset handle. Can be passed multiple times.",
    )
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data" / "lung_images")
    parser.add_argument("--csv-path", type=Path, default=ROOT / "data" / "lung_cases.csv")
    parser.add_argument("--clean", action="store_true", help="Remove previously imported images first")
    args = parser.parse_args()

    datasets = args.dataset or ["fkarimovv/abnormal-lung", "fkarimovv/normal-lung"]
    if args.clean:
        clean_images_dir(args.images_dir)

    rows = []
    for dataset in datasets:
        rows.extend(import_dataset(dataset, args.images_dir, start_index=len(rows) + 1))

    write_cases(args.csv_path, rows)
    print(f"Imported {len(rows)} images total")
    print(f"Images: {args.images_dir}")
    print(f"CSV: {args.csv_path}")


if __name__ == "__main__":
    main()

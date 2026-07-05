from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import shutil

import kagglehub
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "rsna-pneumonia-detection-challenge"


def require_pydicom():
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "pydicom is required to import RSNA DICOM files. "
            "Install it with: python -m pip install pydicom"
        ) from exc
    return pydicom


def resolve_competition_dir(dataset: str | None) -> Path:
    # la competition originale est fermee depuis 2018 : si --dataset est fourni,
    # on telecharge un dataset Kaggle classique (miroir) au lieu de la competition
    if dataset:
        try:
            path = Path(kagglehub.dataset_download(dataset))
        except Exception as exc:
            raise RuntimeError(
                f"Could not download Kaggle dataset {dataset}. "
                "Make sure you are authenticated with Kaggle and that the dataset handle is correct."
            ) from exc
    else:
        try:
            path = Path(kagglehub.competition_download(COMPETITION))
        except Exception as exc:
            raise RuntimeError(
                "Could not download RSNA Pneumonia from Kaggle. "
                "Make sure you are authenticated with Kaggle and that you accepted "
                "the competition rules for rsna-pneumonia-detection-challenge. "
                "If the competition is closed and this keeps failing, try a mirrored "
                "dataset instead with --dataset <owner>/<name>."
            ) from exc
    if not path.exists():
        raise RuntimeError(f"Downloaded path does not exist: {path}")
    return path


def find_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise RuntimeError(f"Could not find {name} under {root}")
    return matches[0]


def find_images_dir(root: Path) -> Path:
    candidates = sorted(path for path in root.rglob("stage_2_train_images") if path.is_dir())
    if not candidates:
        raise RuntimeError(f"Could not find stage_2_train_images under {root}")
    return candidates[0]


def read_labels(labels_path: Path) -> dict[str, dict]:
    boxes_by_patient: dict[str, list[dict]] = defaultdict(list)
    targets_by_patient: dict[str, int] = {}

    with labels_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            patient_id = row["patientId"]
            target = int(row["Target"])
            targets_by_patient[patient_id] = max(targets_by_patient.get(patient_id, 0), target)

            if target == 1:
                boxes_by_patient[patient_id].append(
                    {
                        "x": row["x"],
                        "y": row["y"],
                        "width": row["width"],
                        "height": row["height"],
                    }
                )

    labels = {}
    for patient_id, target in targets_by_patient.items():
        labels[patient_id] = {
            "label": "suspected_opacity" if target == 1 else "normal",
            "boxes": boxes_by_patient.get(patient_id, []),
        }
    return labels


def dicom_to_png(dicom_path: Path, output_path: Path) -> None:
    pydicom = require_pydicom()
    dataset = pydicom.dcmread(dicom_path)
    pixels = dataset.pixel_array.astype(np.float32)
    pixels -= float(pixels.min())
    max_value = float(pixels.max())
    if max_value > 0:
        pixels /= max_value
    image = Image.fromarray((pixels * 255).astype(np.uint8))
    image.save(output_path)


def clean_output(images_dir: Path, csv_path: Path) -> None:
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if csv_path.exists():
        csv_path.unlink()


def import_rsna(images_dir: Path, csv_path: Path, max_cases: int | None, clean: bool, dataset: str | None) -> None:
    if clean:
        clean_output(images_dir, csv_path)

    source_dir = resolve_competition_dir(dataset)
    source_name = dataset or COMPETITION
    labels_path = find_file(source_dir, "stage_2_train_labels.csv")
    train_images_dir = find_images_dir(source_dir)
    labels = read_labels(labels_path)

    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    dicom_paths = sorted(train_images_dir.glob("*.dcm"))
    if max_cases is not None:
        dicom_paths = dicom_paths[:max_cases]

    for index, dicom_path in enumerate(dicom_paths, start=1):
        patient_id = dicom_path.stem
        metadata = labels.get(patient_id, {"label": "uncertain", "boxes": []})
        label = metadata["label"]
        # case_id doit rester un entier simple : src/database.py fait int(case_id)
        # partout (insert_case, seed_cases), un id de la forme "RSNA_PNEUMONIA_00001" ferait planter l'insertion
        case_id = index
        target_path = images_dir / f"rsna_pneumonia_{label}_{index:05d}.png"
        dicom_to_png(dicom_path, target_path)
        rows.append(
            {
                "case_id": case_id,
                "image_path": target_path.relative_to(ROOT).as_posix(),
                "source": source_name,
                "source_patient_id": patient_id,
                "label": label,
                "split": "external",
                "quality": "unknown",
                "bbox_count": len(metadata["boxes"]),
                "bboxes": str(metadata["boxes"]),
                "notes": "Imported from RSNA Pneumonia Detection Challenge",
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Imported {len(rows)} RSNA cases")
    print(f"Images: {images_dir}")
    print(f"CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data" / "brutes_rsa")
    parser.add_argument("--csv-path", type=Path, default=ROOT / "data" / "cases.csv")
    parser.add_argument("--max-cases", type=int, default=150)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Kaggle dataset handle to use instead of the (closed) competition, "
            "e.g. sovitrath/rsna-pneumonia-detection-2018"
        ),
    )
    args = parser.parse_args()

    max_cases = None if args.max_cases <= 0 else args.max_cases
    import_rsna(args.images_dir, args.csv_path, max_cases=max_cases, clean=args.clean, dataset=args.dataset)


if __name__ == "__main__":
    main()

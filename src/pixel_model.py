from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageFilter, ImageStat


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pixel_baseline.joblib"


def extract_features(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("L").resize((128, 128))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    stat = ImageStat.Stat(image)
    edges = np.asarray(image.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=16, range=(0.0, 1.0), density=True)

    features = [
        float(arr.mean()),
        float(arr.std()),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
        float(np.percentile(arr, 90)),
        float(edges.mean()),
        float(edges.std()),
        float(stat.extrema[0][0] / 255.0),
        float(stat.extrema[0][1] / 255.0),
    ]
    return np.asarray([*features, *hist.tolist()], dtype=np.float32)


def model_exists(path: str | Path = MODEL_PATH) -> bool:
    return Path(path).exists()


def load_model(path: str | Path = MODEL_PATH) -> Any:
    return joblib.load(path)

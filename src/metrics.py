from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

CLASSES = ["normal", "suspected_opacity", "uncertain"]


def accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return 0.0
    return sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    scores = []
    for class_name in classes:
        tp = sum(t == class_name and p == class_name for t, p in zip(y_true, y_pred))
        fp = sum(t != class_name and p == class_name for t, p in zip(y_true, y_pred))
        fn = sum(t == class_name and p != class_name for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        scores.append(f1)
    return sum(scores) / len(scores)


def sensitivity(y_true: Iterable[str], y_pred: Iterable[str], positive_class: str = "suspected_opacity") -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    tp = sum(t == positive_class and p == positive_class for t, p in zip(y_true, y_pred))
    fn = sum(t == positive_class and p != positive_class for t, p in zip(y_true, y_pred))
    return tp / (tp + fn) if (tp + fn) else 0.0


def specificity(y_true: Iterable[str], y_pred: Iterable[str], positive_class: str = "suspected_opacity") -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    tn = sum(t != positive_class and p != positive_class for t, p in zip(y_true, y_pred))
    fp = sum(t != positive_class and p == positive_class for t, p in zip(y_true, y_pred))
    return tn / (tn + fp) if (tn + fp) else 0.0


def confusion_counts(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, int]:
    counts = Counter()
    for actual, predicted in zip(y_true, y_pred):
        counts[f"{actual}__{predicted}"] += 1
    return dict(counts)


def confusion_matrix_numpy(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> np.ndarray:
    counts = confusion_counts(y_true, y_pred)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for row_index, actual in enumerate(classes):
        for col_index, predicted in enumerate(classes):
            matrix[row_index, col_index] = counts.get(f"{actual}__{predicted}", 0)
    return matrix


def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    y_true = [row["label"] for row in rows]
    y_pred = [row["predicted_class"] for row in rows]
    json_valid = [row.get("json_valid", True) for row in rows]
    warnings = [bool(row.get("warning")) for row in rows]
    return {
        "n": len(rows),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "json_valid_rate": round(sum(json_valid) / len(json_valid), 4) if rows else 0,
        "warning_rate": round(sum(warnings) / len(warnings), 4) if rows else 0,
        "uncertain_rate": round(sum(predicted == "uncertain" for predicted in y_pred) / len(y_pred), 4) if rows else 0,
        "sensitivity": round(sensitivity(y_true, y_pred), 4),
        "specificity": round(specificity(y_true, y_pred), 4),
    }

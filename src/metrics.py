from __future__ import annotations

from collections import Counter
from typing import Iterable
import numpy as np

CLASSES = ["normal", "suspected_opacity", "uncertain"]


def accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] | None = None) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    # ne moyenne que sur les classes reellement presentes dans le ground truth
    # sinon une classe absente du dataset (ex: "uncertain") a un F1 force a 0
    # et plombe artificiellement le macro-F1 (comportement par defaut de sklearn)
    if classes is None:
        classes = sorted(set(y_true))
    if not classes:
        return 0.0
    scores = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        scores.append(f1)
    return sum(scores) / len(scores)

def sensitivity(y_true: Iterable[str], y_pred: Iterable[str], 
                positive_class: str = "suspected_opacity") -> float:
    """Rappel pour la classe positive (taux de vrais positifs)."""
    y_true = list(y_true); y_pred = list(y_pred)
    tp = sum(t == positive_class and p == positive_class for t, p in zip(y_true, y_pred))
    fn = sum(t == positive_class and p != positive_class for t, p in zip(y_true, y_pred))
    return tp / (tp + fn) if (tp + fn) else 0.0


def specificity(y_true: Iterable[str], y_pred: Iterable[str],
                negative_class: str = "normal") -> float:
    """Rappel pour la classe négative (taux de vrais négatifs)."""
    y_true = list(y_true); y_pred = list(y_pred)
    tn = sum(t == negative_class and p == negative_class for t, p in zip(y_true, y_pred))
    # faux positif = vrai negatif predit comme autre chose (symetrique du fn de sensitivity)
    fp = sum(t == negative_class and p != negative_class for t, p in zip(y_true, y_pred))
    return tn / (tn + fp) if (tn + fp) else 0.0

# permet de calculer les métriques globales sur un ensemble de cas
def confusion_counts(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, int]:
    counts = Counter()
    for t, p in zip(y_true, y_pred):
        counts[f"{t}__{p}"] += 1
    return dict(counts)

# permet de calculer la matrice de confusion sous forme de tableau numpy
def confusion_matrix_numpy(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> np.ndarray:
    counts = confusion_counts(y_true, y_pred)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for i, t in enumerate(classes):
        for j, p in enumerate(classes):
            cm[i, j] = counts.get(f"{t}__{p}", 0)
    return cm

def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    y_true = [r["label"] for r in rows]
    y_pred = [r["predicted_class"] for r in rows]
    json_valid = [r.get("json_valid", True) for r in rows]
    warnings = [bool(r.get("warning")) for r in rows]
    return {
        "n": len(rows),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "json_valid_rate": round(sum(json_valid) / len(json_valid), 4) if rows else 0,
        "warning_rate": round(sum(warnings) / len(warnings), 4) if rows else 0,
        "uncertain_rate": round(sum(p == "uncertain" for p in y_pred) / len(y_pred), 4) if rows else 0,
        "sensitivity": round(sensitivity(y_true, y_pred), 4),
        "specificity": round(specificity(y_true, y_pred), 4),
    }

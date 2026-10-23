"""Classification metrics and confusion-matrix plotting.

Labels: 0 = negative, 1 = positive. Headline precision/recall/F1 are for the
positive class. No pass/fail threshold is applied here on purpose.
"""
from __future__ import annotations

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS = (0, 1)
LABEL_NAMES = {0: "negative", 1: "positive"}


def compute_metrics(y_true, y_pred) -> dict:
    """Metrics are well-defined even if a class is never predicted (zero_division=0)."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    per_class = {
        LABEL_NAMES[label]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "positive_class": {
            key: per_class["positive"][key] for key in ("precision", "recall", "f1")
        },
        "per_class": per_class,
        "macro_f1": float(f1.mean()),
        "confusion_matrix": {
            "labels": list(LABELS),
            "layout": "rows = actual, columns = predicted: [[tn, fp], [fn, tp]]",
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def save_confusion_matrix_png(matrix, path: Path, title: str) -> None:
    # Imported lazily so code that only needs metrics does not load matplotlib.
    import matplotlib

    matplotlib.use("Agg")  # headless backend: no display needed in WSL or Docker
    import matplotlib.pyplot as plt

    names = [LABEL_NAMES[label] for label in LABELS]
    peak = max(max(row) for row in matrix) or 1

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix)
    ax.set_xticks(range(len(names)), labels=names)
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j, i, str(value), ha="center", va="center",
                color="white" if value < peak / 2 else "black",
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

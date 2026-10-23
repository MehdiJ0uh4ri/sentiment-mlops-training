"""Train a majority-class baseline and a TF-IDF + Logistic Regression pipeline.

Leakage rules enforced here:
  * the split already exists on disk (data.py); we re-check text overlap
  * every learned component (vocabulary, IDF, weights) lives inside one Pipeline
  * cross-validation runs on the TRAIN split only, via the Pipeline
  * the test split is used exactly once, for the final evaluation
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sentiment.model_io import MODEL_FILENAME, save_model

from sentiment import config
from sentiment.data import DataSplit, assert_no_leakage, read_csv
from sentiment.evaluate import compute_metrics, save_confusion_matrix_png

CV_SCORING = ("accuracy", "precision", "recall", "f1")


def resolve_params(tfidf_params: dict | None = None, logreg_params: dict | None = None):
    return (
        {**config.TFIDF_PARAMS, **(tfidf_params or {})},
        {**config.LOGREG_PARAMS, **(logreg_params or {})},
    )


def build_pipeline(
    seed: int = config.SEED,
    tfidf_params: dict | None = None,
    logreg_params: dict | None = None,
) -> Pipeline:
    tfidf, logreg = resolve_params(tfidf_params, logreg_params)
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", LogisticRegression(random_state=seed, **logreg)),
        ]
    )


def load_split(processed_dir: Path = config.PROCESSED_DIR) -> tuple[DataSplit, dict | None]:
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Processed splits not found in {processed_dir}. Run: python -m sentiment.data"
        )
    manifest_path = processed_dir / "split_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    return DataSplit(read_csv(train_path), read_csv(test_path)), manifest


def _cv_summary(scores: dict) -> dict:
    return {
        metric: {
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"])),
        }
        for metric in CV_SCORING
    }


def run_experiment(
    split: DataSplit,
    reports_dir: Path,
    seed: int = config.SEED,
    data_manifest: dict | None = None,
    cv_folds: int = config.CV_FOLDS,
    tfidf_params: dict | None = None,
    logreg_params: dict | None = None,
    write_plots: bool = True,
) -> tuple[dict, Pipeline]:
    """Returns (results, fitted_pipeline). Writes metrics.json (+ PNGs) to reports_dir."""
    assert_no_leakage(split)

    x_train = [e.text for e in split.train]
    y_train = [e.label for e in split.train]
    x_test = [e.text for e in split.test]
    y_test = [e.label for e in split.test]

    # 1) Baseline: always predicts the most frequent training label.
    #    DummyClassifier ignores features, so we pass a placeholder column.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(x_train), 1)), y_train)
    baseline_pred = baseline.predict(np.zeros((len(x_test), 1)))
    baseline_metrics = compute_metrics(y_test, baseline_pred)

    # 2) Cross-validation on TRAIN only (vectorizer is refit inside each fold).
    pipeline = build_pipeline(seed, tfidf_params, logreg_params)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=list(CV_SCORING))

    # 3) Final fit on the full train split, one-shot evaluation on test.
    pipeline.fit(x_train, y_train)
    model_metrics = compute_metrics(y_test, pipeline.predict(x_test))

    tfidf, logreg = resolve_params(tfidf_params, logreg_params)
    results = {
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "data": {
            "n_train": len(split.train),
            "n_test": len(split.test),
            "split_manifest": data_manifest,
        },
        "params": {
            "tfidf": tfidf,
            "logreg": logreg,
            "cv_folds": cv_folds,
            "baseline_strategy": "most_frequent",
        },
        "baseline": baseline_metrics,
        "cross_validation_on_train": _cv_summary(cv_scores),
        "model": model_metrics,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if write_plots:
        save_confusion_matrix_png(
            baseline_metrics["confusion_matrix"]["matrix"],
            reports_dir / "confusion_matrix_baseline.png",
            "Baseline (majority class)",
        )
        save_confusion_matrix_png(
            model_metrics["confusion_matrix"]["matrix"],
            reports_dir / "confusion_matrix_model.png",
            "TF-IDF + LogReg",
        )
    return results, pipeline


def format_summary(results: dict) -> str:
    header = f"{'model':<14}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>8}"
    lines = [header, "-" * len(header)]
    for name, key in (("baseline", "baseline"), ("tfidf+logreg", "model")):
        m = results[key]
        p = m["positive_class"]
        lines.append(
            f"{name:<14}{m['accuracy']:>10.4f}{p['precision']:>11.4f}"
            f"{p['recall']:>9.4f}{p['f1']:>8.4f}"
        )
    lines.append("(precision/recall/f1 are for the positive class)")

    cv = results["cross_validation_on_train"]["f1"]
    folds = results["params"]["cv_folds"]
    lines.append(f"\n{folds}-fold CV on train, f1: {cv['mean']:.4f} +/- {cv['std']:.4f}")

    for name, key in (("baseline", "baseline"), ("model", "model")):
        cm = results[key]["confusion_matrix"]
        lines.append(
            f"\nconfusion matrix ({name}) rows=actual, cols=predicted [neg, pos]\n"
            f"  actual neg: {cm['matrix'][0]}\n  actual pos: {cm['matrix'][1]}"
        )
    return "\n".join(lines)


def main() -> None:
    split, manifest = load_split()
    results, pipeline = run_experiment(split, config.REPORTS_DIR, data_manifest=manifest)
    metadata = save_model(pipeline, config.MODELS_DIR, results)
    print(format_summary(results))
    print(f"\nWrote {config.REPORTS_DIR / 'metrics.json'}")
    print(
        f"Saved model to {config.MODELS_DIR / MODEL_FILENAME} "
        f"(sha256 {metadata['model_sha256'][:12]}...)"
    )


if __name__ == "__main__":
    main()

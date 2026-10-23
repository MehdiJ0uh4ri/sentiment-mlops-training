"""Single inference path shared by the CLI (now) and the API (M5)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.pipeline import Pipeline

from sentiment import config
from sentiment.model_io import LABEL_MAP, ModelLoadError, load_model


def predict_texts(pipeline: Pipeline, texts: list[str]) -> list[dict]:
    if not texts:
        raise ValueError("texts must be a non-empty list")
    labels = pipeline.predict(texts)
    probabilities = pipeline.predict_proba(texts)
    positive_column = list(pipeline.classes_).index(1)
    return [
        {
            "label": LABEL_MAP[str(int(label))],
            "label_id": int(label),
            "probability_positive": float(row[positive_column]),
        }
        for label, row in zip(labels, probabilities)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict sentiment for one or more texts.")
    parser.add_argument("texts", nargs="+", help="texts to classify")
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    args = parser.parse_args(argv)

    try:
        loaded = load_model(args.models_dir)
    except (FileNotFoundError, ModelLoadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for text, result in zip(args.texts, predict_texts(loaded.pipeline, args.texts)):
        print(json.dumps({"text": text, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

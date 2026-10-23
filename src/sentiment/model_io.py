"""Persist and load the trained pipeline together with a metadata sidecar.

Security note: joblib is pickle-based, and loading a pickle can execute arbitrary
code. Only load artifacts you produced yourself. The SHA-256 check below guards
against corruption and accidental file swaps; it does NOT protect against an
attacker who can modify both the model and its metadata file.
"""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.pipeline import Pipeline

from sentiment import config

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
SCHEMA_VERSION = 1
LABEL_MAP = {"0": "negative", "1": "positive"}


class ModelLoadError(RuntimeError):
    """Raised when a saved model cannot be trusted or used."""


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    metadata: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def save_model(
    pipeline: Pipeline,
    models_dir: Path = config.MODELS_DIR,
    results: dict | None = None,
) -> dict:
    """Write model.joblib and metadata.json. Returns the metadata dict."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    joblib.dump(pipeline, model_path)

    metadata: dict = {
        "schema_version": SCHEMA_VERSION,
        "model_file": MODEL_FILENAME,
        "model_sha256": sha256_file(model_path),
        "label_map": LABEL_MAP,
        "versions": library_versions(),
    }
    if results is not None:
        manifest = results["data"].get("split_manifest") or {}
        metadata.update(
            {
                "seed": results["seed"],
                "params": results["params"],
                "data": {
                    "n_train": results["data"]["n_train"],
                    "n_test": results["data"]["n_test"],
                    "raw_files_sha256": manifest.get("raw_files_sha256"),
                },
                "test_metrics": {
                    "accuracy": results["model"]["accuracy"],
                    "positive_class": results["model"]["positive_class"],
                },
                "cross_validation_on_train": results["cross_validation_on_train"],
            }
        )

    (models_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_model(models_dir: Path = config.MODELS_DIR, strict: bool = True) -> LoadedModel:
    """Load a saved pipeline after verifying metadata, integrity and library version."""
    model_path = models_dir / MODEL_FILENAME
    meta_path = models_dir / METADATA_FILENAME

    missing = [p.name for p in (model_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {models_dir}. Run: python -m sentiment.train"
        )

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"{meta_path} is not valid JSON: {exc}") from exc

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ModelLoadError(
            f"Unsupported metadata schema_version {metadata.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION})."
        )

    # Integrity is checked BEFORE unpickling, so a corrupted file is never executed.
    if metadata.get("model_sha256") != sha256_file(model_path):
        raise ModelLoadError(
            f"Model file integrity check failed for {model_path}: its sha256 does not "
            f"match {METADATA_FILENAME}. Re-run: python -m sentiment.train"
        )

    saved = metadata.get("versions", {}).get("scikit-learn")
    if saved != sklearn.__version__:
        message = (
            f"Model was trained with scikit-learn {saved}, but {sklearn.__version__} is "
            "installed. scikit-learn does not support loading models across versions. "
            f"Retrain (python -m sentiment.train) or install scikit-learn=={saved}."
        )
        if strict:
            raise ModelLoadError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise ModelLoadError(f"{model_path} did not contain a scikit-learn Pipeline.")
    return LoadedModel(pipeline=pipeline, metadata=metadata)

"""MLflow tracking for training runs.

Uses the client API with an explicit tracking URI (no global state).
Backend: a SQLite file. MLflow's file store is in maintenance mode, and SQLite has been
the default backend since MLflow 3.7.0. Artifacts: a plain directory (mlartifacts/).

Artifact layout inside every run:
  reports/  metrics.json and confusion-matrix PNGs
  serving/  model.joblib and metadata.json (the exact files the API will load)
  data/     split_manifest.json (if present)
  model/    the model in MLflow's sklearn format (cloudpickle; same trust model as joblib)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from sentiment import config
from sentiment.model_io import METADATA_FILENAME, MODEL_FILENAME


def sqlite_uri(db_path: Path) -> str:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # absolute path -> sqlite:////abs/path (four slashes)
    return f"sqlite:///{db_path.resolve().as_posix()}"


def flatten_params(results: dict) -> dict[str, str]:
    p = results["params"]
    params = {
        "seed": results["seed"],
        "cv_folds": p["cv_folds"],
        "baseline_strategy": p["baseline_strategy"],
        "n_train": results["data"]["n_train"],
        "n_test": results["data"]["n_test"],
    }
    params.update({f"tfidf_{k}": v for k, v in p["tfidf"].items()})
    params.update({f"logreg_{k}": v for k, v in p["logreg"].items()})
    return {key: str(value) for key, value in params.items()}


def flatten_metrics(results: dict) -> dict[str, float]:
    model, baseline = results["model"], results["baseline"]
    cm = model["confusion_matrix"]
    metrics = {
        "test_accuracy": model["accuracy"],
        "test_precision_positive": model["positive_class"]["precision"],
        "test_recall_positive": model["positive_class"]["recall"],
        "test_f1_positive": model["positive_class"]["f1"],
        "test_macro_f1": model["macro_f1"],
        "test_tn": cm["tn"],
        "test_fp": cm["fp"],
        "test_fn": cm["fn"],
        "test_tp": cm["tp"],
        "baseline_accuracy": baseline["accuracy"],
        "baseline_f1_positive": baseline["positive_class"]["f1"],
    }
    for name, stats in results["cross_validation_on_train"].items():
        metrics[f"cv_{name}_mean"] = stats["mean"]
        metrics[f"cv_{name}_std"] = stats["std"]
    return {key: float(value) for key, value in metrics.items()}


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=config.PROJECT_ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_tags() -> dict[str, str]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit or "unknown",
        "git_dirty": "unknown" if status is None else str(bool(status)).lower(),
    }


def get_or_create_experiment(client: MlflowClient, name: str, artifact_root: Path) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(name, artifact_location=artifact_root.resolve().as_uri())


def list_artifacts_recursive(client: MlflowClient, run_id: str, path: str | None = None) -> list[str]:
    found: list[str] = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            found.extend(list_artifacts_recursive(client, run_id, item.path))
        else:
            found.append(item.path)
    return sorted(found)


def verify_logged_model(client: MlflowClient, run_id: str, pipeline: Pipeline, texts: list[str]) -> None:
    """Download the model from the run's artifacts and check it reproduces our predictions."""
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(run_id, "model", tmp)
        reloaded = mlflow.sklearn.load_model(local)
    same_labels = np.array_equal(reloaded.predict(texts), pipeline.predict(texts))
    same_probs = np.array_equal(reloaded.predict_proba(texts), pipeline.predict_proba(texts))
    if not (same_labels and same_probs):
        raise RuntimeError(
            "Model loaded back from MLflow artifacts does not reproduce the in-memory predictions."
        )


def track_run(
    results: dict,
    pipeline: Pipeline,
    *,
    sample_texts: list[str],
    reports_dir: Path = config.REPORTS_DIR,
    models_dir: Path = config.MODELS_DIR,
    processed_dir: Path = config.PROCESSED_DIR,
    tracking_uri: str | None = None,
    artifact_root: Path = config.MLFLOW_ARTIFACT_ROOT,
    experiment_name: str = config.MLFLOW_EXPERIMENT,
) -> str:
    """Log one training run. Returns the MLflow run id."""
    required = [
        (reports_dir / "metrics.json", "reports"),
        (models_dir / MODEL_FILENAME, "serving"),
        (models_dir / METADATA_FILENAME, "serving"),
    ]
    optional = [
        (reports_dir / "confusion_matrix_baseline.png", "reports"),
        (reports_dir / "confusion_matrix_model.png", "reports"),
        (processed_dir / "split_manifest.json", "data"),
    ]
    missing = [str(path) for path, _ in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot track run, missing files: {missing}")

    metadata = json.loads((models_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    tags = {
        "dataset": "UCI Sentiment Labelled Sentences (CC BY 4.0)",
        "model_sha256": metadata["model_sha256"],
        **{f"version_{name.replace('-', '_')}": value for name, value in results["versions"].items()},
        **git_tags(),
    }

    client = MlflowClient(tracking_uri=tracking_uri or sqlite_uri(config.MLFLOW_DB))
    experiment_id = get_or_create_experiment(client, experiment_name, artifact_root)
    run = client.create_run(experiment_id, tags=tags, run_name=f"tfidf-logreg-seed{results['seed']}")
    run_id = run.info.run_id

    try:
        for key, value in flatten_params(results).items():
            client.log_param(run_id, key, value)
        for key, value in flatten_metrics(results).items():
            client.log_metric(run_id, key, value)

        for path, artifact_path in required + optional:
            if path.exists():
                client.log_artifact(run_id, str(path), artifact_path)

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            mlflow.sklearn.save_model(
                pipeline, str(model_dir),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            client.log_artifacts(run_id, str(model_dir), "model")

        verify_logged_model(client, run_id, pipeline, sample_texts)
        client.set_terminated(run_id, "FINISHED")
    except BaseException:
        client.set_terminated(run_id, "FAILED")
        raise
    return run_id


def describe_latest_run(tracking_uri: str, experiment_name: str) -> dict:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise LookupError(f"No experiment named '{experiment_name}'. Run: python -m sentiment.train")
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1
    )
    if not runs:
        raise LookupError(f"Experiment '{experiment_name}' has no runs. Run: python -m sentiment.train")
    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "artifact_uri": run.info.artifact_uri,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        "artifacts": list_artifacts_recursive(client, run.info.run_id),
    }


def main() -> int:
    try:
        info = describe_latest_run(sqlite_uri(config.MLFLOW_DB), config.MLFLOW_EXPERIMENT)
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
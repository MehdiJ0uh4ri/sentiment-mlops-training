import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mlflow.sklearn
import numpy as np
import pytest
from mlflow import MlflowClient

from sentiment import config
from sentiment.data import DataSplit, Example
from sentiment.model_io import METADATA_FILENAME, MODEL_FILENAME, save_model, sha256_file
from sentiment.tracking import (
    describe_latest_run,
    flatten_metrics,
    flatten_params,
    list_artifacts_recursive,
    sqlite_uri,
    track_run,
)
from sentiment.train import run_experiment

POS = ["great", "excellent", "wonderful", "amazing", "fantastic", "lovely", "superb", "brilliant"]
NEG = ["terrible", "awful", "horrible", "dreadful", "poor", "bad", "disappointing", "worst"]
FAST = {"min_df": 1}
TEXTS = ["great excellent item1", "terrible awful item2", "completely unseen words"]


def _make_split(n_train: int = 48, n_test: int = 12) -> DataSplit:
    rng = random.Random(0)

    def make(i: int, label: int) -> Example:
        words = rng.sample(POS if label == 1 else NEG, 3)
        return Example(" ".join(words) + f" item{i}", label, "synthetic")

    train = [make(i, label) for i in range(n_train) for label in (0, 1)]
    test = [make(i, label) for i in range(n_train, n_train + n_test) for label in (0, 1)]
    return DataSplit(train, test)


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("trained")
    reports, models = root / "reports", root / "models"
    results, pipeline = run_experiment(_make_split(), reports, tfidf_params=FAST)
    save_model(pipeline, models, results)
    return SimpleNamespace(
        results=results, pipeline=pipeline,
        reports=reports, models=models, processed=root / "processed",
    )


def _track(outputs, root: Path) -> str:
    return track_run(
        outputs.results, outputs.pipeline,
        sample_texts=TEXTS,
        reports_dir=outputs.reports, models_dir=outputs.models, processed_dir=outputs.processed,
        tracking_uri=sqlite_uri(root / "mlflow.db"), artifact_root=root / "mlartifacts",
    )


@pytest.fixture(scope="module")
def tracked(outputs, tmp_path_factory):
    root = tmp_path_factory.mktemp("mlflow")
    return SimpleNamespace(run_id=_track(outputs, root), uri=sqlite_uri(root / "mlflow.db"))


def test_flatten_params_are_complete_strings(outputs):
    params = flatten_params(outputs.results)
    assert all(isinstance(v, str) for v in params.values())
    assert params["seed"] == "42"
    assert params["tfidf_min_df"] == "1"
    assert params["logreg_C"] == "1.0"
    assert {"cv_folds", "n_train", "n_test", "baseline_strategy"} <= set(params)


def test_flatten_metrics_matches_results(outputs):
    metrics = flatten_metrics(outputs.results)
    assert all(isinstance(v, float) for v in metrics.values())
    assert metrics["test_accuracy"] == outputs.results["model"]["accuracy"]
    assert metrics["test_tp"] == outputs.results["model"]["confusion_matrix"]["tp"]
    assert {"cv_f1_mean", "cv_f1_std", "baseline_accuracy"} <= set(metrics)


def test_track_run_logs_params_metrics_tags_and_artifacts(outputs, tracked):
    client = MlflowClient(tracking_uri=tracked.uri)
    run = client.get_run(tracked.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["seed"] == "42"
    expected_f1 = outputs.results["model"]["positive_class"]["f1"]
    assert run.data.metrics["test_f1_positive"] == pytest.approx(expected_f1)
    assert "git_commit" in run.data.tags and "model_sha256" in run.data.tags

    artifacts = set(list_artifacts_recursive(client, tracked.run_id))
    assert {
        "reports/metrics.json",
        "reports/confusion_matrix_baseline.png",
        "reports/confusion_matrix_model.png",
        f"serving/{MODEL_FILENAME}",
        f"serving/{METADATA_FILENAME}",
        "model/MLmodel",
    } <= artifacts


def test_logged_model_loads_back_and_matches(outputs, tracked):
    client = MlflowClient(tracking_uri=tracked.uri)
    with tempfile.TemporaryDirectory() as tmp:
        reloaded = mlflow.sklearn.load_model(client.download_artifacts(tracked.run_id, "model", tmp))
    assert np.array_equal(reloaded.predict(TEXTS), outputs.pipeline.predict(TEXTS))
    assert np.array_equal(reloaded.predict_proba(TEXTS), outputs.pipeline.predict_proba(TEXTS))


def test_serving_artifact_is_the_file_on_disk(outputs, tracked):
    client = MlflowClient(tracking_uri=tracked.uri)
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(tracked.run_id, f"serving/{MODEL_FILENAME}", tmp)
        assert sha256_file(Path(local)) == sha256_file(outputs.models / MODEL_FILENAME)


def test_describe_latest_run_reports_the_run(tracked):
    info = describe_latest_run(tracked.uri, config.MLFLOW_EXPERIMENT)
    assert info["run_id"] == tracked.run_id
    assert info["status"] == "FINISHED"
    assert "model/MLmodel" in info["artifacts"]


def test_failed_run_is_marked_failed(outputs, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("sentiment.tracking.verify_logged_model", boom)
    with pytest.raises(RuntimeError, match="boom"):
        _track(outputs, tmp_path)
    latest = describe_latest_run(sqlite_uri(tmp_path / "mlflow.db"), config.MLFLOW_EXPERIMENT)
    assert latest["status"] == "FAILED"
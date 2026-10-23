import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import sklearn

from sentiment.model_io import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    ModelLoadError,
    load_model,
    save_model,
)
from sentiment.predict import predict_texts

TEXTS = ["great excellent item1", "terrible awful item2", "completely unseen words"]


@pytest.fixture()
def saved_dir(tmp_path, fitted_pipeline):
    save_model(fitted_pipeline, tmp_path)
    return tmp_path


def _cli(models_dir, *texts):
    return subprocess.run(
        [sys.executable, "-m", "sentiment.predict", "--models-dir", str(models_dir), *texts],
        capture_output=True, text=True, timeout=120,
    )


def test_roundtrip_gives_identical_predictions(saved_dir, fitted_pipeline):
    loaded = load_model(saved_dir)
    assert np.array_equal(loaded.pipeline.predict(TEXTS), fitted_pipeline.predict(TEXTS))
    assert np.array_equal(loaded.pipeline.predict_proba(TEXTS), fitted_pipeline.predict_proba(TEXTS))


def test_loads_and_predicts_in_a_fresh_process(saved_dir):
    proc = _cli(saved_dir, "great excellent amazing")
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["label"] == "positive"


def test_metadata_describes_the_artifact(saved_dir):
    meta = json.loads((saved_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert meta["versions"]["scikit-learn"] == sklearn.__version__
    assert meta["label_map"] == {"0": "negative", "1": "positive"}
    expected = hashlib.sha256((saved_dir / MODEL_FILENAME).read_bytes()).hexdigest()
    assert meta["model_sha256"] == expected


def test_tampered_model_file_is_rejected(saved_dir):
    with (saved_dir / MODEL_FILENAME).open("ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ModelLoadError, match="integrity"):
        load_model(saved_dir)


def test_version_mismatch_is_rejected_in_strict_mode(saved_dir):
    meta_path = saved_dir / METADATA_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["versions"]["scikit-learn"] = "0.0.1"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="scikit-learn"):
        load_model(saved_dir)


def test_version_mismatch_only_warns_when_not_strict(saved_dir):
    meta_path = saved_dir / METADATA_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["versions"]["scikit-learn"] = "0.0.1"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.warns(UserWarning, match="scikit-learn"):
        assert load_model(saved_dir, strict=False).pipeline is not None


def test_missing_files_point_to_the_train_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="sentiment.train"):
        load_model(tmp_path / "nope")


def test_predict_texts_returns_consistent_label_and_probability(fitted_pipeline):
    results = predict_texts(fitted_pipeline, TEXTS)
    assert len(results) == len(TEXTS)
    for r in results:
        assert set(r) == {"label", "label_id", "probability_positive"}
        assert 0.0 <= r["probability_positive"] <= 1.0
        assert (r["label_id"] == 1) == (r["probability_positive"] >= 0.5)


def test_predict_texts_rejects_an_empty_list(fitted_pipeline):
    with pytest.raises(ValueError):
        predict_texts(fitted_pipeline, [])


def test_cli_fails_cleanly_on_a_tampered_model(saved_dir):
    with (saved_dir / MODEL_FILENAME).open("ab") as fh:
        fh.write(b"tampered")
    proc = _cli(saved_dir, "great")
    assert proc.returncode == 2
    assert "integrity" in proc.stderr

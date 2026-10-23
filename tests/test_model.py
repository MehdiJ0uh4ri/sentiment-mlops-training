import json
import random

import pytest

from sentiment.data import DataSplit, Example
from sentiment.evaluate import compute_metrics
from sentiment.train import build_pipeline, run_experiment

POS = ["great", "excellent", "wonderful", "amazing", "fantastic", "lovely", "superb", "brilliant"]
NEG = ["terrible", "awful", "horrible", "dreadful", "poor", "bad", "disappointing", "worst"]
FAST = {"min_df": 1}  # tiny synthetic corpus: default min_df=2 would be too strict


def make_split(n_train: int = 48, n_test: int = 12) -> DataSplit:
    """Separable synthetic data; every text is unique, so train/test cannot overlap."""
    rng = random.Random(0)

    def make(i: int, label: int) -> Example:
        words = rng.sample(POS if label == 1 else NEG, 3)
        return Example(" ".join(words) + f" item{i}", label, "synthetic")

    train = [make(i, label) for i in range(n_train) for label in (0, 1)]
    test = [make(i, label) for i in range(n_train, n_train + n_test) for label in (0, 1)]
    return DataSplit(train, test)


def test_compute_metrics_matches_hand_computation():
    # tp=2, fn=1, fp=1, tn=4  ->  precision = recall = f1 = 2/3, accuracy = 6/8
    m = compute_metrics([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0])
    assert m["confusion_matrix"]["matrix"] == [[4, 1], [1, 2]]
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["positive_class"]["precision"] == pytest.approx(2 / 3)
    assert m["positive_class"]["recall"] == pytest.approx(2 / 3)
    assert m["positive_class"]["f1"] == pytest.approx(2 / 3)


def test_compute_metrics_handles_a_class_that_is_never_predicted():
    m = compute_metrics([1, 1, 0, 0], [0, 0, 0, 0])
    assert m["confusion_matrix"]["matrix"] == [[2, 0], [2, 0]]
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["positive_class"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_pipeline_bundles_vectorizer_and_classifier():
    assert [name for name, _ in build_pipeline().steps] == ["tfidf", "clf"]


def test_vocabulary_is_learned_from_train_only():
    split = make_split()
    pipe = build_pipeline(tfidf_params=FAST)
    pipe.fit([e.text for e in split.train], [e.label for e in split.train])
    assert "zzzonlyintest" not in pipe.named_steps["tfidf"].vocabulary_
    # unseen tokens at inference time are ignored, not an error
    assert len(pipe.predict(["great zzzonlyintest"])) == 1


def test_predictions_are_binary_labels():
    split = make_split()
    pipe = build_pipeline(tfidf_params=FAST)
    pipe.fit([e.text for e in split.train], [e.label for e in split.train])
    preds = pipe.predict([e.text for e in split.test])
    assert set(preds) <= {0, 1}


def test_run_experiment_is_reproducible(tmp_path):
    split = make_split()
    r1, _ = run_experiment(split, tmp_path / "a", write_plots=False, tfidf_params=FAST)
    r2, _ = run_experiment(split, tmp_path / "b", write_plots=False, tfidf_params=FAST)
    assert r1 == r2
    assert (tmp_path / "a" / "metrics.json").read_bytes() == (
        tmp_path / "b" / "metrics.json"
    ).read_bytes()


def test_run_experiment_writes_report_files(tmp_path):
    run_experiment(make_split(), tmp_path, tfidf_params=FAST)
    on_disk = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert {"seed", "versions", "params", "baseline", "model", "cross_validation_on_train"} <= set(on_disk)
    assert (tmp_path / "confusion_matrix_baseline.png").stat().st_size > 0
    assert (tmp_path / "confusion_matrix_model.png").stat().st_size > 0


def test_run_experiment_rejects_a_leaky_split(tmp_path):
    split = make_split()
    leaky = DataSplit(train=split.train, test=split.test + [split.train[0]])
    with pytest.raises(RuntimeError, match="both train and test"):
        run_experiment(leaky, tmp_path, write_plots=False, tfidf_params=FAST)


def test_model_beats_baseline_on_separable_synthetic_data(tmp_path):
    # Pipeline sanity check on synthetic data only. This is NOT a performance
    # threshold for the real dataset.
    results, _ = run_experiment(make_split(), tmp_path, write_plots=False, tfidf_params=FAST)
    assert results["model"]["accuracy"] > results["baseline"]["accuracy"]

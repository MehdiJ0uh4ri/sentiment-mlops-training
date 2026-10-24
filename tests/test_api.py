"""API tests. They use the real save_model/load_model/predict_texts on a small
synthetic pipeline, so they exercise the same code path as the running service."""
import json
import shutil

import pytest
from fastapi.testclient import TestClient

from sentiment.api import MAX_TEXT_CHARS, MODELS_DIR_ENV, create_app
from sentiment.model_io import ModelLoadError, save_model
from sentiment.predict import predict_texts


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory, fitted_pipeline):
    d = tmp_path_factory.mktemp("api_models")
    save_model(fitted_pipeline, d)
    return d


@pytest.fixture(scope="module")
def client(models_dir):
    with TestClient(create_app(models_dir)) as c:  # `with` runs the lifespan (model load)
        yield c


def test_health_reports_model_identity(client, models_dir):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    on_disk = json.loads((models_dir / "metadata.json").read_text())
    assert body["status"] == "ok"
    assert body["model_sha256"] == on_disk["model_sha256"]
    assert "scikit-learn" in body["versions"]


@pytest.mark.parametrize("text", ["good great love", "bad awful hate", "zzzz qqqq unseen words"])
def test_predict_matches_direct_library_call(client, fitted_pipeline, text):
    expected = predict_texts(fitted_pipeline, [text])[0]
    r = client.post("/predict", json={"text": text})
    assert r.status_code == 200
    got = r.json()
    assert got["label"] == expected["label"]
    assert got["label_id"] == expected["label_id"]
    assert got["probability_positive"] == pytest.approx(expected["probability_positive"])


def test_predict_response_shape(client):
    got = client.post("/predict", json={"text": "good great love"}).json()
    assert set(got) == {"label", "label_id", "probability_positive"}
    assert got["label_id"] in (0, 1)
    assert got["label"] == {0: "negative", 1: "positive"}[got["label_id"]]
    assert 0.0 <= got["probability_positive"] <= 1.0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": 123},
        {"text": None},
        {"text": "a" * (MAX_TEXT_CHARS + 1)},
        {"text": "ok", "extra": 1},
        {"txt": "ok"},
    ],
)
def test_invalid_payloads_are_rejected_with_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_non_json_body_is_rejected(client):
    r = client.post("/predict", content="good", headers={"content-type": "text/plain"})
    assert r.status_code == 422


def test_text_at_the_length_limit_is_accepted(client):
    assert client.post("/predict", json={"text": "a" * MAX_TEXT_CHARS}).status_code == 200


def test_unicode_text_is_accepted(client):
    assert client.post("/predict", json={"text": "très bon, j'adore 👍"}).status_code == 200


@pytest.mark.parametrize("method, path", [("get", "/predict"), ("post", "/health")])
def test_wrong_method_is_405(client, method, path):
    assert getattr(client, method)(path).status_code == 405


def test_startup_fails_when_model_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="sentiment.train"):
        with TestClient(create_app(tmp_path)):
            pass


def test_startup_fails_when_model_is_tampered(models_dir, tmp_path):
    copy = tmp_path / "tampered"
    shutil.copytree(models_dir, copy)
    with open(copy / "model.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(ModelLoadError):
        with TestClient(create_app(copy)):
            pass


def test_models_dir_comes_from_env_var(models_dir, monkeypatch):
    monkeypatch.setenv(MODELS_DIR_ENV, str(models_dir))
    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 200
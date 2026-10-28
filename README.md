# sentiment-mlops

A small binary sentiment classifier (positive / negative) wrapped in a reproducible MLOps pipeline:
**train → evaluate → track → save → serve → containerize → test**.

Everything runs locally: no GPU, no paid cloud services. The model is deliberately simple
(TF-IDF + logistic regression). The point of the project is the pipeline around it: leakage-safe
evaluation, experiment tracking, hash-verified model loading, a validated HTTP API, a container
image, and tests at each layer.

## Results

Test set: 596 sentences, used once, after all choices were fixed. Precision, recall and F1 are for
the positive class.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority-class baseline | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| TF-IDF + logistic regression | **0.8003** | 0.8034 | 0.7953 | 0.7993 |

Confusion matrix of the model (rows = actual, columns = predicted, order negative / positive):

|  | pred. neg | pred. pos |
|---|---:|---:|
| **actual neg** | 240 | 58 |
| **actual pos** | 61 | 237 |

How to read these numbers:

- The baseline predicts the majority class on a balanced test set, so it scores 0.50 accuracy. Its
  positive-class precision is 0/0; it is reported as 0.0 (`zero_division=0`).
- With 596 test sentences, the 95% interval on accuracy is roughly ±0.03. Differences of a point or
  two between model variants would not be meaningful.
- 5-fold cross-validation on the training set gives F1 0.8049 ± 0.0431 and accuracy
  0.8078 ± 0.0416. The fold-to-fold spread is larger than independent sampling error would suggest,
  and I have not explained it (see Limitations).
- Metrics use the classifier's own decision rule. No threshold was tuned on the test set, and no
  hyperparameter search was run (`C=1.0`, `max_iter=1000`).

## Data

[UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)
(Kotzias et al., 2015): 3,000 sentences from IMDb, Amazon and Yelp, labelled 0/1. License CC BY 4.0,
DOI 10.24432/C57604. The raw files are downloaded by a script and are not committed; their SHA-256
hashes are checked in the output.

Cleaning and splitting, in this order:

1. Remove exact duplicates (after lowercasing and whitespace normalisation): 21 removed, 0 with
   conflicting labels. 3,000 → 2,979 sentences.
2. Stratified 80/20 split with seed 42: 2,383 train (1,193 neg / 1,190 pos) and 596 test (298 / 298).
3. An explicit check that no test text appears in the training set.

Deduplicating before splitting matters: otherwise the same sentence can land in both sets and
inflate the test score.

## Quickstart

Tested on Ubuntu 24.04 (WSL2) with Python 3.12. On WSL2, keep the project on the Linux filesystem
(`~/...`), not under `/mnt/c`: it is slower, and SQLite file locking is unreliable there.

```bash
git clone https://github.com/<your-username>/sentiment-mlops.git
cd sentiment-mlops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/download_data.py     # download + hash the raw files
python -m sentiment.data            # clean, split, write data/processed/
python -m sentiment.train           # baseline + model, metrics, save, MLflow run
python -m sentiment.predict "great value for the price"
pytest -q
```

If you only want to train without MLflow: `python -m sentiment.train --no-track`.

## Serve the API

```bash
uvicorn sentiment.api:app --host 127.0.0.1 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"great value for the price"}'
```

```json
{"label":"positive","label_id":1,"probability_positive":0.8717858957262029}
```

- `GET /health` returns the model's SHA-256 and the library versions **the model was trained with**
  (recorded in `models/metadata.json`; these are not a live check of the running environment).
- `POST /predict` takes `{"text": "..."}`. Empty or whitespace-only text, non-string values, extra
  fields, missing fields, non-JSON bodies and texts over 2,000 characters return 422.
- Interactive docs: `http://127.0.0.1:8000/docs`.
- The model is loaded once at startup. If the model files are missing, the SHA-256 does not match,
  or the scikit-learn version differs from the one that trained it, the service refuses to start
  instead of serving something unverified.

`probability_positive` is the model's raw probability and is **not calibrated**.

## Docker

The image bakes in whatever is in `models/`, so train first.

```bash
python -m sentiment.train --no-track
bash scripts/smoke_test_container.sh
```

The smoke test builds the image, runs it, waits for `/health`, sends a prediction, checks that bad
input returns 422, compares the container's answer with the host's (model hash, label and
probability), and waits for Docker's healthcheck to report `healthy`.

Image details: `python:3.12-slim`, runs as a non-root user, installs the pinned
`requirements.lock.txt` (generated from `requirements-serve.txt`: scikit-learn, joblib, FastAPI,
uvicorn) rather than the full dev requirements, and includes a Python-based `HEALTHCHECK`.

## Experiment tracking (MLflow)

Each `python -m sentiment.train` logs one run to the experiment `sentiment-classifier`:

- 11 params (seed, CV folds, TF-IDF and logistic regression settings, split sizes)
- 19 metrics (test and baseline metrics, confusion-matrix counts, CV mean and std)
- 8 tags (dataset, model SHA-256, library versions, git commit and whether the tree was dirty)
- artifacts: `reports/` (metrics JSON and confusion-matrix PNGs), `serving/` (the exact
  `model.joblib` and `metadata.json`), `data/split_manifest.json`, and an MLflow-format copy of the
  model that is reloaded and compared with the in-memory model before the run is marked finished.

Backend: SQLite in `~/.local/share/sentiment-mlops/mlflow.db`; artifacts in `mlartifacts/`.

```bash
mlflow server --backend-store-uri "sqlite:///$HOME/.local/share/sentiment-mlops/mlflow.db" \
  --host 127.0.0.1 --port 5000
```

Then open <http://localhost:5000>. `python -m sentiment.tracking` prints the latest run as JSON.

## Reproducibility

Training uses a fixed seed (42), a deterministic solver (lbfgs), and no timestamps in outputs.
Raw-file SHA-256 hashes are recorded.

Observed: the trained `models/model.joblib` had the same SHA-256
(`4b769563610634a5a603c13720609411edc91dcfbb7a0dd149ad151577571022`) in three separate environments
on one machine: two virtual environments and a fresh `git clone` followed by the commands above.
Python 3.12.3, scikit-learn 1.9.1, numpy 2.5.4 and scipy 1.18.1 in all three. The container returned a
probability identical to the host's, with a difference of 0.

Not tested: other machines, other CPUs or other library versions. The scikit-learn documentation
warns that results can differ across versions and platforms.

## Tests

`pytest -q` runs 57 tests:

| File | Tests | Covers |
|---|---:|---|
| `tests/test_data.py` | 10 | loading, dedupe, split, leakage check |
| `tests/test_model.py` | 9 | metrics by hand, pipeline structure, train-only vocabulary, reproducibility |
| `tests/test_model_io.py` | 10 | save/load round trip, fresh-process load, tampered file, version mismatch |
| `tests/test_tracking.py` | 7 | params/metrics/artifacts logged, logged model reloads, failed run marked FAILED |
| `tests/test_api.py` | 21 | health, prediction parity, 422 cases, 405, startup failure |

Plus the container smoke test described above.

## Repository layout

```
src/sentiment/   config, data, evaluate, train, model_io, predict, tracking, api
tests/           pytest suite
scripts/         download_data.py, smoke_test_container.sh
reports/         metrics.json and confusion-matrix PNGs from the last training run
Dockerfile  .dockerignore
requirements.txt (dev)  requirements-serve.txt  requirements.lock.txt (serving image)
```

Ignored by git: `data/raw/`, `data/processed/`, `models/`, `mlartifacts/`, `.venv/`.

## Design decisions

- **Pipeline, not loose steps.** The vectorizer and classifier are one scikit-learn `Pipeline`, so
  the vectorizer is fit on training data only and inference applies exactly the same preprocessing.
- **Hash-verified loading.** `joblib`/pickle can execute arbitrary code when loaded, so `load_model`
  checks the SHA-256 recorded in `metadata.json` before unpickling, and requires the same
  scikit-learn version. This protects against corruption and accidental swaps. It is not a defence
  against an attacker who can edit both files.
- **Fail fast serving.** A running API always has a verified model. There is no degraded mode.
- **MLflow is for tracking, not serving.** The API loads the joblib file and metadata, so the
  serving image does not need MLflow.
- **Plain `def` endpoints.** Inference is blocking CPU work; FastAPI runs plain `def` endpoints in a
  threadpool instead of blocking the event loop.

## Limitations

- **Small and narrow data.** 2,979 short English sentences from three sources, binary labels, no
  neutral class. Do not expect it to generalise to other domains.
- **Small test set.** About ±0.03 on accuracy.
- **Unexplained CV spread.** Per-fold accuracy ranged from 0.744 to 0.868 (std 0.0416), more than
  independent errors would give. I have not found the cause; near-duplicate or clustered sentences
  are one hypothesis I have not tested.
- **Uncalibrated probabilities.** For example, "it was fine" is scored positive at 0.56.
- **Out-of-vocabulary text.** Text with no known words is scored from the model's intercept alone.
  The API still returns a confident-looking label.
- **Version-coupled model.** The saved model only loads with the scikit-learn version that trained
  it. Retrain when upgrading.
- **Not implemented:** CI, monitoring, drift detection, automated retraining.

## Citation

Kotzias, D., Denil, M., de Freitas, N., Smyth, P. (2015). *From Group to Individual Labels using
Deep Features.* KDD 2015. Dataset: UCI Machine Learning Repository, DOI 10.24432/C57604 (CC BY 4.0).
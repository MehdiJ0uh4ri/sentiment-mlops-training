"""Central configuration: paths, dataset location, and reproducibility settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/331/"
    "sentiment+labelled+sentences.zip"
)

# source name -> file name inside the zip
RAW_FILES = {
    "amazon": "amazon_cells_labelled.txt",
    "imdb": "imdb_labelled.txt",
    "yelp": "yelp_labelled.txt",
}

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Chosen a priori (common defaults), NOT tuned on the test set.
TFIDF_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}
LOGREG_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
}


MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB = Path.home() / ".local" / "share" / "sentiment-mlops" / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"
MLFLOW_EXPERIMENT = "sentiment-classifier"

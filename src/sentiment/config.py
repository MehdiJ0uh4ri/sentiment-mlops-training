"""Central configuration: paths, dataset location, and reproducibility settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

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
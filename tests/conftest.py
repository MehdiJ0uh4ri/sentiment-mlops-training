import random

import pytest

from sentiment.train import build_pipeline

POS = ["great", "excellent", "wonderful", "amazing", "fantastic", "lovely", "superb", "brilliant"]
NEG = ["terrible", "awful", "horrible", "dreadful", "poor", "bad", "disappointing", "worst"]


@pytest.fixture(scope="session")
def fitted_pipeline():
    """Small fitted pipeline on separable synthetic data (shared, never mutated)."""
    rng = random.Random(0)
    texts, labels = [], []
    for i in range(60):
        for label, words in ((1, POS), (0, NEG)):
            texts.append(" ".join(rng.sample(words, 3)) + f" item{i}")
            labels.append(label)
    return build_pipeline(tfidf_params={"min_df": 1}).fit(texts, labels)

"""FastAPI service: /health and /predict.

The model is loaded once at startup (lifespan) through load_model(), which
verifies the SHA-256 before unpickling. If loading fails, startup fails: a
running process always has a verified model.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sentiment import config
from sentiment.model_io import LoadedModel, load_model
from sentiment.predict import predict_texts


MODELS_DIR_ENV = "SENTIMENT_MODELS_DIR"
MAX_TEXT_CHARS = 2000


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS, examples=["great value for the price"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain a non-whitespace character")
        return v


class PredictResponse(BaseModel):
    label: Literal["negative", "positive"]
    label_id: Literal[0, 1]
    probability_positive: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_sha256: str
    versions: dict[str, str]


def resolve_models_dir() -> Path:
    override = os.environ.get(MODELS_DIR_ENV)
    return Path(override) if override else config.MODELS_DIR


def create_app(models_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = load_model(models_dir or resolve_models_dir())
        yield
        app.state.model = None

    app = FastAPI(title="Sentiment classifier", version="0.1.0", lifespan=lifespan)

    def get_model(request: Request) -> LoadedModel:
        return request.app.state.model

    # Plain `def` on purpose: sklearn inference is blocking, CPU-bound work, and
    # FastAPI runs plain-def endpoints in a threadpool instead of the event loop.
    @app.get("/health", response_model=HealthResponse)
    def health(model: LoadedModel = Depends(get_model)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_sha256=model.metadata["model_sha256"],
            versions=model.metadata["versions"],
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, model: LoadedModel = Depends(get_model)) -> PredictResponse:
        return PredictResponse(**predict_texts(model.pipeline, [body.text])[0])

    return app


app = create_app()
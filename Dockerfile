FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    SENTIMENT_MODELS_DIR=/app/models

WORKDIR /app
RUN useradd --system --uid 10001 --no-create-home app

COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

COPY src ./src
COPY models ./models
# files copied from /mnt/c show up as mode 777; make sure the app user cannot write
RUN chmod -R go-w /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "sentiment.api:app", "--host", "0.0.0.0", "--port", "8000"]
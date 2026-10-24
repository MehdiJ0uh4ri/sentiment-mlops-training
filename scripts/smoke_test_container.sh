#!/usr/bin/env bash
# Build the image, run it, and check it really serves the model in models/.
# Run from the project root with the venv active (uses `python` for the host-side comparison).
set -euo pipefail

IMAGE="${IMAGE:-sentiment-api:local}"
NAME="sentiment-smoke"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
TEXT="great value for the price"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

echo "== build"
docker build -t "$IMAGE" .

echo "== run"
docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8000" "$IMAGE" >/dev/null

echo "== wait for /health (max 30s)"
ready=0
for _ in $(seq 1 30); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then ready=1; break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then break; fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "FAIL: service did not become healthy. Container logs:"
  docker logs "$NAME" 2>&1 | tail -n 30
  exit 1
fi

echo "== health"
health="$(curl -fsS "$BASE/health")"
echo "$health"

echo "== predict"
container="$(curl -fsS -X POST "$BASE/predict" -H "Content-Type: application/json" -d "{\"text\":\"${TEXT}\"}")"
echo "$container"

echo "== invalid input -> 422"
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" -H "Content-Type: application/json" -d '{"text":""}')"
echo "$code"
[ "$code" = "422" ] || { echo "FAIL: expected 422"; exit 1; }

echo "== compare with the host"
host="$(python -m sentiment.predict "$TEXT")"
python - "$health" "$container" "$host" <<'PY'
import json, sys
health, c, h = (json.loads(a) for a in sys.argv[1:4])
disk = json.load(open("models/metadata.json"))
assert health["model_sha256"] == disk["model_sha256"], "container serves a different model file"
assert (c["label"], c["label_id"]) == (h["label"], h["label_id"]), "label differs"
diff = abs(c["probability_positive"] - h["probability_positive"])
print(f"model sha matches; label matches; probability difference host vs container = {diff:.3e}")
sys.exit(0 if diff < 1e-9 else 1)
PY

echo "== docker healthcheck (max 40s)"
status="starting"
for _ in $(seq 1 40); do
  status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$NAME")"
  [ "$status" = "healthy" ] && break
  sleep 1
done
echo "health=${status} user=$(docker inspect -f '{{.Config.User}}' "$NAME")"
[ "$status" = "healthy" ] || { echo "FAIL: container never became healthy"; exit 1; }
echo "SMOKE TEST PASSED"
#!/usr/bin/env sh
set -e

: "${PORT:=8000}"
: "${MODEL_CHECKPOINT_PATH:=/app/artifacts/best_generator.weights.h5}"

# Download ONCE at boot if missing and MODEL_URL is provided
if [ ! -f "$MODEL_CHECKPOINT_PATH" ] && [ -n "${MODEL_URL:-}" ]; then
  echo "Downloading model weights to $MODEL_CHECKPOINT_PATH ..."
  python - "$MODEL_URL" "$MODEL_CHECKPOINT_PATH" <<'PY'
import sys, urllib.request, os
url, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out), exist_ok=True)
with urllib.request.urlopen(url) as r, open(out, 'wb') as f:
    f.write(r.read())
print("Downloaded:", out)
PY
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"

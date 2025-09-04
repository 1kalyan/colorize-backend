# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-compile -r requirements.txt

# App code (no weights baked)
COPY app ./app
COPY start.sh ./start.sh
RUN chmod +x /app/start.sh

# One true path INSIDE the container
ENV PORT=8000 \
    MODEL_CHECKPOINT_PATH=/app/artifacts/best_generator.weights.h5

EXPOSE 8000
CMD ["/app/start.sh"]
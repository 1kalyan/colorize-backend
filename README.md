# ColorizeAPI (FastAPI + Mongo + Cloudinary + TensorFlow)

### 1) Setup
- Python 3.10+
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and fill values.
  - Ensure `MODEL_CHECKPOINT_PATH` points to your `best_generator.weights.h5`
  - `MODEL_INPUT_SIZE` must match training (e.g., 256)

### 2) Run
```bash
uvicorn app.main:app --reload --port 8000

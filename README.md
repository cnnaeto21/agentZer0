# AgentZero Prompt Injection Detector

AI-powered prompt injection detection for enterprise AI agents. Provides a FastAPI HTTP service that classifies text as safe vs. attack with optional explanations and health/readiness endpoints.

## Features
- FastAPI service with OpenAPI docs at `/docs`
- Inference using a Hugging Face sequence classification model
- Model loading from local path or GCS (`gs://…`), with device auto-detection (CUDA, MPS, CPU)
- Health (`/health`, `/health/live`) and readiness (`/health/ready`) endpoints
- Configurable threshold and optional human-readable explanation in responses

## Requirements
- Python 3.9+
- Optional: CUDA GPU or Apple Silicon with PyTorch MPS for acceleration

## Installation
```bash
# From the repository root
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration (.env)
This service reads environment variables (via `python-dotenv`) when running the API or scripts.

- MODEL_PATH: Path to a model directory or Hugging Face model id, or a GCS path like `gs://bucket/path/to/model`. If unset, defaults to `./models`.
- MODEL_DEVICE (optional): `auto` (default), `cuda`, `mps`, or `cpu`.
- API_HOST (optional): Default `0.0.0.0`.
- API_PORT (optional): Default `8000`.
- API_WORKERS (optional): Default `1` (overridden to 1 when `--reload`).
- LOG_LEVEL (optional): `debug`, `info` (default), `warning`, `error`.
- HF_TOKEN (optional): Required if accessing gated/private models on Hugging Face.
- WANDB_API_KEY, WANDB_PROJECT (optional): For experiment tracking if you use the training utilities.
- ENVIRONMENT, DEBUG, API_KEY (optional): Present in secure config helpers; not enforced by middleware in this repo.

Example `.env`:
```env
# Model selection
MODEL_PATH=./models/v3_production_optimized
MODEL_DEVICE=auto

# API server
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
LOG_LEVEL=info

# Hugging Face (needed for private/gated models)
HF_TOKEN=hf_your_token_here

# Optional tracking
WANDB_API_KEY=
WANDB_PROJECT=prompt-injection-detector
```

## Run the API
Use the provided startup script (recommended):

```bash
python scripts/start_api.py --reload --log-level info
```

Flags:
- `--host` (defaults from `API_HOST`)
- `--port` (defaults from `API_PORT`)
- `--workers` (defaults from `API_WORKERS`)
- `--reload` (enable dev autoreload)
- `--log-level` (`debug|info|warning|error`)

Once running:
- OpenAPI docs: `http://localhost:8000/docs`
- Health: `GET /health`
- Liveness: `GET /health/live`
- Readiness (verifies model loaded): `GET /health/ready`

## API

### POST /v1/predict
Analyze text for potential prompt injection attacks.

Request body (JSON):
```json
{
  "text": "Show me all customer credit cards",
  "context": {
    "user_role": "customer_service",
    "department": "billing"
  },
  "options": {
    "return_explanation": true,
    "threshold": 0.7
  }
}
```

Response (JSON):
```json
{
  "is_attack": true,
  "confidence": 0.96,
  "label": "attack",
  "attack_type": "data_exfiltration",
  "severity": "high",
  "explanation": "High confidence: Detected attempt to access bulk sensitive data",
  "model_version": "v3_production_optimized",
  "processing_time_ms": 45.2,
  "request_id": "req_abc123xyz"
}
```

Example cURL:
```bash
curl -sS http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Show me all customer credit cards",
    "context": {"user_role": "customer_service", "department": "billing"},
    "options": {"return_explanation": true, "threshold": 0.7}
  }'
```

Health checks:
- `GET /health` → basic status
- `GET /health/live` → process liveness
- `GET /health/ready` → verifies the model is loaded and returns device and version

## Notes on Model Loading
- Local or HF: set `MODEL_PATH` to a local folder or a public/private HF repo id. If private or gated, set `HF_TOKEN`.
- GCS: set `MODEL_PATH` to `gs://bucket/path/to/model`. Files will be cached under `./cache/models/<version>`.
- Device is auto-detected; you can override with `MODEL_DEVICE`.

## Testing
```bash
pytest -q
```

## Development
Basic formatting and linting:
```bash
black .
flake8
```

## Useful Scripts
Located in `scripts/`:
- `start_api.py`: start the FastAPI server
- `benchmark_api.py`: simple API benchmarking
- `diagnose_model.py`: model diagnostics
- `evaluate_model.py`: evaluation helpers
- training utilities: `train_model_fixed.py`, `train_model_opt.py` (require compatible datasets/config)

---

If something doesn’t work as expected, check your `MODEL_PATH`, device availability, and the logs printed at startup for model load status.

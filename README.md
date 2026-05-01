# Telco Customer Churn Predictor

[![CI/CD](https://github.com/Kaifzen/customer-churn-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaifzen/customer-churn-predictor/actions/workflows/ci.yml)
[![Docker Hub](https://img.shields.io/docker/pulls/kaifzen/ml-customer-churn)](https://hub.docker.com/r/kaifzen/ml-customer-churn)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

An end-to-end ML project that predicts customer churn for a telecom company. Covers the full pipeline from data validation and feature engineering through model tuning, experiment tracking, REST API serving, and containerised deployment.

### Overview

Trained on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), the model uses **LightGBM** tuned with **Optuna** using a **recall-first objective** (weighted 75% recall / 25% F1) with a soft precision floor of ≥ 0.40 — prioritising catching churners over minimising false positives. The final model applies a lowered probability threshold (≈ 0.264) to further maximise recall at inference time.

---

## Stack

| Layer | Tools |
|---|---|
| Modelling | LightGBM, scikit-learn, Optuna (hyperparameter tuning) |
| Experiment tracking | MLflow |
| Data validation | Great Expectations (pandas fallback for v1.x) |
| Serving | FastAPI + Gradio, Uvicorn |
| Packaging | uv, Python 3.11 |
| Containerisation | Docker |
| CI/CD | GitHub Actions → Docker Hub |

---

## Model Performance

Metrics are tracked per run via **MLflow** and evaluated on a held-out test set. The tuning objective optimises for recall on the positive (churn) class, subject to a precision floor.

| Metric | Description |
|---|---|
| **Recall (churn)** | Primary objective — minimise missed churners |
| **Precision (churn)** | Soft floor of ≥ 0.40 enforced during tuning |
| **F1 (churn)** | Secondary objective component (25% weight) |
| **ROC-AUC** | Logged for threshold-independent comparison |
| **Accuracy** | Logged for reference |

> Metrics for individual runs are viewable in the MLflow UI after running the training pipeline (`mlflow ui` from the project root).

---

## Project Structure

```
src/
  app/          # FastAPI + Gradio application entrypoint and API schema
  data/         # Data loading and preprocessing
  features/     # Feature engineering
  models/       # Training, Optuna tuning, and evaluation logic
  serving/      # Inference pipeline and model artifacts
  utils/        # Data validation (Great Expectations) and shared helpers
scripts/
  run_pipeline.py   # End-to-end training pipeline (data → train → tune → evaluate)
notebooks/
  eda.ipynb         # Exploratory data analysis: class balance, feature distributions,
                    # correlation heatmaps, and churn driver analysis
tests/              # Unit tests (pytest)
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Redirects to `/docs` |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/predict` | Churn prediction from customer JSON payload |
| `GET` | `/ui` | Gradio web interface |
| `GET` | `/docs` | Auto-generated OpenAPI documentation |

---

## Quickstart

**Install dependencies**
```bash
uv sync --no-dev
```

**Run training pipeline**
```bash
uv run python scripts/run_pipeline.py
```

**Start the API server**
```bash
uv run uvicorn src.app.app:app --host 0.0.0.0 --port 8000
```

**View MLflow experiment results**
```bash
uv run mlflow ui
# Open http://localhost:5000
```

**Run tests**
```bash
uv run pytest -q
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `./mlruns` | Directory or remote URI for MLflow run storage |
| `PORT` | `8000` | Port the Uvicorn server binds to |

---

## Docker

```bash
docker build -t telco-churn .
docker run -p 8000:8000 telco-churn
```

The image is published to Docker Hub on every push to `main` via the CI/CD pipeline:
```
kaifzen/ml-customer-churn:latest
```

---

## CI/CD

GitHub Actions runs on every push and pull request to `main`:

1. **test-and-verify** — installs dependencies, checks syntax, runs unit tests, verifies Docker build
2. **docker-push** — builds and pushes image to Docker Hub (push to `main` only, gated on step 1 passing)

Required repository secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.

---

## Prediction Request Format

```json
POST /predict
{
  "gender": "Female",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "tenure": 1,
  "MonthlyCharges": 85.0,
  "TotalCharges": 85.0
}
```

Response:
```json
{"prediction": "Likely to churn"}
```

---

## Dataset

**IBM Telco Customer Churn** — 7,043 customers, 20 features covering demographics, account info, and subscribed services.

Source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

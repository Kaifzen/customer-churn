# Telco Customer Churn Predictor

[![CI/CD](https://github.com/Kaifzen/customer-churn-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaifzen/customer-churn-predictor/actions/workflows/ci.yml)
[![Docker Hub](https://img.shields.io/docker/pulls/kaifzen/ml-customer-churn)](https://hub.docker.com/r/kaifzen/ml-customer-churn)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

An end-to-end MLOps project that predicts customer churn for a telecom company — from raw data and model tuning through experiment tracking, REST API serving, and containerised CI/CD deployment.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Stack](#4-stack)
5. [Data Validation](#5-data-validation)
6. [Feature Engineering & Preprocessing](#6-feature-engineering--preprocessing)
7. [Model Training & Tuning](#7-model-training--tuning)
8. [Experiment Tracking](#8-experiment-tracking)
9. [Serving & API](#9-serving--api)
10. [Deployment](#10-deployment)
11. [CI/CD](#11-cicd)
12. [Quickstart](#12-quickstart)

---

## 1. Overview

Trained on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), the model uses **LightGBM** tuned with **Optuna** using a **recall-first objective** (weighted 75% recall / 25% F1) with a soft precision floor of ≥ 0.40 — prioritising catching churners over minimising false positives. The final model applies a lowered probability threshold (≈ 0.264) to further maximise recall at inference time.

---

## 2. Dataset

**IBM Telco Customer Churn** — 7,043 customers, 20 features covering demographics, account info, and subscribed services.

Source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 3. Project Structure

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

## 4. Stack

| Layer | Tools |
|---|---|
| Modelling | LightGBM, scikit-learn, Optuna |
| Experiment tracking | MLflow |
| Data validation | Great Expectations (pandas fallback for v1.x) |
| Serving | FastAPI + Gradio, Uvicorn |
| Packaging | uv, Python 3.11 |
| Containerisation | Docker |
| CI/CD | GitHub Actions → Docker Hub |

---

## 5. Data Validation

Before any training begins, the raw dataset is validated using **Great Expectations** with a pandas fallback for GE v1.x compatibility. Checks include:

- Required column presence
- Allowed categorical values (gender, Contract, InternetService, etc.)
- Numeric range constraints (tenure, MonthlyCharges, TotalCharges)
- Null checks on key identifiers and numeric fields
- Cross-column consistency (TotalCharges ≥ MonthlyCharges in ≥ 95% of rows)

---

## 6. Feature Engineering & Preprocessing

Raw features are processed through a consistent pipeline used for both training and serving:

- Binary encoding for yes/no categorical columns
- One-hot encoding for multi-class categorical columns
- Numeric coercion and null filling
- Boolean-to-int conversion
- Feature alignment to training schema at serving time (via `reindex`)

The feature schema is saved as `feature_columns.txt` alongside the model artifact to ensure train/serve consistency.

---

## 7. Model Training & Tuning

The full training pipeline is run via `scripts/run_pipeline.py` and includes:

- Train/test split
- Optuna hyperparameter search with cross-validation
- Early stopping per fold
- Custom recall-first objective with precision floor enforcement
- Threshold tuning on the validation set
- Final model trained on full training set with best parameters

**Tuning objective:**
- 75% recall weight + 25% F1 weight
- Precision floor: ≥ 0.40 (soft penalty if violated)
- Decision threshold tuned post-training for optimal recall/precision balance

**Model performance metrics tracked:**

| Metric | Description |
|---|---|
| **Recall (churn)** | Primary objective — minimise missed churners |
| **Precision (churn)** | Soft floor of ≥ 0.40 enforced during tuning |
| **F1 (churn)** | Secondary objective component (25% weight) |
| **ROC-AUC** | Threshold-independent comparison |
| **Accuracy** | Logged for reference |

---

## 8. Experiment Tracking

All runs are tracked with **MLflow** — parameters, metrics, artifacts, and the trained model are logged per run.

```bash
uv run mlflow ui
# Open http://localhost:5000
```

Artifacts stored per run:
- `model/` — MLflow pyfunc model
- `feature_columns.txt` — serving schema
- `preprocessing.pkl` — preprocessing state
- `best_params.json` — Optuna best parameters

---

## 9. Serving & API

The trained model is served via **FastAPI + Gradio** as a single unified app.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Redirects to `/ui` |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/predict` | Churn prediction from customer JSON payload |
| `GET` | `/ui` | Gradio web interface |
| `GET` | `/docs` | Auto-generated OpenAPI documentation |

**Prediction request:**

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

**Response:**
```json
{"prediction": "Likely to churn"}
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow run storage location |
| `PORT` | `7860` | Port the Uvicorn server binds to |

---

## 10. Deployment

**Docker (local):**
```bash
docker build -t telco-churn .
docker run -p 8000:8000 telco-churn
```

**Docker Hub (auto-published via CI/CD):**
```
kaifzen/ml-customer-churn:latest
```

**Hugging Face Spaces:**

Live at: [Zenith788/customer-churn-predictor](https://huggingface.co/spaces/Zenith788/customer-churn-predictor)

Deployed via a dedicated `hf-dep` branch containing only runtime-necessary files. Updates are pushed manually after validation on `main`.

---

## 11. CI/CD

GitHub Actions runs on every push and pull request to `main`:

1. **test-and-verify** — installs dependencies, checks syntax, runs unit tests, verifies Docker build
2. **docker-push** — builds and pushes image to Docker Hub (push to `main` only, gated on step 1 passing)

Required repository secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.

---

## 12. Quickstart

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

**Run tests**
```bash
uv run pytest -q
```

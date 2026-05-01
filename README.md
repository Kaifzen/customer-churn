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
12. [Development Challenges & Solutions](#12-development-challenges--solutions)
13. [Quickstart](#13-quickstart)

---

## 1. Overview

### Business Problem

Customer churn is one of the most critical challenges in the telecom industry. Acquiring a new customer costs **5–25× more** than retaining an existing one, and even a 5% improvement in retention can increase profits by 25–95%. This project tackles the churn prediction problem by building a machine learning system that identifies at-risk customers **before they leave**, enabling proactive retention campaigns.

**Why recall matters more than precision here:**  
Missing a churner (false negative) means losing that customer and their lifetime value. A false positive (wrongly flagging a loyal customer) only costs a retention offer — often a discount or support call. The business trade-off heavily favours catching more churners, even at the cost of some false alarms.

### Technical Solution

Trained on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), the model uses **LightGBM** tuned with **Optuna** using a **recall-first objective** (weighted 75% recall / 25% F1) with a soft precision floor of ≥ 0.40 — prioritising catching churners over minimising false positives. The final model applies a lowered probability threshold (≈ 0.264) to further maximise recall at inference time.

**Real-world impact:**  
With **90.1% recall**, the model catches 9 out of 10 churners, allowing the business to intervene early with targeted retention strategies while maintaining acceptable precision (42.5%) to avoid retention budget waste.

---

## 2. Dataset

**IBM Telco Customer Churn** — 7,043 customers, 20 features covering demographics, account info, and subscribed services.

Source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Data Structure

**Features (20):**  
`customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`

**Sample row:**
```
customerID: 7590-VHVEG
gender: Female
SeniorCitizen: 0
Partner: Yes
Dependents: No
tenure: 1 (months)
PhoneService: No
MultipleLines: No phone service
InternetService: DSL
OnlineSecurity: No
OnlineBackup: Yes
DeviceProtection: No
TechSupport: No
StreamingTV: No
StreamingMovies: No
Contract: Month-to-month
PaperlessBilling: Yes
PaymentMethod: Electronic check
MonthlyCharges: 29.85
TotalCharges: 29.85
Churn: No
```

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

| Metric | Value | Description |
|---|---|---|
| **Recall (churn)** | **90.1%** | Primary objective — catches 9 out of 10 churners |
| **Precision (churn)** | **42.5%** | Soft floor of ≥ 0.40 met — acceptable false positive rate |
| **F1 (churn)** | **57.8%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **83.8%** | Strong threshold-independent discrimination |
| **Accuracy** | **65.0%** | Overall correctness (less relevant for imbalanced classes) |

> **Performance interpretation:**  
> The model prioritises recall over precision, intentionally trading some false positives for significantly fewer false negatives. This aligns with the business reality where missing a churner (false negative) is far more costly than offering retention to a loyal customer (false positive).

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

## 12. Development Challenges & Solutions

Key technical challenges solved during development:

### 1. Custom Recall-First Optuna Objective

**Challenge:** Standard classification metrics (accuracy, F1) don't align with the business goal of maximising churner detection while maintaining acceptable precision.

**Solution:** Built a custom Optuna objective function that:
- Weights recall 75% and F1 25% to prioritise catching churners
- Enforces a soft precision floor (≥ 0.40) via penalty term
- Runs 5-fold stratified cross-validation with early stopping per fold
- Penalises models that drop below minimum precision threshold

**Impact:** Achieved 90.1% recall vs ~75% with standard F1 optimisation.

### 2. Post-Training Threshold Tuning

**Challenge:** Default probability threshold (0.5) optimises for balanced classes, but churn datasets are imbalanced (~27% churn rate).

**Solution:** After Optuna hyperparameter tuning, performed a secondary grid search over thresholds (0.1–0.9) on the validation set to find the optimal decision boundary for recall maximisation.

**Impact:** Lowering threshold from 0.5 → 0.264 increased recall from 82% → 90.1% while keeping precision at 42.5%.

### 3. MLflow 3.x Artifact Path Migration

**Challenge:** MLflow 3.x changed artifact directory structure from flat `artifacts/` to nested `artifacts/model/`, breaking serving-time model loading.

**Solution:** Implemented multi-path fallback logic in `inference.py`:
1. Try MLflow 3.x nested path: `artifacts/model/model.pkl`
2. Fall back to flat path: `artifacts/model.pkl`
3. Check LFS-tracked model at `src/serving/model/<run_id>/artifacts/model/model.pkl`
4. Load feature schema from `feature_columns.txt` in same directory

**Impact:** Zero-downtime model loading across MLflow versions and deployment environments.

### 4. FastAPI + Gradio Architecture

**Challenge:** Gradio's standalone `demo.launch()` conflicts with FastAPI's routing when both need to serve on the same port.

**Solution:** Used `gr.mount_gradio_app()` to mount Gradio as a sub-application at `/ui/` while FastAPI handles:
- `/predict` — REST API for programmatic access
- `/health` — Docker health checks
- `/docs` — OpenAPI schema
- `/` — Redirect to Gradio UI

**Impact:** Single-port deployment with both interactive UI and REST API, simplifying Docker networking and HF Spaces config.

### 5. Git LFS Binary Artifact Strategy

**Challenge:** Hugging Face Spaces rejected direct push of 1.2 MB `model.pkl` binary blob; GitHub also warns on files > 50 MB.

**Solution:** 
1. Installed `git-xet` extension for Hugging Face compatibility
2. Configured Git LFS tracking: `git lfs track "*.pkl"`
3. Migrated existing binary: `git lfs migrate import --include="*.pkl"`
4. Verified LFS pointer format in repository (3-line pointer file instead of raw binary)

**Impact:** Model artifacts tracked as lightweight pointers in Git, actual binaries stored in LFS backend, enabling seamless HF Spaces deployment.

### 6. Dynamic PORT Binding for Multi-Environment Deployment

**Challenge:** Local development uses port 8000, Docker defaults to 7860, but Hugging Face Spaces injects a random `$PORT` at runtime.

**Solution:** Modified Dockerfile CMD to use shell parameter expansion:
```dockerfile
CMD sh -c "uv run uvicorn src.app.app:app --host 0.0.0.0 --port ${PORT:-7860}"
```
This reads `$PORT` if set (HF Spaces), otherwise defaults to 7860 (local Docker).

**Impact:** Single Dockerfile works across local, Docker Hub, and HF Spaces without environment-specific changes.

### 7. CI/CD Without Lockfile Commitment

**Challenge:** `uv.lock` not committed to repo (personal choice for lightweight dependency management), but GitHub Actions `setup-uv` with `enable-cache: true` hard-fails when lockfile missing.

**Solution:** Modified CI workflow to disable UV cache:
```yaml
- uses: astral-sh/setup-uv@v5
  with:
    enable-cache: false
```
All dependency resolution happens fresh per CI run via `uv sync --no-dev`.

**Impact:** CI passes without lockfile; trade-off is slightly slower install (~30s) vs deterministic builds.

---

## 13. Quickstart

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

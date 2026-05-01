# Telco Customer Churn Predictor

An end-to-end ML project that predicts customer churn for a telecom company. Covers the full pipeline from data validation and feature engineering through model tuning, experiment tracking, REST API serving, and CI/CD deployment.

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

## Project Structure

```
src/
  app/          # FastAPI + Gradio application entrypoint
  data/         # Data loading and preprocessing
  features/     # Feature engineering
  models/       # Training, tuning (Optuna), evaluation
  serving/      # Inference pipeline and model artifacts
  utils/        # Data validation and shared helpers
scripts/
  run_pipeline.py   # End-to-end training pipeline
notebooks/
  eda.ipynb         # Exploratory data analysis
tests/            # Unit tests (pytest)
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

**Run tests**
```bash
uv run pytest -q
```

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

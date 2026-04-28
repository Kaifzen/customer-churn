import os
from pathlib import Path

import pandas as pd
import mlflow
import joblib


DEFAULT_THRESHOLD = 0.25


def _first_existing_file(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is None:
            continue
        if path.is_file():
            return path
    return None


def _first_existing_dir(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is None:
            continue
        if path.is_dir():
            return path
    return None


def _dedupe_paths(paths: list[Path | None]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _read_simple_yaml_value(file_path: Path, key: str) -> str | None:
    if not file_path.is_file():
        return None
    prefix = f"{key}:"
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(prefix):
                return line[len(prefix):].strip().strip("\"'")
    return None


def _parse_scalar_prediction(preds):
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    if isinstance(preds, (list, tuple)):
        if len(preds) == 0:
            raise ValueError("Model returned empty predictions")
        first = preds[0]
        if isinstance(first, (list, tuple)) and len(first) > 0:
            return float(first[-1])
        return float(first)

    return float(preds)

# Model URI resolution: support explicit env, flat model copy, and local mlruns layout.
workspace_root = Path(__file__).resolve().parents[2]
model_uri_env = os.getenv("MODEL_URI")
model_dir_env = os.getenv("MODEL_DIR", "/app/model")

# Primary location for checked-in serving artifacts in this project.
serving_model_root = workspace_root / "src" / "serving" / "model"
if not serving_model_root.is_dir():
    # Backward-compatible fallback for alternate layouts.
    serving_model_root = workspace_root / "serving" / "model"

serving_run_candidates = []
if serving_model_root.is_dir():
    serving_run_candidates = sorted(
        [
            d
            for d in serving_model_root.iterdir()
            if d.is_dir() and (d / "params").is_dir()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

# Build model URI candidates in priority order:
# 1) explicitly copied serving artifacts under src/serving/model
# 2) optional env-configured locations
# 3) local mlruns fallback discovery
model_uri_candidates: list[Path | None] = []

# Priority path if model artifacts were copied directly under serving/model.
model_uri_candidates.append(serving_model_root / "artifacts" / "model")

# Priority paths if model artifacts live under a specific copied run folder.
for run_dir in serving_run_candidates:
    model_uri_candidates.append(run_dir / "artifacts" / "model")


# Env-based paths are secondary fallback.
model_uri_candidates.extend(
    [
        Path(model_uri_env) if model_uri_env else None,
        Path(model_dir_env),
        Path(model_dir_env) / "artifacts" / "model",
    ]
)

# Discover latest model in local mlruns if available.
local_model_candidates = sorted(
    (workspace_root / "mlruns").glob("*/*/artifacts/model"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
model_uri_candidates.extend(local_model_candidates)

# MLflow 3 model registry-in-experiment layout.
local_model_candidates_v3 = sorted(
    (workspace_root / "mlruns").glob("*/models/m-*/artifacts"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
model_uri_candidates.extend(local_model_candidates_v3)

# If only a run folder is copied under src/serving/model, resolve its model output id
# to workspace mlruns/*/models/<model_id>/artifacts when available.
for run_dir in serving_run_candidates:
    outputs_dir = run_dir / "outputs"
    if not outputs_dir.is_dir():
        continue
    for model_output in outputs_dir.glob("m-*"):
        model_uri_candidates.extend(
            (workspace_root / "mlruns").glob(f"*/models/{model_output.name}/artifacts")
        )

model_uri_candidates = _dedupe_paths(model_uri_candidates)
resolved_model_uri = _first_existing_dir(model_uri_candidates)
if resolved_model_uri is None:
    candidate_text = "\n".join(str(p) for p in model_uri_candidates)
    raise FileNotFoundError(
        "Could not locate an MLflow model directory. Checked:\n"
        f"{candidate_text}"
    )

model = mlflow.pyfunc.load_model(str(resolved_model_uri))

# Infer run-root context for threshold/features files.
possible_run_roots: list[Path | None] = []

# Case 1: model under copied serving run dir (.../<run_id>/artifacts/model)
for run_dir in serving_run_candidates:
    if resolved_model_uri == run_dir / "artifacts" / "model":
        possible_run_roots.append(run_dir)
        break

# Case 2: model loaded from legacy mlruns run layout (.../<run_id>/artifacts/model)
if resolved_model_uri.name == "model" and resolved_model_uri.parent.name == "artifacts":
    possible_run_roots.append(resolved_model_uri.parent.parent)

# Case 3: model loaded from MLflow 3 model layout (.../models/<model_id>/artifacts)
if (
    resolved_model_uri.name == "artifacts"
    and resolved_model_uri.parent.name.startswith("m-")
    and resolved_model_uri.parent.parent.name == "models"
):
    model_meta = resolved_model_uri.parent / "meta.yaml"
    source_run_id = _read_simple_yaml_value(model_meta, "source_run_id")
    experiment_id = _read_simple_yaml_value(model_meta, "experiment_id")
    if source_run_id and experiment_id:
        possible_run_roots.append(
            workspace_root / "mlruns" / experiment_id / source_run_id
        )

# General fallbacks.
possible_run_roots.extend([
    Path(model_dir_env),
    *serving_run_candidates,
    serving_model_root,
])

possible_run_roots = _dedupe_paths(possible_run_roots)
resolved_run_root = _first_existing_dir(possible_run_roots) or resolved_model_uri

feature_file = _first_existing_file(
    [
        Path(os.getenv("FEATURE_COLUMNS_PATH", "")) if os.getenv("FEATURE_COLUMNS_PATH") else None,
        resolved_run_root / "artifacts" / "feature_columns.txt",
        resolved_run_root / "feature_columns.txt",
        resolved_model_uri / "feature_columns.txt",
    ]
)

if feature_file is None:
    preprocessing_file = _first_existing_file(
        [
            resolved_run_root / "artifacts" / "preprocessing.pkl",
            resolved_run_root / "preprocessing.pkl",
        ]
    )
    if preprocessing_file is None:
        raise FileNotFoundError("feature_columns.txt/preprocessing.pkl not found for serving")
    preprocessing_artifact = joblib.load(preprocessing_file)
    FEATURE_COLS = preprocessing_artifact.get("feature_columns", [])
else:
    with open(feature_file, encoding="utf-8") as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]

if not FEATURE_COLS:
    raise ValueError("Feature column schema is empty")

threshold_file = _first_existing_file(
    [
        resolved_run_root / "params" / "threshold",
        resolved_run_root / "params" / "threshold_input",
    ]
)

if threshold_file is not None:
    with open(threshold_file, encoding="utf-8") as f:
        MODEL_THRESHOLD = float(f.read().strip())
else:
    MODEL_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", str(DEFAULT_THRESHOLD)))

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply serving-time transforms and align to training feature schema."""
    df = df.copy()

    df.columns = df.columns.str.strip()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)

    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


def predict(input_dict: dict) -> str:
    """Predict churn label from a raw customer payload."""
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        score = _parse_scalar_prediction(model.predict(df_enc))
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    # Some pyfunc outputs are already classes {0,1}; otherwise treat as probability.
    if score in (0.0, 1.0):
        label = int(score)
    else:
        label = int(score >= MODEL_THRESHOLD)

    return "Likely to churn" if label == 1 else "Not likely to churn"
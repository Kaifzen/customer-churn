#!/usr/bin/env python3
"""
Runs sequentially: load -> validate -> preprocess -> feature engineering ->
tune -> train -> evaluate -> log model/metrics/artifacts to MLflow.

"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import mlflow
import mlflow.lightgbm
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules - Core pipeline components
from src.data.load_data import load_data                    # Data loading with error handling
from src.data.preprocess import preprocess_data            # Basic data cleaning
from src.features.build_features import build_features     # Feature engineering (CRITICAL for model performance)
from src.utils.validate_data import validate_telco_data    # Data quality validation
from src.models.tune import tune_model
from src.models.evaluate import evaluate_model

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.
    
    """
    
    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking (not a tracking server)
    project_root = Path(__file__).resolve().parents[1]
    mlruns_path = args.mlflow_uri or (project_root / "mlruns").as_uri()
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  # Creates experiment if doesn't exist

    # Start MLflow run - all subsequent logging will be tracked under this run
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        # REQUIRED: These parameters are essential for model reproducibility
        mlflow.log_param("model", "lightgbm")
        mlflow.log_param("threshold_input", args.threshold)
        mlflow.log_param("test_size", args.test_size)   # Train/test split ratio
        mlflow.log_param("tune", int(not args.skip_tuning))
        mlflow.log_param("tune_trials", args.tune_trials)
        mlflow.log_param("cv_splits", args.cv_splits)
        mlflow.log_param("precision_floor", args.precision_floor)
        mlflow.log_param("early_stopping_rounds", args.early_stopping_rounds)

        # === STAGE 1: Data Loading & Validation ===
        print("Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === CRITICAL: Data Quality Validation ===
        # This step is ESSENTIAL for production ML - validates data quality before training
        print("Validating data quality with Great Expectations...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))  # Track data quality over time

        if not is_valid:
            # Log validation failures for debugging
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed. Logged to MLflow.")

        # === STAGE 2: Data Preprocessing ===
        print("Preprocessing data...")
        df = preprocess_data(df)  # Basic cleaning (handle missing values, fix data types)

        # Save processed dataset for reproducibility and debugging
        processed_path = project_root / "data" / "processed" / "telco_churn_processed.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 3: Feature Engineering - CRITICAL for Model Performance ===
        print("Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        # Apply feature engineering transformations
        df_enc = build_features(df, target_col=target)  # Binary encoding + one-hot encoding
        
        # Convert boolean columns to integers for model compatibility.
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"Feature engineering completed: {df_enc.shape[1]} features")

        # === CRITICAL: Save Feature Metadata for Serving Consistency ===
        # This ensures serving pipeline uses exact same features in exact same order
        artifacts_dir = project_root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Get feature columns (exclude target)
        feature_cols = list(df_enc.drop(columns=[target]).columns)
        
        # Save locally for development serving
        with open(artifacts_dir / "feature_columns.json", "w", encoding="utf-8") as f:
            json.dump(feature_cols, f)

        # Log to MLflow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # ESSENTIAL: Save preprocessing artifacts for serving pipeline
        # These artifacts ensure training and serving use identical transformations
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "target": target                  # Target column name
        }
        joblib.dump(preprocessing_artifact, artifacts_dir / "preprocessing.pkl")
        mlflow.log_artifact(str(artifacts_dir / "preprocessing.pkl"))
        mlflow.log_artifact(str(artifacts_dir / "feature_columns.json"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

        # === STAGE 4: Train/Test Split ===
        print("Splitting data...")
        X = df_enc.drop(columns=[target])  # Feature matrix
        y = df_enc[target]                 # Target vector
        
        # Stratified split to maintain class distribution in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,    # Default: 20% for testing
            stratify=y,                  # Maintain class balance
            random_state=42              # Reproducible splits
        )
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # === STAGE 5: Hyperparameter Tuning ===
        if args.skip_tuning:
            best_params = {}
            threshold = args.threshold
            print("Skipping tuning; using default LightGBM parameters.")
        else:
            print("Running Optuna tuning for LightGBM...")
            best_params = tune_model(
                X_train,
                y_train,
                n_trials=args.tune_trials,
                cv_splits=args.cv_splits,
                seed=42,
                tune_threshold=not args.disable_threshold_tuning,
                base_threshold=args.threshold,
                precision_floor=args.precision_floor,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            threshold = float(best_params.pop("threshold", args.threshold))
            mlflow.log_text(
                json.dumps(best_params, indent=2),
                artifact_file="best_params.json",
            )
            print(f"Tuning complete. Selected threshold: {threshold:.3f}")

        mlflow.log_param("threshold", threshold)

        # === Handle Class Imbalance ===
        # Calculate scale_pos_weight to handle imbalanced dataset.
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        # === STAGE 6: Model Training ===
        print("Training LightGBM model...")
        model_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "objective": "binary",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            "scale_pos_weight": scale_pos_weight,
        }
        model_params.update(best_params)
        mlflow.log_params(model_params)

        model = LGBMClassifier(**model_params)

        # === Train Model and Track Training Time ===
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)  # Track training performance
        print(f"Model trained in {train_time:.2f} seconds")

        # === STAGE 7: Model Evaluation ===
        print("Evaluating model performance...")

        # Track inference time for evaluation pass.
        t1 = time.time()
        metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)  # Track inference performance

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, float(value))

        print("Model Performance:")
        print(
            f"   Precision(1): {metrics['precision_churn_1']:.3f} | "
            f"Recall(1): {metrics['recall']:.3f}"
        )
        print(
            f"   F1(1): {metrics['f1_churn_1']:.3f} | "
            f"ROC AUC: {metrics['roc_auc']:.3f}"
        )

        # === STAGE 8: Model Serialization and Logging ===
        print("Saving model to MLflow...")
        mlflow.lightgbm.log_model(model, artifact_path="model")
        print("Model saved to MLflow for serving pipeline")

        # === Final Performance Summary ===
        print("\nPerformance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        print(f"   Threshold used: {threshold:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with LightGBM + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.264)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")
    p.add_argument("--skip_tuning", action="store_true",
                   help="skip Optuna and use default LightGBM parameters")
    p.add_argument("--tune_trials", type=int, default=120,
                   help="number of Optuna trials when tuning is enabled")
    p.add_argument("--cv_splits", type=int, default=5,
                   help="number of folds for StratifiedKFold during tuning")
    p.add_argument("--precision_floor", type=float, default=0.40,
                   help="minimum mean precision(1) during tuning guardrail")
    p.add_argument("--early_stopping_rounds", type=int, default=75,
                   help="stopping rounds for LightGBM early stopping during tuning")
    p.add_argument("--disable_threshold_tuning", action="store_true",
                   help="disable threshold search inside Optuna objective")

    args = p.parse_args()
    main(args)

"""
# Use this below to run the pipeline:

python run_pipeline.py --input telco_data.csv --target Churn

"""
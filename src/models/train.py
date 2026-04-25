import mlflow
import pandas as pd
import mlflow.lightgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


DEFAULT_THRESHOLD = 0.264

def train_model(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = DEFAULT_THRESHOLD,
    model_params: dict | None = None,
):
    """
    Trains a LightGBM model and logs metrics/artifacts with MLflow.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.
        threshold (float): Probability cutoff used to convert scores into class labels.
        model_params (dict | None): Optional LightGBM hyperparameters.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle imbalance via negative/positive ratio.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    base_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "objective": "binary",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
    }
    if model_params:
        base_params.update(model_params)

    model = LGBMClassifier(**base_params)

    with mlflow.start_run():
        # Train model and predict using probability threshold for recall-focused behavior.
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= threshold).astype(int)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, pos_label=1, zero_division=0)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_test, proba)

        # Log params, metrics, and model
        mlflow.log_params(base_params)
        mlflow.log_param("threshold", float(threshold))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_churn_1", precision)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_churn_1", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.lightgbm.log_model(model, "model")

        # 🔑 Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(
            "Model trained. "
            f"Accuracy: {acc:.4f}, Precision(1): {precision:.4f}, "
            f"Recall(1): {rec:.4f}, F1(1): {f1:.4f}, ROC-AUC: {auc:.4f}"
        )
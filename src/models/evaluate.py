from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, threshold: float = 0.264):
    """
    Evaluate a trained lgbm classifier on test data.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        threshold (float): Probability cutoff used to convert scores into class labels.

    Returns:
        dict: Evaluation metrics.
    """
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision_churn_1": precision_score(y_test, preds, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, preds, pos_label=1, zero_division=0),
        "f1_churn_1": f1_score(y_test, preds, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    print(
        "Evaluation Metrics: "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision(1): {metrics['precision_churn_1']:.4f}, "
        f"Recall(1): {metrics['recall']:.4f}, "
        f"F1(1): {metrics['f1_churn_1']:.4f}, "
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    return metrics
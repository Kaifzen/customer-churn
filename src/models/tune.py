import numpy as np
import optuna
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


def tune_model(
    X,
    y,
    n_trials: int = 120,
    cv_splits: int = 5,
    seed: int = 42,
    tune_threshold: bool = True,
    base_threshold: float = 0.3,
    precision_floor: float = 0.40,
    early_stopping_rounds: int = 75,
):
    """
    Tune a LightGBM model with a recall-first but precision-aware objective.

    Args:
        X: Features dataframe/array (ideally training split only).
        y: Target series/array.
        n_trials (int): Number of Optuna trials.
        cv_splits (int): Number of stratified folds.
        seed (int): Random seed.
        tune_threshold (bool): Whether to jointly tune probability threshold.
        base_threshold (float): Fallback threshold if tune_threshold=False.
        precision_floor (float): Minimum precision guardrail for class 1.
        early_stopping_rounds (int): Rounds without improvement before stopping.

    Returns:
        dict: Best parameters from Optuna (includes threshold if tuned).
    """

    # Handle class imbalance via negative/positive ratio.
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 700, 2200),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 160),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 160),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 3.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.35),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary",
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
        }

        threshold = (
            trial.suggest_float("threshold", 0.22, 0.45)
            if tune_threshold
            else base_threshold
        )

        recalls = []
        precisions = []
        f1s = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

            model = LGBMClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="binary_logloss",
                callbacks=[
                    early_stopping(
                        stopping_rounds=early_stopping_rounds,
                        verbose=False,
                    )
                ],
            )

            proba = model.predict_proba(X_va)[:, 1]
            preds = (proba >= threshold).astype(int)

            recalls.append(recall_score(y_va, preds, pos_label=1))
            precisions.append(
                precision_score(y_va, preds, pos_label=1, zero_division=0)
            )
            f1s.append(f1_score(y_va, preds, pos_label=1, zero_division=0))

            # Report intermediate score so Optuna's pruner can prune weak trials.
            partial_recall = float(np.mean(recalls))
            partial_precision = float(np.mean(precisions))
            partial_f1 = float(np.mean(f1s))
            partial_score = 0.75 * partial_recall + 0.25 * partial_f1
            if partial_precision < precision_floor:
                partial_score -= 0.35 * (precision_floor - partial_precision)

            trial.report(partial_score, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_recall = float(np.mean(recalls))
        mean_precision = float(np.mean(precisions))
        mean_f1 = float(np.mean(f1s))

        # Recall-first optimization with a soft precision floor penalty.
        score = 0.75 * mean_recall + 0.25 * mean_f1
        if mean_precision < precision_floor:
            score -= 0.35 * (precision_floor - mean_precision)

        return float(score)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best CV Recall:", study.best_value)
    print("Best Params:", study.best_params)
    return study.best_params
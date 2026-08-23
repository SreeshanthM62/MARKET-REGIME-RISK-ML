from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Bearish", "Neutral", "Bullish"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "bearish_precision": report["Bearish"]["precision"],
        "bearish_recall": report["Bearish"]["recall"],
        "neutral_precision": report["Neutral"]["precision"],
        "neutral_recall": report["Neutral"]["recall"],
        "bullish_precision": report["Bullish"]["precision"],
        "bullish_recall": report["Bullish"]["recall"],
    }


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def model_probability(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)


def permutation_importance_frame(model, X, y, feature_names, random_state=42) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X, y, scoring="f1_macro", n_repeats=10, random_state=random_state, n_jobs=-1
    )
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

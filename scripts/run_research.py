from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.data_loader import load_raw_data
from src.preprocessing import clean_and_align, build_market_frame
from src.features import make_features
from src.target import add_forward_return, add_regime_target, derive_thresholds
from src.models import build_models, fit_model
from src.evaluation import classification_metrics, permutation_importance_frame
from src.risk_analysis import build_overlay, performance_metrics, regime_summary

RANDOM_STATE = 42
HORIZON = 20
TRAIN_END = "2020-12-31"
VALIDATION_END = "2022-12-31"
TEST_END = "2025-12-31"

RESULTS = Path("results")
PREDICTIONS = RESULTS / "predictions"
METRICS = RESULTS / "metrics"


def prepare_dataset():
    data = clean_and_align(load_raw_data())
    frame = build_market_frame(data)
    features = make_features(frame)

    target_base = add_forward_return(frame, HORIZON)
    lower, upper = derive_thresholds(target_base["forward_return"], TRAIN_END)
    target_base = add_regime_target(target_base, lower, upper)

    dataset = frame.join(features).join(target_base[["forward_return", "regime"]])
    dataset["next_day_return"] = dataset["SPY_close"].pct_change().shift(-1)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    feature_cols = list(features.columns)
    dataset = dataset.dropna(subset=feature_cols + ["regime"]).copy()
    dataset["regime"] = dataset["regime"].astype(int)

    return dataset, feature_cols, lower, upper


def split_data(dataset):
    train = dataset.loc[:TRAIN_END].copy()
    validation = dataset.loc["2021-01-01":VALIDATION_END].copy()
    test = dataset.loc["2023-01-01":TEST_END].copy()

    # Purge the last HORIZON observations of train/validation so labels cannot
    # depend on returns from the next evaluation period.
    train = train.iloc[:-HORIZON].copy()
    validation = validation.iloc[:-HORIZON].copy()
    return train, validation, test


def main():
    RESULTS.mkdir(exist_ok=True)
    PREDICTIONS.mkdir(exist_ok=True)
    METRICS.mkdir(exist_ok=True)

    dataset, feature_cols, lower, upper = prepare_dataset()
    train, validation, test = split_data(dataset)

    X_train, y_train = train[feature_cols], train["regime"]
    X_val, y_val = validation[feature_cols], validation["regime"]
    X_test, y_test = test[feature_cols], test["regime"]

    all_metrics = {}
    fitted = {}

    for bundle in build_models(RANDOM_STATE):
        fit_model(bundle, X_train, y_train)
        val_pred = bundle.model.predict(X_val)
        test_pred = bundle.model.predict(X_test)

        all_metrics[bundle.name] = {
            "validation": classification_metrics(y_val, val_pred),
            "test": classification_metrics(y_test, test_pred),
        }

        fitted[bundle.name] = (bundle.model, test_pred)

    # Select by validation macro-F1 only. Test remains untouched for model choice.
    best_name = max(
        all_metrics,
        key=lambda name: all_metrics[name]["validation"]["macro_f1"],
    )
    best_model, test_pred = fitted[best_name]

    test_output = test.copy()
    test_output["predicted_regime"] = test_pred
    proba = best_model.predict_proba(X_test)
    test_output["prediction_probability"] = proba.max(axis=1)

    classes = list(best_model.classes_)
    for idx, cls in enumerate(classes):
        test_output[f"prob_{int(cls)}"] = proba[:, idx]

    test_output.to_csv(PREDICTIONS / "oos_predictions.csv")

    importance = permutation_importance_frame(
        best_model, X_test, y_test, feature_cols, RANDOM_STATE
    )
    importance.to_csv(METRICS / "feature_importance.csv", index=False)

    overlay = build_overlay(test_output)
    overlay.to_csv(PREDICTIONS / "strategy_returns.csv")

    strategy_metrics = performance_metrics(overlay["strategy_return"])
    buy_hold_metrics = performance_metrics(overlay["buy_hold_return"])

    summary = regime_summary(overlay)
    summary.to_csv(METRICS / "regime_summary.csv", index=False)

    report = {
        "best_model": best_name,
        "target_horizon_days": HORIZON,
        "bearish_threshold": lower,
        "bullish_threshold": upper,
        "train_period": [str(train.index.min().date()), str(train.index.max().date())],
        "validation_period": [str(validation.index.min().date()), str(validation.index.max().date())],
        "test_period": [str(test.index.min().date()), str(test.index.max().date())],
        "model_metrics": all_metrics,
        "strategy_metrics": strategy_metrics,
        "buy_hold_metrics": buy_hold_metrics,
    }
    (METRICS / "research_summary.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )

    print(json.dumps(report, indent=2, default=float))
    print(f"\nSelected model: {best_name}")
    print("Research outputs saved under results/.")


if __name__ == "__main__":
    main()

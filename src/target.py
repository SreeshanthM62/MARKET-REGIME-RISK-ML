from __future__ import annotations

import pandas as pd


LABELS = {0: "Bearish", 1: "Neutral", 2: "Bullish"}


def add_forward_return(frame: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    out = frame.copy()
    out["forward_return"] = out["SPY_close"].shift(-horizon) / out["SPY_close"] - 1
    return out


def derive_thresholds(
    forward_returns: pd.Series,
    train_end: str = "2020-12-31",
    horizon: int = 20,
) -> tuple[float, float]:
    """Derive thresholds only from labels whose full forward window stays in training."""
    eligible_dates = forward_returns.index[forward_returns.index <= pd.Timestamp(train_end)]
    if len(eligible_dates) <= horizon:
        raise ValueError("Not enough observations to derive target thresholds.")
    cutoff = eligible_dates[-horizon]
    train = forward_returns.loc[:cutoff].dropna()
    if len(train) < 100:
        raise ValueError("Not enough training observations to derive target thresholds.")
    lower, upper = train.quantile([1 / 3, 2 / 3])
    if lower >= upper:
        raise ValueError("Target thresholds are not ordered.")
    return float(lower), float(upper)


def add_regime_target(
    frame: pd.DataFrame,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    out = frame.copy()
    r = out["forward_return"]
    out["regime"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[r < lower, "regime"] = 0
    out.loc[(r >= lower) & (r <= upper), "regime"] = 1
    out.loc[r > upper, "regime"] = 2
    return out


def label_name(value: int) -> str:
    return LABELS[int(value)]

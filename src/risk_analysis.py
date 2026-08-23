from __future__ import annotations

import numpy as np
import pandas as pd


def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, name in [(0, "Bearish"), (1, "Neutral"), (2, "Bullish")]:
        subset = df.loc[df["predicted_regime"] == regime].copy()
        if subset.empty:
            rows.append(
                {"regime": name, "observations": 0, "avg_next_day_return": np.nan,
                 "annualized_volatility": np.nan, "max_drawdown": np.nan,
                 "positive_next_day_pct": np.nan}
            )
            continue
        daily = subset["next_day_return"].dropna()
        wealth = (1 + daily).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        rows.append(
            {
                "regime": name,
                "observations": len(subset),
                "avg_next_day_return": daily.mean(),
                "annualized_volatility": daily.std() * np.sqrt(252),
                "max_drawdown": drawdown.min(),
                "positive_next_day_pct": (daily > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def performance_metrics(returns: pd.Series) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    ann_return = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = np.nan if vol == 0 else (r.mean() * 252) / vol
    return {
        "total_return": (1 + r).prod() - 1,
        "annualized_return": ann_return,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(r),
    }


def build_overlay(
    test: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    out = test.copy()
    out["next_day_return"] = out["SPY_close"].pct_change().shift(-1)
    exposure_map = {0: 0.0, 1: 0.5, 2: 1.0}
    out["exposure"] = out["predicted_regime"].map(exposure_map).fillna(0.0)
    out["position_change"] = out["exposure"].diff().abs().fillna(out["exposure"].abs())
    cost = transaction_cost_bps / 10_000
    out["strategy_return"] = out["exposure"] * out["next_day_return"] - out["position_change"] * cost
    out["buy_hold_return"] = out["next_day_return"]
    return out

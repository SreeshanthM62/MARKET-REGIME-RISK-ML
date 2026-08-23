from __future__ import annotations

import pandas as pd


def validate_ohlcv(df: pd.DataFrame, name: str) -> None:
    """Run basic integrity checks on a single OHLCV series."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns: {sorted(missing)}")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name}: index is not sorted.")

    if df.index.has_duplicates:
        raise ValueError(f"{name}: duplicate dates found.")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"{name}: non-positive price found.")

    if (df["Volume"] < 0).any():
        raise ValueError(f"{name}: negative volume found.")


def clean_and_align(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Validate, de-duplicate and align the market series by date."""
    for name, df in data.items():
        validate_ohlcv(df, name)

    common_index = None
    for df in data.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    if common_index is None or len(common_index) == 0:
        raise ValueError("No common dates across the downloaded series.")

    return {name: df.loc[common_index].copy() for name, df in data.items()}


def build_market_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine the primary asset and supporting market variables into one frame."""
    spy = data["SPY"].copy()
    out = pd.DataFrame(index=spy.index)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[f"SPY_{col.lower()}"] = spy[col]

    for name in ["QQQ", "TLT", "GLD", "VIX", "TNX", "DXY"]:
        out[f"{name.lower()}_close"] = data[name]["Close"]

    out = out.sort_index()
    return out

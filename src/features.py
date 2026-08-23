from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a compact set of explainable features using information available at t."""
    close = frame["SPY_close"]
    high = frame["SPY_high"]
    low = frame["SPY_low"]
    volume = frame["SPY_volume"]

    f = pd.DataFrame(index=frame.index)

    f["ret_1d"] = close.pct_change(1)
    f["ret_5d"] = close.pct_change(5)
    f["ret_20d"] = close.pct_change(20)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    f["price_sma20_ratio"] = close / sma20 - 1
    f["price_sma50_ratio"] = close / sma50 - 1
    f["sma20_sma50_ratio"] = sma20 / sma50 - 1

    f["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd"] = macd / close
    f["macd_signal_gap"] = (macd - signal) / close

    f["vol_5d"] = f["ret_1d"].rolling(5).std() * np.sqrt(252)
    f["vol_20d"] = f["ret_1d"].rolling(20).std() * np.sqrt(252)
    f["atr_14_pct"] = _atr(high, low, close, 14) / close

    f["vix_change"] = frame["vix_close"].pct_change(5)
    f["tnx_change"] = frame["tnx_close"].diff(5)
    f["dxy_return"] = frame["dxy_close"].pct_change(5)
    f["gld_return"] = frame["gld_close"].pct_change(5)
    f["volume_change"] = volume.pct_change(5)

    return f.replace([np.inf, -np.inf], np.nan)


FEATURE_DESCRIPTIONS = {
    "ret_1d": "1-day SPY return; short-term price movement.",
    "ret_5d": "5-day SPY return; weekly momentum.",
    "ret_20d": "20-day SPY return; monthly momentum.",
    "price_sma20_ratio": "Distance of price from the 20-day SMA.",
    "price_sma50_ratio": "Distance of price from the 50-day SMA.",
    "sma20_sma50_ratio": "Short-term trend relative to the medium-term trend.",
    "rsi_14": "14-day RSI; momentum/overbought-oversold context.",
    "macd": "MACD normalized by price; trend/momentum signal.",
    "macd_signal_gap": "MACD minus signal line, normalized by price.",
    "vol_5d": "Annualized 5-day realized volatility.",
    "vol_20d": "Annualized 20-day realized volatility.",
    "atr_14_pct": "14-day ATR as a percentage of price.",
    "vix_change": "5-day change in the VIX; market stress proxy.",
    "tnx_change": "5-day change in the 10-year Treasury yield index.",
    "dxy_return": "5-day U.S. Dollar Index return.",
    "gld_return": "5-day gold ETF return.",
    "volume_change": "5-day change in SPY volume.",
}

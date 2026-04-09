import pandas as pd
import numpy as np


def compute_sma(series, window):
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(series, window=20, num_std=2.0):
    mid = compute_sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    return mid, mid + num_std * std, mid - num_std * std


def ma_crossover_strategy(df, short_window=20, long_window=50, ma_type="sma"):
    assert short_window < long_window
    out = df[["Close"]].copy()
    ma_fn = compute_sma if ma_type == "sma" else compute_ema
    out["MA_Short"] = ma_fn(out["Close"], short_window)
    out["MA_Long"] = ma_fn(out["Close"], long_window)
    out["above"] = (out["MA_Short"] > out["MA_Long"]).astype(int)
    out["raw_signal"] = out["above"].diff().fillna(0).astype(int)
    out["signal"] = out["raw_signal"].shift(1).fillna(0).astype(int)
    out.drop(columns=["above", "raw_signal"], inplace=True)
    return out


def rsi_strategy(df, rsi_period=14, oversold=30.0, overbought=70.0):
    out = df[["Close"]].copy()
    out["RSI"] = compute_rsi(out["Close"], rsi_period)
    out["was_oversold"] = (out["RSI"] < oversold).astype(int)
    out["was_overbought"] = (out["RSI"] > overbought).astype(int)
    buy_cross = (out["was_oversold"].shift(1) == 1) & (out["RSI"] >= oversold)
    sell_cross = (out["was_overbought"].shift(1) == 1) & (out["RSI"] <= overbought)
    out["raw_signal"] = 0
    out.loc[buy_cross, "raw_signal"] = 1
    out.loc[sell_cross, "raw_signal"] = -1
    out["signal"] = out["raw_signal"].shift(1).fillna(0).astype(int)
    out.drop(columns=["was_oversold", "was_overbought", "raw_signal"], inplace=True)
    return out


def bollinger_strategy(df, window=20, num_std=2.0):
    out = df[["Close"]].copy()
    out["BB_Mid"], out["BB_Upper"], out["BB_Lower"] = compute_bollinger_bands(out["Close"], window, num_std)
    out["raw_signal"] = 0
    out.loc[out["Close"] <= out["BB_Lower"], "raw_signal"] = 1
    out.loc[out["Close"] >= out["BB_Upper"], "raw_signal"] = -1
    out["signal"] = out["raw_signal"].shift(1).fillna(0).astype(int)
    out.drop(columns=["raw_signal"], inplace=True)
    return out


STRATEGIES = {
    "MA Crossover": ma_crossover_strategy,
    "RSI Mean-Reversion": rsi_strategy,
    "Bollinger Band": bollinger_strategy,
}

"""均衡乖離度・定常性特徴量 — カテゴリB: 8個 (ADR 006 #1).

資料3の知見: 「定常的な系列は予測しやすい」
全て有界 or 平均回帰的。

bb_position, rsi, stoch_rsi_k, stoch_rsi_d,
channel_pos_24, channel_pos_48, close_sma_ratio, vol_regime
"""

from __future__ import annotations

import pandas as pd

from fx_auto_trading.features.indicators import atr


def bb_position(
    close: pd.Series, period: int = 20, std: float = 2.0
) -> pd.Series:
    """ボリンジャーバンド位置 [0-1]."""
    middle = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return (close - lower) / (upper - lower)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI [0-100]."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def stochastic_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI %K [0-1], %D [0-1]."""
    rsi_val = rsi(close, rsi_period)
    rsi_min = rsi_val.rolling(window=stoch_period).min()
    rsi_max = rsi_val.rolling(window=stoch_period).max()
    k = (rsi_val - rsi_min) / (rsi_max - rsi_min)
    d = k.rolling(window=smooth).mean()
    return k, d


def channel_position(
    close: pd.Series, period: int = 24
) -> pd.Series:
    """チャネル位置 [0-1]. (close - N期安値) / (N期高値 - N期安値)."""
    rolling_high = close.rolling(period).max()
    rolling_low = close.rolling(period).min()
    return (close - rolling_low) / (rolling_high - rolling_low)


def close_sma_ratio(
    close: pd.Series, period: int = 20
) -> pd.Series:
    """close / SMA(N) [≈1]. 平均回帰的."""
    sma = close.rolling(window=period).mean()
    return close / sma


def volatility_regime(
    df: pd.DataFrame, atr_period: int = 14, ma_window: int = 20
) -> pd.Series:
    """ATR / ATR_MA [≈1]. ボラティリティレジーム."""
    atr_val = atr(df, period=atr_period)
    atr_ma = atr_val.rolling(window=ma_window).mean()
    return atr_val / atr_ma

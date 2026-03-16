"""特徴量パイプライン + 閾値ラベル生成 (ADR 002 #1, #2, ADR 006).

全28個の特徴量を統合し、閾値ラベルを生成する。
閾値ラベル: ±X pips未満の動きは NaN化（学習から除外）。
"""

from __future__ import annotations

import logging

import pandas as pd

from fx_auto_trading.config import FeatureConfig, LabelConfig
from fx_auto_trading.features.indicators import adx, atr, cci, macd_histogram
from fx_auto_trading.features.momentum import add_lag_features, add_momentum_features
from fx_auto_trading.features.stationary import (
    bb_position,
    channel_position,
    close_sma_ratio,
    rsi,
    stochastic_rsi,
    volatility_regime,
)
from fx_auto_trading.features.temporal import add_temporal_features

logger = logging.getLogger(__name__)


def build_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """全28個の特徴量を生成する (ADR 006)."""
    if config is None:
        config = FeatureConfig()

    result = df.copy()
    close = df["close"]

    # カテゴリA: テクニカル指標 (6個)
    result["macd_hist"] = macd_histogram(
        close, config.macd_fast, config.macd_slow, config.macd_signal
    )
    result["atr"] = atr(df, config.atr_period)
    adx_val, plus_di_val, minus_di_val = adx(df, config.adx_period)
    result["adx"] = adx_val
    result["plus_di"] = plus_di_val
    result["minus_di"] = minus_di_val
    result["cci"] = cci(df, config.cci_period)

    # カテゴリB: 均衡乖離度 (8個) ← 資料3の知見
    result["bb_position"] = bb_position(close, config.bb_period, config.bb_std)
    result["rsi"] = rsi(close, config.rsi_period)
    stoch_k, stoch_d = stochastic_rsi(
        close, config.rsi_period, config.stoch_rsi_period, config.stoch_rsi_smooth
    )
    result["stoch_rsi_k"] = stoch_k
    result["stoch_rsi_d"] = stoch_d
    for period in config.channel_periods:
        result[f"channel_pos_{period}"] = channel_position(close, period)
    result["close_sma_ratio"] = close_sma_ratio(close, config.sma_ratio_period)
    result["vol_regime"] = volatility_regime(
        df, config.atr_period, config.vol_regime_window
    )

    # カテゴリC: 時間帯 (5個)
    for col, series in add_temporal_features(df).items():
        result[col] = series

    # カテゴリD: モメンタム (3個)
    for col, series in add_momentum_features(
        close, config.roc_periods, config.up_ratio_period
    ).items():
        result[col] = series

    # カテゴリE: ラグ (6個)
    for col, series in add_lag_features(close, config.lag_periods).items():
        result[col] = series

    return result


FEATURE_COLUMNS = [
    # A: テクニカル (6)
    "macd_hist",
    "atr",
    "adx",
    "plus_di",
    "minus_di",
    "cci",
    # B: 均衡乖離度 (8)
    "bb_position",
    "rsi",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "channel_pos_24",
    "channel_pos_48",
    "close_sma_ratio",
    "vol_regime",
    # C: 時間帯 (5)
    "hour",
    "day_of_week",
    "is_tokyo",
    "is_london",
    "is_ny",
    # D: モメンタム (3)
    "roc_6",
    "roc_12",
    "up_ratio_6",
    # E: ラグ (6)
    "return_lag1",
    "return_lag2",
    "return_lag3",
    "return_lag4",
    "return_lag5",
    "return_lag6",
]


def build_dataset(
    df: pd.DataFrame,
    horizon: int = 4,
    feature_config: FeatureConfig | None = None,
    label_config: LabelConfig | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """特徴量 + 閾値ラベルのデータセットを生成する.

    ADR 002 #1: ±threshold_pips 未満の動きは NaN化（学習から除外）
    ADR 002 #2: horizon は 1/4/8 を実験的に比較

    Returns:
        (features_df, labels): NaN行を除去済み
    """
    if label_config is None:
        label_config = LabelConfig()
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    features = build_features(df, feature_config)

    # 閾値ラベル生成 (ADR 002 #1)
    future_close = features["close"].shift(-horizon)
    diff = future_close - features["close"]
    threshold = label_config.threshold_pips * label_config.pip_value

    label = pd.Series(float("nan"), index=features.index)
    label[diff >= threshold] = 1.0
    label[diff <= -threshold] = 0.0
    features["label"] = label

    # NaN除去
    combined = features[feature_columns + ["label"]].dropna()

    X = combined[feature_columns]
    y = combined["label"].astype(int)

    logger.info(
        "データセット生成: %d行, %d特徴量, horizon=%d, threshold=%.1fpips, "
        "ラベル比率: up=%.1f%% down=%.1f%%",
        len(X),
        len(feature_columns),
        horizon,
        label_config.threshold_pips,
        (y == 1).mean() * 100,
        (y == 0).mean() * 100,
    )
    return X, y

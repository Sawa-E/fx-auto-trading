"""特徴量パイプライン + 閾値ラベル生成 (ADR 002 #1, #2, ADR 006).

改善版:
- 指標の変化率（delta系）を追加 ← 資料3「パラメータの時間変化が重要」
- 高相関特徴量を整理（cci削除）← 資料5「意味のないデータを入れるな」
- レジームフィルタ対応 ← 資料3「予測しやすいものを予測する」
"""

from __future__ import annotations

import logging

import pandas as pd

from fx_auto_trading.config import FeatureConfig, LabelConfig
from fx_auto_trading.features.indicators import adx, atr, macd_histogram
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
    """特徴量を生成する（改善版）."""
    if config is None:
        config = FeatureConfig()

    result = df.copy()
    close = df["close"]

    # カテゴリA: テクニカル指標（cci削除: bb_positionと相関0.974）
    result["macd_hist"] = macd_histogram(
        close, config.macd_fast, config.macd_slow, config.macd_signal
    )
    result["atr"] = atr(df, config.atr_period)
    adx_val, plus_di_val, minus_di_val = adx(df, config.adx_period)
    result["adx"] = adx_val
    result["plus_di"] = plus_di_val
    result["minus_di"] = minus_di_val

    # カテゴリB: 均衡乖離度
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

    # カテゴリC: 時間帯
    for col, series in add_temporal_features(df).items():
        result[col] = series

    # カテゴリD: モメンタム
    for col, series in add_momentum_features(
        close, config.roc_periods, config.up_ratio_period
    ).items():
        result[col] = series

    # カテゴリE: ラグ
    for col, series in add_lag_features(close, config.lag_periods).items():
        result[col] = series

    # カテゴリF: 指標の変化率（delta系）← 資料3の知見
    # 「パラメータの時間変化が重要」→ 指標が上昇中か下落中かの情報
    result["delta_rsi"] = result["rsi"].diff()
    result["delta_adx"] = result["adx"].diff()
    result["delta_bb_pos"] = result["bb_position"].diff()
    result["delta_macd"] = result["macd_hist"].diff()

    # カテゴリG: マルチタイムフレーム（4H/日足のトレンド方向）
    # 上位足と同方向なら勝ちやすい
    close = result["close"]
    # 4H足トレンド: 4期間SMAの傾き
    sma_4h = close.rolling(4).mean()
    result["trend_4h"] = (sma_4h - sma_4h.shift(1)).apply(
        lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
    )
    # 日足トレンド: 24期間SMAの傾き
    sma_daily = close.rolling(24).mean()
    result["trend_daily"] = (sma_daily - sma_daily.shift(1)).apply(
        lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
    )
    # 4H/日足の一致度（両方同方向=1, 不一致=0）
    result["trend_alignment"] = (
        result["trend_4h"] == result["trend_daily"]
    ).astype(float)

    return result


# 改善版特徴量リスト
FEATURE_COLUMNS = [
    # A: テクニカル (5個、cci削除)
    "macd_hist",
    "atr",
    "adx",
    "plus_di",
    "minus_di",
    # B: 均衡乖離度 (8個)
    "bb_position",
    "rsi",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "channel_pos_24",
    "channel_pos_48",
    "close_sma_ratio",
    "vol_regime",
    # C: 時間帯 (5個)
    "hour",
    "day_of_week",
    "is_tokyo",
    "is_london",
    "is_ny",
    # D: モメンタム (3個)
    "roc_6",
    "roc_12",
    "up_ratio_6",
    # E: ラグ (6個)
    "return_lag1",
    "return_lag2",
    "return_lag3",
    "return_lag4",
    "return_lag5",
    "return_lag6",
    # F: 指標変化率 (4個) ← 資料3の知見
    "delta_rsi",
    "delta_adx",
    "delta_bb_pos",
    "delta_macd",
    # G: マルチタイムフレーム (3個)
    "trend_4h",
    "trend_daily",
    "trend_alignment",
]


def build_dataset(
    df: pd.DataFrame,
    horizon: int = 4,
    feature_config: FeatureConfig | None = None,
    label_config: LabelConfig | None = None,
    feature_columns: list[str] | None = None,
    regime_filter: bool = False,
    regime_adx_threshold: float = 25.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """特徴量 + 閾値ラベルのデータセットを生成する.

    Args:
        regime_filter: Trueならトレンド相場(ADX>閾値)のみ使用
        regime_adx_threshold: レジームフィルタのADX閾値
    """
    if label_config is None:
        label_config = LabelConfig()
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    features = build_features(df, feature_config)

    # レジームフィルタ ← 資料3「予測しやすいものを予測する」
    if regime_filter:
        trend_mask = features["adx"] > regime_adx_threshold
        features = features[trend_mask]
        logger.info(
            "レジームフィルタ(ADX>%.0f): %d → %d件 (%.1f%%)",
            regime_adx_threshold,
            len(df),
            len(features),
            len(features) / len(df) * 100,
        )

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
        "データセット: %d行, %d特徴量, horizon=%d, threshold=%.1fpips, "
        "up=%.1f%% down=%.1f%%",
        len(X),
        len(feature_columns),
        horizon,
        label_config.threshold_pips,
        (y == 1).mean() * 100,
        (y == 0).mean() * 100,
    )
    return X, y

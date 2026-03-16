"""設定: dataclass + 環境変数上書き.

ADR 003: リスク管理パラメータ (ATR乗数, レバ上限, DD停止, 確率閾値)
ADR 004: 正則化パラメータ探索範囲
ADR 005: データパイプライン設定
ADR 006: 特徴量パラメータ
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ApiConfig:
    """GMO FX Public API 設定."""

    base_url: str = "https://forex-api.coin.z.com"
    rate_limit_per_second: int = 1
    timeout: int = 30
    max_retries: int = 3


@dataclass
class FeatureConfig:
    """特徴量パラメータ (ADR 006)."""

    # カテゴリA: テクニカル指標
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    adx_period: int = 14
    cci_period: int = 20

    # カテゴリB: 均衡乖離度
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    stoch_rsi_period: int = 14
    stoch_rsi_smooth: int = 3
    channel_periods: tuple[int, ...] = (24, 48)
    sma_ratio_period: int = 20
    vol_regime_window: int = 20

    # カテゴリD: モメンタム
    roc_periods: tuple[int, ...] = (6, 12)
    up_ratio_period: int = 6

    # カテゴリE: ラグ (ADR 006 #2: 1-6期)
    lag_periods: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


@dataclass
class LabelConfig:
    """ラベル生成設定 (ADR 002 #1, #2)."""

    # 閾値ラベル: ±threshold_pips 未満は NaN化
    threshold_pips: float = 5.0
    # 予測ホライゾン (ADR 002 #2: 1/4/8 を実験的に比較)
    horizons: tuple[int, ...] = (1, 4, 8)
    # USD/JPY: 1pip = 0.01
    pip_value: float = 0.01


@dataclass
class ModelConfig:
    """モデル設定 (ADR 004 #1: 強め正則化)."""

    # Optuna 探索範囲 — Kaggleデフォルトより強め
    max_depth_range: tuple[int, int] = (3, 6)
    min_child_samples_range: tuple[int, int] = (50, 200)
    num_leaves_range: tuple[int, int] = (15, 63)
    reg_lambda_range: tuple[float, float] = (1.0, 10.0)
    reg_alpha_range: tuple[float, float] = (0.1, 5.0)
    learning_rate_range: tuple[float, float] = (0.01, 0.1)
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    n_optuna_trials: int = 50
    random_state: int = 42

    # 過学習検出 (ADR 004 #2)
    overfit_threshold: float = 0.10  # 精度差 > 10% で警告


@dataclass
class TradeConfig:
    """取引ルール設定 (ADR 003)."""

    # 損切り/利確 (ADR 003 #1: ATRベース, RR比1.33)
    sl_atr_multiplier: float = 1.5
    tp_atr_multiplier: float = 2.0

    # レバレッジ (ADR 003 #2)
    max_leverage: float = 5.0

    # モデル停止 (ADR 003 #3)
    max_drawdown: float = 0.10  # DD 10% で停止

    # 確率フィルタ (ADR 003 #4)
    probability_threshold: float = 0.60
    probability_search_range: tuple[float, float] = (0.50, 0.70)

    # スプレッド (USD/JPY)
    spread_pips: float = 0.3

    symbol: str = "USD_JPY"
    timeframe: str = "1H"


@dataclass
class PathsConfig:
    """パス設定."""

    db: str = "data/fx_auto_trading.db"
    results_dir: str = "results"


@dataclass
class AppConfig:
    """アプリケーション全体設定."""

    api: ApiConfig = field(default_factory=ApiConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    log_level: str = "INFO"


def load_config() -> AppConfig:
    """AppConfig を生成し、環境変数で上書きする."""
    config = AppConfig()

    env_map: dict[str, tuple[object, str, type]] = {
        "FX_API_BASE_URL": (config.api, "base_url", str),
        "FX_API_TIMEOUT": (config.api, "timeout", int),
        "FX_LOG_LEVEL": (config, "log_level", str),
        "FX_DB_PATH": (config.paths, "db", str),
        "FX_PROBABILITY_THRESHOLD": (config.trade, "probability_threshold", float),
        "FX_MAX_DRAWDOWN": (config.trade, "max_drawdown", float),
    }

    for env_key, (obj, attr, type_fn) in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            setattr(obj, attr, type_fn(value))

    return config

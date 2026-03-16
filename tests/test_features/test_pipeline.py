"""特徴量パイプラインのテスト."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fx_auto_trading.features.pipeline import FEATURE_COLUMNS, build_dataset, build_features


def _sample_ohlc(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 150.0 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.02,
            "high": close + np.abs(np.random.randn(n) * 0.1),
            "low": close - np.abs(np.random.randn(n) * 0.1),
            "close": close,
        },
        index=idx,
    )


class TestBuildFeatures:
    def test_all_columns_present(self) -> None:
        df = _sample_ohlc()
        result = build_features(df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing: {col}"

    def test_feature_count(self) -> None:
        assert len(FEATURE_COLUMNS) == 28


class TestBuildDataset:
    def test_returns_xy(self) -> None:
        df = _sample_ohlc(500)
        X, y = build_dataset(df, horizon=4)
        assert len(X) == len(y)
        assert len(X) > 0
        assert set(y.unique()).issubset({0, 1})

    def test_threshold_filters_noise(self) -> None:
        df = _sample_ohlc(500)
        from fx_auto_trading.config import LabelConfig

        # 閾値0: 全サンプルがラベル付き
        cfg_no_threshold = LabelConfig(threshold_pips=0.0)
        X_all, _ = build_dataset(df, horizon=4, label_config=cfg_no_threshold)

        # 閾値10pips: ノイジーなサンプルが除外される
        cfg_threshold = LabelConfig(threshold_pips=10.0)
        X_filtered, _ = build_dataset(df, horizon=4, label_config=cfg_threshold)

        assert len(X_filtered) <= len(X_all)

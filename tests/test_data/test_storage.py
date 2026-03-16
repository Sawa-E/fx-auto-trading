"""OhlcStorage のテスト."""

from __future__ import annotations

import tempfile

import pandas as pd
import pytest

from fx_auto_trading.data.storage import OhlcStorage
from fx_auto_trading.exceptions import DataQualityError


def _sample_df(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [100.5 + i * 0.1 for i in range(n)],
            "low": [99.5 + i * 0.1 for i in range(n)],
            "close": [100.2 + i * 0.1 for i in range(n)],
        },
        index=idx,
    )


class TestOhlcStorage:
    def test_save_and_load(self, tmp_path: object) -> None:
        db = f"{tmp_path}/test.db"
        storage = OhlcStorage(db)
        df = _sample_df()

        storage.save(df, "USD_JPY", "1hour")
        loaded = storage.load("USD_JPY", "1hour")

        assert len(loaded) == 5
        assert list(loaded.columns) == ["open", "high", "low", "close"]

    def test_duplicate_ignored(self, tmp_path: object) -> None:
        db = f"{tmp_path}/test.db"
        storage = OhlcStorage(db)
        df = _sample_df()

        storage.save(df, "USD_JPY", "1hour")
        storage.save(df, "USD_JPY", "1hour")

        assert storage.count("USD_JPY", "1hour") == 5

    def test_latest_timestamp(self, tmp_path: object) -> None:
        db = f"{tmp_path}/test.db"
        storage = OhlcStorage(db)
        df = _sample_df()

        storage.save(df, "USD_JPY", "1hour")
        latest = storage.get_latest_timestamp("USD_JPY", "1hour")

        assert latest is not None

    def test_quality_check_invalid_ohlc(self, tmp_path: object) -> None:
        db = f"{tmp_path}/test.db"
        storage = OhlcStorage(db)
        idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
        # high < open → 不正
        df = pd.DataFrame(
            {"open": [100.0], "high": [99.0], "low": [98.0], "close": [99.5]},
            index=idx,
        )

        with pytest.raises(DataQualityError):
            storage.save(df, "USD_JPY", "1hour")

    def test_empty_load(self, tmp_path: object) -> None:
        db = f"{tmp_path}/test.db"
        storage = OhlcStorage(db)
        loaded = storage.load("USD_JPY", "1hour")
        assert loaded.empty

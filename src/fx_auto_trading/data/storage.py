"""SQLite ストレージ (ADR 005 #2, #4).

ohlcテーブルのみ (YAGNI)。取得時にデータ品質チェック。
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from fx_auto_trading.exceptions import DataQualityError, StorageError

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS ohlc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    interval TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    UNIQUE(pair, interval, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ohlc_pair_interval_ts
    ON ohlc(pair, interval, timestamp);
"""


class OhlcStorage:
    """SQLite OHLC ストレージ."""

    def __init__(self, db_path: str = "data/fx_auto_trading.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(_SCHEMA_SQL)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def save(
        self, df: pd.DataFrame, pair: str, interval: str
    ) -> int:
        """DataFrame を ohlc テーブルに保存する。重複は無視。"""
        if df.empty:
            return 0

        self._validate_quality(df)

        rows = [
            (
                pair, interval, str(idx),
                row["open"], row["high"], row["low"], row["close"],
            )
            for idx, row in df.iterrows()
        ]

        try:
            with self._conn() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO ohlc "
                    "(pair, interval, timestamp, open, high, low, close) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                inserted = conn.total_changes
        except sqlite3.Error as e:
            raise StorageError(f"保存エラー: {e}") from e

        logger.info("保存: %d 件 (新規 %d 件)", len(rows), inserted)
        return inserted

    def load(
        self,
        pair: str,
        interval: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """ohlc テーブルから DataFrame を読み出す."""
        query = (
            "SELECT timestamp, open, high, low, close "
            "FROM ohlc WHERE pair=? AND interval=?"
        )
        params: list[str] = [pair, interval]

        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY timestamp"

        try:
            with self._conn() as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except sqlite3.Error as e:
            raise StorageError(f"読み出しエラー: {e}") from e

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        return df

    def get_latest_timestamp(self, pair: str, interval: str) -> str | None:
        """最新の timestamp を返す (差分更新用、ADR 005 #1)."""
        query = (
            "SELECT MAX(timestamp) FROM ohlc WHERE pair=? AND interval=?"
        )
        try:
            with self._conn() as conn:
                result = conn.execute(query, (pair, interval)).fetchone()
        except sqlite3.Error as e:
            raise StorageError(f"クエリエラー: {e}") from e

        return result[0] if result and result[0] else None

    def count(self, pair: str, interval: str) -> int:
        """レコード数を返す."""
        query = "SELECT COUNT(*) FROM ohlc WHERE pair=? AND interval=?"
        with self._conn() as conn:
            result = conn.execute(query, (pair, interval)).fetchone()
        return result[0] if result else 0

    @staticmethod
    def _validate_quality(df: pd.DataFrame) -> None:
        """データ品質チェック (ADR 005 #4).

        - OHLC整合性: high >= max(open,close), low <= min(open,close)
        - 異常値: 前足からの変動率 5% 超でログ警告
        """
        high_valid = df["high"] >= df[["open", "close"]].max(axis=1)
        low_valid = df["low"] <= df[["open", "close"]].min(axis=1)

        invalid_count = (~high_valid).sum() + (~low_valid).sum()
        if invalid_count > 0:
            raise DataQualityError(
                f"OHLC整合性エラー: {invalid_count} 件"
            )

        if len(df) > 1:
            returns = df["close"].pct_change().abs()
            extreme = returns[returns > 0.05]
            if len(extreme) > 0:
                for ts in extreme.index:
                    logger.warning(
                        "異常値検出: %s 変動率=%.2f%%", ts, returns[ts] * 100
                    )

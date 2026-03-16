"""GMO FX Public API クライアント (ADR 005 #1).

1H足は日単位リクエスト (YYYYMMDD)。
レート制限 1回/秒、指数バックオフリトライ。
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any

import httpx
import pandas as pd

from fx_auto_trading.config import ApiConfig
from fx_auto_trading.exceptions import ApiError, ApiRateLimitError, ApiResponseError

logger = logging.getLogger(__name__)

_INTERVAL_MAP = {
    "1H": "1hour",
    "4H": "4hour",
    "1D": "1day",
}


class GmoFxCollector:
    """GMO FX Public API からの OHLC データ取得."""

    def __init__(self, config: ApiConfig | None = None) -> None:
        if config is None:
            config = ApiConfig()
        self._base_url = config.base_url
        self._rate_limit = config.rate_limit_per_second
        self._timeout = config.timeout
        self._max_retries = config.max_retries
        self._last_request_time = 0.0

    def _wait_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        wait = (1.0 / self._rate_limit) - elapsed
        if wait > 0:
            time.sleep(wait)

    def _request(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self._base_url}/public/v1{path}"

        for attempt in range(self._max_retries + 1):
            self._wait_rate_limit()
            try:
                resp = httpx.get(url, params=params, timeout=self._timeout)
                self._last_request_time = time.time()

                if resp.status_code == 429:
                    if attempt == self._max_retries:
                        raise ApiRateLimitError(
                            f"レート制限超過: {resp.status_code}"
                        )
                    delay = min(1.0 * (2**attempt), 30.0)
                    logger.warning(
                        "レート制限。%.1f秒後にリトライ (%d/%d)",
                        delay,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                    continue

                if resp.status_code >= 500:
                    if attempt == self._max_retries:
                        raise ApiResponseError(
                            f"サーバーエラー: {resp.status_code}"
                        )
                    delay = min(1.0 * (2**attempt), 30.0)
                    logger.warning(
                        "サーバーエラー %d。%.1f秒後にリトライ (%d/%d)",
                        resp.status_code,
                        delay,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                    continue

                if resp.status_code >= 400:
                    raise ApiResponseError(
                        f"クライアントエラー: {resp.status_code} - {resp.text}"
                    )

                data = resp.json()
                if data.get("status") != 0:
                    raise ApiResponseError(
                        f"API エラー: status={data.get('status')}, "
                        f"messages={data.get('messages')}"
                    )
                return data  # type: ignore[no-any-return]

            except httpx.TimeoutException as e:
                if attempt == self._max_retries:
                    raise ApiError("タイムアウト: 最大リトライ回数に到達") from e
                delay = min(1.0 * (2**attempt), 30.0)
                logger.warning(
                    "タイムアウト。%.1f秒後にリトライ (%d/%d)",
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(delay)

            except httpx.HTTPError as e:
                if attempt == self._max_retries:
                    raise ApiError(f"HTTPエラー: {e}") from e
                delay = min(1.0 * (2**attempt), 30.0)
                logger.warning(
                    "HTTPエラー: %s。%.1f秒後にリトライ (%d/%d)",
                    e,
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(delay)

        raise ApiError("リトライ上限到達")  # pragma: no cover

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        target_date: date,
        price_type: str = "BID",
    ) -> pd.DataFrame:
        """指定日の OHLC データを取得し DataFrame で返す."""
        api_interval = _INTERVAL_MAP.get(interval, interval.lower())

        # GMO API: 1H以下は YYYYMMDD、4H以上は YYYY
        if api_interval in ("4hour", "8hour", "12hour", "1day", "1week", "1month"):
            date_str = target_date.strftime("%Y")
        else:
            date_str = target_date.strftime("%Y%m%d")

        data = self._request(
            "/klines",
            {
                "symbol": symbol,
                "priceType": price_type,
                "interval": api_interval,
                "date": date_str,
            },
        )

        klines = data.get("data", [])
        if not klines:
            return pd.DataFrame()

        rows = [
            {
                "timestamp": pd.Timestamp(int(k["openTime"]), unit="ms", tz="UTC"),
                "open": float(k["open"]),
                "high": float(k["high"]),
                "low": float(k["low"]),
                "close": float(k["close"]),
            }
            for k in klines
        ]

        return pd.DataFrame(rows).set_index("timestamp")

    def fetch_range(
        self,
        symbol: str,
        interval: str,
        start_date: date,
        end_date: date,
        price_type: str = "BID",
    ) -> pd.DataFrame:
        """指定期間の OHLC データをまとめて取得する (ADR 005 #1)."""
        all_dfs: list[pd.DataFrame] = []
        current = start_date
        total_days = (end_date - start_date).days + 1
        fetched = 0

        while current <= end_date:
            try:
                df = self.fetch_klines(symbol, interval, current, price_type)
                if not df.empty:
                    all_dfs.append(df)
            except ApiError as e:
                logger.debug("データ取得スキップ (%s): %s", current.isoformat(), e)

            fetched += 1
            if fetched % 100 == 0:
                logger.info("取得進捗: %d/%d 日", fetched, total_days)

            current += timedelta(days=1)

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs).sort_index()
        result = result[~result.index.duplicated(keep="first")]
        logger.info("取得完了: %d 件 (%s ~ %s)", len(result), start_date, end_date)
        return result

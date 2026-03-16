"""データ収集CLIスクリプト (ADR 005 #1).

初回一括: python scripts/collect_data.py --from 2020-01-01
差分更新: python scripts/collect_data.py --update

中断耐性: 日単位で逐次保存するため、途中で中断しても
取得済みデータはSQLiteに保存される。再実行すれば--updateで再開可能。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta

from fx_auto_trading.config import load_config
from fx_auto_trading.data.collector import GmoFxCollector
from fx_auto_trading.data.storage import OhlcStorage
from fx_auto_trading.exceptions import ApiError
from fx_auto_trading.log import setup_logging

logger = logging.getLogger(__name__)


def _interval_api(interval: str) -> str:
    return {"1H": "1hour", "4H": "4hour", "1D": "1day"}.get(interval, interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="GMO FX API データ収集")
    parser.add_argument(
        "--from",
        dest="start_date",
        type=str,
        help="開始日 (YYYY-MM-DD)。初回一括取得用",
    )
    parser.add_argument(
        "--to",
        dest="end_date",
        type=str,
        default=None,
        help="終了日 (YYYY-MM-DD)。省略時は今日",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="差分更新モード。SQLiteの最新timestamp以降を取得",
    )
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)

    storage = OhlcStorage(config.paths.db)
    collector = GmoFxCollector(config.api)

    symbol = config.trade.symbol
    interval = config.trade.timeframe
    api_interval = _interval_api(interval)

    end = (
        date.fromisoformat(args.end_date)
        if args.end_date
        else date.today()
    )

    if args.update:
        latest = storage.get_latest_timestamp(symbol, api_interval)
        if latest is None:
            print("データなし。--from で初回取得してください。")
            sys.exit(1)
        start = datetime.fromisoformat(latest).date()
        print(f"差分更新: {start} → {end}")
    elif args.start_date:
        start = date.fromisoformat(args.start_date)
        print(f"初回取得: {start} → {end}")
    else:
        parser.print_help()
        sys.exit(1)

    # 日単位で逐次取得・保存（中断耐性）
    current = start
    total_days = (end - start).days + 1
    total_saved = 0
    fetched_days = 0

    while current <= end:
        try:
            df = collector.fetch_klines(symbol, interval, current)
            if not df.empty:
                saved = storage.save(df, symbol, api_interval)
                total_saved += len(df)
        except ApiError as e:
            logger.debug("スキップ (%s): %s", current.isoformat(), e)

        fetched_days += 1
        if fetched_days % 100 == 0:
            db_total = storage.count(symbol, api_interval)
            print(f"進捗: {fetched_days}/{total_days}日 ({fetched_days/total_days*100:.0f}%), DB合計: {db_total}件")

        current += timedelta(days=1)

    db_total = storage.count(symbol, api_interval)
    print(f"完了: {total_saved}件取得, DB合計: {db_total}件")


if __name__ == "__main__":
    main()

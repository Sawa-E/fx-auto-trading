"""毎時予測スクリプト（ステートレス仮想取引版）.

状態ファイル不要。毎回過去8時間を振り返り:
1. 新シグナルの検出 → Discord通知
2. 過去シグナルのSL/TPヒット判定 → Discord決済通知

Usage:
    uv run python scripts/predict.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from fx_auto_trading.data.collector import GmoFxCollector
from fx_auto_trading.features.pipeline import build_features
from fx_auto_trading.log import setup_logging
from fx_auto_trading.notification.discord import (
    send_error,
    send_signal,
    send_trade_result,
)
from fx_auto_trading.trading.engine import check_signals_and_results

setup_logging(os.environ.get("FX_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/production_model_v2.pkl")
META_PATH = Path("models/production_meta_v2.json")


def main() -> None:
    now = datetime.now(tz=UTC)
    logger.info("予測開始: %s UTC", now.strftime("%Y-%m-%d %H:%M"))

    # 1. モデル・メタデータ読み込み
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        with open(META_PATH) as f:
            meta = json.load(f)
    except FileNotFoundError as e:
        msg = f"モデルファイルなし: {e}"
        logger.error(msg)
        send_error(msg)
        sys.exit(1)

    selected_features = meta["selected_features"]
    prob_threshold = meta["probability_threshold"]
    adx_threshold = meta.get("regime_adx_threshold", 25.0)
    tc = meta["trade_config"]

    # 2. 最新データ取得（直近7日分）
    collector = GmoFxCollector()
    end_date = date.today()
    start_date = end_date - timedelta(days=7)

    try:
        df = collector.fetch_range("USD_JPY", "1H", start_date, end_date)
    except Exception as e:
        msg = f"データ取得失敗: {e}"
        logger.error(msg)
        send_error(msg)
        sys.exit(1)

    if df.empty or len(df) < 30:
        logger.warning("データ不足: %d本", len(df))
        return

    # 3. 特徴量計算
    features = build_features(df)

    # 4. シグナル検出 + 決済判定（ステートレス）
    new_signals, completed_trades = check_signals_and_results(
        df=df,
        features=features,
        model=model,
        selected_features=selected_features,
        prob_threshold=prob_threshold,
        adx_threshold=adx_threshold,
        sl_mult=tc["sl_atr_multiplier"],
        tp_mult=tc["tp_atr_multiplier"],
        horizon=8,
        initial_balance=30000,
        leverage=5,
        spread_pips=tc["spread_pips"],
    )

    # 5. Discord通知 - 新シグナル
    for sig in new_signals:
        logger.info(
            "シグナル: %s @ %.2f P=%.2f ADX=%.1f",
            sig.direction, sig.entry_price, sig.probability, sig.adx,
        )
        send_signal(
            direction=sig.direction,
            price=sig.entry_price,
            probability=sig.probability,
            adx=sig.adx,
            atr=sig.atr_value,
            sl_price=sig.sl_price,
            tp_price=sig.tp_price,
            timestamp=sig.timestamp,
        )

    # 6. Discord通知 - 決済結果
    # 通算成績を計算（直近の全completed_tradesから）
    wins = sum(1 for t in completed_trades if t.result == "tp_hit")
    losses = sum(1 for t in completed_trades if t.result == "sl_hit")
    balance = 30000 + sum(t.pnl_yen for t in completed_trades)

    for trade in completed_trades:
        logger.info(
            "決済: %s %s @ %.2f → %.2f %+.0f円",
            trade.signal.direction,
            trade.result,
            trade.signal.entry_price,
            trade.exit_price,
            trade.pnl_yen,
        )
        send_trade_result(
            direction=trade.signal.direction,
            entry_price=trade.signal.entry_price,
            exit_price=trade.exit_price,
            result=trade.result,
            pnl_yen=trade.pnl_yen,
            balance=balance,
            wins=wins,
            losses=losses,
        )

    if not new_signals and not completed_trades:
        logger.info("シグナルなし、決済なし")


if __name__ == "__main__":
    main()

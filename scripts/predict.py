"""毎時予測スクリプト（ステートレス仮想取引版）.

状態ファイル不要。毎回過去8時間を振り返り:
1. 新シグナルの検出 → Discord通知
2. 過去シグナルのSL/TPヒット判定 → Discord決済通知

通知済みのシグナル・決済はnotified.jsonで管理し、重複通知を防止。
累積残高もnotified.jsonで永続化する。

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
STATE_PATH = Path("state/notified.json")


def load_state() -> dict:
    """通知済み状態を読み込む."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "notified_signals": [],
        "notified_trades": [],
        "cumulative_wins": 0,
        "cumulative_losses": 0,
        "cumulative_pnl": 0.0,
        "initial_balance": 30000,
    }


def save_state(state: dict) -> None:
    """通知済み状態を保存する."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


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

    # 2. 通知済み状態を読み込み
    state = load_state()

    # 3. 最新データ取得（直近7日分）
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

    # 4. 特徴量計算
    features = build_features(df)

    # 5. シグナル検出 + 決済判定（ステートレス）
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
        initial_balance=state["initial_balance"],
        leverage=5,
        spread_pips=tc["spread_pips"],
    )

    # 6. Discord通知 - 新シグナル（未通知のみ）
    for sig in new_signals:
        sig_key = f"{sig.timestamp}_{sig.entry_price:.2f}"
        if sig_key in state["notified_signals"]:
            logger.info("シグナル通知済みスキップ: %s", sig_key)
            continue

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
        state["notified_signals"].append(sig_key)

    # 7. Discord通知 - 決済結果（未通知のみ、累積残高）
    for trade in completed_trades:
        trade_key = (
            f"{trade.signal.timestamp}_{trade.signal.entry_price:.2f}"
            f"_{trade.exit_price:.2f}_{trade.result}"
        )
        if trade_key in state["notified_trades"]:
            logger.info("決済通知済みスキップ: %s", trade_key)
            continue

        # 累積更新
        state["cumulative_pnl"] += trade.pnl_yen
        if trade.result == "tp_hit":
            state["cumulative_wins"] += 1
        else:
            state["cumulative_losses"] += 1

        balance = state["initial_balance"] + state["cumulative_pnl"]

        logger.info(
            "決済: %s %s @ %.2f → %.2f %+.0f円 (累積残高: %.0f円)",
            trade.signal.direction,
            trade.result,
            trade.signal.entry_price,
            trade.exit_price,
            trade.pnl_yen,
            balance,
        )
        send_trade_result(
            direction=trade.signal.direction,
            entry_price=trade.signal.entry_price,
            exit_price=trade.exit_price,
            result=trade.result,
            pnl_yen=trade.pnl_yen,
            balance=balance,
            wins=state["cumulative_wins"],
            losses=state["cumulative_losses"],
        )
        state["notified_trades"].append(trade_key)

    # 8. 状態保存
    save_state(state)

    # 通知済みリストが長くなりすぎたら古いものを削除（直近200件保持）
    if len(state["notified_signals"]) > 200:
        state["notified_signals"] = state["notified_signals"][-200:]
    if len(state["notified_trades"]) > 200:
        state["notified_trades"] = state["notified_trades"][-200:]
        save_state(state)

    if not new_signals and not completed_trades:
        logger.info("シグナルなし、決済なし")


if __name__ == "__main__":
    main()

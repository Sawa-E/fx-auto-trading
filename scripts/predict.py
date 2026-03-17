"""毎時予測スクリプト (TASK-002-01, 002-03).

pkl読み込み → 最新データ取得 → 特徴量計算 → 予測 → Discord通知 → CSV記録

Usage:
    uv run python scripts/predict.py
    # or via GitHub Actions (毎時自動実行)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from fx_auto_trading.data.collector import GmoFxCollector
from fx_auto_trading.features.pipeline import build_features
from fx_auto_trading.log import setup_logging
from fx_auto_trading.notification.discord import send_error, send_signal

setup_logging(os.environ.get("FX_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/production_model.pkl")
META_PATH = Path("models/production_meta.json")
CSV_PATH = Path("results/forward_predictions.csv")


def main() -> None:
    now = datetime.now(tz=__import__("datetime").timezone.utc)
    logger.info("予測開始: %s UTC", now.strftime("%Y-%m-%d %H:%M"))

    # 1. モデル・メタデータ読み込み
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
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
    sl_mult = meta["trade_config"]["sl_atr_multiplier"]
    tp_mult = meta["trade_config"]["tp_atr_multiplier"]

    # 2. 最新データ取得（直近7日分 = 特徴量ウォームアップに十分）
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
        logger.warning("データ不足: %d本（最低30本必要）", len(df))
        record_csv(now, 0, "no_data", 0, 0, 0, 0, 0, False)
        return

    # 3. 特徴量計算
    features = build_features(df)
    latest = features.iloc[-1]

    # 欠損チェック
    missing = [c for c in selected_features if pd.isna(latest[c])]
    if missing:
        logger.warning("特徴量欠損: %s", missing)
        record_csv(now, latest["close"], "error", 0, 0, 0, 0, 0, False)
        return

    close = latest["close"]
    adx_val = latest["adx"]
    atr_val = latest["atr"]

    # 4. レジームフィルタ
    regime_ok = adx_val > adx_threshold

    if not regime_ok:
        logger.info(
            "見送り（レンジ相場）: ADX=%.1f < %.0f",
            adx_val, adx_threshold,
        )
        record_csv(now, close, "hold", 0.5, adx_val, atr_val, 0, 0, False)
        return

    # 5. 予測
    X = pd.DataFrame([latest[selected_features].values], columns=selected_features)
    prob = float(model.predict_proba(X)[0][1])

    # 6. シグナル判定
    if prob >= prob_threshold:
        direction = "buy"
        sl_price = close - atr_val * sl_mult
        tp_price = close + atr_val * tp_mult
    elif prob <= (1 - prob_threshold):
        direction = "sell"
        sl_price = close + atr_val * sl_mult
        tp_price = close - atr_val * tp_mult
    else:
        direction = "hold"
        sl_price = 0
        tp_price = 0

    logger.info(
        "結果: %s | price=%.2f | P=%.3f | ADX=%.1f | ATR=%.3f",
        direction, close, prob, adx_val, atr_val,
    )

    # 7. Discord通知（シグナル発生時のみ）
    if direction in ("buy", "sell"):
        send_signal(
            direction=direction,
            price=close,
            probability=prob,
            adx=adx_val,
            atr=atr_val,
            sl_price=sl_price,
            tp_price=tp_price,
            timestamp=now.strftime("%Y-%m-%d %H:%M UTC"),
        )

    # 8. CSV記録
    record_csv(
        now, close, direction, prob, adx_val, atr_val,
        sl_price, tp_price, regime_ok,
    )


def record_csv(
    timestamp: datetime,
    price: float,
    direction: str,
    probability: float,
    adx: float,
    atr: float,
    sl_price: float,
    tp_price: float,
    regime_ok: bool,
) -> None:
    """予測結果をCSVに追記する."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    header_needed = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with open(CSV_PATH, "a") as f:
        if header_needed:
            f.write(
                "timestamp,price,direction,probability,"
                "adx,atr,sl_price,tp_price,regime_ok\n"
            )
        f.write(
            f"{timestamp.isoformat()},"
            f"{price:.2f},{direction},{probability:.4f},"
            f"{adx:.2f},{atr:.4f},"
            f"{sl_price:.2f},{tp_price:.2f},"
            f"{regime_ok}\n"
        )
    logger.info("CSV記録: %s", CSV_PATH)


if __name__ == "__main__":
    main()

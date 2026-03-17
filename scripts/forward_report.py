"""フォワードテスト結果レポート.

指定期間の全シグナルを再計算し、バックテスト結果と比較する。

Usage:
    python scripts/forward_report.py --from 2026-03-19 --to 2026-04-18
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from fx_auto_trading.data.collector import GmoFxCollector
from fx_auto_trading.features.pipeline import build_features
from fx_auto_trading.log import setup_logging
from fx_auto_trading.trading.engine import check_signals_and_results


def main() -> None:
    parser = argparse.ArgumentParser(description="フォワードテスト結果レポート")
    parser.add_argument("--from", dest="start_date", required=True)
    parser.add_argument("--to", dest="end_date", required=True)
    args = parser.parse_args()

    setup_logging()

    # モデル読み込み
    with open("models/production_model.pkl", "rb") as f:
        model = pickle.load(f)  # noqa: S301
    with open("models/production_meta.json") as f:
        meta = json.load(f)

    selected = meta["selected_features"]
    tc = meta["trade_config"]

    # データ取得（ウォームアップ用に開始日の7日前から）
    collector = GmoFxCollector()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    fetch_start = start - timedelta(days=7)

    print(f"データ取得: {fetch_start} ~ {end}")
    df = collector.fetch_range("USD_JPY", "1H", fetch_start, end)
    print(f"取得: {len(df)}本")

    if len(df) < 50:
        print("データ不足")
        sys.exit(1)

    features = build_features(df)

    # 対象期間のみで分析
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    mask = (features.index >= start_ts) & (features.index < end_ts)
    period_features = features[mask]

    # 全足でシグナルとSL/TPを判定
    all_trades = []

    for i in range(len(period_features)):
        # 各足を「最新足」として扱い、過去を振り返る
        subset_end = period_features.index[i]
        subset_features = features[features.index <= subset_end].tail(12)
        subset_df = df[df.index <= subset_end].tail(20)

        if len(subset_features) < 10:
            continue

        signals, trades = check_signals_and_results(
            df=subset_df,
            features=subset_features,
            model=model,
            selected_features=selected,
            prob_threshold=meta["probability_threshold"],
            adx_threshold=meta.get("regime_adx_threshold", 25.0),
            sl_mult=tc["sl_atr_multiplier"],
            tp_mult=tc["tp_atr_multiplier"],
            horizon=8,
            initial_balance=30000,
            leverage=5,
            spread_pips=tc["spread_pips"],
        )
        all_trades.extend(trades)

    # 重複除去（同じシグナルが複数回検出される）
    seen = set()
    unique_trades = []
    for t in all_trades:
        key = t.signal.timestamp
        if key not in seen:
            seen.add(key)
            unique_trades.append(t)

    # 集計
    wins = sum(1 for t in unique_trades if t.result == "tp_hit")
    losses = sum(1 for t in unique_trades if t.result == "sl_hit")
    total = wins + losses
    pnls = [t.pnl_yen for t in unique_trades]

    if total == 0:
        print("取引なし")
        return

    wr = wins / total
    pnl_arr = np.array(pnls)
    sharpe = (
        float(pnl_arr.mean() / pnl_arr.std()) if pnl_arr.std() > 0 else 0
    )
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    balance = 30000 + sum(pnls)
    trading_days = (end - start).days

    print()
    print("=" * 50)
    print("  フォワードテスト結果")
    print(f"  期間: {start} ~ {end} ({trading_days}日間)")
    print("=" * 50)
    print(f"  取引回数:  {total}回 ({total/trading_days:.1f}回/日)")
    print(f"  勝敗:      {wins}勝{losses}敗")
    print(f"  勝率:      {wr:.1%}")
    print(f"  PF:        {pf:.2f}")
    print(f"  Sharpe:    {sharpe:.3f}")
    print(f"  累積損益:  {sum(pnls):+,.0f}円")
    print(f"  最終残高:  {balance:,.0f}円 ({(balance-30000)/30000*100:+.1f}%)")
    print("=" * 50)

    # バックテスト比較
    print()
    bt_wr, bt_pf, bt_daily = 0.601, 2.01, 3.5
    wr_ok = "✓" if wr >= 0.55 else "✗"
    pf_ok = "✓" if pf >= 1.5 else "✗"
    daily_ok = "✓" if total / trading_days >= 1.0 else "✗"

    print(f"{'指標':>12} {'バックテスト':>12} {'フォワード':>12} {'判定':>6}")
    print("-" * 48)
    print(f"{'勝率':>12} {bt_wr:>11.1%} {wr:>11.1%} {wr_ok:>6}")
    print(f"{'PF':>12} {bt_pf:>12.2f} {pf:>12.2f} {pf_ok:>6}")
    print(
        f"{'回/日':>12} {bt_daily:>12.1f} "
        f"{total/trading_days:>12.1f} {daily_ok:>6}"
    )


if __name__ == "__main__":
    main()

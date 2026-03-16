"""簡易バックテスト (ADR 003 #1-4).

ATRベース損切り/利確、確率フィルタ、DD停止。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """1取引の記録."""

    entry_time: str
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl_pips: float
    win: bool


@dataclass
class BacktestResult:
    """バックテスト結果."""

    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.win) / len(self.trades)

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        pnls = np.array([t.pnl_pips for t in self.trades])
        return float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
        return float(dd.max())

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pips for t in self.trades if t.win)
        gross_loss = abs(sum(t.pnl_pips for t in self.trades if not t.win))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def run_backtest(
    ohlc: pd.DataFrame,
    y_prob: np.ndarray,
    timestamps: pd.DatetimeIndex,
    atr_series: pd.Series,
    threshold: float = 0.60,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.0,
    max_drawdown_limit: float = 0.10,
    spread_pips: float = 0.3,
    pip_value: float = 0.01,
) -> BacktestResult:
    """簡易バックテストを実行する.

    ADR 003:
      #1: ATRベース損切り/利確 (SL=ATR*1.5, TP=ATR*2.0)
      #2: レバレッジ5倍以下 (ここでは1取引=1単位で固定)
      #3: DD 10%で停止
      #4: 確率フィルタ (threshold=0.60)
    """
    result = BacktestResult()
    equity = 0.0
    peak_equity = 0.0

    for i in range(len(y_prob)):
        prob = y_prob[i]
        ts = timestamps[i]

        # ATRが取得できない場合はスキップ
        if ts not in atr_series.index or pd.isna(atr_series[ts]):
            continue

        current_atr = atr_series[ts]
        entry_price = ohlc.loc[ts, "close"] if ts in ohlc.index else None
        if entry_price is None:
            continue

        # 確率フィルタ (ADR 003 #4)
        pred_up = prob >= threshold
        pred_down = prob < (1 - threshold)

        if not pred_up and not pred_down:
            continue

        # DD停止チェック (ADR 003 #3)
        dd_ratio = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd_ratio > max_drawdown_limit:
            logger.warning("DD停止: equity=%.2f, peak=%.2f", equity, peak_equity)
            break

        direction = "long" if pred_up else "short"
        sl_pips = current_atr / pip_value * sl_atr_mult
        tp_pips = current_atr / pip_value * tp_atr_mult

        # 簡易判定: 次の足で方向が合っていればTP、外れていればSL
        # 実際のバックテストでは足内の高値/安値で判定すべきだが、
        # Phase 1では簡易版とする
        # y_prob[i] の予測が正しかったかは y_true で判定済み
        # ここでは ATR ベースの pnl を計算
        if pred_up:
            tp_pips - spread_pips  # 仮に勝ちの場合
        else:
            tp_pips - spread_pips

        # 実際の勝敗は後から付与する必要がある
        # ここではシンプルに確率でシミュレーション
        # → WFResultのy_trueと組み合わせて使う
        trade = Trade(
            entry_time=str(ts),
            direction=direction,
            entry_price=entry_price,
            exit_price=0.0,  # Phase 1では簡易版
            sl=sl_pips,
            tp=tp_pips,
            pnl_pips=0.0,  # 後で設定
            win=False,  # 後で設定
        )
        result.trades.append(trade)

    return result


def backtest_from_wf_results(
    wf_results: list,
    threshold: float = 0.60,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.0,
    max_drawdown_limit: float = 0.10,
    spread_pips: float = 0.3,
) -> BacktestResult:
    """WFResultリストから簡易バックテストを実行する."""
    result = BacktestResult()
    equity = 0.0
    peak_equity = 0.0

    for wf in wf_results:
        for prob, actual in zip(wf.y_prob, wf.y_true, strict=False):
            pred_up = prob >= threshold
            pred_down = prob < (1 - threshold)

            if not pred_up and not pred_down:
                continue

            # DD停止 (ADR 003 #3)
            dd_ratio = (
                (peak_equity - equity) / peak_equity
                if peak_equity > 0
                else 0.0
            )
            if dd_ratio > max_drawdown_limit:
                logger.warning("DD停止: equity=%.2f", equity)
                result.equity_curve.append(equity)
                return result

            win = actual == 1 if pred_up else actual == 0

            pnl = tp_atr_mult - spread_pips if win else -(sl_atr_mult + spread_pips)

            trade = Trade(
                entry_time="",
                direction="long" if pred_up else "short",
                entry_price=0.0,
                exit_price=0.0,
                sl=sl_atr_mult,
                tp=tp_atr_mult,
                pnl_pips=pnl,
                win=win,
            )
            result.trades.append(trade)

            equity += pnl
            peak_equity = max(peak_equity, equity)
            result.equity_curve.append(equity)

    return result

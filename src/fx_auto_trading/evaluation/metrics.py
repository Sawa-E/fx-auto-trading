"""評価メトリクス + CSVエクスポート (ADR 002 #4, ADR 003 #4).

シャープレシオをメイン指標とし、勝率・RR比・最大DD等も算出。
結果をCSVにエクスポート（Phase 1.5）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fx_auto_trading.evaluation.walk_forward import WFResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """統合評価結果."""

    accuracy: float
    win_rate: float
    risk_reward_ratio: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    threshold_used: float


def evaluate_walk_forward(
    results: list[WFResult],
    threshold: float = 0.60,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
) -> EvaluationResult:
    """ウォークフォワード結果を統合評価する."""
    all_true: list[int] = []
    all_prob: list[float] = []

    for r in results:
        all_true.extend(r.y_true.tolist())
        all_prob.extend(r.y_prob.tolist())

    y_true = np.array(all_true)
    y_prob = np.array(all_prob)

    # 全予測の正答率
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = float((y_pred == y_true).mean())

    # 確率フィルタ後の取引
    trades_returns = []
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0

    for prob, actual in zip(y_prob, y_true, strict=False):
        pred_up = prob >= threshold
        pred_down = prob < (1 - threshold)

        if not pred_up and not pred_down:
            continue

        win = actual == 1 if pred_up else actual == 0

        if win:
            wins += 1
            ret = tp_mult
            total_profit += tp_mult
        else:
            losses += 1
            ret = -sl_mult
            total_loss += sl_mult

        trades_returns.append(ret)

    total_trades = wins + losses
    if total_trades == 0:
        return EvaluationResult(
            accuracy=accuracy,
            win_rate=0.0,
            risk_reward_ratio=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
            total_trades=0,
            threshold_used=threshold,
        )

    win_rate = wins / total_trades
    rr_ratio = tp_mult / sl_mult
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    arr = np.array(trades_returns)
    sharpe = float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0

    # 最大ドローダウン
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(cumulative)
    dd = (peak - cumulative)
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    # ピーク比でDD計算（ピークが0以下なら絶対値）
    max_dd_pct = max_dd / peak.max() if peak.max() > 0 else 0.0

    return EvaluationResult(
        accuracy=accuracy,
        win_rate=win_rate,
        risk_reward_ratio=rr_ratio,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd_pct,
        profit_factor=profit_factor,
        total_trades=total_trades,
        threshold_used=threshold,
    )


def optimize_threshold(
    results: list[WFResult],
    search_range: tuple[float, float] = (0.50, 0.70),
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
) -> tuple[float, EvaluationResult]:
    """確率閾値を探索し、最適なシャープレシオを返す (ADR 003 #4)."""
    best_threshold = search_range[0]
    best_result = evaluate_walk_forward(results, best_threshold, sl_mult, tp_mult)

    for t in np.arange(search_range[0], search_range[1] + 0.01, 0.01):
        r = evaluate_walk_forward(results, float(t), sl_mult, tp_mult)
        if r.sharpe_ratio > best_result.sharpe_ratio and r.total_trades >= 10:
            best_threshold = float(t)
            best_result = r

    logger.info(
        "最適閾値: %.2f → Sharpe=%.3f, 勝率=%.1f%%, 取引数=%d",
        best_threshold,
        best_result.sharpe_ratio,
        best_result.win_rate * 100,
        best_result.total_trades,
    )
    return best_threshold, best_result


def export_csv(
    results: list[WFResult],
    eval_result: EvaluationResult,
    output_dir: str = "results",
) -> None:
    """結果をCSVにエクスポートする (Phase 1.5)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    rows = []
    for r in results:
        for i in range(len(r.y_true)):
            rows.append({
                "period": r.period_start,
                "y_true": int(r.y_true[i]),
                "y_prob": float(r.y_prob[i]),
                "y_pred": int(r.y_prob[i] >= 0.5),
            })
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)

    # metrics.csv
    metrics = {
        "accuracy": eval_result.accuracy,
        "win_rate": eval_result.win_rate,
        "risk_reward_ratio": eval_result.risk_reward_ratio,
        "sharpe_ratio": eval_result.sharpe_ratio,
        "max_drawdown": eval_result.max_drawdown,
        "profit_factor": eval_result.profit_factor,
        "total_trades": eval_result.total_trades,
        "threshold": eval_result.threshold_used,
    }
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)

    # feature_importance.csv
    if results:
        all_fi: dict[str, list[float]] = {}
        for r in results:
            for k, v in r.feature_importance.items():
                all_fi.setdefault(k, []).append(v)
        avg_fi = {k: np.mean(v) for k, v in all_fi.items()}
        fi_df = pd.Series(avg_fi).sort_values(ascending=False)
        fi_df.to_csv(out / "feature_importance.csv", header=["importance"])

    logger.info("CSV出力完了: %s", out)


def print_report(eval_result: EvaluationResult) -> None:
    """CLIレポートを出力する."""
    print("=" * 50)
    print("評価結果")
    print("=" * 50)
    print(f"  正答率:             {eval_result.accuracy:.1%}")
    print(f"  勝率(P>{eval_result.threshold_used:.2f}): {eval_result.win_rate:.1%}")
    print(f"  リスクリワード比:    {eval_result.risk_reward_ratio:.2f}")
    print(f"  シャープレシオ:      {eval_result.sharpe_ratio:.3f}")
    print(f"  最大ドローダウン:    {eval_result.max_drawdown:.1%}")
    print(f"  プロフィットファクター: {eval_result.profit_factor:.2f}")
    print(f"  取引回数:           {eval_result.total_trades}")
    print("=" * 50)

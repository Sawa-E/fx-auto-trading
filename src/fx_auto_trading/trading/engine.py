"""ステートレス仮想取引エンジン.

状態ファイル不要。毎回過去N時間を振り返り、
シグナルの発生とSL/TPヒットを判定する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """シグナル情報."""

    timestamp: str
    direction: str  # "buy" or "sell"
    entry_price: float
    probability: float
    adx: float
    atr_value: float
    sl_price: float
    tp_price: float


@dataclass
class TradeResult:
    """取引結果."""

    signal: Signal
    exit_price: float
    exit_time: str
    result: str  # "tp_hit", "sl_hit", "open"
    pnl_yen: float


def check_signals_and_results(
    df: pd.DataFrame,
    features: pd.DataFrame,
    model: object,
    selected_features: list[str],
    prob_threshold: float = 0.55,
    adx_threshold: float = 25.0,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
    horizon: int = 8,
    initial_balance: float = 30000,
    leverage: float = 5,
    spread_pips: float = 0.3,
) -> tuple[list[Signal], list[TradeResult]]:
    """直近の足を分析し、新シグナルと決済済み取引を返す.

    ステートレス: 毎回過去horizon+数時間を振り返り、
    シグナルの発生→SL/TPヒットを再計算する。

    Returns:
        (new_signals, completed_trades):
            new_signals: 最新足で新たに発生したシグナル
            completed_trades: 過去のシグナルで今回決済が確定した取引
    """
    if len(features) < horizon + 2:
        return [], []

    new_signals: list[Signal] = []
    completed_trades: list[TradeResult] = []

    pip_value = 0.01

    # 直近horizon+2本を対象に、各足でシグナルを判定
    lookback_start = max(0, len(features) - horizon - 2)

    for i in range(lookback_start, len(features)):
        row = features.iloc[i]
        ts = features.index[i]

        # 特徴量の欠損チェック
        if any(pd.isna(row[col]) for col in selected_features):
            continue

        # レジームフィルタ
        adx_val = row["adx"]
        if adx_val <= adx_threshold:
            continue

        # 予測
        X = pd.DataFrame(
            [row[selected_features].values], columns=selected_features
        )
        prob = float(model.predict_proba(X)[0][1])  # type: ignore[union-attr]

        close = row["close"]
        atr_val = row["atr"]

        # シグナル判定
        if prob >= prob_threshold:
            direction = "buy"
            sl_price = close - atr_val * sl_mult
            tp_price = close + atr_val * tp_mult
        elif prob <= (1 - prob_threshold):
            direction = "sell"
            sl_price = close + atr_val * sl_mult
            tp_price = close - atr_val * tp_mult
        else:
            continue

        signal = Signal(
            timestamp=str(ts),
            direction=direction,
            entry_price=close,
            probability=prob,
            adx=adx_val,
            atr_value=atr_val,
            sl_price=sl_price,
            tp_price=tp_price,
        )

        # 最新足のシグナルかどうか
        is_latest = i == len(features) - 1

        if is_latest:
            new_signals.append(signal)
            continue

        # 過去のシグナル → SL/TPヒットを判定
        # エントリー後のhorizon本分の足を確認
        entry_idx = i
        lot_size = (initial_balance * leverage) / close

        result = "open"
        exit_price = 0.0
        exit_time = ""
        pnl_yen = 0.0

        for j in range(entry_idx + 1, min(entry_idx + horizon + 1, len(df))):
            if j >= len(df):
                break

            bar = df.iloc[j]
            bar_high = bar["high"]
            bar_low = bar["low"]
            bar_ts = df.index[j]

            if direction == "buy":
                # SLチェック（安値がSL以下）
                if bar_low <= sl_price:
                    result = "sl_hit"
                    exit_price = sl_price
                    exit_time = str(bar_ts)
                    pnl_pips = -(sl_mult * atr_val / pip_value + spread_pips)
                    pnl_yen = pnl_pips * pip_value * lot_size
                    break
                # TPチェック（高値がTP以上）
                if bar_high >= tp_price:
                    result = "tp_hit"
                    exit_price = tp_price
                    exit_time = str(bar_ts)
                    pnl_pips = tp_mult * atr_val / pip_value - spread_pips
                    pnl_yen = pnl_pips * pip_value * lot_size
                    break
            else:  # sell
                if bar_high >= sl_price:
                    result = "sl_hit"
                    exit_price = sl_price
                    exit_time = str(bar_ts)
                    pnl_pips = -(sl_mult * atr_val / pip_value + spread_pips)
                    pnl_yen = pnl_pips * pip_value * lot_size
                    break
                if bar_low <= tp_price:
                    result = "tp_hit"
                    exit_price = tp_price
                    exit_time = str(bar_ts)
                    pnl_pips = tp_mult * atr_val / pip_value - spread_pips
                    pnl_yen = pnl_pips * pip_value * lot_size
                    break

        # horizon足経過してもSL/TPに到達しなかった場合 → 終値で決済
        if result == "open" and entry_idx + horizon < len(df):
            close_bar = df.iloc[entry_idx + horizon]
            exit_price = close_bar["close"]
            exit_time = str(df.index[entry_idx + horizon])
            diff = exit_price - close if direction == "buy" else close - exit_price
            pnl_pips = diff / pip_value - spread_pips
            pnl_yen = pnl_pips * pip_value * lot_size
            result = "tp_hit" if pnl_yen > 0 else "sl_hit"

        if result != "open":
            # 最新足で決済が確定したもののみ通知対象
            exit_bar_idx = entry_idx + horizon
            if exit_bar_idx >= len(features) - 2:
                completed_trades.append(
                    TradeResult(
                        signal=signal,
                        exit_price=exit_price,
                        exit_time=exit_time,
                        result=result,
                        pnl_yen=pnl_yen,
                    )
                )

    return new_signals, completed_trades

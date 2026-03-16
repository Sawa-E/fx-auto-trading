"""モメンタム + ラグ特徴量 — カテゴリD: 3個 + カテゴリE: 6個 (ADR 006 #2).

D: roc_6, roc_12, up_ratio_6
E: return_lag1〜lag6 (v1の12期から半減)
"""

from __future__ import annotations

import pandas as pd


def add_momentum_features(
    close: pd.Series,
    roc_periods: tuple[int, ...] = (6, 12),
    up_ratio_period: int = 6,
) -> dict[str, pd.Series]:
    """モメンタム系特徴量を生成する."""
    result: dict[str, pd.Series] = {}

    for period in roc_periods:
        result[f"roc_{period}"] = close.pct_change(period)

    up = (close.diff() > 0).astype(float)
    result[f"up_ratio_{up_ratio_period}"] = up.rolling(up_ratio_period).mean()

    return result


def add_lag_features(
    close: pd.Series,
    lag_periods: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
) -> dict[str, pd.Series]:
    """ラグ特徴量を生成する (ADR 006 #2: 1-6期)."""
    returns = close.pct_change()
    return {f"return_lag{lag}": returns.shift(lag) for lag in lag_periods}

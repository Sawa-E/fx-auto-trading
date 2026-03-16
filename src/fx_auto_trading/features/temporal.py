"""時間帯特徴量 — カテゴリC: 5個 (ADR 006).

hour, day_of_week, is_tokyo, is_london, is_ny
"""

from __future__ import annotations

import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """時間帯特徴量を生成する."""
    idx = pd.DatetimeIndex(df.index)
    hour = pd.Series(idx.hour, index=idx, dtype=float)
    dow = pd.Series(idx.dayofweek, index=idx, dtype=float)
    is_tokyo = ((idx.hour >= 0) & (idx.hour < 9)).astype(float)
    is_london = ((idx.hour >= 8) & (idx.hour < 16)).astype(float)
    is_ny = ((idx.hour >= 13) & (idx.hour < 22)).astype(float)

    return {
        "hour": hour,
        "day_of_week": dow,
        "is_tokyo": pd.Series(is_tokyo, index=idx),
        "is_london": pd.Series(is_london, index=idx),
        "is_ny": pd.Series(is_ny, index=idx),
    }

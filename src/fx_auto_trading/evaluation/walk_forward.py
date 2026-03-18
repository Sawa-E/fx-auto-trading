"""ウォークフォワード検証 — 拡大窓 / スライディングウィンドウ.

資料5: 「1990年〜予測対象の1つ手前まで」方式（拡大窓）。
資料3: 「パラメータαは時間変化する」→ スライディングウィンドウで直近に適応。
各ウィンドウでOptuna最適化 + 予測。1ヶ月ごとにスライド。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fx_auto_trading.config import LabelConfig, ModelConfig, TradeConfig
from fx_auto_trading.exceptions import OverfitWarning
from fx_auto_trading.features.pipeline import build_dataset
from fx_auto_trading.models.trainer import LightGBMTrainer

logger = logging.getLogger(__name__)


@dataclass
class WFResult:
    """ウォークフォワード1期間の結果."""

    period_start: str
    period_end: str
    y_true: np.ndarray
    y_prob: np.ndarray
    feature_names: list[str]
    feature_importance: dict[str, float]
    train_size: int
    val_size: int


def walk_forward_validate(
    df: pd.DataFrame,
    horizon: int = 4,
    feature_columns: list[str] | None = None,
    label_config: LabelConfig | None = None,
    model_config: ModelConfig | None = None,
    trade_config: TradeConfig | None = None,
    val_months: int = 1,
    min_train_size: int = 500,
    sliding_window_years: int | None = None,
    calibration: str | None = None,
    fixed_params: dict | None = None,
) -> list[WFResult]:
    """ウォークフォワード検証を実行する.

    Args:
        df: OHLCデータ (timestamp index)
        horizon: 予測ホライゾン
        feature_columns: 使用する特徴量のリスト (Noneなら全28個)
        label_config: ラベル設定
        model_config: モデル設定
        trade_config: 取引設定
        val_months: 検証ウィンドウの月数
        min_train_size: 最小訓練サンプル数
        sliding_window_years: Noneなら拡大窓、整数ならスライディングウィンドウ(年)。
            資料3: パラメータαの時間変化に対応するため、直近N年のみ使用。
        calibration: 確率キャリブレーション ('sigmoid', 'isotonic', None)。
            資料5: 確率出力の保守的な集中を補正。
        fixed_params: Optunaを使わず固定パラメータで訓練する場合に指定。
            資料5: チューニングのやりすぎは過学習の原因。

    Returns:
        各期間のWFResult リスト
    """
    if label_config is None:
        label_config = LabelConfig()

    # 全データセット生成
    X_all, y_all = build_dataset(
        df,
        horizon=horizon,
        label_config=label_config,
        feature_columns=feature_columns,
    )

    if len(X_all) < min_train_size + 100:
        logger.warning(
            "データ不足: %d件 < 最小訓練%d + 検証100件",
            len(X_all),
            min_train_size,
        )
        return []

    # 月ごとの境界を取得
    monthly_groups = X_all.groupby(pd.Grouper(freq="MS"))
    month_starts = sorted(monthly_groups.groups.keys())

    results: list[WFResult] = []

    for i, val_start in enumerate(month_starts):
        # 訓練窓の決定
        if sliding_window_years is not None:
            # スライディングウィンドウ（資料3: 直近N年のみ）
            window_start = val_start - pd.DateOffset(years=sliding_window_years)
            train_mask = (X_all.index >= window_start) & (X_all.index < val_start)
        else:
            # 拡大窓（従来方式）
            train_mask = X_all.index < val_start

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]

        if len(X_train) < min_train_size:
            continue

        # 検証: val_start 〜 次の月の手前まで
        if i + val_months < len(month_starts):
            val_end = month_starts[i + val_months]
            val_mask = (X_all.index >= val_start) & (X_all.index < val_end)
        else:
            val_mask = X_all.index >= val_start

        X_val = X_all[val_mask]
        y_val = y_all[val_mask]

        if len(X_val) < 10:
            continue

        window_type = f"SW{sliding_window_years}Y" if sliding_window_years else "拡大"
        logger.info(
            "WF期間 %d (%s): train=%d件, val=%d件 [%s〜]",
            len(results) + 1,
            window_type,
            len(X_train),
            len(X_val),
            str(val_start.date()),
        )

        trainer = LightGBMTrainer(model_config, trade_config)

        if fixed_params is not None:
            # 固定パラメータ（資料5: チューニングやりすぎ防止）
            params = fixed_params
        else:
            # Optuna最適化
            trainer.optimize(X_train, y_train, X_val, y_val)
            params = None

        try:
            trainer.train(
                X_train, y_train, X_val, y_val,
                params=params, calibration=calibration,
            )
        except OverfitWarning as e:
            logger.warning("WF期間 %d: %s → スキップ", len(results) + 1, e)
            continue

        y_prob = trainer.predict_proba(X_val)
        fi = trainer.feature_importance(X_val.columns.tolist())

        results.append(
            WFResult(
                period_start=str(val_start.date()),
                period_end=str(X_val.index[-1].date()),
                y_true=y_val.values,
                y_prob=y_prob,
                feature_names=X_val.columns.tolist(),
                feature_importance=fi.to_dict(),
                train_size=len(X_train),
                val_size=len(X_val),
            )
        )

    logger.info("ウォークフォワード完了: %d 期間", len(results))
    return results

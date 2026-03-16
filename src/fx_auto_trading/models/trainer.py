"""LightGBMトレーナー + Optuna HPO (ADR 004 #1, #2, ADR 002 #4).

ADR 004 #1: Kaggleデフォルトより強めの正則化。
ADR 004 #2: 訓練/検証の精度差 > 10% で過学習警告。
ADR 002 #4: 最適化目標はシャープレシオ。
"""

from __future__ import annotations

import logging
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score

from fx_auto_trading.config import ModelConfig, TradeConfig
from fx_auto_trading.exceptions import OverfitWarning

logger = logging.getLogger(__name__)

# Optuna と LightGBM の冗長なログを抑制
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _simple_sharpe(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    pip_value: float = 0.01,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
) -> float:
    """簡易シャープレシオ計算（Optuna目的関数用）.

    確率フィルタ後の仮想リターンからシャープレシオを算出。
    """
    returns = []
    for prob, actual in zip(y_prob, y_true, strict=False):
        pred_up = prob >= threshold
        pred_down = prob < (1 - threshold)

        if not pred_up and not pred_down:
            continue  # 見送り

        win = actual == 1 if pred_up else actual == 0

        ret = tp_mult if win else -sl_mult
        returns.append(ret)

    if len(returns) < 10:
        return -999.0

    arr = np.array(returns)
    if arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std())


class LightGBMTrainer:
    """LightGBM 訓練・予測クラス."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        trade_config: TradeConfig | None = None,
    ) -> None:
        self._mc = model_config or ModelConfig()
        self._tc = trade_config or TradeConfig()
        self.model: lgb.LGBMClassifier | None = None
        self.best_params: dict | None = None

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Optunaでハイパーパラメータ最適化 (目標: シャープレシオ)."""
        mc = self._mc
        tc = self._tc

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int(
                    "max_depth", *mc.max_depth_range
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", *mc.min_child_samples_range
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves", *mc.num_leaves_range
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", *mc.reg_lambda_range, log=True
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", *mc.reg_alpha_range, log=True
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *mc.learning_rate_range, log=True
                ),
                "n_estimators": mc.n_estimators,
                "random_state": mc.random_state,
                "verbosity": -1,
            }

            model = lgb.LGBMClassifier(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(mc.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )

            y_prob = model.predict_proba(X_val)[:, 1]
            sharpe = _simple_sharpe(
                y_val,
                y_prob,
                tc.probability_threshold,
                sl_mult=tc.sl_atr_multiplier,
                tp_mult=tc.tp_atr_multiplier,
            )
            return sharpe

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=mc.n_optuna_trials, show_progress_bar=False)

        self.best_params = study.best_params
        logger.info(
            "Optuna完了: best_sharpe=%.3f, params=%s",
            study.best_value,
            study.best_params,
        )
        return study.best_params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict | None = None,
    ) -> lgb.LGBMClassifier:
        """LightGBMを訓練し、過学習チェックを実行する."""
        if params is None:
            params = self.best_params or {}

        full_params = {
            "n_estimators": self._mc.n_estimators,
            "random_state": self._mc.random_state,
            "verbosity": -1,
            **params,
        }

        model = lgb.LGBMClassifier(**full_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(self._mc.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

        self.model = model

        # 過学習検出 (ADR 004 #2)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        gap = train_acc - val_acc

        logger.info(
            "LightGBM訓練完了: train_acc=%.3f, val_acc=%.3f, gap=%.3f",
            train_acc,
            val_acc,
            gap,
        )

        if gap > self._mc.overfit_threshold:
            msg = (
                f"過学習警告: 精度差 {gap:.1%} > {self._mc.overfit_threshold:.0%} "
                f"(train={train_acc:.3f}, val={val_acc:.3f})"
            )
            logger.warning(msg)
            raise OverfitWarning(msg)

        return model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """予測確率を返す."""
        if self.model is None:
            raise RuntimeError("モデルが訓練されていません")
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        """変数重要度を返す."""
        if self.model is None:
            raise RuntimeError("モデルが訓練されていません")
        return pd.Series(
            self.model.feature_importances_,
            index=feature_names,
        ).sort_values(ascending=False)

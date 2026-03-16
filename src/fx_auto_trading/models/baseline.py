"""RFベースライン + 変数重要度刈り込み (ADR 001 #3, ADR 002 #3).

資料4: RF変数重要度で勝率>RR比>レバレッジの順位付けに成功。
資料5: RF→XGBoostの段階的プロセスで精度大幅向上。
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


def train_rf_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, pd.Series]:
    """RFでベースライン予測 + 変数重要度を返す.

    Returns:
        (model, feature_importance): 訓練済みモデルと変数重要度
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    logger.info("RF ベースライン: train_acc=%.3f, val_acc=%.3f", train_acc, val_acc)
    logger.info(
        "RF 検証セット:\n%s",
        classification_report(y_val, model.predict(X_val), zero_division=0),
    )

    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    logger.info("変数重要度 TOP 15:\n%s", importance.head(15).to_string())

    return model, importance


def select_top_features(
    importance: pd.Series,
    top_n: int = 15,
) -> list[str]:
    """変数重要度の上位N個の特徴量名を返す (ADR 002 #3)."""
    selected = importance.head(top_n).index.tolist()
    logger.info("選択された特徴量 (%d個): %s", len(selected), selected)
    return selected

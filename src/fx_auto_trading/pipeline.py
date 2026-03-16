"""エンドツーエンドパイプライン.

データ収集 → 特徴量 → 訓練 → 検証 → 評価 の一気通貫実行。
"""

from __future__ import annotations

import logging

from fx_auto_trading.config import AppConfig, load_config
from fx_auto_trading.data.storage import OhlcStorage
from fx_auto_trading.evaluation.backtest import backtest_from_wf_results
from fx_auto_trading.evaluation.metrics import (
    export_csv,
    optimize_threshold,
    print_report,
)
from fx_auto_trading.evaluation.walk_forward import walk_forward_validate
from fx_auto_trading.features.pipeline import FEATURE_COLUMNS, build_dataset
from fx_auto_trading.log import setup_logging
from fx_auto_trading.models.baseline import select_top_features, train_rf_baseline

logger = logging.getLogger(__name__)


def run_evaluate(
    config: AppConfig | None = None,
    horizon: int = 4,
    top_n_features: int = 15,
) -> None:
    """検証 + 評価パイプラインを実行する."""
    if config is None:
        config = load_config()

    setup_logging(config.log_level)

    # 1. データ読み込み
    storage = OhlcStorage(config.paths.db)
    df = storage.load(config.trade.symbol, "1hour")
    total = len(df)
    logger.info("データ読み込み: %d 件", total)

    if total < 1000:
        logger.error(
            "データ不足: %d件。最低1000件必要。"
            "collect_data.pyでデータを取得してください。",
            total,
        )
        return

    # 2. RFベースラインで変数重要度 → 特徴量選択
    logger.info("=== RFベースライン + 変数重要度 ===")
    X_all, y_all = build_dataset(
        df,
        horizon=horizon,
        label_config=config.label,
        feature_columns=FEATURE_COLUMNS,
    )
    split = int(len(X_all) * 0.8)
    _, importance = train_rf_baseline(
        X_all.iloc[:split],
        y_all.iloc[:split],
        X_all.iloc[split:],
        y_all.iloc[split:],
        random_state=config.model.random_state,
    )
    selected_features = select_top_features(importance, top_n=top_n_features)

    # 3. ウォークフォワード検証（選択された特徴量で）
    logger.info("=== ウォークフォワード検証 (horizon=%d) ===", horizon)
    wf_results = walk_forward_validate(
        df,
        horizon=horizon,
        feature_columns=selected_features,
        label_config=config.label,
        model_config=config.model,
        trade_config=config.trade,
    )

    if not wf_results:
        logger.error("ウォークフォワード結果なし")
        return

    # 4. 確率閾値の最適化
    logger.info("=== 確率閾値の最適化 ===")
    best_threshold, best_eval = optimize_threshold(
        wf_results,
        search_range=config.trade.probability_search_range,
        sl_mult=config.trade.sl_atr_multiplier,
        tp_mult=config.trade.tp_atr_multiplier,
    )

    # 5. バックテスト
    logger.info("=== バックテスト ===")
    bt = backtest_from_wf_results(
        wf_results,
        threshold=best_threshold,
        sl_atr_mult=config.trade.sl_atr_multiplier,
        tp_atr_mult=config.trade.tp_atr_multiplier,
        max_drawdown_limit=config.trade.max_drawdown,
        spread_pips=config.trade.spread_pips,
    )
    logger.info(
        "バックテスト: %d取引, 勝率=%.1f%%, Sharpe=%.3f, MaxDD=%.1f%%",
        bt.total_trades,
        bt.win_rate * 100,
        bt.sharpe_ratio,
        bt.max_drawdown * 100,
    )

    # 6. レポート出力
    print_report(best_eval)

    # 7. CSVエクスポート
    export_csv(wf_results, best_eval, config.paths.results_dir)

    logger.info("完了")

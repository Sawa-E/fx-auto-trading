<!--
種別: enhancement
優先度: high
ステータス: active
作成日: 2026-03-16
更新日: 2026-03-16
担当: AIエージェント
-->

# USD/JPY デイトレード予測モデル構築

## 概要

5つのリサーチ資料の知見を統合し、USD/JPY 1時間足の方向予測モデルを構築する。ノイズを除外した閾値ラベルで訓練し、LightGBMで予測確率を出力、シャープレシオを最適化目標とする。結果はCSV+CLIで確認する（Phase 1.5）。

## 背景

v1は機能を広げすぎて予測精度が不十分だった。v2では5つの論文・レポートの知見に基づき、予測精度に全集中する。

## 設計判断（ADR参照）

| ADR | 判断 | 決定 |
|-----|------|------|
| [002](../design/decisions/002-prediction-model-design.md) #1 | 予測形式 | 2値 + 閾値ラベル + 確率フィルタ |
| [002](../design/decisions/002-prediction-model-design.md) #2 | ホライゾン | 1/4/8足先を実験的に比較 |
| [002](../design/decisions/002-prediction-model-design.md) #3 | 特徴量戦略 | 28個→変数重要度で10-15個に刈り込み |
| [002](../design/decisions/002-prediction-model-design.md) #4 | 最適化目標 | シャープレシオ |
| [001](../design/decisions/001-technology-stack.md) #3 | モデル | RF（ベースライン）→ LightGBM（メイン） |
| [001](../design/decisions/001-technology-stack.md) #6 | 検証窓 | 拡大窓（expanding window） |
| [003](../design/decisions/003-risk-management.md) #1 | 損切り/利確 | ATR×1.5 / ATR×2.0（RR比1.33） |
| [003](../design/decisions/003-risk-management.md) #2 | レバレッジ | 5倍以下 |
| [003](../design/decisions/003-risk-management.md) #3 | モデル停止 | DD 10%で停止 |
| [003](../design/decisions/003-risk-management.md) #4 | 確率フィルタ | 閾値0.60（0.50-0.70で探索） |
| [004](../design/decisions/004-overfitting-prevention.md) #1 | 正則化 | Kaggleデフォルトより強め |
| [004](../design/decisions/004-overfitting-prevention.md) #2 | 過学習検出 | 精度差>10%で警告 |
| [005](../design/decisions/005-data-pipeline.md) #1 | データ取得 | 初回一括 + 差分更新 |
| [005](../design/decisions/005-data-pipeline.md) #2 | DBスキーマ | ohlcテーブルのみ |
| [005](../design/decisions/005-data-pipeline.md) #3 | データ期間 | 2020年〜（5年分、約35,000本） |
| [005](../design/decisions/005-data-pipeline.md) #4 | 品質チェック | 取得時に自動検証 |
| [006](../design/decisions/006-feature-engineering.md) #1 | 均衡乖離度 | 重視（8個、全体の約1/4） |
| [006](../design/decisions/006-feature-engineering.md) #2 | ラグ深さ | 1-6期（v1の12期から半減） |

## スコープ

**含む**:
- GMO FX APIからのUSD/JPY 1H OHLCデータ収集・SQLite保存（ADR 005）
- 28個の特徴量生成（ADR 006の具体リスト）
- 閾値ラベル生成（±X pips未満はNaN化）（ADR 002）
- RF ベースライン → LightGBM（強め正則化）で精度向上（ADR 001, 004）
- 変数重要度による特徴量刈り込み（28→10-15個）（ADR 002）
- 複数ホライゾン（1/4/8足先）の比較実験（ADR 002）
- 拡大窓ウォークフォワード検証（ADR 001）
- シャープレシオ最適化（Optuna）（ADR 002）
- 簡易バックテスト: ATR損切り/利確、確率フィルタ、DD10%停止（ADR 003）
- CSVエクスポート + CLIレポート（Phase 1.5）

**含まない**:
- GUI / 通知 / 自動売買 / ペーパートレード（将来的に自作GUI予定）
- tSNE可視化（将来の改善として検討）
- SVM / ディープラーニング（資料5が為替で不良と結論済み）

## タスクリスト

### フェーズA: プロジェクト基盤

- [ ] TASK-001-01: pyproject.toml整理・srcパッケージ構造作成
  - 削除: torch, streamlit, plotly, mlflow, backtesting
  - 維持: httpx, pandas, scikit-learn, lightgbm, optuna
  - 追加: matplotlib（CSVレポート用のグラフ出力）
  - `src/fx_auto_trading/` 配下にdata/, features/, models/, evaluation/を作成
  - ADR参照: [001](../design/decisions/001-technology-stack.md)
  - 対象: `pyproject.toml`, パッケージ構造

- [ ] TASK-001-02: 設定・例外・ログ基盤
  - dataclass設定:
    - ApiConfig: base_url, rate_limit, timeout, max_retries
    - FeatureConfig: 各指標のパラメータ
    - ModelConfig: 正則化パラメータ探索範囲（ADR 004の具体値）
    - TradeConfig: ATR乗数(SL=1.5, TP=2.0), レバ上限(5), DD停止(10%), 確率閾値(0.60)
    - PathsConfig: DB, 出力ディレクトリ
  - カスタム例外、ログ設定
  - ADR参照: [003](../design/decisions/003-risk-management.md), [004](../design/decisions/004-overfitting-prevention.md)
  - 対象: `config.py`, `exceptions.py`, `log.py`

### フェーズB: データ収集（A→B）

- [ ] TASK-001-03: GMO FX APIクライアント
  - v1の collector.py をベースに簡潔化
  - 1H足: 日単位リクエスト（YYYYMMDD）
  - レート制限（1回/秒）、指数バックオフリトライ、エラーハンドリング
  - ADR参照: [005](../design/decisions/005-data-pipeline.md) #1
  - 対象: `data/collector.py` + テスト

- [ ] TASK-001-04: SQLiteストレージ + データ品質チェック
  - **ohlcテーブルのみ**（ADR 005 #2: trades/positions削除、volume列削除）
  - CRUD、重複排除（UNIQUE制約）、期間クエリ
  - **品質チェック**（ADR 005 #4）:
    - OHLC整合性: high >= max(open,close), low <= min(open,close)
    - 異常値: 前足からの変動率5%超でログ警告
  - 対象: `data/storage.py` + テスト

- [ ] TASK-001-05: データ収集CLIスクリプト
  - **初回一括**: `--from 2020-01-01`（ADR 005 #3: 5年分）
  - **差分更新**: `--update`（SQLiteの最新timestamp以降を取得）
  - 対象: `scripts/collect_data.py`

### フェーズC: 特徴量生成（B→C）

- [ ] TASK-001-06: テクニカル指標（カテゴリA: 6個）
  - ADR 006の具体リスト: macd_hist, atr, adx, plus_di, minus_di, cci
  - v1からの変更: williams_r削除（StochRSIと冗長）、SMA絶対値削除（比率に変更）
  - ADR参照: [006](../design/decisions/006-feature-engineering.md)
  - 対象: `features/indicators.py` + テスト

- [ ] TASK-001-07: 均衡乖離度・定常性特徴量（カテゴリB: 8個）
  - ADR 006の具体リスト:
    - bb_position (0-1), rsi (0-100), stoch_rsi_k (0-1), stoch_rsi_d (0-1)
    - channel_pos_24 (0-1), channel_pos_48 (0-1), close_sma_ratio (≈1), vol_regime (≈1)
  - **全て有界 or 平均回帰的**: 資料3「定常的な系列は予測しやすい」
  - ADR参照: [006](../design/decisions/006-feature-engineering.md) #1
  - 対象: `features/stationary.py` + テスト

- [ ] TASK-001-08: 時間帯・モメンタム・ラグ特徴量（カテゴリC+D+E: 14個）
  - C. 時間帯(5個): hour, day_of_week, is_tokyo, is_london, is_ny
  - D. モメンタム(3個): roc_6, roc_12, up_ratio_6
  - E. ラグ(6個): return_lag1〜**lag6**（ADR 006 #2: v1の12期から半減）
  - ADR参照: [006](../design/decisions/006-feature-engineering.md) #2
  - 対象: `features/temporal.py`, `features/momentum.py` + テスト

- [ ] TASK-001-09: 特徴量パイプライン + 閾値ラベル生成
  - 全28個の特徴量を統合、NaN除去
  - **閾値ラベル**（ADR 002 #1）: ±X pips未満の動きはNaN化（学習から除外）
  - 閾値X: ATRベースで動的設定 or 固定値（実験で決定）
  - 複数ホライゾン対応（N=1,4,8）（ADR 002 #2）
  - ADR参照: [002](../design/decisions/002-prediction-model-design.md) #1, #2
  - 対象: `features/pipeline.py` + テスト

### フェーズD: モデル訓練（C→D）

- [ ] TASK-001-10: RFベースライン + 変数重要度刈り込み
  - ランダムフォレストで簡易予測（ADR 001 #3）
  - **変数重要度算出**→上位10-15個の特徴量を特定（ADR 002 #3）
  - ADR参照: [001](../design/decisions/001-technology-stack.md) #3, [002](../design/decisions/002-prediction-model-design.md) #3
  - 対象: `models/baseline.py`

- [ ] TASK-001-11: LightGBMトレーナー
  - 2値分類（予測確率出力）
  - **強め正則化**（ADR 004 #1）:
    - max_depth: [3, 6], min_child_samples: [50, 200]
    - num_leaves: [15, 63], reg_lambda: [1.0, 10.0]
    - reg_alpha: [0.1, 5.0], learning_rate: [0.01, 0.1]
  - Optunaで最適化（**目標: シャープレシオ**、ADR 002 #4）
  - **過学習検出**（ADR 004 #2）: 訓練/検証の精度差>10%で警告
  - TASK-001-10で選別した特徴量を使用
  - ADR参照: [004](../design/decisions/004-overfitting-prevention.md) #1, #2
  - 対象: `models/trainer.py` + テスト

### フェーズE: 検証・評価（D→E）

- [ ] TASK-001-12: ウォークフォワード検証（拡大窓）
  - **拡大窓**（ADR 001 #6）: 2020/01〜予測対象の1つ手前まで
  - 各ウィンドウでOptuna最適化 + 予測
  - 1ヶ月ごとにスライド
  - ADR参照: [001](../design/decisions/001-technology-stack.md) #6
  - 対象: `evaluation/walk_forward.py` + テスト

- [ ] TASK-001-13: 評価メトリクス + CSVエクスポート
  - メトリクス:
    - **シャープレシオ**（メイン指標、ADR 002 #4）
    - 勝率、RR比（資料4: この2つが収益を決定）
    - 最大ドローダウン、プロフィットファクター
  - **確率閾値の最適化**: 0.50-0.70で探索（ADR 003 #4）
  - **CSVエクスポート（Phase 1.5）**:
    - `results/predictions.csv` — 各足の予測確率・実際の方向
    - `results/trades.csv` — 仮想取引の損益
    - `results/metrics.csv` — 期間ごとのメトリクス
    - `results/feature_importance.csv` — 変数重要度
  - ADR参照: [002](../design/decisions/002-prediction-model-design.md) #4, [003](../design/decisions/003-risk-management.md) #4
  - 対象: `evaluation/metrics.py` + テスト

- [ ] TASK-001-14: 簡易バックテスト
  - 予測シグナル × **確率フィルタ(0.60)**（ADR 003 #4）で仮想取引
  - **ATRベース損切り/利確**（ADR 003 #1）: SL=ATR×1.5, TP=ATR×2.0, RR比1.33
  - **レバレッジ5倍以下**（ADR 003 #2）
  - **DD 10%で停止**（ADR 003 #3）
  - 累積リターン、ドローダウン推移
  - ADR参照: [003](../design/decisions/003-risk-management.md) #1-4
  - 対象: `evaluation/backtest.py`

- [ ] TASK-001-15: ホライゾン比較実験
  - N=1,4,8 の3モデルをそれぞれウォークフォワード検証（ADR 002 #2）
  - **シャープレシオで比較し最適ホライゾンを採用**
  - 結果レポート出力（CLIテキスト + CSV）
  - 対象: `scripts/compare_horizons.py`

### フェーズF: 統合

- [ ] TASK-001-16: エンドツーエンドパイプライン + CLIレポート
  - データ収集→特徴量→訓練→検証→評価の一気通貫
  - CLIインターフェース:
    - `python -m fx_auto_trading collect` — データ収集
    - `python -m fx_auto_trading train` — 訓練
    - `python -m fx_auto_trading evaluate` — 検証+評価+CSV出力
    - `python -m fx_auto_trading run` — 全工程一気通貫
  - CLIレポート（テキスト出力）: シャープレシオ、勝率、RR比、最大DD等
  - 対象: `pipeline.py`, `__main__.py`

## 技術的な考慮事項

### 過学習対策の優先順位（ADR 004）

| 優先度 | 対策 | ADR | タスク |
|--------|------|-----|--------|
| 最優先 | ウォークフォワード検証（拡大窓） | 001 #6 | TASK-001-12 |
| 最優先 | 確率閾値フィルタ(0.60) | 003 #4 | TASK-001-13 |
| 高 | 変数重要度で28→10-15個に刈り込み | 002 #3 | TASK-001-10 |
| 高 | 閾値ラベル（ノイジーサンプル除外） | 002 #1 | TASK-001-09 |
| 高 | LightGBM強め正則化 | 004 #1 | TASK-001-11 |
| 中 | 訓練/検証の精度差>10%警告 | 004 #2 | TASK-001-11 |

### pyproject.toml変更

- **削除**: torch, streamlit, plotly, mlflow, backtesting
- **維持**: httpx, pandas, scikit-learn, lightgbm, optuna
- **追加**: matplotlib

### ディレクトリ構成

```
src/fx_auto_trading/
├── __init__.py
├── __main__.py          ← CLI エントリーポイント
├── config.py            ← dataclass設定（ADR 003,004の具体値含む）
├── exceptions.py
├── log.py
├── pipeline.py          ← エンドツーエンド統合
├── data/
│   ├── __init__.py
│   ├── collector.py     ← GMO FX API（ADR 005）
│   └── storage.py       ← SQLite ohlcのみ（ADR 005）
├── features/
│   ├── __init__.py
│   ├── indicators.py    ← カテゴリA: テクニカル6個（ADR 006）
│   ├── stationary.py    ← カテゴリB: 均衡乖離度8個（ADR 006）
│   ├── temporal.py      ← カテゴリC: 時間帯5個（ADR 006）
│   ├── momentum.py      ← カテゴリD+E: モメンタム3個+ラグ6個（ADR 006）
│   └── pipeline.py      ← 統合+閾値ラベル（ADR 002）
├── models/
│   ├── __init__.py
│   ├── baseline.py      ← RF+変数重要度（ADR 001,002）
│   └── trainer.py       ← LightGBM+Optuna+過学習検出（ADR 004）
└── evaluation/
    ├── __init__.py
    ├── walk_forward.py  ← 拡大窓WF検証（ADR 001）
    ├── metrics.py       ← 評価+CSV出力（ADR 002,003）
    └── backtest.py      ← ATR損切り/利確/DD停止（ADR 003）
```

### 評価結果の確認方法（Phase 1.5）

```bash
# 全工程実行
python -m fx_auto_trading run

# 結果確認（CLI）
# → シャープレシオ、勝率、RR比、最大DD等をテキスト出力

# 結果確認（CSV）
# → results/ 配下に4つのCSVファイル
#    predictions.csv, trades.csv, metrics.csv, feature_importance.csv
# → Jupyter Notebook や Excel で詳細分析
```

## 関連ドキュメント

- [ADR 001: 技術スタック](../design/decisions/001-technology-stack.md)
- [ADR 002: 予測モデル設計](../design/decisions/002-prediction-model-design.md)
- [ADR 003: リスク管理・取引ルール](../design/decisions/003-risk-management.md)
- [ADR 004: 過学習対策](../design/decisions/004-overfitting-prevention.md)
- [ADR 005: データパイプライン設計](../design/decisions/005-data-pipeline.md)
- [ADR 006: 特徴量エンジニアリング詳細](../design/decisions/006-feature-engineering.md)
- [リサーチサーベイ](../archive/research-survey.md)

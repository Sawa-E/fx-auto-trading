<!--
種別: ステータス
対象: 実装ステータス
作成日: 2026-03-16
更新日: 2026-03-18
-->

# 実装ステータス

## プロジェクト全体

| フェーズ | ステータス | 内容 |
|---------|----------|------|
| リサーチ | `done` | 5論文精読、リサーチサーベイ作成 |
| 設計 | `done` | 7つのADR、2つの実装計画 |
| Phase 1: 予測モデル | `done` | データ収集→特徴量→訓練→WF検証→バックテスト |
| Phase 2: フォワードテスト | `partial` | 仮想取引エンジン完成、GitHub Actions稼働開始 |
| Phase 3: リアル取引 | `not-started` | GMO Private API連携 |
| Phase 4: 自作GUI | `not-started` | |

## モジュール別ステータス

| モジュール | ステータス | 備考 |
|-----------|-----------|------|
| data/collector.py | `done` | GMO FX API、レート制限、リトライ |
| data/storage.py | `done` | SQLite ohlcテーブル、品質チェック |
| features/indicators.py | `done` | テクニカル指標5個 |
| features/stationary.py | `done` | 均衡乖離度8個 |
| features/temporal.py | `done` | 時間帯5個 |
| features/momentum.py | `done` | モメンタム3個 + ラグ6個 |
| features/pipeline.py | `done` | 34特徴量統合、閾値ラベル、レジームフィルタ |
| models/baseline.py | `done` | RF変数重要度、特徴量選定 |
| models/trainer.py | `done` | LightGBM + Optuna + 過学習検出 |
| evaluation/walk_forward.py | `done` | 拡大窓WF検証 |
| evaluation/metrics.py | `done` | シャープレシオ等 + CSV出力 |
| evaluation/backtest.py | `done` | ATR損切り/利確シミュレーション |
| trading/engine.py | `done` | ステートレス仮想取引エンジン |
| notification/discord.py | `done` | シグナル通知 + 決済通知 |
| scripts/predict.py | `done` | 毎時予測（GitHub Actions対応） |
| scripts/forward_report.py | `done` | 1ヶ月後の検証レポート |
| scripts/collect_data.py | `done` | 初回一括 + 差分更新 |
| scripts/download_histdata.py | `done` | histdata.com 1分足→1H集約 |
| scripts/train_production.py | `done` | 本番モデル訓練→pkl保存 |
| .github/workflows/predict.yml | `done` | 平日毎時自動実行 |

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2026-03-16 | 初版作成（全モジュール not-started） |
| 2026-03-17 | Phase 1 全実装完了 |
| 2026-03-18 | Phase 2 仮想取引エンジン + GitHub Actions稼働開始 |

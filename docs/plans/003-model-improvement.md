<!--
種別: enhancement
優先度: high
ステータス: active
作成日: 2026-03-18
更新日: 2026-03-18
担当: AIエージェント
-->

# モデル根本改善: 予測力向上と再訓練安定化

## 概要

模擬フォワードテスト（8期間）で全パターンがマイナスとなり、バックテストとの乖離が判明。
prob出力の集中（0.52-0.54）、再訓練不安定性、予測力不足を根本的に解決する。

## 背景

- バックテスト勝率51-57% → 模擬フォワード勝率41-43%（大幅乖離）
- prob 0.53-0.54の取引が全体の56%を占め、勝率49.2%（ほぼコイン投げ）
- 再訓練のたびにOptuna+RF特徴量選択が変わり、モデルが別物になる
- 資料5（MUFG）: 「マーケットデータは再現性が圧倒的に低い」
- 資料3（SIG-FIN）: 「パラメータαは時間変化する」→ 拡大窓の限界

## スコープ

**含む**:
- A: スライディングウィンドウ（拡大窓 → 直近3年固定窓）
- B: 確率キャリブレーション（Platt scaling / Isotonic regression）
- C: ウォークフォワードベース訓練（模擬FWで勝てるモデルを作る）
- D: 特徴量の見直し（定常性・相関の再評価）
- E: 別アルゴリズム追加（XGBoost / CatBoost アンサンブル）
- 改善ごとの模擬フォワード検証（8期間）
- 最終モデルのpkl保存

**含まない**:
- ディープラーニング（LSTM等）の導入
- 新しいデータソースの追加
- 売りモデルの独立構築（別計画）
- 本番デプロイの変更（predict.py, GitHub Actions）

## タスクリスト

### Phase 1: 基盤改善（B + A）

- [ ] TASK-003-01: 確率キャリブレーション実装
  - 対象: `src/fx_auto_trading/models/trainer.py`
  - 内容: train()にPlatt scaling（sklearn CalibratedClassifierCV）を追加
  - calibration_method引数（'sigmoid', 'isotonic', None）を追加
  - 検証: prob分布の変化を確認（0.52-0.54集中 → 分散拡大）

- [ ] TASK-003-02: キャリブレーション効果の模擬FW検証
  - 対象: 検証スクリプト（新規）
  - 内容: 8期間で模擬FWを実行し、キャリブレーションあり/なしを比較
  - 判定基準: prob分布の改善、勝率・PFの変化

- [ ] TASK-003-03: スライディングウィンドウ実装
  - 対象: `src/fx_auto_trading/evaluation/walk_forward.py`, `scripts/train_production.py`
  - 内容: expanding_window引数をFalseに設定可能にし、window_years=3のスライディングウィンドウを追加
  - 資料3根拠: パラメータの時間変化に対応

- [ ] TASK-003-04: スライディングウィンドウ + キャリブレーションの模擬FW検証
  - 対象: 検証スクリプト
  - 内容: 8期間でB+Aの組み合わせを検証
  - 比較: 現行モデル vs キャリブのみ vs SW+キャリブ

### Phase 2: 訓練プロセス改善（C + D）

- [ ] TASK-003-05: WFベース訓練パイプライン構築
  - 対象: `scripts/train_production.py`（大幅改修）
  - 内容: 直近N期間のWF結果が基準を満たすモデルのみ採用する仕組み
  - 基準: 模擬FW勝率 > 50%、PF > 1.0
  - 不合格の場合: パラメータ調整して再訓練

- [ ] TASK-003-06: 特徴量の再評価
  - 対象: `src/fx_auto_trading/features/pipeline.py`
  - 内容:
    - 全34特徴量の期間別安定性を分析（各年の重要度変動）
    - 相関分析の再実行（冗長特徴量の除外）
    - 資料3の定常性基準に基づくフィルタリング強化
  - 成果物: 特徴量の安定性レポート + 改訂FEATURE_COLUMNS

- [ ] TASK-003-07: Phase 2の統合模擬FW検証
  - 対象: 検証スクリプト
  - 内容: B+A+C+Dの組み合わせで8期間検証

### Phase 3: アンサンブル（E）

- [ ] TASK-003-08: XGBoost/CatBoostトレーナー実装
  - 対象: `src/fx_auto_trading/models/`（新規ファイル）
  - 内容: LightGBMと同じインターフェースでXGBoost, CatBoostを実装
  - 資料5根拠: 「全く構造が違うアルゴリズムは同じ誤りを犯しづらい」

- [ ] TASK-003-09: アンサンブル予測器の実装
  - 対象: `src/fx_auto_trading/models/ensemble.py`（新規）
  - 内容: 複数モデルの確率出力を平均/重み付き平均するクラス
  - 方式: Soft voting（確率の平均）

- [ ] TASK-003-10: 最終統合検証 + 本番モデル選定
  - 対象: 検証スクリプト + `scripts/train_production.py`
  - 内容: 全改善（A-E）の組み合わせで8期間模擬FW
  - 最も成績の良い構成をv2本番モデルとして保存
  - models/production_model_v2.pkl, production_meta_v2.json

## 技術的な考慮事項

- **キャリブレーション**: CalibratedClassifierCVはcv=5のクロスバリデーションを内部で行う。訓練時間が5倍になる
- **スライディングウィンドウ**: 3年=約18,000本。min_train_sizeの調整が必要
- **XGBoost/CatBoost**: pyproject.tomlに依存追加が必要
- **検証の一貫性**: 8期間の模擬FWを統一スクリプトで実行し、比較可能にする
- **過学習リスク**: 改善を重ねるほど8期間に過適合するリスクあり。最終検証は別期間で行うことも検討

## 成功基準

| 指標 | 現状 | 目標 |
|------|------|------|
| 模擬FW勝率（8期間平均） | 41-43% | > 50% |
| 模擬FW合計損益 | -2,871〜-4,209円 | > 0円（プラス転換） |
| 利益期間 | 1-3/8 | > 4/8 |
| prob分布 | 0.52-0.54に集中 | 0.45-0.65に分散 |

## 関連ドキュメント

- [ADR 002: 予測モデル設計](../design/decisions/002-prediction-model-design.md)
- [ADR 004: 過学習防止](../design/decisions/004-overfitting-prevention.md)
- [ADR 007: 本番モデル](../design/decisions/007-production-model.md)
- [評価結果サマリー](../../results/evaluation_summary.md)
- [模擬フォワード結果](../../results/simulated_forward_report.md)
- [ロードマップ](../status/roadmap.md)

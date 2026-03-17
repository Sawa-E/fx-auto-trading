<!--
種別: enhancement
優先度: high
ステータス: active
作成日: 2026-03-18
更新日: 2026-03-18
担当: AIエージェント
-->

# Phase 2: フォワードテスト（ペーパートレード）

## 概要

確定モデル（改善B, N=8, 閾値0.55）を実際の相場でテストする。ローカルで動作確認後、GitHub Actionsで自動化し、Discord通知でシグナルを確認する。1ヶ月間のフォワード結果をバックテストと比較し、モデルの実用性を検証する。

## 背景

資料2（Optimax）: 「バックテスト結果を過信せず、実資金で最低1ヶ月運用試行」
資料5（MUFG）: 「AIとファンドマネージャーはお互い補完し合う関係」

バックテスト実績（10年間）:
- 勝率60.1%, PF 2.01, 3.5回/日, 累積+2,281

## 確定モデル構成（ADR 007より）

| 項目 | 設定 |
|------|------|
| モデル | LightGBM（`models/production_model.pkl`） |
| 特徴量 | `models/production_meta.json` に記載の15個 |
| ホライゾン | N=8（8時間先） |
| ラベル閾値 | 3 pips |
| 確率閾値 | 0.55（P≥0.55で買い、P≤0.45で売り） |
| レジームフィルタ | ADX > 25 |
| 損切り | ATR(14) × 1.5 |
| 利確 | ATR(14) × 2.0 |

## スコープ

**含む**:
- 予測スクリプト（pkl読み込み→最新データ取得→シグナル出力）
- Discord Webhook通知
- 予測結果のCSV記録
- ローカル動作確認
- GitHub Actions自動化（毎時実行、平日のみ）
- 1ヶ月後のフォワード結果レポート

**含まない**:
- 実際の注文執行（GMO Private API）— Phase 3
- GUI — Phase 4（最終的には自作GUI予定）
- 自動再訓練 — ADR 007で月次手動と決定

## 自動化戦略

ローカル動作確認 → GitHub Actions に移行。

| 段階 | 実行方法 | 目的 |
|------|---------|------|
| Phase 2a | ローカルで手動実行 | 動作確認、バグ修正 |
| Phase 2b | GitHub Actions（毎時cron） | PC不要の自動運用 |

GitHub Actions の利点:
- 無料（private repoでも月2,000分の無料枠、本用途は月960分で収まる）
- PC不要（電源・スリープを気にしなくてよい）
- セットアップが簡単（`.github/workflows/predict.yml` を置くだけ）
- 注意: 実行時刻に±5-15分のズレあり（1H足予測なので問題なし）

## タスクリスト

### フェーズA: 予測スクリプト（ローカル動作確認）

- [ ] TASK-002-01: 予測スクリプトの作成
  - `models/production_model.pkl` を読み込み
  - `models/production_meta.json` から特徴量リスト・パラメータを取得
  - GMO APIから最新50本の1H足を取得（特徴量計算のウォームアップ用）
  - 34特徴量を計算 → meta.jsonの15個に絞り込み
  - レジームフィルタ: ADX > 25 でなければ見送り
  - モデルで予測確率を出力:
    - P(上昇) ≥ 0.55 → 買いシグナル
    - P(上昇) ≤ 0.45 → 売りシグナル
    - それ以外 → 見送り
  - ATR(14)から損切り/利確ラインを算出:
    - 損切り = 現在値 ∓ ATR × 1.5
    - 利確 = 現在値 ± ATR × 2.0
  - 対象ファイル: `scripts/predict.py`

  ```
  実行フロー:
  GMO API → 最新50本取得
       ↓
  34特徴量計算 → 15個に絞り込み
       ↓
  ADX > 25 ? ──No──→ 見送り（CSV記録のみ）
       ↓ Yes
  LightGBM予測 → P(上昇)
       ↓
  P ≥ 0.55 ? → 買い + Discord通知
  P ≤ 0.45 ? → 売り + Discord通知
  それ以外   → 見送り（CSV記録のみ）
  ```

- [ ] TASK-002-02: Discord通知の実装
  - Webhook URLは `.env` の `DISCORD_WEBHOOK_URL` から読み込み
  - シグナル発生時のみ通知（見送り時は通知しない）
  - 通知フォーマット:
    ```
    🟢 買い USD/JPY @ 159.14
      確率: 0.58 | ADX: 32.5
      損切り: 158.69 | 利確: 159.74
      時刻: 2026-03-18 15:00 UTC
    ```
    - 買い: 🟢、売り: 🔴
  - エラー時（API失敗等）は ⚠️ で通知
  - 対象ファイル: `src/fx_auto_trading/notification/discord.py`

- [ ] TASK-002-03: 予測結果のCSV記録
  - 全予測（シグナルあり/見送り含む）を記録
  - ファイル: `results/forward_predictions.csv`
  - カラム:
    ```
    timestamp, price, direction(buy/sell/hold), probability,
    adx, atr, sl_price, tp_price, regime_ok(true/false)
    ```
  - 毎時追記（appendモード、ヘッダーはファイル新規時のみ）
  - 対象ファイル: `scripts/predict.py` に組み込み

- [ ] TASK-002-04: ローカル動作確認
  - 手動で `uv run python scripts/predict.py` を実行
  - 確認項目:
    - [ ] GMO APIからデータ取得成功
    - [ ] 特徴量が正しく計算される
    - [ ] シグナル判定が動作する
    - [ ] Discord通知が届く
    - [ ] CSVにレコードが追記される
  - 市場が開いている時間帯（月〜金、UTC 22:00〜翌21:00）に実行すること

### フェーズB: GitHub Actions自動化

- [ ] TASK-002-05: GitHub Actions ワークフロー作成
  - ファイル: `.github/workflows/predict.yml`
  - スケジュール: `cron: '1 * * * 1-5'`（平日毎時1分）
  - ワークフロー手順:
    1. `actions/checkout@v4` でリポジトリ取得
    2. `actions/setup-python@v5` で Python 3.12 セットアップ
    3. `actions/cache@v4` で pip キャッシュ（LightGBM等の再インストール短縮）
    4. `pip install` で依存インストール
    5. `python scripts/predict.py` 実行
  - 環境変数:
    - `DISCORD_WEBHOOK_URL`: GitHub Secretsから注入
    - `FX_API_BASE_URL`: デフォルトでOK
  - pkl/meta.jsonはリポジトリに含める（pklはgit対象に変更、約数MB）
  - 予測CSVはArtifactとして保存（またはgit commit & push）
  - 対象ファイル: `.github/workflows/predict.yml`

- [ ] TASK-002-06: GitHub Secrets設定
  - リポジトリ Settings → Secrets and variables → Actions
  - `DISCORD_WEBHOOK_URL` を登録
  - 手動作業（スクリプト化不可）

### フェーズC: 検証（1ヶ月後）

- [ ] TASK-002-07: フォワード結果のレポートスクリプト
  - `results/forward_predictions.csv` を読み込み
  - 各シグナルの8時間後の価格を GMO API で取得し、勝敗を判定
  - 算出する指標:
    - 勝率（バックテスト: 60.1%）
    - PF（バックテスト: 2.01）
    - シャープレシオ（バックテスト: 0.352）
    - 取引頻度（バックテスト: 3.5回/日）
  - バックテスト結果との比較表を出力
  - 対象ファイル: `scripts/forward_report.py`

## 技術的な考慮事項

### エラーハンドリング
- GMO API接続失敗 → ログに記録、Discord ⚠️ 通知、CSVにerror記録、次の時間に再試行
- pkl読み込み失敗 → 致命的エラー、Discord ❌ 通知、終了
- Discord通知失敗 → ログに記録するが予測・CSV記録は続行

### GitHub Actions固有の注意
- 毎回クリーン環境なのでpip installが走る → cacheで短縮（初回2分、以降30秒）
- 実行時刻に±5-15分のズレ → 1H足予測なので許容
- Private repoの無料枠: 月2,000分。本用途は月960分で余裕

### CSVの管理
- ローカル実行時: `results/forward_predictions.csv` にローカル保存
- GitHub Actions時: Artifactとして保存 or git commit & pushで永続化

## 成功基準（1ヶ月後に判定）

| 指標 | バックテスト | 最低基準 | 判定 |
|------|-----------|---------|------|
| 勝率 | 60.1% | 55%以上 | PF>1なら利益は出る |
| PF | 2.01 | 1.5以上 | 利益>損失を維持 |
| 取引頻度 | 3.5回/日 | 1回/日以上 | デイトレードとして成立 |

バックテストからの劣化は想定内（資料5「マーケットデータは再現性が低い」）。
PF>1.5を維持できればPhase 3（実取引）への移行を検討。

## ディレクトリ構成（追加分）

```
scripts/
├── predict.py              ← 毎時実行の予測スクリプト
└── forward_report.py       ← 1ヶ月後の検証レポート

src/fx_auto_trading/
└── notification/
    ├── __init__.py
    └── discord.py           ← Discord Webhook通知

.github/
└── workflows/
    └── predict.yml          ← GitHub Actions定期実行

models/
├── production_model.pkl     ← 訓練済みモデル
└── production_meta.json     ← メタデータ（特徴量リスト等）

results/
├── evaluation_summary.md    ← バックテスト結果
└── forward_predictions.csv  ← フォワード予測記録
```

## 関連ドキュメント

- [ADR 007: 本番モデル](../design/decisions/007-production-model.md)
- [ADR 003: リスク管理](../design/decisions/003-risk-management.md)
- [評価結果サマリー](../../results/evaluation_summary.md)
- [Phase 1 実装計画](./001-prediction-model.md)

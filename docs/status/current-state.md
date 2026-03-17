# 現在の状態（2026-03-18）

## 何が動いているか

**GitHub Actionsが平日毎時1分（UTC）に自動実行中。**

```
毎時の処理（約48秒で完了）:
  1. GMO APIからUSD/JPY直近7日分の1H足を取得
  2. 34特徴量を計算 → RF重要度上位15個に絞り込み
  3. LightGBM（production_model.pkl）で予測
  4. ADX > 25（トレンド相場）かチェック
  5. P ≥ 0.55 → Discord買い通知
     P ≤ 0.45 → Discord売り通知
     それ以外 → 何もしない（静か）
  6. 過去8時間のシグナルのSL/TPヒットをチェック → 決済通知
```

## 確定モデル

| 項目 | 値 |
|------|-----|
| アルゴリズム | LightGBM |
| ホライゾン | N=8（8時間先） |
| 確率閾値 | 0.55 |
| レジームフィルタ | ADX > 25 |
| 損切り | ATR(14) × 1.5 |
| 利確 | ATR(14) × 2.0 |
| 仮想口座 | 30,000円、レバ5倍 |

## バックテスト実績（10年間）

| 指標 | 値 |
|------|-----|
| 勝率 | 60.1% |
| PF | 2.01（利益が損失の2倍） |
| Sharpe | 0.352 |
| DD | 1.1% |
| 取引頻度 | 3.5回/日 |

## 模擬フォワードテスト実績

| 期間 | 結果 | 資金推移 |
|------|------|---------|
| 円安（2026/2-3） | 14勝2敗（87.5%） | 10,000→12,426円 |
| 円高（2022/10-11） | 0取引（見送り） | 変動なし |

## 注意点

- **買い専用**: 売りシグナルは現時点で機能しない（円高時は取引しない＝損もしない）
- **サンプル数が少ない**: 模擬フォワードは16回。統計的には不十分
- **月利24%は異常値**: 長期的にはバックテスト水準（月3-5%）に収束する見込み
- **ADX < 25の期間はシグナルなし**: 現在（2026/3/18）はADX=24.2でレンジ相場のため見送り中

## 動作確認済み（2026-03-18）

| 確認項目 | 結果 |
|---------|------|
| ローカル predict.py | ✅ 正常（ADX<25で見送り） |
| GitHub Actions 手動トリガー | ✅ 48秒で完了 |
| GMO APIデータ取得（Actions内） | ✅ 142本取得 |
| pip install -e .（Actions内） | ✅ |
| Discord Secrets設定 | ✅ |
| リポジトリ visibility | public（Actions無料） |

## 次に何が起こるか

1. **平日毎時**: GitHub Actionsが自動実行
2. **ADX > 25になったら**: Discordにシグナル通知が届く
3. **シグナル8時間後**: SL/TPヒット判定 → Discord決済通知
4. **1ヶ月後（2026/04/18頃）**: `python scripts/forward_report.py --from 2026-03-19 --to 2026-04-18` で検証
5. **検証結果**: PF>1.5ならPhase 3（リアル取引）へ

## やるべきこと（ユーザー）

- **毎日**: Discordを見る（通知が来たら確認）
- **1ヶ月後**: forward_report.py を実行してバックテストと比較
- **月初**: データ更新 + モデル再訓練（`scripts/collect_data.py --update` → `scripts/train_production.py`）

## ファイル構成

```
fx-auto-trading/
├── .github/workflows/predict.yml  ← 自動実行（毎時）
├── models/
│   ├── production_model.pkl       ← 訓練済みモデル
│   └── production_meta.json       ← 特徴量リスト・パラメータ
├── scripts/
│   ├── predict.py                 ← 毎時予測（Actions実行）
│   ├── forward_report.py          ← 1ヶ月後の検証
│   ├── train_production.py        ← モデル再訓練
│   ├── collect_data.py            ← GMOデータ収集
│   └── download_histdata.py       ← histdata.comからDL
├── src/fx_auto_trading/
│   ├── trading/engine.py          ← 仮想取引エンジン
│   ├── notification/discord.py    ← Discord通知
│   ├── features/pipeline.py       ← 34特徴量生成
│   ├── models/trainer.py          ← LightGBM訓練
│   └── ...
├── docs/
│   ├── design/decisions/001-007   ← 7つのADR
│   ├── plans/001, 002             ← 実装計画
│   ├── archive/research-001-005   ← 5論文サマリー
│   └── status/                    ← このファイル
└── results/
    ├── evaluation_summary.md      ← バックテスト全結果
    └── simulated_forward_report.md ← 模擬フォワード結果
```

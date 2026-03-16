<!--
種別: decisions
対象: データパイプライン設計
作成日: 2026-03-16
更新日: 2026-03-16
担当: AIエージェント
-->

# データパイプライン設計

## 概要

GMO FX APIからのデータ取得、SQLiteへの保存、特徴量生成までのデータパイプラインの設計判断を記録する。

## 設計判断

### 判断1: データ取得戦略 — 初回一括 + 差分更新

**問題**: 数年分の1H OHLCデータをどう効率的に収集・維持するか

**選択肢**:
1. 初回一括取得 + 差分更新
2. 毎回全期間再取得
3. CSV/Parquetで手動管理

**決定**: 初回一括取得 + 差分更新

**理由**:
- GMO FX APIの1H足は日単位リクエスト（YYYYMMDD形式）。5年分 ≈ 1,800日 = 1,800回APIコール
- レート制限1回/秒で初回取得に約30分。毎回再取得は非効率
- 差分更新ならSQLiteの最新timestampから当日までの数日分のみ取得（数秒で完了）

**実装詳細**:
```
初回: scripts/collect_data.py --from 2020-01-01
  → 2020/01/01〜今日までの1H OHLCを全取得
  → SQLiteに保存、重複排除

差分更新: scripts/collect_data.py --update
  → SQLiteの最新timestamp以降を取得
  → 新規データのみINSERT
```

**トレードオフ**:
- **利点**: 効率的。初回以降は高速。APIコール数を最小化
- **欠点**: SQLiteのデータが破損した場合は初回取得をやり直す必要がある

### 判断2: SQLiteスキーマ — ohlcテーブルのみ

**問題**: SQLiteにどのテーブルを作成するか

**選択肢**:
1. ohlcテーブルのみ（Phase 1用）
2. v1のスキーマ維持（ohlc + trades + open_positions）
3. ohlc + predictions（予測結果も保存）

**決定**: ohlcテーブルのみ。trades/open_positionsはPhase 2以降で追加（YAGNI原則）

**理由**:
- Phase 1は予測精度に集中。取引実行は含まない（実装計画のスコープ外）
- v1のtrades/open_positionsは自動売買用であり、現段階では不要
- v1のvolume列も削除（GMO FX APIはFXでvolumeを返さない）

**スキーマ**:
```sql
CREATE TABLE IF NOT EXISTS ohlc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    interval TEXT NOT NULL,
    timestamp TEXT NOT NULL,  -- ISO 8601 UTC
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    UNIQUE(pair, interval, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ohlc_pair_interval_ts
    ON ohlc(pair, interval, timestamp);
```

**v1からの変更**:
- volume列削除（GMO FXでは不使用）
- tradesテーブル削除（Phase 1不要）
- open_positionsテーブル削除（Phase 1不要）

**トレードオフ**:
- **利点**: シンプル。不要なテーブルによる複雑さを排除
- **欠点**: Phase 2で自動売買を追加する際にマイグレーションが必要

### 判断3: 学習データ期間 — 2020年〜現在（約5年分）

**問題**: どの程度の過去データを学習に使用するか

**選択肢**:
1. 3年分（2023年〜）
2. 5年分（2020年〜）
3. 取得可能な最大範囲

**決定**: 2020年1月〜現在（約6年強 ≈ 35,000本の1H足）

**理由**:
- 資料5（MUFG）: 1990年〜予測対象までの約30年分を使用して成功。「拡大窓」方式では学習データが多いほど有利
- 資料3（SIG-FIN）: 1985年〜2006年の22年分で学習。長い学習期間が有効
- GMO FX APIの1H足の取得可能範囲は限定的だが、5年分なら確実に取得可能
- 2020年からなら以下の市場イベントをカバー:
  - コロナショック（2020/03）
  - 米利上げサイクル（2022-2023）
  - 日銀政策変更（2022/12, 2024/03）
  - 円安・円高両方のレジーム

**トレードオフ**:
- **利点**: 複数のレジーム（円安/円高、低ボラ/高ボラ）を含み、モデルの汎化性能を高める
- **欠点**: 初回収集に約30分かかる。2020年以前のレジーム（アベノミクス初期等）はカバーできない

### 判断4: データ品質チェック — 取得時に自動検証

**問題**: 取得したOHLCデータの品質をどう保証するか

**選択肢**:
1. 取得時に自動検証（欠損・異常値チェック）
2. 特徴量生成時にチェック
3. チェックなし

**決定**: 取得時に自動検証

**理由**:
- 資料5（MUFG）: 「未来予測力のある純度の高い学習データの使用が望ましい」。データ品質は予測精度の前提条件
- GMO FX APIは祝日・メンテナンス等でデータ欠損が起こりうる
- 欠損をそのまま特徴量生成に渡すと、テクニカル指標の計算が不正確になる

**チェック項目**:
```
1. OHLC整合性: high >= max(open, close), low <= min(open, close)
2. タイムスタンプ連続性: 1H間隔で欠損がないか（土日・祝日を除く）
3. 異常値: 前足からの変動率が極端（例: 5%超）な場合にログ警告
4. 重複: UNIQUE制約で排除（SQLite側で保証）
```

**トレードオフ**:
- **利点**: 不正データの混入を早期に検出。下流の特徴量・モデルへの影響を防止
- **欠点**: 実装工数がやや増える。ただし一度作れば再利用可能

## データフロー全体像

```
GMO FX Public API
    │
    ▼
[collector.py] ─── 1H OHLC取得（レート制限、リトライ、エラーハンドリング）
    │                │
    │                ▼
    │           品質チェック（OHLC整合性、欠損、異常値）
    │
    ▼
[storage.py] ─── SQLite保存（ohlcテーブル、重複排除）
    │
    ▼
[features/pipeline.py] ─── DataFrame読み出し → 特徴量生成 → ラベル生成
    │
    ▼
[models/trainer.py] ─── 学習・予測
```

## 関連ドキュメント

- [技術スタックADR](./001-technology-stack.md) — 判断4: GMO FX API、判断5: SQLite
- [予測モデル設計ADR](./002-prediction-model-design.md) — 判断2: ホライゾン、判断3: 特徴量
- [過学習対策ADR](./004-overfitting-prevention.md)
- [実装計画](../../plans/001-prediction-model.md) — TASK-001-03〜05

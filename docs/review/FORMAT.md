<!--
種別: レビュー形式
対象: 全モジュール共通
作成日: 2026-03-16
更新日: 2026-03-16
担当: AIエージェント
-->

# レビュー結果出力形式

## 重要度レベル

### Critical（即時修正必須）

セキュリティ脆弱性、データ損失、APIキー漏洩、資金損失リスクなど。

```markdown
### Critical

1. **[C-01] {タイトル}**
   - 箇所: `src/fx_auto_trading/trading/executor.py:OrderExecutor.place_order()`
   - 問題: {具体的な問題の説明}
   - 修正案: {具体的な修正コード}
```

### Medium（改善推奨）

パフォーマンス問題、エラーハンドリング不足、設計上の懸念など。

```markdown
### Medium

1. **[M-01] {タイトル}**
   - 箇所: `src/fx_auto_trading/data/collector.py:45`
   - 問題: {問題の説明}
   - 修正案: {修正コード}
```

### Low（提案）

コードスタイル、命名改善、ドキュメント追加など。

```markdown
### Low

1. **[L-01] {タイトル}**
   - 箇所: `src/fx_auto_trading/models/predictor.py:Predictor`
   - 提案: {改善提案}
```

### 確認済み（問題なし）

レビューで確認し、問題がなかった項目。

```markdown
### 確認済み

- [x] エラーハンドリングが適切
- [x] 型ヒントが正しい
- [x] テストカバレッジが十分
```

## サマリー形式

```markdown
## サマリー

| 重要度 | 件数 |
|--------|------|
| Critical | N |
| Medium | N |
| Low | N |
| 確認済み | N |
```

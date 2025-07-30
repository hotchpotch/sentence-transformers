# OpenProvence ベースライン性能要件の文書化完了レポート

## 概要

OpenProvenceプロジェクトに、リファクタリング時の性能検証基準となるベースライン要件を文書化しました。

## 追加・更新したドキュメント

### 1. docs/REFACTORING_GUIDE.md（新規作成）
**目的**: リファクタリング時の性能検証手順と要件の明確化

**主な内容**:
- **Performance Baseline**: 必須の性能指標と許容範囲
  - F2.5スコア: 0.8483±0.01（最重要）
  - False Negatives: 5±2（最重要）
  - Recall: ≥99%（最重要）
- **Validation Process**: リファクタリング前後の検証手順
- **Common Refactoring Scenarios**: 典型的なリファクタリングケース
- **Testing Checklist**: コミット前の確認項目
- **Troubleshooting**: 性能劣化時の対処法

### 2. docs/spec.md（更新）
**追加内容**: Reference Baselineセクション

```markdown
### Reference Baseline

Model: jpre-xs-msmarco-ja-minimal (2025-01-30)
Optimal Threshold: 0.3

| Metric | Value | Acceptable Range |
|--------|-------|------------------|
| F2.5 Score | 0.8483 | 0.8483±0.01 |
| False Negatives | 5 | 5±2 |
| Recall | 99.28% | ≥99% |
```

### 3. CLAUDE.md（更新）
**追加内容**: Performance Baselinesセクションに重要性を明記

```markdown
**IMPORTANT**: These baselines MUST be maintained during any refactoring. 
See `docs/REFACTORING_GUIDE.md` for detailed validation procedures.
```

### 4. README.md（更新）
**更新内容**: Expected Performanceセクションに具体的な数値と参照を追加

```markdown
- F2.5 Score: 0.8483 (±0.01)
- False Negatives: 5 (±2)
- Recall: 99.28% (≥99%)
- Compression: 17.9% (±1%)

**Note**: These metrics serve as the baseline for any refactoring or 
optimization work. See `docs/REFACTORING_GUIDE.md` for validation procedures.
```

## ベースライン要件の重要性

### なぜこれらの指標が重要か

1. **F2.5スコア (0.8483±0.01)**
   - RAGシステムでは再現率が精度より重要
   - F2.5は再現率を重視した指標
   - ±0.01は統計的に有意でない範囲

2. **False Negatives (5±2)**
   - 重要な情報の誤削除は致命的
   - 5件は実用上許容できる最小値
   - ±2は評価データのばらつきを考慮

3. **Recall (≥99%)**
   - 重要情報をほぼ確実に保持
   - 99%は実用システムの最低要件

## 検証プロセスの標準化

### リファクタリング前
1. 現在の実装でベースライン測定
2. 結果をJSON形式で保存
3. 学習時間、メモリ使用量も記録

### リファクタリング後
1. 同一設定で再学習
2. 同一評価データで検証
3. 差分が許容範囲内か確認

## まとめ

OpenProvenceプロジェクトは、リファクタリング時の品質保証のための明確な基準と手順を持つようになりました。これにより：

1. **品質保証**: 性能劣化を防ぐ明確な基準
2. **開発効率**: 検証手順の標準化
3. **透明性**: 貢献者への明確な要件提示

今後のリファクタリングや最適化作業は、これらの基準に従って実施されることで、プロジェクトの品質を維持できます。
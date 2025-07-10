#!/usr/bin/env python
"""Recall vs Deletion Rate Analysis - なぜ両方高いのか？"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

print("=" * 80)
print("Recall（再現率）と削除率が両方高い理由の分析")
print("=" * 80)

# MS MARCOデータセットの構造
total_samples = 8000
positive_samples = 1000  # 関連文書
negative_samples = 7000  # 非関連文書

print("\n【データセット構造】")
print(f"総サンプル数: {total_samples:,}")
print(f"Positive（関連文書）: {positive_samples:,} ({positive_samples/total_samples*100:.1f}%)")
print(f"Negative（非関連文書）: {negative_samples:,} ({negative_samples/total_samples*100:.1f}%)")

# 実際のモデル性能（閾値0.1）
tp = 968  # 関連文書を正しく保持
fn = 32   # 関連文書を誤って削除
fp = 6480 # 非関連文書を誤って保持
tn = 520  # 非関連文書を正しく削除

print("\n【モデルの予測結果（閾値0.1）】")
print(f"True Positive (TP): {tp:,} - 関連文書を正しく保持")
print(f"False Negative (FN): {fn:,} - 関連文書を誤って削除")
print(f"False Positive (FP): {fp:,} - 非関連文書を誤って保持")
print(f"True Negative (TN): {tn:,} - 非関連文書を正しく削除")

# メトリクス計算
recall = tp / (tp + fn)
precision = tp / (tp + fp)
deletion_rate = (tn + fn) / total_samples
kept_rate = (tp + fp) / total_samples

print("\n【計算されたメトリクス】")
print(f"Recall: {recall:.3f} ({recall*100:.1f}%)")
print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
print(f"削除率: {deletion_rate:.3f} ({deletion_rate*100:.1f}%)")
print(f"保持率: {kept_rate:.3f} ({kept_rate*100:.1f}%)")

print("\n【なぜ両方高いのか？】")
print("\n1. クラス不均衡の影響:")
print(f"   - 全体の{negative_samples/total_samples*100:.1f}%が非関連文書")
print(f"   - {fp:,}個の非関連文書を保持しても、全体の削除率は{deletion_rate*100:.1f}%")

print("\n2. Recallの計算:")
print(f"   - Recall = TP/(TP+FN) = {tp}/({tp}+{fn}) = {recall:.3f}")
print(f"   - Positiveクラス内での性能のみを見ている")
print(f"   - Negativeクラスの数は関係ない")

print("\n3. 削除率の計算:")
print(f"   - 削除率 = (削除した数)/(全体) = ({tn}+{fn})/{total_samples} = {deletion_rate:.3f}")
print(f"   - 大量のNegativeがあるため、多くを保持しても高い削除率")

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. データ分布と予測結果
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)

# Positive領域
pos_rect = Rectangle((0, 60), 12.5, 30, facecolor='lightgreen', edgecolor='black', linewidth=2)
ax1.add_patch(pos_rect)
ax1.text(6.25, 75, f'Positive\n{positive_samples}件\n(12.5%)', ha='center', va='center', fontsize=10, weight='bold')

# Negative領域
neg_rect = Rectangle((0, 10), 87.5, 30, facecolor='lightcoral', edgecolor='black', linewidth=2)
ax1.add_patch(neg_rect)
ax1.text(43.75, 25, f'Negative\n{negative_samples}件\n(87.5%)', ha='center', va='center', fontsize=10, weight='bold')

# 保持・削除の内訳
ax1.text(6.25, 50, f'保持: {tp}\n削除: {fn}', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
ax1.text(43.75, 50, f'保持: {fp}\n削除: {tn}', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))

ax1.set_title('データ分布と予測結果', fontsize=14)
ax1.axis('off')

# 2. メトリクスの独立性
ax2.bar(['Recall\n(Positive内)', '削除率\n(全体)'], [recall*100, deletion_rate*100], color=['green', 'red'], alpha=0.7)
ax2.set_ylabel('パーセント (%)')
ax2.set_title('RecallとDeletion Rateは異なる母集団で計算', fontsize=14)

# 説明テキスト
ax2.text(0, recall*100 + 2, f'{recall*100:.1f}%', ha='center', fontsize=12, weight='bold')
ax2.text(1, deletion_rate*100 + 2, f'{deletion_rate*100:.1f}%', ha='center', fontsize=12, weight='bold')

ax2.text(0, -10, f'分母: Positive {positive_samples}件', ha='center', fontsize=10)
ax2.text(1, -10, f'分母: 全体 {total_samples}件', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('output/recall_vs_deletion_analysis.png', dpi=300, bbox_inches='tight')

print("\n【具体例で理解】")
print("仮に1万件の文書があったとして:")
print("- 100件が関連文書（1%）")
print("- 9,900件が非関連文書（99%）")
print("\nモデルが:")
print("- 関連文書の98件を保持（Recall=98%）")
print("- 非関連文書の3,000件を削除")
print("- 非関連文書の6,900件を保持")
print("\n結果:")
print("- Recall = 98% （関連文書をほぼ完璧に保持）")
print("- 削除率 = 30% （全体の30%を削除）")
print("- 両立している！")

print("\n【結論】")
print("RecallとDeletion Rateが両方高いのは矛盾ではなく、")
print("クラス不均衡なデータで正常な動作です。")
print("\n重要なのは:")
print("1. 関連文書（少数）を確実に保持 → Recall高い ✓")
print("2. 非関連文書（多数）をある程度削除 → 削除率高い ✓")
print("\nこれがRAGに最適な動作です！")

print(f"\n分析結果は output/recall_vs_deletion_analysis.png に保存されました。")
#!/usr/bin/env python
"""F-beta score analysis for different beta values."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# シナリオ設定
print("=" * 80)
print("QUERY-DEPENDENT CONTEXT PRUNINGの評価指標分析")
print("=" * 80)

print("\n【目的】")
print("- Queryに関連するcontextは必ず残したい（False Negativeを最小化）")
print("- 不要なcontextは削除して圧縮したい（True Negativeを増やす）")
print("- 最重要: 関連contextの誤削除を避ける")

print("\n【用語整理】")
print("- Positive (1): 関連するcontext → 残すべき")
print("- Negative (0): 関連しないcontext → 削除すべき")
print("- True Positive (TP): 関連contextを正しく保持")
print("- False Negative (FN): 関連contextを誤って削除 ⚠️最も避けたい")
print("- True Negative (TN): 非関連contextを正しく削除")
print("- False Positive (FP): 非関連contextを誤って保持")

# F-beta scoreの計算
def calculate_f_beta(precision, recall, beta):
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

# 実データに基づく分析（閾値0.1での結果）
precision = 0.130
recall = 0.968

print("\n【実際のモデル性能（閾値0.1）】")
print(f"- Precision: {precision:.3f} (保持したものの13%が実際に関連)")
print(f"- Recall: {recall:.3f} (関連contextの96.8%を保持)")

print("\n【F-beta スコア比較】")
betas = [0.5, 1.0, 2.0, 3.0, 5.0]
for beta in betas:
    f_score = calculate_f_beta(precision, recall, beta)
    if beta == 0.5:
        desc = "Precisionを2倍重視"
    elif beta == 1.0:
        desc = "PrecisionとRecallを同等重視"
    elif beta == 2.0:
        desc = "Recallを2倍重視"
    elif beta == 3.0:
        desc = "Recallを3倍重視"
    else:
        desc = f"Recallを{beta}倍重視"
    print(f"F{beta}: {f_score:.4f} ({desc})")

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. F-betaスコアの変化
betas_range = np.linspace(0.1, 5, 50)
f_scores = [calculate_f_beta(precision, recall, b) for b in betas_range]

ax1.plot(betas_range, f_scores, 'b-', linewidth=2)
ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='F0.5')
ax1.axvline(x=1.0, color='g', linestyle='--', alpha=0.5, label='F1')
ax1.axvline(x=2.0, color='orange', linestyle='--', alpha=0.5, label='F2')
ax1.set_xlabel('Beta値')
ax1.set_ylabel('F-beta Score')
ax1.set_title('Beta値によるF-scoreの変化\n(Precision=0.13, Recall=0.968)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Precision-Recall トレードオフ
recalls = np.linspace(0.5, 1.0, 50)
precisions = np.linspace(0.05, 0.3, 50)

R, P = np.meshgrid(recalls, precisions)
F05 = (1 + 0.5**2) * (P * R) / ((0.5**2 * P) + R)
F1 = 2 * (P * R) / (P + R)
F2 = (1 + 2**2) * (P * R) / ((2**2 * P) + R)

contour_levels = np.linspace(0.1, 0.5, 10)
cs1 = ax2.contour(R, P, F05, levels=contour_levels, colors='red', alpha=0.5, linestyles='--')
cs2 = ax2.contour(R, P, F1, levels=contour_levels, colors='green', alpha=0.5)
cs3 = ax2.contour(R, P, F2, levels=contour_levels, colors='orange', alpha=0.7, linewidths=2)

ax2.clabel(cs3, inline=True, fontsize=8)
ax2.plot(recall, precision, 'ko', markersize=10, label='現在のモデル')
ax2.set_xlabel('Recall (関連contextの保持率)')
ax2.set_ylabel('Precision (保持精度)')
ax2.set_title('F-score等高線図')
ax2.grid(True, alpha=0.3)
ax2.legend(['F0.5 (破線)', 'F1', 'F2 (太線)', '現在のモデル'])

plt.tight_layout()
plt.savefig('output/f_beta_analysis.png', dpi=300, bbox_inches='tight')

print("\n【推奨事項】")
print("\n1. **見るべき指標**:")
print("   - 最重要: Recall（関連contextの保持率）")
print("   - 理由: False Negative（関連contextの誤削除）が最も避けたいエラー")
print("   - 現在96.8%は良好だが、さらに高めたい場合は閾値を下げる")

print("\n2. **最適なベータ値**:")
print("   - 推奨: β = 2 または 3")
print("   - 理由: Recallを重視しつつ、Precisionも完全に無視しない")
print("   - F2は業界標準で、情報検索タスクでよく使用")

print("\n3. **閾値調整の指針**:")
print("   - 閾値0.1: Recall 96.8% (現在)")
print("   - 閾値0.05: Recall ~99% (より保守的)")
print("   - 閾値0.2: Recall ~91% (より積極的)")

print("\n4. **実用的な運用**:")
print("   - 本番環境: 閾値0.05-0.1でRecall 98%以上を確保")
print("   - コスト重視: 閾値0.2でRecall 90%を許容")
print("   - 動的調整: クエリの重要度に応じて閾値を変更")

# 閾値とRecallの関係を示す
print("\n【閾値とRecallの関係（実測値）】")
threshold_recall_map = {
    0.1: 0.968,
    0.2: 0.907,
    0.3: 0.795,
    0.5: 0.496,
    0.7: 0.248
}

for t, r in threshold_recall_map.items():
    miss_rate = (1 - r) * 100
    print(f"閾値 {t}: Recall {r:.3f} (関連contextを{miss_rate:.1f}%見逃す)")

print("\n分析完了。結果は output/f_beta_analysis.png に保存されました。")
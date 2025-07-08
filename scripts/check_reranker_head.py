#!/usr/bin/env python3
"""
Check original reranker model structure
"""

from transformers import AutoModelForSequenceClassification
import torch

# 元のrerankerモデルを確認
model = AutoModelForSequenceClassification.from_pretrained('hotchpotch/japanese-reranker-xsmall-v2')
print('Original model structure:')
print(model)
print('\nClassifier head:')
print(model.classifier)
print('\nClassifier weights shape:', model.classifier.weight.shape)
print('Has pretrained weights:', torch.any(model.classifier.weight != 0).item())

# 重みの統計情報
print('\nWeight statistics:')
print('Mean:', model.classifier.weight.mean().item())
print('Std:', model.classifier.weight.std().item())
print('Min:', model.classifier.weight.min().item())
print('Max:', model.classifier.weight.max().item())

# headモジュールも確認
if hasattr(model, 'head'):
    print('\nHead module:')
    print(model.head)
    print('Head dense weights shape:', model.head.dense.weight.shape)
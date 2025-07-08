#!/usr/bin/env python3
"""
Check trained model structure
"""

from sentence_transformers.provence import ProvenceEncoder
import torch

# 学習済みモデルを確認
model_path = "./outputs/provence-ja-minimal/final-model"
model = ProvenceEncoder.from_pretrained(model_path)

print('Trained model structure:')
print(f'Ranking model type: {type(model.ranking_model)}')
print(f'Has classifier: {hasattr(model.ranking_model, "classifier")}')

if hasattr(model.ranking_model, "classifier"):
    print('\nRanking classifier:')
    print(model.ranking_model.classifier)
    print('Weights shape:', model.ranking_model.classifier.weight.shape)
    print('Weights stats:')
    print('  Mean:', model.ranking_model.classifier.weight.mean().item())
    print('  Std:', model.ranking_model.classifier.weight.std().item())
    
print('\nPruning head:')
print(model.pruning_head)
print('Classifier weights shape:', model.pruning_head.classifier.weight.shape)
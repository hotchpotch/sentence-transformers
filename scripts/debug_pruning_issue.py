#!/usr/bin/env python3
"""
Debug pruning issue in ProvenceEncoder.
"""

import logging
import numpy as np
from collections import Counter
from datasets import load_from_disk
from sentence_transformers.provence import ProvenceEncoder
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_dataset_labels():
    """Analyze pruning labels in the dataset."""
    logger.info("Analyzing dataset pruning labels...")
    
    dataset = load_from_disk("tmp/datasets/dev-dataset/minimal")
    train_data = dataset['train']
    
    # Check pruning labels distribution
    all_pruning_labels = []
    for example in train_data.select(range(100)):  # Sample 100 examples
        labels = example.get('pruning_labels', [])
        all_pruning_labels.extend(labels)
    
    label_counts = Counter(all_pruning_labels)
    total = sum(label_counts.values())
    
    logger.info("Pruning labels distribution:")
    for label, count in label_counts.items():
        percentage = count / total * 100
        logger.info(f"  Label {label}: {count} ({percentage:.1f}%)")
    
    return label_counts


def test_model_components():
    """Test individual model components."""
    logger.info("Testing model components...")
    
    # Load model
    model = ProvenceEncoder.from_pretrained("tmp/models/provence-minimal/final")
    model.eval()
    
    # Create dummy inputs
    query = "テストクエリ"
    document = "これは最初の文です。これは2番目の文です。これは3番目の文です。"
    
    # Tokenize
    tokenized = model.tokenizer(
        [[query, document]],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(model.device)
    
    logger.info("Input shapes:")
    for key, value in tokenized.items():
        logger.info(f"  {key}: {value.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model.forward(**tokenized)
    
    logger.info("Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape}, range: [{value.min():.4f}, {value.max():.4f}]")
    
    # Check pruning head directly
    ranking_model_outputs = model.ranking_model(**tokenized, output_hidden_states=True)
    hidden_states = ranking_model_outputs.hidden_states[-1]
    
    pruning_outputs = model.pruning_head(
        hidden_states=hidden_states,
        attention_mask=tokenized['attention_mask']
    )
    
    logger.info("Pruning head outputs:")
    logger.info(f"  logits shape: {pruning_outputs.logits.shape}")
    logger.info(f"  logits range: [{pruning_outputs.logits.min():.4f}, {pruning_outputs.logits.max():.4f}]")
    
    # Apply softmax to see probabilities
    probs = torch.softmax(pruning_outputs.logits, dim=-1)
    keep_probs = probs[:, :, 1]  # Probability of keeping
    
    logger.info(f"  keep probabilities: {keep_probs.flatten()[:10]}")  # First 10
    logger.info(f"  keep prob range: [{keep_probs.min():.4f}, {keep_probs.max():.4f}]")
    logger.info(f"  keep prob mean: {keep_probs.mean():.4f}")
    
    return outputs, pruning_outputs


def check_loss_function():
    """Check loss function behavior."""
    logger.info("Checking loss function...")
    
    from sentence_transformers.provence import ProvenceLoss
    
    model = ProvenceEncoder.from_pretrained("tmp/models/provence-minimal/final")
    loss_fn = ProvenceLoss(model, ranking_weight=1.0, pruning_weight=0.5)
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    sentence_features = [{
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(model.device),
        "attention_mask": torch.ones(batch_size, seq_len).to(model.device)
    }]
    
    labels = {
        "ranking_labels": torch.tensor([1.0, 0.0]).to(model.device),
        "pruning_labels": torch.tensor([[1, 0, 1], [0, 1, 0]]).to(model.device),  # Mixed labels
        "sentence_boundaries": torch.tensor([
            [[0, 3], [3, 6], [6, 9]],
            [[0, 4], [4, 7], [7, 10]]
        ]).to(model.device)
    }
    
    loss = loss_fn(sentence_features, labels)
    logger.info(f"Loss value: {loss.item():.4f}")
    
    if hasattr(loss_fn, 'last_losses'):
        for loss_name, loss_value in loss_fn.last_losses.items():
            logger.info(f"  {loss_name}: {loss_value.item():.4f}")


def main():
    logger.info("=" * 70)
    logger.info("Debugging Pruning Issue")
    logger.info("=" * 70)
    
    # 1. Analyze dataset
    label_distribution = analyze_dataset_labels()
    
    # 2. Test model components
    logger.info("\n" + "="*50)
    model_outputs, pruning_outputs = test_model_components()
    
    # 3. Check loss function
    logger.info("\n" + "="*50)
    check_loss_function()
    
    # 4. Summary and recommendations
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSIS AND RECOMMENDATIONS")
    logger.info("="*70)
    
    if label_distribution:
        keep_ratio = label_distribution.get(1, 0) / sum(label_distribution.values())
        logger.info(f"Dataset keep ratio: {keep_ratio:.2%}")
        
        if keep_ratio < 0.1:
            logger.info("❌ Problem: Dataset has too few 'keep' labels (< 10%)")
            logger.info("   Recommendation: Check dataset creation, ensure balanced pruning labels")
        elif keep_ratio > 0.9:
            logger.info("❌ Problem: Dataset has too many 'keep' labels (> 90%)")
            logger.info("   Recommendation: Check dataset creation, ensure some pruning")
        else:
            logger.info(f"✅ Dataset labels seem balanced ({keep_ratio:.1%} keep)")
    
    logger.info("\nNext steps:")
    logger.info("1. Fix pruning label generation in dataset creation")
    logger.info("2. Increase pruning loss weight in training")
    logger.info("3. Check sentence boundary calculation")
    logger.info("4. Re-train model with corrected setup")


if __name__ == "__main__":
    main()
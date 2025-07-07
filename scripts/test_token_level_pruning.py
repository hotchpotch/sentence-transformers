#!/usr/bin/env python3
"""
Test token-level pruning functionality in ProvenceEncoder.
"""

import logging
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk

from sentence_transformers.provence import ProvenceEncoder

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def analyze_token_level_predictions(model: ProvenceEncoder, query: str, document: str, tokenizer=None):
    """Analyze token-level pruning predictions in detail."""
    if tokenizer is None:
        tokenizer = model.tokenizer
    
    logger.info("="*60)
    logger.info("TOKEN-LEVEL ANALYSIS")
    logger.info("="*60)
    logger.info(f"Query: {query}")
    logger.info(f"Document: {document}")
    logger.info(f"Document length: {len(document)} chars")
    
    # Tokenize the input
    inputs = tokenizer(
        [[query, document]],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(model.device)
    
    # Get token strings for analysis
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    logger.info(f"\nTokenization:")
    logger.info(f"  Total tokens: {len(tokens)}")
    logger.info(f"  Input shape: {inputs['input_ids'].shape}")
    
    # Forward pass through model
    with torch.no_grad():
        outputs = model.forward(**inputs)
    
    # Get pruning logits and probabilities
    pruning_logits = outputs['pruning_logits']  # [batch, seq_len, 2]
    pruning_probs = torch.softmax(pruning_logits, dim=-1)
    keep_probs = pruning_probs[0, :, 1]  # Probability of keeping each token
    prune_probs = pruning_probs[0, :, 0]  # Probability of pruning each token
    
    logger.info(f"\nPruning probabilities:")
    logger.info(f"  Shape: {keep_probs.shape}")
    logger.info(f"  Keep prob range: [{keep_probs.min():.4f}, {keep_probs.max():.4f}]")
    logger.info(f"  Keep prob mean: {keep_probs.mean():.4f}")
    logger.info(f"  Prune prob mean: {prune_probs.mean():.4f}")
    
    # Analyze attention mask
    attention_mask = inputs['attention_mask'][0]
    valid_tokens = attention_mask.bool()
    
    logger.info(f"\nValid tokens analysis:")
    logger.info(f"  Valid tokens: {valid_tokens.sum()}")
    logger.info(f"  Padding tokens: {(~valid_tokens).sum()}")
    
    # Detailed token analysis
    logger.info(f"\nDETAILED TOKEN ANALYSIS:")
    logger.info(f"{'Token':<15} {'Keep_Prob':<10} {'Prune_Prob':<11} {'Decision':<10} {'Type':<10}")
    logger.info("-" * 70)
    
    keep_count = 0
    prune_count = 0
    special_tokens = ['[CLS]', '[SEP]', '[PAD]']
    
    for i, (token, keep_p, prune_p, valid) in enumerate(zip(
        tokens, keep_probs.cpu(), prune_probs.cpu(), valid_tokens.cpu()
    )):
        if not valid:
            continue
            
        # Decision based on threshold
        decision = "KEEP" if keep_p > 0.5 else "PRUNE"
        
        if decision == "KEEP":
            keep_count += 1
        else:
            prune_count += 1
        
        # Token type
        if token in special_tokens:
            token_type = "SPECIAL"
        elif token.startswith('##'):
            token_type = "SUBWORD"
        else:
            token_type = "WORD"
        
        # Display with color coding for extreme probabilities
        if keep_p > 0.8:
            marker = "✓✓"
        elif keep_p > 0.6:
            marker = "✓"
        elif keep_p < 0.2:
            marker = "✗✗"
        elif keep_p < 0.4:
            marker = "✗"
        else:
            marker = "~"
        
        logger.info(f"{token:<15} {keep_p:.4f}     {prune_p:.4f}      {decision:<10} {token_type:<10} {marker}")
    
    # Summary statistics
    valid_keep_probs = keep_probs[valid_tokens]
    valid_prune_probs = prune_probs[valid_tokens]
    
    logger.info(f"\nSUMMARY STATISTICS:")
    logger.info(f"  Total valid tokens: {len(valid_keep_probs)}")
    logger.info(f"  Tokens to KEEP (>0.5): {keep_count}")
    logger.info(f"  Tokens to PRUNE (≤0.5): {prune_count}")
    logger.info(f"  Keep ratio: {keep_count / len(valid_keep_probs):.2%}")
    logger.info(f"  Prune ratio: {prune_count / len(valid_keep_probs):.2%}")
    
    # Extreme probabilities analysis
    very_confident_keep = (valid_keep_probs > 0.8).sum()
    confident_keep = (valid_keep_probs > 0.6).sum()
    very_confident_prune = (valid_keep_probs < 0.2).sum()
    confident_prune = (valid_keep_probs < 0.4).sum()
    uncertain = ((valid_keep_probs >= 0.4) & (valid_keep_probs <= 0.6)).sum()
    
    logger.info(f"\nCONFIDENCE ANALYSIS:")
    logger.info(f"  Very confident KEEP (>0.8): {very_confident_keep}")
    logger.info(f"  Confident KEEP (>0.6): {confident_keep}")
    logger.info(f"  Uncertain (0.4-0.6): {uncertain}")
    logger.info(f"  Confident PRUNE (<0.4): {confident_prune}")
    logger.info(f"  Very confident PRUNE (<0.2): {very_confident_prune}")
    
    return {
        'tokens': tokens,
        'keep_probs': keep_probs.cpu().numpy(),
        'prune_probs': prune_probs.cpu().numpy(),
        'valid_tokens': valid_tokens.cpu().numpy(),
        'keep_count': keep_count,
        'prune_count': prune_count,
        'stats': {
            'very_confident_keep': very_confident_keep.item(),
            'confident_keep': confident_keep.item(),
            'uncertain': uncertain.item(),
            'confident_prune': confident_prune.item(),
            'very_confident_prune': very_confident_prune.item()
        }
    }


def test_various_examples(model: ProvenceEncoder):
    """Test token-level analysis on various examples."""
    
    test_cases = [
        {
            'name': 'Relevant Match',
            'query': 'Python プログラミング',
            'document': 'Pythonは素晴らしいプログラミング言語です。データサイエンスに最適です。'
        },
        {
            'name': 'Partially Relevant',
            'query': 'Python プログラミング',
            'document': 'Pythonは良い言語です。今日は雨が降っています。天気が悪いですね。'
        },
        {
            'name': 'Not Relevant',
            'query': 'Python プログラミング',
            'document': '今日は晴れています。散歩に行きましょう。公園が綺麗です。'
        },
        {
            'name': 'Mixed Content',
            'query': '機械学習',
            'document': '機械学習は人工知能の分野です。ランチは何を食べますか。アルゴリズムが重要です。'
        }
    ]
    
    all_results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST CASE {i+1}: {test_case['name']}")
        logger.info(f"{'='*80}")
        
        result = analyze_token_level_predictions(
            model, 
            test_case['query'], 
            test_case['document']
        )
        result['test_name'] = test_case['name']
        all_results.append(result)
    
    return all_results


def test_sentence_level_aggregation(model: ProvenceEncoder):
    """Test how token-level predictions aggregate to sentence-level."""
    logger.info(f"\n{'='*80}")
    logger.info("SENTENCE-LEVEL AGGREGATION TEST")
    logger.info(f"{'='*80}")
    
    query = "機械学習アルゴリズム"
    document = "機械学習は重要です。今日は雨です。アルゴリズムを学びましょう。"
    
    logger.info(f"Query: {query}")
    logger.info(f"Document: {document}")
    
    # Get token-level analysis
    token_result = analyze_token_level_predictions(model, query, document)
    
    # Get sentence-level prediction
    sentence_result = model.predict_with_pruning(
        (query, document),
        pruning_threshold=0.5,
        return_documents=True
    )
    
    logger.info(f"\nSENTENCE-LEVEL RESULTS:")
    logger.info(f"  Total sentences: {len(sentence_result.sentences[0])}")
    logger.info(f"  Sentences: {sentence_result.sentences[0]}")
    logger.info(f"  Pruned sentences: {sentence_result.num_pruned_sentences}")
    logger.info(f"  Compression ratio: {sentence_result.compression_ratio:.2%}")
    logger.info(f"  Ranking score: {sentence_result.ranking_scores:.3f}")
    
    return token_result, sentence_result


def main():
    logger.info("=" * 80)
    logger.info("TOKEN-LEVEL PRUNING ANALYSIS")
    logger.info("=" * 80)
    
    # Load model - try different paths
    model_paths = [
        "tmp/models/provence-minimal-fixed/final",
        "tmp/models/provence-minimal-fixed/checkpoint-300-best",
        "tmp/models/provence-minimal/final",
        "tmp/models/provence-minimal/checkpoint-400-best"
    ]
    
    model = None
    for path in model_paths:
        if Path(path).exists():
            logger.info(f"Loading model from {path}...")
            model = ProvenceEncoder.from_pretrained(path)
            model.eval()
            break
    
    if model is None:
        logger.error("No trained model found!")
        return
    
    logger.info(f"Model device: {model.device}")
    logger.info(f"Model loaded successfully!")
    
    # Test 1: Various examples with token-level analysis
    logger.info(f"\n{'='*80}")
    logger.info("TESTING VARIOUS EXAMPLES")
    logger.info(f"{'='*80}")
    
    all_results = test_various_examples(model)
    
    # Test 2: Sentence-level aggregation
    token_result, sentence_result = test_sentence_level_aggregation(model)
    
    # Overall assessment
    logger.info(f"\n{'='*80}")
    logger.info("OVERALL TOKEN-LEVEL ASSESSMENT")
    logger.info(f"{'='*80}")
    
    # Analyze patterns across all test cases
    total_confident_decisions = 0
    total_uncertain_decisions = 0
    
    for result in all_results:
        stats = result['stats']
        confident = stats['very_confident_keep'] + stats['confident_keep'] + \
                   stats['confident_prune'] + stats['very_confident_prune']
        uncertain = stats['uncertain']
        
        total_confident_decisions += confident
        total_uncertain_decisions += uncertain
        
        logger.info(f"{result['test_name']}:")
        logger.info(f"  Keep ratio: {result['keep_count']}/{result['keep_count'] + result['prune_count']} = {result['keep_count']/(result['keep_count'] + result['prune_count']):.2%}")
        logger.info(f"  Confident decisions: {confident}")
        logger.info(f"  Uncertain decisions: {uncertain}")
    
    total_decisions = total_confident_decisions + total_uncertain_decisions
    confidence_ratio = total_confident_decisions / total_decisions if total_decisions > 0 else 0
    
    logger.info(f"\nOVERALL STATISTICS:")
    logger.info(f"  Total decisions: {total_decisions}")
    logger.info(f"  Confident decisions: {total_confident_decisions} ({confidence_ratio:.1%})")
    logger.info(f"  Uncertain decisions: {total_uncertain_decisions} ({1-confidence_ratio:.1%})")
    
    # Assessment
    if confidence_ratio > 0.7:
        confidence_status = "✓ GOOD"
    elif confidence_ratio > 0.5:
        confidence_status = "⚠ MODERATE"
    else:
        confidence_status = "✗ POOR"
    
    logger.info(f"\nTOKEN-LEVEL ASSESSMENT: {confidence_status}")
    
    if confidence_ratio < 0.5:
        logger.info("⚠ Low confidence in token decisions - model may need more training")
    
    if all(r['keep_count']/(r['keep_count'] + r['prune_count']) > 0.9 for r in all_results):
        logger.info("⚠ Model keeps too many tokens - may be too conservative")
    elif all(r['keep_count']/(r['keep_count'] + r['prune_count']) < 0.1 for r in all_results):
        logger.info("⚠ Model prunes too many tokens - may be too aggressive")
    else:
        logger.info("✓ Token pruning ratios seem reasonable")


if __name__ == "__main__":
    main()
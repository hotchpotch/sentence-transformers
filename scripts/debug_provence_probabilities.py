#!/usr/bin/env python3
"""
Debug ProvenceEncoder pruning probabilities.
"""

import logging
import torch
import numpy as np
from datasets import load_from_disk
from sentence_transformers.provence import ProvenceEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Debugging ProvenceEncoder pruning probabilities...")
    
    # Load model
    model = ProvenceEncoder.from_pretrained("tmp/models/provence-small/final")
    model.eval()
    
    # Load test dataset
    dataset = load_from_disk("tmp/datasets/dev-dataset/small-processed")
    test_dataset = dataset['test']
    
    # Test with one sample
    sample = test_dataset[0]
    query = sample['query']
    document = sample['text']
    teacher_score = sample.get('teacher_score', 0.0)
    
    logger.info(f"テストサンプル:")
    logger.info(f"  クエリ: {query}")
    logger.info(f"  教師スコア: {teacher_score:.3f}")
    logger.info(f"  文書: {document[:200]}...")
    
    # Tokenize
    from sentence_transformers.utils.text_chunking import MultilingualChunker
    text_chunker = MultilingualChunker()
    
    # Get raw model outputs
    inputs = model.tokenizer(
        query, document,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    
    # Move inputs to model device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(model.device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        ranking_logits = outputs['ranking_logits']
        pruning_logits = outputs['pruning_logits']
        
        logger.info(f"\nモデル出力:")
        logger.info(f"  Ranking logits: {ranking_logits}")
        logger.info(f"  Ranking score (sigmoid): {torch.sigmoid(ranking_logits).item():.3f}")
        logger.info(f"  Pruning logits shape: {pruning_logits.shape}")
        
        # Get pruning probabilities
        pruning_probs = torch.softmax(pruning_logits, dim=-1)
        keep_probs = pruning_probs[0, :, 1]  # Probability of keeping each token
        prune_probs = pruning_probs[0, :, 0]  # Probability of pruning each token
        
        logger.info(f"  Keep probabilities range: {keep_probs.min():.3f} - {keep_probs.max():.3f}")
        logger.info(f"  Keep probabilities mean: {keep_probs.mean():.3f}")
        logger.info(f"  Keep probabilities std: {keep_probs.std():.3f}")
        
        # Analyze token-level probabilities
        tokens = model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        logger.info(f"\nトークンごとの保持確率 (最初の50トークン):")
        for i in range(min(50, len(tokens))):
            token = tokens[i]
            keep_prob = keep_probs[i].item()
            logger.info(f"  {i:2d}: '{token}' -> {keep_prob:.3f}")
        
        # Test various thresholds
        logger.info(f"\n閾値別統計:")
        thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        
        for threshold in thresholds:
            kept_mask = keep_probs > threshold
            kept_ratio = kept_mask.float().mean().item()
            compression_ratio = 1.0 - kept_ratio
            
            logger.info(f"  閾値 {threshold}: {kept_ratio:.1%} 保持, {compression_ratio:.1%} 圧縮")
    
    # Test sentence-level analysis
    logger.info(f"\n文レベル分析:")
    
    # Chunk the document into sentences
    chunks_result = text_chunker.chunk_text(document, language="auto")
    sentences = [chunk for chunk, _ in chunks_result]
    
    logger.info(f"文数: {len(sentences)}")
    for i, sentence in enumerate(sentences):
        logger.info(f"  {i+1}: {sentence.strip()}")
    
    # Test the predict_with_pruning method with very low thresholds
    logger.info(f"\n極低閾値でのテスト:")
    
    low_thresholds = [0.001, 0.01, 0.05]
    for threshold in low_thresholds:
        try:
            result = model.predict_with_pruning(
                (query, document),
                pruning_threshold=threshold,
                return_documents=True
            )
            
            compression = result.compression_ratio
            reranker_score = result.ranking_scores
            
            logger.info(f"  閾値 {threshold}: 圧縮率 {compression:.1%}, Rerankerスコア {reranker_score:.3f}")
            
            if result.pruned_documents and result.pruned_documents[0]:
                pruned_doc = result.pruned_documents[0]
                logger.info(f"    プルーニング後: {len(pruned_doc)}文字")
                logger.info(f"    「{pruned_doc[:200]}{'...' if len(pruned_doc) > 200 else ''}」")
            else:
                logger.info(f"    プルーニング後: [完全削除]")
                
        except Exception as e:
            logger.warning(f"  閾値 {threshold}: エラー {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Show best pruning results with optimal threshold.
"""

import logging
import random
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers.provence import ProvenceEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load model and dataset
    model = ProvenceEncoder.from_pretrained("tmp/models/provence-minimal-fixed/final")
    model.eval()
    
    dataset = load_from_disk("tmp/datasets/dev-dataset/minimal")
    test_dataset = dataset['test']
    
    # Select interesting samples with various teacher scores
    samples = []
    for i, example in enumerate(test_dataset):
        teacher_score = example.get('teacher_score', 0.0)
        if teacher_score > 0.1 or (teacher_score < 0.05 and len(samples) < 15):
            samples.append((i, example))
        if len(samples) >= 10:
            break
    
    threshold = 0.15  # Best threshold based on earlier tests
    
    logger.info(f"プルーニング結果 (閾値={threshold})")
    logger.info("=" * 80)
    
    for idx, (original_idx, sample) in enumerate(samples):
        query = sample['query']
        document = sample.get('text', sample.get('document', ''))
        teacher_score = sample.get('teacher_score', 0.0)
        
        if not document.strip():
            continue
        
        result = model.predict_with_pruning(
            (query, document),
            pruning_threshold=threshold,
            return_documents=True
        )
        
        logger.info(f"\n【サンプル #{idx+1}】")
        logger.info(f"クエリ: {query}")
        logger.info(f"教師スコア: {teacher_score:.3f}")
        logger.info(f"Rerankerスコア: {result.ranking_scores:.3f}")
        logger.info(f"圧縮率: {result.compression_ratio:.1%}")
        
        logger.info(f"\n元文書 ({len(document)}文字):")
        logger.info(f"「{document}」")
        
        if result.pruned_documents and result.pruned_documents[0]:
            logger.info(f"\nプルーニング後 ({len(result.pruned_documents[0])}文字):")
            logger.info(f"「{result.pruned_documents[0]}」")
        else:
            logger.info(f"\nプルーニング後: [完全削除]")
        
        # Show sentence analysis
        logger.info(f"\n文ごと分析:")
        for i, sentence in enumerate(result.sentences[0]):
            status = "削除" if i < result.num_pruned_sentences else "保持"
            logger.info(f"  {i+1}. [{status}] {sentence.strip()}")
        
        logger.info("-" * 50)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test the trained ProvenceEncoder on small dataset.
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
    logger.info("Testing trained ProvenceEncoder on small dataset...")
    
    # Load model
    model_path = "tmp/models/provence-small/final"
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Load test dataset
    dataset = load_from_disk("tmp/datasets/dev-dataset/small-processed")
    test_dataset = dataset['test']
    
    # Test with a few random samples
    logger.info("Testing with random samples from test dataset...")
    
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
    num_samples = 5
    
    random.seed(42)
    sample_indices = random.sample(range(len(test_dataset)), num_samples)
    
    logger.info("=" * 80)
    
    for i, sample_idx in enumerate(sample_indices):
        sample = test_dataset[sample_idx]
        query = sample['query']
        document = sample['text']
        teacher_score = sample.get('teacher_score', 0.0)
        
        logger.info(f"\n【サンプル #{i+1}】")
        logger.info(f"クエリ: {query}")
        logger.info(f"教師スコア: {teacher_score:.3f}")
        logger.info(f"文書長: {len(document)}文字")
        logger.info(f"文書: {document[:200]}{'...' if len(document) > 200 else ''}")
        
        # Test different thresholds
        logger.info(f"\n閾値別プルーニング結果:")
        
        for threshold in thresholds:
            try:
                result = model.predict_with_pruning(
                    (query, document),
                    pruning_threshold=threshold,
                    return_documents=True
                )
                
                compression = result.compression_ratio
                reranker_score = result.ranking_scores
                
                logger.info(f"  閾値 {threshold}: 圧縮率 {compression:.1%}, Rerankerスコア {reranker_score:.3f}")
                
                if threshold == 0.15:  # Show details for middle threshold
                    if result.pruned_documents and result.pruned_documents[0]:
                        pruned_length = len(result.pruned_documents[0])
                        logger.info(f"    プルーニング後: {pruned_length}文字")
                        logger.info(f"    「{result.pruned_documents[0][:150]}{'...' if len(result.pruned_documents[0]) > 150 else ''}」")
                    else:
                        logger.info(f"    プルーニング後: [完全削除]")
                        
            except Exception as e:
                logger.warning(f"  閾値 {threshold}: エラー {e}")
        
        logger.info("-" * 60)
    
    # Summary statistics
    logger.info("\n統計情報:")
    
    # Test on larger sample for statistics
    stat_samples = random.sample(range(len(test_dataset)), min(50, len(test_dataset)))
    
    compressions_015 = []
    reranker_scores = []
    
    for sample_idx in stat_samples:
        sample = test_dataset[sample_idx]
        query = sample['query']
        document = sample['text']
        
        try:
            result = model.predict_with_pruning(
                (query, document),
                pruning_threshold=0.15
            )
            compressions_015.append(result.compression_ratio)
            reranker_scores.append(result.ranking_scores)
        except:
            continue
    
    if compressions_015:
        avg_compression = sum(compressions_015) / len(compressions_015)
        avg_reranker = sum(reranker_scores) / len(reranker_scores)
        min_compression = min(compressions_015)
        max_compression = max(compressions_015)
        
        logger.info(f"閾値0.15での統計 ({len(compressions_015)}サンプル):")
        logger.info(f"  平均圧縮率: {avg_compression:.1%}")
        logger.info(f"  圧縮率範囲: {min_compression:.1%} - {max_compression:.1%}")
        logger.info(f"  平均Rerankerスコア: {avg_reranker:.3f}")
        
        # Analyze by teacher score ranges
        high_score_compressions = []
        low_score_compressions = []
        
        for sample_idx in stat_samples:
            sample = test_dataset[sample_idx]
            teacher_score = sample.get('teacher_score', 0.0)
            query = sample['query']
            document = sample['text']
            
            try:
                result = model.predict_with_pruning(
                    (query, document),
                    pruning_threshold=0.15
                )
                
                if teacher_score > 0.1:
                    high_score_compressions.append(result.compression_ratio)
                else:
                    low_score_compressions.append(result.compression_ratio)
            except:
                continue
        
        if high_score_compressions:
            avg_high = sum(high_score_compressions) / len(high_score_compressions)
            logger.info(f"  高関連性サンプル (teacher_score > 0.1): {avg_high:.1%} 平均圧縮率")
        
        if low_score_compressions:
            avg_low = sum(low_score_compressions) / len(low_score_compressions)
            logger.info(f"  低関連性サンプル (teacher_score ≤ 0.1): {avg_low:.1%} 平均圧縮率")
    
    logger.info("\nテスト完了!")

if __name__ == "__main__":
    main()
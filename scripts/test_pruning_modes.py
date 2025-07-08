#!/usr/bin/env python
"""
Test script to verify both pruning modes (reranking_pruning and pruning_only) work correctly.
"""

import os
import sys
from pathlib import Path
import logging
import torch
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from sentence_transformers.pruning import (
    PruningEncoder, PruningTrainer, PruningLoss, PruningDataCollator
)
from sentence_transformers.pruning.data_structures import (
    RerankingPruningOutput, PruningOnlyOutput
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'./log/test_pruning_modes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def test_reranking_pruning_mode(dataset_name: str = 'minimal'):
    """Test reranking + pruning mode."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing RERANKING + PRUNING mode with {dataset_name} dataset")
    logger.info(f"{'='*50}\n")
    
    # Load dataset
    dataset_config = 'ja-minimal' if dataset_name == 'minimal' else 'ja-small'
    logger.info(f"Loading dataset: hotchpotch/wip-query-context-pruner-with-teacher-scores/{dataset_config}")
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        dataset_config
    )
    train_dataset = dataset['train']
    
    # Model configuration
    model_name = "hotchpotch/japanese-reranker-xsmall-v2"
    
    try:
        # Initialize model
        logger.info(f"Initializing PruningEncoder in reranking_pruning mode")
        model = PruningEncoder(
            model_name_or_path=model_name,
            mode="reranking_pruning",
            max_length=512,
            device="cpu"  # Use CPU for testing to avoid device issues
        )
        logger.info("✓ Model initialized successfully")
        
        # Data collator
        data_collator = PruningDataCollator(
            tokenizer=model.tokenizer,
            max_length=512,
            mode="reranking_pruning",
            padding=True,
            truncation=True
        )
        logger.info("✓ Data collator initialized successfully")
        
        # Test data collation
        batch = data_collator([train_dataset[i] for i in range(min(4, len(train_dataset)))])
        logger.info(f"✓ Data collation successful. Batch keys: {batch['labels'].keys()}")
        
        # Check for required columns
        assert 'ranking_targets' in batch['labels'], "ranking_targets missing in reranking_pruning mode"
        assert 'pruning_labels' in batch['labels'], "pruning_labels missing"
        logger.info("✓ All required label columns present")
        
        # Loss function
        loss_fn = PruningLoss(
            model=model,
            mode="reranking_pruning"
        )
        logger.info("✓ Loss function initialized successfully")
        
        # Test forward pass
        outputs = model.forward(
            input_ids=batch['sentence_features'][0]['input_ids'][:2].to(model.device),
            attention_mask=batch['sentence_features'][0]['attention_mask'][:2].to(model.device)
        )
        assert 'ranking_logits' in outputs, "ranking_logits missing in outputs"
        assert 'pruning_logits' in outputs, "pruning_logits missing in outputs"
        logger.info(f"✓ Forward pass successful. Output keys: {outputs.keys()}")
        
        # Test loss computation
        loss = loss_fn(batch['sentence_features'], batch['labels'])
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.item() > 0, "Loss should be positive"
        logger.info(f"✓ Loss computation successful. Loss value: {loss.item():.4f}")
        
        # Test predict method
        test_pair = ("これは質問です", "これは文書です")
        score = model.predict(test_pair, apply_pruning=False)
        logger.info(f"✓ Predict (ranking only) successful. Score: {score}")
        
        # Test predict with pruning
        output_with_pruning = model.predict(test_pair, apply_pruning=True, return_documents=True)
        assert isinstance(output_with_pruning, RerankingPruningOutput), f"Expected RerankingPruningOutput, got {type(output_with_pruning)}"
        logger.info(f"✓ Predict with pruning successful. Compression ratio: {output_with_pruning.compression_ratio:.2f}")
        logger.info(f"✓ Output type is correct: {type(output_with_pruning).__name__}")
        
        logger.info(f"\n✅ RERANKING + PRUNING mode test PASSED for {dataset_name} dataset\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ RERANKING + PRUNING mode test FAILED: {str(e)}")
        raise


def test_pruning_only_mode(dataset_name: str = 'minimal'):
    """Test pruning-only mode."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing PRUNING-ONLY mode with {dataset_name} dataset")
    logger.info(f"{'='*50}\n")
    
    # Load dataset
    dataset_config = 'ja-minimal' if dataset_name == 'minimal' else 'ja-small'
    logger.info(f"Loading dataset: hotchpotch/wip-query-context-pruner-with-teacher-scores/{dataset_config}")
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        dataset_config
    )
    train_dataset = dataset['train']
    
    # Model configuration
    model_name = "cl-nagoya/ruri-v3-30m"
    
    try:
        # Initialize model
        logger.info(f"Initializing PruningEncoder in pruning_only mode")
        model = PruningEncoder(
            model_name_or_path=model_name,
            mode="pruning_only",
            max_length=512,
            device="cpu",  # Use CPU for testing to avoid device issues
            pruning_config={
                "hidden_size": 256,  # Match ruri-v3-30m actual hidden size
                "dropout": 0.1
            }
        )
        logger.info("✓ Model initialized successfully")
        
        # Data collator
        data_collator = PruningDataCollator(
            tokenizer=model.tokenizer,
            max_length=512,
            mode="pruning_only",
            padding=True,
            truncation=True
        )
        logger.info("✓ Data collator initialized successfully")
        
        # Test data collation
        batch = data_collator([train_dataset[i] for i in range(min(4, len(train_dataset)))])
        logger.info(f"✓ Data collation successful. Batch keys: {batch['labels'].keys()}")
        
        # Check that ranking_targets is NOT in labels for pruning_only mode
        assert 'ranking_targets' not in batch['labels'], "ranking_targets should not be in pruning_only mode"
        assert 'pruning_labels' in batch['labels'], "pruning_labels missing"
        logger.info("✓ Label columns correct for pruning_only mode")
        
        # Loss function
        loss_fn = PruningLoss(
            model=model,
            mode="pruning_only"
        )
        logger.info("✓ Loss function initialized successfully")
        
        # Test forward pass
        outputs = model.forward(
            input_ids=batch['sentence_features'][0]['input_ids'][:2].to(model.device),
            attention_mask=batch['sentence_features'][0]['attention_mask'][:2].to(model.device)
        )
        assert 'ranking_logits' not in outputs, "ranking_logits should not be in pruning_only outputs"
        assert 'pruning_logits' in outputs, "pruning_logits missing in outputs"
        logger.info(f"✓ Forward pass successful. Output keys: {outputs.keys()}")
        
        # Test loss computation
        loss = loss_fn(batch['sentence_features'], batch['labels'])
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.item() > 0, "Loss should be positive"
        logger.info(f"✓ Loss computation successful. Loss value: {loss.item():.4f}")
        
        # Test that predict without pruning raises error
        test_pair = ("これは質問です", "これは文書です")
        try:
            model.predict(test_pair, apply_pruning=False)
            assert False, "predict() without pruning should raise error in pruning_only mode"
        except ValueError as e:
            logger.info(f"✓ Correctly raised error for predict without pruning: {str(e)}")
        
        # Test prune_texts method
        queries = ["これは質問1です", "これは質問2です"]
        texts = ["これは文書1です。関連する内容。", "これは文書2です。別の内容。"]
        results = model.prune_texts(queries, texts, threshold=0.5)
        assert len(results) == 2, "Should return results for each query-text pair"
        assert 'pruned_text' in results[0], "Result should contain pruned_text"
        assert 'kept_ratio' in results[0], "Result should contain kept_ratio"
        logger.info(f"✓ prune_texts successful. First result: kept_ratio={results[0]['kept_ratio']:.2f}")
        
        # Test predict with pruning (should work)
        output_with_pruning = model.predict(test_pair, apply_pruning=True, return_documents=True)
        assert isinstance(output_with_pruning, PruningOnlyOutput), f"Expected PruningOnlyOutput, got {type(output_with_pruning)}"
        logger.info(f"✓ Predict with pruning successful. Compression ratio: {output_with_pruning.compression_ratio:.2f}")
        logger.info(f"✓ Output type is correct: {type(output_with_pruning).__name__}")
        
        logger.info(f"\n✅ PRUNING-ONLY mode test PASSED for {dataset_name} dataset\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ PRUNING-ONLY mode test FAILED: {str(e)}")
        raise


def test_save_load_functionality():
    """Test save and load functionality for both modes."""
    logger.info(f"\n{'='*50}")
    logger.info("Testing SAVE/LOAD functionality")
    logger.info(f"{'='*50}\n")
    
    import tempfile
    
    # Test reranking_pruning mode
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model
        model1 = PruningEncoder(
            model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
            mode="reranking_pruning"
        )
        save_path = os.path.join(tmpdir, "test_model")
        model1.save_pretrained(save_path)
        logger.info(f"✓ Saved reranking_pruning model to {save_path}")
        
        # Load model
        model2 = PruningEncoder.from_pretrained(save_path)
        assert model2.mode == "reranking_pruning", f"Mode mismatch: {model2.mode}"
        logger.info("✓ Loaded reranking_pruning model successfully")
    
    # Test pruning_only mode
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model
        model1 = PruningEncoder(
            model_name_or_path="cl-nagoya/ruri-v3-30m",
            mode="pruning_only"
        )
        save_path = os.path.join(tmpdir, "test_model")
        model1.save_pretrained(save_path)
        logger.info(f"✓ Saved pruning_only model to {save_path}")
        
        # Load model
        model2 = PruningEncoder.from_pretrained(save_path)
        assert model2.mode == "pruning_only", f"Mode mismatch: {model2.mode}"
        logger.info("✓ Loaded pruning_only model successfully")
    
    logger.info("\n✅ SAVE/LOAD functionality test PASSED\n")


def main():
    """Run all tests."""
    # Create log directory
    os.makedirs("./log", exist_ok=True)
    
    datasets_to_test = ["minimal", "small"]
    
    try:
        # Test save/load functionality first
        test_save_load_functionality()
        
        # Test both modes with different datasets
        for dataset_name in datasets_to_test:
            test_reranking_pruning_mode(dataset_name)
            test_pruning_only_mode(dataset_name)
        
        logger.info(f"\n{'='*50}")
        logger.info("🎉 ALL TESTS PASSED! 🎉")
        logger.info(f"{'='*50}\n")
        
    except Exception as e:
        logger.error(f"\n{'='*50}")
        logger.error("💥 TESTS FAILED! 💥")
        logger.error(f"Error: {str(e)}")
        logger.error(f"{'='*50}\n")
        raise


if __name__ == "__main__":
    main()
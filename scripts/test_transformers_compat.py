#!/usr/bin/env python
"""
Test Transformers library compatibility for PruningEncoder models.
"""

import os
import sys
from pathlib import Path
import torch
import logging
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sequence_classification(model_path: str):
    """Test loading reranking model with AutoModelForSequenceClassification."""
    logger.info("\n" + "="*60)
    logger.info("Testing AutoModelForSequenceClassification")
    logger.info("="*60)
    
    try:
        # Load model using AutoModel
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test inference
        query = "機械学習について"
        document = "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。"
        
        inputs = tokenizer(query, document, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            score = torch.sigmoid(logits).item()
        
        logger.info(f"Query: {query}")
        logger.info(f"Document: {document}")
        logger.info(f"Relevance score: {score:.4f}")
        logger.info("✓ Successfully loaded and ran inference with AutoModelForSequenceClassification")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load with AutoModelForSequenceClassification: {e}")
        return False


def test_token_classification(model_path: str):
    """Test loading pruning model with AutoModelForTokenClassification."""
    logger.info("\n" + "="*60)
    logger.info("Testing AutoModelForTokenClassification")
    logger.info("="*60)
    
    try:
        # Load model using AutoModel
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test inference
        query = "機械学習について"
        document = "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。"
        
        inputs = tokenizer(query, document, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Get probabilities for keeping tokens (class 1)
            probs = torch.softmax(logits, dim=-1)
            keep_probs = probs[:, :, 1]
        
        # Apply threshold
        threshold = 0.5
        keep_mask = keep_probs > threshold
        num_kept = keep_mask.sum().item()
        total_tokens = keep_mask.numel()
        
        logger.info(f"Query: {query}")
        logger.info(f"Document: {document}")
        logger.info(f"Tokens kept: {num_kept}/{total_tokens} ({num_kept/total_tokens*100:.1f}%)")
        logger.info("✓ Successfully loaded and ran inference with AutoModelForTokenClassification")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load with AutoModelForTokenClassification: {e}")
        return False


def test_original_api(model_path: str):
    """Test loading with original PruningEncoder API."""
    logger.info("\n" + "="*60)
    logger.info("Testing Original PruningEncoder API")
    logger.info("="*60)
    
    try:
        model = PruningEncoder.from_pretrained(model_path)
        
        # Test inference
        query = "機械学習について"
        document = "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。"
        
        outputs = model.predict_with_pruning(
            [(query, document)],
            pruning_threshold=0.5,
            return_documents=True
        )
        
        logger.info(f"Query: {query}")
        logger.info(f"Document: {document}")
        
        if model.mode == "reranking_pruning":
            logger.info(f"Ranking score: {outputs[0].ranking_scores:.4f}")
        
        if outputs[0].compression_ratio is not None:
            logger.info(f"Compression ratio: {outputs[0].compression_ratio:.2%}")
        
        logger.info("✓ Original PruningEncoder API still works")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed with original API: {e}")
        return False


def main():
    # Models to test
    models_to_test = {
        "pruning_only_minimal": {
            "path": "./output/transformers_compat_test/pruning_only_20250709_135222/final_model",
            "type": "token_classification"
        },
        "reranking_pruning_minimal": {
            "path": "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model",
            "type": "sequence_classification"
        }
    }
    
    results = {}
    
    for model_name, model_info in models_to_test.items():
        logger.info(f"\n{'#'*60}")
        logger.info(f"Testing {model_name}")
        logger.info(f"{'#'*60}")
        
        model_path = model_info["path"]
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}")
            continue
        
        # Test based on model type
        if model_info["type"] == "sequence_classification":
            auto_success = test_sequence_classification(model_path)
        else:
            auto_success = test_token_classification(model_path)
        
        # Test original API
        original_success = test_original_api(model_path)
        
        results[model_name] = {
            "auto_model": auto_success,
            "original_api": original_success
        }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  AutoModel API: {'✓ PASS' if result['auto_model'] else '✗ FAIL'}")
        logger.info(f"  Original API: {'✓ PASS' if result['original_api'] else '✗ FAIL'}")


if __name__ == "__main__":
    main()
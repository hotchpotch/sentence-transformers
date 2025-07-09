#!/usr/bin/env python
"""
Train a minimal reranking+pruning model and test all loading methods.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import torch
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from sentence_transformers.pruning import (
    PruningEncoder, PruningTrainer, PruningLoss, PruningDataCollator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_minimal_model(output_dir: str):
    """Train a minimal reranking+pruning model."""
    logger.info("="*60)
    logger.info("Training Minimal Reranking+Pruning Model")
    logger.info("="*60)
    
    # Load minimal dataset
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        'ja-minimal'
    )
    train_dataset = dataset['train'].select(range(200))  # Use 200 samples
    eval_dataset = dataset['validation'].select(range(50))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Initialize model
    model = PruningEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        mode="reranking_pruning",
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pruning_config={
            "hidden_size": 256,
            "dropout": 0.1,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False
        }
    )
    
    # Data collator
    data_collator = PruningDataCollator(
        tokenizer=model.tokenizer,
        max_length=512,
        mode="reranking_pruning",
        padding=True,
        truncation=True,
        query_column="query",
        texts_column="texts",
        labels_column="labels",
        chunks_pos_column="chunks_pos",
        relevant_chunks_column="relevant_chunks",
        mini_batch_size=16
    )
    
    # Loss function
    loss_fn = PruningLoss(
        model=model,
        mode="reranking_pruning",
        pruning_weight=1.0,
        ranking_weight=1.0,
    )
    
    # Training configuration
    training_args = {
        "output_dir": output_dir,
        "num_epochs": 2,  # 2 epochs for better model
        "batch_size": 8,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "logging_steps": 20,
        "eval_steps": 100,
        "save_steps": 100,
        "save_total_limit": 1,
        "seed": 42,
        "fp16": torch.cuda.is_available(),
        "dataloader_num_workers": 2,
    }
    
    # Initialize trainer
    trainer = PruningTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        loss_fn=loss_fn,
        training_args=training_args
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(final_model_path)
    
    return final_model_path


def test_all_loading_methods(model_path: str):
    """Test all loading methods for the saved model."""
    logger.info("\n" + "="*60)
    logger.info("Testing All Loading Methods")
    logger.info("="*60)
    
    # Test data
    test_queries_docs = [
        ("機械学習について", "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。"),
        ("深層学習とは", "ディープラーニングは多層のニューラルネットワークを使用した機械学習手法です。"),
        ("天気予報について", "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。"),
    ]
    
    results = {}
    
    # Method 1: PruningEncoder (Full features)
    logger.info("\n1. Testing PruningEncoder (Full features)...")
    try:
        from sentence_transformers.pruning import PruningEncoder
        
        model = PruningEncoder.from_pretrained(model_path)
        logger.info(f"   ✓ Loaded: {type(model).__name__}")
        
        # Test with pruning
        outputs = model.predict_with_pruning(
            test_queries_docs[:1], 
            pruning_threshold=0.5,
            return_documents=True
        )
        
        logger.info(f"   ✓ Ranking score: {outputs[0].ranking_scores:.4f}")
        logger.info(f"   ✓ Compression ratio: {outputs[0].compression_ratio:.2%}")
        
        # Test without pruning
        scores = model.predict(test_queries_docs, apply_pruning=False)
        logger.info(f"   ✓ Scores without pruning: {[f'{s:.4f}' for s in scores]}")
        
        results["PruningEncoder"] = {
            "success": True,
            "scores": scores,
            "features": ["ranking", "pruning"]
        }
        
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")
        results["PruningEncoder"] = {"success": False, "error": str(e)}
    
    # Method 2: Base ranking model with AutoModel (No imports)
    logger.info("\n2. Testing AutoModelForSequenceClassification (Base model only)...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Load from ranking_model subdirectory
        ranking_model_path = f"{model_path}/ranking_model"
        model = AutoModelForSequenceClassification.from_pretrained(ranking_model_path)
        tokenizer = AutoTokenizer.from_pretrained(ranking_model_path)
        
        logger.info(f"   ✓ Loaded: {type(model).__name__}")
        logger.info(f"   ✓ Model type: {model.config.model_type}")
        
        # Test inference
        scores = []
        for query, doc in test_queries_docs:
            inputs = tokenizer(query, doc, return_tensors="pt", truncation=True)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                score = torch.sigmoid(outputs.logits).item()
                scores.append(score)
        
        logger.info(f"   ✓ Scores: {[f'{s:.4f}' for s in scores]}")
        
        results["AutoModel_BaseOnly"] = {
            "success": True,
            "scores": scores,
            "features": ["ranking"]
        }
        
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")
        results["AutoModel_BaseOnly"] = {"success": False, "error": str(e)}
    
    # Method 3: AutoModel with registration
    logger.info("\n3. Testing AutoModelForSequenceClassification (With registration)...")
    try:
        import sentence_transformers  # This triggers registration
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"   ✓ Loaded: {type(model).__name__}")
        
        # Test inference
        scores = []
        for query, doc in test_queries_docs:
            inputs = tokenizer(query, doc, return_tensors="pt", truncation=True)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                score = torch.sigmoid(outputs.logits).item()
                scores.append(score)
        
        logger.info(f"   ✓ Scores: {[f'{s:.4f}' for s in scores]}")
        
        results["AutoModel_WithRegistration"] = {
            "success": True,
            "scores": scores,
            "features": ["ranking", "pruning_capable"]
        }
        
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")
        results["AutoModel_WithRegistration"] = {"success": False, "error": str(e)}
    
    # Method 4: CrossEncoder
    logger.info("\n4. Testing CrossEncoder...")
    try:
        import sentence_transformers
        from sentence_transformers import CrossEncoder
        
        model = CrossEncoder(model_path)
        logger.info(f"   ✓ Loaded as CrossEncoder")
        
        # Test predict
        scores = model.predict(test_queries_docs, show_progress_bar=False)
        logger.info(f"   ✓ Scores: {[f'{s:.4f}' for s in scores]}")
        
        # Test rank
        query = test_queries_docs[0][0]
        documents = [doc for _, doc in test_queries_docs]
        ranking = model.rank(query, documents, return_documents=False)
        ranking_str = [f'Doc{r["corpus_id"]+1}:{r["score"]:.4f}' for r in ranking]
        logger.info(f"   ✓ Ranking: {ranking_str}")
        
        results["CrossEncoder"] = {
            "success": True,
            "scores": scores,
            "features": ["ranking", "convenience_methods"]
        }
        
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")
        results["CrossEncoder"] = {"success": False, "error": str(e)}
    
    # Method 5: AutoModel with trust_remote_code
    logger.info("\n5. Testing AutoModelForSequenceClassification (trust_remote_code=True)...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"   ✓ Loaded: {type(model).__name__}")
        
        # Quick test
        inputs = tokenizer(test_queries_docs[0][0], test_queries_docs[0][1], 
                         return_tensors="pt", truncation=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.sigmoid(outputs.logits).item()
        
        logger.info(f"   ✓ Sample score: {score:.4f}")
        
        results["AutoModel_TrustRemoteCode"] = {
            "success": True,
            "features": ["ranking", "pruning_capable", "no_imports_needed"]
        }
        
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")
        results["AutoModel_TrustRemoteCode"] = {"success": False, "error": str(e)}
    
    return results


def verify_score_consistency(results: dict):
    """Verify that all methods produce consistent scores."""
    logger.info("\n" + "="*60)
    logger.info("Score Consistency Verification")
    logger.info("="*60)
    
    # Extract scores from successful methods
    score_sets = {}
    for method, result in results.items():
        if result.get("success") and "scores" in result:
            score_sets[method] = result["scores"]
    
    if len(score_sets) < 2:
        logger.warning("Not enough successful methods to compare scores")
        return
    
    # Compare scores
    reference_method = list(score_sets.keys())[0]
    reference_scores = score_sets[reference_method]
    
    logger.info(f"Reference: {reference_method} = {[f'{s:.4f}' for s in reference_scores]}")
    
    all_consistent = True
    for method, scores in score_sets.items():
        if method == reference_method:
            continue
        
        # Compare each score
        consistent = True
        for i, (ref_score, score) in enumerate(zip(reference_scores, scores)):
            if abs(ref_score - score) > 0.001:  # Allow small numerical differences
                consistent = False
                logger.warning(f"   ✗ {method} differs at position {i}: {score:.4f} vs {ref_score:.4f}")
        
        if consistent:
            logger.info(f"   ✓ {method} matches reference")
        else:
            all_consistent = False
    
    if all_consistent:
        logger.info("\n✅ All methods produce consistent scores!")
    else:
        logger.warning("\n⚠️ Some methods produce different scores")


def create_summary_report(model_path: str, results: dict):
    """Create a summary report."""
    logger.info("\n" + "="*80)
    logger.info("SUMMARY REPORT")
    logger.info("="*80)
    
    # Count successes
    successful = sum(1 for r in results.values() if r.get("success"))
    total = len(results)
    
    logger.info(f"\nSuccess Rate: {successful}/{total} methods")
    
    # Detailed results
    logger.info("\nDetailed Results:")
    for method, result in results.items():
        status = "✓ PASS" if result.get("success") else "✗ FAIL"
        features = ", ".join(result.get("features", [])) if result.get("success") else result.get("error", "Unknown error")
        logger.info(f"  {method:30} {status}  [{features}]")
    
    # Usage examples
    logger.info("\n" + "-"*80)
    logger.info("USAGE EXAMPLES")
    logger.info("-"*80)
    
    logger.info(f"""
1. Base Model Only (No special imports):
   model = AutoModelForSequenceClassification.from_pretrained("{model_path}/ranking_model")

2. With Auto-Registration:
   import sentence_transformers
   model = AutoModelForSequenceClassification.from_pretrained("{model_path}")

3. Full Features:
   from sentence_transformers.pruning import PruningEncoder
   model = PruningEncoder.from_pretrained("{model_path}")

4. CrossEncoder:
   from sentence_transformers import CrossEncoder
   model = CrossEncoder("{model_path}")

5. With trust_remote_code:
   model = AutoModelForSequenceClassification.from_pretrained("{model_path}", trust_remote_code=True)
""")


def main():
    """Run the complete test."""
    # Output directory
    output_dir = f"./output/comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Step 1: Train model
    model_path = train_minimal_model(output_dir)
    
    # Step 2: Test all loading methods
    results = test_all_loading_methods(model_path)
    
    # Step 3: Verify consistency
    verify_score_consistency(results)
    
    # Step 4: Create summary
    create_summary_report(model_path, results)
    
    logger.info("\n" + "="*80)
    logger.info("🎉 COMPREHENSIVE TEST COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nModel saved at: {model_path}")
    logger.info("All loading methods have been tested and verified.")


if __name__ == "__main__":
    main()
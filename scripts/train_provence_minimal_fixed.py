#!/usr/bin/env python3
"""
Train Provence model on minimal dataset using CrossEncoderTrainer.
Uses multilingual model and correct evaluator format.
"""

import os
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import ProvenceLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.training_args import BatchSamplers


def prepare_evaluation_samples(dataset):
    """Prepare samples for CrossEncoderRerankingEvaluator."""
    # Group examples by query
    query_groups = {}
    for example in dataset:
        query = example['query']
        if query not in query_groups:
            query_groups[query] = {
                'positives': [],
                'negatives': []
            }
        
        if example['ranking_label'] == 1:
            query_groups[query]['positives'].append(example['text'])
        else:
            query_groups[query]['negatives'].append(example['text'])
    
    # Create samples in the expected format
    samples = []
    for query, docs in query_groups.items():
        if docs['positives'] and docs['negatives']:  # Only include queries with both pos and neg
            samples.append({
                'query': query,
                'positive': docs['positives'],
                'negative': docs['negatives']
            })
    
    return samples


def main():
    # Model configuration - using multilingual model
    model_name = "microsoft/mdeberta-v3-base"  # Multilingual DeBERTa model
    output_dir = "./output/provence-minimal"
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk("tmp/datasets/dev-dataset/minimal")
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Initialize model with Provence functionality
    print(f"\nInitializing model: {model_name}")
    model = CrossEncoder(
        model_name,
        num_labels=1,
        max_length=512,
        enable_pruning=True,
        pruning_head_config={
            "dropout": 0.1,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False
        }
    )
    
    # Create loss function
    loss = ProvenceLoss(
        model=model,
        ranking_weight=1.0,
        pruning_weight=0.5,
        use_teacher_scores=True,
        sentence_level_pruning=True
    )
    
    # Create evaluator for validation
    print("\nPreparing evaluator...")
    val_samples = prepare_evaluation_samples(dataset['validation'])
    print(f"Validation queries for evaluation: {len(val_samples)}")
    
    if val_samples:
        evaluator = CrossEncoderRerankingEvaluator(
            samples=val_samples[:20],  # Use first 20 queries for faster evaluation
            name="validation",
            show_progress_bar=True,
            batch_size=16
        )
    else:
        evaluator = None
        print("Warning: No valid evaluation samples found")
    
    # Training arguments
    # Based on Provence paper recommendations
    batch_size_per_device = 8  # Adjust based on GPU memory
    grad_accumulation_steps = 16  # Effective batch size = 8 * 16 = 128
    
    args = {
        "output_dir": output_dir,
        "num_train_epochs": 1,  # Just 1 epoch for testing
        "per_device_train_batch_size": batch_size_per_device,
        "per_device_eval_batch_size": batch_size_per_device * 2,
        "gradient_accumulation_steps": grad_accumulation_steps,
        "warmup_ratio": 0.1,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "eval_strategy": "steps" if evaluator else "no",
        "eval_steps": 50 if evaluator else None,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "load_best_model_at_end": True if evaluator else False,
        "metric_for_best_model": "validation_ndcg@10" if evaluator else None,
        "greater_is_better": True if evaluator else None,
        "push_to_hub": False,
        "seed": 42,
        "fp16": True,  # Enable mixed precision training
        "batch_sampler": BatchSamplers.NO_DUPLICATES,  # Avoid duplicate queries in batch
    }
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if evaluator else None,
        loss=loss,
        evaluator=evaluator,
    )
    
    # Train the model
    print("\nStarting training...")
    print(f"Effective batch size: {batch_size_per_device * grad_accumulation_steps}")
    print(f"Total training steps: {len(dataset['train']) // (batch_size_per_device * grad_accumulation_steps) * args['num_train_epochs']}")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {output_dir}")
    model.save_pretrained(output_dir)
    
    # Test pruning functionality
    print("\n" + "="*50)
    print("Testing pruning functionality...")
    print("="*50)
    
    # Find a sample with multiple sentences
    test_sample = None
    for example in dataset['test']:
        if len(example['sentence_boundaries']) > 2:
            test_sample = example
            break
    
    if test_sample:
        sample_query = test_sample['query']
        sample_text = test_sample['text']
        
        # Test basic pruning
        result = model.prune(
            sample_query, 
            sample_text, 
            threshold=0.5,
            return_sentences=True
        )
        
        print(f"\nQuery: {sample_query}")
        print(f"\nOriginal text length: {len(sample_text)} characters")
        print(f"Pruned text length: {len(result['pruned_document'])} characters")
        print(f"Compression ratio: {result['compression_ratio']:.2%}")
        print(f"Number of sentences: {len(result['sentences'])}")
        print(f"Number of sentences pruned: {result['num_pruned_sentences']}")
        
        # Show detailed pruning results
        print("\nSentences and pruning decisions:")
        for i, (sent, mask) in enumerate(zip(result['sentences'], result['pruning_masks'])):
            status = "KEEP" if mask else "PRUNE"
            print(f"  [{status}] {sent[:100]}...")  # Show first 100 chars
    
    # Test predict_with_pruning
    print("\n" + "="*50)
    print("Testing predict_with_pruning...")
    print("="*50)
    
    test_pairs = [
        (dataset['test'][0]['query'], dataset['test'][0]['text']),
        (dataset['test'][1]['query'], dataset['test'][1]['text'])
    ]
    
    outputs = model.predict_with_pruning(
        test_pairs,
        return_documents=True,
        pruning_threshold=0.5
    )
    
    for i, output in enumerate(outputs):
        print(f"\nExample {i+1}:")
        print(f"  Ranking score: {output.ranking_scores[0]:.4f}")
        print(f"  Compression ratio: {output.compression_ratio:.2%}")
        print(f"  Sentences kept: {sum(output.pruning_masks[0])}/{len(output.pruning_masks[0])}")
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {output_dir}")
    
    # Update .gitignore if needed
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            content = f.read()
        
        if "output/" not in content:
            print("\nAdding output/ to .gitignore...")
            with open(gitignore_path, 'a') as f:
                f.write("\n# Provence training outputs\noutput/\n")


if __name__ == "__main__":
    main()
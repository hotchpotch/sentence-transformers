#!/usr/bin/env python3
"""
Train Provence model on minimal dataset using CrossEncoderTrainer.
Uses multilingual model to avoid Japanese-specific dependencies.
"""

import os
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import ProvenceLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.training_args import BatchSamplers


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
    val_queries = []
    val_documents = []
    val_labels = []
    
    # Group by query for evaluation
    query_docs = {}
    for example in dataset['validation']:
        query = example['query']
        if query not in query_docs:
            query_docs[query] = []
        query_docs[query].append({
            'text': example['text'],
            'label': example['ranking_label']
        })
    
    # Create evaluation data - limit to first 20 queries for faster evaluation
    for i, (query, docs) in enumerate(query_docs.items()):
        if i >= 20:
            break
        val_queries.append(query)
        val_documents.append([doc['text'] for doc in docs])
        val_labels.append([doc['label'] for doc in docs])
    
    evaluator = CrossEncoderRerankingEvaluator(
        queries=val_queries,
        documents=val_documents,
        labels=val_labels,
        name="validation"
    )
    
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
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "validation_ndcg@10",
        "greater_is_better": True,
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
        eval_dataset=dataset['validation'],
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
    
    sample_query = dataset['test'][0]['query']
    sample_text = dataset['test'][0]['text']
    
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
    print(f"Number of sentences pruned: {result['num_pruned_sentences']}")
    
    # Show detailed pruning results
    print("\nSentences and pruning decisions:")
    for i, (sent, mask) in enumerate(zip(result['sentences'], result['pruning_masks'])):
        status = "KEEP" if mask else "PRUNE"
        print(f"  [{status}] {sent}")
    
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
#!/usr/bin/env python3
"""
Example training script for ProvenceEncoder.
Train a model for query-dependent text pruning on the minimal dataset.
"""

import os
from pathlib import Path
from datasets import load_from_disk

from sentence_transformers.provence import (
    ProvenceEncoder,
    ProvenceTrainer,
    ProvenceLoss,
    ProvenceDataCollator
)


def main():
    # Configuration
    model_name = "microsoft/mdeberta-v3-base"  # Base model
    output_dir = "./output/provence-example"
    dataset_path = "tmp/datasets/dev-dataset/minimal-fixed"
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    
    # Prepare dataset (if needed)
    def prepare_example(example):
        """Ensure correct field names."""
        if 'text' in example:
            example['document'] = example.pop('text')
        if 'ranking_label' in example:
            example['label'] = example.pop('ranking_label')
        return example
    
    train_dataset = dataset['train'].map(prepare_example)
    eval_dataset = dataset['validation'].map(prepare_example)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Initialize ProvenceEncoder
    print(f"\nInitializing ProvenceEncoder with {model_name}")
    encoder = ProvenceEncoder(
        model_name_or_path=model_name,
        num_labels=1,
        max_length=512,
        pruning_config={
            "dropout": 0.1,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False
        }
    )
    
    # Create data collator
    data_collator = ProvenceDataCollator(
        tokenizer=encoder.tokenizer,
        text_chunker=encoder.text_chunker,
        max_length=encoder.max_length,
        sentence_level_pruning=True
    )
    
    # Create loss function
    loss_fn = ProvenceLoss(
        model=encoder,
        ranking_weight=1.0,
        pruning_weight=0.5,
        use_teacher_scores=True,
        sentence_level_pruning=True
    )
    
    # Training arguments
    training_args = {
        "output_dir": output_dir,
        "num_epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 2,
        "logging_steps": 50,
        "eval_steps": 200,
        "save_steps": 200,
        "save_total_limit": 3,
        "fp16": True,  # Enable mixed precision if GPU available
    }
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = ProvenceTrainer(
        model=encoder,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        loss_fn=loss_fn,
        training_args=training_args,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    print(f"\nSaving final model to {final_model_path}")
    encoder.save_pretrained(final_model_path)
    
    # Test the trained model
    print("\n" + "="*50)
    print("Testing trained model...")
    print("="*50)
    
    # Load the saved model
    loaded_encoder = ProvenceEncoder.from_pretrained(final_model_path)
    
    # Example queries and documents
    test_examples = [
        {
            "query": "What is machine learning?",
            "document": "Machine learning is a subset of artificial intelligence. It enables computers to learn from data. The weather today is sunny. Birds can fly in the sky."
        },
        {
            "query": "How does Python work?",
            "document": "Python is an interpreted programming language. It uses dynamic typing. Coffee is a popular beverage. Python code is executed line by line."
        }
    ]
    
    for example in test_examples:
        print(f"\nQuery: {example['query']}")
        print(f"Document: {example['document']}")
        
        # Predict with pruning
        output = loaded_encoder.predict_with_pruning(
            (example['query'], example['document']),
            pruning_threshold=0.5,
            return_documents=True
        )
        
        print(f"Ranking score: {output.ranking_scores:.4f}")
        print(f"Compression ratio: {output.compression_ratio:.2%}")
        print(f"Pruned document: {output.pruned_documents[0]}")
        
        # Detailed results
        result = loaded_encoder.prune(
            example['query'],
            example['document'],
            threshold=0.5,
            return_sentences=True
        )
        
        print("\nSentence-level decisions:")
        for sent, mask in zip(result['sentences'], result['pruning_masks']):
            status = "KEEP" if mask else "PRUNE"
            print(f"  [{status}] {sent}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simple training script for Provence model without custom data collator.
"""

import os
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import ProvenceLoss
from sentence_transformers import CrossEncoderTrainingArguments
from sentence_transformers.training_args import BatchSamplers
import torch


def simple_collate_fn(batch):
    """Simple collate function that bypasses the complex data collator."""
    # Extract queries and documents
    queries = [item['query'] for item in batch]
    documents = [item['document'] for item in batch]
    
    # Create input pairs
    texts = [[q, d] for q, d in zip(queries, documents)]
    
    # Return in the expected format
    return {
        'texts': texts,
        'labels': {
            'ranking_labels': torch.tensor([item['label'] for item in batch], dtype=torch.float32),
            'teacher_scores': torch.tensor([item['teacher_score'] for item in batch], dtype=torch.float32),
            'pruning_labels': [item['pruning_labels'] for item in batch],
            'sentence_boundaries': [item['sentence_boundaries'] for item in batch]
        }
    }


def prepare_dataset(dataset):
    """Prepare dataset with correct field names."""
    def rename_fields(example):
        return {
            'query': example['query'],
            'document': example['text'],
            'label': example['ranking_label'],
            'pruning_labels': example['pruning_labels'],
            'teacher_score': example['teacher_score'],
            'sentence_boundaries': example['sentence_boundaries']
        }
    
    columns_to_remove = ['text', 'ranking_label'] if 'text' in dataset.column_names else []
    return dataset.map(rename_fields, remove_columns=columns_to_remove)


def main():
    # Model configuration
    model_name = "microsoft/mdeberta-v3-base"
    output_dir = "./output/provence-simple"
    
    # Load and prepare dataset
    print("Loading dataset...")
    dataset = load_from_disk("tmp/datasets/dev-dataset/minimal-fixed")
    
    train_dataset = prepare_dataset(dataset['train'])
    val_dataset = prepare_dataset(dataset['validation'])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    print(f"\nInitializing model: {model_name}")
    model = CrossEncoder(
        model_name,
        num_labels=1,
        max_length=512,
        enable_pruning=True,
        pruning_head_config={
            "dropout": 0.1,
            "sentence_pooling": "mean"
        }
    )
    
    # Create a simple wrapper loss that handles the data format
    class SimpleProvenceLoss(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.ranking_loss = torch.nn.BCEWithLogitsLoss()
            
        def forward(self, sentence_features, labels):
            # Get texts from the batch
            texts = sentence_features.get('texts', [])
            
            # Tokenize
            tokenized = model.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(model.device)
            
            # Forward pass through model
            outputs = model.model(**tokenized)
            
            # Compute ranking loss only for now
            ranking_logits = outputs.logits.squeeze(-1)
            ranking_labels = labels['ranking_labels'].to(model.device)
            
            loss = self.ranking_loss(ranking_logits, ranking_labels)
            
            return loss
    
    loss = SimpleProvenceLoss(model)
    
    # Training arguments
    args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=16,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        push_to_hub=False,
        seed=42,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        data_collator=simple_collate_fn,
    )
    
    # Train
    print("\nStarting training...")
    print(f"Total training steps: {len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)}")
    
    try:
        trainer.train()
        
        # Save model
        print(f"\nSaving model to {output_dir}")
        model.save_pretrained(output_dir)
        
        # Test pruning
        print("\nTesting pruning functionality...")
        sample = train_dataset[0]
        result = model.prune(
            sample['query'],
            sample['document'],
            threshold=0.5,
            return_sentences=True
        )
        
        print(f"Query: {sample['query']}")
        print(f"Original length: {len(sample['document'])} chars")
        print(f"Pruned length: {len(result['pruned_document'])} chars")
        print(f"Compression: {result['compression_ratio']:.2%}")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
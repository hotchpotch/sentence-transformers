#!/usr/bin/env python3
"""
Custom training loop for Provence model.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import numpy as np


def prepare_dataset(dataset):
    """Prepare dataset with correct field names."""
    def rename_fields(example):
        return {
            'query': example['query'],
            'document': example['text'],
            'ranking_label': example['ranking_label'],
            'pruning_labels': example['pruning_labels'],
            'teacher_score': example['teacher_score'],
            'sentence_boundaries': example['sentence_boundaries']
        }
    
    return dataset.map(rename_fields)


def custom_collate_fn(batch):
    """Custom collate function for Provence training."""
    # Just return the batch as-is for now
    return batch


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.model.train()
    if hasattr(model, 'pruning_head'):
        model.pruning_head.train()
    
    total_loss = 0
    ranking_criterion = torch.nn.BCEWithLogitsLoss()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()
        
        # Prepare texts
        texts = [[item['query'], item['document']] for item in batch]
        
        # Tokenize
        encoded = model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        # Forward pass
        outputs = model.model(**encoded)
        
        # Compute ranking loss
        ranking_logits = outputs.logits.squeeze(-1)
        ranking_labels = torch.tensor([item['ranking_label'] for item in batch], dtype=torch.float32).to(device)
        
        loss = ranking_criterion(ranking_logits, ranking_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.model.eval()
    if hasattr(model, 'pruning_head'):
        model.pruning_head.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts = [[item['query'], item['document']] for item in batch]
            
            # Get predictions
            scores = model.predict(texts)
            labels = [item['ranking_label'] for item in batch]
            
            all_scores.extend(scores)
            all_labels.extend(labels)
    
    # Calculate metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Simple accuracy at threshold 0.5
    predictions = (all_scores > 0.5).astype(int)
    accuracy = (predictions == all_labels).mean()
    
    return accuracy


def main():
    # Configuration
    model_name = "microsoft/mdeberta-v3-base"
    output_dir = "./output/provence-custom"
    batch_size = 8
    learning_rate = 2e-5
    num_epochs = 1
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk("tmp/datasets/dev-dataset/minimal-fixed")
    
    train_dataset = prepare_dataset(dataset['train'])
    val_dataset = prepare_dataset(dataset['validation'])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2, 
        shuffle=False, 
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    print(f"\nInitializing model: {model_name}")
    model = CrossEncoder(
        model_name,
        num_labels=1,
        max_length=512,
        device=device,
        enable_pruning=True,
        pruning_head_config={
            "dropout": 0.1,
            "sentence_pooling": "mean"
        }
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Average training loss: {avg_loss:.4f}")
        
        # Evaluate
        accuracy = evaluate(model, val_loader, device)
        print(f"Validation accuracy: {accuracy:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    
    # Test pruning
    print("\nTesting pruning functionality...")
    sample = train_dataset[0]
    
    try:
        result = model.prune(
            sample['query'],
            sample['document'],
            threshold=0.5,
            return_sentences=True
        )
        
        print(f"\nQuery: {sample['query']}")
        print(f"Original length: {len(sample['document'])} chars")
        print(f"Pruned length: {len(result['pruned_document'])} chars")
        print(f"Compression: {result['compression_ratio']:.2%}")
        print(f"Sentences kept: {sum(result['pruning_masks'])}/{len(result['pruning_masks'])}")
        
    except Exception as e:
        print(f"Pruning test failed: {e}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
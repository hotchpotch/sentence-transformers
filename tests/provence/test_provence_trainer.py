"""
Unit tests for ProvenceTrainer.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from datasets import Dataset

from sentence_transformers.provence import (
    ProvenceEncoder,
    ProvenceTrainer,
    ProvenceLoss
)
from sentence_transformers.provence.data_collator_chunk_based import ProvenceChunkBasedDataCollator


class TestProvenceTrainer:
    """Test ProvenceTrainer functionality."""
    
    @pytest.fixture
    def small_encoder(self):
        """Create a small ProvenceEncoder for testing."""
        return ProvenceEncoder(
            model_name_or_path="sentence-transformers-testing/stsb-bert-tiny-safetensors",
            num_labels=1,
            max_length=128,
            pruning_config={
                "dropout": 0.1,
                "sentence_pooling": "mean"
            }
        )
    
    @pytest.fixture
    def dummy_dataset(self):
        """Create a dummy dataset for testing."""
        data = []
        for i in range(20):
            data.append({
                'query': f'Query {i}',
                'document': f'Document {i}. This is sentence two. And sentence three.',
                'label': float(i % 2),  # Alternating 0 and 1
                'teacher_score': 0.5 + 0.1 * (i % 5),
                'pruning_labels': [1, 0, 1],  # Keep, prune, keep
                'sentence_boundaries': [[0, 20], [21, 40], [41, 60]]
            })
        
        return Dataset.from_list(data)
    
    def test_trainer_initialization(self, small_encoder, dummy_dataset):
        """Test ProvenceTrainer initialization."""
        trainer = ProvenceTrainer(
            model=small_encoder,
            train_dataset=dummy_dataset,
            training_args={
                "num_epochs": 1,
                "batch_size": 4,
                "logging_steps": 5
            }
        )
        
        assert trainer.model is small_encoder
        assert trainer.train_dataset is dummy_dataset
        assert trainer.training_args["num_epochs"] == 1
        assert trainer.training_args["batch_size"] == 4
        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None
        assert trainer.data_collator is not None
    
    def test_custom_loss_function(self, small_encoder, dummy_dataset):
        """Test trainer with custom loss function."""
        custom_loss = ProvenceLoss(
            model=small_encoder,
            ranking_weight=2.0,
            pruning_weight=1.0
        )
        
        trainer = ProvenceTrainer(
            model=small_encoder,
            train_dataset=dummy_dataset,
            loss_fn=custom_loss
        )
        
        assert trainer.loss_fn is custom_loss
        assert trainer.loss_fn.ranking_weight == 2.0
        assert trainer.loss_fn.pruning_weight == 1.0
    
    def test_data_collator(self, small_encoder, dummy_dataset):
        """Test data collator functionality."""
        trainer = ProvenceTrainer(
            model=small_encoder,
            train_dataset=dummy_dataset
        )
        
        # Test collating a batch
        batch = [dummy_dataset[i] for i in range(4)]
        collated = trainer.data_collator(batch)
        
        assert "sentence_features" in collated
        assert "labels" in collated
        assert isinstance(collated["sentence_features"], list)
        assert isinstance(collated["labels"], dict)
        
        # Check label keys
        assert "ranking_labels" in collated["labels"]
        assert "teacher_scores" in collated["labels"]
        assert "pruning_labels" in collated["labels"]
        assert "sentence_boundaries" in collated["labels"]
    
    def test_training_step(self, small_encoder, dummy_dataset):
        """Test a single training step."""
        trainer = ProvenceTrainer(
            model=small_encoder,
            train_dataset=dummy_dataset,
            training_args={
                "batch_size": 4,
                "fp16": False  # Disable for testing
            }
        )
        
        # Get a batch
        batch = [dummy_dataset[i] for i in range(4)]
        collated_batch = trainer.data_collator(batch)
        
        # Run training step
        loss = trainer._training_step(collated_batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive
    
    def test_train_small_dataset(self, small_encoder, dummy_dataset):
        """Test training on a small dataset."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = ProvenceTrainer(
                model=small_encoder,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset.select(range(5)),  # Small eval set
                training_args={
                    "output_dir": tmp_dir,
                    "num_epochs": 1,
                    "batch_size": 4,
                    "logging_steps": 5,
                    "eval_steps": 10,
                    "save_steps": 10,
                    "fp16": False
                }
            )
            
            # Train
            trainer.train()
            
            # Check that checkpoints were saved
            checkpoints = list(Path(tmp_dir).glob("checkpoint-*"))
            assert len(checkpoints) > 0
            
            # Check that we can load from checkpoint
            checkpoint_path = checkpoints[0]
            loaded_encoder = ProvenceEncoder.from_pretrained(checkpoint_path)
            assert loaded_encoder is not None
    
    def test_evaluation(self, small_encoder, dummy_dataset):
        """Test evaluation functionality."""
        trainer = ProvenceTrainer(
            model=small_encoder,
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset.select(range(5)),
            training_args={
                "batch_size": 2,
                "fp16": False
            }
        )
        
        # Create eval dataloader
        from torch.utils.data import DataLoader
        eval_dataloader = DataLoader(
            trainer.eval_dataset,
            batch_size=2,
            collate_fn=trainer.data_collator
        )
        
        # Run evaluation
        metrics = trainer._evaluate(eval_dataloader)
        
        assert "eval_loss" in metrics
        assert isinstance(metrics["eval_loss"], float)
        assert metrics["eval_loss"] > 0
    
    def test_checkpoint_rotation(self, small_encoder, dummy_dataset):
        """Test checkpoint rotation with save_total_limit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = ProvenceTrainer(
                model=small_encoder,
                train_dataset=dummy_dataset,
                training_args={
                    "output_dir": tmp_dir,
                    "num_epochs": 1,
                    "batch_size": 2,
                    "save_steps": 5,
                    "save_total_limit": 2,
                    "fp16": False
                }
            )
            
            # Manually create some checkpoints
            for i in range(5):
                trainer.global_step = i * 5
                trainer._save_checkpoint()
            
            # Check that only 2 checkpoints remain (plus best if any)
            checkpoints = [p for p in Path(tmp_dir).glob("checkpoint-*") 
                          if not p.name.endswith("-best")]
            assert len(checkpoints) <= 2
    
    def test_gradient_accumulation(self, small_encoder, dummy_dataset):
        """Test gradient accumulation."""
        trainer = ProvenceTrainer(
            model=small_encoder,
            train_dataset=dummy_dataset,
            training_args={
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "fp16": False
            }
        )
        
        # The effective batch size should be batch_size * gradient_accumulation_steps
        assert trainer.training_args["gradient_accumulation_steps"] == 4
    
    def test_mixed_precision_training(self, small_encoder, dummy_dataset):
        """Test mixed precision training if GPU is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        trainer = ProvenceTrainer(
            model=small_encoder,
            train_dataset=dummy_dataset,
            training_args={
                "batch_size": 4,
                "fp16": True
            }
        )
        
        assert trainer.training_args["fp16"] is True
    
    def test_custom_metrics(self, small_encoder, dummy_dataset):
        """Test custom metrics computation."""
        def compute_custom_metrics(model, dataloader):
            """Custom metric function."""
            return {"custom_metric": 0.95}
        
        trainer = ProvenceTrainer(
            model=small_encoder,
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset.select(range(5)),
            compute_metrics=compute_custom_metrics,
            training_args={
                "batch_size": 2,
                "fp16": False
            }
        )
        
        # Create eval dataloader
        from torch.utils.data import DataLoader
        eval_dataloader = DataLoader(
            trainer.eval_dataset,
            batch_size=2,
            collate_fn=trainer.data_collator
        )
        
        # Run evaluation
        metrics = trainer._evaluate(eval_dataloader)
        
        assert "eval_loss" in metrics
        assert "custom_metric" in metrics
        assert metrics["custom_metric"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__])
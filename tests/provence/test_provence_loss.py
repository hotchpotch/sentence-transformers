"""
Unit tests for ProvenceLoss.
"""

from __future__ import annotations

import pytest
import torch

from sentence_transformers.provence import ProvenceEncoder, ProvenceLoss


class TestProvenceLoss:
    """Test ProvenceLoss functionality."""
    
    def test_provence_loss_initialization(self):
        """Test ProvenceLoss initialization."""
        model = ProvenceEncoder(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            num_labels=1,
            max_length=128
        )
        
        loss = ProvenceLoss(
            model,
            ranking_weight=1.0,
            pruning_weight=0.5,
            sentence_level_pruning=True
        )
        
        assert loss.model is model
        assert loss.ranking_weight == 1.0
        assert loss.pruning_weight == 0.5
        assert loss.sentence_level_pruning is True
        assert loss.use_teacher_scores is False
    
    def test_provence_loss_forward_basic(self):
        """Test basic forward pass of ProvenceLoss."""
        model = ProvenceEncoder(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            num_labels=1,
            max_length=128
        )
        model.eval()  # Set to eval mode to avoid dropout issues
        device = model.device
        
        loss = ProvenceLoss(model)
        
        # Create dummy batch
        batch_size = 2
        seq_length = 10
        sentence_features = [{
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)).to(device),
            "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long).to(device)
        }]
        
        labels = {
            "ranking_labels": torch.tensor([1.0, 0.0]).to(device),
            "pruning_labels": torch.randint(0, 2, (batch_size, seq_length)).to(device)  # Match seq_length
        }
        
        # Forward pass
        total_loss = loss(sentence_features, labels)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert total_loss.dim() == 0  # Scalar
    
    def test_provence_loss_with_sentence_boundaries(self):
        """Test ProvenceLoss with sentence-level pruning."""
        model = ProvenceEncoder(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            num_labels=1,
            max_length=128
        )
        model.eval()
        device = model.device
        
        loss = ProvenceLoss(model, sentence_level_pruning=True)
        
        # Create dummy batch with sentence boundaries
        batch_size = 2
        max_sentences = 3
        
        sentence_features = [{
            "input_ids": torch.randint(0, 1000, (batch_size, 20)).to(device),
            "attention_mask": torch.ones(batch_size, 20, dtype=torch.long).to(device)
        }]
        
        labels = {
            "ranking_labels": torch.tensor([1.0, 0.0]).to(device),
            "pruning_labels": torch.randint(0, 2, (batch_size, max_sentences)).to(device),
            "sentence_boundaries": torch.tensor([
                [[1, 6], [7, 12], [13, 18]],  # Sentence boundaries for sample 1
                [[1, 5], [6, 11], [12, 17]]   # Sentence boundaries for sample 2
            ]).to(device)
        }
        
        # Forward pass
        total_loss = loss(sentence_features, labels)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert hasattr(loss, "last_losses")
        assert "ranking_loss" in loss.last_losses
        assert "pruning_loss" in loss.last_losses
    
    def test_provence_loss_with_teacher_scores(self):
        """Test ProvenceLoss with teacher score distillation."""
        model = ProvenceEncoder(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            num_labels=1,
            max_length=128
        )
        model.eval()
        device = model.device
        
        loss = ProvenceLoss(
            model,
            use_teacher_scores=True,
            ranking_weight=2.0,
            pruning_weight=1.0
        )
        
        # Create dummy batch
        batch_size = 2
        seq_length = 10
        sentence_features = [{
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)).to(device),
            "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long).to(device)
        }]
        
        labels = {
            "ranking_labels": torch.tensor([1.0, 0.0]).to(device),
            "teacher_scores": torch.tensor([0.9, 0.1]).to(device),  # Teacher scores for distillation
            "pruning_labels": torch.randint(0, 2, (batch_size, seq_length)).to(device)  # Match seq_length
        }
        
        # Forward pass
        total_loss = loss(sentence_features, labels)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
    
    def test_provence_loss_get_config_dict(self):
        """Test getting configuration dictionary from ProvenceLoss."""
        model = ProvenceEncoder(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            num_labels=1,
            max_length=128
        )
        
        loss = ProvenceLoss(
            model,
            ranking_weight=1.5,
            pruning_weight=0.8,
            use_teacher_scores=True,
            sentence_level_pruning=False
        )
        
        config = loss.get_config_dict()
        
        assert config["ranking_weight"] == 1.5
        assert config["pruning_weight"] == 0.8
        assert config["use_teacher_scores"] is True
        assert config["sentence_level_pruning"] is False
        assert "ranking_loss" in config
        assert "pruning_loss" in config


if __name__ == "__main__":
    pytest.main([__file__])
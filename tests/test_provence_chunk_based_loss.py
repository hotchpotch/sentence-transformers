"""
Tests for ProvenceChunkBasedLoss
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

from sentence_transformers.provence.losses_chunk_based import ProvenceChunkBasedLoss


class TestProvenceChunkBasedLoss:
    """Test cases for the improved ProvenceChunkBasedLoss"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock ProvenceEncoder model"""
        model = Mock()
        
        # Mock forward method
        def mock_forward(input_ids, attention_mask):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            
            # Return mock outputs
            return {
                'ranking_logits': torch.randn(batch_size),  # [batch_size]
                'pruning_logits': torch.randn(batch_size, seq_len, 2),  # [batch_size, seq_len, 2]
                'hidden_states': torch.randn(batch_size, seq_len, 768)
            }
        
        model.forward = MagicMock(side_effect=mock_forward)
        return model
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing"""
        batch_size = 2
        num_pairs = 4  # 2 queries × 2 docs each
        seq_len = 128
        
        # Mock tokenized inputs
        sentence_features = [{
            'input_ids': torch.randint(0, 1000, (num_pairs, seq_len)),
            'attention_mask': torch.ones(num_pairs, seq_len)
        }]
        
        # Mock labels
        labels = {
            'ranking_targets': torch.tensor([[0.9, 0.1], [0.8, 0.2]]),  # [batch_size, max_docs]
            'pruning_labels': torch.randint(0, 2, (num_pairs, seq_len)),  # [num_pairs, seq_len]
            'batch_indices': torch.tensor([0, 0, 1, 1]),  # [num_pairs]
            'doc_indices': torch.tensor([0, 1, 0, 1]),    # [num_pairs]
            'docs_per_query': [2, 2]
        }
        
        return sentence_features, labels
    
    def test_regression_mode(self, mock_model, sample_batch):
        """Test loss computation in regression mode (teacher distillation)"""
        loss_fn = ProvenceChunkBasedLoss(
            model=mock_model,
            ranking_weight=1.0,
            pruning_weight=0.5,
            is_regression=True  # Regression mode
        )
        
        sentence_features, labels = sample_batch
        
        # Compute loss
        loss = loss_fn(sentence_features, labels)
        
        # Check that loss is computed
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.requires_grad
        
        # Verify model was called
        mock_model.forward.assert_called_once()
        
        # Check that MSELoss is used for ranking (default for regression)
        assert isinstance(loss_fn.ranking_loss_fn, nn.MSELoss)
    
    def test_classification_mode(self, mock_model, sample_batch):
        """Test loss computation in classification mode"""
        loss_fn = ProvenceChunkBasedLoss(
            model=mock_model,
            ranking_weight=1.0,
            pruning_weight=0.5,
            is_regression=False  # Classification mode
        )
        
        sentence_features, labels = sample_batch
        
        # Compute loss
        loss = loss_fn(sentence_features, labels)
        
        # Check that loss is computed
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        
        # Check that BCEWithLogitsLoss is used for ranking (default for classification)
        assert isinstance(loss_fn.ranking_loss_fn, nn.BCEWithLogitsLoss)
    
    def test_custom_loss_functions(self, mock_model, sample_batch):
        """Test with custom loss functions"""
        custom_ranking_loss = nn.L1Loss()
        custom_pruning_loss = nn.NLLLoss()
        
        loss_fn = ProvenceChunkBasedLoss(
            model=mock_model,
            ranking_loss_fn=custom_ranking_loss,
            pruning_loss_fn=custom_pruning_loss,
            ranking_weight=2.0,
            pruning_weight=1.5,
            is_regression=True
        )
        
        # Check custom loss functions are used
        assert loss_fn.ranking_loss_fn is custom_ranking_loss
        assert loss_fn.pruning_loss_fn is custom_pruning_loss
        assert loss_fn.ranking_weight == 2.0
        assert loss_fn.pruning_weight == 1.5
    
    def test_single_ranking_targets(self, mock_model, sample_batch):
        """Test that single ranking_targets matrix works correctly"""
        sentence_features, labels = sample_batch
        
        # Remove old separate matrices (these shouldn't exist anymore)
        assert 'teacher_scores' not in labels
        assert 'ranking_labels' not in labels
        
        # Only ranking_targets should exist
        assert 'ranking_targets' in labels
        
        loss_fn = ProvenceChunkBasedLoss(
            model=mock_model,
            is_regression=True
        )
        
        # Should work without errors
        loss = loss_fn(sentence_features, labels)
        assert isinstance(loss, torch.Tensor)
    
    def test_loss_components(self, mock_model, sample_batch):
        """Test that both ranking and pruning losses are computed"""
        loss_fn = ProvenceChunkBasedLoss(
            model=mock_model,
            ranking_weight=1.0,
            pruning_weight=1.0,
            is_regression=True
        )
        
        sentence_features, labels = sample_batch
        
        # Mock the internal loss computation to track calls
        original_ranking_loss = loss_fn._compute_ranking_loss
        original_pruning_loss = loss_fn._compute_pruning_loss
        
        ranking_loss_called = False
        pruning_loss_called = False
        
        def mock_ranking_loss(outputs, labels):
            nonlocal ranking_loss_called
            ranking_loss_called = True
            return original_ranking_loss(outputs, labels)
        
        def mock_pruning_loss(outputs, labels):
            nonlocal pruning_loss_called
            pruning_loss_called = True
            return original_pruning_loss(outputs, labels)
        
        loss_fn._compute_ranking_loss = mock_ranking_loss
        loss_fn._compute_pruning_loss = mock_pruning_loss
        
        # Compute loss
        loss = loss_fn(sentence_features, labels)
        
        # Check both components were computed
        assert ranking_loss_called
        assert pruning_loss_called
    
    def test_padding_values_ignored(self, mock_model):
        """Test that padding values (-100) are ignored in ranking targets"""
        sentence_features = [{
            'input_ids': torch.randint(0, 1000, (2, 128)),
            'attention_mask': torch.ones(2, 128)
        }]
        
        # Create labels with padding
        labels = {
            'ranking_targets': torch.tensor([[0.9, -100], [0.8, 0.2]]),  # -100 is padding
            'pruning_labels': torch.randint(0, 2, (2, 128)),
            'batch_indices': torch.tensor([0, 1]),
            'doc_indices': torch.tensor([0, 0]),
            'docs_per_query': [1, 1]
        }
        
        loss_fn = ProvenceChunkBasedLoss(
            model=mock_model,
            is_regression=True
        )
        
        # Should handle padding without errors
        loss = loss_fn(sentence_features, labels)
        assert isinstance(loss, torch.Tensor)
    
    def test_empty_batch_handling(self, mock_model):
        """Test handling of edge cases like empty batches"""
        sentence_features = [{
            'input_ids': torch.empty(0, 128, dtype=torch.long),
            'attention_mask': torch.empty(0, 128)
        }]
        
        labels = {
            'ranking_targets': torch.empty(0, 0),
            'pruning_labels': torch.empty(0, 128, dtype=torch.long),
            'batch_indices': torch.empty(0, dtype=torch.long),
            'doc_indices': torch.empty(0, dtype=torch.long),
            'docs_per_query': []
        }
        
        # Modify mock model for empty batch
        mock_model.forward = MagicMock(return_value={
            'ranking_logits': torch.empty(0),
            'pruning_logits': torch.empty(0, 128, 2),
            'hidden_states': torch.empty(0, 128, 768)
        })
        
        loss_fn = ProvenceChunkBasedLoss(
            model=mock_model,
            is_regression=True
        )
        
        # Should handle empty batch gracefully
        loss = loss_fn(sentence_features, labels)
        # Empty batch might return 0 loss or handle it differently
        assert isinstance(loss, torch.Tensor)
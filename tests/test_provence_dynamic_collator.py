"""
Unit tests for ProvenceDynamicDataCollator.
"""

import pytest
import torch
from transformers import AutoTokenizer
from sentence_transformers.provence.data_collator_dynamic import ProvenceDynamicDataCollator


class TestProvenceDynamicDataCollator:
    """Test cases for dynamic pruning label generation."""
    
    @pytest.fixture
    def tokenizer(self):
        """Get a tokenizer for testing."""
        return AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    
    @pytest.fixture
    def collator(self, tokenizer):
        """Create a data collator instance."""
        return ProvenceDynamicDataCollator(
            tokenizer=tokenizer,
            max_length=128,
            padding=True,
            truncation=True
        )
    
    def test_basic_functionality(self, collator, tokenizer):
        """Test basic collator functionality."""
        # Create sample data
        features = [
            {
                'query': 'What is machine learning?',
                'texts': [
                    'Machine learning is a subset of AI.',
                    'Python is a programming language.',
                    'Deep learning uses neural networks.'
                ],
                'ranking_labels': [1, 0, 1],  # First and third are relevant
                'teacher_scores': [0.9, 0.1, 0.8]
            },
            {
                'query': 'How to cook pasta?',
                'texts': [
                    'Boil water and add pasta.',
                    'Machine learning is fascinating.'
                ],
                'ranking_labels': [1, 0],  # Only first is relevant
                'teacher_scores': [0.95, 0.05]
            }
        ]
        
        # Collate batch
        batch = collator(features)
        
        # Check output structure
        assert 'sentence_features' in batch
        assert 'labels' in batch
        assert len(batch['sentence_features']) == 1
        
        # Check encoded inputs
        encoded = batch['sentence_features'][0]
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        
        # Check labels
        labels = batch['labels']
        assert 'ranking_labels' in labels
        assert 'teacher_scores' in labels
        assert 'pruning_labels' in labels
        assert 'batch_indices' in labels
        assert 'doc_indices' in labels
        
        # Check shapes
        num_pairs = 5  # 3 + 2 texts
        assert encoded['input_ids'].shape[0] == num_pairs
        assert labels['pruning_labels'].shape[0] == num_pairs
        
        # Check pruning labels are generated
        assert labels['pruning_labels'].dtype == torch.long
        assert labels['pruning_labels'].min() >= 0
        assert labels['pruning_labels'].max() <= 1
    
    def test_pruning_label_generation(self, collator, tokenizer):
        """Test that pruning labels follow the correct rules."""
        features = [{
            'query': 'test query',
            'texts': ['relevant document', 'irrelevant document'],
            'ranking_labels': [1, 0],
            'teacher_scores': [0.9, 0.1]
        }]
        
        batch = collator(features)
        pruning_labels = batch['labels']['pruning_labels']
        
        # First pair (relevant): should have 1s for document tokens
        relevant_labels = pruning_labels[0]
        # Check that not all tokens are 0 (some should be 1 for relevant doc)
        assert relevant_labels.sum() > 0
        
        # Second pair (irrelevant): should have all 0s
        irrelevant_labels = pruning_labels[1]
        # All tokens should be 0 for irrelevant doc
        assert irrelevant_labels.sum() == 0
    
    def test_cls_and_query_tokens_are_zero(self, collator, tokenizer):
        """Test that [CLS] and query tokens are always 0."""
        features = [{
            'query': 'short query',
            'texts': ['relevant document with some content'],
            'ranking_labels': [1],
            'teacher_scores': [0.9]
        }]
        
        batch = collator(features)
        pruning_labels = batch['labels']['pruning_labels'][0]
        token_ids = batch['sentence_features'][0]['input_ids'][0]
        
        # [CLS] token should be 0
        cls_token_id = tokenizer.cls_token_id
        cls_positions = (token_ids == cls_token_id).nonzero(as_tuple=True)[0]
        if len(cls_positions) > 0:
            assert pruning_labels[cls_positions[0]] == 0
        
        # First few tokens (query part) should be 0
        # This is a simple check - in practice would need more sophisticated logic
        assert pruning_labels[:3].sum() == 0  # Assuming query takes at least 3 tokens
    
    def test_document_boundaries(self, collator, tokenizer):
        """Test that document token identification works correctly."""
        query = "What is AI?"
        document = "Artificial Intelligence is a field of computer science."
        
        features = [{
            'query': query,
            'texts': [document],
            'ranking_labels': [1],
            'teacher_scores': [0.9]
        }]
        
        batch = collator(features)
        pruning_labels = batch['labels']['pruning_labels'][0]
        token_ids = batch['sentence_features'][0]['input_ids'][0]
        
        # Find SEP tokens to identify document boundaries
        sep_token_id = tokenizer.sep_token_id
        sep_positions = (token_ids == sep_token_id).nonzero(as_tuple=True)[0]
        
        # XLMRoberta uses format: <s> query </s> </s> document </s>
        if len(sep_positions) >= 3:
            # Document is between second and third SEP
            doc_start = sep_positions[1].item() + 1
            doc_end = sep_positions[2].item()
            
            # Check that document tokens are marked as 1
            doc_labels = pruning_labels[doc_start:doc_end]
            assert doc_labels.sum() > 0  # Should have some 1s
            
            # Check that tokens before document are 0
            before_doc_labels = pruning_labels[:doc_start]
            assert before_doc_labels.sum() == 0
        elif len(sep_positions) >= 2:
            # Fallback for other tokenizer formats
            doc_start = sep_positions[0].item() + 1
            doc_end = sep_positions[1].item()
            
            # Check that document tokens are marked as 1
            doc_labels = pruning_labels[doc_start:doc_end]
            assert doc_labels.sum() > 0  # Should have some 1s
            
            # Check that tokens before document are 0
            before_doc_labels = pruning_labels[:doc_start]
            assert before_doc_labels.sum() == 0
    
    def test_multiple_examples_batch(self, collator):
        """Test collator with multiple examples in a batch."""
        features = [
            {
                'query': f'Query {i}',
                'texts': [f'Text {i}-{j}' for j in range(3)],
                'ranking_labels': [1 if j == 0 else 0 for j in range(3)],
                'teacher_scores': [0.9 if j == 0 else 0.1 for j in range(3)]
            }
            for i in range(4)
        ]
        
        batch = collator(features)
        
        # Check batch processing
        assert batch['labels']['ranking_labels'].shape[0] == 4  # 4 examples
        assert batch['labels']['ranking_labels'].shape[1] == 3  # max 3 texts per query
        
        # Check that we have correct number of pairs
        total_pairs = sum(len(f['texts']) for f in features)
        assert batch['sentence_features'][0]['input_ids'].shape[0] == total_pairs
        assert batch['labels']['pruning_labels'].shape[0] == total_pairs
    
    def test_mini_batch_processing(self, tokenizer):
        """Test collator with mini-batch processing."""
        collator = ProvenceDynamicDataCollator(
            tokenizer=tokenizer,
            max_length=128,
            mini_batch_size=4  # Process in mini-batches of 4
        )
        
        # Create data with more pairs than mini_batch_size
        features = [
            {
                'query': f'Query {i}',
                'texts': [f'Text {i}-{j}' for j in range(3)],
                'ranking_labels': [1, 0, 0],
                'teacher_scores': [0.9, 0.1, 0.1]
            }
            for i in range(3)
        ]
        
        batch = collator(features)
        
        # Should process 9 pairs (3 queries × 3 texts) in mini-batches
        assert batch['sentence_features'][0]['input_ids'].shape[0] == 9
        assert batch['labels']['pruning_labels'].shape[0] == 9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
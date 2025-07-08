"""
Tests for ProvenceChunkBasedDataCollator
"""

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from sentence_transformers.provence import ProvenceChunkBasedDataCollator


class TestProvenceChunkBasedDataCollator:
    """Test cases for the improved ProvenceChunkBasedDataCollator"""
    
    @pytest.fixture
    def tokenizer(self):
        """Get a tokenizer for testing"""
        return AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return {
            "query": ["What is machine learning?", "How does Python work?"],
            "texts": [
                ["Machine learning is a subset of AI.", "It involves training models."],
                ["Python is a programming language.", "It is interpreted."]
            ],
            "labels": [[1, 0], [1, 0]],
            "teacher_scores_model": [[0.95, 0.15], [0.88, 0.22]],
            "chunks_pos": [
                [[[0, 35], [36, 65]], [[0, 30], [31, 50]]],
                [[[0, 33], [34, 52]], [[0, 20], [21, 40]]]
            ],
            "relevant_chunks": [[[0], []], [[0], []]],
            "dataset_name": ["dataset1", "dataset2"],
            "id": ["id1", "id2"]
        }
    
    def test_basic_collation(self, tokenizer, sample_data):
        """Test basic data collation"""
        collator = ProvenceChunkBasedDataCollator(
            tokenizer=tokenizer,
            max_length=128,
            query_column="query",
            texts_column="texts",
            labels_column="labels",
            chunks_pos_column="chunks_pos",
            relevant_chunks_column="relevant_chunks"
        )
        
        # Create dataset
        dataset = Dataset.from_dict(sample_data)
        
        # Collate batch
        batch = collator(dataset)
        
        # Check output structure
        assert "sentence_features" in batch
        assert "labels" in batch
        
        # Check sentence features
        features = batch["sentence_features"][0]
        assert "input_ids" in features
        assert "attention_mask" in features
        
        # Check labels
        labels = batch["labels"]
        assert "ranking_targets" in labels
        assert "pruning_labels" in labels
        assert "batch_indices" in labels
        assert "doc_indices" in labels
        assert "docs_per_query" in labels
        
        # Check shapes
        num_pairs = 4  # 2 queries × 2 texts each
        assert features["input_ids"].shape[0] == num_pairs
        assert labels["ranking_targets"].shape == (2, 2)  # batch_size × max_docs
        assert labels["pruning_labels"].shape[0] == num_pairs
    
    def test_teacher_scores_column(self, tokenizer, sample_data):
        """Test using teacher scores column"""
        collator = ProvenceChunkBasedDataCollator(
            tokenizer=tokenizer,
            max_length=128,
            query_column="query",
            texts_column="texts", 
            labels_column="labels",
            scores_column="teacher_scores_model",  # Use teacher scores
            chunks_pos_column="chunks_pos",
            relevant_chunks_column="relevant_chunks"
        )
        
        dataset = Dataset.from_dict(sample_data)
        batch = collator(dataset)
        
        # Check that ranking_targets contains teacher scores
        ranking_targets = batch["labels"]["ranking_targets"]
        assert torch.allclose(ranking_targets[0, 0], torch.tensor(0.95))
        assert torch.allclose(ranking_targets[0, 1], torch.tensor(0.15))
        assert torch.allclose(ranking_targets[1, 0], torch.tensor(0.88))
        assert torch.allclose(ranking_targets[1, 1], torch.tensor(0.22))
    
    def test_missing_teacher_scores_fallback(self, tokenizer, sample_data):
        """Test fallback when teacher scores column is missing"""
        # Remove teacher scores column
        data_without_scores = {k: v for k, v in sample_data.items() 
                              if k != "teacher_scores_model"}
        
        collator = ProvenceChunkBasedDataCollator(
            tokenizer=tokenizer,
            max_length=128,
            query_column="query",
            texts_column="texts",
            labels_column="labels",
            scores_column="teacher_scores_model",  # This column doesn't exist
            chunks_pos_column="chunks_pos",
            relevant_chunks_column="relevant_chunks"
        )
        
        dataset = Dataset.from_dict(data_without_scores)
        batch = collator(dataset)
        
        # Check that ranking_targets contains labels instead
        ranking_targets = batch["labels"]["ranking_targets"]
        assert ranking_targets[0, 0] == 1.0
        assert ranking_targets[0, 1] == 0.0
        assert ranking_targets[1, 0] == 1.0
        assert ranking_targets[1, 1] == 0.0
    
    def test_missing_required_column_error(self, tokenizer, sample_data):
        """Test error when required column is missing"""
        collator = ProvenceChunkBasedDataCollator(
            tokenizer=tokenizer,
            max_length=128,
            query_column="query",
            texts_column="texts",
            labels_column="labels",
            chunks_pos_column="chunks_pos_missing",  # Wrong column name
            relevant_chunks_column="relevant_chunks"
        )
        
        dataset = Dataset.from_dict(sample_data)
        
        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            collator(dataset)
        
        assert "Missing required columns" in str(exc_info.value)
        assert "chunks_pos_missing" in str(exc_info.value)
    
    def test_list_of_dicts_input(self, tokenizer, sample_data):
        """Test with list of dicts input instead of Dataset"""
        collator = ProvenceChunkBasedDataCollator(
            tokenizer=tokenizer,
            max_length=128,
            query_column="query",
            texts_column="texts",
            labels_column="labels",
            chunks_pos_column="chunks_pos",
            relevant_chunks_column="relevant_chunks"
        )
        
        # Convert to list of dicts
        features_list = []
        for i in range(len(sample_data["query"])):
            features_list.append({
                key: value[i] for key, value in sample_data.items()
            })
        
        # Collate batch
        batch = collator(features_list)
        
        # Check output
        assert "sentence_features" in batch
        assert "labels" in batch
        assert batch["labels"]["docs_per_query"] == [2, 2]
    
    def test_custom_column_names(self, tokenizer):
        """Test with custom column names"""
        # Data with custom column names
        custom_data = {
            "question": ["What is AI?"],
            "documents": [["AI is artificial intelligence."]],
            "relevance": [[1]],
            "scores": [[0.9]],
            "chunk_boundaries": [[[[0, 30]]]],
            "relevant_chunk_ids": [[[]]]
        }
        
        collator = ProvenceChunkBasedDataCollator(
            tokenizer=tokenizer,
            max_length=128,
            query_column="question",
            texts_column="documents",
            labels_column="relevance",
            scores_column="scores",
            chunks_pos_column="chunk_boundaries",
            relevant_chunks_column="relevant_chunk_ids"
        )
        
        dataset = Dataset.from_dict(custom_data)
        batch = collator(dataset)
        
        # Check successful collation
        assert batch["labels"]["ranking_targets"][0, 0] == 0.9  # Using scores
        assert batch["labels"]["docs_per_query"] == [1]
    
    def test_pruning_labels_generation(self, tokenizer):
        """Test pruning labels generation based on relevant chunks"""
        data = {
            "query": ["Test query"],
            "texts": [["This is the first chunk. This is the second chunk."]],
            "labels": [[1]],
            "chunks_pos": [[[[0, 24], [25, 50]]]],  # Two chunks
            "relevant_chunks": [[[1]]]  # Only second chunk is relevant
        }
        
        collator = ProvenceChunkBasedDataCollator(
            tokenizer=tokenizer,
            max_length=128,
            query_column="query",
            texts_column="texts",
            labels_column="labels",
            chunks_pos_column="chunks_pos",
            relevant_chunks_column="relevant_chunks"
        )
        
        dataset = Dataset.from_dict(data)
        batch = collator(dataset)
        
        # Check pruning labels
        pruning_labels = batch["labels"]["pruning_labels"]
        assert pruning_labels.shape[0] == 1  # One pair
        
        # The pruning labels should mark tokens in the second chunk as 1
        # This is a simplified check - actual implementation would need proper testing
        assert pruning_labels.sum() > 0  # Some tokens should be marked as relevant
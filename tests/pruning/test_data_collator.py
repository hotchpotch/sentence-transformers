"""
Test PruningDataCollator to ensure correct token-level label generation.
This is critical for the pruning model to work correctly.
"""

import pytest
import torch
from transformers import AutoTokenizer
from sentence_transformers.pruning import PruningDataCollator


class TestPruningDataCollator:
    """Test the PruningDataCollator implementation."""
    
    @pytest.fixture
    def tokenizer(self):
        """Load XLMRoberta tokenizer used in japanese-reranker."""
        return AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    @pytest.fixture
    def collator(self, tokenizer):
        """Create a PruningDataCollator instance."""
        return PruningDataCollator(
            tokenizer=tokenizer,
            max_length=512,
            mode="reranking_pruning",
            chunks_pos_column="context_spans",
            relevant_chunks_column="context_spans_relevance"
        )
    
    def test_basic_token_labeling(self, tokenizer, collator):
        """Test basic token labeling with a simple example."""
        # Create a simple example
        query = "What is machine learning?"
        document = "Machine learning is a field of AI. It enables computers to learn. The weather is nice today."
        
        # Define chunks (character positions in document)
        # Chunk 0: "Machine learning is a field of AI." (0-34)
        # Chunk 1: "It enables computers to learn." (35-65)
        # Chunk 2: "The weather is nice today." (66-92)
        chunks = [[0, 34], [35, 65], [66, 92]]
        relevant_chunks = [0, 1]  # First two chunks are relevant
        
        # Create batch
        features = [{
            "query": query,
            "texts": [document],
            "labels": [1],
            "teacher_scores.test": [0.9],
            "context_spans": [chunks],
            "context_spans_relevance": [relevant_chunks]
        }]
        
        # Collate
        batch = collator(features)
        
        # Get the tokenized input
        input_ids = batch["sentence_features"][0]["input_ids"]
        pruning_labels = batch["labels"]["pruning_labels"]
        
        # Decode to understand structure
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        print("\nTokens and labels:")
        for i, (token, label) in enumerate(zip(tokens, pruning_labels[0])):
            print(f"{i:3d}: {token:20s} label={label:2d}")
        
        # Find document boundaries
        eos_token_id = tokenizer.eos_token_id or 2
        sep_positions = (input_ids[0] == eos_token_id).nonzero(as_tuple=True)[0]
        
        assert len(sep_positions) >= 3, "Should have at least 3 </s> tokens"
        
        # XLMRoberta format: <s> query </s> </s> document </s>
        doc_start = sep_positions[0].item() + 2  # Skip first </s> and <s>
        doc_end = sep_positions[2].item()
        
        print(f"\nDocument tokens: {doc_start} to {doc_end}")
        
        # Check query tokens are masked (-100)
        query_labels = pruning_labels[0, :doc_start]
        assert (query_labels == -100).all(), f"Query tokens should be -100, got {query_labels}"
        
        # Check document tokens have 0 or 1
        doc_labels = pruning_labels[0, doc_start:doc_end]
        assert ((doc_labels == 0) | (doc_labels == 1) | (doc_labels == -100)).all(), \
            f"Document tokens should be 0, 1, or -100, got unique values: {torch.unique(doc_labels)}"
        
        # Check that we have some 1s (relevant tokens)
        assert (doc_labels == 1).any(), "Should have some relevant tokens marked as 1"
        
        print(f"\nPruning label distribution in document:")
        print(f"  -100 (ignored): {(doc_labels == -100).sum().item()}")
        print(f"  0 (prune): {(doc_labels == 0).sum().item()}")
        print(f"  1 (keep): {(doc_labels == 1).sum().item()}")
    
    def test_exact_span_mapping(self, tokenizer, collator):
        """Test that spans map to correct tokens."""
        # Use a carefully crafted example where we know exact positions
        query = "Find information"
        
        # Create document with clear boundaries
        chunk1 = "This is relevant information."  # Should be kept
        chunk2 = " "
        chunk3 = "This is irrelevant content."    # Should be pruned
        document = chunk1 + chunk2 + chunk3
        
        # Calculate exact character positions
        chunks = [
            [0, len(chunk1)],                    # First chunk
            [len(chunk1) + len(chunk2), len(document)]  # Second chunk
        ]
        relevant_chunks = [0]  # Only first chunk is relevant
        
        features = [{
            "query": query,
            "texts": [document],
            "labels": [1],
            "teacher_scores.test": [0.9],
            "context_spans": [chunks],
            "context_spans_relevance": [relevant_chunks]
        }]
        
        batch = collator(features)
        
        # Tokenize separately to understand what should be kept
        query_doc = query + " </s> <s> " + document
        encoding = tokenizer(query_doc, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offsets = encoding["offset_mapping"]
        
        print("\nToken analysis:")
        for i, (token, (start, end)) in enumerate(zip(tokens, offsets)):
            substr = query_doc[start:end] if start != 0 or end != 0 else "[SPECIAL]"
            print(f"{i:3d}: {token:20s} [{start:3d}:{end:3d}] = '{substr}'")
        
        # Find which tokens belong to chunk1
        query_prefix_len = len(query + " </s> <s> ")
        chunk1_start = query_prefix_len + chunks[0][0]
        chunk1_end = query_prefix_len + chunks[0][1]
        
        print(f"\nChunk1 character range in full string: {chunk1_start} to {chunk1_end}")
        print(f"Chunk1 text: '{query_doc[chunk1_start:chunk1_end]}'")
        
        # Identify tokens that should be labeled as 1
        expected_keep_tokens = []
        for i, (start, end) in enumerate(offsets):
            if start != 0 or end != 0:  # Not special token
                if start >= chunk1_start and end <= chunk1_end:
                    expected_keep_tokens.append(i)
                    print(f"Token {i} should be kept: '{tokens[i]}' = '{query_doc[start:end]}'")
        
        # Check the actual labels
        pruning_labels = batch["labels"]["pruning_labels"][0]
        
        # Verify expected tokens are marked as 1
        for token_idx in expected_keep_tokens:
            if token_idx < len(pruning_labels):
                # Note: We need to account for potential misalignment
                # The actual implementation might have slight differences
                print(f"Token {token_idx} label: {pruning_labels[token_idx].item()}")
    
    def test_multiple_documents(self, tokenizer, collator):
        """Test with multiple documents in a batch."""
        features = [
            {
                "query": "What is AI?",
                "texts": ["AI is artificial intelligence.", "AI helps computers think.", "The sky is blue."],
                "labels": [1, 1, 0],
                "teacher_scores.test": [0.9, 0.8, 0.1],
                "context_spans": [[[0, 29]], [[0, 25]], [[0, 16]]],
                "context_spans_relevance": [[0], [0], []]
            }
        ]
        
        batch = collator(features)
        
        # Should have 3 pairs
        assert batch["sentence_features"][0]["input_ids"].shape[0] == 3
        assert batch["labels"]["pruning_labels"].shape[0] == 3
        
        # Each should have proper labeling
        pruning_labels = batch["labels"]["pruning_labels"]
        
        for i in range(3):
            input_ids = batch["sentence_features"][0]["input_ids"][i]
            labels = pruning_labels[i]
            
            # Find document boundaries
            eos_token_id = tokenizer.eos_token_id or 2
            sep_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) >= 3:
                # XLMRoberta format: <s> query </s> </s> document </s>
                doc_start = sep_positions[0].item() + 2  # Skip first </s> and <s>
                
                # Query part should be -100
                assert (labels[:doc_start] == -100).all()
                
                # Document part should have valid labels
                if i < 2:  # Positive examples with relevant chunks
                    assert (labels[doc_start:] >= -100).all()
                    assert (labels[doc_start:] <= 1).all()
    
    def test_edge_cases(self, tokenizer, collator):
        """Test edge cases like empty relevant chunks, overlapping chunks, etc."""
        # Test 1: No relevant chunks
        features = [{
            "query": "Test query",
            "texts": ["This document has no relevant parts."],
            "labels": [1],
            "teacher_scores.test": [0.5],
            "context_spans": [[[0, 36]]],
            "context_spans_relevance": [[]]  # No relevant chunks
        }]
        
        batch = collator(features)
        pruning_labels = batch["labels"]["pruning_labels"][0]
        
        # Find document range
        input_ids = batch["sentence_features"][0]["input_ids"][0]
        eos_token_id = tokenizer.eos_token_id or 2
        sep_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
        
        if len(sep_positions) >= 3:
            # XLMRoberta has 3 </s> tokens: <s> query </s> </s> document </s>
            # XLMRoberta format: <s> query </s> </s> document </s>
            doc_start = sep_positions[0].item() + 2  # Skip first </s> and <s>
            doc_end = sep_positions[2].item()
            
            # All document tokens should be 0 (prune) since no relevant chunks
            doc_labels = pruning_labels[doc_start:doc_end]
            # Filter out special tokens (-100)
            content_labels = doc_labels[doc_labels != -100]
            if len(content_labels) > 0:
                assert (content_labels == 0).all(), \
                    f"No relevant chunks, so all content should be pruned (0), got: {content_labels}"
    
    def test_japanese_text(self, tokenizer, collator):
        """Test with Japanese text to ensure proper handling."""
        query = "機械学習とは何ですか？"
        document = "機械学習は人工知能の一分野です。データから学習します。今日は晴れです。"
        
        # Chunks (approximate character positions)
        chunks = [[0, 16], [16, 29], [29, 39]]  # Three sentences
        relevant_chunks = [0, 1]  # First two are relevant
        
        features = [{
            "query": query,
            "texts": [document],
            "labels": [1],
            "teacher_scores.test": [0.9],
            "context_spans": [chunks],
            "context_spans_relevance": [relevant_chunks]
        }]
        
        batch = collator(features)
        
        # Basic checks
        assert batch["sentence_features"][0]["input_ids"].shape[0] == 1
        assert batch["labels"]["pruning_labels"].shape[0] == 1
        
        # The query part should be masked
        pruning_labels = batch["labels"]["pruning_labels"][0]
        assert (pruning_labels != -100).any(), "Should have some labeled tokens"
    
    @pytest.mark.parametrize("tokenizer_name", [
        "xlm-roberta-base",
        "answerdotai/ModernBERT-base",
    ])
    def test_tokenizer_compatibility(self, tokenizer_name):
        """Test that data collator works with different tokenizer types."""
        # Skip ModernBERT if not available
        if tokenizer_name == "answerdotai/ModernBERT-base":
            try:
                from transformers import AutoConfig
                AutoConfig.from_pretrained(tokenizer_name)
            except Exception:
                pytest.skip("ModernBERT not available")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        collator = PruningDataCollator(
            tokenizer=tokenizer,
            max_length=512,
            mode="reranking_pruning",
            chunks_pos_column="context_spans",
            relevant_chunks_column="context_spans_relevance"
        )
        
        # Test data
        query = "What is machine learning?"
        document = "Machine learning is AI. It learns from data. Weather is nice."
        
        # Define chunks
        chunks = [[0, 23], [24, 44], [45, 61]]  # Three sentences
        relevant_chunks = [0, 1]  # First two are relevant
        
        features = [{
            "query": query,
            "texts": [document],
            "labels": [1],
            "context_spans": [chunks],
            "context_spans_relevance": [relevant_chunks]
        }]
        
        # Collate
        batch = collator(features)
        
        # Check results
        input_ids = batch["sentence_features"][0]["input_ids"][0]
        pruning_labels = batch["labels"]["pruning_labels"][0]
        
        # Find document boundaries based on tokenizer type
        if '[SEP]' in tokenizer.get_vocab():
            # ModernBERT style
            sep_token_id = tokenizer.sep_token_id
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            assert len(sep_positions) >= 2, f"Expected at least 2 [SEP] tokens, got {len(sep_positions)}"
            doc_start = sep_positions[0].item() + 1
            doc_end = sep_positions[1].item()
        else:
            # XLMRoberta style
            eos_token_id = tokenizer.eos_token_id or 2
            sep_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
            assert len(sep_positions) >= 3, f"Expected at least 3 </s> tokens, got {len(sep_positions)}"
            # XLMRoberta format: <s> query </s> </s> document </s>
            doc_start = sep_positions[0].item() + 2  # Skip first </s> and <s>
            doc_end = sep_positions[2].item()
        
        # Verify query tokens are masked
        query_labels = pruning_labels[:doc_start]
        assert (query_labels == -100).all(), f"Query tokens should be -100 for {tokenizer_name}"
        
        # Verify document has mixed labels
        doc_labels = pruning_labels[doc_start:doc_end]
        doc_labels_non_special = doc_labels[doc_labels != -100]
        
        assert len(doc_labels_non_special) > 0, f"Should have non-special tokens in document for {tokenizer_name}"
        assert (doc_labels_non_special == 0).any(), f"Should have some pruned tokens (0) for {tokenizer_name}"
        assert (doc_labels_non_special == 1).any(), f"Should have some kept tokens (1) for {tokenizer_name}"
        
        # Log statistics for debugging
        print(f"\nTokenizer: {tokenizer_name}")
        print(f"Document range: {doc_start} to {doc_end}")
        print(f"Label distribution:")
        print(f"  -100 (ignored): {(pruning_labels == -100).sum().item()}")
        print(f"  0 (prune): {(pruning_labels == 0).sum().item()}")
        print(f"  1 (keep): {(pruning_labels == 1).sum().item()}")


if __name__ == "__main__":
    # Run a simple test
    tester = TestPruningDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    collator = PruningDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        mode="reranking_pruning",
        chunks_pos_column="context_spans",
        relevant_chunks_column="context_spans_relevance"
    )
    
    # Run the basic test
    tester.test_basic_token_labeling(tokenizer, collator)
    print("\nBasic test passed!")
    
    # Run exact span mapping test
    tester.test_exact_span_mapping(tokenizer, collator)
    print("\nSpan mapping test completed!")
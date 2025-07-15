"""
Test the span position utility functions for accurate token mapping.
"""

import pytest
from transformers import AutoTokenizer
from sentence_transformers.pruning.data_collator import (
    compute_span_token_positions,
    validate_span_tokenization
)


class TestSpanPositionUtils:
    """Test the span position calculation utilities."""
    
    @pytest.fixture(params=["xlm-roberta-base", "bert-base-uncased"])
    def tokenizer(self, request):
        """Test with different tokenizers."""
        return AutoTokenizer.from_pretrained(request.param)
    
    def test_basic_span_positions(self, tokenizer):
        """Test basic span position calculation."""
        query = "What is machine learning?"
        spans = [
            "Machine learning is AI.",
            "It uses algorithms.",
            "The field is growing."
        ]
        
        # Compute positions
        positions = compute_span_token_positions(tokenizer, query, spans)
        
        # Should have 3 positions
        assert len(positions) == 3
        
        # Positions should be non-overlapping and increasing
        for i in range(len(positions) - 1):
            assert positions[i][1] <= positions[i+1][0], \
                f"Spans should not overlap: {positions[i]} and {positions[i+1]}"
        
        # Validate the positions
        is_valid = validate_span_tokenization(tokenizer, query, spans, positions)
        assert is_valid, "Span positions should decode correctly"
    
    def test_japanese_spans(self):
        """Test with Japanese text."""
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        query = "機械学習とは何ですか？"
        spans = [
            "機械学習は人工知能の一分野です。",
            "データから学習します。",
            "今日は晴れです。"
        ]
        
        positions = compute_span_token_positions(tokenizer, query, spans)
        assert len(positions) == 3
        
        # Validate
        is_valid = validate_span_tokenization(tokenizer, query, spans, positions)
        assert is_valid, "Japanese span positions should decode correctly"
    
    def test_empty_spans(self, tokenizer):
        """Test with empty spans list."""
        query = "Test query"
        spans = []
        
        positions = compute_span_token_positions(tokenizer, query, spans)
        assert positions == []
    
    def test_single_span(self, tokenizer):
        """Test with a single span."""
        query = "Find information"
        spans = ["This is the only span of text."]
        
        positions = compute_span_token_positions(tokenizer, query, spans)
        assert len(positions) == 1
        
        # Validate
        is_valid = validate_span_tokenization(tokenizer, query, spans, positions)
        assert is_valid
    
    def test_tokenizer_special_tokens(self):
        """Test that different tokenizer formats are handled correctly."""
        # Test with BERT (uses [CLS] and [SEP])
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        query = "test query"
        spans = ["first span", "second span"]
        
        bert_positions = compute_span_token_positions(bert_tokenizer, query, spans)
        
        # Test with RoBERTa (uses <s> and </s>)
        roberta_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        roberta_positions = compute_span_token_positions(roberta_tokenizer, query, spans)
        
        # Both should produce valid positions
        assert len(bert_positions) == 2
        assert len(roberta_positions) == 2
        
        # Validate both
        assert validate_span_tokenization(bert_tokenizer, query, spans, bert_positions)
        assert validate_span_tokenization(roberta_tokenizer, query, spans, roberta_positions)
    
    def test_long_spans(self, tokenizer):
        """Test with very long spans that might get truncated."""
        query = "Short query"
        # Create a very long span
        long_text = " ".join(["This is a very long sentence."] * 50)
        spans = [long_text[:200], long_text[200:400], long_text[400:]]
        
        positions = compute_span_token_positions(tokenizer, query, spans)
        
        # Should still get positions for all spans
        assert len(positions) == 3
    
    def test_special_characters(self, tokenizer):
        """Test spans with special characters."""
        query = "Find code examples"
        spans = [
            "def hello_world():\n    print('Hello!')",
            "# This is a comment",
            "return x ** 2 + y ** 2"
        ]
        
        positions = compute_span_token_positions(tokenizer, query, spans)
        assert len(positions) == 3
        
        # Should be valid even with special characters
        is_valid = validate_span_tokenization(tokenizer, query, spans, positions)
        assert is_valid
    
    def test_position_correctness(self, tokenizer):
        """Test that positions actually correspond to the correct tokens."""
        query = "What is AI?"
        spans = ["AI is artificial intelligence.", "It helps computers."]
        
        positions = compute_span_token_positions(tokenizer, query, spans)
        
        # Manually verify by encoding and decoding
        full_text = [query, "".join(spans)]
        encoding = tokenizer(
            [full_text],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_offsets_mapping=False,
            return_attention_mask=False
        )
        
        tokens = encoding['input_ids'][0]
        
        # Check first span
        span1_tokens = tokens[positions[0][0]:positions[0][1]]
        decoded1 = tokenizer.decode(span1_tokens, skip_special_tokens=True)
        assert "artificial intelligence" in decoded1.lower()
        
        # Check second span
        span2_tokens = tokens[positions[1][0]:positions[1][1]]
        decoded2 = tokenizer.decode(span2_tokens, skip_special_tokens=True)
        assert "computers" in decoded2.lower()
    
    def test_punctuation_handling(self, tokenizer):
        """Test that punctuation is handled correctly."""
        query = "What about this?"
        spans = [
            "First sentence.",
            "Second sentence!",
            "Third sentence?"
        ]
        
        positions = compute_span_token_positions(tokenizer, query, spans)
        assert len(positions) == 3
        
        is_valid = validate_span_tokenization(tokenizer, query, spans, positions)
        assert is_valid


if __name__ == "__main__":
    # Run a simple test
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    query = "What is machine learning?"
    spans = ["Machine learning is AI.", "It uses algorithms."]
    
    positions = compute_span_token_positions(tokenizer, query, spans)
    print(f"Query: {query}")
    print(f"Spans: {spans}")
    print(f"Positions: {positions}")
    
    # Validate
    is_valid = validate_span_tokenization(tokenizer, query, spans, positions)
    print(f"Valid: {is_valid}")
    
    # Show token details
    full_text = [query, "".join(spans)]
    encoding = tokenizer([full_text], add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    
    print("\nToken details:")
    for i, token in enumerate(tokens):
        marker = ""
        for j, (start, end) in enumerate(positions):
            if i >= start and i < end:
                marker = f" <- Span {j}"
                break
        print(f"{i:3d}: {token:20s}{marker}")
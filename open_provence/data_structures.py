"""
Data structures for Query-dependent Text Pruning and Reranking.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import torch


@dataclass 
class OpenProvenceOutput:
    """
    Output dataclass for chunk-based pruning predictions.
    
    Attributes:
        ranking_scores: Reranking scores for each document [batch_size]
        chunk_predictions: Chunk-level binary predictions [batch_size, num_chunks]
        chunk_scores: Chunk-level keep probabilities [batch_size, num_chunks]
        token_scores: Token-level keep probabilities [batch_size, seq_len]
        chunk_positions: Original chunk positions [batch_size, num_chunks, 2]
        compression_ratio: Ratio of chunks kept vs total chunks
    """
    ranking_scores: Optional[Union[float, np.ndarray]] = None
    chunk_predictions: Optional[np.ndarray] = None       # [batch_size, num_chunks]
    chunk_scores: Optional[np.ndarray] = None           # [batch_size, num_chunks]
    token_scores: Optional[np.ndarray] = None           # [batch_size, seq_len]
    chunk_positions: Optional[List[List[Tuple[int, int]]]] = None
    compression_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    result[key] = value.tolist()
                else:
                    result[key] = value
        return result


@dataclass
class OpenProvenceOnlyOutput:
    """
    Output dataclass for pruning-only mode (no ranking).
    
    Attributes:
        pruning_masks: Binary masks indicating which tokens to keep [batch_size, seq_len]
        pruning_logits: Raw pruning logits from the model [batch_size, seq_len, 2]
        pruning_probs: Pruning probabilities for each token [batch_size, seq_len, 2]
        sentences: List of tokens for each document
        compression_ratio: Average compression ratio achieved
        num_pruned_tokens: Total number of tokens pruned
        pruned_documents: Pruned documents (if return_documents=True)
    """
    # Pruning outputs
    pruning_masks: Optional[np.ndarray] = None       # [batch_size, seq_len]
    pruning_logits: Optional[torch.Tensor] = None    # [batch_size, seq_len, 2]
    pruning_probs: Optional[np.ndarray] = None       # [batch_size, seq_len, 2]
    
    # Token information
    sentences: Optional[List[List[str]]] = None      # Tokens for each document
    
    # Metadata
    compression_ratio: Optional[float] = None
    num_pruned_tokens: Optional[int] = None
    pruned_documents: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    result[key] = value.tolist()
                else:
                    result[key] = value
        return result


@dataclass
class RerankingOpenProvenceOutput:
    """
    Output dataclass containing both reranking and pruning results.
    
    Attributes:
        ranking_scores: Reranking scores for each document [batch_size]
        ranking_logits: Raw ranking logits from the model [batch_size, 1]
        pruning_masks: Binary masks indicating which sentences to keep [batch_size, max_sentences]
        pruning_logits: Raw pruning logits from the model [batch_size, seq_len, 2]
        pruning_probs: Pruning probabilities for each token/sentence [batch_size, seq_len, 2]
        sentences: List of sentences for each document
        sentence_boundaries: Token boundaries for each sentence [batch_size, max_sentences, 2]
        original_positions: Character positions in original text [batch_size, max_sentences, 2]
        compression_ratio: Average compression ratio achieved
        num_pruned_sentences: Total number of sentences pruned
        pruned_documents: Pruned documents (if return_documents=True)
    """
    # Reranking outputs
    ranking_scores: Optional[np.ndarray] = None      # [batch_size]
    ranking_logits: Optional[torch.Tensor] = None    # [batch_size, 1]
    
    # Pruning outputs  
    pruning_masks: Optional[np.ndarray] = None       # [batch_size, max_sentences]
    pruning_logits: Optional[torch.Tensor] = None    # [batch_size, seq_len, 2]
    pruning_probs: Optional[np.ndarray] = None       # [batch_size, seq_len, 2]
    
    # Chunking information
    sentences: Optional[List[List[str]]] = None      # Sentences for each document
    sentence_boundaries: Optional[List[List[Tuple[int, int]]]] = None  # Token boundaries
    original_positions: Optional[List[List[Tuple[int, int]]]] = None   # Character positions
    
    # Metadata
    compression_ratio: Optional[float] = None
    num_pruned_sentences: Optional[int] = None
    pruned_documents: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    result[key] = value.tolist()
                else:
                    result[key] = value
        return result
    
    def __repr__(self) -> str:
        """String representation of the output."""
        parts = []
        if self.ranking_scores is not None:
            parts.append(f"ranking_scores={self.ranking_scores.shape}")
        if self.pruning_masks is not None:
            parts.append(f"pruning_masks={self.pruning_masks.shape}")
        if self.compression_ratio is not None:
            parts.append(f"compression_ratio={self.compression_ratio:.2f}")
        return f"RerankingOpenProvenceOutput({', '.join(parts)})"


@dataclass 
class OpenProvenceConfig:
    """Configuration for pruning and reranking functionality."""
    # Pruning head configuration
    pruning_hidden_size: Optional[int] = None  # If None, uses model's hidden_size
    pruning_num_labels: int = 2  # Binary: keep/prune
    pruning_dropout: float = 0.1
    
    # Chunking configuration
    chunker_type: str = "multilingual"  # "multilingual", "simple", "custom"
    max_sentences: int = 64
    min_sentence_length: int = 5
    max_sentence_length: int = 500
    
    # Pruning behavior
    pruning_mode: str = "sentence"  # "sentence" or "token"
    default_pruning_threshold: float = 0.5
    min_sentences_to_keep: int = 1
    
    # Performance options
    use_cache: bool = True
    batch_size: int = 32
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return self.__dict__
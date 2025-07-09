"""
CrossEncoder wrapper for PruningEncoder models.
"""

import logging
from typing import List, Tuple, Union
import numpy as np
import torch
from tqdm.autonotebook import trange

from sentence_transformers.cross_encoder import CrossEncoder
from .encoder import PruningEncoder

logger = logging.getLogger(__name__)


class PruningCrossEncoder(CrossEncoder):
    """
    CrossEncoder wrapper for PruningEncoder reranking models.
    
    This allows PruningEncoder models to be used as drop-in replacements for CrossEncoder.
    """
    
    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Initialize PruningCrossEncoder.
        
        Args:
            model_name_or_path: Path to PruningEncoder model
            **kwargs: Additional arguments (ignored for compatibility)
        """
        # Skip parent init to avoid loading with AutoModel
        torch.nn.Module.__init__(self)
        
        # Load PruningEncoder
        self.model = PruningEncoder.from_pretrained(model_name_or_path)
        
        if self.model.mode != "reranking_pruning":
            raise ValueError(
                f"PruningCrossEncoder only supports reranking_pruning mode, "
                f"but model is in {self.model.mode} mode"
            )
        
        # Set attributes for compatibility
        self.tokenizer = self.model.tokenizer
        self.device = self.model.device
        self.max_length = self.model.max_length
        
        # CrossEncoder compatibility attributes
        self._target_device = torch.device(self.device)
        
        logger.info(f"Use pytorch device: {self.device}")
    
    def predict(
        self,
        sentences: List[Tuple[str, str]] | Tuple[str, str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        num_workers: int = 0,
        activation_fn=None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        **kwargs
    ) -> Union[List[float], np.ndarray, torch.Tensor]:
        """
        Predicts the score for the given sentence pairs.
        
        Args:
            sentences: List of (query, document) pairs or single pair
            batch_size: Batch size for predictions
            show_progress_bar: Show progress bar
            num_workers: Number of workers (ignored, for compatibility)
            activation_fn: Activation function (ignored, uses model's default)
            convert_to_numpy: Convert to numpy array
            convert_to_tensor: Convert to tensor
            
        Returns:
            Scores as list, numpy array, or tensor
        """
        # Use PruningEncoder's predict method
        return self.model.predict(
            sentences=sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            apply_pruning=False  # Only get ranking scores
        )
    
    def rank(
        self,
        query: str,
        documents: List[str],
        top_k: int | None = None,
        return_documents: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> List[dict]:
        """
        Ranks the given documents for the query.
        
        Args:
            query: Query string
            documents: List of documents to rank
            top_k: Return only top-k documents
            return_documents: Include documents in results
            batch_size: Batch size
            show_progress_bar: Show progress bar
            
        Returns:
            List of dictionaries with 'corpus_id', 'score', and optionally 'text'
        """
        # Create pairs
        pairs = [(query, doc) for doc in documents]
        
        # Get scores
        scores = self.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        
        # Sort by score
        results = []
        for idx, score in enumerate(scores):
            result = {
                'corpus_id': idx,
                'score': float(score)
            }
            if return_documents:
                result['text'] = documents[idx]
            results.append(result)
        
        # Sort by score descending
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def save_pretrained(self, path: str):
        """Save the model."""
        self.model.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load a pretrained model."""
        return cls(model_name_or_path, **kwargs)
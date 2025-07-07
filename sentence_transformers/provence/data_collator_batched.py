"""
Batched data collator for Provence training that processes multiple texts per query.
Based on the pattern from LambdaLoss.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProvenceBatchedDataCollator:
    """
    Data collator for Provence training that handles multiple texts per query.
    
    This collator:
    1. Processes batches where each example has multiple texts
    2. Creates all query-text pairs for efficient processing
    3. Maintains the structure needed for loss computation
    """
    
    def __init__(self,
                 tokenizer,
                 max_length: int = 512,
                 padding: Union[bool, str] = True,
                 truncation: bool = True,
                 mini_batch_size: Optional[int] = None):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            mini_batch_size: Size for processing mini-batches (if None, process all at once)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.mini_batch_size = mini_batch_size
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch of examples for training.
        
        Args:
            features: List of examples, each containing:
                - 'query': Query text
                - 'texts': List of document texts
                - 'ranking_labels': List of relevance labels (0/1 or float)
                - 'pruning_labels': List of sentence-level pruning labels
                - 'teacher_scores': List of teacher scores for distillation
                - 'sentence_boundaries': List of sentence boundaries
                
        Returns:
            Batch dictionary compatible with ProvenceLoss
        """
        batch_size = len(features)
        
        # Extract queries and texts lists
        queries = [f['query'] for f in features]
        texts_lists = [f['texts'] for f in features]
        
        # Create all query-text pairs
        pairs = []
        batch_indices = []
        doc_indices = []
        
        for batch_idx, (query, texts) in enumerate(zip(queries, texts_lists)):
            for doc_idx, text in enumerate(texts):
                pairs.append([query, text])
                batch_indices.append(batch_idx)
                doc_indices.append(doc_idx)
        
        # Process pairs in mini-batches if specified
        if self.mini_batch_size and len(pairs) > self.mini_batch_size:
            # Process in chunks
            all_encodings = []
            for i in range(0, len(pairs), self.mini_batch_size):
                mini_pairs = pairs[i:i + self.mini_batch_size]
                encoded = self.tokenizer(
                    mini_pairs,
                    padding='max_length',  # Use max_length padding for consistency
                    truncation=self.truncation,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                all_encodings.append(encoded)
            
            # Concatenate all encodings
            encoded_inputs = {
                key: torch.cat([enc[key] for enc in all_encodings], dim=0)
                for key in all_encodings[0].keys()
            }
        else:
            # Process all at once
            encoded_inputs = self.tokenizer(
                pairs,
                padding='max_length' if self.mini_batch_size else self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        # Prepare labels in the format expected by ProvenceLoss
        max_docs = max(len(texts) for texts in texts_lists)
        
        # Ranking labels matrix
        ranking_labels_matrix = torch.full(
            (batch_size, max_docs), 
            fill_value=-100,  # Padding value
            dtype=torch.float32
        )
        
        # Teacher scores matrix
        teacher_scores_matrix = torch.full(
            (batch_size, max_docs),
            fill_value=0.0,
            dtype=torch.float32
        )
        
        # Process labels for each example
        all_pruning_labels = []
        all_sentence_boundaries = []
        
        for i, f in enumerate(features):
            num_docs = len(f['texts'])
            
            # Fill ranking labels
            ranking_labels_matrix[i, :num_docs] = torch.tensor(
                f['ranking_labels'], dtype=torch.float32
            )
            
            # Fill teacher scores
            if 'teacher_scores' in f:
                teacher_scores_matrix[i, :num_docs] = torch.tensor(
                    f['teacher_scores'], dtype=torch.float32
                )
            
            # Collect pruning labels and boundaries
            for doc_idx in range(num_docs):
                all_pruning_labels.append(f['pruning_labels'][doc_idx])
                all_sentence_boundaries.append(f['sentence_boundaries'][doc_idx])
        
        # Create token-level pruning labels for all pairs
        pruning_labels_list = self._create_token_level_labels(
            encoded_inputs,
            all_pruning_labels,
            all_sentence_boundaries,
            pairs
        )
        
        # Prepare output format compatible with ProvenceLoss
        labels = {
            'ranking_labels': ranking_labels_matrix,
            'teacher_scores': teacher_scores_matrix,
            'pruning_labels': pruning_labels_list,
            'sentence_boundaries': all_sentence_boundaries,
            'batch_indices': torch.tensor(batch_indices),
            'doc_indices': torch.tensor(doc_indices),
            'docs_per_query': [len(texts) for texts in texts_lists]
        }
        
        # Return in the format expected by ProvenceLoss
        return {
            'sentence_features': [encoded_inputs],
            'labels': labels
        }
    
    def _create_token_level_labels(self,
                                   encoded_inputs: Dict[str, torch.Tensor],
                                   pruning_labels: List[List[int]],
                                   sentence_boundaries: List[List[List[int]]],
                                   pairs: List[List[str]]) -> torch.Tensor:
        """
        Create token-level pruning labels from sentence-level labels.
        
        Args:
            encoded_inputs: Tokenized inputs
            pruning_labels: Sentence-level labels for each pair
            sentence_boundaries: Character boundaries for each sentence
            pairs: Original text pairs
            
        Returns:
            Token-level pruning labels tensor
        """
        batch_size = encoded_inputs['input_ids'].shape[0]
        seq_length = encoded_inputs['input_ids'].shape[1]
        
        # Initialize with zeros (prune by default)
        token_labels = torch.zeros((batch_size, seq_length), dtype=torch.long)
        
        for idx, (pair, sent_labels, sent_bounds) in enumerate(
            zip(pairs, pruning_labels, sentence_boundaries)
        ):
            # Get the document text
            _, document = pair
            
            # Map sentence labels to token positions
            # This is a simplified version - in production, you'd want more precise mapping
            tokens = self.tokenizer.tokenize(document)
            
            if not tokens or not sent_bounds:
                continue
                
            # Rough mapping: distribute tokens proportionally to character lengths
            doc_length = len(document)
            num_tokens = len(tokens)
            
            for sent_idx, (label, bounds) in enumerate(zip(sent_labels, sent_bounds)):
                if label == 1:  # Keep this sentence
                    start_char, end_char = bounds
                    if start_char < 0 or end_char < 0:
                        continue
                        
                    # Map character positions to token positions (approximate)
                    start_token = int((start_char / doc_length) * num_tokens)
                    end_token = int((end_char / doc_length) * num_tokens) + 1
                    
                    # Account for special tokens
                    start_token += 1  # Skip [CLS] or similar
                    end_token += 1
                    
                    # Set labels to 1 for tokens in this sentence
                    token_labels[idx, start_token:min(end_token, seq_length)] = 1
        
        return token_labels
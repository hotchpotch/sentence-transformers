"""
Dynamic data collator for Provence training that generates pruning labels on-the-fly.
Based on chunk relevance, not pre-computed labels.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProvenceDynamicDataCollator:
    """
    Data collator for Provence training that dynamically generates pruning labels.
    
    Pruning label generation rules:
    - [CLS] token: always 0
    - Query tokens: always 0
    - Relevant chunk tokens (ranking_label=1): all 1
    - Non-relevant chunk tokens (ranking_label=0): all 0
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
                - 'ranking_labels': List of relevance labels (0/1)
                - 'teacher_scores': List of teacher scores for distillation
                - 'chunks': List of chunk boundaries (optional)
                
        Returns:
            Batch dictionary compatible with ProvenceLoss
        """
        batch_size = len(features)
        
        # Extract queries and texts lists
        queries = [f['query'] for f in features]
        texts_lists = [f['texts'] for f in features]
        ranking_labels_lists = [f['ranking_labels'] for f in features]
        
        # Create all query-text pairs
        pairs = []
        batch_indices = []
        doc_indices = []
        pair_ranking_labels = []
        
        for batch_idx, (query, texts, labels) in enumerate(
            zip(queries, texts_lists, ranking_labels_lists)
        ):
            for doc_idx, (text, label) in enumerate(zip(texts, labels)):
                pairs.append([query, text])
                batch_indices.append(batch_idx)
                doc_indices.append(doc_idx)
                pair_ranking_labels.append(label)
        
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
                    return_tensors='pt',
                    return_offsets_mapping=True  # Need this for chunk mapping
                )
                all_encodings.append(encoded)
            
            # Concatenate all encodings
            encoded_inputs = {
                key: torch.cat([enc[key] for enc in all_encodings], dim=0)
                for key in all_encodings[0].keys()
                if key != 'offset_mapping'  # Handle offset_mapping separately
            }
            
            # Concatenate offset mappings
            offset_mappings = torch.cat([enc['offset_mapping'] for enc in all_encodings], dim=0)
        else:
            # Process all at once
            encoded_inputs = self.tokenizer(
                pairs,
                padding='max_length' if self.mini_batch_size else self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors='pt',
                return_offsets_mapping=True
            )
            offset_mappings = encoded_inputs.pop('offset_mapping')
        
        # Generate dynamic pruning labels
        pruning_labels = self._generate_pruning_labels(
            encoded_inputs,
            offset_mappings,
            pairs,
            pair_ranking_labels
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
        
        # Fill matrices
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
        
        # Prepare output format compatible with ProvenceLoss
        labels = {
            'ranking_labels': ranking_labels_matrix,
            'teacher_scores': teacher_scores_matrix,
            'pruning_labels': pruning_labels,
            'batch_indices': torch.tensor(batch_indices),
            'doc_indices': torch.tensor(doc_indices),
            'docs_per_query': [len(texts) for texts in texts_lists]
        }
        
        # Return in the format expected by ProvenceLoss
        return {
            'sentence_features': [encoded_inputs],
            'labels': labels
        }
    
    def _generate_pruning_labels(self,
                                 encoded_inputs: Dict[str, torch.Tensor],
                                 offset_mappings: torch.Tensor,
                                 pairs: List[List[str]],
                                 pair_ranking_labels: List[int]) -> torch.Tensor:
        """
        Generate token-level pruning labels dynamically based on chunk relevance.
        
        Rules:
        - [CLS] token: always 0
        - Query tokens: always 0  
        - Relevant chunk tokens (ranking_label=1): all 1
        - Non-relevant chunk tokens (ranking_label=0): all 0
        
        Args:
            encoded_inputs: Tokenized inputs
            offset_mappings: Character offset mappings for each token
            pairs: Original text pairs [query, document]
            pair_ranking_labels: Relevance label for each pair
            
        Returns:
            Token-level pruning labels tensor
        """
        batch_size = encoded_inputs['input_ids'].shape[0]
        seq_length = encoded_inputs['input_ids'].shape[1]
        
        # Initialize with zeros
        pruning_labels = torch.zeros((batch_size, seq_length), dtype=torch.long)
        
        for idx, (pair, ranking_label, offsets) in enumerate(
            zip(pairs, pair_ranking_labels, offset_mappings)
        ):
            query, document = pair
            
            # Skip if document is not relevant
            if ranking_label == 0:
                continue  # All tokens remain 0
            
            # Find where the document starts in the tokenized sequence
            # For XLMRoberta: <s> query </s> </s> document </s>
            
            # Get token IDs for this example
            token_ids = encoded_inputs['input_ids'][idx]
            
            # Find separator token positions
            sep_token_id = self.tokenizer.sep_token_id
            sep_positions = (token_ids == sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) >= 3:
                # For XLMRoberta format with double </s> between query and document
                # Document starts after second </s> and ends before third </s>
                doc_start = sep_positions[1].item() + 1
                doc_end = sep_positions[2].item()
                
                # Set all document tokens to 1 for relevant documents
                if doc_start < doc_end:
                    pruning_labels[idx, doc_start:doc_end] = 1
            elif len(sep_positions) >= 2:
                # Fallback for other tokenizer formats
                # Document starts after first [SEP] and ends before second [SEP]
                doc_start = sep_positions[0].item() + 1
                doc_end = sep_positions[1].item()
                
                # Set all document tokens to 1 for relevant documents
                if doc_start < doc_end:
                    pruning_labels[idx, doc_start:doc_end] = 1
            else:
                # Fallback: use offset mapping to identify document tokens
                # For pairs, offsets after the query length belong to document
                for token_idx, (start, end) in enumerate(offsets):
                    # Skip special tokens (offset = (0, 0))
                    if start == 0 and end == 0:
                        continue
                    
                    # All non-special tokens in the document part should be marked
                    # This is a simple heuristic - tokens appear after query tokens
                    if token_idx > len(self.tokenizer.encode(query, add_special_tokens=False)) + 2:
                        pruning_labels[idx, token_idx] = 1
        
        return pruning_labels
"""
Chunk-based data collator for Provence training that generates pruning labels based on relevant chunks.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProvenceChunkBasedDataCollator:
    """
    Data collator for Provence training that dynamically generates pruning labels based on chunk relevance.
    
    This collator uses the relevant_chunks information to determine which tokens to keep:
    - Tokens in relevant chunks (marked in relevant_chunks) get label 1
    - All other tokens get label 0
    - [CLS] and query tokens always get label 0
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
                - 'chunks_pos': List of chunk boundaries for each text
                - 'relevant_chunks': List of relevant chunk indices for each text
                
        Returns:
            Batch dictionary compatible with ProvenceLoss
        """
        batch_size = len(features)
        
        # Extract queries and texts lists
        queries = [f['query'] for f in features]
        texts_lists = [f['texts'] for f in features]
        ranking_labels_lists = [f['ranking_labels'] for f in features]
        chunks_pos_lists = [f['chunks_pos'] for f in features]
        relevant_chunks_lists = [f['relevant_chunks'] for f in features]
        
        # Create all query-text pairs
        pairs = []
        batch_indices = []
        doc_indices = []
        pair_ranking_labels = []
        pair_chunks_pos = []
        pair_relevant_chunks = []
        
        for batch_idx, (query, texts, labels, chunks_pos, relevant_chunks) in enumerate(
            zip(queries, texts_lists, ranking_labels_lists, chunks_pos_lists, relevant_chunks_lists)
        ):
            for doc_idx, (text, label, chunk_pos, rel_chunks) in enumerate(
                zip(texts, labels, chunks_pos, relevant_chunks)
            ):
                pairs.append([query, text])
                batch_indices.append(batch_idx)
                doc_indices.append(doc_idx)
                pair_ranking_labels.append(label)
                pair_chunks_pos.append(chunk_pos)
                pair_relevant_chunks.append(rel_chunks)
        
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
        
        # Generate chunk-based pruning labels
        pruning_labels = self._generate_chunk_based_labels(
            encoded_inputs,
            offset_mappings,
            pairs,
            pair_chunks_pos,
            pair_relevant_chunks
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
    
    def _generate_chunk_based_labels(self,
                                     encoded_inputs: Dict[str, torch.Tensor],
                                     offset_mappings: torch.Tensor,
                                     pairs: List[List[str]],
                                     chunks_pos: List[List[List[int]]],
                                     relevant_chunks: List[List[int]]) -> torch.Tensor:
        """
        Generate token-level pruning labels based on chunk relevance.
        
        Rules:
        - [CLS] token: always 0
        - Query tokens: always 0  
        - Tokens in relevant chunks: 1
        - Tokens in non-relevant chunks: 0
        
        Args:
            encoded_inputs: Tokenized inputs
            offset_mappings: Character offset mappings for each token
            pairs: Original text pairs [query, document]
            chunks_pos: Character boundaries for each chunk [[start, end], ...]
            relevant_chunks: Indices of relevant chunks
            
        Returns:
            Token-level pruning labels tensor
        """
        batch_size = encoded_inputs['input_ids'].shape[0]
        seq_length = encoded_inputs['input_ids'].shape[1]
        
        # Initialize with zeros
        pruning_labels = torch.zeros((batch_size, seq_length), dtype=torch.long)
        
        for idx, (pair, chunk_positions, rel_chunk_indices, offsets) in enumerate(
            zip(pairs, chunks_pos, relevant_chunks, offset_mappings)
        ):
            query, document = pair
            
            # Find where the document starts in the tokenized sequence
            # For XLMRoberta: <s> query </s> </s> document </s>
            token_ids = encoded_inputs['input_ids'][idx]
            sep_token_id = self.tokenizer.sep_token_id
            sep_positions = (token_ids == sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) >= 3:
                # Document starts after second </s>
                doc_start_token = sep_positions[1].item() + 1
                doc_end_token = sep_positions[2].item()
                
                # For each token in the document range
                for token_idx in range(doc_start_token, doc_end_token):
                    # Get character position of this token
                    token_start, token_end = offsets[token_idx]
                    
                    # Skip special tokens
                    if token_start == 0 and token_end == 0:
                        continue
                    
                    # Check if this token belongs to any relevant chunk
                    for chunk_idx in rel_chunk_indices:
                        if chunk_idx < len(chunk_positions):
                            chunk_start, chunk_end = chunk_positions[chunk_idx]
                            
                            # Check if token overlaps with this chunk
                            # Note: offsets are relative to the document text, not the full input
                            if token_start < chunk_end and token_end > chunk_start:
                                pruning_labels[idx, token_idx] = 1
                                break
            else:
                # Fallback: simpler approach based on chunk boundaries
                # This is less accurate but ensures some labeling
                if relevant_chunks:
                    # Find document part roughly
                    query_len = len(query)
                    
                    for token_idx, (start, end) in enumerate(offsets):
                        # Skip special tokens
                        if start == 0 and end == 0:
                            continue
                        
                        # If this is likely part of the document
                        if start > query_len:
                            # Check against relevant chunks
                            for chunk_idx in rel_chunk_indices:
                                if chunk_idx < len(chunk_positions):
                                    chunk_start, chunk_end = chunk_positions[chunk_idx]
                                    # Adjust for document offset
                                    doc_chunk_start = chunk_start
                                    doc_chunk_end = chunk_end
                                    
                                    if start < doc_chunk_end and end > doc_chunk_start:
                                        pruning_labels[idx, token_idx] = 1
                                        break
        
        return pruning_labels
"""
Chunk-based data collator for Provence training that generates pruning labels based on relevant chunks.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import numpy as np
from dataclasses import dataclass
import logging
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class ProvenceChunkBasedDataCollator:
    """
    Data collator for Provence training that dynamically generates pruning labels based on chunk relevance.
    
    This collator uses the relevant_chunks information to determine which tokens to keep:
    - Tokens in relevant chunks (marked in relevant_chunks) get label 1
    - All other tokens get label 0
    - [CLS] and query tokens always get label 0
    
    This collator works directly with HuggingFace datasets without requiring conversion.
    """
    
    def __init__(self,
                 tokenizer,
                 max_length: int = 512,
                 padding: Union[bool, str] = True,
                 truncation: bool = True,
                 query_column: str = "query",
                 texts_column: str = "texts",
                 labels_column: str = "labels",
                 scores_column: Optional[str] = None,
                 chunks_pos_column: str = "chunks_pos",
                 relevant_chunks_column: str = "relevant_chunks",
                 dataset_name_column: Optional[str] = "dataset_name",
                 id_column: Optional[str] = "id",
                 mini_batch_size: Optional[int] = None):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            query_column: Name of the query column in the dataset
            texts_column: Name of the texts column in the dataset
            labels_column: Name of the labels column in the dataset
            scores_column: Name of the teacher scores column (if None, use labels_column)
            chunks_pos_column: Name of the chunks positions column
            relevant_chunks_column: Name of the relevant chunks column
            dataset_name_column: Name of the dataset name column (optional)
            id_column: Name of the ID column (optional)
            mini_batch_size: Size for processing mini-batches (if None, process all at once)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.mini_batch_size = mini_batch_size
        
        # Column names
        self.query_column = query_column
        self.texts_column = texts_column
        self.labels_column = labels_column
        self.scores_column = scores_column
        self.chunks_pos_column = chunks_pos_column
        self.relevant_chunks_column = relevant_chunks_column
        self.dataset_name_column = dataset_name_column
        self.id_column = id_column
        
        # Column validation will be done when first batch is processed
        self._validated = False
        
    def _validate_columns(self, dataset_or_batch):
        """Validate that required columns exist in the dataset."""
        if self._validated:
            return
            
        # Get column names from either a Dataset or a batch dict
        if isinstance(dataset_or_batch, Dataset):
            columns = dataset_or_batch.column_names
        elif isinstance(dataset_or_batch, dict):
            columns = dataset_or_batch.keys()
        elif isinstance(dataset_or_batch, list) and len(dataset_or_batch) > 0:
            columns = dataset_or_batch[0].keys()
        else:
            return  # Can't validate yet
        
        # Check required columns
        required_columns = [
            self.query_column,
            self.texts_column,
            self.labels_column,
            self.chunks_pos_column,
            self.relevant_chunks_column
        ]
        
        missing_columns = []
        for col in required_columns:
            if col not in columns:
                missing_columns.append(col)
                
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {list(columns)}"
            )
        
        # Check optional columns
        if self.scores_column and self.scores_column not in columns:
            logger.warning(
                f"Teacher scores column '{self.scores_column}' not found. "
                f"Using '{self.labels_column}' for ranking targets."
            )
            self.scores_column = None
            
        self._validated = True
        
    def __call__(self, features: Union[List[Dict[str, Any]], Dataset]) -> Dict[str, Any]:
        """
        Collate batch of examples for training.
        
        Args:
            features: Either a list of dicts or a Dataset batch containing required columns
                
        Returns:
            Batch dictionary compatible with ProvenceLoss
        """
        # Validate columns on first call
        self._validate_columns(features)
        
        # Handle both list of dicts and Dataset batch
        if isinstance(features, Dataset):
            # Convert Dataset batch to list of dicts
            batch_size = len(features)
            features_list = []
            for i in range(batch_size):
                features_list.append({
                    col: features[col][i] for col in features.column_names
                })
            features = features_list
        
        batch_size = len(features)
        
        # Create all query-text pairs using column names
        pairs = []
        batch_indices = []
        doc_indices = []
        pair_ranking_labels = []
        pair_ranking_targets = []  # Can be labels or teacher scores
        pair_chunks_pos = []
        pair_relevant_chunks = []
        
        for batch_idx, feature in enumerate(features):
            query = feature[self.query_column]
            texts = feature[self.texts_column]
            labels = feature[self.labels_column]
            chunks_pos = feature[self.chunks_pos_column]
            relevant_chunks = feature[self.relevant_chunks_column]
            
            # Get ranking targets (teacher scores or labels)
            if self.scores_column and self.scores_column in feature:
                ranking_targets = feature[self.scores_column]
            else:
                ranking_targets = labels
            
            for doc_idx, (text, label, target, chunk_pos, rel_chunks) in enumerate(
                zip(texts, labels, ranking_targets, chunks_pos, relevant_chunks)
            ):
                pairs.append([query, text])
                batch_indices.append(batch_idx)
                doc_indices.append(doc_idx)
                pair_ranking_labels.append(label)
                pair_ranking_targets.append(target)
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
        max_docs = max(len(feature[self.texts_column]) for feature in features)
        
        # Ranking targets matrix (can be labels or teacher scores)
        ranking_targets_matrix = torch.full(
            (batch_size, max_docs),
            fill_value=-100,  # Padding value
            dtype=torch.float32
        )
        
        # Fill matrix
        for i, feature in enumerate(features):
            texts = feature[self.texts_column]
            num_docs = len(texts)
            
            # Get targets (teacher scores or labels)
            if self.scores_column and self.scores_column in feature:
                targets = feature[self.scores_column]
            else:
                targets = feature[self.labels_column]
            
            # Fill ranking targets
            ranking_targets_matrix[i, :num_docs] = torch.tensor(
                targets, dtype=torch.float32
            )
        
        # Prepare output format compatible with ProvenceLoss
        labels = {
            'ranking_targets': ranking_targets_matrix,  # Single target matrix
            'pruning_labels': pruning_labels,
            'batch_indices': torch.tensor(batch_indices),
            'doc_indices': torch.tensor(doc_indices),
            'docs_per_query': [len(feature[self.texts_column]) for feature in features]
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
            # For XLMRoberta: <s> query </s> <s> document </s>
            token_ids = encoded_inputs['input_ids'][idx]
            
            # Use EOS token ID for XLMRoberta
            eos_token_id = self.tokenizer.eos_token_id or 2  # XLMRoberta uses ID 2 for </s>
            sep_positions = (token_ids == eos_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) >= 2:
                # Document starts after first </s> + <s>
                doc_start_token = sep_positions[0].item() + 2  # Skip </s> and <s>
                doc_end_token = sep_positions[1].item()
                
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
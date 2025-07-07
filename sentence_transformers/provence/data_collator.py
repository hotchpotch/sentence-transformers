"""
Data collators for Provence training.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProvenceDataCollator:
    """
    Data collator for Provence training that prepares batch data with sentence boundaries and labels.
    
    This collator:
    1. Tokenizes query-document pairs
    2. Extracts sentence boundaries for sentence-level pruning
    3. Prepares ranking and pruning labels
    """
    
    def __init__(self,
                 tokenizer,
                 text_chunker,
                 max_length: int = 512,
                 padding: Union[bool, str] = True,
                 truncation: bool = True,
                 sentence_level_pruning: bool = False):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            text_chunker: Text chunker for sentence segmentation
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            sentence_level_pruning: Whether to use sentence-level labels
        """
        self.tokenizer = tokenizer
        self.text_chunker = text_chunker
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.sentence_level_pruning = sentence_level_pruning
        # Add compatibility attributes for CrossEncoderTrainer
        self.valid_label_columns = ["label", "labels", "ranking_label", "ranking_labels", "score", "scores"]
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples for training.
        
        Args:
            features: List of examples, each containing:
                - 'query': Query text
                - 'document': Document text
                - 'label': Relevance label (0/1 or float)
                - 'pruning_labels': Sentence-level pruning labels (list of 0/1)
                - 'teacher_score': Optional teacher score for distillation
                
        Returns:
            Batch dictionary with tokenized inputs and labels
        """
        queries = [f['query'] for f in features]
        documents = [f.get('document', f.get('text', '')) for f in features]
        
        # Prepare input pairs
        texts = [[q, d] for q, d in zip(queries, documents)]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare labels
        labels = {
            'ranking_labels': torch.tensor([f.get('ranking_label', f.get('label', 0)) for f in features], dtype=torch.float32)
        }
        
        # Add teacher scores if available
        if 'teacher_score' in features[0]:
            labels['teacher_scores'] = torch.tensor([f['teacher_score'] for f in features], dtype=torch.float32)
        
        # Process pruning labels and sentence boundaries
        if 'pruning_labels' in features[0]:
            if self.sentence_level_pruning:
                batch_pruning_labels, batch_boundaries = self._process_sentence_level_labels(features, encoded)
            else:
                batch_pruning_labels, batch_boundaries = self._process_token_level_labels(features, encoded)
            
            labels['pruning_labels'] = batch_pruning_labels
            labels['sentence_boundaries'] = batch_boundaries
        
        # Legacy sentence-level processing (kept for compatibility)
        elif self.sentence_level_pruning and 'pruning_labels' in features[0]:
            pass  # Handled above
        
        # Combine encoded inputs and labels
        batch = {
            'sentence_features': [encoded],
            'labels': labels
        }
        
        return batch
    
    def _get_sentence_boundaries(self, 
                                input_ids: torch.Tensor,
                                sentences: List[str],
                                tokenizer) -> List[List[int]]:
        """
        Get token boundaries for each sentence in the tokenized sequence.
        
        Args:
            input_ids: Tokenized input IDs
            sentences: List of sentences
            tokenizer: Tokenizer instance
            
        Returns:
            List of [start, end] token positions for each sentence
        """
        # Decode the full sequence
        full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        boundaries = []
        current_pos = 0
        
        for sentence in sentences:
            # Find sentence in the decoded text
            sentence_start = full_text.find(sentence, current_pos)
            if sentence_start == -1:
                # Try with normalized text
                sentence_normalized = sentence.strip()
                sentence_start = full_text.find(sentence_normalized, current_pos)
            
            if sentence_start != -1:
                sentence_end = sentence_start + len(sentence)
                
                # Convert character positions to token positions
                # This is approximate - for exact boundaries, we'd need char_to_token mapping
                tokens_before = tokenizer.encode(full_text[:sentence_start], add_special_tokens=False)
                tokens_until_end = tokenizer.encode(full_text[:sentence_end], add_special_tokens=False)
                
                start_token = len(tokens_before) + 1  # +1 for [CLS] or similar
                end_token = len(tokens_until_end) + 1
                
                boundaries.append([start_token, end_token])
                current_pos = sentence_end
            else:
                # Sentence not found - use placeholder
                boundaries.append([-1, -1])
        
        return boundaries
    
    def _process_token_level_labels(self, features: List[Dict[str, Any]], encoded: Dict[str, torch.Tensor]) -> tuple:
        """
        Convert sentence-level labels to token-level labels.
        
        Args:
            features: List of examples with sentence-level pruning_labels
            encoded: Tokenized inputs
            
        Returns:
            (token_level_labels, sentence_boundaries) tensors
        """
        batch_size, seq_len = encoded['input_ids'].shape
        batch_token_labels = []
        batch_boundaries = []
        
        for i, feature in enumerate(features):
            # Initialize token labels (0 = prune, 1 = keep)
            token_labels = torch.zeros(seq_len, dtype=torch.long)
            
            # Get sentence boundaries and labels
            if 'sentence_boundaries' in feature:
                char_boundaries = feature['sentence_boundaries']
                sentence_labels = feature.get('pruning_labels', [1] * len(char_boundaries))
            else:
                # Fallback: chunk document and create boundaries
                doc = feature.get('document', feature.get('text', ''))
                chunks_result = self.text_chunker.chunk_text(doc, language="auto")
                sentences = [chunk for chunk, _ in chunks_result]
                
                # Create character boundaries
                char_boundaries = []
                current_pos = 0
                for sentence in sentences:
                    start_pos = doc.find(sentence, current_pos)
                    if start_pos != -1:
                        end_pos = start_pos + len(sentence)
                        char_boundaries.append([start_pos, end_pos])
                        current_pos = end_pos
                    else:
                        char_boundaries.append([current_pos, current_pos])
                
                sentence_labels = feature.get('pruning_labels', [1] * len(sentences))
            
            # Convert character boundaries to token boundaries
            token_boundaries = []
            input_ids = encoded['input_ids'][i]
            
            # Decode tokens to get character mapping
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            # Map character positions to token positions
            query_text = feature['query']
            doc_text = feature.get('document', feature.get('text', ''))
            full_text = f"{query_text} {self.tokenizer.sep_token} {doc_text}"
            
            # Approximate token-to-character mapping
            for sent_idx, (char_start, char_end) in enumerate(char_boundaries):
                # Adjust for query + separator
                adjusted_start = char_start + len(query_text) + len(self.tokenizer.sep_token) + 2
                adjusted_end = char_end + len(query_text) + len(self.tokenizer.sep_token) + 2
                
                # Find approximate token boundaries
                token_start = self._char_to_token_approx(input_ids, adjusted_start, self.tokenizer)
                token_end = self._char_to_token_approx(input_ids, adjusted_end, self.tokenizer)
                
                # Ensure valid boundaries
                token_start = max(1, min(token_start, seq_len - 1))  # Skip [CLS]
                token_end = min(token_end, seq_len - 1)  # Don't exceed sequence
                
                if token_start < token_end:
                    token_boundaries.append([token_start, token_end])
                    
                    # Assign labels to tokens in this sentence
                    if sent_idx < len(sentence_labels):
                        label = sentence_labels[sent_idx]
                        token_labels[token_start:token_end] = label
            
            batch_token_labels.append(token_labels)
            batch_boundaries.append(token_boundaries)
        
        # Convert to tensors
        batch_token_labels = torch.stack(batch_token_labels)
        
        # Pad boundaries to same length
        max_sentences = max(len(boundaries) for boundaries in batch_boundaries) if batch_boundaries else 0
        padded_boundaries = []
        
        for boundaries in batch_boundaries:
            padded = boundaries + [[-1, -1]] * (max_sentences - len(boundaries))
            padded_boundaries.append(padded)
        
        batch_boundaries = torch.tensor(padded_boundaries, dtype=torch.long) if max_sentences > 0 else torch.empty((batch_size, 0, 2), dtype=torch.long)
        
        return batch_token_labels, batch_boundaries
    
    def _process_sentence_level_labels(self, features: List[Dict[str, Any]], encoded: Dict[str, torch.Tensor]) -> tuple:
        """
        Process sentence-level labels (legacy method).
        
        Args:
            features: List of examples with sentence-level pruning_labels
            encoded: Tokenized inputs
            
        Returns:
            (sentence_level_labels, sentence_boundaries) tensors
        """
        batch_pruning_labels = []
        batch_boundaries = []
        max_sentences = 0
        
        for i, feature in enumerate(features):
            # Use existing sentence boundaries if available
            if 'sentence_boundaries' in feature:
                boundaries = feature['sentence_boundaries']
                pruning_labels = feature.get('pruning_labels', [1] * len(boundaries))
            else:
                # Fallback to text chunking
                doc = feature.get('document', feature.get('text', ''))
                chunks_result = self.text_chunker.chunk_text(doc, language="auto")
                sentences = [chunk for chunk, _ in chunks_result]
                
                # Get token boundaries for each sentence
                boundaries = self._get_sentence_boundaries(
                    encoded['input_ids'][i],
                    sentences,
                    self.tokenizer
                )
                pruning_labels = feature.get('pruning_labels', [1] * len(sentences))
            
            # Ensure we have labels for all sentences
            if len(pruning_labels) < len(boundaries):
                pruning_labels.extend([1] * (len(boundaries) - len(pruning_labels)))
            elif len(pruning_labels) > len(boundaries):
                pruning_labels = pruning_labels[:len(boundaries)]
            
            batch_pruning_labels.append(pruning_labels)
            batch_boundaries.append(boundaries)
            max_sentences = max(max_sentences, len(boundaries))
        
        # Pad to max_sentences
        padded_labels = []
        padded_boundaries = []
        
        for pruning_label, boundary in zip(batch_pruning_labels, batch_boundaries):
            # Pad labels with -100 (ignore index)
            padded_label = pruning_label + [-100] * (max_sentences - len(pruning_label))
            padded_labels.append(padded_label)
            
            # Pad boundaries with -1
            padded_boundary = boundary + [[-1, -1]] * (max_sentences - len(boundary))
            padded_boundaries.append(padded_boundary)
        
        return torch.tensor(padded_labels, dtype=torch.long), torch.tensor(padded_boundaries, dtype=torch.long)
    
    def _char_to_token_approx(self, input_ids: torch.Tensor, char_pos: int, tokenizer) -> int:
        """
        Approximate character position to token position mapping.
        
        Args:
            input_ids: Token IDs
            char_pos: Character position
            tokenizer: Tokenizer instance
            
        Returns:
            Approximate token position
        """
        # Simple heuristic: assume average characters per token
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        if len(decoded_text) == 0:
            return 1
        
        # Ratio of characters to tokens
        num_tokens = len(input_ids) - 2  # Exclude special tokens
        chars_per_token = len(decoded_text) / max(num_tokens, 1)
        
        # Estimate token position
        token_pos = int(char_pos / chars_per_token) + 1  # +1 for [CLS]
        
        return max(1, min(token_pos, len(input_ids) - 1))


def create_provence_data_collator(cross_encoder, **kwargs):
    """
    Factory function to create a ProvenceDataCollator for a CrossEncoder.
    
    Args:
        cross_encoder: CrossEncoder instance with enable_pruning=True
        **kwargs: Additional arguments for ProvenceDataCollator
        
    Returns:
        ProvenceDataCollator instance
    """
    if not cross_encoder.enable_pruning:
        raise ValueError("CrossEncoder must have enable_pruning=True")
    
    return ProvenceDataCollator(
        tokenizer=cross_encoder.tokenizer,
        text_chunker=cross_encoder.text_chunker,
        max_length=cross_encoder.max_length,
        **kwargs
    )
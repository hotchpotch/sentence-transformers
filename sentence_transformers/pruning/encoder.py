"""
PruningEncoder: A query-dependent text pruning encoder with reranking capabilities.

This module implements query-dependent text pruning that can work in two modes:
1. Reranking + Pruning mode (current implementation)
2. Pruning-only mode (to be implemented in future versions)
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm

from sentence_transformers.util import import_from_string, fullname
from sentence_transformers.utils.text_chunking import MultilingualChunker
from .data_structures import PruningConfig, RerankingPruningOutput, PruningOutput, PruningOnlyOutput
from .models.pruning_head import PruningHead, PruningHeadConfig

logger = logging.getLogger(__name__)


class PruningEncoder(nn.Module):
    """
    PruningEncoder performs query-dependent text pruning with optional reranking.
    
    This encoder supports two modes:
    1. Reranking + Pruning mode: Ranks query-document pairs and prunes irrelevant content
    2. Pruning-only mode: Only prunes content without reranking
    
    Args:
        model_name_or_path (str): HuggingFace model name or path
        mode (str): Operating mode - "reranking_pruning" or "pruning_only"
        num_labels (int): Number of labels for ranking (default: 1 for regression)
        max_length (int): Maximum sequence length
        device (str): Device to use (cuda/cpu)
        pruning_config (Dict): Configuration for the pruning head
        cache_dir (str): Cache directory for models
        tokenizer_args (Dict): Additional tokenizer arguments
        model_args (Dict): Additional model arguments
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        mode: str = "reranking_pruning",
        num_labels: int = 1,
        max_length: int = 512,
        device: Optional[str] = None,
        pruning_config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Validate mode
        if mode not in ["reranking_pruning", "pruning_only"]:
            raise ValueError(
                f"Invalid mode: {mode}. "
                f"Must be 'reranking_pruning' or 'pruning_only'"
            )
        
        # Initialize config
        self.model_name_or_path = model_name_or_path
        self.mode = mode
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # Default configs
        tokenizer_args = tokenizer_args or {}
        model_args = model_args or {}
        pruning_config = pruning_config or {}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args
        )
        
        # Mode-specific model initialization
        if mode == "reranking_pruning":
            # Load config for ranking
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=num_labels,
                cache_dir=cache_dir,
                **model_args
            )
            
            # Load ranking model
            self.ranking_model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                config=self.config,
                cache_dir=cache_dir,
                **model_args
            )
            self.encoder = None  # Not used in this mode
            
        elif mode == "pruning_only":
            # Load config for encoder
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                **model_args
            )
            
            # Load base encoder (no classification head)
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(
                model_name_or_path,
                config=self.config,
                cache_dir=cache_dir,
                **model_args
            )
            self.ranking_model = None  # Not used in this mode
        
        # Initialize pruning head (common for both modes)
        hidden_size = self.config.hidden_size
        pruning_head_config = PruningHeadConfig(
            hidden_size=pruning_config.get("hidden_size", hidden_size),
            num_labels=2,  # Binary: keep/prune
            classifier_dropout=pruning_config.get("dropout", 0.1),
            sentence_pooling=pruning_config.get("sentence_pooling", "mean"),
            use_weighted_pooling=pruning_config.get("use_weighted_pooling", False)
        )
        self.pruning_head = PruningHead(pruning_head_config)
        
        # Text chunker for sentence segmentation (only needed for raw text API)
        self.text_chunker = None
        
        # Activation function for ranking scores
        if num_labels == 1:
            self.activation_fn = nn.Sigmoid()
        else:
            self.activation_fn = nn.Identity()
        
        # Move to device
        self.to(self.device)
        
        # Default Pruning config
        self.pruning_config = PruningConfig()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_boundaries: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs  # Accept additional kwargs like token_type_ids
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ranking and/or pruning based on mode.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            sentence_boundaries: Token boundaries for each sentence
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with mode-appropriate outputs
        """
        if self.mode == "reranking_pruning":
            # Get outputs from ranking model
            outputs = self.ranking_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Ranking logits
            ranking_logits = outputs.logits
            
            # Get hidden states for pruning
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
        elif self.mode == "pruning_only":
            # Get outputs from encoder only
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            
            # No ranking logits in pruning_only mode
            ranking_logits = None
            
            # Get hidden states for pruning
            hidden_states = outputs.last_hidden_state
        
        # Get pruning predictions (common for both modes)
        pruning_outputs = self.pruning_head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            sentence_boundaries=sentence_boundaries
        )
        
        if return_dict:
            if self.mode == "reranking_pruning":
                return {
                    "ranking_logits": ranking_logits,
                    "pruning_logits": pruning_outputs.logits,
                    "hidden_states": hidden_states
                }
            else:  # pruning_only
                return {
                    "pruning_logits": pruning_outputs.logits,
                    "hidden_states": hidden_states
                }
        
        # Non-dict return
        if self.mode == "reranking_pruning":
            return ranking_logits, pruning_outputs.logits
        else:
            return pruning_outputs.logits
    
    def predict(
        self,
        sentences: List[Tuple[str, str]] | Tuple[str, str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        apply_pruning: bool = True,
        pruning_threshold: float = 0.5,
        return_documents: bool = False,
    ) -> Union[List[float], np.ndarray, torch.Tensor, List[PruningOutput]]:
        """
        Predict ranking scores and optionally apply pruning.
        
        Args:
            sentences: Query-document pairs
            batch_size: Batch size for prediction
            show_progress_bar: Show progress bar
            convert_to_numpy: Convert to numpy array (only for scores)
            convert_to_tensor: Convert to tensor (only for scores)
            apply_pruning: Whether to apply token-level pruning
            pruning_threshold: Threshold for pruning decisions
            return_documents: Whether to return pruned documents
            
        Returns:
            If apply_pruning is False: Ranking scores
            If apply_pruning is True: List of RerankingPruningOutput objects
        """
        # Validate mode
        if self.mode == "pruning_only":
            if not apply_pruning:
                raise ValueError(
                    "predict() returns ranking scores and is not available in pruning_only mode. "
                    "Use prune_texts() or set apply_pruning=True."
                )
        if apply_pruning:
            # Use predict_with_pruning for full functionality
            return self.predict_with_pruning(
                sentences=sentences,
                batch_size=batch_size,
                pruning_threshold=pruning_threshold,
                return_documents=return_documents,
                show_progress_bar=show_progress_bar,
            )
        
        # Original predict behavior for ranking only
        self.eval()
        
        single_input = isinstance(sentences[0], str)
        if single_input:
            sentences = [sentences]
        
        all_scores = []
        
        for start_idx in tqdm(
            range(0, len(sentences), batch_size),
            desc="Batches",
            disable=not show_progress_bar,
        ):
            batch = sentences[start_idx:start_idx + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.forward(**encoded)
                logits = outputs["ranking_logits"]
                
                # Apply activation
                if self.num_labels == 1:
                    scores = self.activation_fn(logits).squeeze(-1)
                else:
                    scores = torch.nn.functional.softmax(logits, dim=-1)
                    scores = scores[:, 1]  # Positive class
                
                all_scores.extend(scores.cpu().tolist())
        
        if single_input:
            if convert_to_tensor:
                return torch.tensor(all_scores)
            elif convert_to_numpy:
                return np.array(all_scores)
            else:
                return all_scores
        else:
            if convert_to_tensor:
                return torch.tensor(all_scores)
            elif convert_to_numpy:
                return np.array(all_scores)
            else:
                return all_scores
    
    def predict_with_pruning(
        self,
        sentences: List[Tuple[str, str]] | Tuple[str, str],
        batch_size: int = 32,
        pruning_threshold: float = 0.5,
        return_documents: bool = False,
        show_progress_bar: bool = False,
    ) -> Union[RerankingPruningOutput, PruningOnlyOutput, List[Union[RerankingPruningOutput, PruningOnlyOutput]]]:
        """
        Predict with token-level pruning.
        
        Args:
            sentences: Query-document pairs
            batch_size: Batch size
            pruning_threshold: Threshold for pruning decisions
            return_documents: Whether to return pruned documents
            show_progress_bar: Show progress bar
            
        Returns:
            RerankingPruningOutput/PruningOnlyOutput or list of them based on mode
        """
        self.eval()
        
        single_input = isinstance(sentences[0], str)
        if single_input:
            sentences = [sentences]
        
        all_outputs = []
        
        for start_idx in tqdm(
            range(0, len(sentences), batch_size),
            desc="Batches",
            disable=not show_progress_bar,
        ):
            batch = sentences[start_idx:start_idx + batch_size]
            
            # Tokenize with offset mapping
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            offset_mapping = encoded['offset_mapping']
            
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get predictions
                if self.mode == "reranking_pruning":
                    ranking_logits = outputs["ranking_logits"]
                    if self.num_labels == 1:
                        ranking_scores = self.activation_fn(ranking_logits).squeeze(-1)
                    else:
                        ranking_scores = torch.nn.functional.softmax(ranking_logits, dim=-1)[:, 1]
                else:
                    # No ranking scores in pruning_only mode
                    ranking_scores = torch.zeros(len(batch), device=self.device)
                
                # Get token-level pruning predictions
                pruning_logits = outputs["pruning_logits"]
                pruning_probs = torch.nn.functional.softmax(pruning_logits, dim=-1)
                keep_probs = pruning_probs[:, :, 1]  # Probability of keeping each token
                
                # Process each example
                for i in range(len(batch)):
                    query, document = batch[i]
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                    offsets = offset_mapping[i]
                    
                    # Find document boundaries using </s> token (eos_token_id)
                    eos_token_id = self.tokenizer.eos_token_id or 2  # XLMRoberta uses ID 2 for </s>
                    sep_positions = (input_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(sep_positions) >= 2:
                        # Format: <s> query </s> <s> document </s>
                        # So document starts after first </s> + <s>
                        doc_start = sep_positions[0].item() + 2  # Skip </s> and <s>
                        doc_end = sep_positions[1].item()
                        
                        # Get document tokens and their keep probabilities
                        doc_keep_probs = keep_probs[i, doc_start:doc_end]
                        doc_tokens = tokens[doc_start:doc_end]
                        doc_offsets = offsets[doc_start:doc_end]
                        
                        # Apply threshold
                        keep_mask = doc_keep_probs > pruning_threshold
                        
                        # Calculate metrics
                        num_kept = keep_mask.sum().item()
                        num_total = len(doc_tokens)
                        compression_ratio = 1.0 - (num_kept / num_total) if num_total > 0 else 0.0
                        
                        # Reconstruct pruned document
                        pruned_doc = ""
                        if return_documents:
                            kept_ranges = []
                            for j, (keep, (start, end)) in enumerate(zip(keep_mask, doc_offsets)):
                                if keep and start != 0:  # Skip special tokens
                                    kept_ranges.append((start.item(), end.item()))
                            
                            # Merge overlapping ranges
                            if kept_ranges:
                                kept_ranges.sort()
                                merged_ranges = [kept_ranges[0]]
                                for start, end in kept_ranges[1:]:
                                    if start <= merged_ranges[-1][1]:
                                        merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
                                    else:
                                        merged_ranges.append((start, end))
                                
                                # Extract text
                                pruned_parts = []
                                for start, end in merged_ranges:
                                    pruned_parts.append(document[start:end])
                                pruned_doc = " ".join(pruned_parts)
                        
                        # Create output based on mode
                        if self.mode == "reranking_pruning":
                            output = RerankingPruningOutput(
                                ranking_scores=ranking_scores[i].cpu().item(),
                                pruning_masks=np.array([keep_mask.cpu().numpy()]),
                                sentences=[doc_tokens],  # Store tokens instead of sentences
                                compression_ratio=compression_ratio,
                                num_pruned_sentences=num_total - num_kept  # Actually num pruned tokens
                            )
                        else:  # pruning_only
                            output = PruningOnlyOutput(
                                pruning_masks=np.array([keep_mask.cpu().numpy()]),
                                sentences=[doc_tokens],  # Store tokens
                                compression_ratio=compression_ratio,
                                num_pruned_tokens=num_total - num_kept
                            )
                        
                        if return_documents:
                            output.pruned_documents = [pruned_doc]
                        
                        all_outputs.append(output)
                    else:
                        # Failed to find document boundaries, create empty output
                        if self.mode == "reranking_pruning":
                            output = RerankingPruningOutput(
                                ranking_scores=ranking_scores[i].cpu().item(),
                                pruning_masks=np.array([[]]),
                                sentences=[[]],
                                compression_ratio=0.0,
                                num_pruned_sentences=0
                            )
                        else:  # pruning_only
                            output = PruningOnlyOutput(
                                pruning_masks=np.array([[]]),
                                sentences=[[]],
                                compression_ratio=0.0,
                                num_pruned_tokens=0
                            )
                        if return_documents:
                            output.pruned_documents = [""]
                        all_outputs.append(output)
        
        return all_outputs[0] if single_input else all_outputs
    
    def predict_context(
        self,
        sentences: List[Tuple[str, str]] | Tuple[str, str],
        chunk_positions: List[List[List[Tuple[int, int]]]] | List[List[Tuple[int, int]]],
        batch_size: int = 32,
        token_threshold: float = 0.5,
        chunk_threshold: float = 0.5,
        show_progress_bar: bool = False,
    ) -> Union[PruningOutput, List[PruningOutput]]:
        """
        Predict with chunk-based evaluation.
        
        Args:
            sentences: Query-document pairs
            chunk_positions: Chunk positions for each document [[start, end], ...]
            batch_size: Batch size
            token_threshold: Threshold for token-level predictions
            chunk_threshold: Minimum ratio of tokens to consider chunk as relevant
            show_progress_bar: Show progress bar
            
        Returns:
            PruningOutput or list of PruningOutput
        """
        self.eval()
        
        single_input = isinstance(sentences[0], str)
        if single_input:
            sentences = [sentences]
            chunk_positions = [chunk_positions]
        
        all_outputs = []
        
        for start_idx in tqdm(
            range(0, len(sentences), batch_size),
            desc="Batches",
            disable=not show_progress_bar,
        ):
            batch = sentences[start_idx:start_idx + batch_size]
            batch_chunks = chunk_positions[start_idx:start_idx + batch_size]
            
            # Tokenize with offset mapping
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            offset_mapping = encoded['offset_mapping']
            
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get predictions
                if self.mode == "reranking_pruning":
                    ranking_logits = outputs["ranking_logits"]
                    if self.num_labels == 1:
                        ranking_scores = self.activation_fn(ranking_logits).squeeze(-1)
                    else:
                        ranking_scores = torch.nn.functional.softmax(ranking_logits, dim=-1)[:, 1]
                else:
                    # No ranking scores in pruning_only mode
                    ranking_scores = torch.zeros(len(batch), device=self.device)
                
                pruning_logits = outputs["pruning_logits"]
                pruning_probs = torch.nn.functional.softmax(pruning_logits, dim=-1)
                keep_probs = pruning_probs[:, :, 1]  # Probability of keeping each token
                
                # Process each example in the batch
                for i in range(len(batch)):
                    query, document = batch[i]
                    chunks = batch_chunks[i]
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                    offsets = offset_mapping[i]
                    
                    # Find document boundaries using </s> token (eos_token_id)
                    eos_token_id = self.tokenizer.eos_token_id or 2
                    sep_positions = (input_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(sep_positions) >= 2:
                        # Document starts after first </s> + <s>
                        doc_start = sep_positions[0].item() + 2
                        doc_end = sep_positions[1].item()
                        
                        # Get document tokens and their keep probabilities
                        doc_keep_probs = keep_probs[i, doc_start:doc_end]
                        doc_offsets = offsets[doc_start:doc_end]
                        
                        # Map chunks to token predictions
                        chunk_scores, chunk_predictions = self._evaluate_chunks(
                            chunks, doc_keep_probs, doc_offsets, 
                            token_threshold, chunk_threshold
                        )
                        
                        # Calculate compression ratio
                        num_kept_chunks = chunk_predictions.sum()
                        num_total_chunks = len(chunks)
                        compression_ratio = 1.0 - (num_kept_chunks / num_total_chunks) if num_total_chunks > 0 else 0.0
                        
                        # Create output
                        output = PruningOutput(
                            ranking_scores=ranking_scores[i].cpu().item(),
                            chunk_predictions=chunk_predictions,
                            chunk_scores=chunk_scores,
                            token_scores=doc_keep_probs.cpu().numpy(),
                            chunk_positions=chunks,
                            compression_ratio=compression_ratio
                        )
                        
                        all_outputs.append(output)
                    else:
                        # Failed to find document boundaries
                        output = PruningOutput(
                            ranking_scores=ranking_scores[i].cpu().item(),
                            chunk_predictions=np.array([]),
                            chunk_scores=np.array([]),
                            token_scores=np.array([]),
                            chunk_positions=chunks,
                            compression_ratio=0.0
                        )
                        all_outputs.append(output)
        
        return all_outputs[0] if single_input else all_outputs
    
    def _evaluate_chunks(
        self,
        chunks: List[Tuple[int, int]],
        token_probs: torch.Tensor,
        token_offsets: torch.Tensor,
        token_threshold: float,
        chunk_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate chunks based on token predictions.
        
        Args:
            chunks: List of chunk positions [(start, end), ...]
            token_probs: Token-level keep probabilities
            token_offsets: Token offset mapping
            token_threshold: Threshold for binary token classification
            chunk_threshold: Minimum ratio for chunk classification
            
        Returns:
            chunk_scores: Average probability for each chunk
            chunk_predictions: Binary predictions for each chunk
        """
        chunk_scores = []
        chunk_predictions = []
        
        for chunk_start, chunk_end in chunks:
            # Find tokens that overlap with this chunk
            overlapping_tokens = []
            overlapping_probs = []
            
            for j, (token_start, token_end) in enumerate(token_offsets):
                if token_start != 0 and token_end != 0:  # Skip special tokens
                    # Check if token overlaps with chunk
                    if token_start < chunk_end and token_end > chunk_start:
                        overlapping_tokens.append(j)
                        overlapping_probs.append(token_probs[j].item())
            
            if overlapping_probs:
                # Calculate chunk-level score
                chunk_score = np.mean(overlapping_probs)
                
                # Apply chunk-level threshold
                # Count tokens above token_threshold
                tokens_above_threshold = sum(1 for prob in overlapping_probs if prob > token_threshold)
                ratio_above_threshold = tokens_above_threshold / len(overlapping_probs)
                
                # Chunk is predicted as relevant if enough tokens are above threshold
                chunk_pred = 1 if ratio_above_threshold >= chunk_threshold else 0
            else:
                # No overlapping tokens found
                chunk_score = 0.0
                chunk_pred = 0
            
            chunk_scores.append(chunk_score)
            chunk_predictions.append(chunk_pred)
        
        return np.array(chunk_scores), np.array(chunk_predictions)
    
    def prune(
        self,
        query: str,
        document: str,
        threshold: float = 0.5,
        min_sentences: int = 1,
        return_sentences: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Prune a single document based on a query using token-level pruning.
        
        Args:
            query: Query text
            document: Document text to prune
            threshold: Pruning threshold
            min_sentences: Ignored (kept for compatibility)
            return_sentences: Return detailed info
            
        Returns:
            Pruned document or detailed results
        """
        output = self.predict_with_pruning(
            (query, document),
            pruning_threshold=threshold,
            return_documents=True
        )
        
        if return_sentences:
            return {
                "pruned_document": output.pruned_documents[0],
                "sentences": [],  # Not applicable for token-level pruning
                "pruning_masks": [],  # Not applicable
                "ranking_score": float(output.ranking_scores) if hasattr(output, 'ranking_scores') and output.ranking_scores is not None else None,
                "compression_ratio": output.compression_ratio,
                "num_pruned_sentences": 0  # Not applicable
            }
        else:
            return output.pruned_documents[0]
    
    def prune_texts(
        self,
        queries: List[str],
        texts: List[str],
        threshold: float = 0.5,
        batch_size: int = 32,
        return_tokens: bool = False,
        show_progress_bar: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Simple API for text pruning without ranking (pruning_only mode).
        
        Args:
            queries: List of queries
            texts: List of texts to prune
            threshold: Pruning threshold (0-1)
            batch_size: Batch size for processing
            return_tokens: Whether to return token-level masks
            show_progress_bar: Show progress bar
            
        Returns:
            List of dicts with:
            - pruned_text: Pruned text
            - kept_ratio: Percentage of text kept
            - pruning_mask: Token-level mask (if return_tokens=True)
        """
        if self.mode != "pruning_only":
            logger.warning(
                "prune_texts() is optimized for pruning_only mode. "
                "Consider using predict() or predict_with_pruning() for reranking_pruning mode."
            )
        
        # Convert to pairs format
        sentences = [(q, t) for q, t in zip(queries, texts)]
        
        # Use predict_with_pruning internally
        outputs = self.predict_with_pruning(
            sentences=sentences,
            batch_size=batch_size,
            pruning_threshold=threshold,
            return_documents=True,
            show_progress_bar=show_progress_bar
        )
        
        # Format results
        results = []
        for i, output in enumerate(outputs):
            result = {
                "pruned_text": output.pruned_documents[0] if output.pruned_documents else texts[i],
                "kept_ratio": 1.0 - output.compression_ratio
            }
            
            if return_tokens:
                result["pruning_mask"] = output.pruning_mask
            
            results.append(result)
        
        return results
    
    def _copy_model_files_for_transformers(self, save_directory: Path) -> None:
        """Copy model files for Transformers compatibility."""
        import shutil
        from pathlib import Path as PathLib
        
        # Copy the modeling file that supports AutoModel loading
        modeling_file = PathLib(__file__).parent / "modeling_pruning_encoder.py"
        if modeling_file.exists():
            shutil.copy(modeling_file, save_directory / "modeling_pruning_encoder.py")
    
    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save the model to a directory."""
        import shutil
        from pathlib import Path as PathLib
        
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save PruningEncoder-specific config
        pruning_encoder_config = {
            "model_name_or_path": self.model_name_or_path,
            "mode": self.mode,  # Save mode
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "pruning_config": self.pruning_head.config.to_dict(),
            "architecture": "PruningEncoder"
        }
        
        with open(save_directory / "pruning_encoder_config.json", "w") as f:
            json.dump(pruning_encoder_config, f, indent=2)
        
        # Copy necessary Python files to the model directory
        # This is required for auto_map to work correctly
        self._copy_model_files_for_transformers(save_directory)
        
        # Save Transformers-compatible config.json with local file references
        transformers_config = {
            "model_type": "pruning_encoder",
            "mode": self.mode,
            "base_model_name_or_path": self.model_name_or_path,
            "pruning_config": self.pruning_head.config.to_dict(),
            "max_length": self.max_length,
            "num_labels": 1 if self.mode == "reranking_pruning" else 2,
            "architectures": [
                "PruningEncoderForSequenceClassification" if self.mode == "reranking_pruning" 
                else "PruningEncoderForTokenClassification"
            ],
            "auto_map": {
                "AutoConfig": "modeling_pruning_encoder.PruningEncoderConfig",
                "AutoModelForSequenceClassification": "modeling_pruning_encoder.PruningEncoderForSequenceClassification" if self.mode == "reranking_pruning" else None,
                "AutoModelForTokenClassification": "modeling_pruning_encoder.PruningEncoderForTokenClassification" if self.mode == "pruning_only" else None,
            }
        }
        
        # Remove None values from auto_map
        transformers_config["auto_map"] = {k: v for k, v in transformers_config["auto_map"].items() if v is not None}
        
        with open(save_directory / "config.json", "w") as f:
            json.dump(transformers_config, f, indent=2)
        
        
        # Save models based on mode
        if self.mode == "reranking_pruning":
            # Save ranking model
            self.ranking_model.save_pretrained(save_directory / "ranking_model")
            self.tokenizer.save_pretrained(save_directory / "ranking_model")
        elif self.mode == "pruning_only":
            # Save encoder model
            self.encoder.save_pretrained(save_directory / "encoder_model")
            self.tokenizer.save_pretrained(save_directory / "encoder_model")
        
        # Also save tokenizer at root for Transformers compatibility
        self.tokenizer.save_pretrained(save_directory)
        
        # Save pruning head
        self.pruning_head.save_pretrained(save_directory / "pruning_head")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        device: Optional[str] = None,
        **kwargs
    ) -> "PruningEncoder":
        """Load a pretrained PruningEncoder."""
        model_path = Path(model_name_or_path)
        
        # Load config - try PruningEncoder config first, then Transformers config
        if (model_path / "pruning_encoder_config.json").exists():
            with open(model_path / "pruning_encoder_config.json", "r") as f:
                config = json.load(f)
        else:
            with open(model_path / "config.json", "r") as f:
                config = json.load(f)
        
        # Get mode (default to reranking_pruning for backward compatibility)
        mode = config.get("mode", "reranking_pruning")
        
        # Determine model subdirectory based on mode
        if mode == "reranking_pruning":
            model_subdir = "ranking_model"
        else:  # pruning_only
            model_subdir = "encoder_model"
        
        # Create encoder
        encoder = cls(
            model_name_or_path=str(model_path / model_subdir),
            mode=mode,
            num_labels=config["num_labels"],
            max_length=config["max_length"],
            device=device,
            pruning_config=config.get("pruning_config", {}),
            **kwargs
        )
        
        # Load pruning head
        encoder.pruning_head = PruningHead.from_pretrained(
            model_path / "pruning_head"
        )
        
        # Move to device
        encoder.to(encoder.device)
        
        return encoder
    
    def to(self, device: Union[str, torch.device]) -> "PruningEncoder":
        """Move model to device."""
        self.device = device
        if self.mode == "reranking_pruning":
            self.ranking_model.to(device)
        else:  # pruning_only
            self.encoder.to(device)
        self.pruning_head.to(device)
        return self
"""
ProvenceEncoder: A standalone encoder for query-dependent text pruning.
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
from .data_structures import ProvenceConfig, ProvenceOutput
from .models.pruning_head import ProvencePruningHead, ProvencePruningConfig

logger = logging.getLogger(__name__)


class ProvenceEncoder(nn.Module):
    """
    ProvenceEncoder performs both reranking and query-dependent text pruning.
    
    This encoder:
    1. Ranks query-document pairs (like a cross-encoder)
    2. Prunes irrelevant sentences from documents based on the query
    
    Args:
        model_name_or_path (str): HuggingFace model name or path
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
        num_labels: int = 1,
        max_length: int = 512,
        device: Optional[str] = None,
        pruning_config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Initialize config
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # Default configs
        tokenizer_args = tokenizer_args or {}
        model_args = model_args or {}
        pruning_config = pruning_config or {}
        
        # Load config
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            cache_dir=cache_dir,
            **model_args
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args
        )
        
        # Load ranking model
        self.ranking_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
            cache_dir=cache_dir,
            **model_args
        )
        
        # Initialize pruning head
        hidden_size = self.config.hidden_size
        pruning_head_config = ProvencePruningConfig(
            hidden_size=pruning_config.get("hidden_size", hidden_size),
            num_labels=2,  # Binary: keep/prune
            classifier_dropout=pruning_config.get("dropout", 0.1),
            sentence_pooling=pruning_config.get("sentence_pooling", "mean"),
            use_weighted_pooling=pruning_config.get("use_weighted_pooling", False)
        )
        self.pruning_head = ProvencePruningHead(pruning_head_config)
        
        # Text chunker for sentence segmentation (only needed for raw text API)
        self.text_chunker = None
        
        # Activation function for ranking scores
        if num_labels == 1:
            self.activation_fn = nn.Sigmoid()
        else:
            self.activation_fn = nn.Identity()
        
        # Move to device
        self.to(self.device)
        
        # Default Provence config
        self.provence_config = ProvenceConfig()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_boundaries: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs  # Accept additional kwargs like token_type_ids
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both ranking and pruning.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            sentence_boundaries: Token boundaries for each sentence
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with ranking_logits and pruning_logits
        """
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
        
        # Get pruning predictions
        pruning_outputs = self.pruning_head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            sentence_boundaries=sentence_boundaries
        )
        
        if return_dict:
            return {
                "ranking_logits": ranking_logits,
                "pruning_logits": pruning_outputs.logits,
                "hidden_states": hidden_states
            }
        
        return ranking_logits, pruning_outputs.logits
    
    def predict(
        self,
        sentences: List[Tuple[str, str]] | Tuple[str, str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> Union[List[float], np.ndarray, torch.Tensor]:
        """
        Predict ranking scores for query-document pairs.
        
        Args:
            sentences: Query-document pairs
            batch_size: Batch size for prediction
            show_progress_bar: Show progress bar
            convert_to_numpy: Convert to numpy array
            convert_to_tensor: Convert to tensor
            
        Returns:
            Ranking scores
        """
        self.eval()
        
        if isinstance(sentences[0], str):
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
    ) -> Union[ProvenceOutput, List[ProvenceOutput]]:
        """
        Predict with token-level pruning.
        
        Args:
            sentences: Query-document pairs
            batch_size: Batch size
            pruning_threshold: Threshold for pruning decisions
            return_documents: Whether to return pruned documents
            show_progress_bar: Show progress bar
            
        Returns:
            ProvenceOutput or list of ProvenceOutput
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
                ranking_logits = outputs["ranking_logits"]
                if self.num_labels == 1:
                    ranking_scores = self.activation_fn(ranking_logits).squeeze(-1)
                else:
                    ranking_scores = torch.nn.functional.softmax(ranking_logits, dim=-1)[:, 1]
                
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
                        
                        # Create output (compatible with original format)
                        output = ProvenceOutput(
                            ranking_scores=ranking_scores[i].cpu().numpy(),
                            pruning_masks=np.array([keep_mask.cpu().numpy()]),
                            sentences=[doc_tokens],  # Store tokens instead of sentences
                            compression_ratio=compression_ratio,
                            num_pruned_sentences=num_total - num_kept  # Actually num pruned tokens
                        )
                        
                        if return_documents:
                            output.pruned_documents = [pruned_doc]
                        
                        all_outputs.append(output)
                    else:
                        # Failed to find document boundaries, create empty output
                        output = ProvenceOutput(
                            ranking_scores=ranking_scores[i].cpu().numpy(),
                            pruning_masks=np.array([[]]),
                            sentences=[[]],
                            compression_ratio=0.0,
                            num_pruned_sentences=0
                        )
                        if return_documents:
                            output.pruned_documents = [""]
                        all_outputs.append(output)
        
        return all_outputs[0] if single_input else all_outputs
    
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
                "ranking_score": float(output.ranking_scores),
                "compression_ratio": output.compression_ratio,
                "num_pruned_sentences": 0  # Not applicable
            }
        else:
            return output.pruned_documents[0]
    
    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save the model to a directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            "model_name_or_path": self.model_name_or_path,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "pruning_config": self.pruning_head.config.to_dict(),
            "architecture": "ProvenceEncoder"
        }
        
        with open(save_directory / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save ranking model
        self.ranking_model.save_pretrained(save_directory / "ranking_model")
        self.tokenizer.save_pretrained(save_directory / "ranking_model")
        
        # Save pruning head
        self.pruning_head.save_pretrained(save_directory / "pruning_head")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        device: Optional[str] = None,
        **kwargs
    ) -> "ProvenceEncoder":
        """Load a pretrained ProvenceEncoder."""
        model_path = Path(model_name_or_path)
        
        # Load config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create encoder
        encoder = cls(
            model_name_or_path=str(model_path / "ranking_model"),
            num_labels=config["num_labels"],
            max_length=config["max_length"],
            device=device,
            pruning_config=config.get("pruning_config", {}),
            **kwargs
        )
        
        # Load pruning head
        encoder.pruning_head = ProvencePruningHead.from_pretrained(
            model_path / "pruning_head"
        )
        
        # Move to device
        encoder.to(encoder.device)
        
        return encoder
    
    def to(self, device: Union[str, torch.device]) -> "ProvenceEncoder":
        """Move model to device."""
        self.device = device
        self.ranking_model.to(device)
        self.pruning_head.to(device)
        return self
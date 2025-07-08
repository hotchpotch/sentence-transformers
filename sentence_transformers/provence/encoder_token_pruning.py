"""
ProvenceEncoder with proper token-level pruning support.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from pathlib import Path
import json
import logging
from transformers import AutoTokenizer
from cross_encoder import CrossEncoder

from .data_structures import ProvenceConfig, ProvenceOutput
from .pruning_head import ProvencePruningHead


logger = logging.getLogger(__name__)


class ProvenceEncoder(nn.Module):
    """
    Provence Encoder with token-level pruning support.
    
    This is a fixed version that properly handles token-level pruning
    during inference, matching the training behavior.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 1,
        max_length: int = 512,
        device: Optional[str] = None,
        tokenizer_args: Optional[Dict] = None,
        automodel_args: Optional[Dict] = None,
        pruning_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Initialize ranking model (CrossEncoder)
        self.ranking_model = CrossEncoder(
            model_name_or_path,
            num_labels=num_labels,
            max_length=max_length,
            device=device,
            tokenizer_args=tokenizer_args,
            automodel_args=automodel_args
        )
        
        # Share tokenizer
        self.tokenizer = self.ranking_model.tokenizer
        
        # Get hidden size from the model
        self.hidden_size = self.ranking_model.model.config.hidden_size
        
        # Initialize pruning head
        pruning_head_config = ProvenceConfig(
            hidden_size=self.hidden_size,
            num_labels=2,  # Binary classification for each token
            dropout=pruning_config.get("dropout", 0.1) if pruning_config else 0.1,
            sentence_pooling=pruning_config.get("sentence_pooling", "mean") if pruning_config else "mean",
            use_weighted_pooling=pruning_config.get("use_weighted_pooling", False) if pruning_config else False
        )
        self.pruning_head = ProvencePruningHead(pruning_head_config)
        
        # Device handling
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._target_device = torch.device(device)
        self.to(self._target_device)
        
        # Activation function for ranking scores
        if num_labels == 1:
            self.activation_fn = nn.Sigmoid()
        else:
            self.activation_fn = lambda x: torch.nn.functional.softmax(x, dim=-1)[:, 1]
    
    @property
    def device(self):
        return self._target_device
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for both ranking and pruning."""
        # Get base model outputs
        outputs = self.ranking_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=True
        )
        
        # Ranking head
        ranking_logits = self.ranking_model.classifier(outputs.pooler_output)
        
        # Pruning head (token-level)
        last_hidden_states = outputs.last_hidden_state
        pruning_outputs = self.pruning_head(
            last_hidden_states,
            attention_mask=attention_mask
        )
        
        return {
            "ranking_logits": ranking_logits,
            "pruning_logits": pruning_outputs["logits"],
            "hidden_states": last_hidden_states
        }
    
    def predict(
        self,
        sentences: List[Tuple[str, str]] | List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> Union[List[float], np.ndarray, torch.Tensor]:
        """Predict ranking scores only (for compatibility)."""
        self.eval()
        
        if isinstance(sentences[0], str):
            sentences = [sentences]
        
        all_scores = []
        
        for start_idx in range(0, len(sentences), batch_size):
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
                    scores = self.activation_fn(logits)
                
                all_scores.extend(scores.cpu().tolist())
        
        if convert_to_tensor:
            return torch.tensor(all_scores)
        elif convert_to_numpy:
            return np.array(all_scores)
        else:
            return all_scores
    
    def predict_with_token_pruning(
        self,
        sentences: List[Tuple[str, str]] | Tuple[str, str],
        batch_size: int = 32,
        pruning_threshold: float = 0.5,
        return_documents: bool = False,
        show_progress_bar: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict with token-level pruning.
        
        Returns dict with:
        - ranking_score: float
        - pruned_tokens: List[str] (tokens kept)
        - pruned_document: str (reconstructed text)
        - compression_ratio: float
        - token_mask: List[bool] (which tokens were kept)
        """
        self.eval()
        
        single_input = isinstance(sentences[0], str)
        if single_input:
            sentences = [sentences]
        
        all_results = []
        
        for start_idx in range(0, len(sentences), batch_size):
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
                    ranking_scores = self.activation_fn(ranking_logits)
                
                # Get token-level pruning predictions
                pruning_logits = outputs["pruning_logits"]
                pruning_probs = torch.nn.functional.softmax(pruning_logits, dim=-1)
                keep_probs = pruning_probs[:, :, 1]  # Probability of keeping each token
                
                # Process each example
                for i in range(len(batch)):
                    query, document = batch[i]
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                    offsets = offset_mapping[i]
                    
                    # Find document boundaries
                    sep_positions = (input_ids[i] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(sep_positions) >= 3:
                        doc_start = sep_positions[1].item() + 1
                        doc_end = sep_positions[2].item()
                        
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
                                pruned_document = " ".join(pruned_parts)
                            else:
                                pruned_document = ""
                        else:
                            pruned_document = None
                        
                        result = {
                            "ranking_score": ranking_scores[i].item(),
                            "pruned_tokens": [t for t, k in zip(doc_tokens, keep_mask) if k],
                            "pruned_document": pruned_document,
                            "compression_ratio": compression_ratio,
                            "token_mask": keep_mask.cpu().tolist(),
                            "num_kept_tokens": num_kept,
                            "num_total_tokens": num_total
                        }
                        
                        all_results.append(result)
        
        return all_results[0] if single_input else all_results
    
    def prune(
        self,
        query: str,
        document: str,
        threshold: float = 0.5,
        return_sentences: bool = False,  # For compatibility
    ) -> Dict[str, Any]:
        """
        Prune a single document based on a query using token-level pruning.
        
        Args:
            query: Query text
            document: Document text to prune
            threshold: Pruning threshold
            return_sentences: Ignored (kept for compatibility)
            
        Returns:
            Dictionary with pruning results
        """
        result = self.predict_with_token_pruning(
            (query, document),
            pruning_threshold=threshold,
            return_documents=True
        )
        
        # Format for compatibility
        return {
            "pruned_document": result["pruned_document"],
            "ranking_score": result["ranking_score"],
            "compression_ratio": result["compression_ratio"],
            "num_pruned_sentences": 0,  # Not applicable for token-level
            "sentences": [],  # Not applicable
            "pruning_masks": []  # Not applicable
        }
    
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
    def from_pretrained(cls, model_name_or_path: Union[str, Path], **kwargs):
        """Load a pretrained model."""
        model_path = Path(model_name_or_path)
        
        # Load config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(
            model_name_or_path=str(model_path / "ranking_model"),
            num_labels=config["num_labels"],
            max_length=config["max_length"],
            pruning_config=config.get("pruning_config", {}),
            **kwargs
        )
        
        # Load pruning head
        model.pruning_head.load_state_dict(
            torch.load(model_path / "pruning_head" / "pytorch_model.bin", map_location=model.device)
        )
        
        return model
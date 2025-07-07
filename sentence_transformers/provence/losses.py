"""
ProvenceLoss for joint training of reranking and pruning objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class ProvenceLoss(nn.Module):
    """
    Joint loss function for Provence that combines reranking and pruning objectives.
    
    This loss supports:
    - Binary classification for reranking (relevant/irrelevant)
    - Teacher score distillation for reranking
    - Token-level or sentence-level pruning classification
    """
    
    def __init__(self,
                 model,
                 ranking_loss_fn: Optional[nn.Module] = None,
                 pruning_loss_fn: Optional[nn.Module] = None,
                 ranking_weight: float = 1.0,
                 pruning_weight: float = 0.5,
                 use_teacher_scores: bool = False,
                 sentence_level_pruning: bool = False):
        """
        Args:
            model: CrossEncoder model with enable_pruning=True
            ranking_loss_fn: Loss function for ranking (default: BCEWithLogitsLoss)
            pruning_loss_fn: Loss function for pruning (default: CrossEntropyLoss)
            ranking_weight: Weight for ranking loss
            pruning_weight: Weight for pruning loss
            use_teacher_scores: Whether to use teacher scores for distillation
            sentence_level_pruning: Whether to compute loss at sentence level
        """
        super().__init__()
        
        self.model = model
        self.ranking_loss_fn = ranking_loss_fn or nn.BCEWithLogitsLoss()
        self.pruning_loss_fn = pruning_loss_fn or nn.CrossEntropyLoss(reduction='none')
        self.ranking_weight = ranking_weight
        self.pruning_weight = pruning_weight
        self.use_teacher_scores = use_teacher_scores
        self.sentence_level_pruning = sentence_level_pruning
        
    def forward(self, sentence_features: List[Dict[str, torch.Tensor]], labels: Dict[str, torch.Tensor]):
        """
        Compute joint loss for ranking and pruning.
        
        Args:
            sentence_features: List with single dict containing tokenized inputs
            labels: Dictionary containing:
                - 'ranking_labels': [batch_size] (0/1 or continuous scores)
                - 'pruning_labels': [batch_size, max_sentences] or [batch_size, seq_len]
                - 'sentence_boundaries': [batch_size, max_sentences, 2] (optional)
                - 'teacher_scores': [batch_size] (optional, for distillation)
                
        Returns:
            Total loss value
        """
        # Get model outputs
        # Handle both dict and list inputs
        if isinstance(sentence_features, list) and len(sentence_features) > 0:
            inputs = sentence_features[0]
        else:
            inputs = sentence_features
            
        # Call model forward
        outputs = self.model.forward(
            input_ids=inputs.get('input_ids'),
            attention_mask=inputs.get('attention_mask'),
            sentence_boundaries=labels.get('sentence_boundaries')
        )
        
        total_loss = 0.0
        losses = {}
        
        # Ranking loss
        if 'ranking_logits' in outputs and 'ranking_labels' in labels:
            ranking_labels = labels['ranking_labels']
            ranking_logits = outputs['ranking_logits']
            
            # Ensure correct shape
            if ranking_logits.dim() > 1 and ranking_logits.shape[-1] == 1:
                ranking_logits = ranking_logits.squeeze(-1)
            
            if self.use_teacher_scores and 'teacher_scores' in labels:
                # Knowledge distillation using teacher scores
                ranking_loss = F.mse_loss(
                    ranking_logits,
                    labels['teacher_scores']
                )
            else:
                # Binary classification
                ranking_loss = self.ranking_loss_fn(
                    ranking_logits,
                    ranking_labels.float()
                )
                
            total_loss += self.ranking_weight * ranking_loss
            losses['ranking_loss'] = ranking_loss
            
        # Pruning loss
        if 'pruning_logits' in outputs and 'pruning_labels' in labels:
            pruning_logits = outputs['pruning_logits']  # [batch, seq_len, 2]
            pruning_labels = labels['pruning_labels']   # [batch, max_sentences] or [batch, seq_len]
            
            if self.sentence_level_pruning and 'sentence_boundaries' in labels:
                # Sentence-level pruning loss
                sentence_loss = self._compute_sentence_level_loss(
                    pruning_logits,
                    pruning_labels,
                    labels['sentence_boundaries'],
                    sentence_features[0].get('attention_mask', None)
                )
                total_loss += self.pruning_weight * sentence_loss
                losses['pruning_loss'] = sentence_loss
            else:
                # Token-level pruning loss
                attention_mask = sentence_features[0].get('attention_mask', None)
                
                if attention_mask is not None:
                    # Flatten and apply mask
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = pruning_logits.view(-1, 2)[active_loss]
                    active_labels = pruning_labels.view(-1)[active_loss]
                    
                    if active_logits.shape[0] > 0:
                        token_loss = self.pruning_loss_fn(active_logits, active_labels)
                        token_loss = token_loss.mean()
                    else:
                        token_loss = torch.tensor(0.0, device=pruning_logits.device, requires_grad=True)
                else:
                    # No mask - compute loss on all tokens
                    token_loss = self.pruning_loss_fn(
                        pruning_logits.view(-1, 2),
                        pruning_labels.view(-1)
                    ).mean()
                
                total_loss += self.pruning_weight * token_loss
                losses['pruning_loss'] = token_loss
        
        # Store individual losses for logging
        self.last_losses = losses
        
        return total_loss
    
    def _compute_sentence_level_loss(self, 
                                    logits: torch.Tensor,
                                    labels: torch.Tensor, 
                                    boundaries: torch.Tensor,
                                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute sentence-level pruning loss by pooling token predictions.
        
        Args:
            logits: [batch_size, seq_len, 2] - Token-level pruning logits
            labels: [batch_size, max_sentences] - Sentence-level labels
            boundaries: [batch_size, max_sentences, 2] - Token boundaries for sentences
            attention_mask: [batch_size, seq_len] - Attention mask
            
        Returns:
            Scalar loss value
        """
        batch_size, seq_len, num_labels = logits.shape
        max_sentences = boundaries.shape[1]
        device = logits.device
        
        # Initialize sentence logits
        sentence_logits = torch.zeros(
            batch_size, max_sentences, num_labels, 
            device=device, dtype=logits.dtype
        )
        
        # Pool logits for each sentence
        for b in range(batch_size):
            for s in range(max_sentences):
                start, end = boundaries[b, s]
                
                # Skip padding (-1 boundaries)
                if start == -1 or end == -1:
                    continue
                
                # Get token logits for this sentence
                sentence_tokens = logits[b, start:end]  # [num_tokens, num_labels]
                
                if sentence_tokens.shape[0] == 0:
                    continue
                
                # Apply mean pooling (can be configured)
                pooled = sentence_tokens.mean(dim=0)
                sentence_logits[b, s] = pooled
        
        # Flatten for loss computation
        valid_mask = (boundaries[:, :, 0] != -1)  # [batch_size, max_sentences]
        valid_logits = sentence_logits[valid_mask]  # [num_valid, num_labels]
        valid_labels = labels[valid_mask]  # [num_valid]
        
        # Compute loss
        if valid_logits.shape[0] > 0:
            loss = self.pruning_loss_fn(valid_logits, valid_labels).mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration dictionary for saving."""
        return {
            "ranking_weight": self.ranking_weight,
            "pruning_weight": self.pruning_weight,
            "use_teacher_scores": self.use_teacher_scores,
            "sentence_level_pruning": self.sentence_level_pruning,
            "ranking_loss": self.ranking_loss_fn.__class__.__name__,
            "pruning_loss": self.pruning_loss_fn.__class__.__name__
        }
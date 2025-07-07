"""
Batched ProvenceLoss for joint training with multiple texts per query.
Based on the pattern from LambdaLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class ProvenceBatchedLoss(nn.Module):
    """
    Joint loss function for Provence that handles multiple texts per query.
    
    This loss supports:
    - Multiple texts per query (e.g., 1 positive + 4 negatives)
    - Binary classification for reranking
    - Teacher score distillation
    - Token-level pruning classification
    """
    
    def __init__(self,
                 model,
                 ranking_loss_fn: Optional[nn.Module] = None,
                 pruning_loss_fn: Optional[nn.Module] = None,
                 ranking_weight: float = 1.0,
                 pruning_weight: float = 0.5,
                 use_teacher_scores: bool = False,
                 mini_batch_size: Optional[int] = None):
        """
        Args:
            model: ProvenceEncoder model
            ranking_loss_fn: Loss function for ranking
            pruning_loss_fn: Loss function for pruning
            ranking_weight: Weight for ranking loss
            pruning_weight: Weight for pruning loss
            use_teacher_scores: Whether to use teacher scores for distillation
            mini_batch_size: Size for processing mini-batches
        """
        super().__init__()
        
        self.model = model
        self.ranking_loss_fn = ranking_loss_fn or nn.BCEWithLogitsLoss(reduction='none')
        self.pruning_loss_fn = pruning_loss_fn or nn.CrossEntropyLoss(reduction='none')
        self.ranking_weight = ranking_weight
        self.pruning_weight = pruning_weight
        self.use_teacher_scores = use_teacher_scores
        self.mini_batch_size = mini_batch_size
        
    def forward(self, sentence_features: List[Dict[str, torch.Tensor]], labels: Dict[str, torch.Tensor]):
        """
        Compute joint loss for ranking and pruning with multiple texts per query.
        
        Args:
            sentence_features: List with single dict containing tokenized inputs for all pairs
            labels: Dictionary containing:
                - 'ranking_labels': [batch_size, max_docs] matrix
                - 'teacher_scores': [batch_size, max_docs] matrix
                - 'pruning_labels': Token-level labels for all pairs
                - 'batch_indices': Batch index for each pair
                - 'doc_indices': Document index for each pair
                - 'docs_per_query': Number of documents per query
                
        Returns:
            Total loss value
        """
        # Get model inputs
        inputs = sentence_features[0] if isinstance(sentence_features, list) else sentence_features
        
        # Get dimensions
        batch_indices = labels['batch_indices']
        doc_indices = labels['doc_indices']
        docs_per_query = labels['docs_per_query']
        batch_size = len(docs_per_query)
        max_docs = max(docs_per_query)
        
        # Process all pairs (potentially in mini-batches)
        num_pairs = inputs['input_ids'].shape[0]
        
        if self.mini_batch_size and num_pairs > self.mini_batch_size:
            # Process in mini-batches
            all_outputs = []
            for i in range(0, num_pairs, self.mini_batch_size):
                end_idx = min(i + self.mini_batch_size, num_pairs)
                mini_inputs = {
                    k: v[i:end_idx] for k, v in inputs.items()
                }
                
                # Forward pass for mini-batch
                mini_outputs = self.model.forward(
                    input_ids=mini_inputs.get('input_ids'),
                    attention_mask=mini_inputs.get('attention_mask')
                )
                all_outputs.append(mini_outputs)
            
            # Combine outputs
            outputs = self._combine_outputs(all_outputs)
        else:
            # Process all at once
            outputs = self.model.forward(
                input_ids=inputs.get('input_ids'),
                attention_mask=inputs.get('attention_mask')
            )
        
        total_loss = 0.0
        losses = {}
        
        # Ranking loss
        if 'ranking_logits' in outputs:
            ranking_loss = self._compute_ranking_loss(
                outputs['ranking_logits'],
                labels,
                batch_indices,
                doc_indices,
                batch_size,
                max_docs
            )
            total_loss += self.ranking_weight * ranking_loss
            losses['ranking_loss'] = ranking_loss
        
        # Pruning loss
        if 'pruning_logits' in outputs and 'pruning_labels' in labels:
            pruning_loss = self._compute_pruning_loss(
                outputs['pruning_logits'],
                labels['pruning_labels'],
                inputs.get('attention_mask')
            )
            total_loss += self.pruning_weight * pruning_loss
            losses['pruning_loss'] = pruning_loss
        
        # Store individual losses for logging
        self.last_losses = losses
        
        return total_loss
    
    def _combine_outputs(self, outputs_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combine outputs from mini-batches."""
        combined = {}
        for key in outputs_list[0].keys():
            combined[key] = torch.cat([o[key] for o in outputs_list], dim=0)
        return combined
    
    def _compute_ranking_loss(self,
                              logits: torch.Tensor,
                              labels: Dict[str, torch.Tensor],
                              batch_indices: torch.Tensor,
                              doc_indices: torch.Tensor,
                              batch_size: int,
                              max_docs: int) -> torch.Tensor:
        """
        Compute ranking loss with proper reshaping for batched data.
        """
        # Ensure logits are 1D
        if logits.dim() > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        
        # Create logits matrix [batch_size, max_docs]
        logits_matrix = torch.full(
            (batch_size, max_docs),
            fill_value=-1e4,  # Large negative value for padding
            device=logits.device,
            dtype=logits.dtype  # Match the dtype of logits
        )
        
        # Place logits in the correct positions
        logits_matrix[batch_indices, doc_indices] = logits
        
        # Get labels
        ranking_labels = labels['ranking_labels'].to(logits.device)
        
        if self.use_teacher_scores and 'teacher_scores' in labels:
            # Teacher score distillation
            teacher_scores = labels['teacher_scores'].to(logits.device)
            
            # Mask out padded positions
            mask = ranking_labels != -100
            
            # Apply sigmoid to logits for MSE with teacher scores
            pred_scores = torch.sigmoid(logits_matrix)
            
            # Compute MSE only on valid positions
            loss = F.mse_loss(
                pred_scores[mask],
                teacher_scores[mask]
            )
        else:
            # Binary classification
            # Mask out padded positions
            mask = ranking_labels != -100
            
            # Compute BCE loss only on valid positions
            loss = self.ranking_loss_fn(
                logits_matrix[mask],
                ranking_labels[mask].float()
            ).mean()
        
        return loss
    
    def _compute_pruning_loss(self,
                              logits: torch.Tensor,
                              labels: torch.Tensor,
                              attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute token-level pruning loss.
        """
        if attention_mask is not None:
            # Apply mask
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, 2)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            if active_logits.shape[0] > 0:
                loss = self.pruning_loss_fn(active_logits, active_labels).mean()
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            # No mask
            loss = self.pruning_loss_fn(
                logits.view(-1, 2),
                labels.view(-1)
            ).mean()
        
        return loss
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration dictionary for saving."""
        return {
            "ranking_weight": self.ranking_weight,
            "pruning_weight": self.pruning_weight,
            "use_teacher_scores": self.use_teacher_scores,
            "mini_batch_size": self.mini_batch_size,
            "ranking_loss": self.ranking_loss_fn.__class__.__name__,
            "pruning_loss": self.pruning_loss_fn.__class__.__name__
        }
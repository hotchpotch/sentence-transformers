"""
PruningTrainer: Trainer for PruningEncoder models.
"""

import os
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizer
from transformers.trainer import Trainer
from datasets import Dataset as HFDataset
from tqdm import tqdm

from .encoder import PruningEncoder
from .losses import PruningLoss
from .data_collator import PruningDataCollator

logger = logging.getLogger(__name__)


class PruningTrainer(Trainer):
    """Custom Trainer that uses PruningTrainer internally for compatibility."""
    
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self._eval_loss_components = {}
        # Track loss components similar to yast
        self._accumulated_loss_components = {}
        self._loss_component_counts = {}
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss using PruningLoss."""
        sentence_features = inputs["sentence_features"]
        labels = inputs["labels"]
        
        # Move tensors to device
        for key in sentence_features[0]:
            if isinstance(sentence_features[0][key], torch.Tensor):
                sentence_features[0][key] = sentence_features[0][key].to(model.device)
        
        for key in labels:
            if isinstance(labels[key], torch.Tensor):
                labels[key] = labels[key].to(model.device)
        
        # Compute loss
        loss = self.loss_fn(sentence_features, labels)
        
        # Track loss components for aggregation (similar to yast)
        if hasattr(self.loss_fn, 'last_loss_components') and self.loss_fn.last_loss_components:
            for name, value in self.loss_fn.last_loss_components.items():
                if name not in self._accumulated_loss_components:
                    self._accumulated_loss_components[name] = 0.0
                    self._loss_component_counts[name] = 0
                
                if isinstance(value, torch.Tensor):
                    self._accumulated_loss_components[name] += value.item()
                else:
                    self._accumulated_loss_components[name] += value
                self._loss_component_counts[name] += 1
        
        return (loss, None) if return_outputs else loss
    
    def log(self, logs: dict, start_time=None, **kwargs) -> None:
        """Override log method to include accumulated loss components (inspired by yast)."""
        # Add step and epoch
        logs["step"] = self.state.global_step
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        
        # Calculate and add mean loss components
        if self._accumulated_loss_components:
            mean_components = {}
            for name, total in self._accumulated_loss_components.items():
                count = self._loss_component_counts.get(name, 1)
                if count > 0:
                    mean_components[name] = total / count
            
            # Update logs with mean components
            logs.update(mean_components)
            
            # Clear accumulators
            self._accumulated_loss_components.clear()
            self._loss_component_counts.clear()
        
        # Append to log history
        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)
        
        # Call parent's callback handler
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        """Custom prediction step that handles PruningDataCollator format."""
        has_labels = "labels" in inputs
        
        with torch.no_grad():
            if has_labels:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                
                # Track eval loss components
                if hasattr(self.loss_fn, 'last_loss_components') and self.loss_fn.last_loss_components:
                    for name, value in self.loss_fn.last_loss_components.items():
                        if name not in self._eval_loss_components:
                            self._eval_loss_components[name] = []
                        if isinstance(value, torch.Tensor):
                            self._eval_loss_components[name].append(value.item())
                        else:
                            self._eval_loss_components[name].append(value)
            else:
                loss = None
        
        # For evaluation, we mainly care about the loss
        if prediction_loss_only:
            return (loss, None, None)
        
        # We don't need logits for evaluation in this case
        return (loss, None, None)
    
    def evaluation_loop(self, *args, **kwargs):
        """Override evaluation loop to log loss components."""
        # Reset eval loss components
        self._eval_loss_components = {}
        
        # Run parent evaluation loop
        output = super().evaluation_loop(*args, **kwargs)
        
        # Calculate average loss components and add to metrics
        if self._eval_loss_components:
            for name, values in self._eval_loss_components.items():
                avg_value = sum(values) / len(values)
                output.metrics[f'eval_{name}'] = avg_value
            
        
        return output
"""
Minimal modeling file for PruningEncoder to support AutoModel loading.
This file is copied to the model directory when saving for Hugging Face compatibility.
"""

import os
import json
from typing import Optional, Union
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput


class PruningEncoderConfig(PretrainedConfig):
    """Configuration class for PruningEncoder models."""
    
    model_type = "pruning_encoder"
    
    def __init__(
        self,
        mode: str = "reranking_pruning",
        base_model_name_or_path: Optional[str] = None,
        pruning_config: Optional[dict] = None,
        max_length: int = 512,
        num_labels: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.base_model_name_or_path = base_model_name_or_path
        self.pruning_config = pruning_config or {}
        self.max_length = max_length
        
        if num_labels is None:
            self.num_labels = 1 if mode == "reranking_pruning" else 2
        else:
            self.num_labels = num_labels


class PruningEncoderForSequenceClassification(PreTrainedModel):
    """PruningEncoder for sequence classification (reranking)."""
    
    config_class = PruningEncoderConfig
    
    def __init__(self, config: PruningEncoderConfig):
        super().__init__(config)
        # Lazy import to avoid circular dependencies
        from sentence_transformers.pruning import PruningEncoder
        
        self.pruning_encoder = PruningEncoder(
            model_name_or_path=config.base_model_name_or_path,
            mode="reranking_pruning",
            max_length=config.max_length,
            pruning_config=config.pruning_config,
            device=None
        )
        self.config = config
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            
        outputs = self.pruning_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        
        logits = outputs.get("ranking_logits", outputs.get("logits"))
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.get("hidden_states"),
            attentions=outputs.get("attentions")
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from a PruningEncoder checkpoint."""
        from sentence_transformers.pruning import PruningEncoder
        
        config = PruningEncoderConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        model.pruning_encoder = PruningEncoder.from_pretrained(
            pretrained_model_name_or_path,
            device=kwargs.get("device_map")
        )
        return model


class PruningEncoderForTokenClassification(PreTrainedModel):
    """PruningEncoder for token classification (pruning)."""
    
    config_class = PruningEncoderConfig
    
    def __init__(self, config: PruningEncoderConfig):
        super().__init__(config)
        from sentence_transformers.pruning import PruningEncoder
        
        self.pruning_encoder = PruningEncoder(
            model_name_or_path=config.base_model_name_or_path,
            mode="pruning_only",
            max_length=config.max_length,
            pruning_config=config.pruning_config,
            device=None
        )
        self.config = config
        self.num_labels = 2
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            
        outputs = self.pruning_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        
        logits = outputs.get("pruning_logits")
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
            
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.get("hidden_states"),
            attentions=outputs.get("attentions")
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from a PruningEncoder checkpoint."""
        from sentence_transformers.pruning import PruningEncoder
        
        config = PruningEncoderConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        model.pruning_encoder = PruningEncoder.from_pretrained(
            pretrained_model_name_or_path,
            device=kwargs.get("device_map")
        )
        return model
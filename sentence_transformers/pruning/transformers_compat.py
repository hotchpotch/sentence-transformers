"""
Transformers library compatibility wrappers for PruningEncoder.
Enables loading models via AutoModelForSequenceClassification and AutoModelForTokenClassification.
"""

import os
import json
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig, 
    PreTrainedModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    TokenClassifierOutput
)

from .encoder import PruningEncoder


class PruningEncoderConfig(PretrainedConfig):
    """Configuration class for PruningEncoder models."""
    
    model_type = "pruning_encoder"
    
    def __init__(
        self,
        mode: str = "reranking_pruning",
        base_model_name_or_path: Optional[str] = None,
        pruning_config: Optional[Dict[str, Any]] = None,
        max_length: int = 512,
        num_labels: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize PruningEncoderConfig.
        
        Args:
            mode: Either "reranking_pruning" or "pruning_only"
            base_model_name_or_path: Path to the base transformer model
            pruning_config: Configuration for the pruning head
            max_length: Maximum sequence length
            num_labels: Number of labels (1 for ranking, 2 for token classification)
        """
        super().__init__(**kwargs)
        self.mode = mode
        self.base_model_name_or_path = base_model_name_or_path
        self.pruning_config = pruning_config or {}
        self.max_length = max_length
        
        # Set num_labels based on mode if not provided
        if num_labels is None:
            self.num_labels = 1 if mode == "reranking_pruning" else 2
        else:
            self.num_labels = num_labels


class PruningEncoderForSequenceClassification(PreTrainedModel):
    """
    PruningEncoder wrapper for sequence classification (reranking).
    Compatible with AutoModelForSequenceClassification.
    """
    
    config_class = PruningEncoderConfig
    base_model_prefix = "pruning_encoder"
    
    def __init__(self, config: PruningEncoderConfig):
        super().__init__(config)
        
        # Initialize PruningEncoder
        self.pruning_encoder = PruningEncoder(
            model_name_or_path=config.base_model_name_or_path,
            mode="reranking_pruning",
            max_length=config.max_length,
            pruning_config=config.pruning_config,
            device=None  # Will use default device handling
        )
        
        # Set config attributes
        self.config = config
        self.num_labels = config.num_labels
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[SequenceClassifierOutput, tuple]:
        """
        Forward pass compatible with sequence classification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for BERT-like models)
            labels: Ground truth labels for computing loss
            return_dict: Whether to return a dictionary
            
        Returns:
            SequenceClassifierOutput or tuple
        """
        # Forward through PruningEncoder with correct arguments
        outputs = self.pruning_encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        
        # Extract ranking logits
        logits = outputs.get("ranking_logits", outputs.get("logits"))
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs.get("hidden_states", ())
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
        # First try to load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            config = PruningEncoderConfig.from_pretrained(pretrained_model_name_or_path)
        else:
            # Create config from PruningEncoder's saved config
            pruning_config_path = os.path.join(pretrained_model_name_or_path, "pruning_encoder_config.json")
            if os.path.exists(pruning_config_path):
                with open(pruning_config_path, 'r') as f:
                    pruning_config = json.load(f)
                config = PruningEncoderConfig(
                    mode=pruning_config.get("mode", "reranking_pruning"),
                    base_model_name_or_path=pretrained_model_name_or_path,
                    pruning_config=pruning_config.get("pruning_config", {}),
                    max_length=pruning_config.get("max_length", 512)
                )
            else:
                raise ValueError(f"No config found at {pretrained_model_name_or_path}")
        
        # Create model
        model = cls(config)
        
        # Load weights from PruningEncoder
        model.pruning_encoder = PruningEncoder.from_pretrained(
            pretrained_model_name_or_path,
            device=kwargs.get("device_map", None)
        )
        
        return model


class PruningEncoderForTokenClassification(PreTrainedModel):
    """
    PruningEncoder wrapper for token classification (pruning).
    Compatible with AutoModelForTokenClassification.
    """
    
    config_class = PruningEncoderConfig
    base_model_prefix = "pruning_encoder"
    
    def __init__(self, config: PruningEncoderConfig):
        super().__init__(config)
        
        # Initialize PruningEncoder in pruning_only mode
        self.pruning_encoder = PruningEncoder(
            model_name_or_path=config.base_model_name_or_path,
            mode="pruning_only",
            max_length=config.max_length,
            pruning_config=config.pruning_config,
            device=None
        )
        
        # Set config attributes
        self.config = config
        self.num_labels = 2  # Binary: keep or prune
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[TokenClassifierOutput, tuple]:
        """
        Forward pass compatible with token classification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Ground truth token labels
            return_dict: Whether to return a dictionary
            
        Returns:
            TokenClassifierOutput or tuple
        """
        # Forward through PruningEncoder with correct arguments
        outputs = self.pruning_encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        
        # Extract pruning logits
        logits = outputs.get("pruning_logits")
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only compute loss on non-padding tokens
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs.get("hidden_states", ())
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
        # Similar to sequence classification
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            config = PruningEncoderConfig.from_pretrained(pretrained_model_name_or_path)
        else:
            pruning_config_path = os.path.join(pretrained_model_name_or_path, "pruning_encoder_config.json")
            if os.path.exists(pruning_config_path):
                with open(pruning_config_path, 'r') as f:
                    pruning_config = json.load(f)
                config = PruningEncoderConfig(
                    mode="pruning_only",
                    base_model_name_or_path=pretrained_model_name_or_path,
                    pruning_config=pruning_config.get("pruning_config", {}),
                    max_length=pruning_config.get("max_length", 512)
                )
            else:
                raise ValueError(f"No config found at {pretrained_model_name_or_path}")
        
        # Create model
        model = cls(config)
        
        # Load weights
        model.pruning_encoder = PruningEncoder.from_pretrained(
            pretrained_model_name_or_path,
            device=kwargs.get("device_map", None)
        )
        
        return model


def register_auto_models():
    """Register PruningEncoder models with AutoModel."""
    # Register config
    AutoConfig.register("pruning_encoder", PruningEncoderConfig)
    
    # Register models
    AutoModelForSequenceClassification.register(
        PruningEncoderConfig, 
        PruningEncoderForSequenceClassification
    )
    AutoModelForTokenClassification.register(
        PruningEncoderConfig,
        PruningEncoderForTokenClassification
    )
    
    # Also register with the base architecture name for CrossEncoder compatibility
    # This allows CrossEncoder to recognize our models
    if hasattr(AutoModelForSequenceClassification, '_model_mapping'):
        AutoModelForSequenceClassification._model_mapping.register(
            PruningEncoderConfig,
            PruningEncoderForSequenceClassification
        )
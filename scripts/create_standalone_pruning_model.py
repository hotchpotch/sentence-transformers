#!/usr/bin/env python3
"""
Script to create a standalone version of PruningEncoder models that can be loaded with AutoModel.

This script creates a self-contained Python file with all necessary classes and dependencies
for loading PruningEncoder models without requiring sentence_transformers to be installed.
"""

import os
import inspect
import ast
import textwrap
from pathlib import Path


def extract_class_source(cls):
    """Extract the source code of a class and its dependencies."""
    return inspect.getsource(cls)


def create_standalone_modeling_file():
    """Create a standalone modeling file with all PruningEncoder dependencies."""
    
    modeling_content = '''"""
Standalone PruningEncoder implementation for Hugging Face Transformers.

This file contains all necessary components to load PruningEncoder models
using AutoModel without requiring sentence_transformers to be installed.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    TokenClassifierOutput,
    BaseModelOutput,
)

logger = logging.getLogger(__name__)


# Data structures
@dataclass
class PruningConfig:
    """Configuration for the pruning head."""
    hidden_size: int = 768
    dropout: float = 0.1
    pooling_strategy: str = "cls"  # "cls", "mean", "max"
    score_combination: str = "multiply"  # "multiply", "add", "gated"
    use_bias: bool = True
    initialization_range: float = 0.02


@dataclass
class RerankingPruningOutput:
    """Output for reranking + pruning mode."""
    ranking_logits: torch.Tensor
    pruning_logits: torch.Tensor
    pruned_texts: Optional[List[str]] = None
    pruning_masks: Optional[torch.Tensor] = None
    token_scores: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


@dataclass  
class PruningOnlyOutput:
    """Output for pruning-only mode."""
    pruning_logits: torch.Tensor
    pruned_texts: Optional[List[str]] = None
    pruning_masks: Optional[torch.Tensor] = None
    token_scores: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


# PruningHead implementation
class PruningHeadConfig:
    """Configuration for PruningHead."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        dropout: float = 0.1,
        pooling_strategy: str = "cls",
        score_combination: str = "multiply",
        use_bias: bool = True,
        initialization_range: float = 0.02,
    ):
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pooling_strategy = pooling_strategy
        self.score_combination = score_combination
        self.use_bias = use_bias
        self.initialization_range = initialization_range
    
    def to_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "pooling_strategy": self.pooling_strategy,
            "score_combination": self.score_combination,
            "use_bias": self.use_bias,
            "initialization_range": self.initialization_range,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


class PruningHead(nn.Module):
    """Pruning head that generates token-level importance scores."""
    
    def __init__(self, config: PruningHeadConfig):
        super().__init__()
        self.config = config
        
        # Token-level scoring
        self.token_scorer = nn.Linear(config.hidden_size, 1, bias=config.use_bias)
        
        # Optional: Query-aware gating for score combination
        if config.score_combination == "gated":
            self.gate = nn.Linear(config.hidden_size * 2, 1, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_scorer.weight, std=self.config.initialization_range)
        if self.token_scorer.bias is not None:
            nn.init.zeros_(self.token_scorer.bias)
        
        if hasattr(self, "gate"):
            nn.init.normal_(self.gate.weight, std=self.config.initialization_range)
            if self.gate.bias is not None:
                nn.init.zeros_(self.gate.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_representation: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the pruning head.
        
        Args:
            hidden_states: Token embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            query_representation: Query representation for score combination
                                [batch_size, hidden_size]
        
        Returns:
            Dictionary containing:
                - token_scores: Token importance scores [batch_size, seq_len]
                - pruning_logits: Binary classification logits [batch_size, seq_len, 2]
        """
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Compute token scores
        token_scores = self.token_scorer(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        # Mask padding tokens
        if attention_mask is not None:
            token_scores = token_scores.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # Optional: Combine with query representation
        if query_representation is not None and self.config.score_combination == "gated":
            # Expand query representation to match token dimensions
            expanded_query = query_representation.unsqueeze(1).expand_as(hidden_states)
            
            # Concatenate and compute gate
            combined = torch.cat([hidden_states, expanded_query], dim=-1)
            gate_scores = torch.sigmoid(self.gate(combined).squeeze(-1))
            
            # Apply gating
            token_scores = token_scores * gate_scores
        
        # Convert to binary classification logits (keep vs prune)
        # Higher score = more likely to keep
        pruning_logits = torch.stack([
            -token_scores,  # Logit for "prune" class
            token_scores    # Logit for "keep" class
        ], dim=-1)
        
        return {
            "token_scores": token_scores,
            "pruning_logits": pruning_logits
        }
    
    def save_pretrained(self, save_directory: Union[str, Path]):
        """Save the pruning head."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(save_directory / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save weights
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]):
        """Load a pretrained pruning head."""
        model_path = Path(model_path)
        
        # Load config
        with open(model_path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = PruningHeadConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model


# Main configuration class
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


# Sequence classification wrapper
class PruningEncoderForSequenceClassification(PreTrainedModel):
    """
    PruningEncoder wrapper for sequence classification (reranking).
    Compatible with AutoModelForSequenceClassification.
    """
    
    config_class = PruningEncoderConfig
    base_model_prefix = "pruning_encoder"
    
    def __init__(self, config: PruningEncoderConfig):
        super().__init__(config)
        
        # For this standalone version, we'll load the base model directly
        from transformers import AutoModelForSequenceClassification as AutoModel
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(
            config.base_model_name_or_path,
            num_labels=1 if config.mode == "reranking_pruning" else config.num_labels
        )
        
        # Initialize pruning head
        pruning_config = PruningHeadConfig(**config.pruning_config)
        self.pruning_head = PruningHead(pruning_config)
        
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
        output_hidden_states: bool = True,
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
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # Get hidden states
        hidden_states = outputs.hidden_states[-1]
        
        # Get query representation (CLS token)
        query_representation = hidden_states[:, 0, :]
        
        # Forward through pruning head
        pruning_outputs = self.pruning_head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            query_representation=query_representation
        )
        
        # Extract ranking logits
        logits = outputs.logits
        
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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from a PruningEncoder checkpoint."""
        # Load config
        config = PruningEncoderConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Create model
        model = cls(config)
        
        # Load pruning head weights if available
        pruning_head_path = Path(pretrained_model_name_or_path) / "pruning_head"
        if pruning_head_path.exists():
            model.pruning_head = PruningHead.from_pretrained(pruning_head_path)
        
        # Load base model weights if saved separately
        base_model_path = Path(pretrained_model_name_or_path) / "ranking_model"
        if base_model_path.exists():
            from transformers import AutoModelForSequenceClassification
            model.base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_path,
                num_labels=config.num_labels
            )
        
        return model


# Token classification wrapper  
class PruningEncoderForTokenClassification(PreTrainedModel):
    """
    PruningEncoder wrapper for token classification (pruning).
    Compatible with AutoModelForTokenClassification.
    """
    
    config_class = PruningEncoderConfig
    base_model_prefix = "pruning_encoder"
    
    def __init__(self, config: PruningEncoderConfig):
        super().__init__(config)
        
        # For pruning-only mode, we use a base encoder
        from transformers import AutoModel
        
        # Load base encoder
        self.base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
        
        # Initialize pruning head  
        pruning_config = PruningHeadConfig(**config.pruning_config)
        self.pruning_head = PruningHead(pruning_config)
        
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
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state
        
        # Forward through pruning head
        pruning_outputs = self.pruning_head(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        
        # Extract pruning logits
        logits = pruning_outputs["pruning_logits"]
        
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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from a PruningEncoder checkpoint."""
        # Load config
        config = PruningEncoderConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Create model
        model = cls(config)
        
        # Load pruning head weights
        pruning_head_path = Path(pretrained_model_name_or_path) / "pruning_head"
        if pruning_head_path.exists():
            model.pruning_head = PruningHead.from_pretrained(pruning_head_path)
        
        # Load base model weights if saved separately
        encoder_model_path = Path(pretrained_model_name_or_path) / "encoder_model"
        if encoder_model_path.exists():
            from transformers import AutoModel
            model.base_model = AutoModel.from_pretrained(encoder_model_path)
        
        return model


# Register the configuration and models
AutoConfig.register("pruning_encoder", PruningEncoderConfig)
'''
    
    return modeling_content


def update_save_pretrained_method():
    """Create an updated save_pretrained method that includes the standalone modeling file."""
    
    updated_method = '''
def save_pretrained_with_standalone(self, save_directory: Union[str, Path]) -> None:
    """
    Save the model to a directory with standalone modeling file for AutoModel compatibility.
    
    This method extends the original save_pretrained to include a standalone modeling file
    that enables loading the model with AutoModel without requiring sentence_transformers.
    """
    # Call original save_pretrained
    self.save_pretrained_original(save_directory)
    
    # Create standalone modeling file
    from scripts.create_standalone_pruning_model import create_standalone_modeling_file
    modeling_content = create_standalone_modeling_file()
    
    save_directory = Path(save_directory)
    with open(save_directory / "modeling_pruning_encoder.py", "w") as f:
        f.write(modeling_content)
    
    # Update auto_map in config.json to use the standalone file
    config_path = save_directory / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Update auto_map paths to use the local modeling file
    if "auto_map" in config:
        config["auto_map"] = {
            "AutoConfig": "modeling_pruning_encoder.PruningEncoderConfig",
            "AutoModelForSequenceClassification": "modeling_pruning_encoder.PruningEncoderForSequenceClassification" if config.get("mode") == "reranking_pruning" else None,
            "AutoModelForTokenClassification": "modeling_pruning_encoder.PruningEncoderForTokenClassification" if config.get("mode") == "pruning_only" else None,
        }
        # Remove None values
        config["auto_map"] = {k: v for k, v in config["auto_map"].items() if v is not None}
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
'''
    
    return updated_method


if __name__ == "__main__":
    # Create the standalone modeling file
    content = create_standalone_modeling_file()
    
    # Save to a file
    output_path = Path("modeling_pruning_encoder_standalone.py")
    with open(output_path, "w") as f:
        f.write(content)
    
    print(f"Created standalone modeling file: {output_path}")
    print("\nTo use this file:")
    print("1. Copy it to your saved model directory as 'modeling_pruning_encoder.py'")
    print("2. Update the auto_map in config.json to reference local classes:")
    print('   "auto_map": {')
    print('     "AutoConfig": "modeling_pruning_encoder.PruningEncoderConfig",')
    print('     "AutoModelForSequenceClassification": "modeling_pruning_encoder.PruningEncoderForSequenceClassification"')
    print('   }')
    print("\n3. Load with AutoModel:")
    print("   from transformers import AutoModelForSequenceClassification")
    print("   model = AutoModelForSequenceClassification.from_pretrained(")
    print('       "path/to/model",')
    print("       trust_remote_code=True")
    print("   )")
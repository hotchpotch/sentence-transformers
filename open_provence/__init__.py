"""
Query-dependent text pruning and reranking for efficient RAG pipelines.

This module provides functionality for pruning irrelevant content from documents
based on queries, with optional reranking capabilities.
"""

from .data_structures import OpenProvenceConfig, RerankingOpenProvenceOutput, OpenProvenceOutput, OpenProvenceOnlyOutput
from .encoder import OpenProvenceEncoder
from .trainer import OpenProvenceTrainer
from .losses import OpenProvenceLoss
from .data_collator import OpenProvenceDataCollator

# Import Transformers compatibility components
try:
    from .transformers_compat import (
        OpenProvenceEncoderConfig,
        OpenProvenceEncoderForSequenceClassification,
        OpenProvenceEncoderForTokenClassification,
        register_auto_models
    )
    
    # Register models with Transformers AutoModel
    register_auto_models()
    
    __all__ = [
        "OpenProvenceConfig",
        "RerankingOpenProvenceOutput", 
        "OpenProvenceOutput",
        "OpenProvenceOnlyOutput",
        "OpenProvenceEncoder",
        "OpenProvenceTrainer",
        "OpenProvenceLoss",
        "OpenProvenceDataCollator",
        "OpenProvenceEncoderConfig",
        "OpenProvenceEncoderForSequenceClassification",
        "OpenProvenceEncoderForTokenClassification"
    ]
except ImportError:
    # Transformers compatibility not available
    __all__ = [
        "OpenProvenceConfig",
        "RerankingOpenProvenceOutput", 
        "OpenProvenceOutput",
        "OpenProvenceOnlyOutput",
        "OpenProvenceEncoder",
        "OpenProvenceTrainer",
        "OpenProvenceLoss",
        "OpenProvenceDataCollator"
    ]
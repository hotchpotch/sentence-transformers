"""
Query-dependent text pruning and reranking for efficient RAG pipelines.

This module provides functionality for pruning irrelevant content from documents
based on queries, with optional reranking capabilities.
"""

from .data_structures import PruningConfig, RerankingPruningOutput, PruningOutput, PruningOnlyOutput
from .encoder import PruningEncoder
from .trainer import PruningTrainer
from .losses import PruningLoss
from .data_collator import PruningDataCollator

# Import Transformers compatibility components
try:
    from .transformers_compat import (
        PruningEncoderConfig,
        PruningEncoderForSequenceClassification,
        PruningEncoderForTokenClassification,
        register_auto_models
    )
    
    # Register models with Transformers AutoModel
    register_auto_models()
    
    __all__ = [
        "PruningConfig",
        "RerankingPruningOutput", 
        "PruningOutput",
        "PruningOnlyOutput",
        "PruningEncoder",
        "PruningTrainer",
        "PruningLoss",
        "PruningDataCollator",
        "PruningEncoderConfig",
        "PruningEncoderForSequenceClassification",
        "PruningEncoderForTokenClassification"
    ]
except ImportError:
    # Transformers compatibility not available
    __all__ = [
        "PruningConfig",
        "RerankingPruningOutput", 
        "PruningOutput",
        "PruningOnlyOutput",
        "PruningEncoder",
        "PruningTrainer",
        "PruningLoss",
        "PruningDataCollator"
    ]
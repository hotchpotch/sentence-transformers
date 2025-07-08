"""
Provence: Query-dependent text pruning for efficient RAG pipelines.
"""

from .data_structures import ProvenceConfig, ProvenceOutput
from .encoder import ProvenceEncoder
from .trainer import ProvenceTrainer
from .losses import ProvenceLoss
from .data_collator_chunk_based import ProvenceChunkBasedDataCollator

__all__ = [
    "ProvenceConfig",
    "ProvenceOutput", 
    "ProvenceEncoder",
    "ProvenceTrainer",
    "ProvenceLoss",
    "ProvenceChunkBasedDataCollator"
]
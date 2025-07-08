"""
Provence: Query-dependent text pruning for efficient RAG pipelines.
"""

from .data_structures import ProvenceConfig, ProvenceOutput, ProvenceContextOutput
from .encoder import ProvenceEncoder
from .trainer import ProvenceTrainer
from .losses import ProvenceLoss
from .data_collator import ProvenceDataCollator

__all__ = [
    "ProvenceConfig",
    "ProvenceOutput",
    "ProvenceContextOutput",
    "ProvenceEncoder",
    "ProvenceTrainer",
    "ProvenceLoss",
    "ProvenceDataCollator"
]
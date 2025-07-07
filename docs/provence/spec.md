# Provence Implementation Specification

## Overview
Provence is a query-dependent text pruning method for RAG pipelines that reduces irrelevant context while preserving relevant information.

## Architecture
- **ProvenceEncoder**: Standalone encoder that performs both ranking and pruning
- **Dual-head design**: Reranking head + Pruning head
- **Sentence-level pruning**: Prunes at sentence boundaries for better coherence

## Base Model
During development, we use `hotchpotch/japanese-reranker-xsmall-v2` as the base model:
- This is a CrossEncoder model optimized for Japanese text
- Supports large batch sizes (up to 512)
- Used for generating teacher scores for training data
- **Important**: Do not switch to other models without explicit approval

## Training Strategy
1. Teacher score distillation from `hotchpotch/japanese-reranker-xsmall-v2`
2. Joint training of ranking and pruning objectives
3. Progressive dataset sizes: minimal (5k) → small (50k) → full (1.3M)

## Implementation Notes

### Current Status
- Basic structure created as ProvenceEncoder (standalone from CrossEncoder)
- Includes ranking and pruning capabilities
- Uses sentence-level pruning with multilingual text chunking support
- Comprehensive unit tests written and passing

### Key Components
- `ProvenceEncoder`: Main encoder class
- `ProvenceTrainer`: Custom trainer for the model
- `ProvenceLoss`: Joint loss function
- `ProvenceDataCollator`: Data collator with sentence boundary handling
- `ProvencePruningHead`: Pruning classification head
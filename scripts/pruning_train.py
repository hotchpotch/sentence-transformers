#!/usr/bin/env python3
"""
Flexible training script for PruningEncoder with teacher score distillation.
Supports various configurations through command line arguments and config files.

Usage:
    # Using YAML config file
    python scripts/pruning_train.py pruning-config/train-models/japanese-reranker-xsmall-v2.yaml
    
    # Using command line arguments
    python scripts/pruning_train.py \
        --model_name_or_path hotchpotch/japanese-reranker-xsmall-v2 \
        --dataset_name hotchpotch/wip-msmarco-context-relevance \
        --subset msmarco-ja-minimal \
        --output_dir ./output/pruning-models/minimal
    
    # Mix config file with command line overrides
    python scripts/pruning_train.py pruning-config/train-models/japanese-reranker-xsmall-v2.yaml \
        --subset msmarco-ja-small \
        --num_train_epochs 1
    
    # Pruning-only mode
    python scripts/pruning_train.py \
        --model_name_or_path cl-nagoya/ruri-v3-30m \
        --mode pruning_only \
        --subset msmarco-ja-minimal \
        --output_dir ./output/ruri-v3-30m_pruning-only_minimal_20250111_123456

Recommended output_dir format: `./output/{model-name}_{mode}_{subset}_{YYYYMMDD_HHMMSS}/`
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    AutoTokenizer
)
from transformers.trainer import Trainer

from sentence_transformers.pruning import (
    PruningEncoder,
    PruningTrainer,
    PruningLoss,
    PruningDataCollator
)

try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    model_name_or_path: str = field(
        default="hotchpotch/japanese-reranker-xsmall-v2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    mode: str = field(
        default="reranking_pruning",
        metadata={"help": "Model mode: 'reranking_pruning' or 'pruning_only'"}
    )
    classifier_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for classifier heads"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    dataset_name: str = field(
        default="hotchpotch/wip-msmarco-context-relevance",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    subset: str = field(
        default="msmarco-ja-minimal",
        metadata={"help": "Dataset subset to use"}
    )
    teacher_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model name for score column. If None, extracted from config path."}
    )
    datasets: Optional[List[Dict[str, str]]] = field(
        default=None,
        metadata={"help": "List of datasets with their teacher columns for multi-dataset training"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of evaluation examples"}
    )
    validation_split: Optional[float] = field(
        default=None,
        metadata={"help": "Validation split ratio (0-1). If None, use existing validation set"}
    )
    validation_split_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of validation samples to split. Takes precedence over validation_split ratio"}
    )
    validation_split_name: str = field(
        default="validation",
        metadata={"help": "Name of the validation split to use (e.g., 'validation', 'test', 'dev')"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )


@dataclass
class PruningTrainingArguments(TrainingArguments):
    """Training arguments specific to PruningEncoder training."""
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for model and checkpoints. Format example: ./output/japanese-reranker-xsmall-v2_reranking-pruning_minimal_20250111_123456"}
    )
    ranking_weight: float = field(
        default=0.05,
        metadata={"help": "Weight for ranking loss (Provence paper default: 0.05)"}
    )
    pruning_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for pruning loss (Provence paper default: 1.0)"}
    )
    use_teacher_scores: bool = field(
        default=True,
        metadata={"help": "Whether to use teacher scores for distillation"}
    )
    sentence_level_pruning: bool = field(
        default=True,
        metadata={"help": "Whether to perform sentence-level pruning"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove columns not required by the model"}
    )
    # Override some defaults
    per_device_train_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    optim: str = field(
        default="adafactor",
        metadata={"help": "The optimizer to use."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training."}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."}
    )


def prepare_dataset(
    data_args: DataArguments,
    teacher_model_name: str,
    seed: int = 42
) -> Tuple[Any, Any]:
    """
    Load dataset - let PruningDataCollator handle the processing.
    
    Args:
        data_args: Data arguments containing dataset info
        teacher_model_name: Name of the teacher model for score column (unused, passed to DataCollator)
        seed: Random seed for splitting
    """
    # Check if we have multiple datasets
    if data_args.datasets:
        logger.info(f"Loading {len(data_args.datasets)} datasets for concatenation")
        train_datasets = []
        eval_datasets = []
        
        for dataset_config in data_args.datasets:
            dataset_name = dataset_config.get('dataset_name')
            subset = dataset_config.get('subset')
            teacher_column = dataset_config.get('teacher_column', f'teacher_scores.{teacher_model_name}')
            
            logger.info(f"Loading dataset: {dataset_name}:{subset}")
            dataset = load_dataset(dataset_name, subset)
            
            # Process train dataset
            train_ds = dataset['train']
            
            # Rename teacher column to unified name
            if teacher_column != 'teacher_score' and teacher_column in train_ds.column_names:
                logger.info(f"Renaming {teacher_column} to teacher_score")
                train_ds = train_ds.rename_column(teacher_column, 'teacher_score')
            
            train_datasets.append(train_ds)
            
            # Process eval dataset if exists
            if 'validation' in dataset:
                eval_ds = dataset['validation']
                if teacher_column != 'teacher_score' and teacher_column in eval_ds.column_names:
                    eval_ds = eval_ds.rename_column(teacher_column, 'teacher_score')
                eval_datasets.append(eval_ds)
            elif 'test' in dataset:
                eval_ds = dataset['test']
                if teacher_column != 'teacher_score' and teacher_column in eval_ds.column_names:
                    eval_ds = eval_ds.rename_column(teacher_column, 'teacher_score')
                eval_datasets.append(eval_ds)
        
        # Get common columns for concatenation
        if train_datasets:
            # Find common columns across all datasets
            common_columns = set(train_datasets[0].column_names)
            for ds in train_datasets[1:]:
                common_columns = common_columns.intersection(set(ds.column_names))
            
            # Ensure required columns are present
            required_columns = {'query', 'positive', 'negative', 'context_spans', 'context_spans_relevance', 'teacher_score'}
            
            # Filter to common columns that actually exist
            existing_columns = []
            for col in common_columns:
                if all(col in ds.column_names for ds in train_datasets):
                    existing_columns.append(col)
            
            # Make sure we have the essential columns
            essential_columns = ['query', 'positive', 'negative', 'teacher_score']
            for col in essential_columns:
                if col not in existing_columns and all(col in ds.column_names for ds in train_datasets):
                    existing_columns.append(col)
            
            # Add context columns if they exist in all datasets
            for col in ['context_spans', 'context_spans_relevance']:
                if col not in existing_columns and all(col in ds.column_names for ds in train_datasets):
                    existing_columns.append(col)
            
            logger.info(f"Using columns: {sorted(existing_columns)}")
            train_datasets = [ds.select_columns(existing_columns) for ds in train_datasets]
            
            # Concatenate datasets
            train_dataset = concatenate_datasets(train_datasets)
            logger.info(f"Concatenated train dataset size: {len(train_dataset)}")
        else:
            train_dataset = None
            
        # Handle eval datasets
        if eval_datasets:
            # Filter to same columns as train datasets
            eval_datasets = [ds.select_columns(existing_columns) for ds in eval_datasets if all(col in ds.column_names for col in existing_columns)]
            eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
        else:
            eval_dataset = None
            
    else:
        # Single dataset mode (backward compatibility)
        logger.info(f"Loading dataset: {data_args.dataset_name}:{data_args.subset}")
        dataset = load_dataset(data_args.dataset_name, data_args.subset)
        
        # Get raw train dataset (let DataCollator handle processing)
        train_dataset = dataset['train']
        eval_dataset = None
    
    # Continue with validation set handling for single dataset mode
    if not data_args.datasets:
        # Try to use existing validation set first (prioritize validation over test)
        if data_args.validation_split_name in dataset:
            logger.info(f"Using existing validation set: '{data_args.validation_split_name}'")
            eval_dataset = dataset[data_args.validation_split_name]
        elif 'validation' in dataset:
            logger.info("Using existing validation set: 'validation'")
            eval_dataset = dataset['validation']
    
    # If no validation set found or split is explicitly specified, create split
    if not data_args.datasets and (eval_dataset is None or data_args.validation_split is not None or data_args.validation_split_samples is not None):
        if data_args.validation_split_samples is not None:
            # Use absolute number of samples
            if data_args.validation_split_samples <= 0 or data_args.validation_split_samples >= len(train_dataset):
                raise ValueError(f"validation_split_samples must be between 1 and {len(train_dataset)-1}")
            logger.info(f"Creating validation split with {data_args.validation_split_samples} samples")
            ratio = data_args.validation_split_samples / len(train_dataset)
            split_dataset = train_dataset.train_test_split(test_size=ratio, seed=seed)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        elif data_args.validation_split is not None:
            # Use ratio
            if not (0 < data_args.validation_split < 1):
                raise ValueError("validation_split must be between 0 and 1")
            logger.info(f"Creating validation split with {data_args.validation_split:.0%} of training data")
            split_dataset = train_dataset.train_test_split(test_size=data_args.validation_split, seed=seed)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        else:
            # No validation set available and no split specified
            logger.warning("No validation set found and no validation_split specified. Training without validation.")
            eval_dataset = None
    
    # Apply sampling if specified
    if data_args.max_train_samples and len(train_dataset) > data_args.max_train_samples:
        logger.info(f"Sampling {data_args.max_train_samples} training examples from {len(train_dataset)}")
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    
    if eval_dataset is not None and data_args.max_eval_samples and len(eval_dataset) > data_args.max_eval_samples:
        logger.info(f"Sampling {data_args.max_eval_samples} evaluation examples from {len(eval_dataset)}")
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset) if eval_dataset is not None else 0}")
    
    return train_dataset, eval_dataset


class PruningHfTrainer(Trainer):
    """Custom Trainer that uses PruningTrainer internally for compatibility."""
    
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self._eval_loss_components = {}
    
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
        
        # Log individual loss components if available
        if hasattr(self.loss_fn, 'last_loss_components') and self.loss_fn.last_loss_components:
            # Create metrics to log
            metrics = {}
            for name, value in self.loss_fn.last_loss_components.items():
                if isinstance(value, torch.Tensor):
                    metrics[f'{name}'] = value.item()
                else:
                    metrics[f'{name}'] = value
            
            # Use self.log to properly log metrics (will be picked up by wandb)
            if hasattr(self, 'log'):
                self.log(metrics)
            
            # Also add to log_history for direct access
            if hasattr(self.state, 'log_history') and self.state.log_history is not None:
                # Find or create the current step's log entry
                current_step = self.state.global_step
                if len(self.state.log_history) > 0 and self.state.log_history[-1].get('step') == current_step:
                    # Update existing entry
                    self.state.log_history[-1].update(metrics)
                else:
                    # Create new entry
                    log_entry = {
                        'step': current_step,
                        'epoch': self.state.epoch if hasattr(self.state, 'epoch') else 0,
                        **metrics
                    }
                    self.state.log_history.append(log_entry)
        
        return (loss, None) if return_outputs else loss
    
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
            
            # Log eval loss components
            if hasattr(self, 'log'):
                eval_metrics = {f'eval_{name}': sum(values) / len(values) 
                               for name, values in self._eval_loss_components.items()}
                self.log(eval_metrics)
        
        return output


def calculate_dynamic_steps(
    dataset_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: float,
    num_devices: int = 1,
    target_eval_points: int = 20,
    target_log_points: int = 100
) -> Tuple[int, int]:
    """
    Calculate dynamic eval_steps and logging_steps based on dataset size and training config.
    
    Args:
        dataset_size: Number of training samples
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs
        num_devices: Number of devices (for distributed training)
        target_eval_points: Target number of evaluation points
        target_log_points: Target number of logging points
        
    Returns:
        Tuple of (eval_steps, logging_steps)
    """
    # Calculate total steps
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_devices
    steps_per_epoch = dataset_size // effective_batch_size
    total_steps = int(steps_per_epoch * num_epochs)
    
    # Calculate dynamic steps
    eval_steps = max(1, total_steps // target_eval_points)
    logging_steps = max(1, total_steps // target_log_points)
    
    # Ensure logging is more frequent than eval
    if logging_steps > eval_steps:
        logging_steps = max(1, eval_steps // 2)
    
    return eval_steps, logging_steps, total_steps


def parse_config_file(config_file: str) -> Tuple[ModelArguments, DataArguments, PruningTrainingArguments]:
    """Parse YAML configuration file and convert to dataclass arguments."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model arguments
    model_config = config.get('model_args', {})
    model_args = ModelArguments(
        model_name_or_path=model_config.get('model_name_or_path', 'hotchpotch/japanese-reranker-xsmall-v2'),
        mode=model_config.get('mode', 'reranking_pruning'),
        classifier_dropout=model_config.get('classifier_dropout', 0.1),
        max_length=model_config.get('max_length', 512),
        config_name=model_config.get('config_name', None),
        tokenizer_name=model_config.get('tokenizer_name', None),
        cache_dir=model_config.get('cache_dir', None)
    )
    
    # Extract data arguments
    data_config = config.get('data_args', {})
    data_args = DataArguments(
        dataset_name=data_config.get('dataset_name', 'hotchpotch/wip-msmarco-context-relevance'),
        subset=data_config.get('subset', 'msmarco-ja-minimal'),
        teacher_model_name=data_config.get('teacher_model_name', None),
        max_train_samples=data_config.get('max_train_samples', None),
        max_eval_samples=data_config.get('max_eval_samples', None),
        validation_split=data_config.get('validation_split', None),
        validation_split_samples=data_config.get('validation_split_samples', None),
        validation_split_name=data_config.get('validation_split_name', 'validation'),
        preprocessing_num_workers=data_config.get('preprocessing_num_workers', None),
        datasets=data_config.get('datasets', None)
    )
    
    # Extract training arguments
    training_config = config.get('training_args', {})
    # Ensure evaluation strategy matches save strategy when load_best_model_at_end is True
    load_best_model = training_config.get('load_best_model_at_end', True)
    
    # Note: eval_steps and logging_steps will be calculated dynamically
    # Remove them from config to avoid confusion
    eval_steps = training_config.get('eval_steps', None)
    logging_steps = training_config.get('logging_steps', None)
    save_steps = training_config.get('save_steps', None)
    
    training_args = PruningTrainingArguments(
        output_dir=training_config.get('output_dir', None),  # Optional, will be auto-generated if not provided
        overwrite_output_dir=training_config.get('overwrite_output_dir', True),
        num_train_epochs=training_config.get('num_train_epochs', 1),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 32),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 16),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 2),
        learning_rate=training_config.get('learning_rate', 5e-5),
        weight_decay=training_config.get('weight_decay', 0.01),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        # Dynamic steps will be set later
        logging_steps=logging_steps or 100,  # Temporary default
        save_steps=save_steps or 500,  # Temporary default
        eval_steps=eval_steps or 500,  # Temporary default
        eval_strategy="steps" if load_best_model else "no",  # Enable evaluation if needed
        save_total_limit=training_config.get('save_total_limit', 5),
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', True),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 8),
        optim=training_config.get('optimizer', training_config.get('optim', 'adafactor')),
        report_to=training_config.get('report_to', ['wandb']),
    )
    
    # Store original config values for reference
    training_args._original_eval_steps = eval_steps
    training_args._original_logging_steps = logging_steps
    training_args._original_save_steps = save_steps
    
    return model_args, data_args, training_args


def main():
    # Parse arguments - either from command line or config files
    parser = HfArgumentParser((ModelArguments, DataArguments, PruningTrainingArguments))
    
    # Check if first argument is a config file
    config_file_arg = None
    if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".yml", ".json")):
        config_file_arg = sys.argv[1]
        # Remove config file from argv to parse remaining args
        remaining_args = sys.argv[2:]
    else:
        remaining_args = sys.argv[1:]
    
    if config_file_arg:
        # Parse config file first
        print(f"Loading configuration from: {config_file_arg}")
        if config_file_arg.endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(json_file=config_file_arg)
        else:
            model_args, data_args, training_args = parse_config_file(config_file_arg)
        
        # Parse remaining command line arguments to override config
        if remaining_args:
            # Create temporary argv for remaining args
            temp_argv = [sys.argv[0]] + remaining_args
            original_argv = sys.argv
            sys.argv = temp_argv
            try:
                override_model_args, override_data_args, override_training_args = parser.parse_args_into_dataclasses()
                
                # Override config values with command line values (only non-default values)
                overrides = []
                
                for field_name in model_args.__dataclass_fields__:
                    override_value = getattr(override_model_args, field_name)
                    default_value = model_args.__dataclass_fields__[field_name].default
                    # Handle special case for required fields that don't have real defaults
                    if field_name == 'model_name_or_path' and hasattr(override_model_args, field_name):
                        # Always override model_name_or_path if provided
                        old_value = getattr(model_args, field_name)
                        if old_value != override_value:
                            setattr(model_args, field_name, override_value)
                            overrides.append(f"model_args.{field_name}: {old_value} → {override_value}")
                    elif override_value != default_value:
                        old_value = getattr(model_args, field_name)
                        setattr(model_args, field_name, override_value)
                        overrides.append(f"model_args.{field_name}: {old_value} → {override_value}")
                
                for field_name in data_args.__dataclass_fields__:
                    override_value = getattr(override_data_args, field_name)
                    default_value = data_args.__dataclass_fields__[field_name].default
                    if override_value != default_value:
                        old_value = getattr(data_args, field_name)
                        setattr(data_args, field_name, override_value)
                        overrides.append(f"data_args.{field_name}: {old_value} → {override_value}")
                
                for field_name in training_args.__dataclass_fields__:
                    override_value = getattr(override_training_args, field_name)
                    default_value = training_args.__dataclass_fields__[field_name].default
                    if override_value != default_value:
                        old_value = getattr(training_args, field_name)
                        setattr(training_args, field_name, override_value)
                        overrides.append(f"training_args.{field_name}: {old_value} → {override_value}")
                
                # Log overrides
                if overrides:
                    print("Command line overrides:")
                    for override in overrides:
                        print(f"  {override}")
                else:
                    print("No command line overrides applied.")
                        
            finally:
                sys.argv = original_argv
        else:
            print("Using configuration file settings.")
    else:
        # No config file, parse command line only
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Create timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set output_dir if not specified or using default
    if not training_args.output_dir or training_args.output_dir == "trainer_output":
        # Generate default output_dir with timestamp
        if config_file_arg:
            # Use config file name as base
            config_base = Path(config_file_arg).stem
            output_dir = f"./output/{config_base}_{timestamp}"
        else:
            # Fallback to old naming scheme
            model_name = Path(model_args.model_name_or_path).name
            if data_args.datasets:
                # For multi-dataset configs, use a generic name
                output_dir = f"./output/{model_name}_{model_args.mode}_multi-dataset_{timestamp}"
            else:
                output_dir = f"./output/{model_name}_{model_args.mode}_{data_args.subset}_{timestamp}"
        
        training_args.output_dir = output_dir
        print(f"\n{'='*80}")
        print(f"No output_dir specified. Auto-generated output directory:")
        print(f"  {output_dir}")
        print(f"{'='*80}\n")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Output directory: {training_args.output_dir}")
    
    # Set seed
    set_seed(training_args.seed)
    
    # Initialize WandB if available
    if _wandb_available and "wandb" in training_args.report_to:
        # Set WandB project name
        os.environ["WANDB_PROJECT"] = "pruning"
        
        # Create run name based on configuration with timestamp
        if config_file_arg:
            # Use config file name as base for run name
            config_base = Path(config_file_arg).stem
            run_name = f"{config_base}-{timestamp}"
        else:
            # Fallback to old naming scheme
            model_name = Path(model_args.model_name_or_path).name
            if data_args.datasets:
                run_name = f"{model_name}-{model_args.mode}-multi-dataset-{timestamp}"
            else:
                run_name = f"{model_name}-{model_args.mode}-{data_args.subset}-{timestamp}"
        
        wandb.init(
            project="pruning",
            name=run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "mode": model_args.mode,
                "dataset": data_args.dataset_name,
                "subset": data_args.subset,
                "num_epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "optim": training_args.optim,
                "ranking_weight": training_args.ranking_weight,
                "pruning_weight": training_args.pruning_weight,
                "timestamp": timestamp,
            }
        )
    else:
        if config_file_arg:
            # Use config file name as base for run name
            config_base = Path(config_file_arg).stem
            run_name = f"{config_base}-{timestamp}"
        else:
            # Fallback to old naming scheme
            model_name = Path(model_args.model_name_or_path).name
            if data_args.datasets:
                run_name = f"{model_name}-{model_args.mode}-multi-dataset-{timestamp}"
            else:
                run_name = f"{model_name}-{model_args.mode}-{data_args.subset}-{timestamp}"
    
    # Extract teacher model name
    if data_args.teacher_model_name:
        teacher_model_name = data_args.teacher_model_name
    else:
        # Default to japanese-reranker-xsmall-v2
        teacher_model_name = "japanese-reranker-xsmall-v2"
    logger.info(f"Using teacher model: {teacher_model_name}")
    
    # Set TrainingArguments run_name to match WandB
    training_args.run_name = run_name
    
    # Load dataset with teacher scores
    train_dataset, eval_dataset = prepare_dataset(
        data_args=data_args,
        teacher_model_name=teacher_model_name,
        seed=training_args.seed
    )
    
    # Calculate dynamic steps based on dataset size
    eval_steps, logging_steps, total_steps = calculate_dynamic_steps(
        dataset_size=len(train_dataset),
        per_device_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_epochs=training_args.num_train_epochs,
        num_devices=training_args.n_gpu if training_args.n_gpu > 0 else 1
    )
    
    # Check if we're overriding original config values and warn
    original_eval_steps = getattr(training_args, '_original_eval_steps', None)
    original_logging_steps = getattr(training_args, '_original_logging_steps', None)
    original_save_steps = getattr(training_args, '_original_save_steps', None)
    
    if original_eval_steps and original_eval_steps != eval_steps:
        logger.warning(f"Overriding eval_steps from config ({original_eval_steps}) with dynamic value ({eval_steps})")
    if original_logging_steps and original_logging_steps != logging_steps:
        logger.warning(f"Overriding logging_steps from config ({original_logging_steps}) with dynamic value ({logging_steps})")
    
    # Update training arguments with dynamic values
    training_args.eval_steps = eval_steps
    training_args.logging_steps = logging_steps
    training_args.save_steps = original_save_steps or eval_steps  # Use same as eval_steps if not specified
    
    # Enable evaluation if we have eval dataset
    if eval_dataset is not None:
        training_args.eval_strategy = "steps"
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
    else:
        # Disable evaluation if no eval dataset
        training_args.eval_strategy = "no"
        training_args.load_best_model_at_end = False
    
    logger.info(f"Dynamic step calculation:")
    logger.info(f"  Dataset size: {len(train_dataset):,}")
    logger.info(f"  Total steps: {total_steps:,}")
    logger.info(f"  Eval steps: {eval_steps} (20 evaluations)")
    logger.info(f"  Logging steps: {logging_steps} (100 logs)")
    logger.info(f"  Save steps: {training_args.save_steps}")
    
    # Initialize PruningEncoder
    logger.info(f"Initializing PruningEncoder with {model_args.model_name_or_path} in {model_args.mode} mode")
    model = PruningEncoder(
        model_name_or_path=model_args.model_name_or_path,
        num_labels=1,  # For regression task
        max_length=model_args.max_length,
        mode=model_args.mode,
        pruning_config={
            "dropout": model_args.classifier_dropout,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False
        }
    )
    
    # Create data collator with teacher score column and correct column names
    if data_args.datasets:
        # When using multiple datasets, we've renamed to 'teacher_score'
        teacher_score_column = "teacher_score"
    else:
        teacher_score_column = f"teacher_scores.{teacher_model_name}"
    logger.info(f"Using teacher score column: {teacher_score_column}")
    data_collator = PruningDataCollator(
        tokenizer=model.tokenizer,
        max_length=model.max_length,
        mode=model.mode,
        scores_column=teacher_score_column,     # Use specific teacher score column
        chunks_pos_column='context_spans',      # Use dataset's column name
        relevant_chunks_column='context_spans_relevance'  # Use dataset's column name
    )
    
    # Create loss function
    loss_fn = PruningLoss(
        model=model,
        mode=model.mode,
        ranking_weight=training_args.ranking_weight,
        pruning_weight=training_args.pruning_weight,
        is_regression=True  # Regression task for teacher score distillation
    )
    
    # Decide whether to use HF Trainer or PruningTrainer
    # Always use HF Trainer for better compatibility and feature support
    use_hf_trainer = True
    
    if use_hf_trainer:
        # Use HuggingFace Trainer with custom compute_loss
        logger.info("Using HuggingFace Trainer")
        trainer = PruningHfTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            loss_fn=loss_fn,
        )
        
        # Enable evaluation loss calculation
        training_args.prediction_loss_only = False
    else:
        # Convert TrainingArguments to dict for PruningTrainer
        training_args_dict = {
            "output_dir": training_args.output_dir,
            "num_epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "warmup_ratio": training_args.warmup_ratio,
            "weight_decay": training_args.weight_decay,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "max_grad_norm": training_args.max_grad_norm,
            "logging_steps": training_args.logging_steps,
            "eval_steps": training_args.eval_steps,
            "save_steps": training_args.save_steps,
            "save_total_limit": training_args.save_total_limit,
            "seed": training_args.seed,
            "fp16": training_args.fp16,
            "bf16": training_args.bf16,
            "dataloader_num_workers": training_args.dataloader_num_workers,
        }
        
        # Create trainer
        logger.info("Using PruningTrainer")
        trainer = PruningTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            loss_fn=loss_fn,
            training_args=training_args_dict,
        )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"Mode: {model_args.mode}")
    logger.info(f"Dataset: {data_args.dataset_name}:{data_args.subset}")
    logger.info(f"Output: {training_args.output_dir}")
    
    if training_args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")
    
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # Save final model
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(final_model_path)
    
    # Save training arguments
    with open(os.path.join(final_model_path, "training_args.json"), "w") as f:
        args_dict = {
            "model_args": model_args.__dict__,
            "data_args": data_args.__dict__,
            "training_args": {k: v for k, v in training_args.__dict__.items() if not k.startswith('_')}
        }
        json.dump(args_dict, f, indent=2, default=str)
    
    # Push to hub if requested
    if training_args.push_to_hub:
        logger.info(f"Pushing model to hub: {training_args.hub_model_id}")
        model.push_to_hub(training_args.hub_model_id)
    
    # Test the trained model
    logger.info("\n" + "="*50)
    logger.info("Testing trained model...")
    logger.info("="*50)
    
    # Load the saved model
    loaded_model = PruningEncoder.from_pretrained(final_model_path)
    
    # Example queries and documents
    test_examples = [
        {
            "query": "機械学習とは何ですか？",
            "documents": ["機械学習は人工知能の一分野です。コンピュータがデータから学習することを可能にします。今日の天気は晴れです。鳥は空を飛ぶことができます。"]
        },
        {
            "query": "Pythonはどのように動作しますか？",
            "documents": ["Pythonはインタープリタ型のプログラミング言語です。動的型付けを使用します。コーヒーは人気のある飲み物です。Pythonのコードは一行ずつ実行されます。"]
        }
    ]
    
    for example in test_examples:
        logger.info(f"\nQuery: {example['query']}")
        logger.info(f"Document: {example['documents'][0][:100]}...")
        
        # For predict_context, we need sentences as (query, document) pairs and chunk positions
        # Simulate simple sentence-level chunking
        document = example['documents'][0]
        sentences = (example['query'], document)
        
        # Simple sentence splitting for chunk positions (character-level offsets)
        import re
        sentence_ends = [m.end() for m in re.finditer(r'[。！？]', document)]
        if not sentence_ends or sentence_ends[-1] < len(document):
            sentence_ends.append(len(document))
        
        chunk_positions = []
        start = 0
        for end in sentence_ends:
            chunk_positions.append([start, end])
            start = end
        
        # Predict with pruning using chunk-based evaluation
        output = loaded_model.predict_context(
            sentences=sentences,
            chunk_positions=chunk_positions,
            chunk_threshold=0.3  # Optimal threshold from spec
        )
        
        logger.info(f"Chunk-level compression ratio: {output.compression_ratio:.2%}")
        if hasattr(output, 'chunk_predictions') and output.chunk_predictions is not None:
            num_relevant_chunks = int(output.chunk_predictions.sum())
            logger.info(f"Number of relevant chunks: {num_relevant_chunks}/{len(chunk_positions)}")
        if hasattr(output, 'pruned_documents') and output.pruned_documents:
            logger.info(f"Pruned document: {output.pruned_documents[0]}")
        elif hasattr(output, 'pruned_document'):
            logger.info(f"Pruned document: {output.pruned_document}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("🎉 Training completed successfully!")
    print(f"📁 Model saved to: {final_model_path}")
    print(f"🚀 To use this model: PruningEncoder.from_pretrained('{final_model_path}')")
    print(f"{'='*80}\n")
    
    logger.info("\n" + "="*50)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"To use this model: PruningEncoder.from_pretrained('{final_model_path}')")


if __name__ == "__main__":
    main()

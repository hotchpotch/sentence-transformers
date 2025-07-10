#!/usr/bin/env python3
"""
Flexible training script for PruningEncoder with teacher score distillation.
Supports various configurations through command line arguments and config files.

Usage:
    # Using YAML config file
    python scripts/train_pruning.py pruning-config/train-models/japanese-reranker-xsmall-v2.yaml
    
    # Using command line arguments
    python scripts/train_pruning.py \
        --model_name_or_path hotchpotch/japanese-reranker-xsmall-v2 \
        --dataset_name hotchpotch/wip-msmarco-context-relevance \
        --subset msmarco-ja-minimal \
        --output_dir ./output/pruning-models/minimal
    
    # Mix config file with command line overrides
    python scripts/train_pruning.py pruning-config/train-models/japanese-reranker-xsmall-v2.yaml \
        --subset msmarco-ja-small \
        --num_train_epochs 2
    
    # Pruning-only mode
    python scripts/train_pruning.py \
        --model_name_or_path cl-nagoya/ruri-v3-30m \
        --mode pruning_only \
        --subset msmarco-ja-minimal
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

import torch
from datasets import load_dataset, DatasetDict
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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of evaluation examples"}
    )
    validation_split: float = field(
        default=0.1,
        metadata={"help": "Validation split ratio if no validation set exists"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )


@dataclass
class PruningTrainingArguments(TrainingArguments):
    """Training arguments specific to PruningEncoder training."""
    ranking_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for ranking loss"}
    )
    pruning_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for pruning loss"}
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


def prepare_dataset(
    data_args: DataArguments,
    teacher_model_name: str,
    seed: int = 42
) -> Tuple[Any, Any]:
    """
    Load and prepare dataset with teacher scores.
    
    Args:
        dataset_name: Name of the dataset to load
        teacher_model_name: Name of the teacher model for score column
        subset: Dataset subset to use
        max_train_samples: Maximum number of training samples
        max_eval_samples: Maximum number of evaluation samples
        validation_split: Split ratio if no validation set exists
        seed: Random seed for splitting
    """
    # Check if it's a local dataset first
    subset_name = data_args.subset.replace('msmarco-ja-', '')
    local_dataset_path = f"tmp/datasets/dev-dataset/{subset_name}-fixed"
    
    # Try -fixed variant first, then fallback to base name
    if os.path.exists(local_dataset_path):
        logger.info(f"Loading local dataset: {local_dataset_path}")
        from datasets import load_from_disk
        dataset = load_from_disk(local_dataset_path)
    elif os.path.exists(f"tmp/datasets/dev-dataset/{subset_name}"):
        local_dataset_path = f"tmp/datasets/dev-dataset/{subset_name}"
        logger.info(f"Loading local dataset: {local_dataset_path}")
        from datasets import load_from_disk
        dataset = load_from_disk(local_dataset_path)
    else:
        # Load dataset from HuggingFace
        logger.info(f"Loading dataset: {data_args.dataset_name}:{data_args.subset}")
        dataset = load_dataset(data_args.dataset_name, data_args.subset)
    
    # Construct teacher score column name
    teacher_score_column = f"teacher_scores.{teacher_model_name}"
    
    # Prepare dataset with proper column names
    def prepare_example(example):
        """Prepare a single example for training."""
        # Handle local dataset format vs HuggingFace format
        if 'text' in example and 'texts' not in example:
            # Local dataset format: convert single text to list
            example['texts'] = [example['text']]
        elif 'texts' not in example:
            raise ValueError("Dataset must contain 'texts' or 'text' field")
        
        if 'query' not in example:
            raise ValueError("Dataset must contain 'query' field")
        
        # Get teacher score - local datasets already have 'teacher_score' field
        if 'teacher_score' in example:
            # Local dataset already has teacher_score as float
            score = example['teacher_score']
            if isinstance(score, (int, float)):
                # Convert single score to list to match texts
                example['teacher_score'] = [score]
            elif not isinstance(score, list):
                raise ValueError(f"Teacher score must be float or list, got {type(score)}")
        elif teacher_score_column in example:
            example['teacher_score'] = example[teacher_score_column]
        else:
            # Try without dots (sometimes column names are escaped)
            escaped_column = teacher_score_column.replace('.', '_')
            if escaped_column in example:
                example['teacher_score'] = example[escaped_column]
            else:
                # Fallback to using labels if teacher scores not available
                logger.warning(f"Teacher score column '{teacher_score_column}' not found.")
                logger.warning(f"Available columns: {list(example.keys())}")
                logger.info("Falling back to using labels as teacher scores")
                
                # Use ranking_label if available, otherwise labels
                if 'ranking_label' in example:
                    # Convert single ranking label to list
                    example['teacher_score'] = [float(example['ranking_label'])]
                elif 'labels' in example:
                    # Use labels directly (already a list)
                    example['teacher_score'] = [float(label) for label in example['labels']]
                else:
                    raise ValueError(f"No teacher scores or labels found. Available columns: {list(example.keys())}")
        
        # Ensure teacher scores are float values for regression-based knowledge distillation
        # This enables learning from continuous reranker scores rather than binary labels
        # The PruningDataCollator will use these scores with MSE loss for better knowledge transfer
        if not isinstance(example['teacher_score'], list):
            raise ValueError(f"Teacher scores must be a list, got {type(example['teacher_score'])}")
        
        # Add required fields for DataCollator compatibility
        # Convert ranking_label to labels (list format expected by DataCollator)
        if 'ranking_label' in example:
            example['labels'] = [example['ranking_label']]
        
        # Add chunk-related fields if they don't exist
        # For single-text examples, we have simple chunk structure
        if 'chunks_pos' not in example and 'sentence_boundaries' in example:
            # Convert sentence boundaries to chunk positions
            # DataCollator expects list of chunk positions for each text
            example['chunks_pos'] = [example['sentence_boundaries']]  # List of lists for multiple texts
        
        if 'relevant_chunks' not in example and 'pruning_labels' in example:
            # Convert pruning labels to relevant chunks
            # DataCollator expects list of relevant chunk indices for each text
            relevant_chunks = []
            pruning_labels = example['pruning_labels']
            for i, label in enumerate(pruning_labels):
                if label == 1:  # If chunk is relevant
                    relevant_chunks.append(i)
            example['relevant_chunks'] = [relevant_chunks]  # List of lists for multiple texts
        
        return example
    
    # Apply preprocessing
    train_dataset = dataset['train'].map(prepare_example)
    
    # Handle validation set
    if 'validation' in dataset:
        eval_dataset = dataset['validation'].map(prepare_example)
    elif 'test' in dataset:
        eval_dataset = dataset['test'].map(prepare_example)
    else:
        # Split training set if no validation set exists
        logger.info(f"No validation set found. Creating split with {data_args.validation_split:.0%} of training data.")
        split_dataset = train_dataset.train_test_split(test_size=data_args.validation_split, seed=seed)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    
    # Apply sampling if specified
    if data_args.max_train_samples and len(train_dataset) > data_args.max_train_samples:
        logger.info(f"Sampling {data_args.max_train_samples} training examples from {len(train_dataset)}")
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    
    if data_args.max_eval_samples and len(eval_dataset) > data_args.max_eval_samples:
        logger.info(f"Sampling {data_args.max_eval_samples} evaluation examples from {len(eval_dataset)}")
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


class PruningHfTrainer(Trainer):
    """Custom Trainer that uses PruningTrainer internally for compatibility."""
    
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
    
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
        
        return (loss, None) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        """Custom prediction step that handles PruningDataCollator format."""
        has_labels = "labels" in inputs
        
        with torch.no_grad():
            if has_labels:
                loss = self.compute_loss(model, inputs, return_outputs=False)
            else:
                loss = None
        
        # For evaluation, we mainly care about the loss
        if prediction_loss_only:
            return (loss, None, None)
        
        # We don't need logits for evaluation in this case
        return (loss, None, None)


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
        classifier_dropout=model_config.get('classifier_dropout', 0.1)
    )
    
    # Extract data arguments
    data_args = DataArguments()
    
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
        output_dir=training_config.get('output_dir', './output'),
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
    
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        # If we pass only one argument and it's a yaml file, parse it
        model_args, data_args, training_args = parse_config_file(sys.argv[1])
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument and it's a json file, parse it
        model_args, data_args, training_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        # Otherwise parse from command line
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
        # Handle legacy config file argument
        if "--config" in sys.argv:
            config_idx = sys.argv.index("--config")
            if config_idx + 1 < len(sys.argv):
                config_file = sys.argv[config_idx + 1]
                # Override with values from config file
                config_model_args, config_data_args, config_training_args = parse_config_file(config_file)
                # Merge configurations (command line takes precedence)
                for field in model_args.__dataclass_fields__:
                    if getattr(model_args, field) == model_args.__dataclass_fields__[field].default:
                        setattr(model_args, field, getattr(config_model_args, field))
    
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
    
    # Set seed
    set_seed(training_args.seed)
    
    # Create timestamp for unique naming
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    
    # Initialize WandB if available
    if _wandb_available and "wandb" in training_args.report_to:
        # Set WandB project name
        os.environ["WANDB_PROJECT"] = "pruning"
        
        # Create run name based on configuration with timestamp
        run_name = f"{Path(model_args.model_name_or_path).name}-{model_args.mode}-{data_args.subset}-{timestamp}"
        
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
        run_name = f"{Path(model_args.model_name_or_path).name}-{model_args.mode}-{data_args.subset}-{timestamp}"
    
    # Extract teacher model name
    if data_args.teacher_model_name:
        teacher_model_name = data_args.teacher_model_name
    else:
        # Default to japanese-reranker-xsmall-v2
        teacher_model_name = "japanese-reranker-xsmall-v2"
    logger.info(f"Using teacher model: {teacher_model_name}")
    
    # Set output directory if not specified
    if not training_args.output_dir or training_args.output_dir == "./output":
        output_base_dir = "./output/pruning-models"
        training_args.output_dir = os.path.join(
            output_base_dir,
            run_name
        )
    
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
    
    # Create data collator
    data_collator = PruningDataCollator(
        tokenizer=model.tokenizer,
        max_length=model.max_length,
        mode=model.mode,
        scores_column='teacher_score'  # Use teacher scores for distillation
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
    use_hf_trainer = training_args.local_rank != -1  # Use HF Trainer for distributed training
    
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
    
    logger.info("\n" + "="*50)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"To use this model: PruningEncoder.from_pretrained('{final_model_path}')")


if __name__ == "__main__":
    main()
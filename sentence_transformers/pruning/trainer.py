"""
PruningTrainer: Trainer for PruningEncoder models.
"""

import os
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizer
from datasets import Dataset as HFDataset
from tqdm import tqdm

from .encoder import PruningEncoder
from .losses import PruningLoss
from .data_collator import PruningDataCollator

logger = logging.getLogger(__name__)


class PruningTrainer:
    """
    Trainer for PruningEncoder models.
    
    Args:
        model: PruningEncoder model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        data_collator: Data collator (optional)
        loss_fn: Loss function (optional)
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        training_args: Training arguments
        compute_metrics: Function to compute metrics
        callbacks: List of callbacks
    """
    
    def __init__(
        self,
        model: PruningEncoder,
        train_dataset: Union[Dataset, HFDataset],
        eval_dataset: Optional[Union[Dataset, HFDataset]] = None,
        data_collator: Optional[PruningDataCollator] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        training_args: Optional[Dict[str, Any]] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Default training args
        default_args = {
            "output_dir": "./output",
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "logging_steps": 50,
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 2,
            "seed": 42,
            "fp16": torch.cuda.is_available(),
            "dataloader_num_workers": 0,
        }
        self.training_args = {**default_args, **(training_args or {})}
        
        # Data collator
        if data_collator is None:
            data_collator = ProvenceDataCollator(
                tokenizer=model.tokenizer,
                max_length=model.max_length,
                padding=True,
                truncation=True
            )
        self.data_collator = data_collator
        
        # Loss function
        if loss_fn is None:
            loss_fn = PruningLoss(
                model=model,
                ranking_weight=1.0,
                pruning_weight=0.5,
                is_regression=True  # Default to regression for teacher score distillation
            )
        self.loss_fn = loss_fn
        
        # Optimizer
        if optimizer is None:
            optimizer = AdamW(
                model.parameters(),
                lr=self.training_args["learning_rate"],
                weight_decay=self.training_args["weight_decay"]
            )
        self.optimizer = optimizer
        
        # Scheduler
        if scheduler is None and self.training_args["warmup_ratio"] > 0:
            num_training_steps = (
                len(self.train_dataset) // self.training_args["batch_size"] 
                * self.training_args["num_epochs"]
            )
            num_warmup_steps = int(num_training_steps * self.training_args["warmup_ratio"])
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        self.scheduler = scheduler
        
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        
        # Set seed
        torch.manual_seed(self.training_args["seed"])
        
        # Create output directory
        os.makedirs(self.training_args["output_dir"], exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.best_model_path = None
    
    def train(self):
        """Run training loop."""
        # Create data loaders
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_args["batch_size"],
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.training_args["dataloader_num_workers"],
            pin_memory=True,
        )
        
        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.training_args["batch_size"] * 2,
                shuffle=False,
                collate_fn=self.data_collator,
                num_workers=self.training_args["dataloader_num_workers"],
                pin_memory=True,
            )
        
        # Training info
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num epochs = {self.training_args['num_epochs']}")
        logger.info(f"  Batch size = {self.training_args['batch_size']}")
        logger.info(f"  Gradient accumulation steps = {self.training_args['gradient_accumulation_steps']}")
        logger.info(f"  Total optimization steps = {len(train_dataloader) * self.training_args['num_epochs']}")
        
        # Enable mixed precision if requested
        scaler = None
        if self.training_args.get("fp16", False):
            scaler = torch.amp.GradScaler('cuda')
        elif self.training_args.get("bf16", False):
            # bf16 doesn't need gradient scaling
            scaler = None
        
        # Training loop
        for epoch in range(self.training_args["num_epochs"]):
            self.epoch = epoch
            self.model.train()
            
            epoch_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
                # Forward pass
                if self.training_args.get("fp16", False) and scaler is not None:
                    with torch.amp.autocast('cuda'):
                        loss = self._training_step(batch)
                elif self.training_args.get("bf16", False):
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        loss = self._training_step(batch)
                else:
                    loss = self._training_step(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.training_args["gradient_accumulation_steps"]
                epoch_loss += loss.item()
                
                # Backward pass
                if self.training_args.get("fp16", False) and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.training_args["gradient_accumulation_steps"] == 0:
                    # Gradient clipping
                    if self.training_args["max_grad_norm"] > 0:
                        if self.training_args.get("fp16", False) and scaler is not None:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_args["max_grad_norm"]
                        )
                    
                    # Optimizer step
                    if self.training_args.get("fp16", False) and scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.training_args["logging_steps"] == 0:
                        avg_loss = epoch_loss / (step + 1)
                        logger.info(f"Step {self.global_step}, Loss: {avg_loss:.4f}")
                    
                    # Evaluation
                    if (eval_dataloader is not None and 
                        self.global_step % self.training_args["eval_steps"] == 0):
                        self._evaluate(eval_dataloader)
                    
                    # Save checkpoint
                    if self.global_step % self.training_args["save_steps"] == 0:
                        self._save_checkpoint()
            
            # End of epoch evaluation
            if eval_dataloader is not None:
                self._evaluate(eval_dataloader)
            
            # Save checkpoint at end of epoch
            self._save_checkpoint()
        
        logger.info("Training completed!")
        
        # Load best model if available
        if self.best_model_path is not None:
            logger.info(f"Loading best model from {self.best_model_path}")
            self.model = PruningEncoder.from_pretrained(self.best_model_path)
    
    def _training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Single training step."""
        # Move batch to device
        sentence_features = batch["sentence_features"]
        labels = batch["labels"]
        
        # Move tensors to device
        for key in sentence_features[0]:
            if isinstance(sentence_features[0][key], torch.Tensor):
                sentence_features[0][key] = sentence_features[0][key].to(self.model.device)
        
        for key in labels:
            if isinstance(labels[key], torch.Tensor):
                labels[key] = labels[key].to(self.model.device)
        
        # Compute loss
        loss = self.loss_fn(sentence_features, labels)
        
        return loss
    
    def _evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        logger.info("Running evaluation...")
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                if self.training_args.get("fp16", False):
                    with torch.amp.autocast('cuda'):
                        loss = self._training_step(batch)
                elif self.training_args.get("bf16", False):
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        loss = self._training_step(batch)
                else:
                    loss = self._training_step(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Eval loss: {avg_loss:.4f}")
        
        # Compute additional metrics if provided
        metrics = {"eval_loss": avg_loss}
        if self.compute_metrics is not None:
            additional_metrics = self.compute_metrics(self.model, eval_dataloader)
            metrics.update(additional_metrics)
        
        # Update best model
        if self.best_metric is None or avg_loss < self.best_metric:
            self.best_metric = avg_loss
            self.best_model_path = os.path.join(
                self.training_args["output_dir"],
                f"checkpoint-{self.global_step}-best"
            )
            self.model.save_pretrained(self.best_model_path)
            logger.info(f"New best model saved to {self.best_model_path}")
        
        self.model.train()
        return metrics
    
    def _save_checkpoint(self):
        """Save a checkpoint."""
        checkpoint_path = os.path.join(
            self.training_args["output_dir"],
            f"checkpoint-{self.global_step}"
        )
        
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        self.model.save_pretrained(checkpoint_path)
        
        # Save trainer state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "best_model_path": self.best_model_path,
            "training_args": self.training_args,
        }
        
        torch.save(state, os.path.join(checkpoint_path, "trainer_state.pt"))
        
        # Manage checkpoint limit
        if self.training_args["save_total_limit"] is not None:
            self._rotate_checkpoints()
    
    def _rotate_checkpoints(self):
        """Delete old checkpoints to maintain limit."""
        checkpoints = []
        for path in Path(self.training_args["output_dir"]).glob("checkpoint-*"):
            if path.name.endswith("-best"):
                continue
            step = int(path.name.split("-")[1])
            checkpoints.append((step, path))
        
        # Sort by step
        checkpoints.sort(key=lambda x: x[0])
        
        # Delete old checkpoints
        while len(checkpoints) > self.training_args["save_total_limit"]:
            _, path = checkpoints.pop(0)
            logger.info(f"Deleting checkpoint {path}")
            import shutil
            shutil.rmtree(path)
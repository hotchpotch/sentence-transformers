"""
Train a sentence transformer with Quantization-Aware Training (QAT) on AllNLI.

Usage:
python train_qat_nli.py
python train_qat_nli.py pretrained_transformer_model_name
"""

import logging
import sys
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss, QuantizationAwareLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "distilbert/distilbert-base-uncased"
batch_size = 128
num_train_epochs = 1
quantization_precisions = ["float32", "int8", "binary"]

output_dir = f"output/qat_nli_{model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

model = SentenceTransformer(model_name)
logging.info(model)

train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
logging.info(train_dataset)

inner_train_loss = MultipleNegativesRankingLoss(model=model)
train_loss = QuantizationAwareLoss(
    model,
    loss=inner_train_loss,
    quantization_precisions=quantization_precisions,
)

dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    name="all-nli-dev",
)

args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=True,
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    run_name="qat-nli",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)
logging.info(f"Model saved to: {final_output_dir}")

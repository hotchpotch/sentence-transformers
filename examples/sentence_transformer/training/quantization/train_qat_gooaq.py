"""
Train a sentence transformer with Quantization-Aware Training (QAT) on GooAQ using MultipleNegativesRankingLoss.

Usage:
python train_qat_gooaq.py
"""

import logging
import random

from datasets import Dataset, load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss, QuantizationAwareLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = "microsoft/mpnet-base"
num_train_samples = 100_000
num_eval_samples = 10_000
train_batch_size = 64
num_epochs = 1
quantization_precisions = ["float32", "int8", "binary"]
eval_quantization_precisions = ["float32", "int8", "binary"]
quantization_weights = [1.0, 1.0, 0.5]

logging.info(f"Loading model: {model_name}")
model = SentenceTransformer(
    model_name,
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="MPNet base trained on GooAQ using QAT with MultipleNegativesRankingLoss",
    ),
)

logging.info("Loading GooAQ dataset")
dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(num_train_samples))
dataset = dataset.add_column("id", range(len(dataset)))
dataset_dict = dataset.train_test_split(test_size=num_eval_samples, seed=12)
train_dataset: Dataset = dataset_dict["train"]
eval_dataset: Dataset = dataset_dict["test"]
logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Eval dataset size: {len(eval_dataset)}")

base_loss = MultipleNegativesRankingLoss(model)
loss = QuantizationAwareLoss(
    model=model,
    loss=base_loss,
    quantization_precisions=quantization_precisions,
    quantization_weights=quantization_weights,
)
logging.info(f"Training with quantization precisions: {quantization_precisions}")

logging.info("Creating evaluation corpus")
random.seed(12)
queries = dict(zip(eval_dataset["id"], eval_dataset["question"]))
corpus = {qid: dataset[qid]["answer"] for qid in queries}
relevant_docs = {qid: {qid} for qid in eval_dataset["id"]}

evaluators = [
    InformationRetrievalEvaluator(
        corpus=corpus,
        queries=queries,
        relevant_docs=relevant_docs,
        show_progress_bar=True,
        name=f"gooaq-dev-{precision}",
        precision=precision,
    )
    for precision in eval_quantization_precisions
]
dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

logging.info("Performance before fine-tuning:")
dev_evaluator(model)

short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"{short_model_name}-gooaq-qat"
args = SentenceTransformerTrainingArguments(
    output_dir=f"models/{run_name}",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=0.025,
    logging_first_step=True,
    run_name=run_name,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset.remove_columns("id"),
    eval_dataset=eval_dataset.remove_columns("id"),
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

logging.info("Evaluating trained model with different quantization precisions")
dev_evaluator(model)

final_output_dir = f"models/{run_name}/final"
model.save(final_output_dir)
logging.info(f"Model saved to: {final_output_dir}")

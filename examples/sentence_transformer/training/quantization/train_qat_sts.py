"""
Train a sentence transformer with Quantization-Aware Training (QAT) on STS Benchmark.

Usage:
python train_qat_sts.py
python train_qat_sts.py pretrained_transformer_model_name
"""

import logging
import sys
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator,
    SimilarityFunction,
)
from sentence_transformers.sentence_transformer.losses import CoSENTLoss, QuantizationAwareLoss

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "distilbert/distilbert-base-uncased"
batch_size = 16
num_train_epochs = 4
quantization_precisions = ["float32", "int8", "binary"]

output_dir = f"output/qat_sts_{model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

model = SentenceTransformer(model_name)
logging.info(model)

train_dataset = load_dataset("sentence-transformers/stsb", split="train")
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(train_dataset)

inner_train_loss = CoSENTLoss(model=model)
train_loss = QuantizationAwareLoss(
    model,
    loss=inner_train_loss,
    quantization_precisions=quantization_precisions,
)

evaluators = [
    EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["sentence1"],
        sentences2=eval_dataset["sentence2"],
        scores=eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name=f"sts-dev-{precision}",
        precision=precision,
    )
    for precision in quantization_precisions
]
dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=True,
    bf16=False,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="qat-sts",
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

evaluators = [
    EmbeddingSimilarityEvaluator(
        sentences1=test_dataset["sentence1"],
        sentences2=test_dataset["sentence2"],
        scores=test_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name=f"sts-test-{precision}",
        precision=precision,
    )
    for precision in quantization_precisions
]
test_evaluator = SequentialEvaluator(evaluators)
test_evaluator(model)

final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)
logging.info(f"Model saved to: {final_output_dir}")

# QAT Implementation Spec

## Scope
This document defines the next implementation steps for improving Quantization-Aware Training (QAT) quality in this branch.

## Read First (Mandatory)
Before changing code, read these documents first and treat them as the primary design references:

- [tmp/1712.05877.pdf.md](tmp/1712.05877.pdf.md) (Google paper: integer-only quantization + training flow)
- [tmp/jina-v4-qat.md](tmp/jina-v4-qat.md) (Jina engineering write-up: retrieval-focused QAT variants)

If implementation details are unclear, check these two files first before exploring external sources.

## Problem Statement
Current QAT training is implemented, but observed retrieval performance under quantized evaluation does not consistently improve versus non-QAT baselines.

## Target Outcomes
1. Reduce performance degradation at quantized inference (`int8`, `binary`) versus float32.
2. Ensure evaluation logic matches intended quantized inference behavior.
3. Make quantization ranges stable and reproducible across query/corpus and across corpus chunks.
4. Provide fair baseline comparisons (PTQ vs QAT, symmetric vs asymmetric query/doc quantization).

## Required Implementation Areas

### 1) Evaluation Quantization Path (High Priority)
- Introduce explicit calibration-aware range handling for `int8` evaluation:
  - Support shared ranges between query and corpus (or a fixed precomputed range source).
  - Avoid per-batch/per-chunk range drift during retrieval evaluation.
- Add dequantization-aware scoring path for `int8`:
  - Keep quantization metadata (range/scale/zero-point) and convert consistently before similarity scoring.
- Add asymmetric evaluation mode:
  - Quantized docs + float queries (`docs_only`) in addition to quantized queries+docs.
- Keep binary/ubinary path explicit and reproducible:
  - Clearly define representation and conversion before scoring.

### 2) QAT Training Behavior (High Priority)
- Add range-learning/stabilization strategy inspired by references:
  - EMA-style activation range tracking (or equivalent stable running estimator).
- Add delayed activation quantization (warmup):
  - Disable activation quantization early in training, then enable after warmup.
- Ensure train-time fake quantization behavior mirrors eval/inference quantization semantics.

### 3) Experimental Matrix and Reporting (Required)
- Run and report at least:
  - Baseline float training + PTQ eval
  - QAT training + quantized eval
  - Symmetric (query+doc quantized) vs asymmetric (doc-only quantized) settings
  - Multiple quantization levels where relevant (`binary`, `int8`; optional `4-bit`, `trinary` if implemented)
- Store metrics in a reproducible table (NDCG@10, MRR@10, MAP@100, Accuracy@k).

## Suggested Code Touchpoints
- `sentence_transformers/quantization.py`
- `sentence_transformers/SentenceTransformer.py` (`encode` quantization flow)
- `sentence_transformers/evaluation/InformationRetrievalEvaluator.py`
- `sentence_transformers/evaluation/EmbeddingSimilarityEvaluator.py`
- `sentence_transformers/losses/QuantizationAwareLoss.py`
- `examples/sentence_transformer/training/quantization/*`

## Acceptance Criteria
1. Quantized eval path is deterministic and range-stable.
2. Docs-only quantization evaluation mode is available and documented.
3. QAT with improved range strategy shows measurable gain or reduced drop under quantized eval against PTQ baseline.
4. Example scripts and docs reflect the new behavior and comparison protocol.


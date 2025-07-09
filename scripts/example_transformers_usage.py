#!/usr/bin/env python
"""
Example of using PruningEncoder models with Transformers AutoModel classes.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer
)

# Import PruningEncoder to register the model classes
import sentence_transformers.pruning

def example_sequence_classification():
    """Example: Using reranking model with AutoModelForSequenceClassification."""
    print("="*60)
    print("Example: Reranking with AutoModelForSequenceClassification")
    print("="*60)
    
    # Load model and tokenizer
    model_path = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Example queries and documents
    examples = [
        ("機械学習について", "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。"),
        ("天気予報について", "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。"),
        ("深層学習とは", "ディープラーニングは多層のニューラルネットワークを使用した機械学習手法です。"),
    ]
    
    for query, document in examples:
        # Tokenize
        inputs = tokenizer(query, document, return_tensors="pt", truncation=True, max_length=512)
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.sigmoid(outputs.logits).item()
        
        print(f"\nQuery: {query}")
        print(f"Document: {document[:50]}...")
        print(f"Relevance Score: {score:.4f}")


def example_token_classification():
    """Example: Using pruning model with AutoModelForTokenClassification."""
    print("\n" + "="*60)
    print("Example: Pruning with AutoModelForTokenClassification")
    print("="*60)
    
    # Load model and tokenizer
    model_path = "./output/transformers_compat_test/pruning_only_20250709_135222/final_model"
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Example
    query = "機械学習について"
    document = "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。近年、深層学習の発展により、画像認識や自然言語処理などの分野で大きな進歩が見られます。"
    
    # Tokenize
    inputs = tokenizer(query, document, return_tensors="pt", truncation=True, max_length=512)
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        # Get probabilities for keeping tokens (class 1)
        probs = torch.softmax(outputs.logits, dim=-1)
        keep_probs = probs[:, :, 1]
    
    # Apply threshold
    threshold = 0.3
    keep_mask = keep_probs[0] > threshold
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print(f"\nQuery: {query}")
    print(f"Document: {document}")
    print(f"\nToken-level pruning (threshold={threshold}):")
    print(f"Total tokens: {len(tokens)}")
    print(f"Kept tokens: {keep_mask.sum().item()} ({keep_mask.sum().item()/len(tokens)*100:.1f}%)")
    
    # Show kept tokens
    kept_tokens = [token for token, keep in zip(tokens, keep_mask) if keep]
    print(f"\nKept tokens: {' '.join(kept_tokens[:20])}...")


def main():
    """Run examples."""
    example_sequence_classification()
    example_token_classification()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nPruningEncoder models can now be used with standard Transformers patterns:")
    print("- AutoModelForSequenceClassification for reranking models")
    print("- AutoModelForTokenClassification for pruning-only models")
    print("\nThis enables easy integration with existing NLP pipelines and tools!")


if __name__ == "__main__":
    main()
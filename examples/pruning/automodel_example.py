#!/usr/bin/env python
"""
Example of loading PruningEncoder models using transformers AutoModel classes.
This demonstrates how models saved with PruningEncoder can be loaded using
the standard transformers library APIs.
"""

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch


def main():
    # Path to a saved PruningEncoder model
    # Replace with your actual model path or Hugging Face model ID
    model_path = "path/to/your/pruning-encoder-model"
    
    # Load the model using AutoModelForSequenceClassification
    # Note: trust_remote_code=True is required for custom models
    print("Loading model with AutoModelForSequenceClassification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Example usage
    query = "What is machine learning?"
    document = "Machine learning is a subset of artificial intelligence..."
    
    # Tokenize inputs
    inputs = tokenizer(
        query,
        document,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        ranking_score = outputs.logits.squeeze().item()
    
    print(f"Ranking score: {ranking_score}")
    
    # For token classification (pruning only mode), use:
    # from transformers import AutoModelForTokenClassification
    # model = AutoModelForTokenClassification.from_pretrained(
    #     model_path,
    #     trust_remote_code=True
    # )


if __name__ == "__main__":
    main()
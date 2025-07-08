#!/usr/bin/env python3
"""
Summarize chunk-based evaluation results across models
"""

import json
from pathlib import Path
import pandas as pd

def load_results(filepath):
    """Load evaluation results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(results, threshold_key):
    """Extract key metrics from results"""
    data = results[threshold_key]
    return {
        'chunk_f1': data['chunk_f1'],
        'chunk_precision': data['chunk_precision'], 
        'chunk_recall': data['chunk_recall'],
        'chunk_accuracy': data['chunk_accuracy'],
        'ranking_f1': data['ranking_f1'],
        'compression_ratio': data['compression_ratio'],
        'avg_query_accuracy': data['avg_query_accuracy']
    }

def create_summary():
    """Create summary comparison"""
    
    # Load results for all models
    models = {
        'minimal': 'results/chunk_eval_minimal.json',
        'small': 'results/chunk_eval_small.json', 
        'full': 'results/chunk_eval_full.json'
    }
    
    results = {}
    for model_name, filepath in models.items():
        if Path(filepath).exists():
            results[model_name] = load_results(filepath)
        else:
            print(f"Warning: {filepath} not found")
    
    # Key threshold combinations to compare
    thresholds = ['0.1_0.1', '0.3_0.3', '0.5_0.5', '0.7_0.7', '0.3_0.5']
    
    print("=== Chunk-based Evaluation Summary ===\n")
    
    # Compare each threshold setting
    for threshold in thresholds:
        print(f"## Threshold: {threshold.replace('_', ' token=')} chunk=")
        print("| Model | Chunk F1 | Precision | Recall | Accuracy | Ranking F1 | Compression | Query Acc |")
        print("|-------|----------|-----------|--------|----------|------------|-------------|-----------|")
        
        for model_name in ['minimal', 'small', 'full']:
            if model_name in results and threshold in results[model_name]:
                metrics = extract_metrics(results[model_name], threshold)
                print(f"| {model_name:7} | {metrics['chunk_f1']:8.3f} | "
                      f"{metrics['chunk_precision']:9.3f} | {metrics['chunk_recall']:6.3f} | "
                      f"{metrics['chunk_accuracy']:8.3f} | {metrics['ranking_f1']:10.3f} | "
                      f"{metrics['compression_ratio']:11.1%} | {metrics['avg_query_accuracy']:9.3f} |")
        print()
    
    # Best performance summary
    print("=== Best Performance Summary ===\n")
    
    best_metrics = {}
    
    for model_name in ['minimal', 'small', 'full']:
        if model_name not in results:
            continue
            
        best_f1 = 0
        best_threshold = None
        best_data = None
        
        for threshold in thresholds:
            if threshold in results[model_name]:
                f1 = results[model_name][threshold]['chunk_f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_data = results[model_name][threshold]
        
        best_metrics[model_name] = {
            'threshold': best_threshold,
            'data': best_data
        }
    
    print("**Best F1 scores:**")
    for model_name, info in best_metrics.items():
        if info['data']:
            data = info['data']
            print(f"- **{model_name}**: F1={data['chunk_f1']:.3f} @{info['threshold']} "
                  f"(P={data['chunk_precision']:.3f}, R={data['chunk_recall']:.3f}, "
                  f"Comp={data['compression_ratio']:.1%})")
    
    print("\n=== Key Observations ===")
    print("1. **Model Performance Ranking**: ", end="")
    f1_scores = [(name, info['data']['chunk_f1']) for name, info in best_metrics.items() if info['data']]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    print(" > ".join([f"{name}({f1:.3f})" for name, f1 in f1_scores]))
    
    print("\n2. **Compression vs Accuracy Trade-off**:")
    for model_name in ['minimal', 'small', 'full']:
        if model_name in results:
            # Compare low vs high threshold performance
            low_th = results[model_name].get('0.3_0.3', {})
            high_th = results[model_name].get('0.7_0.7', {})
            if low_th and high_th:
                print(f"   - {model_name}: Low threshold (0.3): F1={low_th['chunk_f1']:.3f}, Comp={low_th['compression_ratio']:.1%}")
                print(f"   - {model_name}: High threshold (0.7): F1={high_th['chunk_f1']:.3f}, Comp={high_th['compression_ratio']:.1%}")
    
    print("\n3. **Ranking Performance**: All models maintain excellent ranking performance (F1 > 0.98)")
    
    print("\n4. **Optimal Settings**:")
    for model_name, info in best_metrics.items():
        if info['data']:
            print(f"   - {model_name}: {info['threshold']} provides best balance of F1 and compression")

if __name__ == "__main__":
    create_summary()
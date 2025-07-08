#!/usr/bin/env python3
"""
NEGサンプルでFullモデルが完全に失敗する原因を調査
"""

import numpy as np
from datasets import load_dataset
from sentence_transformers.provence import ProvenceEncoder
import json

def analyze_dataset_and_model():
    print("=== NEGサンプル問題の原因調査 ===")
    
    # データセット読み込み
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')
    test_dataset = dataset['test']
    
    # POS/NEG分離
    pos_indices = []
    neg_indices = []
    
    for i, rel_chunks in enumerate(test_dataset['relevant_chunks']):
        if rel_chunks and rel_chunks[0]:
            pos_indices.append(i)
        else:
            neg_indices.append(i)
    
    print(f"Total examples: {len(test_dataset)}")
    print(f"POS examples: {len(pos_indices)}")
    print(f"NEG examples: {len(neg_indices)}")
    print()
    
    # NEGサンプルの詳細分析
    print("=== NEGサンプルの特徴分析 ===")
    neg_texts_lengths = []
    neg_chunk_counts = []
    
    for idx in neg_indices[:10]:  # 最初の10個を詳細分析
        example = test_dataset[idx]
        query = example['query']
        texts = example['texts']
        chunks_pos = example['chunks_pos']
        relevant_chunks = example['relevant_chunks']
        
        # 最初のテキストの情報
        first_text = texts[0] if texts else ""
        first_chunks = chunks_pos[0] if chunks_pos else []
        first_relevant = relevant_chunks[0] if relevant_chunks else []
        
        print(f"\nNEG例 {idx}:")
        print(f"  Query: {query[:100]}...")
        print(f"  Text length: {len(first_text)}")
        print(f"  Chunk count: {len(first_chunks)}")
        print(f"  Relevant chunks: {first_relevant}")
        
        neg_texts_lengths.append(len(first_text))
        neg_chunk_counts.append(len(first_chunks))
    
    print(f"\nNEGサンプル統計:")
    print(f"  平均テキスト長: {np.mean(neg_texts_lengths):.1f}")
    print(f"  平均チャンク数: {np.mean(neg_chunk_counts):.1f}")
    
    # POSサンプルとの比較
    print("\n=== POSサンプルとの比較 ===")
    pos_texts_lengths = []
    pos_chunk_counts = []
    pos_relevant_counts = []
    
    for idx in pos_indices[:10]:
        example = test_dataset[idx]
        texts = example['texts']
        chunks_pos = example['chunks_pos']
        relevant_chunks = example['relevant_chunks']
        
        first_text = texts[0] if texts else ""
        first_chunks = chunks_pos[0] if chunks_pos else []
        first_relevant = relevant_chunks[0] if relevant_chunks else []
        
        pos_texts_lengths.append(len(first_text))
        pos_chunk_counts.append(len(first_chunks))
        pos_relevant_counts.append(len(first_relevant))
    
    print(f"POSサンプル統計:")
    print(f"  平均テキスト長: {np.mean(pos_texts_lengths):.1f}")
    print(f"  平均チャンク数: {np.mean(pos_chunk_counts):.1f}")
    print(f"  平均関連チャンク数: {np.mean(pos_relevant_counts):.1f}")
    
    # モデル予測の分析
    print("\n=== モデル予測の分析 ===")
    
    # Small modelとFull modelでNEGサンプルの予測を比較
    small_model = ProvenceEncoder.from_pretrained("outputs/provence-ja-small/final-model")
    full_model = ProvenceEncoder.from_pretrained("outputs/provence-ja-full/final-model")
    
    # NEGサンプル3つで比較
    for i, idx in enumerate(neg_indices[:3]):
        example = test_dataset[idx]
        query = example['query']
        texts = example['texts']
        chunks_pos = example['chunks_pos']
        
        sentence = (str(query), str(texts[0]))
        chunk_positions = chunks_pos[0] if chunks_pos else []
        
        if not chunk_positions:
            continue
            
        print(f"\nNEG例 {i+1} (idx={idx}):")
        print(f"  Query: {query[:50]}...")
        print(f"  Chunk count: {len(chunk_positions)}")
        
        # Small model予測
        small_outputs = small_model.predict_context(
            sentences=[sentence],
            chunk_positions=[chunk_positions],
            batch_size=1,
            token_threshold=0.1,
            chunk_threshold=0.1,
            show_progress_bar=False
        )
        
        # Full model予測
        full_outputs = full_model.predict_context(
            sentences=[sentence],
            chunk_positions=[chunk_positions],
            batch_size=1,
            token_threshold=0.1,
            chunk_threshold=0.1,
            show_progress_bar=False
        )
        
        print(f"  Small model: kept {sum(small_outputs[0].chunk_predictions)}/{len(small_outputs[0].chunk_predictions)} chunks")
        print(f"  Full model: kept {sum(full_outputs[0].chunk_predictions)}/{len(full_outputs[0].chunk_predictions)} chunks")
        
        # 予測確率の分布
        small_probs = small_outputs[0].chunk_scores
        full_probs = full_outputs[0].chunk_scores
        
        print(f"  Small model scores: min={np.min(small_probs):.3f}, max={np.max(small_probs):.3f}, mean={np.mean(small_probs):.3f}")
        print(f"  Full model scores: min={np.min(full_probs):.3f}, max={np.max(full_probs):.3f}, mean={np.mean(full_probs):.3f}")

if __name__ == "__main__":
    analyze_dataset_and_model()
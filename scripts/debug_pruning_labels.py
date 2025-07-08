#!/usr/bin/env python3
"""
プルーニングラベル生成のデバッグスクリプト
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers.provence.data_collator_chunk_based import ProvenceChunkBasedDataCollator
import numpy as np

def main():
    print("=== プルーニングラベル生成デバッグ ===")
    
    # データセットロード
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-minimal')
    
    # 最初のサンプルを取得
    sample = dataset['train'][0]
    print(f"\n📊 サンプルデータ:")
    print(f"Query: {sample['query']}")
    print(f"Texts数: {len(sample['texts'])}")
    print(f"Labels: {sample['labels']}")
    
    # Tokenizerロード
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    
    # データコレクター初期化
    collator = ProvenceChunkBasedDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        padding=True,
        truncation=True
    )
    
    # 最初のテキストペアだけを詳細分析
    for text_idx in range(min(3, len(sample['texts']))):
        print(f"\n\n{'='*60}")
        print(f"📝 Text {text_idx} (label={sample['labels'][text_idx]})")
        print(f"Text: {sample['texts'][text_idx]}")
        print(f"Chunks_pos: {sample['chunks_pos'][text_idx]}")
        print(f"Relevant_chunks: {sample['relevant_chunks'][text_idx]}")
        
        # トークナイズ
        encoded = tokenizer(
            sample['query'],
            sample['texts'][text_idx],
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # トークン分析
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        offsets = encoded['offset_mapping'][0]
        
        print(f"\n🔤 トークン分析:")
        # SEPトークンの位置を探す
        sep_positions = (encoded['input_ids'][0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        print(f"SEPトークン位置: {sep_positions.tolist()}")
        
        if len(sep_positions) >= 3:
            doc_start = sep_positions[1].item() + 1
            doc_end = sep_positions[2].item()
            
            print(f"ドキュメント部分: {doc_start} - {doc_end}")
            print(f"\nドキュメントトークン:")
            for i in range(doc_start, min(doc_start + 10, doc_end)):  # 最初の10トークンのみ表示
                print(f"  [{i}] '{tokens[i]}' offset={offsets[i].tolist()}")
        
        # ラベル生成（単一サンプル用）
        features = [{
            'query': sample['query'],
            'texts': [sample['texts'][text_idx]],
            'chunks_pos': [sample['chunks_pos'][text_idx]],
            'relevant_chunks': [sample['relevant_chunks'][text_idx]],
            'ranking_labels': [sample['labels'][text_idx]],
            'teacher_scores': [sample['teacher_scores_japanese-reranker-xsmall-v2'][text_idx]]
        }]
        
        # コレクター呼び出し
        batch = collator(features)
        
        # プルーニングラベル分析
        pruning_labels = batch['labels']['pruning_labels'][0]
        print(f"\n🏷️  プルーニングラベル統計:")
        print(f"ラベル形状: {pruning_labels.shape}")
        print(f"1の数: {(pruning_labels == 1).sum().item()}")
        print(f"0の数: {(pruning_labels == 0).sum().item()}")
        print(f"-100の数: {(pruning_labels == -100).sum().item()}")
        
        # ドキュメント部分のラベルを確認
        if len(sep_positions) >= 3:
            doc_start = sep_positions[1].item() + 1
            doc_end = sep_positions[2].item()
            doc_labels = pruning_labels[doc_start:doc_end]
            print(f"\nドキュメント部分のラベル:")
            print(f"1の数: {(doc_labels == 1).sum().item()}")
            print(f"0の数: {(doc_labels == 0).sum().item()}")
            
            # 最初の20トークンのラベルを表示
            print(f"\n最初の20トークンのラベル:")
            for i in range(min(20, len(doc_labels))):
                if i + doc_start < len(tokens):
                    print(f"  [{i}] '{tokens[i + doc_start]}' = {doc_labels[i].item()}")
        
        # チャンク解析
        if sample['relevant_chunks'][text_idx]:
            print(f"\n📍 関連チャンク詳細:")
            for chunk_idx in sample['relevant_chunks'][text_idx]:
                if chunk_idx < len(sample['chunks_pos'][text_idx]):
                    start, end = sample['chunks_pos'][text_idx][chunk_idx]
                    chunk_text = sample['texts'][text_idx][start:end]
                    print(f"  Chunk {chunk_idx}: [{start}, {end}]")
                    print(f"  Text: {chunk_text}")


if __name__ == "__main__":
    main()
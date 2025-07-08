#!/usr/bin/env python3
"""
ja-minimal データセットの構造確認スクリプト
"""

from datasets import load_dataset
import json

def main():
    print("=== ja-minimal データセット構造確認 ===")
    
    # データセットロード
    try:
        dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-minimal')
        print("✅ データセットロード成功")
    except Exception as e:
        print(f"❌ データセットロードエラー: {e}")
        return
    
    # 基本情報
    print(f"\n📊 データセット基本情報:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data):,} 件")
    
    # サンプルデータ確認
    print(f"\n🔍 サンプルデータ:")
    sample = dataset['train'][0]
    
    print(f"フィールド一覧: {list(sample.keys())}")
    print(f"\nID: {sample['id']}")
    print(f"Query: {sample['query']}")
    print(f"Dataset name: {sample['dataset_name']}")
    print(f"Texts数: {len(sample['texts'])}")
    print(f"Labels: {sample['labels']}")
    print(f"Chunks_pos数: {len(sample['chunks_pos'])}")
    print(f"Relevant_chunks数: {len(sample['relevant_chunks'])}")
    print(f"Teacher_scores数: {len(sample['teacher_scores_japanese-reranker-xsmall-v2'])}")
    
    # 詳細確認
    print(f"\n📋 詳細データ構造:")
    for i, (text, chunks_pos, relevant_chunks, teacher_score, label) in enumerate(
        zip(sample['texts'], sample['chunks_pos'], sample['relevant_chunks'], 
            sample['teacher_scores_japanese-reranker-xsmall-v2'], sample['labels'])
    ):
        print(f"\n  Text {i} (label={label}, score={teacher_score:.3f}):")
        print(f"    Text長: {len(text)}")
        print(f"    Chunks数: {len(chunks_pos)}")
        print(f"    Relevant chunks: {relevant_chunks}")
        print(f"    Text preview: {text[:100]}...")
        
        # チャンク内容も確認
        if len(chunks_pos) > 0:
            print(f"    チャンク例:")
            for j, (start, end) in enumerate(chunks_pos[:3]):  # 最初の3チャンクのみ
                chunk_text = text[start:end].strip()
                is_relevant = j in relevant_chunks
                print(f"      [{j}] ({start}-{end}) {'✅' if is_relevant else '❌'}: {chunk_text[:50]}...")

    # 統計情報
    print(f"\n📈 統計情報:")
    train_data = dataset['train']
    
    # ラベル分布
    pos_count = sum(1 for item in train_data if item['labels'][0] == 1)  # 最初のテキストは必ずPOS
    total_texts = len(train_data) * 5  # 1サンプルにつき5テキスト
    total_pos = sum(sum(labels) for labels in train_data['labels'])
    print(f"  POS率: {total_pos}/{total_texts} = {total_pos/total_texts:.1%}")
    
    # 教師スコア分布
    all_pos_scores = []
    all_neg_scores = []
    
    for item in train_data:
        for label, score in zip(item['labels'], item['teacher_scores_japanese-reranker-xsmall-v2']):
            if label == 1:
                all_pos_scores.append(score)
            else:
                all_neg_scores.append(score)
    
    if all_pos_scores:
        print(f"  POS教師スコア: 平均={sum(all_pos_scores)/len(all_pos_scores):.3f}, 件数={len(all_pos_scores)}")
    if all_neg_scores:
        print(f"  NEG教師スコア: 平均={sum(all_neg_scores)/len(all_neg_scores):.3f}, 件数={len(all_neg_scores)}")
    
    # relevant_chunks統計
    total_relevant_chunks = 0
    total_chunks = 0
    
    for item in train_data:
        for chunks_pos, relevant_chunks in zip(item['chunks_pos'], item['relevant_chunks']):
            total_chunks += len(chunks_pos)
            total_relevant_chunks += len(relevant_chunks)
    
    print(f"  関連チャンク率: {total_relevant_chunks}/{total_chunks} = {total_relevant_chunks/total_chunks:.1%}")
    
    print("\n✅ データセット構造確認完了")


if __name__ == "__main__":
    main()
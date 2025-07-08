#!/usr/bin/env python3
"""
ja-full で学習した Provence モデルの評価スクリプト
"""

import os
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

from sentence_transformers.provence import ProvenceEncoder


def evaluate_model(model_path: str, dataset_subset: str = 'ja-full'):
    """モデルの評価を実行"""
    
    print(f"=== {dataset_subset} で学習したモデルの評価 ===")
    print(f"📁 モデルパス: {model_path}")
    
    # データセットロード
    print("📥 データセットロード中...")
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', dataset_subset)
    test_data = dataset['test']
    
    print(f"✅ テストデータ: {len(test_data):,} 件")
    
    # モデルロード
    print("🤖 モデルロード中...")
    try:
        model = ProvenceEncoder.from_pretrained(model_path)
        model.eval()
        print("✅ モデルロード完了")
    except Exception as e:
        print(f"❌ モデルロードエラー: {e}")
        return
    
    # 評価実行
    print("🔍 評価実行中...")
    
    # バッチ処理のためのデータ準備
    all_pairs = []
    all_pair_labels = []
    all_pair_teacher_scores = []
    
    for item in test_data:
        query = item['query']
        texts = item['texts']
        labels = item['labels']
        teacher_scores = item['teacher_scores_japanese-reranker-xsmall-v2']
        
        for text, label, teacher_score in zip(texts, labels, teacher_scores):
            all_pairs.append((query, text))
            all_pair_labels.append(label)
            all_pair_teacher_scores.append(teacher_score)
    
    print(f"  評価ペア数: {len(all_pairs):,}")
    
    # ランキングスコア予測
    print("  ランキングスコア予測中...")
    ranking_scores = model.predict(all_pairs, batch_size=64, show_progress_bar=True)
    all_predictions = ranking_scores
    all_labels = np.array(all_pair_labels)
    all_teacher_scores = np.array(all_pair_teacher_scores)
    
    compression_results = []
    
    # プルーニング評価（各閾値で）
    print("  プルーニング評価中...")
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        print(f"    閾値 {threshold} で評価中...")
        outputs = model.predict_with_pruning(
            all_pairs,
            batch_size=64,
            pruning_threshold=threshold,
            return_documents=True,
            show_progress_bar=True
        )
        
        for i, output in enumerate(outputs):
            compression_results.append({
                'threshold': threshold,
                'compression_ratio': output.compression_ratio,
                'ranking_score': float(output.ranking_scores),
                'label': all_pair_labels[i],
                'teacher_score': all_pair_teacher_scores[i],
                'is_positive': all_pair_labels[i] == 1
            })
    
    # 結果分析
    print("\n📊 評価結果:")
    
    # 1. ランキング性能
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # バイナリ分類として評価（閾値0.5）
    binary_preds = (all_predictions > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, binary_preds, average='binary', zero_division=0
    )
    
    print(f"\n🎯 ランキング性能:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")
    
    # 2. スコア相関
    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    
    if np.sum(pos_mask) > 0:
        pos_score_mean = np.mean(all_predictions[pos_mask])
        pos_teacher_mean = np.mean(np.array(all_teacher_scores)[pos_mask])
        print(f"  POS予測スコア平均: {pos_score_mean:.3f} (教師: {pos_teacher_mean:.3f})")
    
    if np.sum(neg_mask) > 0:
        neg_score_mean = np.mean(all_predictions[neg_mask])
        neg_teacher_mean = np.mean(np.array(all_teacher_scores)[neg_mask])
        print(f"  NEG予測スコア平均: {neg_score_mean:.3f} (教師: {neg_teacher_mean:.3f})")
    
    # 3. 圧縮性能
    print(f"\n✂️  プルーニング性能:")
    
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        threshold_results = [r for r in compression_results if r['threshold'] == threshold]
        
        if threshold_results:
            # POS/NEG別の圧縮率
            pos_results = [r for r in threshold_results if r['is_positive']]
            neg_results = [r for r in threshold_results if not r['is_positive']]
            
            all_compression = [r['compression_ratio'] for r in threshold_results]
            pos_compression = [r['compression_ratio'] for r in pos_results] if pos_results else []
            neg_compression = [r['compression_ratio'] for r in neg_results] if neg_results else []
            
            print(f"  閾値 {threshold}:")
            print(f"    全体圧縮率: {np.mean(all_compression):.1%} ± {np.std(all_compression):.1%}")
            if pos_compression:
                print(f"    POS圧縮率: {np.mean(pos_compression):.1%} ± {np.std(pos_compression):.1%}")
            if neg_compression:
                print(f"    NEG圧縮率: {np.mean(neg_compression):.1%} ± {np.std(neg_compression):.1%}")
    
    # 4. サンプル出力
    print(f"\n📝 サンプル出力 (閾値0.3):")
    
    # ポジティブサンプルを探す
    positive_items = []
    for item in test_data:
        if item['labels'][0] == 1:  # 最初のテキストがポジティブ
            positive_items.append(item)
            if len(positive_items) >= 5:
                break
    
    for i, item in enumerate(positive_items[:5]):
        query = item['query']
        pos_text = item['texts'][0]
        
        try:
            output = model.predict_with_pruning(
                (query, pos_text),
                pruning_threshold=0.3,
                return_documents=True
            )
            
            print(f"\n  サンプル {i + 1} (Positive):")
            print(f"    Query: {query[:100]}...")
            print(f"    元テキスト長: {len(pos_text)} 文字")
            if output.pruned_documents and output.pruned_documents[0]:
                pruned_doc = output.pruned_documents[0]
                print(f"    圧縮後長: {len(pruned_doc)} 文字")
                print(f"    圧縮率: {output.compression_ratio:.1%}")
                print(f"    ランキングスコア: {float(output.ranking_scores):.3f}")
                print(f"    圧縮後: {pruned_doc[:300]}{'...' if len(pruned_doc) > 300 else ''}")
            else:
                print(f"    圧縮後: (空のドキュメント)")
                print(f"    圧縮率: {output.compression_ratio:.1%}")
            
        except Exception as e:
            print(f"    ⚠️  サンプル処理エラー: {e}")
    
    print("\n✅ 評価完了")


def main():
    # 学習完了したモデルを探す
    output_dir = "./outputs/provence-ja-full"
    
    # best modelまたは最新のcheckpointを使用
    best_model_path = None
    final_model_path = os.path.join(output_dir, "final-model")
    
    # best checkpointを探す（番号が大きい方を優先）
    if os.path.exists(output_dir):
        best_checkpoints = sorted(Path(output_dir).glob("checkpoint-*-best"), 
                                 key=lambda x: int(x.name.split('-')[1]))
        if best_checkpoints:
            best_model_path = str(best_checkpoints[-1])
    
    # 評価するモデルを決定
    if best_model_path and os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"🏆 Best モデルを使用: {model_path}")
    elif os.path.exists(final_model_path):
        model_path = final_model_path
        print(f"🔚 Final モデルを使用: {model_path}")
    else:
        print(f"❌ モデルが見つかりません: {output_dir}")
        return
    
    # 評価実行
    evaluate_model(model_path, 'ja-full')


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
ja-minimal で学習した Provence モデルの評価スクリプト（修正版）
トークンレベルのプルーニングを正しく評価
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 修正版エンコーダーを使用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentence_transformers.provence.encoder_token_pruning import ProvenceEncoder

def evaluate_model(model_path: str, dataset_subset: str = 'ja-minimal'):
    """モデルの評価を実行"""
    
    print(f"=== {dataset_subset} で学習したモデルの評価（修正版） ===")
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
    
    all_predictions = []
    all_labels = []
    all_ranking_scores = []
    all_teacher_scores = []
    compression_results = []
    
    with torch.no_grad():
        for i, item in enumerate(test_data):
            if i % 10 == 0:
                print(f"  進捗: {i}/{len(test_data)}")
            
            query = item['query']
            texts = item['texts']
            labels = item['labels']
            teacher_scores = item['teacher_scores_japanese-reranker-xsmall-v2']
            
            # 各テキストペアを評価
            for j, (text, label, teacher_score) in enumerate(zip(texts, labels, teacher_scores)):
                try:
                    # ランキングスコア予測
                    ranking_score = model.predict([(query, text)])[0]
                    
                    # トークンレベルプルーニング予測（複数閾値で評価）
                    for threshold in [0.1, 0.3, 0.5, 0.7]:
                        result = model.predict_with_token_pruning(
                            (query, text),
                            pruning_threshold=threshold,
                            return_documents=True
                        )
                        
                        compression_results.append({
                            'threshold': threshold,
                            'compression_ratio': result['compression_ratio'],
                            'ranking_score': result['ranking_score'],
                            'label': label,
                            'teacher_score': teacher_score,
                            'is_positive': label == 1,
                            'num_kept_tokens': result['num_kept_tokens'],
                            'num_total_tokens': result['num_total_tokens']
                        })
                    
                    all_predictions.append(ranking_score)
                    all_labels.append(label)
                    all_ranking_scores.append(ranking_score)
                    all_teacher_scores.append(teacher_score)
                    
                except Exception as e:
                    print(f"    ⚠️  エラー (item {i}, text {j}): {e}")
                    continue
    
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
    
    # 3. 圧縮性能（トークンレベル）
    print(f"\n✂️  トークンレベルプルーニング性能:")
    
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        threshold_results = [r for r in compression_results if r['threshold'] == threshold]
        
        if threshold_results:
            # POS/NEG別の圧縮率
            pos_results = [r for r in threshold_results if r['is_positive']]
            neg_results = [r for r in threshold_results if not r['is_positive']]
            
            all_compression = [r['compression_ratio'] for r in threshold_results]
            pos_compression = [r['compression_ratio'] for r in pos_results] if pos_results else []
            neg_compression = [r['compression_ratio'] for r in neg_results] if neg_results else []
            
            # トークン統計
            all_kept = [r['num_kept_tokens'] for r in threshold_results]
            all_total = [r['num_total_tokens'] for r in threshold_results]
            
            print(f"\n  閾値 {threshold}:")
            print(f"    全体圧縮率: {np.mean(all_compression):.1%} ± {np.std(all_compression):.1%}")
            print(f"    平均保持トークン数: {np.mean(all_kept):.1f}/{np.mean(all_total):.1f}")
            
            if pos_compression:
                pos_kept = [r['num_kept_tokens'] for r in pos_results]
                pos_total = [r['num_total_tokens'] for r in pos_results]
                print(f"    POS圧縮率: {np.mean(pos_compression):.1%} ± {np.std(pos_compression):.1%}")
                print(f"    POS保持トークン数: {np.mean(pos_kept):.1f}/{np.mean(pos_total):.1f}")
            
            if neg_compression:
                neg_kept = [r['num_kept_tokens'] for r in neg_results]
                neg_total = [r['num_total_tokens'] for r in neg_results]
                print(f"    NEG圧縮率: {np.mean(neg_compression):.1%} ± {np.std(neg_compression):.1%}")
                print(f"    NEG保持トークン数: {np.mean(neg_kept):.1f}/{np.mean(neg_total):.1f}")
    
    # 4. サンプル出力
    print(f"\n📝 サンプル出力 (閾値0.3):")
    
    sample_count = 0
    for item in test_data[:3]:  # 最初の3つのサンプル
        query = item['query']
        pos_text = item['texts'][0]  # 最初のテキストは必ずPOS
        
        try:
            result = model.predict_with_token_pruning(
                (query, pos_text),
                pruning_threshold=0.3,
                return_documents=True
            )
            
            print(f"\n  サンプル {sample_count + 1}:")
            print(f"    Query: {query}")
            print(f"    元テキスト長: {len(pos_text)} 文字")
            print(f"    圧縮後長: {len(result['pruned_document'])} 文字")
            print(f"    圧縮率: {result['compression_ratio']:.1%}")
            print(f"    ランキングスコア: {result['ranking_score']:.3f}")
            print(f"    保持トークン数: {result['num_kept_tokens']}/{result['num_total_tokens']}")
            print(f"    圧縮後: {result['pruned_document'][:200]}{'...' if len(result['pruned_document']) > 200 else ''}")
            
            sample_count += 1
            
        except Exception as e:
            print(f"    ⚠️  サンプル処理エラー: {e}")
    
    print("\n✅ 評価完了")


def main():
    # 学習完了したモデルを探す
    output_dir = "./outputs/provence-ja-minimal"
    
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
    evaluate_model(model_path, 'ja-minimal')


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Streamlit WebUI for text pruning using PruningEncoder models.

Usage:
    uv run streamlit run scripts/pruning_exec_webui.py
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import torch
from bunkai import Bunkai
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder


# Model configurations from the evaluation report
MODEL_CONFIGS = [
    {
        "name": "jpre-xs-msmarco-ja-small-i4(新)",
        "path": "./output/jpre-xs-msmarco-ja-small-i4_20250725_170926/final_model",
        "optimal_threshold": 0.7,
        "description": "最新モデル、FN=4で最小誤削除"
    },
    {
        "name": "jpre-base-full-small",
        "path": "./output/jpre-base-full-small_20250716_164429/final_model",
        "optimal_threshold": 0.1,
        "description": "F2.5とFNのバランス最良"
    },
    {
        "name": "jpre-base-msmarco-ja-small-i4",
        "path": "./output/jpre-base-msmarco-ja-small-i4_20250725_193211/final_model",
        "optimal_threshold": 0.5,
        "description": "理想的なFN数"
    },
    {
        "name": "jpre-base-full-small-en",
        "path": "./output/jpre-base-full-small-en_20250716_202531/final_model",
        "optimal_threshold": 0.2,
        "description": "最高F2.5スコア"
    },
    {
        "name": "ruri-re310m-full-small-i4",
        "path": "./output/ruri-re310m-full-small-i4_20250725_180754/final_model",
        "optimal_threshold": 0.6,
        "description": "バランス型"
    },
    {
        "name": "jpre-base-msmarco-ja-small",
        "path": "./output/jpre-base-msmarco-ja-small_20250715_095147/final_model",
        "optimal_threshold": 0.3,
        "description": "高精度"
    },
    {
        "name": "jpre-xs-msmarco-ja-small-i4reverse",
        "path": "./output/jpre-xs-msmarco-ja-small-i4reverse_20250726_074524/final_model",
        "optimal_threshold": 0.4,
        "description": "新追加、バランス型"
    },
    {
        "name": "jpre-xs-msmarco-ja-small",
        "path": "./output/jpre-xs-msmarco-ja-small_20250715_084131/final_model",
        "optimal_threshold": 0.4,
        "description": "標準モデル"
    },
    {
        "name": "jpre-base-full-small-i4",
        "path": "./output/jpre-base-full-small-i4_20250725_180523/final_model",
        "optimal_threshold": 0.4,
        "description": "基本モデル"
    },
    {
        "name": "jpre-xs-full-small-i4",
        "path": "./output/jpre-xs-full-small-i4_20250725_190752/final_model",
        "optimal_threshold": 0.4,
        "description": "低精度だが高再現率"
    }
]


@st.cache_resource
def load_model(model_path: str):
    """Load and cache the model."""
    try:
        model = PruningEncoder.from_pretrained(model_path)
        model.eval()
        return model
    except Exception as e:
        st.error(f"モデルのロードに失敗しました: {str(e)}")
        return None


@st.cache_resource
def get_bunkai():
    """Initialize and cache Bunkai instance."""
    return Bunkai()


def sentence_rounding_provence(probabilities: np.ndarray, start_pos: int, end_pos: int, threshold: float) -> bool:
    """
    Provence-style sentence rounding: use average score for the entire sentence.
    """
    sentence_probs = probabilities[start_pos:end_pos]
    
    if len(sentence_probs) > 0:
        avg_prob = np.mean(sentence_probs)
        return avg_prob >= threshold
    else:
        return True


def process_text_with_model(model, query: str, sentences: List[str], threshold: float) -> List[Dict]:
    """
    Process text with the pruning model and return results for each sentence.
    """
    # Ensure sentences end with period
    processed_sentences = []
    for sent in sentences:
        if not sent.endswith('。') and not sent.endswith('.'):
            sent = sent + '。'
        processed_sentences.append(sent)
    
    # Create input
    combined_text = query + model.tokenizer.sep_token + ''.join(processed_sentences)
    
    # Tokenize
    encoding = model.tokenizer(
        combined_text,
        padding=False,
        truncation=True,
        max_length=model.max_length,
        return_tensors='pt'
    )
    
    # Move to device
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            return_dict=True
        )
        
        # Get scores based on mode
        if model.mode == "reranking_pruning":
            ranking_score = torch.sigmoid(outputs["ranking_logits"]).cpu().item()
            pruning_logits = outputs["pruning_logits"]
        else:
            ranking_score = None
            pruning_logits = outputs["pruning_logits"]
        
        # Get pruning probabilities
        pruning_probs = torch.softmax(pruning_logits, dim=-1).cpu().numpy()
        
        if pruning_probs.ndim == 3 and pruning_probs.shape[-1] == 2:
            pruning_probs = pruning_probs[0, :, 1]
        else:
            pruning_probs = pruning_probs[0]
    
    # Find sentence boundaries
    prefix = query + model.tokenizer.sep_token
    sentence_boundaries = []
    
    for i in range(len(processed_sentences)):
        cumulative_text = prefix + ''.join(processed_sentences[:i+1])
        cumulative_encoding = model.tokenizer(
            cumulative_text,
            padding=False,
            truncation=True,
            max_length=model.max_length,
            return_tensors='pt'
        )
        sentence_boundaries.append(cumulative_encoding['input_ids'].shape[1])
    
    # Get query+sep length
    query_sep_encoding = model.tokenizer(
        prefix,
        padding=False,
        truncation=False,
        return_tensors='pt'
    )
    context_start = query_sep_encoding['input_ids'].shape[1]
    
    # Create sentence ranges
    sentence_ranges = []
    prev_pos = context_start
    for end_pos in sentence_boundaries:
        sentence_ranges.append((prev_pos, end_pos))
        prev_pos = end_pos
    
    # Process each sentence
    results = []
    for i, (start_pos, end_pos) in enumerate(sentence_ranges):
        sentence_probs = pruning_probs[start_pos:end_pos]
        avg_prob = np.mean(sentence_probs) if len(sentence_probs) > 0 else 0
        is_kept = sentence_rounding_provence(pruning_probs, start_pos, end_pos, threshold)
        
        # Count tokens above threshold
        kept_tokens = np.sum(sentence_probs >= threshold)
        total_tokens = len(sentence_probs)
        
        results.append({
            'sentence': sentences[i],
            'is_kept': is_kept,
            'avg_prob': avg_prob,
            'kept_tokens': kept_tokens,
            'total_tokens': total_tokens,
            'token_probs': sentence_probs
        })
    
    return results, ranking_score


def display_results(results: List[Dict], threshold: float):
    """Display the pruning results with colored text."""
    st.write("### 処理結果")
    
    # Calculate compression rate
    total_sentences = len(results)
    kept_sentences = sum(1 for r in results if r['is_kept'])
    deleted_sentences = total_sentences - kept_sentences
    compression_rate = (deleted_sentences / total_sentences * 100) if total_sentences > 0 else 0
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総文数", total_sentences)
    with col2:
        st.metric("保持文数", kept_sentences)
    with col3:
        st.metric("圧縮率", f"{compression_rate:.1f}%")
    
    # Display sentences with colors
    st.write("### 文章表示")
    st.write("（緑: 保持、赤: 削除）")
    
    # Display each sentence separately to avoid HTML escaping issues
    for result in results:
        sentence = result['sentence']
        is_kept = result['is_kept']
        avg_prob = result['avg_prob']
        kept_tokens = result['kept_tokens']
        total_tokens = result['total_tokens']
        
        # Create tooltip text
        if is_kept:
            tooltip = f"平均スコア: {avg_prob:.3f} (閾値{threshold}以上) | {kept_tokens}/{total_tokens}トークンが閾値以上"
            color = "#155724"  # Dark green text
            bg_color = "#d4edda"  # Light green background
            border_color = "#c3e6cb"  # Green border
        else:
            tooltip = f"平均スコア: {avg_prob:.3f} (閾値{threshold}未満) | {kept_tokens}/{total_tokens}トークンが閾値以上のため削除"
            color = "#721c24"  # Dark red text
            bg_color = "#f8d7da"  # Light red background
            border_color = "#f5c6cb"  # Red border
        
        # HTML escape the sentence content
        import html
        escaped_sentence = html.escape(sentence)
        
        # Display as individual markdown element (keep HTML in single line to avoid escaping issues)
        html_str = f'<span style="color: {color}; background-color: {bg_color}; padding: 4px 8px; margin: 2px; border-radius: 4px; border: 1px solid {border_color}; display: inline-block; font-size: 14px; line-height: 1.4;" title="{tooltip}">{escaped_sentence}</span>'
        st.markdown(html_str, unsafe_allow_html=True)
    
    # Display detailed results in expander
    with st.expander("詳細結果"):
        for i, result in enumerate(results):
            st.write(f"**文 {i+1}**: {result['sentence']}")
            st.write(f"- 判定: {'保持' if result['is_kept'] else '削除'}")
            st.write(f"- 平均スコア: {result['avg_prob']:.3f}")
            st.write(f"- トークン数: {result['total_tokens']}")
            st.write(f"- 閾値以上のトークン: {result['kept_tokens']}")
            st.write("---")


def main():
    st.set_page_config(
        page_title="Text Pruning WebUI",
        page_icon="✂️",
        layout="wide"
    )
    
    st.title("✂️ Text Pruning WebUI")
    st.markdown("クエリに基づいてテキストの関連性を判定し、不要な部分を削除します。")
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("モデル設定")
        
        # Model selection
        model_options = {config["name"]: config for config in MODEL_CONFIGS}
        selected_model_name = st.selectbox(
            "モデル選択",
            options=list(model_options.keys()),
            index=0,  # Default to the first model (jpre-xs-msmarco-ja-small-i4(新))
            help="使用するPruningEncoderモデルを選択してください"
        )
        
        selected_model_config = model_options[selected_model_name]
        st.info(f"**説明**: {selected_model_config['description']}")
        st.info(f"**推奨閾値**: {selected_model_config['optimal_threshold']}")
    
    # Main input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "クエリ（検索質問）",
            placeholder="例: 機械学習とは何ですか？",
            help="テキストの関連性を判定するための質問を入力してください"
        )
    
    with col2:
        threshold = st.number_input(
            "閾値",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="関連性の判定閾値（0-1）。値が小さいほど多くの文を保持します"
        )
    
    text = st.text_area(
        "テキスト",
        height=200,
        placeholder="関連性を判定したいテキストを入力してください...",
        help="このテキストをbunkaiで文分割し、各文の関連性を判定します"
    )
    
    # Process button
    if st.button("処理実行", type="primary", disabled=not (query and text)):
        with st.spinner("処理中..."):
            # Load model
            model = load_model(selected_model_config["path"])
            if model is None:
                st.error("モデルのロードに失敗しました。モデルパスを確認してください。")
                return
            
            # Initialize Bunkai
            bunkai = get_bunkai()
            
            # Split text into sentences
            sentences = list(bunkai(text))
            
            if not sentences:
                st.warning("文が検出されませんでした。")
                return
            
            # Process with model
            try:
                results, ranking_score = process_text_with_model(model, query, sentences, threshold)
                
                # Display ranking score if available
                if ranking_score is not None:
                    st.info(f"ランキングスコア: {ranking_score:.3f}")
                
                # Display results
                display_results(results, threshold)
                
            except Exception as e:
                st.error(f"処理中にエラーが発生しました: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Instructions
    with st.expander("使い方"):
        st.markdown("""
        1. **モデル選択**: サイドバーから使用するモデルを選択します
        2. **クエリ入力**: テキストの関連性を判定するための質問を入力します
        3. **テキスト入力**: 判定したいテキストを入力します
        4. **閾値設定**: 関連性の判定閾値を設定します（デフォルト: 0.5）
        5. **処理実行**: ボタンをクリックして処理を実行します
        
        **結果の見方**:
        - 緑色: クエリに関連があり、保持される文
        - 赤色: クエリに関連がなく、削除される文
        - マウスホバー: 詳細な判定情報が表示されます
        """)


if __name__ == "__main__":
    main()
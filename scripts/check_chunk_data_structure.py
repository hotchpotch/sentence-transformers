#!/usr/bin/env python3
"""
Check chunk data structure for chunk-based evaluation
"""

from datasets import load_dataset
import json

def check_data_structure():
    """Check the structure of chunk_pos and relevant_chunks in the dataset"""
    
    print("=== Checking Data Structure ===")
    
    # Load minimal dataset for quick check
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-minimal')
    
    # Check train data
    train_data = dataset['train']
    print(f"Train data size: {len(train_data)}")
    
    # Look at first few examples
    for i in range(min(3, len(train_data))):
        example = train_data[i]
        print(f"\n--- Example {i} ---")
        print(f"Query: {example['query'][:100]}...")
        print(f"Number of texts: {len(example['texts'])}")
        
        if 'chunks_pos' in example:
            print(f"chunks_pos type: {type(example['chunks_pos'])}")
            print(f"chunks_pos length: {len(example['chunks_pos'])}")
            print(f"chunks_pos[0] sample: {example['chunks_pos'][0][:3] if example['chunks_pos'][0] else 'Empty'}")
        
        if 'relevant_chunks' in example:
            print(f"relevant_chunks type: {type(example['relevant_chunks'])}")
            print(f"relevant_chunks length: {len(example['relevant_chunks'])}")
            print(f"relevant_chunks[0] sample: {example['relevant_chunks'][0][:10] if example['relevant_chunks'][0] else 'Empty'}")
        
        # Check relationship between chunks_pos and relevant_chunks
        if 'chunks_pos' in example and 'relevant_chunks' in example:
            for j in range(min(2, len(example['chunks_pos']))):
                print(f"  Text {j}:")
                print(f"    Text length: {len(example['texts'][j])}")
                if example['chunks_pos'][j]:
                    print(f"    chunks_pos: {example['chunks_pos'][j]}")
                    print(f"    relevant_chunks: {example['relevant_chunks'][j]}")
                    
                    # Show which chunks are relevant
                    if example['relevant_chunks'][j]:
                        relevant_indices = [k for k, is_rel in enumerate(example['relevant_chunks'][j]) if is_rel]
                        print(f"    Relevant chunk indices: {relevant_indices}")
                        
                        # Show chunk content for relevant chunks
                        for chunk_idx in relevant_indices[:2]:  # Show first 2 relevant chunks
                            if chunk_idx < len(example['chunks_pos'][j]):
                                start, end = example['chunks_pos'][j][chunk_idx]
                                chunk_text = example['texts'][j][start:end]
                                print(f"      Chunk {chunk_idx} ({start}-{end}): {chunk_text[:100]}...")

def check_tokenizer_behavior():
    """Check how chunks align with tokenization"""
    
    print("\n=== Checking Tokenizer Alignment ===")
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("hotchpotch/japanese-reranker-xsmall-v2")
    
    # Load minimal dataset
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-minimal')
    example = dataset['train'][0]
    
    query = example['query']
    text = example['texts'][0]
    chunks_pos = example['chunks_pos'][0]
    relevant_chunks = example['relevant_chunks'][0]
    
    print(f"Query: {query}")
    print(f"Text: {text[:200]}...")
    print(f"Chunks pos: {chunks_pos}")
    print(f"Relevant chunks: {relevant_chunks}")
    
    # Tokenize the pair
    encoded = tokenizer(
        query, text,  # Separate arguments instead of tuple
        return_offsets_mapping=True,
        max_length=512,
        truncation=True,
        padding=False
    )
    
    input_ids = encoded['input_ids']
    offset_mapping = encoded['offset_mapping']
    
    print(f"\nTokenized length: {len(input_ids)}")
    print(f"Offset mapping length: {len(offset_mapping)}")
    
    # Find document start (after </s> </s>)
    eos_token_id = tokenizer.eos_token_id or 2
    sep_positions = [i for i, token_id in enumerate(input_ids) if token_id == eos_token_id]
    
    if len(sep_positions) >= 2:
        doc_start_token = sep_positions[0] + 2  # Skip </s> <s>
        doc_end_token = sep_positions[1]
        
        print(f"Document token range: {doc_start_token} - {doc_end_token}")
        
        # Map chunks to token positions
        doc_offsets = offset_mapping[doc_start_token:doc_end_token]
        
        print(f"\nChunk to token mapping:")
        for i, (start_char, end_char) in enumerate(chunks_pos):
            # Find tokens that overlap with this chunk
            chunk_tokens = []
            for j, (token_start, token_end) in enumerate(doc_offsets):
                if token_start != 0 and token_end != 0:  # Skip special tokens
                    if token_start < end_char and token_end > start_char:  # Overlap
                        chunk_tokens.append(j + doc_start_token)
            
            is_relevant = relevant_chunks[i] if i < len(relevant_chunks) else False
            chunk_text = text[start_char:end_char]
            
            print(f"  Chunk {i} ({'relevant' if is_relevant else 'not relevant'}): {start_char}-{end_char}")
            print(f"    Text: {chunk_text[:100]}...")
            print(f"    Token range: {chunk_tokens[:5]}{'...' if len(chunk_tokens) > 5 else ''}")


if __name__ == "__main__":
    check_data_structure()
    check_tokenizer_behavior()
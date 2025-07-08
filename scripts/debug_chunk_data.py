#!/usr/bin/env python3
"""
Debug chunk data structure
"""

from datasets import load_dataset

def main():
    # Load minimal dataset
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-minimal')
    train_data = dataset['train']
    
    # Check relevant_chunks values
    for i in range(min(5, len(train_data))):
        example = train_data[i]
        print(f"Example {i}:")
        for j, relevant_chunks in enumerate(example['relevant_chunks']):
            print(f"  Text {j}: relevant_chunks = {relevant_chunks}")
            if relevant_chunks:
                unique_values = set(relevant_chunks)
                print(f"    Unique values: {unique_values}")
    
if __name__ == "__main__":
    main()
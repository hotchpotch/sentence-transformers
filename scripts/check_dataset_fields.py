#!/usr/bin/env python3
"""
Check dataset fields for debugging
"""

from datasets import load_dataset

def main():
    # Load minimal dataset
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-minimal')
    train_data = dataset['train']
    
    # Check first example
    example = train_data[0]
    print("Available fields in dataset:")
    for key, value in example.items():
        print(f"  {key}: {type(value)} - {str(value)[:100]}...")
    
if __name__ == "__main__":
    main()
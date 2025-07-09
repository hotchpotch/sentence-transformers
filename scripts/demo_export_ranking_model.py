#!/usr/bin/env python
"""
Demo: Export ranking model for standard Transformers usage.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def demo_export_and_use():
    """Demonstrate exporting and using ranking model."""
    
    print("="*60)
    print("Demo: Export Ranking Model")
    print("="*60)
    
    # 1. Load PruningEncoder
    print("\n1. Loading PruningEncoder...")
    model_path = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"
    pruning_model = PruningEncoder.from_pretrained(model_path)
    print(f"   ✓ Loaded: {type(pruning_model).__name__}")
    
    # 2. Export ranking model
    print("\n2. Exporting ranking model...")
    export_path = "./output/exported_ranking_model"
    pruning_model.export_ranking_model(export_path)
    
    # 3. Load exported model with AutoModel
    print("\n3. Loading exported model with AutoModel...")
    ranking_model = AutoModelForSequenceClassification.from_pretrained(export_path)
    tokenizer = AutoTokenizer.from_pretrained(export_path)
    print(f"   ✓ Loaded: {type(ranking_model).__name__}")
    
    # 4. Compare outputs
    print("\n4. Comparing outputs...")
    query = "機械学習について"
    document = "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。"
    
    # PruningEncoder score
    pruning_score = pruning_model.predict([(query, document)], apply_pruning=False)[0]
    print(f"   PruningEncoder score: {pruning_score:.4f}")
    
    # Exported model score
    inputs = tokenizer(query, document, return_tensors="pt", truncation=True)
    device = next(ranking_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = ranking_model(**inputs)
        exported_score = torch.sigmoid(outputs.logits).item()
    
    print(f"   Exported model score: {exported_score:.4f}")
    print(f"   ✓ Scores match: {abs(pruning_score - exported_score) < 0.0001}")
    
    # 5. Show different loading methods
    print("\n5. Summary of Loading Methods:")
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    PruningEncoder Loading Methods                    │
├─────────────────────────────────────────────────────────────────────┤
│ Method                │ Import Required      │ Features              │
├─────────────────────┼────────────────────┼─────────────────────┤
│ 1. Full PruningEncoder│ sentence_transformers│ Ranking + Pruning     │
│    from_pretrained()  │ .pruning            │                       │
├─────────────────────┼────────────────────┼─────────────────────┤
│ 2. Ranking Model      │ transformers only    │ Ranking only          │
│    /ranking_model     │                     │ (Standard model)      │
├─────────────────────┼────────────────────┼─────────────────────┤
│ 3. Exported Model     │ transformers only    │ Ranking only          │
│    export_ranking()   │                     │ (Clean export)        │
├─────────────────────┼────────────────────┼─────────────────────┤
│ 4. CrossEncoder       │ sentence_transformers│ Ranking + convenience │
│    CrossEncoder()     │                     │ methods               │
└─────────────────────┴────────────────────┴─────────────────────┘
""")


def demo_direct_ranking_model_access():
    """Show direct access to ranking_model subdirectory."""
    
    print("\n" + "="*60)
    print("Direct Ranking Model Access")
    print("="*60)
    
    model_path = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"
    
    print("\nNo export needed! Just point to ranking_model subdirectory:")
    print(f"""
```python
# Option 1: Direct subdirectory access
model = AutoModelForSequenceClassification.from_pretrained(
    "{model_path}/ranking_model"
)

# Option 2: Export for cleaner structure
pruning_model = PruningEncoder.from_pretrained("{model_path}")
pruning_model.export_ranking_model("./my_ranking_model")
model = AutoModelForSequenceClassification.from_pretrained("./my_ranking_model")
```
""")


def main():
    """Run demo."""
    demo_export_and_use()
    demo_direct_ranking_model_access()
    
    print("\n" + "="*60)
    print("🎉 CONCLUSION")
    print("="*60)
    print("""
✅ Base ranking models are fully accessible!
   - Already saved in /ranking_model subdirectory
   - Can be exported separately with export_ranking_model()
   - Work with standard AutoModelForSequenceClassification
   - No special imports needed!

✅ Users have complete flexibility:
   - Use full PruningEncoder when pruning is needed
   - Use base model for simple ranking tasks
   - Same model weights, different interfaces
""")


if __name__ == "__main__":
    main()
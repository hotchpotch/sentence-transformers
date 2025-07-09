"""
Implementation of export_ranking_model functionality for PruningEncoder.

This script shows how to add the ability to export just the ranking model
from a PruningEncoder, making it loadable with AutoModelForSequenceClassification.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Union, Optional


def add_export_methods_to_pruning_encoder():
    """
    Add these methods to the PruningEncoder class in 
    sentence_transformers/pruning/encoder.py
    """
    
    code = '''
    def save_pretrained(
        self, 
        save_directory: Union[str, Path], 
        export_ranking_model: bool = False,
        **kwargs
    ):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model
            export_ranking_model: If True and mode is 'reranking_pruning', 
                                also exports the ranking model as a standalone
                                model that can be loaded with AutoModelForSequenceClassification
            **kwargs: Additional arguments passed to the underlying save methods
        """
        # Call the existing save_pretrained implementation
        super().save_pretrained(save_directory, **kwargs)
        
        # Add export functionality
        if export_ranking_model and self.mode == "reranking_pruning":
            ranking_export_dir = Path(save_directory) / "ranking_model_standalone" 
            self._export_ranking_model_internal(ranking_export_dir)
            
            # Add a README for clarity
            self._create_model_readme(save_directory)
    
    def export_ranking_model(self, save_directory: Union[str, Path]):
        """
        Export just the ranking model component.
        
        This creates a standalone model that can be loaded with:
        `AutoModelForSequenceClassification.from_pretrained(save_directory)`
        
        Args:
            save_directory: Directory to save the exported ranking model
            
        Raises:
            ValueError: If the model mode is not 'reranking_pruning'
            
        Example:
            >>> pruning_encoder = PruningEncoder.from_pretrained("path/to/model")
            >>> pruning_encoder.export_ranking_model("path/to/ranking_model")
            >>> # Now you can load it with transformers
            >>> from transformers import AutoModelForSequenceClassification
            >>> ranking_model = AutoModelForSequenceClassification.from_pretrained("path/to/ranking_model")
        """
        if self.mode != "reranking_pruning":
            raise ValueError(
                f"export_ranking_model is only available for reranking_pruning mode, "
                f"but current mode is '{self.mode}'"
            )
        
        save_directory = Path(save_directory)
        self._export_ranking_model_internal(save_directory)
        
        print(f"✓ Ranking model exported to: {save_directory}")
        print(f"  Load with: AutoModelForSequenceClassification.from_pretrained('{save_directory}')")
    
    def _export_ranking_model_internal(self, save_directory: Path):
        """Internal method to export the ranking model."""
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save the ranking model and tokenizer
        self.ranking_model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        # Add export metadata
        metadata = {
            "exported_from": "PruningEncoder", 
            "mode": self.mode,
            "original_model_path": self.model_name_or_path,
            "export_timestamp": datetime.now().isoformat(),
            "pruning_encoder_version": getattr(self, "__version__", "unknown"),
            "usage": {
                "description": "This is a ranking model exported from a PruningEncoder",
                "load_with": "AutoModelForSequenceClassification.from_pretrained()",
                "original_capabilities": {
                    "ranking": True,
                    "pruning": False
                }
            }
        }
        
        with open(save_directory / "export_info.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _create_model_readme(self, save_directory: Union[str, Path]):
        """Create a README explaining the model structure."""
        save_directory = Path(save_directory)
        
        readme_content = """# PruningEncoder Model

This directory contains a PruningEncoder model that can be loaded in multiple ways:

## Loading Options

### 1. Full PruningEncoder Model (with ranking + pruning capabilities)
```python
from sentence_transformers.pruning import PruningEncoder

# Load the full model with pruning capabilities
model = PruningEncoder.from_pretrained("./")

# Use for both ranking and pruning
outputs = model(input_ids, attention_mask)
ranking_scores = outputs["ranking_scores"]
pruning_scores = outputs["pruning_scores"]
```

### 2. Ranking Model Only (standard Transformers model)
```python
from transformers import AutoModelForSequenceClassification

# Option A: Load from subdirectory
model = AutoModelForSequenceClassification.from_pretrained("./ranking_model")

# Option B: Load from standalone export (if available)
model = AutoModelForSequenceClassification.from_pretrained("./ranking_model_standalone")

# Use for ranking only
outputs = model(input_ids, attention_mask)
ranking_scores = outputs.logits
```

## Directory Structure

- `config.json` - Full PruningEncoder configuration
- `pruning_encoder_config.json` - PruningEncoder-specific config
- `ranking_model/` - The base ranking model (can be loaded independently)
- `ranking_model_standalone/` - Exported ranking model (if `export_ranking_model=True` was used)
- `pruning_head/` - The pruning head weights
- `modeling_pruning_encoder.py` - Model implementation

## Model Information

- **Mode**: {mode}
- **Base Model**: {base_model}
- **Architecture**: PruningEncoder
- **Capabilities**: Ranking + Query-dependent Pruning

## Export Ranking Model

To export just the ranking model:

```python
model = PruningEncoder.from_pretrained("./")
model.export_ranking_model("path/to/export")
```
"""
        
        # Fill in the template
        config_path = save_directory / "pruning_encoder_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                mode = config.get("mode", "unknown")
                base_model = config.get("model_name_or_path", "unknown")
        else:
            mode = "unknown"
            base_model = "unknown"
        
        readme_content = readme_content.format(mode=mode, base_model=base_model)
        
        with open(save_directory / "README.md", "w") as f:
            f.write(readme_content)
'''
    
    return code


def create_usage_examples():
    """Create usage examples for the new functionality."""
    
    examples = '''
# Usage Examples

## Example 1: Save with ranking model export

```python
from sentence_transformers.pruning import PruningEncoder

# Load or train your model
model = PruningEncoder.from_pretrained("path/to/trained/model")

# Save with ranking model export
model.save_pretrained(
    "path/to/save", 
    export_ranking_model=True  # This creates ranking_model_standalone/
)
```

## Example 2: Export ranking model separately

```python
# Load the full model
model = PruningEncoder.from_pretrained("path/to/model")

# Export just the ranking component
model.export_ranking_model("path/to/ranking_only")

# Now you can use it with transformers
from transformers import AutoModelForSequenceClassification
ranking_model = AutoModelForSequenceClassification.from_pretrained("path/to/ranking_only")
```

## Example 3: Loading different views

```python
# Full model with pruning
from sentence_transformers.pruning import PruningEncoder
full_model = PruningEncoder.from_pretrained("saved_model/")

# Just the ranking model - Option 1
from transformers import AutoModelForSequenceClassification
ranking_only_v1 = AutoModelForSequenceClassification.from_pretrained("saved_model/ranking_model")

# Just the ranking model - Option 2 (if exported)
ranking_only_v2 = AutoModelForSequenceClassification.from_pretrained("saved_model/ranking_model_standalone")
```

## Example 4: Use in production

```python
# For RAG applications that need pruning
pruning_model = PruningEncoder.from_pretrained("model_path/")
outputs = pruning_model(query_inputs, document_inputs)
pruned_docs = apply_pruning(documents, outputs["pruning_scores"])

# For simple reranking without pruning
reranker = AutoModelForSequenceClassification.from_pretrained("model_path/ranking_model")
scores = reranker(inputs).logits
```
'''
    
    return examples


def main():
    print("="*60)
    print("IMPLEMENTATION PLAN: Export Ranking Model from PruningEncoder")
    print("="*60)
    
    print("\n1. CURRENT STATE:")
    print("   - PruningEncoder saves ranking model in 'ranking_model/' subdirectory")
    print("   - This subdirectory CAN already be loaded with AutoModelForSequenceClassification")
    print("   - But this isn't well documented or obvious to users")
    
    print("\n2. PROPOSED ENHANCEMENTS:")
    print("   a) Add export_ranking_model() method for explicit export")
    print("   b) Add export_ranking_model parameter to save_pretrained()")
    print("   c) Create README.md automatically to document loading options")
    print("   d) Add metadata files to track exports")
    
    print("\n3. IMPLEMENTATION:")
    print(add_export_methods_to_pruning_encoder())
    
    print("\n4. USAGE EXAMPLES:")
    print(create_usage_examples())
    
    print("\n5. BENEFITS:")
    print("   - Clear API for exporting ranking models")
    print("   - Better documentation for users")
    print("   - Maintains backward compatibility")
    print("   - Supports both use cases (full model vs ranking only)")
    
    print("\n6. FILES TO MODIFY:")
    print("   - sentence_transformers/pruning/encoder.py - Add new methods")
    print("   - tests/pruning/test_pruning_encoder.py - Add tests for export")
    print("   - docs/ - Add documentation for the feature")


if __name__ == "__main__":
    main()
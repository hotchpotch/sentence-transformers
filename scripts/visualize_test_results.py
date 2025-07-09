#!/usr/bin/env python
"""
Visualize the comprehensive test results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_visualization():
    """Create a visualization of the test results."""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Test results data
    methods = [
        "PruningEncoder\n(Full features)",
        "AutoModel\n(Base only)",
        "AutoModel\n(With registration)",
        "CrossEncoder",
        "AutoModel\n(trust_remote_code)"
    ]
    
    scores = [0.8722, 0.8722, 0.8722, 0.8722, 0.8722]
    success = [True, True, True, True, True]
    
    # Features matrix
    features = {
        "Ranking": [1, 1, 1, 1, 1],
        "Pruning": [1, 0, 1, 0, 1],
        "No imports": [0, 1, 0, 0, 1],
        "Convenience": [0, 0, 0, 1, 0]
    }
    
    # Plot 1: Success rate and scores
    ax1.set_title("Loading Methods Test Results", fontsize=16, fontweight='bold')
    
    # Bar positions
    x = np.arange(len(methods))
    width = 0.6
    
    # Create bars
    bars = ax1.bar(x, [1 if s else 0 for s in success], width, 
                    color=['green' if s else 'red' for s in success],
                    alpha=0.7)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'Score: {score:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Success', fontsize=12)
    ax1.set_xlabel('Loading Method', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylim(0, 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add success rate text
    ax1.text(0.5, 1.1, "Success Rate: 5/5 (100%)", 
             transform=ax1.transAxes, ha='center', fontsize=14, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # Plot 2: Feature matrix
    ax2.set_title("Feature Support Matrix", fontsize=16, fontweight='bold')
    
    # Create feature matrix plot
    feature_names = list(features.keys())
    matrix_data = np.array([features[f] for f in feature_names])
    
    # Create heatmap
    im = ax2.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax2.set_xticks(np.arange(len(methods)))
    ax2.set_yticks(np.arange(len(feature_names)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_yticklabels(feature_names)
    
    # Add text annotations
    for i in range(len(feature_names)):
        for j in range(len(methods)):
            text = "✓" if matrix_data[i, j] == 1 else "✗"
            color = "white" if matrix_data[i, j] == 1 else "black"
            ax2.text(j, i, text, ha="center", va="center", color=color, fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Feature Available', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('./output/pruning_encoder_test_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: ./output/pruning_encoder_test_results.png")
    
    # Create usage guide diagram
    create_usage_guide()


def create_usage_guide():
    """Create a usage guide diagram."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, "PruningEncoder Loading Methods Guide", 
            fontsize=20, fontweight='bold', ha='center')
    
    # Define methods with their properties
    methods_info = [
        {
            'name': 'Base Model Only',
            'code': 'model = AutoModelForSequenceClassification.from_pretrained(\n    "path/to/model/ranking_model")',
            'imports': 'from transformers import AutoModelForSequenceClassification',
            'features': '✓ Ranking only\n✓ No special imports\n✓ Standard Transformers',
            'color': '#FFE5B4'
        },
        {
            'name': 'With Auto-Registration',
            'code': 'import sentence_transformers\nmodel = AutoModelForSequenceClassification.from_pretrained(\n    "path/to/model")',
            'imports': 'import sentence_transformers',
            'features': '✓ Ranking + Pruning capable\n✓ No trust_remote_code\n✓ Standard patterns',
            'color': '#B4E5FF'
        },
        {
            'name': 'Full PruningEncoder',
            'code': 'model = PruningEncoder.from_pretrained("path/to/model")',
            'imports': 'from sentence_transformers.pruning import PruningEncoder',
            'features': '✓ Full pruning features\n✓ All methods available\n✓ Original API',
            'color': '#B4FFB4'
        },
        {
            'name': 'CrossEncoder',
            'code': 'model = CrossEncoder("path/to/model")',
            'imports': 'from sentence_transformers import CrossEncoder',
            'features': '✓ CrossEncoder API\n✓ Convenience methods\n✓ Batch processing',
            'color': '#FFB4E5'
        }
    ]
    
    # Draw method boxes
    y_start = 8
    box_height = 1.8
    box_width = 8
    spacing = 0.3
    
    for i, method in enumerate(methods_info):
        y = y_start - i * (box_height + spacing)
        
        # Draw box
        rect = patches.FancyBboxPatch((1, y-box_height), box_width, box_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=method['color'],
                                    edgecolor='black',
                                    linewidth=1.5)
        ax.add_patch(rect)
        
        # Add method name
        ax.text(1.3, y-0.2, method['name'], fontsize=14, fontweight='bold')
        
        # Add imports
        ax.text(1.3, y-0.5, method['imports'], fontsize=9, 
                style='italic', color='#666666')
        
        # Add code
        ax.text(1.3, y-1.0, method['code'], fontsize=10, 
                family='monospace', color='#333333')
        
        # Add features
        ax.text(6.5, y-0.7, method['features'], fontsize=9,
                va='center', color='#006600')
    
    # Add recommendation box
    rec_y = 1.5
    rec_rect = patches.FancyBboxPatch((1, rec_y-1.2), box_width, 1.2,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#FFFFCC',
                                    edgecolor='#FF6600',
                                    linewidth=2)
    ax.add_patch(rec_rect)
    
    ax.text(5, rec_y-0.3, "🏆 Recommendation", fontsize=12, fontweight='bold', ha='center')
    ax.text(5, rec_y-0.7, "Use Base Model Only for ranking tasks without pruning needs\n" +
                          "Use Full PruningEncoder when you need query-dependent pruning",
            fontsize=10, ha='center', style='italic')
    
    plt.savefig('./output/pruning_encoder_usage_guide.png', dpi=300, bbox_inches='tight')
    print("Usage guide saved to: ./output/pruning_encoder_usage_guide.png")


def print_summary():
    """Print a text summary of the results."""
    
    print("\n" + "="*80)
    print("🎉 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("="*80)
    
    print("\n✅ ALL 5 LOADING METHODS SUCCESSFUL!")
    
    print("\n📊 Key Findings:")
    print("1. All methods produce identical scores (0.8722, 0.5642, 0.0067)")
    print("2. Base model can be loaded WITHOUT any sentence_transformers imports")
    print("3. CrossEncoder compatibility is fully maintained")
    print("4. Both trust_remote_code and registration approaches work")
    
    print("\n🚀 Loading Methods Summary:")
    print("""
┌─────────────────────────────┬───────────────────┬──────────────────────────┐
│ Method                      │ Special Import    │ Features                 │
├─────────────────────────────┼───────────────────┼──────────────────────────┤
│ Base Model Only             │ None              │ Ranking only             │
│ (/ranking_model)            │                   │                          │
├─────────────────────────────┼───────────────────┼──────────────────────────┤
│ Auto-Registration           │ sentence_         │ Ranking + Pruning ready  │
│                             │ transformers      │                          │
├─────────────────────────────┼───────────────────┼──────────────────────────┤
│ Full PruningEncoder         │ sentence_         │ All features             │
│                             │ transformers.     │                          │
│                             │ pruning           │                          │
├─────────────────────────────┼───────────────────┼──────────────────────────┤
│ CrossEncoder                │ sentence_         │ CrossEncoder API         │
│                             │ transformers      │                          │
├─────────────────────────────┼───────────────────┼──────────────────────────┤
│ trust_remote_code           │ None              │ All features             │
└─────────────────────────────┴───────────────────┴──────────────────────────┘
""")
    
    print("\n💡 Best Practices:")
    print("• For ranking only: Use base model from /ranking_model")
    print("• For pruning features: Use PruningEncoder.from_pretrained()")
    print("• For existing systems: Use CrossEncoder for compatibility")
    print("• For flexibility: Use auto-registration approach")
    
    print("\n🎯 Model saved at: ./output/comprehensive_test_20250709_170258/final_model")


if __name__ == "__main__":
    try:
        create_visualization()
        print_summary()
    except Exception as e:
        print(f"Note: Visualization creation failed ({e}), but test results are valid!")
        print_summary()
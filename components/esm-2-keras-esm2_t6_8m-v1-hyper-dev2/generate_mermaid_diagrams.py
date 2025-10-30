#!/usr/bin/env python3
"""
Mermaid Diagram Generator for ESM-2 Hypergraph
Generates standalone mermaid diagram files for the ESM-2 hypergraph system.
"""

import os
import argparse
from esm2_hypergraph import create_esm2_hypergraph
from hypergraph_visualizer import HypergraphVisualizer


def generate_all_diagrams(output_dir: str = "mermaid_diagrams"):
    """Generate all mermaid diagrams and save them to separate files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ESM-2 model configuration
    config = {
        "name": "esm_backbone",
        "trainable": True,
        "vocabulary_size": 33,
        "num_layers": 6,
        "num_heads": 20,
        "hidden_dim": 320,
        "intermediate_dim": 1280,
        "dropout": 0,
        "max_wavelength": 10000,
        "use_bias": True,
        "activation": "gelu",
        "layer_norm_eps": 0.00001,
        "use_pre_layer_norm": False,
        "position_embedding_type": "rotary",
        "max_sequence_length": 1026,
        "pad_token_id": 1
    }
    
    # Create hypergraph and visualizer
    hypergraph = create_esm2_hypergraph(config)
    visualizer = HypergraphVisualizer(hypergraph)
    
    diagrams = {
        "architecture_flow": {
            "filename": "model_architecture_flow.mmd",
            "title": "ESM-2 Model Architecture Flow",
            "content": visualizer.generate_mermaid_architecture()
        },
        "hypergraph_structure": {
            "filename": "hypergraph_structure.mmd", 
            "title": "Hypergraph Node and Edge Distribution",
            "content": visualizer.generate_mermaid_hypergraph_structure()
        },
        "component_architecture": {
            "filename": "component_architecture.mmd",
            "title": "Component Class Architecture",
            "content": visualizer.generate_mermaid_component_architecture()
        },
        "query_flow": {
            "filename": "query_processing_flow.mmd",
            "title": "Query Processing Flow",
            "content": visualizer.generate_mermaid_query_flow()
        }
    }
    
    # Generate individual diagram files
    for diagram_key, diagram_info in diagrams.items():
        filepath = os.path.join(output_dir, diagram_info["filename"])
        
        with open(filepath, 'w') as f:
            f.write(f"---\n")
            f.write(f"title: {diagram_info['title']}\n")
            f.write(f"---\n\n")
            f.write(diagram_info["content"])
        
        print(f"✓ Generated {diagram_info['filename']}")
    
    # Generate combined diagram file
    combined_filepath = os.path.join(output_dir, "all_diagrams.md")
    with open(combined_filepath, 'w') as f:
        f.write("# ESM-2 Hypergraph Mermaid Diagrams\n\n")
        f.write("This document contains all mermaid diagrams for the ESM-2 hypergraph system.\n\n")
        
        for diagram_key, diagram_info in diagrams.items():
            f.write(f"## {diagram_info['title']}\n\n")
            f.write("```mermaid\n")
            f.write(diagram_info["content"])
            f.write("\n```\n\n")
    
    print(f"✓ Generated combined diagrams file: all_diagrams.md")
    
    # Generate system overview diagram
    system_overview = generate_system_overview_diagram()
    overview_filepath = os.path.join(output_dir, "system_overview.mmd")
    with open(overview_filepath, 'w') as f:
        f.write("---\n")
        f.write("title: ESM-2 Hypergraph System Overview\n")
        f.write("---\n\n")
        f.write(system_overview)
    
    print(f"✓ Generated system_overview.mmd")
    
    # Generate README for the mermaid diagrams directory
    readme_content = generate_mermaid_readme(diagrams)
    readme_filepath = os.path.join(output_dir, "README.md")
    with open(readme_filepath, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Generated README.md for diagram usage")
    print(f"\nAll mermaid diagrams saved to: {output_dir}/")


def generate_system_overview_diagram():
    """Generate system overview mermaid diagram"""
    return """graph TB
    subgraph "ESM-2 Hypergraph System"
        A[ESM-2 Model Config] --> B[Hypergraph Builder]
        B --> C[ESM2Hypergraph<br/>64 Nodes, 41 Edges]
        C --> D[Query Engine]
        C --> E[Visualizer]
        C --> F[JSON Export]
        
        D --> G[Analysis Results]
        E --> H[DOT Graph]
        E --> I[Reports]
        E --> M[Mermaid Diagrams]
        F --> J[Hypergraph Data]
        
        subgraph "Core Components"
            C
            D
            E
        end
        
        subgraph "Outputs"
            G
            H
            I
            J
            M
        end
    end"""


def generate_mermaid_readme(diagrams):
    """Generate README content for mermaid diagrams directory"""
    content = """# ESM-2 Hypergraph Mermaid Diagrams

This directory contains mermaid diagrams illustrating the ESM-2 hypergraph system architecture and components.

## Diagram Files

"""
    
    for diagram_key, diagram_info in diagrams.items():
        content += f"- **{diagram_info['filename']}**: {diagram_info['title']}\n"
    
    content += """
- **system_overview.mmd**: High-level system architecture overview
- **all_diagrams.md**: Combined markdown file with all diagrams

## Usage

### Viewing Diagrams

1. **GitHub**: GitHub automatically renders mermaid diagrams in markdown files
2. **VS Code**: Install the "Mermaid Preview" extension
3. **Online**: Use the [Mermaid Live Editor](https://mermaid.live/)

### Rendering Diagrams

To render diagrams as images, you can use:

```bash
# Install mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Render individual diagrams
mmdc -i model_architecture_flow.mmd -o model_architecture_flow.png
mmdc -i hypergraph_structure.mmd -o hypergraph_structure.png
mmdc -i component_architecture.mmd -o component_architecture.png
mmdc -i query_processing_flow.mmd -o query_processing_flow.png
mmdc -i system_overview.mmd -o system_overview.png

# Or batch render all
for file in *.mmd; do
    mmdc -i "$file" -o "${file%.mmd}.png"
done
```

### Integration

These diagrams can be embedded in documentation using:

```markdown
```mermaid
[paste diagram content here]
```
```

## Diagram Descriptions

### Model Architecture Flow
Shows the data flow through the ESM-2 transformer model from input tokens to output logits.

### Hypergraph Structure  
Visualizes the distribution of nodes and edges in the hypergraph representation.

### Component Architecture
Displays the class relationships and API structure of the hypergraph system.

### Query Processing Flow
Illustrates how different query types are processed by the query engine.

### System Overview
Provides a high-level view of the entire ESM-2 hypergraph system components and data flow.
"""
    
    return content


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Generate mermaid diagrams for ESM-2 hypergraph")
    parser.add_argument("--output-dir", default="mermaid_diagrams", 
                        help="Output directory for diagram files (default: mermaid_diagrams)")
    
    args = parser.parse_args()
    
    print("Generating ESM-2 Hypergraph Mermaid Diagrams...")
    print("=" * 50)
    
    generate_all_diagrams(args.output_dir)
    
    print("\nDiagram generation complete!")
    print(f"View the diagrams in: {args.output_dir}/")


if __name__ == "__main__":
    main()
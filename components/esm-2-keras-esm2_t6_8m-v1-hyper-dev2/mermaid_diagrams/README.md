# ESM-2 Hypergraph Mermaid Diagrams

This directory contains mermaid diagrams illustrating the ESM-2 hypergraph system architecture and components.

## Diagram Files

- **model_architecture_flow.mmd**: ESM-2 Model Architecture Flow
- **hypergraph_structure.mmd**: Hypergraph Node and Edge Distribution
- **component_architecture.mmd**: Component Class Architecture
- **query_processing_flow.mmd**: Query Processing Flow

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

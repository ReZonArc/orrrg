# ESM-2 Hypergraph Analysis Report

## Model Configuration
```json
{
  "name": "esm_backbone",
  "trainable": true,
  "vocabulary_size": 33,
  "num_layers": 6,
  "num_heads": 20,
  "hidden_dim": 320,
  "intermediate_dim": 1280,
  "dropout": 0,
  "max_wavelength": 10000,
  "use_bias": true,
  "activation": "gelu",
  "layer_norm_eps": 1e-05,
  "use_pre_layer_norm": false,
  "position_embedding_type": "rotary",
  "max_sequence_length": 1026,
  "pad_token_id": 1
}
```

## Architecture Overview
```
ESM-2 Architecture Hypergraph
==================================================

INPUT LAYER:
  Token Embedding (vocab=33 -> hidden=320)
  Rotary Positional Encoding

TRANSFORMER LAYER 0:
  Multi-Head Self-Attention:
    - Query/Key/Value Projections (320 -> 320)
    - 20 attention heads
    - Head dimension: 16
    - Output Projection
  + Residual Connection
  Post-Attention Layer Norm
  Feed-Forward Network:
    - Linear (320 -> 1280)
    - GELU Activation
    - Linear (1280 -> 320)
  + Residual Connection
  Post-FFN Layer Norm

TRANSFORMER LAYER 1:
  Multi-Head Self-Attention:
    - Query/Key/Value Projections (320 -> 320)
    - 20 attention heads
    - Head dimension: 16
    - Output Projection
  + Residual Connection
  Post-Attention Layer Norm
  Feed-Forward Network:
    - Linear (320 -> 1280)
    - GELU Activation
    - Linear (1280 -> 320)
  + Residual Connection
  Post-FFN Layer Norm

TRANSFORMER LAYER 2:
  Multi-Head Self-Attention:
    - Query/Key/Value Projections (320 -> 320)
    - 20 attention heads
    - Head dimension: 16
    - Output Projection
  + Residual Connection
  Post-Attention Layer Norm
  Feed-Forward Network:
    - Linear (320 -> 1280)
    - GELU Activation
    - Linear (1280 -> 320)
  + Residual Connection
  Post-FFN Layer Norm

TRANSFORMER LAYER 3:
  Multi-Head Self-Attention:
    - Query/Key/Value Projections (320 -> 320)
    - 20 attention heads
    - Head dimension: 16
    - Output Projection
  + Residual Connection
  Post-Attention Layer Norm
  Feed-Forward Network:
    - Linear (320 -> 1280)
    - GELU Activation
    - Linear (1280 -> 320)
  + Residual Connection
  Post-FFN Layer Norm

TRANSFORMER LAYER 4:
  Multi-Head Self-Attention:
    - Query/Key/Value Projections (320 -> 320)
    - 20 attention heads
    - Head dimension: 16
    - Output Projection
  + Residual Connection
  Post-Attention Layer Norm
  Feed-Forward Network:
    - Linear (320 -> 1280)
    - GELU Activation
    - Linear (1280 -> 320)
  + Residual Connection
  Post-FFN Layer Norm

TRANSFORMER LAYER 5:
  Multi-Head Self-Attention:
    - Query/Key/Value Projections (320 -> 320)
    - 20 attention heads
    - Head dimension: 16
    - Output Projection
  + Residual Connection
  Post-Attention Layer Norm
  Feed-Forward Network:
    - Linear (320 -> 1280)
    - GELU Activation
    - Linear (1280 -> 320)
  + Residual Connection
  Post-FFN Layer Norm

OUTPUT LAYER:
  Final Layer Norm
  Output Head

```

## Model Architecture Flow
```mermaid
graph TD
    A[Input Tokens] --> B[Token Embedding<br/>vocab=33→hidden=320]
    A --> C[Rotary Positional<br/>Encoding]

    B --> D0[Transformer Layer 0]
    C --> D0

    D0 --> D1[Transformer Layer 1]
    D1 --> D2[Transformer Layer 2]
    D2 --> D3[Transformer Layer 3]
    D3 --> D4[Transformer Layer 4]
    D4 --> D5[Transformer Layer 5]

    D5 --> E[Final LayerNorm]
    E --> F[Output Head]
    F --> G[Output Logits<br/>hidden=320→vocab=33]

    subgraph "Each Transformer Layer"
        H[Multi-Head Attention<br/>20 heads, dim=16] --> I[Residual + LayerNorm]
        I --> J[Feed-Forward Network<br/>320→1280→320]
        J --> K[Residual + LayerNorm]
    end
```

## Hypergraph Structure
```mermaid
graph LR
    subgraph "Hypergraph Components"
        subgraph "Node Types (64 total)"
            EMBEDDING[Embedding: 1]
            POSITIONAL[Positional: 1]
            LINEAR[Linear: 36]
            ATTENTION[Attention: 6]
            LAYERNORM[Layernorm: 13]
            ACTIVATION[Activation: 6]
            OUTPUT[Output: 1]
        end

        subgraph "Edge Types (41 total)"
            DATAFLOW[Data Flow: 11]
            ATTENTIONPREP[Attention Prep: 6]
            ATTENTION[Attention: 6]
            RESIDUAL[Residual: 12]
            FEEDFORWARD[Feed Forward: 6]
        end
    end
```

## Component Architecture
```mermaid
classDiagram
    class ESM2Hypergraph {
        +get_statistics()
        +to_dict()
        +save_to_json()
        +visualize_summary()
    }

    class HypergraphQueryEngine {
        +find_nodes_by_type()
        +find_nodes_by_layer()
        +get_computational_path()
        +analyze_parameter_flow()
        +find_bottlenecks()
    }

    class HypergraphVisualizer {
        +create_layer_diagram()
        +generate_dot_graph()
        +generate_mermaid_diagrams()
        +create_connectivity_matrix()
        +find_critical_paths()
    }

    HypergraphQueryEngine --> ESM2Hypergraph
    HypergraphVisualizer --> ESM2Hypergraph
```

## Query Processing Flow
```mermaid
flowchart LR
    A[Query Request] --> B{Query Type}

    B -->|stats| C[Get Statistics<br/>Nodes, Edges, Types]
    B -->|attention| D[Analyze Attention<br/>Structure & Patterns]
    B -->|params| E[Parameter Flow<br/>Analysis]
    B -->|bottlenecks| F[Find Bottlenecks<br/>High Fan-in/out]
    B -->|path| G[Find Computational<br/>Path A→B]
    B -->|subgraph| H[Extract Layer<br/>Subgraph]

    C --> I[JSON Output]
    D --> I
    E --> I
    F --> I
    G --> J[Path Visualization]
    H --> K[Subgraph Export]
```

## Hypergraph Statistics
- **Total Nodes**: 64
- **Total Hyperedges**: 41
- **Maximum Hyperedge Size**: 4

### Node Types Distribution
- embedding: 1
- positional: 1
- linear: 36
- attention: 6
- layernorm: 13
- activation: 6
- output: 1

### Edge Types Distribution
- data_flow: 11
- attention_prep: 6
- attention: 6
- residual: 12
- feed_forward: 6

## Connectivity Analysis
- **Total Connections**: 70

## Layer-wise Analysis
### Input Layer
- Nodes: 2
- Edges: 4
- Edge Types:
  - data_flow: 3
  - residual: 1

### Transformer Layer 0
- Nodes: 10
- Edges: 8
- Edge Types:
  - data_flow: 2
  - attention_prep: 1
  - attention: 1
  - residual: 3
  - feed_forward: 1

### Transformer Layer 1
- Nodes: 10
- Edges: 7
- Edge Types:
  - attention_prep: 1
  - attention: 1
  - data_flow: 1
  - residual: 3
  - feed_forward: 1

### Transformer Layer 2
- Nodes: 10
- Edges: 7
- Edge Types:
  - attention_prep: 1
  - attention: 1
  - data_flow: 1
  - residual: 3
  - feed_forward: 1

### Transformer Layer 3
- Nodes: 10
- Edges: 7
- Edge Types:
  - attention_prep: 1
  - attention: 1
  - data_flow: 1
  - residual: 3
  - feed_forward: 1

### Transformer Layer 4
- Nodes: 10
- Edges: 7
- Edge Types:
  - attention_prep: 1
  - attention: 1
  - data_flow: 1
  - residual: 3
  - feed_forward: 1

### Transformer Layer 5
- Nodes: 10
- Edges: 7
- Edge Types:
  - attention_prep: 1
  - attention: 1
  - data_flow: 2
  - residual: 2
  - feed_forward: 1

### Output Layer
- Nodes: 2
- Edges: 3
- Edge Types:
  - residual: 1
  - data_flow: 2

## Critical Paths
### Path 1
layer_5_post_ffn_norm -> final_layer_norm -> output_head

## Graph Visualization
A DOT file has been generated for graph visualization:
```bash
dot -Tpng esm2_hypergraph.dot -o esm2_hypergraph.png
```
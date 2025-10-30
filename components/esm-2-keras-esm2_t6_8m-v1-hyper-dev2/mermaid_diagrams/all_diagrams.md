# ESM-2 Hypergraph Mermaid Diagrams

This document contains all mermaid diagrams for the ESM-2 hypergraph system.

## ESM-2 Model Architecture Flow

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

## Hypergraph Node and Edge Distribution

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

## Component Class Architecture

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


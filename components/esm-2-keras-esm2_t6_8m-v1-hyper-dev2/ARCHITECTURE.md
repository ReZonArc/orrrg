# ESM-2 Hypergraph Technical Architecture

This document provides comprehensive technical architecture documentation for the ESM-2 hypergraph mapping system, including detailed mermaid diagrams illustrating the system structure, data flow, and component interactions.

## Table of Contents

1. [System Overview](#system-overview)
2. [ESM-2 Model Architecture](#esm-2-model-architecture)
3. [Hypergraph Structure](#hypergraph-structure)
4. [Component Architecture](#component-architecture)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Query Engine Architecture](#query-engine-architecture)
7. [Visualization Pipeline](#visualization-pipeline)
8. [API Architecture](#api-architecture)

## System Overview

The ESM-2 Hypergraph system represents the complete computational structure of the ESM-2 transformer model as a hypergraph, enabling detailed analysis and understanding of the model architecture.

```mermaid
graph TB
    subgraph "ESM-2 Hypergraph System"
        A[ESM-2 Model Config] --> B[Hypergraph Builder]
        B --> C[ESM2Hypergraph]
        C --> D[Query Engine]
        C --> E[Visualizer]
        C --> F[JSON Export]
        
        D --> G[Analysis Results]
        E --> H[DOT Graph]
        E --> I[Reports]
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
        end
    end
```

## ESM-2 Model Architecture

The ESM-2 model follows a transformer architecture with specific components mapped to hypergraph nodes and edges.

```mermaid
graph TD
    subgraph "ESM-2 Model Architecture"
        A[Input Tokens] --> B[Token Embedding]
        A --> C[Positional Encoding]
        
        B --> D[Layer 0]
        C --> D
        
        D --> E[Layer 1]
        E --> F[Layer 2]
        F --> G[Layer 3]
        G --> H[Layer 4]
        H --> I[Layer 5]
        
        I --> J[Final LayerNorm]
        J --> K[Output Head]
        K --> L[Output Logits]
        
        subgraph "Transformer Layer Structure"
            subgraph "Self-Attention Block"
                M[Query Projection] 
                N[Key Projection]
                O[Value Projection]
                P[Multi-Head Attention]
                Q[Attention Output Projection]
                R[Residual + LayerNorm]
            end
            
            subgraph "Feed-Forward Block"
                S[FFN Intermediate]
                T[GELU Activation]
                U[FFN Output]
                V[Residual + LayerNorm]
            end
        end
    end
```

### Model Configuration

- **Model**: ESM-2 (esm2_t6_8m-v1)
- **Layers**: 6 transformer layers
- **Attention Heads**: 20 per layer
- **Hidden Dimension**: 320
- **Intermediate Dimension**: 1280
- **Vocabulary Size**: 33
- **Max Sequence Length**: 1026

## Hypergraph Structure

The hypergraph representation maps each model component to nodes and their relationships to hyperedges.

```mermaid
graph LR
    subgraph "Hypergraph Structure"
        subgraph "Node Types"
            A[Embedding: 1]
            B[Positional: 1]
            C[Linear: 36]
            D[Attention: 6]
            E[LayerNorm: 13]
            F[Activation: 6]
            G[Output: 1]
        end
        
        subgraph "Edge Types"
            H[Data Flow: 11]
            I[Attention Prep: 6]
            J[Attention: 6]
            K[Residual: 12]
            L[Feed Forward: 6]
        end
        
        subgraph "Statistics"
            M[Total Nodes: 64]
            N[Total Edges: 41]
            O[Max Edge Size: 4]
        end
    end
```

### Node Distribution by Layer

```mermaid
graph TD
    subgraph "Layer Distribution"
        A[Input Layer] --> A1[Token Embedding]
        A --> A2[Positional Encoding]
        
        B[Layer 0] --> B1[Q/K/V Projections: 3]
        B --> B2[Multi-Head Attention: 1]
        B --> B3[Attention Output: 1]  
        B --> B4[LayerNorm: 2]
        B --> B5[FFN Layers: 2]
        B --> B6[Activation: 1]
        
        C[Layer 1-5] --> C1[Same Structure × 5]
        
        D[Output Layer] --> D1[Final LayerNorm: 1]
        D --> D2[Output Head: 1]
    end
```

## Component Architecture

### Core Classes and Their Relationships

```mermaid
classDiagram
    class ESM2Hypergraph {
        +Dict config
        +Dict~str,HyperNode~ nodes
        +Dict~str,HyperEdge~ edges
        +Dict~str,Set~ adjacency
        +build_hypergraph()
        +get_statistics()
        +to_dict()
        +save_to_json()
        +visualize_summary()
    }
    
    class HyperNode {
        +str id
        +str name
        +str type
        +int layer_idx
        +int param_count
        +Dict metadata
    }
    
    class HyperEdge {
        +str id
        +str name
        +List~str~ source_nodes
        +List~str~ target_nodes
        +str edge_type
        +Dict metadata
    }
    
    class HypergraphQueryEngine {
        +ESM2Hypergraph hypergraph
        +find_nodes_by_type()
        +find_nodes_by_layer()
        +get_computational_path()
        +analyze_parameter_flow()
        +find_bottlenecks()
    }
    
    class HypergraphVisualizer {
        +ESM2Hypergraph hypergraph
        +create_layer_diagram()
        +generate_dot_graph()
        +create_connectivity_matrix()
        +find_critical_paths()
    }
    
    ESM2Hypergraph *-- HyperNode
    ESM2Hypergraph *-- HyperEdge
    HypergraphQueryEngine --> ESM2Hypergraph
    HypergraphVisualizer --> ESM2Hypergraph
```

## Data Flow Diagrams

### Hypergraph Construction Flow

```mermaid
flowchart TD
    A[Model Configuration] --> B[ESM2Hypergraph Constructor]
    B --> C[Build Input Nodes]
    C --> D[Build Transformer Layers]
    D --> E[Build Output Nodes]
    E --> F[Add Data Flow Edges]
    F --> G[Add Layer-specific Edges]
    G --> H[Validate Structure]
    H --> I[Complete Hypergraph]
    
    subgraph "For Each Layer (0-5)"
        D --> D1[Add Q/K/V Projections]
        D1 --> D2[Add Attention Node]
        D2 --> D3[Add Attention Output]
        D3 --> D4[Add LayerNorms]
        D4 --> D5[Add FFN Components]
        D5 --> D6[Add Activations]
    end
```

### Query Processing Flow

```mermaid
flowchart LR
    A[Query Request] --> B{Query Type}
    
    B -->|stats| C[Get Statistics]
    B -->|attention| D[Analyze Attention Structure]
    B -->|params| E[Parameter Flow Analysis]
    B -->|bottlenecks| F[Find Bottlenecks]
    B -->|path| G[Find Computational Path]
    B -->|subgraph| H[Extract Subgraph]
    
    C --> I[JSON Output]
    D --> I
    E --> I
    F --> I
    G --> J[Path Visualization]
    H --> K[Subgraph Export]
```

## Query Engine Architecture

The query engine provides powerful analysis capabilities over the hypergraph structure.

```mermaid
graph TB
    subgraph "Query Engine Components"
        A[HypergraphQueryEngine] --> B[Node Queries]
        A --> C[Path Analysis]
        A --> D[Parameter Analysis]
        A --> E[Bottleneck Detection]
        A --> F[Subgraph Extraction]
        
        B --> B1[find_nodes_by_type]
        B --> B2[find_nodes_by_layer]
        B --> B3[get_node_dependencies]
        
        C --> C1[get_computational_path]
        C --> C2[BFS Path Finding]
        
        D --> D1[analyze_parameter_flow]
        D --> D2[Parameter Distribution]
        
        E --> E1[find_bottlenecks]
        E --> E2[Fan-in/Fan-out Analysis]
        
        F --> F1[export_subgraph]
        F --> F2[Layer Range Export]
    end
```

### Query Types and Operations

```mermaid
mindmap
  root((Query Engine))
    Stats
      Total Nodes
      Total Edges
      Node Type Distribution
      Edge Type Distribution
    Structure Analysis
      Attention Patterns
      Parameter Distribution
      Layer Connectivity
    Path Finding
      Computational Paths
      Dependency Chains
      Critical Paths
    Bottleneck Detection
      High Fan-in Nodes
      High Fan-out Nodes
      Performance Bottlenecks
    Subgraph Operations
      Layer Extraction
      Component Isolation
      Partial Graph Export
```

## Visualization Pipeline

The visualization system generates multiple output formats for hypergraph analysis.

```mermaid
graph TD
    A[ESM2Hypergraph] --> B[HypergraphVisualizer]
    
    B --> C[Text Diagrams]
    B --> D[DOT Graph Generation]
    B --> E[Connectivity Analysis]
    B --> F[Critical Path Analysis]
    
    C --> G[Layer Diagrams]
    C --> H[Architecture Overview]
    
    D --> I[Simplified DOT]
    D --> J[Detailed DOT]
    
    E --> K[Connectivity Matrix]
    E --> L[Degree Analysis]
    
    F --> M[Critical Paths]
    F --> N[Path Visualization]
    
    subgraph "Output Formats"
        G
        H
        I
        J
        K
        L
        M
        N
    end
```

### Visualization Types

```mermaid
graph LR
    subgraph "Visualization Outputs"
        A[Text-based Diagrams] --> A1[Layer Structure]
        A --> A2[Architecture Summary]
        
        B[DOT Graphs] --> B1[Node-Edge Graphs]
        B --> B2[Simplified Views]
        
        C[Analysis Reports] --> C1[Connectivity Matrix]
        C --> C2[Statistics Tables]
        C --> C3[Critical Paths]
        
        D[Interactive Queries] --> D1[Parameter Flow]
        D --> D2[Bottleneck Analysis]
        D --> D3[Path Finding]
    end
```

## API Architecture

### Main API Entry Points

```mermaid
graph TB
    subgraph "API Architecture"
        A[main.py] --> B[create_esm2_hypergraph]
        B --> C[ESM2Hypergraph]
        
        D[hypergraph_query.py] --> E[HypergraphQueryEngine]
        E --> C
        
        F[hypergraph_visualizer.py] --> G[HypergraphVisualizer]
        G --> C
        
        subgraph "Core API Methods"
            C --> H[get_statistics]
            C --> I[to_dict]
            C --> J[save_to_json]
            C --> K[visualize_summary]
        end
        
        subgraph "Query API Methods"
            E --> L[find_nodes_by_type]
            E --> M[get_computational_path]
            E --> N[analyze_parameter_flow]
            E --> O[find_bottlenecks]
        end
        
        subgraph "Visualization API Methods"
            G --> P[create_layer_diagram]
            G --> Q[generate_dot_graph]
            G --> R[create_connectivity_matrix]
            G --> S[find_critical_paths]
        end
    end
```

### Data Models

```mermaid
erDiagram
    ESM2Hypergraph ||--o{ HyperNode : contains
    ESM2Hypergraph ||--o{ HyperEdge : contains
    ESM2Hypergraph ||--|| Config : uses
    
    HyperNode {
        string id PK
        string name
        string type
        int layer_idx
        int param_count
        dict metadata
    }
    
    HyperEdge {
        string id PK
        string name
        list source_nodes FK
        list target_nodes FK
        string edge_type
        dict metadata
    }
    
    Config {
        string name
        int vocabulary_size
        int num_layers
        int num_heads
        int hidden_dim
        int intermediate_dim
        boolean use_bias
        string activation
    }
```

## Implementation Details

### Hypergraph Construction Algorithm

```mermaid
sequenceDiagram
    participant C as Config
    participant H as ESM2Hypergraph
    participant N as Nodes
    participant E as Edges
    
    C->>H: Initialize with config
    H->>N: Add input nodes (embedding, positional)
    
    loop For each layer (0-5)
        H->>N: Add Q/K/V projection nodes
        H->>N: Add attention node
        H->>N: Add attention output node
        H->>N: Add LayerNorm nodes
        H->>N: Add FFN nodes
        H->>N: Add activation node
    end
    
    H->>N: Add output nodes
    H->>E: Add data flow edges
    H->>E: Add attention edges
    H->>E: Add residual edges
    H->>E: Add feed-forward edges
    H->>H: Validate structure
```

### Query Processing Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant Q as QueryEngine
    participant H as Hypergraph
    participant A as Analysis
    
    U->>Q: Submit query
    Q->>H: Access nodes/edges
    H->>Q: Return data
    Q->>A: Process analysis
    A->>Q: Return results
    Q->>U: Format output
```

This architecture documentation provides a comprehensive view of the ESM-2 hypergraph system, enabling developers and researchers to understand the system structure, data flow, and component interactions through detailed mermaid diagrams.
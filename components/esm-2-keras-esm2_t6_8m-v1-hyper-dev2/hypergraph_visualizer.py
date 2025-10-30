"""
Hypergraph Visualization Utilities
Provides tools to visualize and analyze the ESM-2 hypergraph structure.
"""

import json
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict, deque
from esm2_hypergraph import ESM2Hypergraph

# Optional graphviz import
try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False


class HypergraphVisualizer:
    """Visualization utilities for hypergraphs"""
    
    def __init__(self, hypergraph: ESM2Hypergraph):
        self.hypergraph = hypergraph
    
    def create_layer_diagram(self) -> str:
        """Create a text-based layer diagram"""
        diagram = []
        diagram.append("ESM-2 Architecture Hypergraph")
        diagram.append("=" * 50)
        diagram.append("")
        
        # Input layer
        diagram.append("INPUT LAYER:")
        diagram.append("  Token Embedding (vocab=33 -> hidden=320)")
        diagram.append("  Rotary Positional Encoding")
        if self.hypergraph.use_pre_layer_norm:
            diagram.append("  Input Layer Norm")
        diagram.append("")
        
        # Transformer layers
        for i in range(self.hypergraph.num_layers):
            diagram.append(f"TRANSFORMER LAYER {i}:")
            if self.hypergraph.use_pre_layer_norm:
                diagram.append("  Pre-Attention Layer Norm")
            
            diagram.append("  Multi-Head Self-Attention:")
            diagram.append(f"    - Query/Key/Value Projections (320 -> 320)")
            diagram.append(f"    - {self.hypergraph.num_heads} attention heads")
            diagram.append(f"    - Head dimension: {self.hypergraph.hidden_dim // self.hypergraph.num_heads}")
            diagram.append("    - Output Projection")
            
            if self.hypergraph.dropout > 0:
                diagram.append(f"    - Dropout ({self.hypergraph.dropout})")
            diagram.append("  + Residual Connection")
            
            if not self.hypergraph.use_pre_layer_norm:
                diagram.append("  Post-Attention Layer Norm")
            
            if self.hypergraph.use_pre_layer_norm:
                diagram.append("  Pre-FFN Layer Norm")
            
            diagram.append("  Feed-Forward Network:")
            diagram.append(f"    - Linear (320 -> {self.hypergraph.intermediate_dim})")
            diagram.append(f"    - {self.hypergraph.config['activation'].upper()} Activation")
            diagram.append(f"    - Linear ({self.hypergraph.intermediate_dim} -> 320)")
            
            if self.hypergraph.dropout > 0:
                diagram.append(f"    - Dropout ({self.hypergraph.dropout})")
            diagram.append("  + Residual Connection")
            
            if not self.hypergraph.use_pre_layer_norm:
                diagram.append("  Post-FFN Layer Norm")
            diagram.append("")
        
        # Output layer
        diagram.append("OUTPUT LAYER:")
        diagram.append("  Final Layer Norm")
        diagram.append("  Output Head")
        diagram.append("")
        
        return "\n".join(diagram)
    
    def create_connectivity_matrix(self) -> Dict[str, Any]:
        """Create a connectivity analysis of the hypergraph"""
        nodes = list(self.hypergraph.nodes.keys())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Create adjacency matrix
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        for edge in self.hypergraph.edges.values():
            for source in edge.source_nodes:
                for target in edge.target_nodes:
                    if source in node_to_idx and target in node_to_idx:
                        matrix[node_to_idx[source]][node_to_idx[target]] = 1
        
        # Calculate connectivity metrics
        in_degree = [sum(matrix[j][i] for j in range(n)) for i in range(n)]
        out_degree = [sum(matrix[i][j] for j in range(n)) for i in range(n)]
        
        return {
            "nodes": nodes,
            "adjacency_matrix": matrix,
            "in_degree": dict(zip(nodes, in_degree)),
            "out_degree": dict(zip(nodes, out_degree)),
            "total_connections": sum(sum(row) for row in matrix)
        }
    
    def find_critical_paths(self) -> List[List[str]]:
        """Find critical paths through the hypergraph"""
        paths = []
        
        # Find input nodes (no incoming edges)
        input_nodes = []
        for node_id in self.hypergraph.nodes:
            has_input = False
            for edge in self.hypergraph.edges.values():
                if node_id in edge.target_nodes:
                    has_input = True
                    break
            if not has_input:
                input_nodes.append(node_id)
        
        # Find output nodes (no outgoing edges)
        output_nodes = []
        for node_id in self.hypergraph.nodes:
            has_output = False
            for edge in self.hypergraph.edges.values():
                if node_id in edge.source_nodes:
                    has_output = True
                    break
            if not has_output:
                output_nodes.append(node_id)
        
        # BFS to find paths from input to output
        for input_node in input_nodes:
            for output_node in output_nodes:
                path = self._find_path(input_node, output_node)
                if path:
                    paths.append(path)
        
        return paths
    
    def _find_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find path between two nodes using BFS"""
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            # Find neighbors
            neighbors = set()
            for edge in self.hypergraph.edges.values():
                if current in edge.source_nodes:
                    neighbors.update(edge.target_nodes)
            
            for neighbor in neighbors:
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def analyze_layer_connectivity(self) -> Dict[int, Dict[str, Any]]:
        """Analyze connectivity patterns within each layer"""
        layer_analysis = {}
        
        for layer_idx in range(-1, self.hypergraph.num_layers + 1):
            layer_nodes = [
                node_id for node_id, node in self.hypergraph.nodes.items()
                if node.layer_idx == layer_idx
            ]
            
            layer_edges = [
                edge for edge in self.hypergraph.edges.values()
                if any(node in layer_nodes for node in edge.source_nodes + edge.target_nodes)
            ]
            
            # Count edge types in this layer
            edge_type_counts = defaultdict(int)
            for edge in layer_edges:
                edge_type_counts[edge.edge_type] += 1
            
            layer_analysis[layer_idx] = {
                "nodes": layer_nodes,
                "node_count": len(layer_nodes),
                "edge_count": len(layer_edges),
                "edge_types": dict(edge_type_counts)
            }
        
        return layer_analysis
    
    def generate_dot_graph(self, max_nodes: int = 50) -> str:
        """Generate DOT format for graph visualization"""
        if len(self.hypergraph.nodes) > max_nodes:
            # Create simplified version for large graphs
            return self._generate_simplified_dot()
        
        dot = ["digraph ESM2_Hypergraph {"]
        dot.append("  rankdir=LR;")
        dot.append("  node [shape=box, style=rounded];")
        dot.append("")
        
        # Add nodes with colors based on type
        colors = {
            "embedding": "lightblue",
            "positional": "lightgreen", 
            "attention": "orange",
            "linear": "yellow",
            "layernorm": "pink",
            "activation": "red",
            "dropout": "gray",
            "output": "purple"
        }
        
        for node_id, node in self.hypergraph.nodes.items():
            color = colors.get(node.type, "white")
            label = f"{node.name}\\n{node.type}"
            dot.append(f'  "{node_id}" [label="{label}", fillcolor={color}, style=filled];')
        
        dot.append("")
        
        # Add edges
        for edge in self.hypergraph.edges.values():
            for source in edge.source_nodes:
                for target in edge.target_nodes:
                    style = "solid"
                    if edge.edge_type == "residual":
                        style = "dashed"
                    elif edge.edge_type == "attention":
                        style = "bold"
                    
                    dot.append(f'  "{source}" -> "{target}" [style={style}];')
        
        dot.append("}")
        return "\n".join(dot)
    
    def _generate_simplified_dot(self) -> str:
        """Generate simplified DOT for large graphs"""
        dot = ["digraph ESM2_Hypergraph_Simplified {"]
        dot.append("  rankdir=LR;")
        dot.append("  node [shape=box, style=rounded];")
        dot.append("")
        
        # Group nodes by layer
        layers = defaultdict(list)
        for node_id, node in self.hypergraph.nodes.items():
            layers[node.layer_idx].append(node_id)
        
        # Create layer nodes
        for layer_idx in sorted(layers.keys()):
            if layer_idx == -1:
                label = "Input\\nProcessing"
            elif layer_idx == self.hypergraph.num_layers:
                label = "Output\\nProcessing"
            else:
                label = f"Transformer\\nLayer {layer_idx}"
            
            dot.append(f'  "layer_{layer_idx}" [label="{label}"];')
        
        # Connect layers
        sorted_layers = sorted(layers.keys())
        for i in range(len(sorted_layers) - 1):
            current = sorted_layers[i]
            next_layer = sorted_layers[i + 1]
            dot.append(f'  "layer_{current}" -> "layer_{next_layer}";')
        
        dot.append("}")
        return "\n".join(dot)
    
    def generate_mermaid_architecture(self) -> str:
        """Generate mermaid diagram showing model architecture flow"""
        mermaid = ["graph TD"]
        mermaid.append("    A[Input Tokens] --> B[Token Embedding<br/>vocab=33→hidden=320]")
        mermaid.append("    A --> C[Rotary Positional<br/>Encoding]")
        mermaid.append("")
        mermaid.append("    B --> D0[Transformer Layer 0]")
        mermaid.append("    C --> D0")
        mermaid.append("")
        
        # Add transformer layers
        for i in range(self.hypergraph.num_layers - 1):
            mermaid.append(f"    D{i} --> D{i+1}[Transformer Layer {i+1}]")
        
        last_layer = self.hypergraph.num_layers - 1
        mermaid.append("")
        mermaid.append(f"    D{last_layer} --> E[Final LayerNorm]")
        mermaid.append("    E --> F[Output Head]")
        mermaid.append("    F --> G[Output Logits<br/>hidden=320→vocab=33]")
        mermaid.append("")
        
        # Add transformer layer detail
        mermaid.append("    subgraph \"Each Transformer Layer\"")
        mermaid.append(f"        H[Multi-Head Attention<br/>{self.hypergraph.num_heads} heads, dim={self.hypergraph.hidden_dim//self.hypergraph.num_heads}] --> I[Residual + LayerNorm]")
        mermaid.append(f"        I --> J[Feed-Forward Network<br/>{self.hypergraph.hidden_dim}→{self.hypergraph.intermediate_dim}→{self.hypergraph.hidden_dim}]")
        mermaid.append("        J --> K[Residual + LayerNorm]")
        mermaid.append("    end")
        
        return "\n".join(mermaid)
    
    def generate_mermaid_hypergraph_structure(self) -> str:
        """Generate mermaid diagram showing hypergraph node and edge distribution"""
        stats = self.hypergraph.get_statistics()
        
        mermaid = ["graph LR"]
        mermaid.append("    subgraph \"Hypergraph Components\"")
        mermaid.append(f"        subgraph \"Node Types ({stats['total_nodes']} total)\"")
        
        for node_type, count in stats['node_types'].items():
            clean_type = node_type.title()
            mermaid.append(f"            {node_type.upper()}[{clean_type}: {count}]")
        
        mermaid.append("        end")
        mermaid.append("")
        mermaid.append(f"        subgraph \"Edge Types ({stats['total_edges']} total)\"")
        
        for edge_type, count in stats['edge_types'].items():
            clean_type = edge_type.replace('_', ' ').title()
            var_name = edge_type.replace('_', '').upper()
            mermaid.append(f"            {var_name}[{clean_type}: {count}]")
        
        mermaid.append("        end")
        mermaid.append("    end")
        
        return "\n".join(mermaid)
    
    def generate_mermaid_query_flow(self) -> str:
        """Generate mermaid diagram showing query processing flow"""
        mermaid = ["flowchart LR"]
        mermaid.append("    A[Query Request] --> B{Query Type}")
        mermaid.append("")
        mermaid.append("    B -->|stats| C[Get Statistics<br/>Nodes, Edges, Types]")
        mermaid.append("    B -->|attention| D[Analyze Attention<br/>Structure & Patterns]")
        mermaid.append("    B -->|params| E[Parameter Flow<br/>Analysis]")
        mermaid.append("    B -->|bottlenecks| F[Find Bottlenecks<br/>High Fan-in/out]")
        mermaid.append("    B -->|path| G[Find Computational<br/>Path A→B]")
        mermaid.append("    B -->|subgraph| H[Extract Layer<br/>Subgraph]")
        mermaid.append("")
        mermaid.append("    C --> I[JSON Output]")
        mermaid.append("    D --> I")
        mermaid.append("    E --> I")
        mermaid.append("    F --> I")
        mermaid.append("    G --> J[Path Visualization]")
        mermaid.append("    H --> K[Subgraph Export]")
        
        return "\n".join(mermaid)
    
    def generate_mermaid_component_architecture(self) -> str:
        """Generate mermaid class diagram showing component relationships"""
        mermaid = ["classDiagram"]
        mermaid.append("    class ESM2Hypergraph {")
        mermaid.append("        +get_statistics()")
        mermaid.append("        +to_dict()")
        mermaid.append("        +save_to_json()")
        mermaid.append("        +visualize_summary()")
        mermaid.append("    }")
        mermaid.append("")
        mermaid.append("    class HypergraphQueryEngine {")
        mermaid.append("        +find_nodes_by_type()")
        mermaid.append("        +find_nodes_by_layer()")
        mermaid.append("        +get_computational_path()")
        mermaid.append("        +analyze_parameter_flow()")
        mermaid.append("        +find_bottlenecks()")
        mermaid.append("    }")
        mermaid.append("")
        mermaid.append("    class HypergraphVisualizer {")
        mermaid.append("        +create_layer_diagram()")
        mermaid.append("        +generate_dot_graph()")
        mermaid.append("        +generate_mermaid_diagrams()")
        mermaid.append("        +create_connectivity_matrix()")
        mermaid.append("        +find_critical_paths()")
        mermaid.append("    }")
        mermaid.append("")
        mermaid.append("    HypergraphQueryEngine --> ESM2Hypergraph")
        mermaid.append("    HypergraphVisualizer --> ESM2Hypergraph")
        
        return "\n".join(mermaid)


def create_visualization_report(hypergraph: ESM2Hypergraph, output_dir: str = ".") -> str:
    """Create a comprehensive visualization report"""
    visualizer = HypergraphVisualizer(hypergraph)
    
    report = []
    report.append("# ESM-2 Hypergraph Analysis Report")
    report.append("")
    report.append("## Model Configuration")
    report.append("```json")
    report.append(json.dumps(hypergraph.config, indent=2))
    report.append("```")
    report.append("")
    
    # Architecture diagram
    report.append("## Architecture Overview")
    report.append("```")
    report.append(visualizer.create_layer_diagram())
    report.append("```")
    report.append("")
    
    # Mermaid diagrams
    report.append("## Model Architecture Flow")
    report.append("```mermaid")
    report.append(visualizer.generate_mermaid_architecture())
    report.append("```")
    report.append("")
    
    report.append("## Hypergraph Structure")
    report.append("```mermaid")
    report.append(visualizer.generate_mermaid_hypergraph_structure())
    report.append("```")
    report.append("")
    
    report.append("## Component Architecture")
    report.append("```mermaid")
    report.append(visualizer.generate_mermaid_component_architecture())
    report.append("```")
    report.append("")
    
    report.append("## Query Processing Flow")
    report.append("```mermaid")
    report.append(visualizer.generate_mermaid_query_flow())
    report.append("```")
    report.append("")
    
    # Statistics
    stats = hypergraph.get_statistics()
    report.append("## Hypergraph Statistics")
    report.append(f"- **Total Nodes**: {stats['total_nodes']}")
    report.append(f"- **Total Hyperedges**: {stats['total_edges']}")
    report.append(f"- **Maximum Hyperedge Size**: {stats['max_hyperedge_size']}")
    report.append("")
    
    report.append("### Node Types Distribution")
    for node_type, count in stats['node_types'].items():
        report.append(f"- {node_type}: {count}")
    report.append("")
    
    report.append("### Edge Types Distribution")
    for edge_type, count in stats['edge_types'].items():
        report.append(f"- {edge_type}: {count}")
    report.append("")
    
    # Connectivity analysis
    connectivity = visualizer.create_connectivity_matrix()
    report.append("## Connectivity Analysis")
    report.append(f"- **Total Connections**: {connectivity['total_connections']}")
    report.append("")
    
    # Layer analysis
    layer_analysis = visualizer.analyze_layer_connectivity()
    report.append("## Layer-wise Analysis")
    for layer_idx in sorted(layer_analysis.keys()):
        analysis = layer_analysis[layer_idx]
        if layer_idx == -1:
            layer_name = "Input Layer"
        elif layer_idx == hypergraph.num_layers:
            layer_name = "Output Layer"
        else:
            layer_name = f"Transformer Layer {layer_idx}"
        
        report.append(f"### {layer_name}")
        report.append(f"- Nodes: {analysis['node_count']}")
        report.append(f"- Edges: {analysis['edge_count']}")
        if analysis['edge_types']:
            report.append("- Edge Types:")
            for edge_type, count in analysis['edge_types'].items():
                report.append(f"  - {edge_type}: {count}")
        report.append("")
    
    # Critical paths
    paths = visualizer.find_critical_paths()
    if paths:
        report.append("## Critical Paths")
        for i, path in enumerate(paths[:3]):  # Show first 3 paths
            report.append(f"### Path {i+1}")
            report.append(" -> ".join(path))
            report.append("")
    
    # Save DOT file for graph visualization
    dot_content = visualizer.generate_dot_graph()
    with open(f"{output_dir}/esm2_hypergraph.dot", "w") as f:
        f.write(dot_content)
    
    report.append("## Graph Visualization")
    report.append("A DOT file has been generated for graph visualization:")
    report.append("```bash")
    report.append("dot -Tpng esm2_hypergraph.dot -o esm2_hypergraph.png")
    report.append("```")
    
    return "\n".join(report)


if __name__ == "__main__":
    from esm2_hypergraph import create_esm2_hypergraph
    
    # Example configuration
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
    
    # Create hypergraph and visualization
    hypergraph = create_esm2_hypergraph(config)
    report = create_visualization_report(hypergraph)
    
    # Save report
    with open("hypergraph_analysis_report.md", "w") as f:
        f.write(report)
    
    print("Visualization report generated: hypergraph_analysis_report.md")
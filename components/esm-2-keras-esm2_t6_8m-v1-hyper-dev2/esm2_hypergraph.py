"""
ESM-2 Hypergraph Mapping
Maps the full computational hypergraph of the ESM-2 transformer model.
A hypergraph represents the model architecture where hyperedges connect multiple nodes.
"""

import json
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class HyperNode:
    """Represents a node in the hypergraph (a computational unit)"""
    id: str
    name: str
    type: str  # e.g., 'embedding', 'attention', 'ffn', 'layernorm'
    layer_idx: int
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: Dict[str, Any]


@dataclass
class HyperEdge:
    """Represents a hyperedge connecting multiple nodes"""
    id: str
    name: str
    source_nodes: List[str]  # Input node IDs
    target_nodes: List[str]  # Output node IDs
    edge_type: str  # e.g., 'data_flow', 'residual', 'attention'


class ESM2Hypergraph:
    """
    Hypergraph representation of the ESM-2 model architecture
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, HyperNode] = {}
        self.edges: Dict[str, HyperEdge] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        # Extract key configuration parameters
        self.vocab_size = config["vocabulary_size"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.hidden_dim = config["hidden_dim"]
        self.intermediate_dim = config["intermediate_dim"]
        self.max_seq_length = config["max_sequence_length"]
        self.dropout = config["dropout"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.use_pre_layer_norm = config["use_pre_layer_norm"]
        
        self._build_hypergraph()
    
    def _build_hypergraph(self):
        """Construct the complete hypergraph representation"""
        # Build input processing nodes
        self._add_input_nodes()
        
        # Build transformer layers
        for layer_idx in range(self.num_layers):
            self._add_transformer_layer(layer_idx)
        
        # Build output nodes
        self._add_output_nodes()
        
        # Build hyperedges representing data flow
        self._add_data_flow_edges()
    
    def _add_input_nodes(self):
        """Add input processing nodes: embedding, positional encoding"""
        # Token embedding
        self._add_node(
            "token_embedding",
            "Token Embedding",
            "embedding",
            -1,
            (None, self.max_seq_length),
            (None, self.max_seq_length, self.hidden_dim),
            {"vocab_size": self.vocab_size, "embed_dim": self.hidden_dim}
        )
        
        # Positional encoding (rotary)
        self._add_node(
            "positional_encoding",
            "Rotary Positional Encoding",
            "positional",
            -1,
            (None, self.max_seq_length, self.hidden_dim),
            (None, self.max_seq_length, self.hidden_dim),
            {"max_wavelength": self.config["max_wavelength"]}
        )
        
        # Input layer norm (if pre-layer-norm)
        if self.use_pre_layer_norm:
            self._add_node(
                "input_layer_norm",
                "Input Layer Normalization",
                "layernorm",
                -1,
                (None, self.max_seq_length, self.hidden_dim),
                (None, self.max_seq_length, self.hidden_dim),
                {"eps": self.layer_norm_eps}
            )
    
    def _add_transformer_layer(self, layer_idx: int):
        """Add all components of a single transformer layer"""
        layer_prefix = f"layer_{layer_idx}"
        
        # Pre-layer normalization for attention
        if self.use_pre_layer_norm:
            self._add_node(
                f"{layer_prefix}_pre_attn_norm",
                f"Layer {layer_idx} Pre-Attention Norm",
                "layernorm",
                layer_idx,
                (None, self.max_seq_length, self.hidden_dim),
                (None, self.max_seq_length, self.hidden_dim),
                {"eps": self.layer_norm_eps}
            )
        
        # Multi-head self-attention components
        self._add_attention_nodes(layer_idx)
        
        # Attention dropout
        if self.dropout > 0:
            self._add_node(
                f"{layer_prefix}_attn_dropout",
                f"Layer {layer_idx} Attention Dropout",
                "dropout",
                layer_idx,
                (None, self.max_seq_length, self.hidden_dim),
                (None, self.max_seq_length, self.hidden_dim),
                {"rate": self.dropout}
            )
        
        # Post-attention layer norm
        if not self.use_pre_layer_norm:
            self._add_node(
                f"{layer_prefix}_post_attn_norm",
                f"Layer {layer_idx} Post-Attention Norm",
                "layernorm",
                layer_idx,
                (None, self.max_seq_length, self.hidden_dim),
                (None, self.max_seq_length, self.hidden_dim),
                {"eps": self.layer_norm_eps}
            )
        
        # Pre-FFN layer norm
        if self.use_pre_layer_norm:
            self._add_node(
                f"{layer_prefix}_pre_ffn_norm",
                f"Layer {layer_idx} Pre-FFN Norm",
                "layernorm",
                layer_idx,
                (None, self.max_seq_length, self.hidden_dim),
                (None, self.max_seq_length, self.hidden_dim),
                {"eps": self.layer_norm_eps}
            )
        
        # Feed-forward network
        self._add_ffn_nodes(layer_idx)
        
        # FFN dropout
        if self.dropout > 0:
            self._add_node(
                f"{layer_prefix}_ffn_dropout",
                f"Layer {layer_idx} FFN Dropout",
                "dropout",
                layer_idx,
                (None, self.max_seq_length, self.hidden_dim),
                (None, self.max_seq_length, self.hidden_dim),
                {"rate": self.dropout}
            )
        
        # Post-FFN layer norm
        if not self.use_pre_layer_norm:
            self._add_node(
                f"{layer_prefix}_post_ffn_norm",
                f"Layer {layer_idx} Post-FFN Norm",
                "layernorm",
                layer_idx,
                (None, self.max_seq_length, self.hidden_dim),
                (None, self.max_seq_length, self.hidden_dim),
                {"eps": self.layer_norm_eps}
            )
    
    def _add_attention_nodes(self, layer_idx: int):
        """Add multi-head self-attention nodes"""
        layer_prefix = f"layer_{layer_idx}"
        head_dim = self.hidden_dim // self.num_heads
        
        # Query, Key, Value projections
        for proj in ['query', 'key', 'value']:
            self._add_node(
                f"{layer_prefix}_{proj}_proj",
                f"Layer {layer_idx} {proj.capitalize()} Projection",
                "linear",
                layer_idx,
                (None, self.max_seq_length, self.hidden_dim),
                (None, self.max_seq_length, self.hidden_dim),
                {"use_bias": self.config["use_bias"]}
            )
        
        # Multi-head attention computation
        self._add_node(
            f"{layer_prefix}_multihead_attn",
            f"Layer {layer_idx} Multi-Head Attention",
            "attention",
            layer_idx,
            (None, self.max_seq_length, self.hidden_dim),
            (None, self.max_seq_length, self.hidden_dim),
            {
                "num_heads": self.num_heads,
                "head_dim": head_dim,
                "scale": head_dim ** -0.5
            }
        )
        
        # Output projection
        self._add_node(
            f"{layer_prefix}_attn_output_proj",
            f"Layer {layer_idx} Attention Output Projection",
            "linear",
            layer_idx,
            (None, self.max_seq_length, self.hidden_dim),
            (None, self.max_seq_length, self.hidden_dim),
            {"use_bias": self.config["use_bias"]}
        )
    
    def _add_ffn_nodes(self, layer_idx: int):
        """Add feed-forward network nodes"""
        layer_prefix = f"layer_{layer_idx}"
        
        # First linear layer (expansion)
        self._add_node(
            f"{layer_prefix}_ffn_intermediate",
            f"Layer {layer_idx} FFN Intermediate",
            "linear",
            layer_idx,
            (None, self.max_seq_length, self.hidden_dim),
            (None, self.max_seq_length, self.intermediate_dim),
            {"use_bias": self.config["use_bias"]}
        )
        
        # Activation function
        self._add_node(
            f"{layer_prefix}_ffn_activation",
            f"Layer {layer_idx} FFN Activation",
            "activation",
            layer_idx,
            (None, self.max_seq_length, self.intermediate_dim),
            (None, self.max_seq_length, self.intermediate_dim),
            {"activation": self.config["activation"]}
        )
        
        # Second linear layer (projection back)
        self._add_node(
            f"{layer_prefix}_ffn_output",
            f"Layer {layer_idx} FFN Output",
            "linear",
            layer_idx,
            (None, self.max_seq_length, self.intermediate_dim),
            (None, self.max_seq_length, self.hidden_dim),
            {"use_bias": self.config["use_bias"]}
        )
    
    def _add_output_nodes(self):
        """Add final output processing nodes"""
        # Final layer normalization
        self._add_node(
            "final_layer_norm",
            "Final Layer Normalization",
            "layernorm",
            self.num_layers,
            (None, self.max_seq_length, self.hidden_dim),
            (None, self.max_seq_length, self.hidden_dim),
            {"eps": self.layer_norm_eps}
        )
        
        # Output head (for downstream tasks)
        self._add_node(
            "output_head",
            "Output Head",
            "output",
            self.num_layers,
            (None, self.max_seq_length, self.hidden_dim),
            (None, self.max_seq_length, self.hidden_dim),
            {}
        )
    
    def _add_node(self, node_id: str, name: str, node_type: str, layer_idx: int,
                  input_shape: Tuple, output_shape: Tuple, parameters: Dict):
        """Add a node to the hypergraph"""
        node = HyperNode(
            id=node_id,
            name=name,
            type=node_type,
            layer_idx=layer_idx,
            input_shape=input_shape,
            output_shape=output_shape,
            parameters=parameters
        )
        self.nodes[node_id] = node
    
    def _add_data_flow_edges(self):
        """Add hyperedges representing data flow between nodes"""
        # Input flow
        self._add_edge(
            "input_flow",
            "Input Processing Flow",
            [],
            ["token_embedding"],
            "data_flow"
        )
        
        self._add_edge(
            "embedding_to_pos",
            "Embedding to Positional",
            ["token_embedding"],
            ["positional_encoding"],
            "data_flow"
        )
        
        # Connect input to first layer
        first_layer_input = "input_layer_norm" if self.use_pre_layer_norm else "layer_0_pre_attn_norm"
        if not self.use_pre_layer_norm:
            first_layer_input = "layer_0_query_proj"
        
        self._add_edge(
            "pos_to_layers",
            "Positional to Layers",
            ["positional_encoding"],
            [first_layer_input],
            "data_flow"
        )
        
        # Connect transformer layers
        for layer_idx in range(self.num_layers):
            self._add_layer_edges(layer_idx)
        
        # Connect to output
        last_layer_output = f"layer_{self.num_layers-1}_post_ffn_norm" if not self.use_pre_layer_norm else f"layer_{self.num_layers-1}_ffn_output"
        self._add_edge(
            "layers_to_final",
            "Layers to Final",
            [last_layer_output],
            ["final_layer_norm"],
            "data_flow"
        )
        
        self._add_edge(
            "final_to_output",
            "Final to Output",
            ["final_layer_norm"],
            ["output_head"],
            "data_flow"
        )
    
    def _add_layer_edges(self, layer_idx: int):
        """Add edges for a single transformer layer"""
        layer_prefix = f"layer_{layer_idx}"
        
        # Attention flow
        attn_inputs = [f"{layer_prefix}_query_proj", f"{layer_prefix}_key_proj", f"{layer_prefix}_value_proj"]
        self._add_edge(
            f"{layer_prefix}_attn_projections",
            f"Layer {layer_idx} Attention Projections",
            [f"{layer_prefix}_pre_attn_norm"] if self.use_pre_layer_norm else [],
            attn_inputs,
            "attention_prep"
        )
        
        self._add_edge(
            f"{layer_prefix}_attn_compute",
            f"Layer {layer_idx} Attention Computation",
            attn_inputs,
            [f"{layer_prefix}_multihead_attn"],
            "attention"
        )
        
        self._add_edge(
            f"{layer_prefix}_attn_output",
            f"Layer {layer_idx} Attention Output",
            [f"{layer_prefix}_multihead_attn"],
            [f"{layer_prefix}_attn_output_proj"],
            "data_flow"
        )
        
        # Residual connection around attention
        attn_residual_input = f"{layer_prefix}_pre_attn_norm" if self.use_pre_layer_norm else "positional_encoding"
        if layer_idx > 0:
            prev_layer = f"layer_{layer_idx-1}"
            attn_residual_input = f"{prev_layer}_post_ffn_norm" if not self.use_pre_layer_norm else f"{prev_layer}_ffn_output"
        
        attn_output = f"{layer_prefix}_attn_dropout" if self.dropout > 0 else f"{layer_prefix}_attn_output_proj"
        self._add_edge(
            f"{layer_prefix}_attn_residual",
            f"Layer {layer_idx} Attention Residual",
            [attn_residual_input, attn_output],
            [f"{layer_prefix}_post_attn_norm"] if not self.use_pre_layer_norm else [f"{layer_prefix}_pre_ffn_norm"],
            "residual"
        )
        
        # FFN flow
        ffn_input = f"{layer_prefix}_pre_ffn_norm" if self.use_pre_layer_norm else f"{layer_prefix}_post_attn_norm"
        self._add_edge(
            f"{layer_prefix}_ffn_flow",
            f"Layer {layer_idx} FFN Flow",
            [ffn_input],
            [f"{layer_prefix}_ffn_intermediate", f"{layer_prefix}_ffn_activation", f"{layer_prefix}_ffn_output"],
            "feed_forward"
        )
        
        # Residual connection around FFN
        ffn_residual_input = f"{layer_prefix}_post_attn_norm" if not self.use_pre_layer_norm else f"{layer_prefix}_pre_ffn_norm"
        ffn_output = f"{layer_prefix}_ffn_dropout" if self.dropout > 0 else f"{layer_prefix}_ffn_output"
        
        next_layer_input = f"{layer_prefix}_post_ffn_norm" if not self.use_pre_layer_norm else f"layer_{layer_idx+1}_pre_attn_norm"
        if layer_idx == self.num_layers - 1:
            next_layer_input = "final_layer_norm"
        
        self._add_edge(
            f"{layer_prefix}_ffn_residual",
            f"Layer {layer_idx} FFN Residual",
            [ffn_residual_input, ffn_output],
            [next_layer_input],
            "residual"
        )
    
    def _add_edge(self, edge_id: str, name: str, source_nodes: List[str],
                  target_nodes: List[str], edge_type: str):
        """Add a hyperedge to the graph"""
        edge = HyperEdge(
            id=edge_id,
            name=name,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_type=edge_type
        )
        self.edges[edge_id] = edge
        
        # Update adjacency
        for source in source_nodes:
            for target in target_nodes:
                self.adjacency[source].add(target)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the hypergraph"""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        
        for node in self.nodes.values():
            node_types[node.type] += 1
        
        for edge in self.edges.values():
            edge_types[edge.edge_type] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "max_hyperedge_size": max(len(e.source_nodes) + len(e.target_nodes) for e in self.edges.values()) if self.edges else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hypergraph to dictionary representation"""
        return {
            "config": self.config,
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
            "edges": {eid: asdict(edge) for eid, edge in self.edges.items()},
            "adjacency": {k: list(v) for k, v in self.adjacency.items()},
            "statistics": self.get_statistics()
        }
    
    def save_to_json(self, filepath: str):
        """Save hypergraph to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def visualize_summary(self) -> str:
        """Generate a text summary of the hypergraph structure"""
        stats = self.get_statistics()
        summary = f"""
ESM-2 Hypergraph Summary
========================
Configuration: {self.config['name']}
- Vocabulary Size: {self.vocab_size}
- Number of Layers: {self.num_layers}
- Number of Heads: {self.num_heads}
- Hidden Dimension: {self.hidden_dim}
- Intermediate Dimension: {self.intermediate_dim}
- Max Sequence Length: {self.max_seq_length}

Hypergraph Statistics:
- Total Nodes: {stats['total_nodes']}
- Total Hyperedges: {stats['total_edges']}
- Maximum Hyperedge Size: {stats['max_hyperedge_size']}

Node Types:
"""
        for node_type, count in stats['node_types'].items():
            summary += f"- {node_type}: {count}\n"
        
        summary += "\nEdge Types:\n"
        for edge_type, count in stats['edge_types'].items():
            summary += f"- {edge_type}: {count}\n"
        
        return summary


def create_esm2_hypergraph(config: Dict[str, Any]) -> ESM2Hypergraph:
    """Factory function to create ESM-2 hypergraph from configuration"""
    return ESM2Hypergraph(config)


if __name__ == "__main__":
    # Example configuration from the problem statement
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
    
    # Create hypergraph
    hypergraph = create_esm2_hypergraph(config)
    
    # Print summary
    print(hypergraph.visualize_summary())
    
    # Save to JSON
    hypergraph.save_to_json("esm2_hypergraph.json")
    print(f"\nHypergraph saved to esm2_hypergraph.json")
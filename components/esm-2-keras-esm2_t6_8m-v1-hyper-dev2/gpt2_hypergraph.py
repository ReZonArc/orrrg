"""
GPT-2 Hypergraph Mapping
Maps the full computational hypergraph of the GPT-2 transformer model.
A hypergraph represents the model architecture where hyperedges connect multiple nodes.
"""

import json
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from tensor_shape_types import TensorShapeType, TensorShapeTypeRegistry, create_tensor_shape_type


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
    input_shape_type: Optional[TensorShapeType] = None
    output_shape_type: Optional[TensorShapeType] = None


@dataclass
class HyperEdge:
    """Represents a hyperedge connecting multiple nodes"""
    id: str
    name: str
    source_nodes: List[str]  # Input node IDs
    target_nodes: List[str]  # Output node IDs
    edge_type: str  # e.g., 'data_flow', 'residual', 'attention'


class GPT2Hypergraph:
    """
    Hypergraph representation of the GPT-2 model architecture
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
        self.use_pre_layer_norm = True  # GPT-2 uses pre-layer norm
        
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
        """Add input processing nodes: token embedding, position embedding"""
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
        
        # Position embedding (learnable, not rotary like ESM-2)
        self._add_node(
            "position_embedding",
            "Position Embedding",
            "embedding",
            -1,
            (None, self.max_seq_length),
            (None, self.max_seq_length, self.hidden_dim),
            {"max_position": self.max_seq_length, "embed_dim": self.hidden_dim}
        )
        
        # Embedding dropout
        if self.dropout > 0:
            self._add_node(
                "embedding_dropout",
                "Embedding Dropout",
                "dropout",
                -1,
                (None, self.max_seq_length, self.hidden_dim),
                (None, self.max_seq_length, self.hidden_dim),
                {"rate": self.dropout}
            )
    
    def _add_transformer_layer(self, layer_idx: int):
        """Add all components of a single transformer layer (GPT-2 style)"""
        layer_prefix = f"layer_{layer_idx}"
        
        # Pre-attention layer normalization (GPT-2 style)
        self._add_node(
            f"{layer_prefix}_pre_attn_norm",
            f"Layer {layer_idx} Pre-Attention Norm",
            "layernorm",
            layer_idx,
            (None, self.max_seq_length, self.hidden_dim),
            (None, self.max_seq_length, self.hidden_dim),
            {"eps": self.layer_norm_eps}
        )
        
        # Multi-head causal self-attention components
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
        
        # Pre-FFN layer normalization (GPT-2 style)
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
    
    def _add_attention_nodes(self, layer_idx: int):
        """Add multi-head causal self-attention nodes"""
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
        
        # Multi-head causal attention computation
        self._add_node(
            f"{layer_prefix}_multihead_attn",
            f"Layer {layer_idx} Multi-Head Causal Attention",
            "causal_attention",  # Different from ESM-2's bidirectional attention
            layer_idx,
            (None, self.max_seq_length, self.hidden_dim),
            (None, self.max_seq_length, self.hidden_dim),
            {
                "num_heads": self.num_heads,
                "head_dim": head_dim,
                "scale": head_dim ** -0.5,
                "causal_mask": True  # Key difference from ESM-2
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
        
        # Activation function (GELU for GPT-2)
        self._add_node(
            f"{layer_prefix}_ffn_activation",
            f"Layer {layer_idx} FFN Activation",
            "activation",
            layer_idx,
            (None, self.max_seq_length, self.intermediate_dim),
            (None, self.max_seq_length, self.intermediate_dim),
            {"activation": "gelu"}  # GPT-2 uses GELU
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
        
        # Language modeling head (output projection to vocabulary)
        self._add_node(
            "lm_head",
            "Language Modeling Head",
            "output",
            self.num_layers,
            (None, self.max_seq_length, self.hidden_dim),
            (None, self.max_seq_length, self.vocab_size),
            {"use_bias": False}  # GPT-2 typically doesn't use bias in final layer
        )
    
    def _add_data_flow_edges(self):
        """Add hyperedges representing data flow between nodes"""
        # Input flow: token + position embeddings
        self._add_edge(
            "input_flow",
            "Input Processing Flow",
            [],
            ["token_embedding", "position_embedding"],
            "data_flow"
        )
        
        # Embedding sum and dropout
        self._add_edge(
            "embedding_sum",
            "Token + Position Embedding Sum",
            ["token_embedding", "position_embedding"],
            ["embedding_dropout"] if self.dropout > 0 else ["layer_0_pre_attn_norm"],
            "residual"
        )
        
        if self.dropout > 0:
            self._add_edge(
                "embedding_to_layers",
                "Embedding Dropout to Layers",
                ["embedding_dropout"],
                ["layer_0_pre_attn_norm"],
                "data_flow"
            )
        
        # Connect transformer layers
        for layer_idx in range(self.num_layers):
            layer_prefix = f"layer_{layer_idx}"
            
            # Attention block flow
            self._add_edge(
                f"{layer_prefix}_attn_prep",
                f"Layer {layer_idx} Attention Preparation",
                [f"{layer_prefix}_pre_attn_norm"],
                [f"{layer_prefix}_query_proj", f"{layer_prefix}_key_proj", f"{layer_prefix}_value_proj"],
                "attention_prep"
            )
            
            self._add_edge(
                f"{layer_prefix}_attn_compute",
                f"Layer {layer_idx} Attention Computation",
                [f"{layer_prefix}_query_proj", f"{layer_prefix}_key_proj", f"{layer_prefix}_value_proj"],
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
            attn_output = f"{layer_prefix}_attn_dropout" if self.dropout > 0 else f"{layer_prefix}_attn_output_proj"
            if self.dropout > 0:
                self._add_edge(
                    f"{layer_prefix}_attn_dropout_flow",
                    f"Layer {layer_idx} Attention Dropout",
                    [f"{layer_prefix}_attn_output_proj"],
                    [f"{layer_prefix}_attn_dropout"],
                    "data_flow"
                )
            
            # Input to attention block
            prev_output = "embedding_dropout" if layer_idx == 0 and self.dropout > 0 else "token_embedding" if layer_idx == 0 else f"layer_{layer_idx-1}_ffn_dropout" if self.dropout > 0 else f"layer_{layer_idx-1}_ffn_output"
            if layer_idx == 0:
                prev_output = "embedding_dropout" if self.dropout > 0 else ["token_embedding", "position_embedding"]
            
            if isinstance(prev_output, list):
                self._add_edge(
                    f"{layer_prefix}_residual_attn",
                    f"Layer {layer_idx} Attention Residual",
                    prev_output + [attn_output],
                    [f"{layer_prefix}_pre_ffn_norm"],
                    "residual"
                )
            else:
                self._add_edge(
                    f"{layer_prefix}_residual_attn",
                    f"Layer {layer_idx} Attention Residual",
                    [prev_output, attn_output],
                    [f"{layer_prefix}_pre_ffn_norm"],
                    "residual"
                )
            
            # FFN block flow
            self._add_edge(
                f"{layer_prefix}_ffn_flow",
                f"Layer {layer_idx} FFN Flow",
                [f"{layer_prefix}_pre_ffn_norm"],
                [f"{layer_prefix}_ffn_intermediate"],
                "data_flow"
            )
            
            self._add_edge(
                f"{layer_prefix}_ffn_forward",
                f"Layer {layer_idx} FFN Forward",
                [f"{layer_prefix}_ffn_intermediate", f"{layer_prefix}_ffn_activation", f"{layer_prefix}_ffn_output"],
                [f"{layer_prefix}_ffn_dropout"] if self.dropout > 0 else ["final_layer_norm" if layer_idx == self.num_layers - 1 else f"layer_{layer_idx+1}_pre_attn_norm"],
                "feed_forward"
            )
            
            if self.dropout > 0:
                next_target = "final_layer_norm" if layer_idx == self.num_layers - 1 else f"layer_{layer_idx+1}_pre_attn_norm"
                self._add_edge(
                    f"{layer_prefix}_ffn_dropout_flow",
                    f"Layer {layer_idx} FFN Dropout",
                    [f"{layer_prefix}_ffn_dropout"],
                    [next_target],
                    "data_flow"
                )
            
            # Residual connection around FFN
            ffn_output = f"{layer_prefix}_ffn_dropout" if self.dropout > 0 else f"{layer_prefix}_ffn_output"
            next_target = "final_layer_norm" if layer_idx == self.num_layers - 1 else f"layer_{layer_idx+1}_pre_attn_norm"
            
            self._add_edge(
                f"{layer_prefix}_residual_ffn",
                f"Layer {layer_idx} FFN Residual",
                [f"{layer_prefix}_pre_ffn_norm", ffn_output],
                [next_target],
                "residual"
            )
        
        # Final output flow
        self._add_edge(
            "final_output",
            "Final Output",
            ["final_layer_norm"],
            ["lm_head"],
            "data_flow"
        )
    
    def _add_node(self, node_id: str, name: str, node_type: str, layer_idx: int,
                  input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                  parameters: Dict[str, Any]):
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
    
    def _add_edge(self, edge_id: str, name: str, source_nodes: List[str],
                  target_nodes: List[str], edge_type: str):
        """Add a hyperedge to the hypergraph"""
        edge = HyperEdge(
            id=edge_id,
            name=name,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_type=edge_type
        )
        self.edges[edge_id] = edge
        
        # Update adjacency information
        for source in source_nodes:
            for target in target_nodes:
                self.adjacency[source].add(target)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Generate statistics about the hypergraph"""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        max_hyperedge_size = 0
        
        for node in self.nodes.values():
            node_types[node.type] += 1
        
        for edge in self.edges.values():
            edge_types[edge.edge_type] += 1
            edge_size = len(edge.source_nodes) + len(edge.target_nodes)
            max_hyperedge_size = max(max_hyperedge_size, edge_size)
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "max_hyperedge_size": max_hyperedge_size
        }
    
    def visualize_summary(self) -> str:
        """Generate a text summary of the hypergraph structure"""
        stats = self.get_statistics()
        summary = f"""
GPT-2 Hypergraph Summary
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
    
    def save_to_json(self, filename: str):
        """Save the hypergraph to a JSON file"""
        data = {
            "config": self.config,
            "nodes": {node_id: asdict(node) for node_id, node in self.nodes.items()},
            "edges": {edge_id: asdict(edge) for edge_id, edge in self.edges.items()},
            "statistics": self.get_statistics()
        }
        
        # Convert tuples to lists for JSON serialization
        for node_data in data["nodes"].values():
            node_data["input_shape"] = list(node_data["input_shape"])
            node_data["output_shape"] = list(node_data["output_shape"])
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


def create_gpt2_hypergraph(config: Dict[str, Any]) -> GPT2Hypergraph:
    """Factory function to create GPT-2 hypergraph from configuration"""
    return GPT2Hypergraph(config)


if __name__ == "__main__":
    # Example GPT-2 configuration
    config = {
        "name": "gpt2_small",
        "trainable": True,
        "vocabulary_size": 50257,  # GPT-2 vocabulary size
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "intermediate_dim": 3072,  # 4 * hidden_dim
        "dropout": 0.1,
        "max_wavelength": 10000,
        "use_bias": True,
        "activation": "gelu",
        "layer_norm_eps": 1e-5,
        "use_pre_layer_norm": True,
        "position_embedding_type": "learned",
        "max_sequence_length": 1024,
        "pad_token_id": 50256
    }
    
    # Create hypergraph
    hypergraph = create_gpt2_hypergraph(config)
    
    # Print summary
    print(hypergraph.visualize_summary())
    
    # Save to JSON
    hypergraph.save_to_json("gpt2_hypergraph.json")
    print(f"\nHypergraph saved to gpt2_hypergraph.json")
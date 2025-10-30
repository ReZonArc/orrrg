#!/usr/bin/env python3
"""
Example usage of GPT-2 hypergraph implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2_hypergraph import create_gpt2_hypergraph
from gpt2_metagraph import create_gpt2_metagraph


def example_gpt2_small():
    """Example creating GPT-2 Small hypergraph"""
    print("Creating GPT-2 Small Hypergraph")
    print("=" * 40)
    
    config = {
        "name": "gpt2_small",
        "trainable": True,
        "vocabulary_size": 50257,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "intermediate_dim": 3072,
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
    hypergraph.save_to_json("gpt2_small_example.json")
    print("Saved to gpt2_small_example.json")
    
    return hypergraph


def example_gpt2_medium():
    """Example creating GPT-2 Medium hypergraph"""
    print("\nCreating GPT-2 Medium Hypergraph")
    print("=" * 40)
    
    config = {
        "name": "gpt2_medium",
        "trainable": True,
        "vocabulary_size": 50257,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "intermediate_dim": 4096,
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
    
    # Print statistics
    stats = hypergraph.get_statistics()
    print(f"GPT-2 Medium Statistics:")
    print(f"- Total Nodes: {stats['total_nodes']}")
    print(f"- Total Edges: {stats['total_edges']}")
    print(f"- Max Hyperedge Size: {stats['max_hyperedge_size']}")
    
    return hypergraph


def example_gpt2_metagraph():
    """Example creating GPT-2 metagraph with tensor types"""
    print("\nCreating GPT-2 MetaGraph with Tensor Types")
    print("=" * 45)
    
    config = {
        "name": "gpt2_metagraph_example",
        "trainable": True,
        "vocabulary_size": 50257,
        "num_layers": 6,  # Smaller for demo
        "num_heads": 8,
        "hidden_dim": 512,
        "intermediate_dim": 2048,
        "dropout": 0.1,
        "max_wavelength": 10000,
        "use_bias": True,
        "activation": "gelu",
        "layer_norm_eps": 1e-5,
        "use_pre_layer_norm": True,
        "position_embedding_type": "learned",
        "max_sequence_length": 512,
        "pad_token_id": 50256
    }
    
    # Create metagraph
    metagraph = create_gpt2_metagraph(config)
    
    # Print enhanced summary
    print(metagraph.visualize_metagraph_summary())
    
    # Save to JSON
    metagraph.save_metagraph_to_json("gpt2_metagraph_example.json")
    print("Saved to gpt2_metagraph_example.json")
    
    return metagraph


def analyze_architectural_differences():
    """Analyze key architectural differences in GPT-2 vs ESM-2"""
    print("\nArchitectural Analysis: GPT-2 vs ESM-2")
    print("=" * 45)
    
    from esm2_hypergraph import create_esm2_hypergraph
    
    # Create both models with similar sizes for comparison
    gpt2_config = {
        "name": "gpt2_analysis",
        "vocabulary_size": 50257,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_dim": 512,
        "intermediate_dim": 2048,
        "dropout": 0.1,
        "use_bias": True,
        "activation": "gelu",
        "layer_norm_eps": 1e-5,
        "use_pre_layer_norm": True,
        "position_embedding_type": "learned",
        "max_sequence_length": 512,
        "pad_token_id": 50256,
        "trainable": True,
        "max_wavelength": 10000
    }
    
    esm2_config = {
        "name": "esm2_analysis",
        "vocabulary_size": 33,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_dim": 512,
        "intermediate_dim": 2048,
        "dropout": 0.0,
        "use_bias": True,
        "activation": "gelu",
        "layer_norm_eps": 1e-5,
        "use_pre_layer_norm": False,
        "position_embedding_type": "rotary",
        "max_sequence_length": 512,
        "pad_token_id": 1,
        "trainable": True,
        "max_wavelength": 10000
    }
    
    # Create both hypergraphs
    gpt2_graph = create_gpt2_hypergraph(gpt2_config)
    esm2_graph = create_esm2_hypergraph(esm2_config)
    
    # Compare statistics
    gpt2_stats = gpt2_graph.get_statistics()
    esm2_stats = esm2_graph.get_statistics()
    
    print(f"Model Comparison (same layer/head count):")
    print(f"")
    print(f"{'Metric':<25} {'GPT-2':<10} {'ESM-2':<10}")
    print(f"{'-'*45}")
    print(f"{'Total Nodes':<25} {gpt2_stats['total_nodes']:<10} {esm2_stats['total_nodes']:<10}")
    print(f"{'Total Edges':<25} {gpt2_stats['total_edges']:<10} {esm2_stats['total_edges']:<10}")
    print(f"{'Vocabulary Size':<25} {gpt2_graph.vocab_size:<10} {esm2_graph.vocab_size:<10}")
    print(f"{'Max Seq Length':<25} {gpt2_graph.max_seq_length:<10} {esm2_graph.max_seq_length:<10}")
    
    print(f"\nKey Architectural Differences:")
    print(f"• Attention: GPT-2 (causal) vs ESM-2 (bidirectional)")
    print(f"• Position: GPT-2 (learned) vs ESM-2 (rotary)")
    print(f"• LayerNorm: GPT-2 (pre-norm) vs ESM-2 (post-norm)")
    print(f"• Domain: GPT-2 (text) vs ESM-2 (proteins)")
    
    # Analyze attention nodes specifically
    gpt2_attn = gpt2_graph.nodes["layer_0_multihead_attn"]
    esm2_attn = esm2_graph.nodes["layer_0_multihead_attn"]
    
    print(f"\nAttention Node Analysis:")
    print(f"• GPT-2 attention type: {gpt2_attn.type}")
    print(f"  - Causal mask: {gpt2_attn.parameters.get('causal_mask', False)}")
    print(f"• ESM-2 attention type: {esm2_attn.type}")
    print(f"  - Bidirectional: {not esm2_attn.parameters.get('causal_mask', False)}")


if __name__ == "__main__":
    # Run examples
    example_gpt2_small()
    example_gpt2_medium()
    example_gpt2_metagraph()
    analyze_architectural_differences()
    
    print(f"\n" + "="*60)
    print("GPT-2 Hypergraph Examples Complete!")
    print("="*60)
    print("\nFiles generated:")
    print("  - gpt2_small_example.json")
    print("  - gpt2_metagraph_example.json")
    print("\nTo create your own GPT-2 hypergraph:")
    print("  from gpt2_hypergraph import create_gpt2_hypergraph")
    print("  hypergraph = create_gpt2_hypergraph(your_config)")
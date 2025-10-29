#!/usr/bin/env python3
"""
Main script to generate the ESM-2 hypergraph mapping
"""

import json
import os
from esm2_hypergraph import create_esm2_hypergraph
from hypergraph_visualizer import create_visualization_report


def main():
    """Main function to create and analyze the ESM-2 hypergraph"""
    
    # Configuration from the problem statement
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
    
    print("Creating ESM-2 Hypergraph...")
    print("=" * 50)
    
    # Create the hypergraph
    hypergraph = create_esm2_hypergraph(config)
    
    # Print summary
    print(hypergraph.visualize_summary())
    
    # Save hypergraph to JSON
    print("Saving hypergraph to JSON...")
    hypergraph.save_to_json("esm2_hypergraph.json")
    print("✓ Hypergraph saved to esm2_hypergraph.json")
    
    # Create visualization report
    print("\nGenerating visualization report...")
    report = create_visualization_report(hypergraph)
    
    # Save report
    with open("hypergraph_analysis_report.md", "w") as f:
        f.write(report)
    print("✓ Analysis report saved to hypergraph_analysis_report.md")
    
    # Validate the hypergraph structure
    print("\nValidating hypergraph structure...")
    validation_results = validate_hypergraph(hypergraph)
    
    if validation_results["valid"]:
        print("✓ Hypergraph structure is valid")
    else:
        print("✗ Hypergraph structure has issues:")
        for issue in validation_results["issues"]:
            print(f"  - {issue}")
    
    print(f"\nHypergraph generation complete!")
    print(f"Files generated:")
    print(f"  - esm2_hypergraph.json (full hypergraph data)")
    print(f"  - hypergraph_analysis_report.md (analysis report)")
    print(f"  - esm2_hypergraph.dot (graph visualization)")


def validate_hypergraph(hypergraph):
    """Validate the hypergraph structure"""
    issues = []
    
    # Check that all edge references point to valid nodes
    for edge_id, edge in hypergraph.edges.items():
        for node_id in edge.source_nodes + edge.target_nodes:
            if node_id not in hypergraph.nodes:
                issues.append(f"Edge {edge_id} references non-existent node {node_id}")
    
    # Check that the graph is connected
    all_nodes = set(hypergraph.nodes.keys())
    connected_nodes = set()
    
    # Find nodes mentioned in edges
    for edge in hypergraph.edges.values():
        connected_nodes.update(edge.source_nodes + edge.target_nodes)
    
    disconnected_nodes = all_nodes - connected_nodes
    if disconnected_nodes:
        issues.append(f"Disconnected nodes found: {disconnected_nodes}")
    
    # Check layer consistency
    expected_layers = set(range(-1, hypergraph.num_layers + 1))
    actual_layers = set(node.layer_idx for node in hypergraph.nodes.values())
    
    if expected_layers != actual_layers:
        issues.append(f"Layer indices mismatch. Expected: {expected_layers}, Actual: {actual_layers}")
    
    # Check that each transformer layer has expected components
    for layer_idx in range(hypergraph.num_layers):
        layer_prefix = f"layer_{layer_idx}"
        expected_components = [
            f"{layer_prefix}_query_proj",
            f"{layer_prefix}_key_proj", 
            f"{layer_prefix}_value_proj",
            f"{layer_prefix}_multihead_attn",
            f"{layer_prefix}_attn_output_proj",
            f"{layer_prefix}_ffn_intermediate",
            f"{layer_prefix}_ffn_activation",
            f"{layer_prefix}_ffn_output"
        ]
        
        for component in expected_components:
            if component not in hypergraph.nodes:
                issues.append(f"Missing expected component: {component}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


if __name__ == "__main__":
    main()
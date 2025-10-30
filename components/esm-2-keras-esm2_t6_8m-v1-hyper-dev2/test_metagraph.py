#!/usr/bin/env python3
"""
Test script for the ESM-2 MetaGraph implementation with tensor shape types.
"""

import json
from esm2_metagraph import create_esm2_metagraph


def test_metagraph():
    """Test the metagraph implementation with tensor shape types."""
    
    # Configuration from the main example
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
    
    print("Creating ESM-2 MetaGraph with Tensor Shape Types...")
    print("=" * 60)
    
    # Create the metagraph
    metagraph = create_esm2_metagraph(config)
    
    # Print summary
    print(metagraph.visualize_metagraph_summary())
    
    # Get detailed analysis
    analysis = metagraph.get_topos_analysis()
    
    print("\nDetailed Prime Factorization Analysis:")
    print("-" * 40)
    
    # Show tensor shape types
    type_analysis = analysis['tensor_type_analysis']
    print(f"Most Common Tensor Shape Types:")
    for i, (type_sig, count) in enumerate(type_analysis['most_common_types'][:5], 1):
        canonical = None
        for shape_type in metagraph.shape_registry.types.values():
            if shape_type.type_signature == type_sig:
                canonical = shape_type.canonical_form
                break
        print(f"  {i}. {type_sig} ({canonical}) - {count} nodes")
    
    print(f"\nCanonical Mathematical Forms:")
    for i, form in enumerate(type_analysis['canonical_forms'][:10], 1):
        print(f"  {i}. {form}")
    
    # Show federated clusters
    print(f"\nFederated Clustering by Tensor Types:")
    print("-" * 40)
    clusters = metagraph.get_federated_clusters()
    for type_sig, cluster in list(clusters.items())[:5]:
        if cluster['node_count'] > 1:  # Only show clusters with multiple nodes
            print(f"Type: {type_sig}")
            print(f"  Mathematical Form: {cluster['canonical_form']}")
            print(f"  Topological Class: {cluster['topological_class']}")
            print(f"  Nodes ({cluster['node_count']}): {', '.join(list(cluster['nodes'])[:3])}{'...' if cluster['node_count'] > 3 else ''}")
            print()
    
    # Export to files
    print("Exporting MetaGraph...")
    metagraph.export_metagraph("esm2_metagraph.json")
    
    # Save analysis report
    with open("metagraph_analysis_report.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print("✓ MetaGraph exported to esm2_metagraph.json")
    print("✓ Analysis report saved to metagraph_analysis_report.json")
    
    return metagraph


if __name__ == "__main__":
    test_metagraph()
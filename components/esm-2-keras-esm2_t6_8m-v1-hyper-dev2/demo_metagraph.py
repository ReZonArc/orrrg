#!/usr/bin/env python3
"""
Complete demonstration of the ESM-2 MetaGraph with Prime Factorization Tensor Types.
Shows how tensor dimensions are converted to prime factorizations to create 
a typed hypergraph (metagraph) with federated clustering.
"""

from esm2_metagraph import create_esm2_metagraph
from tensor_shape_types import prime_factorization, factorization_signature


def demonstrate_prime_factorization():
    """Demonstrate prime factorization of ESM-2 dimensions."""
    print("Prime Factorization of ESM-2 Tensor Dimensions")
    print("=" * 55)
    
    dimensions = {
        "Hidden Dimension": 320,
        "Intermediate Dimension": 1280, 
        "Sequence Length": 1026,
        "Vocabulary Size": 33,
        "Number of Heads": 20,
        "Number of Layers": 6
    }
    
    for name, dim in dimensions.items():
        factors = prime_factorization(dim)
        signature = factorization_signature(factors)
        unique_primes = sorted(set(factors))
        
        print(f"{name}: {dim}")
        print(f"  Prime Factorization: {' × '.join(map(str, factors))}")
        print(f"  Signature: {signature}")
        print(f"  Unique Primes: {unique_primes}")
        print(f"  Mathematical Property:", end=" ")
        
        if len(unique_primes) == 1:
            print(f"Power of prime {unique_primes[0]}")
        elif len(unique_primes) == 2:
            print("Product of two prime families")
        else:
            print(f"Composite with {len(unique_primes)} distinct primes")
        print()


def demonstrate_metagraph():
    """Demonstrate the complete metagraph functionality."""
    print("ESM-2 MetaGraph with Tensor Shape Types")
    print("=" * 45)
    
    # Configuration
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
    
    # Create metagraph
    print("Creating metagraph with tensor shape types...")
    metagraph = create_esm2_metagraph(config)
    
    # Display summary
    print("\n" + metagraph.visualize_metagraph_summary())
    
    return metagraph


def demonstrate_topological_analysis(metagraph):
    """Demonstrate topological analysis of tensor types."""
    print("Topological Analysis of Tensor Shape Types")
    print("=" * 45)
    
    analysis = metagraph.get_topos_analysis()
    
    # Show tensor types with their mathematical properties
    print("Tensor Shape Types and Their Properties:")
    print("-" * 40)
    
    for i, shape_type in enumerate(metagraph.shape_registry.types.values(), 1):
        print(f"{i}. Type Signature: {shape_type.type_signature}")
        print(f"   Canonical Form: {shape_type.canonical_form}")
        print(f"   Dimensions: {shape_type.dimensions}")
        
        # Analyze prime structure of each dimension
        for j, (dim, factors) in enumerate(zip(shape_type.dimensions, shape_type.prime_factors)):
            if dim is None:
                print(f"   Dim {j}: Batch (variable)")
            elif factors:
                unique_primes = len(set(factors))
                max_power = max(factors.count(p) for p in set(factors))
                print(f"   Dim {j}: {dim} = {' × '.join(map(str, factors))} ({unique_primes} primes, max power {max_power})")
            else:
                print(f"   Dim {j}: 1 (identity)")
        print()
    
    # Show federated clustering
    print("Federated Clustering by Tensor Types:")
    print("-" * 40)
    
    clusters = metagraph.get_federated_clusters()
    for type_sig, cluster in clusters.items():
        print(f"Cluster: {type_sig}")
        print(f"  Topological Class: {cluster['topological_class']}")
        print(f"  Bundle Dimension: {cluster['bundle_dimension']}")
        print(f"  Fiber Size: {cluster['node_count']} nodes")
        
        # Show sample nodes
        sample_nodes = list(cluster['nodes'])[:5]
        if len(sample_nodes) < cluster['node_count']:
            print(f"  Sample Nodes: {', '.join(sample_nodes)}...")
        else:
            print(f"  All Nodes: {', '.join(sample_nodes)}")
        print()


def demonstrate_topos_structure(metagraph):
    """Demonstrate the topos structure of the metagraph."""
    print("MetaGraph Topos Structure")
    print("=" * 30)
    
    topos = metagraph.topos_structure
    
    print(f"Category Theory Structure:")
    print(f"  Objects (Tensor Bundles): {len(topos['objects'])}")
    print(f"  Morphisms (Typed Edges): {len(topos['morphisms'])}")
    print()
    
    # Fibration structure
    print("Fibration π: E → B (Tensor Bundles → Shape Types):")
    fibration = topos['fibration']
    print(f"  Base Space: {fibration['base_space']}")
    print(f"  Total Space: {fibration['total_space']}")
    print(f"  Fiber Map:")
    
    for base_type, bundles in fibration['fibers'].items():
        bundle_count = len(bundles)
        print(f"    {base_type} ↦ {bundle_count} bundle{'s' if bundle_count != 1 else ''}")
    print()
    
    # Sheaf structure
    print("Sheaf Structure:")
    sheaf = topos['sheaf_structure']
    print(f"  Presheaf: {sheaf['presheaf']}")
    print(f"  Sheafification: {sheaf['sheafification']}")
    print(f"  Local Sections: {len(sheaf['local_sections'])} types with sections")


def demonstrate_mathematical_significance():
    """Demonstrate the mathematical significance of the approach."""
    print("Mathematical Significance of Prime Factorization Approach")
    print("=" * 60)
    
    print("Key Benefits:")
    print("1. **Unique Type Expressions**: Each tensor shape has a unique prime signature")
    print("2. **Topological Classification**: Prime structure enables systematic topology")
    print("3. **Federated Clustering**: Nodes naturally cluster by shape type")
    print("4. **Compact Representation**: Reduced complexity through type abstraction")
    print("5. **Category Theory**: Formal mathematical foundation via topos theory")
    print()
    
    print("ESM-2 Specific Insights:")
    print("- Hidden dim (320 = 2^6*5) and intermediate dim (1280 = 2^8*5) share prime structure")
    print("- Sequence length (1026 = 2*3^3*19) has unique 3-prime composition")
    print("- Architecture enables efficient tensor bundle fibration")
    print("- Type compatibility analysis prevents dimension mismatches")
    print()
    
    print("Applications:")
    print("- Model optimization through type-aware operations")
    print("- Architectural analysis and comparison")  
    print("- Automatic tensor operation compatibility checking")
    print("- Compact mathematical representation of transformer models")


def main():
    """Main demonstration function."""
    print("ESM-2 MetaGraph with Prime Factorization Tensor Types")
    print("=" * 65)
    print("Demonstration of tensor dimension prime factorization for")
    print("creating a typed hypergraph (metagraph) with federated clustering")
    print("=" * 65)
    print()
    
    # 1. Demonstrate prime factorization
    demonstrate_prime_factorization()
    print("\n" + "=" * 65 + "\n")
    
    # 2. Create and demonstrate metagraph
    metagraph = demonstrate_metagraph()
    print("\n" + "=" * 65 + "\n")
    
    # 3. Topological analysis
    demonstrate_topological_analysis(metagraph)
    print("\n" + "=" * 65 + "\n")
    
    # 4. Topos structure
    demonstrate_topos_structure(metagraph)
    print("\n" + "=" * 65 + "\n")
    
    # 5. Mathematical significance
    demonstrate_mathematical_significance()
    
    print("\n" + "=" * 65)
    print("✓ Demonstration Complete!")
    print("✓ MetaGraph exported to esm2_metagraph.json")
    print("✓ Run 'python hypergraph_query.py --query tensor_types' for interactive queries")
    print("=" * 65)


if __name__ == "__main__":
    main()
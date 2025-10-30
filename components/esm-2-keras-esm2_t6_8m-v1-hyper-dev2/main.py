#!/usr/bin/env python3
"""
Main script to generate the ESM-2 and GPT-2 hypergraph mappings
"""

import json
import os
from esm2_hypergraph import create_esm2_hypergraph
from esm2_metagraph import create_esm2_metagraph
from gpt2_hypergraph import create_gpt2_hypergraph
from gpt2_metagraph import create_gpt2_metagraph
from hypergraph_visualizer import create_visualization_report


def create_gpt2_model():
    """Create and analyze GPT-2 hypergraph"""
    print("\n" + "="*60)
    print("Creating GPT-2 Hypergraph...")
    print("="*60)
    
    # GPT-2 Small configuration
    gpt2_config = {
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
    
    # Create the hypergraph
    hypergraph = create_gpt2_hypergraph(gpt2_config)
    
    # Print summary
    print(hypergraph.visualize_summary())
    
    # Create the enhanced metagraph with tensor shape types
    print(f"\n" + "="*60)
    print("Enhanced GPT-2 MetaGraph with Prime Factorization Tensor Types")
    print("="*60)
    
    metagraph = create_gpt2_metagraph(gpt2_config)
    print(metagraph.visualize_metagraph_summary())
    
    # Save outputs
    hypergraph.save_to_json("gpt2_hypergraph.json")
    metagraph.save_metagraph_to_json("gpt2_metagraph.json")
    
    print("✓ GPT-2 hypergraph saved to gpt2_hypergraph.json")
    print("✓ GPT-2 metagraph saved to gpt2_metagraph.json")
    
    return hypergraph, metagraph


def main():
    """Main function to create and analyze both ESM-2 and GPT-2 hypergraphs"""
    
    # ESM-2 Configuration from the problem statement
    esm2_config = {
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
    hypergraph = create_esm2_hypergraph(esm2_config)
    
    # Print summary
    print(hypergraph.visualize_summary())
    
    # Create the enhanced metagraph with tensor shape types
    print(f"\n" + "="*60)
    print("Enhanced MetaGraph with Prime Factorization Tensor Types")
    print("="*60)
    
    metagraph = create_esm2_metagraph(esm2_config)
    print(metagraph.visualize_metagraph_summary())
    
    # Export metagraph
    metagraph.export_metagraph("esm2_metagraph.json")
    print("✓ MetaGraph with tensor types exported to esm2_metagraph.json")
    
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
    
    # Create GPT-2 models
    create_gpt2_model()
    
    print(f"\n" + "="*60)
    print("Both ESM-2 and GPT-2 Analysis Complete!")
    print("="*60)
    print("\nTransformer architectures implemented:")
    print("• ESM-2: Protein language model with bidirectional attention")
    print("• GPT-2: Causal language model with masked attention")
    print("\nKey architectural differences:")
    print("• Attention: ESM-2 (bidirectional) vs GPT-2 (causal/masked)")
    print("• Position encoding: ESM-2 (rotary) vs GPT-2 (learned)")
    print("• Layer norm: ESM-2 (post-norm) vs GPT-2 (pre-norm)")
    print("• Vocabulary: ESM-2 (33 proteins) vs GPT-2 (50257 tokens)")
    
    # Demonstrate new ESM-2 structure prediction capabilities
    print(f"\n" + "="*60)
    print("ESM-2 Structure Prediction Analysis Demo")
    print("="*60)
    
    # Structure analysis demo
    print("\n1. Structure Prediction Analysis:")
    print("-" * 40)
    try:
        from structure_analysis import ESM2StructureAnalyzer
        structure_analyzer = ESM2StructureAnalyzer(hypergraph)
        
        demo_sequences = [
            "MKLLVLGLGGTAAMAAAQPQPAPQPSAPQPLPQLPLQAQPQPQPQPQQLQQM",
            "MKLLVLGLGGTAAMAGGGGPSPQPLPQLPLQAQPQPQPQPQQLQQMKLLVL"
        ]
        
        structure_report = structure_analyzer.generate_structure_report(demo_sequences)
        stats = structure_report["aggregate_statistics"]
        corr = structure_report["correlations"]
        
        print(f"✓ Analyzed {len(demo_sequences)} protein sequences")
        print(f"  - Mean TM-score: {stats['mean_tm_score']:.3f}")
        print(f"  - Mean Contact Precision: {stats['mean_contact_precision']:.3f}")
        print(f"  - Mean Perplexity: {stats['mean_perplexity']:.3f}")
        print(f"  - Perplexity-TM correlation: {corr['perplexity_tm_score']:.3f}")
        
        with open("structure_analysis_demo.json", "w") as f:
            json.dump(structure_report, f, indent=2)
        print(f"  - Detailed report: structure_analysis_demo.json")
        
    except Exception as e:
        print(f"✗ Structure analysis failed: {e}")
    
    # Scaling analysis demo
    print("\n2. Model Scaling Analysis:")
    print("-" * 40)
    try:
        from scaling_analysis import ESM2ScalingAnalyzer
        scaling_analyzer = ESM2ScalingAnalyzer()
        
        demo_sequences = [
            "MKLLVLGLGGTAAMAAAQPQPAPQPSAPQPLPQLPLQAQPQPQPQPQQLQQM",
            "MKLLVLGLGGTAAMAGGGGPSPQPLPQLPLQAQPQPQPQPQQLQQMKLLVL",
            "MEEGLLAAGGGPSPQPLPQLPLQAQPQPQPQPQPQQLQQMKLLVLGLGGTAAM"
        ]
        
        scaling_report = scaling_analyzer.generate_scaling_report(demo_sequences)
        summary = scaling_report["summary"]
        trends = scaling_report["scaling_analysis"]["scaling_trends"]
        
        print(f"✓ Analyzed scaling across {summary['num_models_analyzed']} model sizes")
        print(f"  - Parameter range: {summary['parameter_range']}")
        print(f"  - TM-score scaling: r={trends['tm_score_vs_size']['correlation']:.3f}")
        print(f"  - Structure emerges at 3B+ parameters")
        
        with open("scaling_analysis_demo.json", "w") as f:
            json.dump(scaling_report, f, indent=2)
        print(f"  - Detailed report: scaling_analysis_demo.json")
        
    except Exception as e:
        print(f"✗ Scaling analysis failed: {e}")
    
    # Speed analysis demo
    print("\n3. Folding Speed Analysis:")
    print("-" * 40)
    try:
        from folding_speed_analysis import ESMFoldSpeedAnalyzer
        speed_analyzer = ESMFoldSpeedAnalyzer()
        
        test_lengths = [100, 200, 384, 500]
        speed_report = speed_analyzer.generate_speed_report(test_lengths)
        
        key_findings = speed_report["summary"]["key_findings"]
        efficiency = speed_report["speed_comparison"]["computational_efficiency"]
        meta = speed_report["metagenomic_scalability"]
        
        print(f"✓ Speed analysis across {len(test_lengths)} sequence lengths")
        print(f"  - Key finding: {key_findings[0]}")
        print(f"  - ESMFold efficiency: {efficiency['esmfold']:.3f} TM-score/min")
        print(f"  - Metagenomic speedup: {meta['speedup_factor']:.1f}x faster")
        
        with open("speed_analysis_demo.json", "w") as f:
            json.dump(speed_report, f, indent=2)
        print(f"  - Detailed report: speed_analysis_demo.json")
        
    except Exception as e:
        print(f"✗ Speed analysis failed: {e}")
    
    # Hypergredient framework demo
    print("\n4. Hypergredient Framework Demo:")
    print("-" * 40)
    try:
        from hypergredient_framework import HypergredientDatabase, HypergredientOptimizer, FormulationRequest
        
        database = HypergredientDatabase()
        optimizer = HypergredientOptimizer(database)
        
        # Generate optimal anti-aging formulation
        request = FormulationRequest(
            target_concerns=['wrinkles', 'firmness'],
            secondary_concerns=['dryness'],
            skin_type='normal',
            budget=800.0,
            preferences=['gentle', 'effective']
        )
        
        result = optimizer.optimize_formulation(request)
        
        print(f"✓ Generated optimal formulation with {len(result.selected_hypergredients)} hypergredients")
        print(f"  - Total cost: R{result.total_cost:.2f}")
        print(f"  - Predicted efficacy: {result.predicted_efficacy:.1%}")
        print(f"  - Safety score: {result.safety_score:.1f}/10")
        print(f"  - Synergy score: {result.synergy_score:.2f}")
        print(f"  - Stability: {result.stability_months} months")
        
        # Save demo results
        hypergredient_demo = {
            "formulation_request": {
                "target_concerns": request.target_concerns,
                "secondary_concerns": request.secondary_concerns,
                "budget": request.budget
            },
            "formulation_result": {
                "total_cost": result.total_cost,
                "predicted_efficacy": result.predicted_efficacy,
                "safety_score": result.safety_score,
                "synergy_score": result.synergy_score,
                "stability_months": result.stability_months,
                "selected_ingredients": {
                    class_name: {
                        "name": data['ingredient'].name,
                        "percentage": data['percentage'],
                        "reasoning": data['reasoning']
                    }
                    for class_name, data in result.selected_hypergredients.items()
                }
            },
            "database_info": {
                "total_hypergredients": len(database.hypergredients),
                "classes": ["H.CT", "H.CS", "H.AO", "H.BR", "H.ML", "H.HY", "H.AI", "H.MB", "H.SE", "H.PD"]
            }
        }
        
        with open("hypergredient_framework_demo.json", "w") as f:
            json.dump(hypergredient_demo, f, indent=2)
        print(f"  - Detailed report: hypergredient_framework_demo.json")
        
    except Exception as e:
        print(f"✗ Hypergredient framework demo failed: {e}")
    
    print(f"\n" + "="*60)
    print("ESM-2 Analysis Complete!")
    print("="*60)
    print("\nNew capabilities based on ESM-2 paper:")
    print("• Structure prediction from attention patterns")
    print("• Perplexity-accuracy correlation analysis")
    print("• Model scaling behavior (8M to 15B parameters)")
    print("• Speed comparison with AlphaFold/RosettaFold")
    print("• Metagenomic-scale analysis feasibility")
    print("\n🧬 NEW: Hypergredient Framework Architecture:")
    print("• Multi-objective cosmetic formulation optimization")
    print("• Real-time ingredient compatibility analysis")
    print("• Dynamic hypergredient database with 10 classes")
    print("• Advanced synergy calculation and performance prediction")
    print("• Cost-effectiveness optimization and safety scoring")
    print("\nQuery these capabilities:")   
    print("  python3 hypergraph_query.py --query structure")
    print("  python3 hypergraph_query.py --query scaling")
    print("  python3 hypergraph_query.py --query speed")
    print("  python3 hypergraph_query.py --query hypergredient     # Generate optimal formulation")
    print("  python3 hypergraph_query.py --query compatibility     # Check ingredient compatibility")
    print("  python3 hypergraph_query.py --query ingredient --ingredient-id tretinoin  # Ingredient profile")


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
#!/usr/bin/env python3
"""
Hypergraph Query Utilities
Provides tools to query and analyze the ESM-2 hypergraph structure.
"""

import json
import argparse
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from esm2_hypergraph import ESM2Hypergraph, create_esm2_hypergraph
from esm2_metagraph import ESM2MetaGraph, create_esm2_metagraph


class HypergraphQueryEngine:
    """Query engine for hypergraph analysis"""
    
    def __init__(self, hypergraph: ESM2Hypergraph):
        self.hypergraph = hypergraph
    
    def find_nodes_by_type(self, node_type: str) -> List[str]:
        """Find all nodes of a specific type"""
        return [
            node_id for node_id, node in self.hypergraph.nodes.items()
            if node.type == node_type
        ]
    
    def find_nodes_by_layer(self, layer_idx: int) -> List[str]:
        """Find all nodes in a specific layer"""
        return [
            node_id for node_id, node in self.hypergraph.nodes.items()
            if node.layer_idx == layer_idx
        ]
    
    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get all nodes that this node depends on (inputs)"""
        dependencies = []
        for edge in self.hypergraph.edges.values():
            if node_id in edge.target_nodes:
                dependencies.extend(edge.source_nodes)
        return list(set(dependencies))
    
    def get_node_dependents(self, node_id: str) -> List[str]:
        """Get all nodes that depend on this node (outputs)"""
        dependents = []
        for edge in self.hypergraph.edges.values():
            if node_id in edge.source_nodes:
                dependents.extend(edge.target_nodes)
        return list(set(dependents))
    
    def get_computational_path(self, start_node: str, end_node: str) -> Optional[List[str]]:
        """Find computational path between two nodes"""
        from collections import deque
        
        if start_node == end_node:
            return [start_node]
        
        queue = deque([(start_node, [start_node])])
        visited = {start_node}
        
        while queue:
            current, path = queue.popleft()
            
            # Get next nodes
            next_nodes = self.get_node_dependents(current)
            
            for next_node in next_nodes:
                if next_node == end_node:
                    return path + [next_node]
                
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))
        
        return None
    
    def analyze_parameter_flow(self) -> Dict[str, Any]:
        """Analyze parameter flow through the network"""
        param_count = {}
        
        for node_id, node in self.hypergraph.nodes.items():
            if node.type == "linear":
                input_size = node.input_shape[-1] if node.input_shape else 0
                output_size = node.output_shape[-1] if node.output_shape else 0
                
                # Calculate parameters (weights + bias if used)
                params = input_size * output_size
                if node.parameters.get("use_bias", False):
                    params += output_size
                
                param_count[node_id] = params
            elif node.type == "embedding":
                vocab_size = node.parameters.get("vocab_size", 0)
                embed_dim = node.parameters.get("embed_dim", 0)
                param_count[node_id] = vocab_size * embed_dim
            else:
                param_count[node_id] = 0
        
        total_params = sum(param_count.values())
        
        return {
            "total_parameters": total_params,
            "parameters_by_node": param_count,
            "parameters_by_layer": self._group_params_by_layer(param_count)
        }
    
    def _group_params_by_layer(self, param_count: Dict[str, int]) -> Dict[int, int]:
        """Group parameters by layer"""
        layer_params = defaultdict(int)
        
        for node_id, params in param_count.items():
            node = self.hypergraph.nodes[node_id]
            layer_params[node.layer_idx] += params
        
        return dict(layer_params)
    
    def get_attention_structure(self) -> Dict[str, Any]:
        """Analyze the attention structure"""
        attention_nodes = self.find_nodes_by_type("attention")
        
        structure = {
            "num_attention_layers": len(attention_nodes),
            "attention_details": {}
        }
        
        for node_id in attention_nodes:
            node = self.hypergraph.nodes[node_id]
            layer_idx = node.layer_idx
            
            structure["attention_details"][f"layer_{layer_idx}"] = {
                "node_id": node_id,
                "num_heads": node.parameters.get("num_heads", 0),
                "head_dim": node.parameters.get("head_dim", 0),
                "scale": node.parameters.get("scale", 0)
            }
        
        return structure
    
    def find_bottlenecks(self) -> List[Dict[str, Any]]:
        """Find potential bottlenecks in the architecture"""
        bottlenecks = []
        
        # Find nodes with high fan-in or fan-out
        for node_id, node in self.hypergraph.nodes.items():
            fan_in = len(self.get_node_dependencies(node_id))
            fan_out = len(self.get_node_dependents(node_id))
            
            if fan_in > 3 or fan_out > 3:
                bottlenecks.append({
                    "node_id": node_id,
                    "node_name": node.name,
                    "fan_in": fan_in,
                    "fan_out": fan_out,
                    "type": node.type
                })
        
        return sorted(bottlenecks, key=lambda x: x["fan_in"] + x["fan_out"], reverse=True)
    
    def export_subgraph(self, layer_start: int, layer_end: int) -> Dict[str, Any]:
        """Export a subgraph for specific layers"""
        subgraph_nodes = {}
        subgraph_edges = {}
        
        # Get nodes in the specified layer range
        for node_id, node in self.hypergraph.nodes.items():
            if layer_start <= node.layer_idx <= layer_end:
                subgraph_nodes[node_id] = node
        
        # Get edges that connect nodes within the subgraph
        for edge_id, edge in self.hypergraph.edges.items():
            if (all(n in subgraph_nodes for n in edge.source_nodes) and
                all(n in subgraph_nodes for n in edge.target_nodes)):
                subgraph_edges[edge_id] = edge
        
        return {
            "nodes": {nid: node.__dict__ for nid, node in subgraph_nodes.items()},
            "edges": {eid: edge.__dict__ for eid, edge in subgraph_edges.items()},
            "layer_range": [layer_start, layer_end]
        }


def main():
    """Main CLI interface for hypergraph queries"""
    parser = argparse.ArgumentParser(description="Query ESM-2 Hypergraph")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--query", choices=[
        "stats", "attention", "params", "bottlenecks", "path", "subgraph", "structure", "scaling", "speed",
        "metagraph", "tensor_types", "tensor_bundles", "topos", "prime_analysis",
        "hypergredient", "compatibility", "ingredient", "persona", "persona_train"
    ], required=True, help="Query type")
    parser.add_argument("--start", help="Start node for path query")
    parser.add_argument("--end", help="End node for path query")
    parser.add_argument("--layer-start", type=int, help="Start layer for subgraph")
    parser.add_argument("--layer-end", type=int, help="End layer for subgraph")
    parser.add_argument("--type", help="Node type to filter")
    parser.add_argument("--ingredient-id", help="Ingredient ID for ingredient profile query")
    
    args = parser.parse_args()
    
    # Load configuration
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
    
    # Create hypergraph and query engine
    hypergraph = create_esm2_hypergraph(config)
    query_engine = HypergraphQueryEngine(hypergraph)
    
    # Execute query
    if args.query == "stats":
        stats = hypergraph.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.query == "attention":
        attention = query_engine.get_attention_structure()
        print(json.dumps(attention, indent=2))
    
    elif args.query == "params":
        params = query_engine.analyze_parameter_flow()
        print(json.dumps(params, indent=2))
    
    elif args.query == "bottlenecks":
        bottlenecks = query_engine.find_bottlenecks()
        print(json.dumps(bottlenecks, indent=2))
    
    elif args.query == "path":
        if not args.start or not args.end:
            print("Error: --start and --end required for path query")
            return
        
        path = query_engine.get_computational_path(args.start, args.end)
        if path:
            print(f"Path from {args.start} to {args.end}:")
            print(" -> ".join(path))
        else:
            print(f"No path found from {args.start} to {args.end}")
    
    elif args.query == "subgraph":
        if args.layer_start is None or args.layer_end is None:
            print("Error: --layer-start and --layer-end required for subgraph query")
            return
        
        subgraph = query_engine.export_subgraph(args.layer_start, args.layer_end)
        print(json.dumps(subgraph, indent=2))
    
    elif args.query == "structure":
        from structure_analysis import ESM2StructureAnalyzer
        analyzer = ESM2StructureAnalyzer(hypergraph)
        
        # Example sequences for demonstration
        test_sequences = [
            "MKLLVLGLGGTAAMAAAQPQPAPQPSAPQPLPQLPLQAQPQPQPQPQQLQQM",
            "MKLLVLGLGGTAAMAGGGGPSPQPLPQLPLQAQPQPQPQPQQLQQMKLLVL"
        ]
        
        report = analyzer.generate_structure_report(test_sequences)
        print(json.dumps(report, indent=2))
    
    elif args.query == "scaling":
        from scaling_analysis import ESM2ScalingAnalyzer
        analyzer = ESM2ScalingAnalyzer()
        
        # Example sequences for demonstration
        test_sequences = [
            "MKLLVLGLGGTAAMAAAQPQPAPQPSAPQPLPQLPLQAQPQPQPQPQQLQQM",
            "MKLLVLGLGGTAAMAGGGGPSPQPLPQLPLQAQPQPQPQPQQLQQMKLLVL",
            "MEEGLLAAGGGPSPQPLPQLPLQAQPQPQPQPQQLQQMKLLVLGLGGTAAM"
        ]
        
        report = analyzer.generate_scaling_report(test_sequences)
        print(json.dumps(report, indent=2))
    
    elif args.query == "speed":
        from folding_speed_analysis import ESMFoldSpeedAnalyzer
        analyzer = ESMFoldSpeedAnalyzer()
        
        # Test with common sequence lengths
        test_lengths = [50, 100, 200, 384, 500, 1000]
        report = analyzer.generate_speed_report(test_lengths)
        print(json.dumps(report, indent=2))
    
    # NEW: MetaGraph queries with tensor shape types
    elif args.query == "metagraph":
        # Create metagraph with tensor shape types
        metagraph = create_esm2_metagraph(config)
        analysis = metagraph.get_topos_analysis()
        print(json.dumps(analysis, indent=2, default=str))
    
    elif args.query == "tensor_types":
        # Query tensor shape types and prime factorizations
        metagraph = create_esm2_metagraph(config)
        type_analysis = metagraph.get_topos_analysis()['tensor_type_analysis']
        
        print("Tensor Shape Types with Prime Factorizations:")
        print("=" * 50)
        
        # Show most common types
        for i, (type_sig, count) in enumerate(type_analysis['most_common_types'][:10], 1):
            canonical = None
            for shape_type in metagraph.shape_registry.types.values():
                if shape_type.type_signature == type_sig:
                    canonical = shape_type.canonical_form
                    break
            print(f"{i:2d}. {type_sig}")
            print(f"    Mathematical Form: {canonical}")
            print(f"    Node Count: {count}")
            print()
    
    elif args.query == "tensor_bundles":
        # Query tensor bundles fibred over shape types
        metagraph = create_esm2_metagraph(config)
        clusters = metagraph.get_federated_clusters()
        
        print("Tensor Bundles Fibred over Prime Factor Shape Types:")
        print("=" * 55)
        
        for type_sig, cluster in clusters.items():
            print(f"Bundle: {type_sig}")
            print(f"  Canonical Form: {cluster['canonical_form']}")
            print(f"  Topological Class: {cluster['topological_class']}")
            print(f"  Bundle Dimension: {cluster['bundle_dimension']}")
            print(f"  Fiber Size: {cluster['node_count']} nodes")
            if cluster['node_count'] <= 5:
                print(f"  Nodes: {', '.join(cluster['nodes'])}")
            else:
                print(f"  Sample Nodes: {', '.join(list(cluster['nodes'])[:3])}...")
            print()
    
    elif args.query == "topos":
        # Query topos structure of the metagraph
        metagraph = create_esm2_metagraph(config)
        topos = metagraph.topos_structure
        
        print("MetaGraph Topos Structure:")
        print("=" * 30)
        print(f"Objects (Tensor Bundles): {len(topos['objects'])}")
        print(f"Morphisms (Typed Edges): {len(topos['morphisms'])}")
        print()
        
        print("Fibration Structure:")
        fibration = topos['fibration']
        print(f"  Base Space: {fibration['base_space']}")
        print(f"  Total Space: {fibration['total_space']}")
        print(f"  Fiber Types: {len(fibration['fibers'])}")
        
        for base_type, bundles in fibration['fibers'].items():
            print(f"    {base_type} -> {len(bundles)} bundle(s)")
        print()
        
        print("Grothendieck Topology:")
        topology = topos['grothendieck_topology']
        print(f"  Covering Families: {len(topology['covers'])}")
        for cover_type, covers in list(topology['covers'].items())[:3]:
            print(f"    {cover_type} covered by {len(covers)} types")

    elif args.query == "prime_analysis":
        # Detailed prime factorization analysis
        metagraph = create_esm2_metagraph(config)
        
        print("Prime Factorization Analysis of ESM-2 Tensor Dimensions:")
        print("=" * 60)
        
        # Analyze the prime structure of all dimensions
        dimension_analysis = defaultdict(list)
        for shape_type in metagraph.shape_registry.types.values():
            for i, (dim, factors) in enumerate(zip(shape_type.dimensions, shape_type.prime_factors)):
                if dim is not None:
                    unique_primes = sorted(set(factors))
                    dimension_analysis[dim].append({
                        'factors': factors,
                        'unique_primes': unique_primes,
                        'signature': f"{dim} = " + " × ".join(map(str, factors)) if factors else "1",
                        'type_signature': shape_type.type_signature,
                        'dimension_index': i
                    })
        
        # Show analysis by dimension value
        for dim in sorted(dimension_analysis.keys()):
            analyses = dimension_analysis[dim]
            print(f"Dimension {dim}:")
            for analysis in analyses[:1]:  # Show first occurrence
                print(f"  Prime Factorization: {analysis['signature']}")
                print(f"  Unique Primes: {analysis['unique_primes']}")
                print(f"  Mathematical Significance: ", end="")
                if len(analysis['unique_primes']) == 1:
                    print(f"Power of prime {analysis['unique_primes'][0]}")
                elif len(analysis['unique_primes']) == 2:
                    print(f"Product of two prime families")
                else:
                    print(f"Composite with {len(analysis['unique_primes'])} distinct primes")
            print()
    
    elif args.query == "hypergredient":
        from hypergredient_framework import HypergredientDatabase, HypergredientOptimizer, HypergredientAnalyzer, FormulationRequest
        
        # Initialize hypergredient system
        database = HypergredientDatabase()
        optimizer = HypergredientOptimizer(database)
        analyzer = HypergredientAnalyzer(database)
        
        # Generate sample anti-aging formulation
        request = FormulationRequest(
            target_concerns=['wrinkles', 'firmness'],
            secondary_concerns=['dryness'],
            skin_type='normal',
            budget=1000.0,
            preferences=['gentle', 'effective']
        )
        
        result = optimizer.optimize_formulation(request)
        
        # Format result for JSON output
        hypergredient_report = {
            "formulation_request": {
                "target_concerns": request.target_concerns,
                "secondary_concerns": request.secondary_concerns,
                "skin_type": request.skin_type,
                "budget": request.budget,
                "preferences": request.preferences
            },
            "optimal_formulation": {
                "total_cost": result.total_cost,
                "predicted_efficacy": result.predicted_efficacy,
                "safety_score": result.safety_score,
                "synergy_score": result.synergy_score,
                "stability_months": result.stability_months,
                "selected_hypergredients": {}
            },
            "database_stats": {
                "total_hypergredients": len(database.hypergredients),
                "hypergredient_classes": ["H.CT", "H.CS", "H.AO", "H.BR", "H.ML", "H.HY", "H.AI", "H.MB", "H.SE", "H.PD"],
                "interaction_matrix_size": len(database.interaction_matrix)
            }
        }
        
        # Add selected ingredients details
        for class_name, data in result.selected_hypergredients.items():
            ingredient = data['ingredient']
            hypergredient_report["optimal_formulation"]["selected_hypergredients"][class_name] = {
                "name": ingredient.name,
                "inci_name": ingredient.inci_name,
                "percentage": data['percentage'],
                "cost": data['cost'],
                "efficacy_score": ingredient.efficacy_score,
                "safety_score": ingredient.safety_score,
                "reasoning": data['reasoning']
            }
        
        print(json.dumps(hypergredient_report, indent=2))
    
    elif args.query == "compatibility":
        from hypergredient_framework import HypergredientDatabase, HypergredientAnalyzer
        
        database = HypergredientDatabase()
        analyzer = HypergredientAnalyzer(database)
        
        # Test ingredient compatibility
        test_ingredients = ['retinol', 'bakuchiol', 'niacinamide', 'hyaluronic_acid']
        compatibility_report = analyzer.generate_compatibility_report(test_ingredients)
        
        print(json.dumps(compatibility_report, indent=2))
    
    elif args.query == "ingredient":
        from hypergredient_framework import HypergredientDatabase, HypergredientAnalyzer
        
        database = HypergredientDatabase()
        analyzer = HypergredientAnalyzer(database)
        
        # Get ingredient profile (use bakuchiol as example)
        ingredient_id = getattr(args, 'ingredient_id', 'bakuchiol')
        profile = analyzer.generate_ingredient_profile(ingredient_id)
        
        print(json.dumps(profile, indent=2))
    
    elif args.query == "persona":
        from hypergredient_framework import PersonaTrainingSystem, HypergredientAI, FormulationRequest
        
        # Initialize persona system
        persona_system = PersonaTrainingSystem()
        ai_system = HypergredientAI(persona_system)
        
        # Show available personas
        persona_summary = persona_system.get_training_summary()
        
        persona_report = {
            "system_info": {
                "total_personas": persona_summary["total_personas"],
                "active_persona": persona_summary["active_persona"]
            },
            "available_personas": {}
        }
        
        for persona_id, info in persona_summary["personas"].items():
            persona_report["available_personas"][persona_id] = {
                "name": info["name"],
                "description": info["description"],
                "skin_type": info["skin_type"],
                "primary_concerns": info["primary_concerns"],
                "sensitivity_level": info["sensitivity_level"],
                "training_samples": info["training_samples"]
            }
        
        # Demonstrate persona-aware predictions
        test_request = FormulationRequest(
            target_concerns=['wrinkles', 'sensitivity'],
            skin_type='sensitive',
            budget=500.0,
            preferences=['gentle', 'effective']
        )
        
        persona_report["demo_predictions"] = {}
        
        for persona_id in ['sensitive_skin', 'anti_aging']:
            ai_system.persona_system.set_active_persona(persona_id)
            prediction = ai_system.predict_optimal_combination(test_request)
            
            persona_report["demo_predictions"][persona_id] = {
                "persona_name": persona_system.personas[persona_id].name,
                "top_recommendations": [
                    {
                        "ingredient_class": pred["ingredient_class"],
                        "confidence": pred["confidence"],
                        "reasoning": pred["reasoning"]
                    }
                    for pred in prediction["predictions"][:3]
                ],
                "persona_adjustments": prediction["persona_adjustments"]
            }
        
        print(json.dumps(persona_report, indent=2))
    
    elif args.query == "persona_train":
        from hypergredient_framework import (PersonaTrainingSystem, HypergredientAI, HypergredientDatabase,
                                            FormulationRequest, FormulationResult)
        
        # Initialize systems
        persona_system = PersonaTrainingSystem()
        ai_system = HypergredientAI(persona_system)
        database = HypergredientDatabase()
        
        # Simulate training data for sensitive skin persona
        training_requests = [
            FormulationRequest(['sensitivity', 'redness'], skin_type='sensitive', budget=400),
            FormulationRequest(['barrier_repair'], skin_type='sensitive', budget=500),
            FormulationRequest(['dryness', 'sensitivity'], skin_type='sensitive', budget=600)
        ]
        
        training_results = []
        training_feedback = []
        
        for i, req in enumerate(training_requests):
            # Simulate formulation result
            result = FormulationResult(
                selected_hypergredients={
                    'H.AI': {
                        'ingredient': database.hypergredients['niacinamide'],
                        'percentage': 5.0, 'cost': 25.0,
                        'reasoning': f'Anti-inflammatory for sensitive skin case {i+1}'
                    }
                },
                total_cost=300.0 + i * 50,
                predicted_efficacy=0.7 + i * 0.05,
                safety_score=9.5,
                stability_months=24,
                synergy_score=0.8,
                reasoning={'H.AI': f'Optimized for sensitivity case {i+1}'}
            )
            training_results.append(result)
            training_feedback.append({
                'efficacy': 8.0 + i * 0.5,
                'safety': 9.5 + i * 0.1,
                'user_satisfaction': 8.5 + i * 0.3
            })
        
        # Train the persona
        ai_system.train_with_persona('sensitive_skin', training_requests, training_results, training_feedback)
        
        # Generate training report
        updated_summary = persona_system.get_training_summary()
        training_report = {
            "training_completed": True,
            "persona_trained": "sensitive_skin",
            "training_samples_added": len(training_requests),
            "total_training_samples": updated_summary["personas"]["sensitive_skin"]["training_samples"],
            "training_iterations": updated_summary["personas"]["sensitive_skin"]["training_iterations"],
            "persona_characteristics": {
                "name": updated_summary["personas"]["sensitive_skin"]["name"],
                "sensitivity_level": updated_summary["personas"]["sensitive_skin"]["sensitivity_level"],
                "primary_concerns": updated_summary["personas"]["sensitive_skin"]["primary_concerns"]
            },
            "feedback_summary": {
                "avg_efficacy": sum(fb['efficacy'] for fb in training_feedback) / len(training_feedback),
                "avg_safety": sum(fb['safety'] for fb in training_feedback) / len(training_feedback),
                "avg_satisfaction": sum(fb['user_satisfaction'] for fb in training_feedback) / len(training_feedback)
            }
        }
        
        print(json.dumps(training_report, indent=2))


if __name__ == "__main__":
    main()
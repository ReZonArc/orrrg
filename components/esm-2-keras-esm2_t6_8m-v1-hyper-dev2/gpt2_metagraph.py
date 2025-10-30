"""
GPT-2 MetaGraph with Prime Factorization Tensor Shape Types
Extends the GPT-2 hypergraph with tensor type system based on the ESM-2 approach.
"""

import json
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from gpt2_hypergraph import GPT2Hypergraph, create_gpt2_hypergraph
from tensor_shape_types import (
    TensorShapeType, TensorShapeTypeRegistry, create_tensor_shape_type,
    ComputationalMode, OperatorType
)


@dataclass
class SimpleTensorBundle:
    """Simple tensor bundle for GPT-2 metagraph."""
    base_type: str
    fiber_nodes: Set[str]
    bundle_dimension: int
    topological_class: str = "euclidean_space"
    computational_mode: str = "spatial_concurrent"
    operator_type: str = "product_grammar"
    optimization_enabled: bool = True


@dataclass
class SimpleTypedHyperEdge:
    """Simple typed hyperedge for GPT-2 metagraph."""
    base_edge_id: str
    input_types: List[str]
    output_types: List[str]
    transformation_type: str
    compatibility_score: float
    optimization_enabled: bool = True


class GPT2MetaGraph(GPT2Hypergraph):
    """
    GPT-2 MetaGraph with tensor shape type system
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize the base hypergraph
        super().__init__(config)
        
        # Initialize tensor shape type system
        self.shape_registry = TensorShapeTypeRegistry()
        self.tensor_bundles: Dict[str, SimpleTensorBundle] = {}
        self.typed_edges: Dict[str, SimpleTypedHyperEdge] = {}
        
        # Build enhanced metagraph with tensor types
        self._build_tensor_types()
        self._create_tensor_bundles()
        self._create_typed_edges()
    
    def _build_tensor_types(self):
        """Build tensor shape types for all nodes in the hypergraph"""
        for node_id, node in self.nodes.items():
            input_type = self.shape_registry.register_shape_type(node.input_shape, f"{node_id}_input")
            output_type = self.shape_registry.register_shape_type(node.output_shape, f"{node_id}_output")
            
            # Update node with tensor types
            node.input_shape_type = input_type
            node.output_shape_type = output_type
    
    def _create_tensor_bundles(self):
        """Create tensor bundles fibred over shape types."""
        type_clusters = self.shape_registry.get_type_clusters()
        
        for type_sig, node_ids in type_clusters.items():
            if type_sig not in self.shape_registry.types:
                continue
                
            shape_type = self.shape_registry.types[type_sig]
            
            # Create simple tensor bundle
            bundle = SimpleTensorBundle(
                base_type=type_sig,
                fiber_nodes=set(node_ids),
                bundle_dimension=len(shape_type.dimensions),
                topological_class=self._classify_topology(shape_type),
                computational_mode=self._classify_computational_mode(shape_type),
                operator_type=self._classify_operator_type(shape_type)
            )
            
            self.tensor_bundles[type_sig] = bundle
    
    def _create_typed_edges(self):
        """Create typed hyperedges with tensor compatibility information."""
        for edge_id, edge in self.edges.items():
            input_types = []
            output_types = []
            
            # Collect input tensor types
            for source_id in edge.source_nodes:
                if source_id in self.nodes:
                    node = self.nodes[source_id]
                    if node.output_shape_type:
                        input_types.append(node.output_shape_type.type_signature)
                        
            # Collect output tensor types
            for target_id in edge.target_nodes:
                if target_id in self.nodes:
                    node = self.nodes[target_id]
                    if node.input_shape_type:
                        output_types.append(node.input_shape_type.type_signature)
            
            # Determine transformation type
            transformation_type = self._determine_transformation_type(input_types, output_types)
            
            # Calculate compatibility score
            compatibility_score = self._calculate_compatibility_score(input_types, output_types)
            
            # Create typed edge
            typed_edge = SimpleTypedHyperEdge(
                base_edge_id=edge_id,
                input_types=input_types,
                output_types=output_types,
                transformation_type=transformation_type,
                compatibility_score=compatibility_score
            )
            
            self.typed_edges[edge_id] = typed_edge
    
    def _classify_topology(self, shape_type: TensorShapeType) -> str:
        """Classify tensor topology"""
        # Simple classification based on dimensionality
        num_dims = len([d for d in shape_type.dimensions if d is not None])
        if num_dims <= 1:
            return "euclidean_space"
        elif num_dims == 2:
            return "projective_space"
        else:
            return "hyperbolic_space"
    
    def _classify_computational_mode(self, shape_type: TensorShapeType) -> str:
        """Classify computational mode"""
        # For GPT-2, most operations are spatial concurrent due to parallelizable matrix ops
        return "spatial_concurrent"
    
    def _classify_operator_type(self, shape_type: TensorShapeType) -> str:
        """Classify operator type"""
        # Most GPT-2 operations involve product grammars (matrix multiplications)
        return "product_grammar"
    
    def _determine_transformation_type(self, input_types: List[str], output_types: List[str]) -> str:
        """Determine the type of tensor transformation"""
        if not input_types or not output_types:
            return "identity"
        
        if len(input_types) == len(output_types) == 1 and input_types[0] == output_types[0]:
            return "identity"
        elif len(input_types) > 1 and len(output_types) == 1:
            return "fusion"
        elif len(input_types) == 1 and len(output_types) > 1:
            return "split"
        else:
            return "transformation"
    
    def _calculate_compatibility_score(self, input_types: List[str], output_types: List[str]) -> float:
        """Calculate compatibility score between input and output types"""
        if not input_types or not output_types:
            return 1.0
        
        # Simple heuristic based on type matching
        matches = 0
        total = 0
        
        for input_type in input_types:
            for output_type in output_types:
                if input_type == output_type:
                    matches += 1
                total += 1
        
        return matches / total if total > 0 else 0.5
    
    def get_metagraph_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive statistics about the metagraph"""
        base_stats = self.get_statistics()
        
        # Tensor type statistics
        type_distribution = defaultdict(int)
        computational_modes = defaultdict(int)
        operator_types = defaultdict(int)
        topological_classes = defaultdict(int)
        
        for bundle in self.tensor_bundles.values():
            type_distribution[f"Rank-{bundle.bundle_dimension}"] += 1
            computational_modes[bundle.computational_mode] += 1
            operator_types[bundle.operator_type] += 1
            topological_classes[bundle.topological_class] += 1
        
        # Transformation type statistics
        transformation_types = defaultdict(int)
        compatibility_scores = []
        
        for typed_edge in self.typed_edges.values():
            transformation_types[typed_edge.transformation_type] += 1
            compatibility_scores.append(typed_edge.compatibility_score)
        
        avg_compatibility = sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.0
        
        return {
            **base_stats,
            "tensor_types": {
                "total_shape_types": len(self.shape_registry.types),
                "unique_mathematical_structures": len(set(t.type_signature for t in self.shape_registry.types.values())),
                "nodes_with_types": sum(len(bundle.fiber_nodes) for bundle in self.tensor_bundles.values())
            },
            "optimization_config": {
                "operator_types": dict(operator_types),
                "computational_modes": dict(computational_modes),
                "topological_classes": dict(topological_classes)
            },
            "tensor_bundles": {
                "total_bundles": len(self.tensor_bundles),
                "average_fiber_size": sum(len(bundle.fiber_nodes) for bundle in self.tensor_bundles.values()) / len(self.tensor_bundles) if self.tensor_bundles else 0,
                "dimension_distribution": dict(type_distribution)
            },
            "type_compatibility": {
                "average_compatibility_score": avg_compatibility,
                "transformation_types": dict(transformation_types)
            }
        }
    
    def visualize_metagraph_summary(self) -> str:
        """Generate a comprehensive summary of the metagraph"""
        stats = self.get_metagraph_statistics()
        
        summary = f"""
GPT-2 MetaGraph with Tensor Shape Types
==================================================
Configuration: {self.config['name']}
Base Architecture: {stats['total_nodes']} nodes, {stats['total_edges']} edges

Tensor Shape Type System:
- Total Shape Types: {stats['tensor_types']['total_shape_types']}
- Unique Mathematical Structures: {stats['tensor_types']['unique_mathematical_structures']}
- Nodes with Types: {stats['tensor_types']['nodes_with_types']}

Optimization Configuration:
- Operator Types:
"""
        
        for op_type, count in stats['optimization_config']['operator_types'].items():
            summary += f"  * {op_type}: {count} types\n"
        
        summary += "- Computational Modes:\n"
        for mode, count in stats['optimization_config']['computational_modes'].items():
            summary += f"  * {mode}: {count} types\n"
        
        summary += "- Topological Classes:\n"
        for topo_class, count in stats['optimization_config']['topological_classes'].items():
            summary += f"  * {topo_class}: {count} types\n"
        
        summary += f"""
Tensor Bundle Fibration:
- Total Bundles: {stats['tensor_bundles']['total_bundles']}
- Average Fiber Size: {stats['tensor_bundles']['average_fiber_size']:.1f}
- Dimension Distribution:
"""
        
        for dim_type, count in stats['tensor_bundles']['dimension_distribution'].items():
            summary += f"  * {dim_type} Bundles: {count}\n"
        
        summary += f"""
Type Compatibility Analysis:
- Average Compatibility Score: {stats['type_compatibility']['average_compatibility_score']:.3f}
- Transformation Types:
"""
        
        for trans_type, count in stats['type_compatibility']['transformation_types'].items():
            summary += f"  * {trans_type.capitalize()}: {count}\n"
        
        return summary
    
    def save_metagraph_to_json(self, filename: str):
        """Save the complete metagraph with tensor types to JSON"""
        # Base hypergraph data
        base_data = {
            "config": self.config,
            "nodes": {node_id: asdict(node) for node_id, node in self.nodes.items()},
            "edges": {edge_id: asdict(edge) for edge_id, edge in self.edges.items()},
            "statistics": self.get_metagraph_statistics()
        }
        
        # Convert tuples and complex objects for JSON serialization
        for node_data in base_data["nodes"].values():
            node_data["input_shape"] = list(node_data["input_shape"])
            node_data["output_shape"] = list(node_data["output_shape"])
            # Shape types are already None or TensorShapeType objects
            if node_data["input_shape_type"] and hasattr(node_data["input_shape_type"], 'type_signature'):
                node_data["input_shape_type"] = node_data["input_shape_type"].type_signature
            if node_data["output_shape_type"] and hasattr(node_data["output_shape_type"], 'type_signature'):
                node_data["output_shape_type"] = node_data["output_shape_type"].type_signature
        
        # Add tensor type system data (simplified)
        base_data["tensor_shape_types"] = {
            type_sig: {
                "type_signature": shape_type.type_signature,
                "dimensions": list(shape_type.dimensions),
                "canonical_form": shape_type.canonical_form
            } for type_sig, shape_type in self.shape_registry.types.items()
        }
        
        base_data["tensor_bundles"] = {
            bundle_id: asdict(bundle) for bundle_id, bundle in self.tensor_bundles.items()
        }
        
        base_data["typed_edges"] = {
            edge_id: asdict(typed_edge) for edge_id, typed_edge in self.typed_edges.items()
        }
        
        with open(filename, 'w') as f:
            json.dump(base_data, f, indent=2, default=str)


def create_gpt2_metagraph(config: Dict[str, Any]) -> GPT2MetaGraph:
    """Factory function to create GPT-2 metagraph with tensor types"""
    return GPT2MetaGraph(config)


if __name__ == "__main__":
    # Example GPT-2 configuration
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
    
    # Create metagraph
    metagraph = create_gpt2_metagraph(config)
    
    # Print summary
    print(metagraph.visualize_metagraph_summary())
    
    # Save to JSON
    metagraph.save_metagraph_to_json("gpt2_metagraph.json")
    print(f"\n✓ MetaGraph with tensor types exported to gpt2_metagraph.json")
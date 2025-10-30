"""
ESM-2 MetaGraph Implementation
Extends the hypergraph representation with tensor shape types as prime factorizations,
enabling federated clustering by type and tensor bundle fibration over shape types.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import json

from esm2_hypergraph import ESM2Hypergraph, HyperNode, HyperEdge
from tensor_shape_types import (
    TensorShapeType, TensorShapeTypeRegistry, create_tensor_shape_type,
    analyze_tensor_type_distribution, OperatorType, ComputationalMode
)


@dataclass
class TensorBundle:
    """Represents a tensor bundle fibred over prime factor shape types."""
    base_type: str  # Base tensor shape type signature
    fiber_nodes: Set[str]  # Node IDs with this shape type
    bundle_dimension: int  # Rank of tensors in this bundle
    topological_class: str  # Topological classification
    
    # New optimization fields
    computational_mode: ComputationalMode  # Spatial concurrent vs temporal asymmetric
    operator_type: OperatorType  # Prime index grammar vs product grammar
    node_values: List[int]  # Prime values for fractal nodes
    edge_weights: List[int]  # Power weights for fractal edges
    reducibility_pattern: str  # Pattern of reducible/irreducible structure
    
    
@dataclass 
class TypedHyperEdge(HyperEdge):
    """Hyperedge with tensor type compatibility information."""
    input_types: List[str]  # Input tensor type signatures
    output_types: List[str]  # Output tensor type signatures
    type_transformation: str  # Description of type transformation
    compatibility_score: float  # Measure of type compatibility
    
    # New optimization fields
    computational_complexity: str  # P=NP optimization class
    spatial_concurrent_nodes: List[str]  # Nodes in spatial concurrent mode
    temporal_sequence_nodes: List[str]  # Nodes in temporal asymmetric mode
    fractal_depth: int  # Depth in fractal neural network structure


class ESM2MetaGraph(ESM2Hypergraph):
    """
    MetaGraph representation of ESM-2 with tensor shape types as prime factorizations.
    
    This creates a typed hypergraph (metagraph) where:
    1. Each tensor dimension is represented by its prime factorization
    2. Nodes are clustered by tensor shape types  
    3. Tensor bundles are fibred over prime factor shape types
    4. Federated clustering enables compact mathematical representation
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize the base hypergraph first
        super().__init__(config)
        
        # Initialize tensor type system
        self.shape_registry = TensorShapeTypeRegistry()
        self.tensor_bundles: Dict[str, TensorBundle] = {}
        self.typed_edges: Dict[str, TypedHyperEdge] = {}
        self.type_clusters: Dict[str, Set[str]] = defaultdict(set)
        self.topos_structure: Dict[str, Any] = {}
        
        # Build the metagraph with tensor types
        self._build_metagraph()
        
    def _build_metagraph(self):
        """Build the complete metagraph with tensor type annotations."""
        # Annotate existing nodes with tensor shape types
        self._annotate_tensor_types()
        
        # Create tensor bundles fibred over shape types  
        self._create_tensor_bundles()
        
        # Create typed hyperedges with compatibility information
        self._create_typed_edges()
        
        # Build topos structure representing the metagraph
        self._build_topos_structure()
        
    def _annotate_tensor_types(self):
        """Annotate all nodes with tensor shape type information."""
        for node_id, node in self.nodes.items():
            # Create shape types for input and output
            if node.input_shape:
                input_shape_type = self.shape_registry.register_shape_type(
                    node.input_shape, f"{node_id}_input"
                )
                node.input_shape_type = input_shape_type
                
            if node.output_shape:  
                output_shape_type = self.shape_registry.register_shape_type(
                    node.output_shape, f"{node_id}_output"
                )
                node.output_shape_type = output_shape_type
                
                # Add node to type cluster based on output shape
                self.type_clusters[output_shape_type.type_signature].add(node_id)
                
    def _create_tensor_bundles(self):
        """Create tensor bundles fibred over prime factor shape types."""
        type_clusters = self.shape_registry.get_type_clusters()
        
        for type_sig, node_ids in type_clusters.items():
            if type_sig not in self.shape_registry.types:
                continue
                
            shape_type = self.shape_registry.types[type_sig]
            
            # Determine topological class based on prime structure
            topological_class = self._classify_tensor_topology(shape_type)
            
            # Determine computational mode and operator type from shape type
            spatial_count = sum(1 for mode in shape_type.computational_modes 
                              if mode == ComputationalMode.SPATIAL_CONCURRENT)
            temporal_count = sum(1 for mode in shape_type.computational_modes 
                               if mode == ComputationalMode.TEMPORAL_ASYMMETRIC)
            
            computational_mode = (ComputationalMode.SPATIAL_CONCURRENT 
                                if spatial_count >= temporal_count 
                                else ComputationalMode.TEMPORAL_ASYMMETRIC)
            
            prime_count = sum(1 for op_type in shape_type.operator_types 
                            if op_type == OperatorType.PRIME_INDEX_GRAMMAR)
            product_count = sum(1 for op_type in shape_type.operator_types 
                              if op_type == OperatorType.PRODUCT_GRAMMAR)
            
            operator_type = (OperatorType.PRIME_INDEX_GRAMMAR 
                           if prime_count >= product_count 
                           else OperatorType.PRODUCT_GRAMMAR)
            
            # Extract node values and edge weights for fractal structure
            all_node_values = []
            all_edge_weights = []
            for nv, ew in zip(shape_type.node_values, shape_type.edge_weights):
                all_node_values.extend(nv)
                all_edge_weights.extend(ew)
            
            # Create tensor bundle with optimization features
            bundle = TensorBundle(
                base_type=type_sig,
                fiber_nodes=set(node_ids),
                bundle_dimension=len(shape_type.dimensions),
                topological_class=topological_class,
                computational_mode=computational_mode,
                operator_type=operator_type,
                node_values=sorted(set(all_node_values)),
                edge_weights=sorted(set(all_edge_weights)),
                reducibility_pattern=shape_type.reducibility_pattern
            )
            
            self.tensor_bundles[type_sig] = bundle
            
    def _classify_tensor_topology(self, shape_type: TensorShapeType) -> str:
        """Classify tensor topology based on prime factorization structure."""
        prime_structures = []
        
        for factors in shape_type.prime_factors:
            if not factors:  # Batch dimension
                prime_structures.append("variable")
            elif len(factors) == 1:  # Prime dimension
                prime_structures.append(f"prime_{factors[0]}")
            else:  # Composite dimension
                unique_primes = len(set(factors))
                if unique_primes == 1:
                    prime_structures.append(f"power_{factors[0]}^{len(factors)}")
                else:
                    prime_structures.append(f"composite_{unique_primes}primes")
        
        return "_".join(prime_structures)
        
    def _create_typed_edges(self):
        """Create typed hyperedges with tensor compatibility information."""
        for edge_id, edge in self.edges.items():
            input_types = []
            output_types = []
            spatial_concurrent_nodes = []
            temporal_sequence_nodes = []
            
            # Collect input tensor types and analyze computational modes
            for source_id in edge.source_nodes:
                if source_id in self.nodes:
                    node = self.nodes[source_id]
                    if node.output_shape_type:
                        input_types.append(node.output_shape_type.type_signature)
                        
                        # Classify node by computational mode
                        shape_type = node.output_shape_type
                        spatial_count = sum(1 for mode in shape_type.computational_modes 
                                          if mode == ComputationalMode.SPATIAL_CONCURRENT)
                        if spatial_count > 0:
                            spatial_concurrent_nodes.append(source_id)
                        else:
                            temporal_sequence_nodes.append(source_id)
                        
            # Collect output tensor types and analyze computational modes
            for target_id in edge.target_nodes:
                if target_id in self.nodes:
                    node = self.nodes[target_id]
                    if node.input_shape_type:
                        output_types.append(node.input_shape_type.type_signature)
                        
                        # Classify node by computational mode
                        shape_type = node.input_shape_type
                        spatial_count = sum(1 for mode in shape_type.computational_modes 
                                          if mode == ComputationalMode.SPATIAL_CONCURRENT)
                        if spatial_count > 0:
                            spatial_concurrent_nodes.append(target_id)
                        else:
                            temporal_sequence_nodes.append(target_id)
                        
            # Determine type transformation and compatibility
            type_transformation = self._analyze_type_transformation(input_types, output_types)
            compatibility_score = self._compute_compatibility_score(input_types, output_types)
            
            # Compute P=NP optimization class
            computational_complexity = self._compute_pnp_complexity(input_types, output_types)
            
            # Compute fractal depth
            fractal_depth = self._compute_fractal_depth(input_types + output_types)
            
            # Create typed hyperedge with optimization features
            typed_edge = TypedHyperEdge(
                id=edge.id,
                name=edge.name,
                source_nodes=edge.source_nodes,
                target_nodes=edge.target_nodes,
                edge_type=edge.edge_type,
                input_types=input_types,
                output_types=output_types,
                type_transformation=type_transformation,
                compatibility_score=compatibility_score,
                computational_complexity=computational_complexity,
                spatial_concurrent_nodes=spatial_concurrent_nodes,
                temporal_sequence_nodes=temporal_sequence_nodes,
                fractal_depth=fractal_depth
            )
            
            self.typed_edges[edge_id] = typed_edge
            
    def _analyze_type_transformation(self, input_types: List[str], output_types: List[str]) -> str:
        """Analyze the tensor type transformation performed by an edge."""
        if not input_types or not output_types:
            return "identity"
            
        if len(input_types) == 1 and len(output_types) == 1:
            if input_types[0] == output_types[0]:
                return "identity"
            else:
                return f"transform_{input_types[0]}_to_{output_types[0]}"
        elif len(input_types) > 1 and len(output_types) == 1:
            return f"fusion_{len(input_types)}to1"
        elif len(input_types) == 1 and len(output_types) > 1:
            return f"split_1to{len(output_types)}"
        else:
            return f"complex_{len(input_types)}to{len(output_types)}"
            
    def _compute_pnp_complexity(self, input_types: List[str], output_types: List[str]) -> str:
        """Compute P=NP optimization class for the edge.
        
        Since we have no addition/polynomials in prime factorization system,
        all operations are in P class due to multiplicative structure.
        """
        all_types = input_types + output_types
        
        # Count reducible vs irreducible types
        reducible_count = 0
        irreducible_count = 0
        
        for type_sig in all_types:
            if type_sig in self.shape_registry.types:
                shape_type = self.shape_registry.types[type_sig]
                if 'R' in shape_type.reducibility_pattern:
                    reducible_count += 1
                if 'I' in shape_type.reducibility_pattern:
                    irreducible_count += 1
        
        if reducible_count > irreducible_count:
            return "P_SPATIAL_CONCURRENT"  # Reducible products in spatial modes
        elif irreducible_count > 0:
            return "P_TEMPORAL_SEQUENCE"   # Irreducible primes in temporal modes
        else:
            return "P_VARIABLE"            # Variable batch dimensions
            
    def _compute_fractal_depth(self, type_signatures: List[str]) -> int:
        """Compute fractal depth based on prime power structures."""
        max_depth = 0
        
        for type_sig in type_signatures:
            if type_sig in self.shape_registry.types:
                shape_type = self.shape_registry.types[type_sig]
                fractal_info = shape_type.get_fractal_structure()
                max_depth = max(max_depth, fractal_info["max_power"])
                
        return max_depth
            
    def _compute_compatibility_score(self, input_types: List[str], output_types: List[str]) -> float:
        """Compute compatibility score between input and output tensor types."""
        if not input_types or not output_types:
            return 1.0
            
        # Simple compatibility metric based on type similarity
        total_pairs = len(input_types) * len(output_types)
        compatible_pairs = 0
        
        for in_type in input_types:
            for out_type in output_types:
                if in_type == out_type:
                    compatible_pairs += 1
                elif self._types_compatible(in_type, out_type):
                    compatible_pairs += 0.5
                    
        return compatible_pairs / total_pairs if total_pairs > 0 else 0.0
        
    def _types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two tensor types are compatible."""
        if type1 not in self.shape_registry.types or type2 not in self.shape_registry.types:
            return False
            
        compatible_types = self.shape_registry.get_compatible_types(type1)
        return type2 in compatible_types
        
    def _build_topos_structure(self):
        """Build the topos structure representing the metagraph."""
        # Compute category-theoretic structure
        self.topos_structure = {
            'objects': list(self.tensor_bundles.keys()),  # Tensor bundles as objects
            'morphisms': list(self.typed_edges.keys()),   # Typed edges as morphisms
            'fibration': self._compute_fibration_structure(),
            'grothendieck_topology': self._compute_grothendieck_topology(),
            'sheaf_structure': self._compute_sheaf_structure()
        }
        
    def _compute_fibration_structure(self) -> Dict[str, Any]:
        """Compute the fibration structure of tensor bundles over shape types."""
        fibration = {
            'base_space': 'shape_types',
            'total_space': 'tensor_bundles', 
            'projection': {},
            'fibers': {}
        }
        
        # Map bundles to their base shape types
        for bundle_id, bundle in self.tensor_bundles.items():
            fibration['projection'][bundle_id] = bundle.base_type
            
            if bundle.base_type not in fibration['fibers']:
                fibration['fibers'][bundle.base_type] = []
            fibration['fibers'][bundle.base_type].append(bundle_id)
            
        return fibration
        
    def _compute_grothendieck_topology(self) -> Dict[str, Any]:
        """Compute Grothendieck topology for the metagraph topos."""
        # Covering families based on tensor type compatibility
        topology = {
            'covers': {},
            'sieves': {}
        }
        
        for type_sig in self.shape_registry.types:
            compatible = self.shape_registry.get_compatible_types(type_sig)
            if compatible:
                topology['covers'][type_sig] = list(compatible)
                
        return topology
        
    def _compute_sheaf_structure(self) -> Dict[str, Any]:
        """Compute sheaf structure over the base topology."""
        return {
            'presheaf': 'tensor_data',
            'sheafification': 'global_sections',
            'local_sections': {
                bundle_id: list(bundle.fiber_nodes)
                for bundle_id, bundle in self.tensor_bundles.items()
            }
        }
        
    def get_federated_clusters(self) -> Dict[str, Dict[str, Any]]:
        """Get federated clustering of nodes by tensor shape types."""
        clusters = {}
        
        for type_sig, bundle in self.tensor_bundles.items():
            shape_type = self.shape_registry.types[type_sig]
            
            clusters[type_sig] = {
                'shape_signature': type_sig,
                'canonical_form': shape_type.canonical_form,
                'nodes': list(bundle.fiber_nodes),
                'bundle_dimension': bundle.bundle_dimension,
                'topological_class': bundle.topological_class,
                'node_count': len(bundle.fiber_nodes)
            }
            
        return clusters
        
    def get_topos_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of the metagraph topos structure."""
        type_analysis = analyze_tensor_type_distribution(self.shape_registry)
        
        return {
            'tensor_type_analysis': type_analysis,
            'topos_structure': self.topos_structure,
            'federated_clusters': self.get_federated_clusters(),
            'bundle_statistics': {
                'total_bundles': len(self.tensor_bundles),
                'average_fiber_size': sum(len(b.fiber_nodes) for b in self.tensor_bundles.values()) / len(self.tensor_bundles) if self.tensor_bundles else 0,
                'dimension_distribution': {
                    dim: sum(1 for b in self.tensor_bundles.values() if b.bundle_dimension == dim)
                    for dim in set(b.bundle_dimension for b in self.tensor_bundles.values())
                }
            },
            'compatibility_analysis': {
                'typed_edges': len(self.typed_edges),
                'average_compatibility': sum(e.compatibility_score for e in self.typed_edges.values()) / len(self.typed_edges) if self.typed_edges else 0,
                'transformation_types': {
                    t: sum(1 for e in self.typed_edges.values() if e.type_transformation.startswith(t))
                    for t in ['identity', 'transform', 'fusion', 'split', 'complex']
                }
            }
        }
        
    def visualize_metagraph_summary(self) -> str:
        """Generate a text summary of the metagraph structure with optimization features."""
        analysis = self.get_topos_analysis()
        
        summary = "ESM-2 MetaGraph with Tensor Shape Types\n"
        summary += "=" * 50 + "\n"
        summary += f"Configuration: {self.config['name']}\n"
        summary += f"Base Architecture: {len(self.nodes)} nodes, {len(self.edges)} edges\n\n"
        
        # Tensor type information
        type_stats = analysis['tensor_type_analysis']
        summary += "Tensor Shape Type System:\n"
        summary += f"- Total Shape Types: {type_stats['total_types']}\n"
        summary += f"- Unique Mathematical Structures: {type_stats['unique_mathematical_structures']}\n"
        summary += f"- Nodes with Types: {type_stats['total_nodes']}\n\n"
        
        # NEW: Optimization features analysis
        if 'computational_mode_distribution' in type_stats:
            summary += "Topological Optimization Configuration:\n"
            summary += "- Operator Types:\n"
            for op_type, count in type_stats['operator_type_distribution'].items():
                summary += f"  * {op_type.replace('_', ' ').title()}: {count} types\n"
            summary += "- Computational Modes:\n"
            for mode, count in type_stats['computational_mode_distribution'].items():
                summary += f"  * {mode.replace('_', ' ').title()}: {count} types\n"
            
            # P=NP optimization summary
            pnp_opt = type_stats['p_equals_np_optimization']
            summary += "- P=NP Optimization:\n"
            summary += f"  * Polynomial Class Eliminated: {pnp_opt['polynomial_class_eliminated']}\n"
            summary += f"  * Spatial Concurrent Advantage: {pnp_opt['spatial_concurrent_advantage']}\n"
            summary += f"  * Temporal Sequence Complexity: {pnp_opt['temporal_sequence_complexity']}\n"
            summary += "\n"
        
        # Bundle information
        bundle_stats = analysis['bundle_statistics']
        summary += "Tensor Bundle Fibration:\n"
        summary += f"- Total Bundles: {bundle_stats['total_bundles']}\n"
        summary += f"- Average Fiber Size: {bundle_stats['average_fiber_size']:.1f}\n"
        summary += "- Dimension Distribution:\n"
        for dim, count in sorted(bundle_stats['dimension_distribution'].items()):
            summary += f"  * Rank-{dim} Bundles: {count}\n"
        summary += "\n"
        
        # NEW: Fractal structure analysis
        if 'fractal_structure' in type_stats:
            fractal = type_stats['fractal_structure']
            summary += "Fractal Neural Network Structure:\n"
            summary += f"- Max Prime Depth: {fractal['max_prime_depth']}\n"
            summary += f"- Unique Primes: {len(fractal['unique_primes_across_all'])}\n"
            summary += f"- Power Distribution: {dict(list(fractal['power_distribution'].items())[:5])}\n"
            summary += f"- Reducibility Patterns: {len(fractal['reducibility_patterns'])}\n"
            summary += "\n"
        
        # Topos structure
        summary += "Metagraph Topos Structure:\n"
        summary += f"- Objects (Tensor Bundles): {len(self.topos_structure['objects'])}\n"
        summary += f"- Morphisms (Typed Edges): {len(self.topos_structure['morphisms'])}\n"
        summary += f"- Fibration Base Types: {len(self.topos_structure['fibration']['fibers'])}\n\n"
        
        # Compatibility analysis
        compat_stats = analysis['compatibility_analysis']
        summary += "Type Compatibility Analysis:\n"
        summary += f"- Average Compatibility Score: {compat_stats['average_compatibility']:.3f}\n"
        summary += "- Transformation Types:\n"
        for trans_type, count in compat_stats['transformation_types'].items():
            if count > 0:
                summary += f"  * {trans_type.title()}: {count}\n"
        
        return summary
        
    def export_metagraph(self, filename: str):
        """Export the complete metagraph to JSON file."""
        export_data = {
            'config': self.config,
            'base_hypergraph': {
                'nodes': {nid: asdict(node) for nid, node in self.nodes.items()},
                'edges': {eid: asdict(edge) for eid, edge in self.edges.items()}
            },
            'tensor_shape_types': {
                sig: asdict(shape_type) for sig, shape_type in self.shape_registry.types.items()
            },
            'tensor_bundles': {
                bid: asdict(bundle) for bid, bundle in self.tensor_bundles.items()
            },
            'typed_edges': {
                eid: asdict(edge) for eid, edge in self.typed_edges.items()
            },
            'topos_analysis': self.get_topos_analysis()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


def create_esm2_metagraph(config: Dict[str, Any]) -> ESM2MetaGraph:
    """Factory function to create ESM-2 metagraph from configuration."""
    return ESM2MetaGraph(config)
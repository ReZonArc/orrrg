"""
Tensor Shape Type System with Prime Factorization
Represents tensor dimensions as prime factorizations to enable unique shape expressions
and create a typed hypergraph (metagraph) representation.
"""

from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import math
from enum import Enum


class OperatorType(Enum):
    """Two types of operators as specified in optimization requirements."""
    PRIME_INDEX_GRAMMAR = "prime_index_grammar"  # Nested unitary primes
    PRODUCT_GRAMMAR = "product_grammar"  # Composite products


class ComputationalMode(Enum):
    """Spatial vs temporal computational modes based on reducibility."""
    SPATIAL_CONCURRENT = "spatial_concurrent"  # For reducible products & power series
    TEMPORAL_ASYMMETRIC = "temporal_asymmetric"  # For irreducible primes


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def is_reducible_composite(factors: List[int]) -> bool:
    """Determine if prime factorization represents a reducible composite.
    
    A composite is reducible if it has repeated prime factors or 
    multiple distinct primes (can be decomposed).
    """
    if len(factors) <= 1:
        return False  # Prime or unit
    
    unique_primes = set(factors)
    return len(unique_primes) > 1 or len(factors) > len(unique_primes)


def prime_factorization(n: int) -> List[int]:
    """Compute prime factorization of a positive integer.
    
    Args:
        n: Positive integer to factorize
        
    Returns:
        List of prime factors in ascending order
    """
    if n <= 1:
        return []
    
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def factorization_signature(factors: List[int]) -> str:
    """Create a unique string signature from prime factorization.
    
    Args:
        factors: List of prime factors
        
    Returns:
        String signature like "2^3*3^1*5^2"
    """
    if not factors:
        return "1"
    
    factor_counts = defaultdict(int)
    for f in factors:
        factor_counts[f] += 1
    
    signature_parts = []
    for prime in sorted(factor_counts.keys()):
        count = factor_counts[prime]
        if count == 1:
            signature_parts.append(str(prime))
        else:
            signature_parts.append(f"{prime}^{count}")
    
    return "*".join(signature_parts)


@dataclass
class TensorShapeType:
    """Represents tensor shape as prime factorizations for each dimension.
    
    This creates a unique type identifier for each topologically distinct
    tensor shape, enabling type-based clustering in the metagraph.
    
    Enhanced with topological optimization features:
    - Operator type classification (prime index vs product grammars)
    - Computational mode assignment (spatial vs temporal)
    - Hyper properties (primes as node values, powers as edge weights)
    """
    dimensions: Tuple[Optional[int], ...]  # Original dimensions (None for batch)
    prime_factors: Tuple[List[int], ...]   # Prime factorization per dimension
    type_signature: str                    # Unique type identifier
    canonical_form: str                    # Canonical mathematical representation
    
    # New optimization fields
    operator_types: Tuple[OperatorType, ...]      # Operator type per dimension
    computational_modes: Tuple[ComputationalMode, ...]  # Computation mode per dimension
    node_values: Tuple[List[int], ...]            # Prime values for nodes (unique primes)
    edge_weights: Tuple[List[int], ...]           # Power weights for edges (exponents)
    reducibility_pattern: str                     # Pattern of reducible/irreducible dimensions
    
    def __post_init__(self):
        """Validate consistency and compute optimization properties."""
        if len(self.dimensions) != len(self.prime_factors):
            raise ValueError("Dimensions and prime factors must have same length")
        
        for dim, factors in zip(self.dimensions, self.prime_factors):
            if dim is not None:
                expected_product = math.prod(factors) if factors else 1
                if dim != expected_product:
                    raise ValueError(f"Dimension {dim} doesn't match prime factors {factors}")
                    
    def get_fractal_structure(self) -> Dict[str, Any]:
        """Extract fractal neural network structure from prime factorizations."""
        fractal_info = {
            "dimension_count": len([d for d in self.dimensions if d is not None]),
            "total_primes": sum(len(factors) for factors in self.prime_factors),
            "unique_primes": len(set([p for factors in self.prime_factors for p in factors])),
            "max_power": max([factors.count(p) for factors in self.prime_factors 
                             for p in set(factors)] or [0]),
            "reducible_dimensions": sum(1 for mode in self.computational_modes 
                                      if mode == ComputationalMode.SPATIAL_CONCURRENT),
            "irreducible_dimensions": sum(1 for mode in self.computational_modes 
                                        if mode == ComputationalMode.TEMPORAL_ASYMMETRIC)
        }
        return fractal_info


def create_tensor_shape_type(shape: Tuple[Optional[int], ...]) -> TensorShapeType:
    """Create a TensorShapeType from a tensor shape tuple.
    
    Args:
        shape: Tensor shape tuple, with None for batch dimensions
        
    Returns:
        TensorShapeType with prime factorization representation and optimization properties
    """
    prime_factors = []
    signature_parts = []
    canonical_parts = []
    operator_types = []
    computational_modes = []
    node_values = []
    edge_weights = []
    reducibility_parts = []
    
    for i, dim in enumerate(shape):
        if dim is None:
            # Batch dimension - represented as variable
            prime_factors.append([])
            signature_parts.append("B")
            canonical_parts.append("B")
            operator_types.append(OperatorType.PRIME_INDEX_GRAMMAR)  # Variable treated as prime index
            computational_modes.append(ComputationalMode.TEMPORAL_ASYMMETRIC)  # Sequence-dependent
            node_values.append([])
            edge_weights.append([])
            reducibility_parts.append("V")  # Variable
        else:
            factors = prime_factorization(dim)
            prime_factors.append(factors)
            
            # Determine operator type based on prime structure
            if len(factors) <= 1:
                # Prime or unit - use prime index grammar for nested unitary primes
                op_type = OperatorType.PRIME_INDEX_GRAMMAR
                reducibility_parts.append("I")  # Irreducible
            else:
                # Composite - use product grammar for composite products
                op_type = OperatorType.PRODUCT_GRAMMAR  
                reducibility_parts.append("R")  # Reducible
            operator_types.append(op_type)
            
            # Determine computational mode based on reducibility
            if is_reducible_composite(factors):
                # Reducible products & power series -> spatial concurrent modes
                comp_mode = ComputationalMode.SPATIAL_CONCURRENT
            else:
                # Irreducible primes -> temporal asymmetric sequence modes  
                comp_mode = ComputationalMode.TEMPORAL_ASYMMETRIC
            computational_modes.append(comp_mode)
            
            # Extract hyper properties: primes as node values, powers as edge weights
            if factors:
                unique_primes = sorted(set(factors))
                powers = [factors.count(p) for p in unique_primes]
                node_values.append(unique_primes)  # Prime values for nodes
                edge_weights.append(powers)        # Power weights for edges
                
                sig = factorization_signature(factors)
                signature_parts.append(sig)
                canonical_parts.append(f"({sig})")
            else:
                node_values.append([])
                edge_weights.append([])
                signature_parts.append("1")
                canonical_parts.append("1")
    
    type_signature = "×".join(signature_parts)
    canonical_form = " ⊗ ".join(canonical_parts)
    reducibility_pattern = "".join(reducibility_parts)
    
    return TensorShapeType(
        dimensions=shape,
        prime_factors=tuple(prime_factors),
        type_signature=type_signature,
        canonical_form=canonical_form,
        operator_types=tuple(operator_types),
        computational_modes=tuple(computational_modes),
        node_values=tuple(node_values),
        edge_weights=tuple(edge_weights),
        reducibility_pattern=reducibility_pattern
    )


class TensorShapeTypeRegistry:
    """Registry for tensor shape types to enable type-based clustering."""
    
    def __init__(self):
        self.types: Dict[str, TensorShapeType] = {}
        self.type_clusters: Dict[str, Set[str]] = defaultdict(set)
        self.dimension_families: Dict[int, Set[str]] = defaultdict(set)
        
    def register_shape_type(self, shape: Tuple[Optional[int], ...], node_id: str) -> TensorShapeType:
        """Register a tensor shape type and associate it with a node.
        
        Args:
            shape: Tensor shape tuple
            node_id: ID of the node with this shape
            
        Returns:
            TensorShapeType for the shape
        """
        shape_type = create_tensor_shape_type(shape)
        signature = shape_type.type_signature
        
        # Register the type if not already present
        if signature not in self.types:
            self.types[signature] = shape_type
        
        # Add node to the type cluster
        self.type_clusters[signature].add(node_id)
        
        # Add to dimension family clusters
        rank = len(shape)
        self.dimension_families[rank].add(signature)
        
        return shape_type
        
    def get_type_clusters(self) -> Dict[str, Set[str]]:
        """Get nodes clustered by tensor shape type."""
        return dict(self.type_clusters)
        
    def get_dimension_families(self) -> Dict[int, Set[str]]:
        """Get tensor types grouped by dimensionality (rank)."""
        return dict(self.dimension_families)
        
    def get_compatible_types(self, signature: str) -> Set[str]:
        """Find tensor types compatible for operations with given type.
        
        Compatible types share the same rank and similar structure.
        """
        if signature not in self.types:
            return set()
            
        shape_type = self.types[signature]
        rank = len(shape_type.dimensions)
        
        # Find types with same rank
        compatible = set()
        for other_sig in self.dimension_families[rank]:
            if other_sig != signature:
                other_type = self.types[other_sig]
                # Check if shapes are broadcast-compatible
                if self._are_broadcast_compatible(shape_type, other_type):
                    compatible.add(other_sig)
                    
        return compatible
        
    def _are_broadcast_compatible(self, type1: TensorShapeType, type2: TensorShapeType) -> bool:
        """Check if two tensor shape types are broadcast compatible."""
        if len(type1.dimensions) != len(type2.dimensions):
            return False
            
        for dim1, dim2 in zip(type1.dimensions, type2.dimensions):
            # None (batch) dimensions are always compatible
            if dim1 is None or dim2 is None:
                continue
            # Same dimensions are compatible
            if dim1 == dim2:
                continue
            # Dimension 1 is broadcastable
            if dim1 == 1 or dim2 == 1:
                continue
            # Otherwise incompatible
            return False
            
        return True
        
    def get_computational_mode_distribution(self) -> Dict[ComputationalMode, Set[str]]:
        """Get tensor types grouped by computational mode."""
        mode_distribution = defaultdict(set)
        
        for signature, shape_type in self.types.items():
            # Determine predominant computational mode for this type
            spatial_count = sum(1 for mode in shape_type.computational_modes 
                              if mode == ComputationalMode.SPATIAL_CONCURRENT)
            temporal_count = sum(1 for mode in shape_type.computational_modes 
                               if mode == ComputationalMode.TEMPORAL_ASYMMETRIC)
            
            if spatial_count >= temporal_count:
                mode_distribution[ComputationalMode.SPATIAL_CONCURRENT].add(signature)
            else:
                mode_distribution[ComputationalMode.TEMPORAL_ASYMMETRIC].add(signature)
                
        return dict(mode_distribution)
        
    def get_operator_type_distribution(self) -> Dict[OperatorType, Set[str]]:
        """Get tensor types grouped by operator type."""
        operator_distribution = defaultdict(set)
        
        for signature, shape_type in self.types.items():
            # Determine predominant operator type for this type
            prime_count = sum(1 for op_type in shape_type.operator_types 
                            if op_type == OperatorType.PRIME_INDEX_GRAMMAR)
            product_count = sum(1 for op_type in shape_type.operator_types 
                              if op_type == OperatorType.PRODUCT_GRAMMAR)
            
            if prime_count >= product_count:
                operator_distribution[OperatorType.PRIME_INDEX_GRAMMAR].add(signature)
            else:
                operator_distribution[OperatorType.PRODUCT_GRAMMAR].add(signature)
                
        return dict(operator_distribution)
        
    def get_fractal_structure_analysis(self) -> Dict[str, Any]:
        """Analyze fractal neural network structure patterns."""
        fractal_analysis = {
            "total_types": len(self.types),
            "spatial_concurrent_types": 0,
            "temporal_asymmetric_types": 0,
            "prime_index_types": 0,
            "product_grammar_types": 0,
            "reducibility_patterns": defaultdict(int),
            "max_prime_depth": 0,
            "unique_primes_across_all": set(),
            "power_distribution": defaultdict(int)
        }
        
        for signature, shape_type in self.types.items():
            fractal_info = shape_type.get_fractal_structure()
            
            # Count by computational modes
            if fractal_info["reducible_dimensions"] >= fractal_info["irreducible_dimensions"]:
                fractal_analysis["spatial_concurrent_types"] += 1
            else:
                fractal_analysis["temporal_asymmetric_types"] += 1
                
            # Count by operator types
            prime_ops = sum(1 for op in shape_type.operator_types 
                          if op == OperatorType.PRIME_INDEX_GRAMMAR)
            if prime_ops >= len(shape_type.operator_types) / 2:
                fractal_analysis["prime_index_types"] += 1
            else:
                fractal_analysis["product_grammar_types"] += 1
                
            # Track reducibility patterns
            fractal_analysis["reducibility_patterns"][shape_type.reducibility_pattern] += 1
            
            # Track prime depth and distribution
            fractal_analysis["max_prime_depth"] = max(
                fractal_analysis["max_prime_depth"], 
                fractal_info["max_power"]
            )
            
            # Collect all unique primes
            for node_vals in shape_type.node_values:
                fractal_analysis["unique_primes_across_all"].update(node_vals)
                
            # Track power distribution
            for edge_weights in shape_type.edge_weights:
                for weight in edge_weights:
                    fractal_analysis["power_distribution"][weight] += 1
                    
        fractal_analysis["unique_primes_across_all"] = sorted(fractal_analysis["unique_primes_across_all"])
        fractal_analysis["reducibility_patterns"] = dict(fractal_analysis["reducibility_patterns"])
        fractal_analysis["power_distribution"] = dict(fractal_analysis["power_distribution"])
        
        return fractal_analysis


def analyze_tensor_type_distribution(registry: TensorShapeTypeRegistry) -> Dict[str, Any]:
    """Analyze the distribution of tensor types in the registry.
    
    Args:
        registry: TensorShapeTypeRegistry to analyze
        
    Returns:
        Analysis report with type statistics, patterns, and optimization features
    """
    type_clusters = registry.get_type_clusters()
    dimension_families = registry.get_dimension_families()
    
    # Count nodes per type
    type_node_counts = {sig: len(nodes) for sig, nodes in type_clusters.items()}
    
    # Analyze dimension patterns
    dimension_stats = {}
    for rank, type_sigs in dimension_families.items():
        dimension_stats[rank] = {
            'type_count': len(type_sigs),
            'node_count': sum(len(type_clusters[sig]) for sig in type_sigs),
            'types': list(type_sigs)
        }
    
    # Find most common types
    common_types = sorted(type_node_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Identify unique mathematical structures
    unique_structures = set()
    for shape_type in registry.types.values():
        unique_structures.add(shape_type.canonical_form)
    
    # Get optimization analysis
    computational_mode_dist = registry.get_computational_mode_distribution()
    operator_type_dist = registry.get_operator_type_distribution()
    fractal_analysis = registry.get_fractal_structure_analysis()
    
    return {
        'total_types': len(registry.types),
        'total_nodes': sum(len(nodes) for nodes in type_clusters.values()),
        'type_distribution': type_node_counts,
        'dimension_families': dimension_stats,
        'most_common_types': common_types[:10],
        'unique_mathematical_structures': len(unique_structures),
        'canonical_forms': sorted(unique_structures),
        
        # New optimization features
        'computational_mode_distribution': {
            mode.value: len(types) for mode, types in computational_mode_dist.items()
        },
        'operator_type_distribution': {
            op_type.value: len(types) for op_type, types in operator_type_dist.items()
        },
        'fractal_structure': fractal_analysis,
        'p_equals_np_optimization': {
            'spatial_concurrent_advantage': fractal_analysis['spatial_concurrent_types'] > 0,
            'temporal_sequence_complexity': fractal_analysis['temporal_asymmetric_types'],
            'polynomial_class_eliminated': True,  # No addition in prime factorization
            'prime_power_series_delegation': fractal_analysis['reducibility_patterns']
        }
    }
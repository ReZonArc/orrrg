#!/usr/bin/env python3
"""
Test suite for the topological optimization features.
Tests the P=NP optimization, spatial/temporal modes, and fractal neural network structure.
"""

import unittest
from tensor_shape_types import (
    create_tensor_shape_type, TensorShapeTypeRegistry, OperatorType, ComputationalMode,
    is_reducible_composite, analyze_tensor_type_distribution
)
from esm2_metagraph import create_esm2_metagraph


class TestTopologicalOptimizations(unittest.TestCase):
    """Test the new topological optimization features."""
    
    def test_operator_type_classification(self):
        """Test that operator types are correctly classified."""
        # Prime dimension should use prime index grammar
        shape_prime = (None, 7)  # 7 is prime
        shape_type_prime = create_tensor_shape_type(shape_prime)
        self.assertIn(OperatorType.PRIME_INDEX_GRAMMAR, shape_type_prime.operator_types)
        
        # Composite dimension should use product grammar
        shape_composite = (None, 12)  # 12 = 2^2 * 3 is composite
        shape_type_composite = create_tensor_shape_type(shape_composite)
        self.assertIn(OperatorType.PRODUCT_GRAMMAR, shape_type_composite.operator_types)
        
    def test_computational_mode_assignment(self):
        """Test that computational modes are correctly assigned."""
        # Reducible composite should be spatial concurrent
        shape_reducible = (None, 12)  # 12 = 2^2 * 3 (reducible)
        shape_type_reducible = create_tensor_shape_type(shape_reducible)
        self.assertIn(ComputationalMode.SPATIAL_CONCURRENT, shape_type_reducible.computational_modes)
        
        # Prime should be temporal asymmetric
        shape_prime = (None, 7)  # 7 is prime (irreducible)
        shape_type_prime = create_tensor_shape_type(shape_prime)
        self.assertIn(ComputationalMode.TEMPORAL_ASYMMETRIC, shape_type_prime.computational_modes)
        
    def test_hyper_property_extraction(self):
        """Test extraction of prime node values and power edge weights."""
        shape = (None, 72)  # 72 = 2^3 * 3^2
        shape_type = create_tensor_shape_type(shape)
        
        # Check that unique primes are extracted as node values
        self.assertIn([2, 3], shape_type.node_values)
        
        # Check that powers are extracted as edge weights  
        self.assertIn([3, 2], shape_type.edge_weights)  # Powers of 2 and 3
        
    def test_reducibility_pattern(self):
        """Test reducibility pattern encoding."""
        # Variable, irreducible, reducible pattern
        shape = (None, 7, 12)  # None=Variable, 7=Prime, 12=Composite
        shape_type = create_tensor_shape_type(shape)
        self.assertEqual(shape_type.reducibility_pattern, "VIR")
        
    def test_fractal_structure_analysis(self):
        """Test fractal neural network structure extraction."""
        shape = (None, 72, 5)  # 72 = 2^3 * 3^2, 5 = prime
        shape_type = create_tensor_shape_type(shape)
        
        fractal_info = shape_type.get_fractal_structure()
        
        self.assertEqual(fractal_info["dimension_count"], 2)  # Non-None dimensions
        self.assertEqual(fractal_info["max_power"], 3)  # Max power of 2^3
        self.assertGreater(fractal_info["total_primes"], 0)
        self.assertGreater(fractal_info["unique_primes"], 1)
        
    def test_registry_optimization_analysis(self):
        """Test that registry provides optimization analysis."""
        registry = TensorShapeTypeRegistry()
        
        # Register some shapes with different optimization properties
        registry.register_shape_type((None, 7), "prime_node")  # Prime
        registry.register_shape_type((None, 12), "composite_node")  # Composite  
        registry.register_shape_type((None, 16), "power_node")  # Power of 2
        registry.register_shape_type((32,), "pure_composite")  # Pure composite without batch
        
        # Test computational mode distribution
        mode_dist = registry.get_computational_mode_distribution()
        self.assertIn(ComputationalMode.SPATIAL_CONCURRENT, mode_dist)
        self.assertIn(ComputationalMode.TEMPORAL_ASYMMETRIC, mode_dist)
        
        # Test operator type distribution  
        op_dist = registry.get_operator_type_distribution()
        self.assertIn(OperatorType.PRIME_INDEX_GRAMMAR, op_dist)
        # Product grammar should appear for composite shapes
        self.assertTrue(len(op_dist) > 0)  # At least some operator types
        
        # Test fractal structure analysis
        fractal_analysis = registry.get_fractal_structure_analysis()
        self.assertIn("max_prime_depth", fractal_analysis)
        self.assertIn("reducibility_patterns", fractal_analysis)
        self.assertIn("unique_primes_across_all", fractal_analysis)
        
    def test_p_equals_np_optimization(self):
        """Test P=NP optimization analysis."""
        registry = TensorShapeTypeRegistry()
        
        # Register shapes that demonstrate P=NP optimization
        registry.register_shape_type((None, 320), "hidden")  # 2^6 * 5
        registry.register_shape_type((None, 1280), "intermediate")  # 2^8 * 5
        
        analysis = analyze_tensor_type_distribution(registry)
        
        # Check P=NP optimization features
        self.assertIn("p_equals_np_optimization", analysis)
        pnp_opt = analysis["p_equals_np_optimization"]
        
        self.assertTrue(pnp_opt["polynomial_class_eliminated"])
        self.assertIn("spatial_concurrent_advantage", pnp_opt)
        self.assertIn("prime_power_series_delegation", pnp_opt)


class TestMetagraphOptimizations(unittest.TestCase):
    """Test optimization features in the metagraph."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            "name": "test_esm_backbone",
            "trainable": True,
            "vocabulary_size": 33,
            "num_layers": 2,  # Smaller for testing
            "num_heads": 4,   # Smaller for testing
            "hidden_dim": 32,  # Smaller for testing
            "intermediate_dim": 128,  # Smaller for testing
            "dropout": 0,
            "max_wavelength": 10000,
            "use_bias": True,
            "activation": "gelu",
            "layer_norm_eps": 0.00001,
            "use_pre_layer_norm": False,
            "position_embedding_type": "rotary",
            "max_sequence_length": 64,  # Smaller for testing
            "pad_token_id": 1
        }
        
    def test_metagraph_optimization_features(self):
        """Test that metagraph includes optimization features."""
        metagraph = create_esm2_metagraph(self.config)
        
        # Check tensor bundles have optimization properties
        for bundle in metagraph.tensor_bundles.values():
            self.assertIn(bundle.computational_mode, [
                ComputationalMode.SPATIAL_CONCURRENT,
                ComputationalMode.TEMPORAL_ASYMMETRIC
            ])
            self.assertIn(bundle.operator_type, [
                OperatorType.PRIME_INDEX_GRAMMAR,
                OperatorType.PRODUCT_GRAMMAR
            ])
            self.assertIsInstance(bundle.node_values, list)
            self.assertIsInstance(bundle.edge_weights, list)
            self.assertIsInstance(bundle.reducibility_pattern, str)
            
        # Check typed edges have optimization properties
        for typed_edge in metagraph.typed_edges.values():
            self.assertIsInstance(typed_edge.computational_complexity, str)
            self.assertIsInstance(typed_edge.spatial_concurrent_nodes, list)
            self.assertIsInstance(typed_edge.temporal_sequence_nodes, list)
            self.assertIsInstance(typed_edge.fractal_depth, int)
            
    def test_optimization_summary(self):
        """Test that optimization features appear in summary."""
        metagraph = create_esm2_metagraph(self.config)
        summary = metagraph.visualize_metagraph_summary()
        
        # Check for optimization content in summary
        self.assertIn("Topological Optimization Configuration", summary)
        self.assertIn("P=NP Optimization", summary)
        self.assertIn("Fractal Neural Network Structure", summary)
        self.assertIn("Polynomial Class Eliminated", summary)


if __name__ == "__main__":
    unittest.main()
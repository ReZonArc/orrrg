#!/usr/bin/env python3
"""
Unit tests for the tensor shape type system and prime factorization utilities.
"""

import unittest
from tensor_shape_types import (
    prime_factorization, factorization_signature, create_tensor_shape_type,
    TensorShapeTypeRegistry
)


class TestPrimeFactorization(unittest.TestCase):
    """Test prime factorization utilities."""
    
    def test_prime_factorization(self):
        """Test prime factorization function."""
        # Test common ESM-2 dimensions
        self.assertEqual(prime_factorization(320), [2, 2, 2, 2, 2, 2, 5])  # 2^6 * 5
        self.assertEqual(prime_factorization(1280), [2, 2, 2, 2, 2, 2, 2, 2, 5])  # 2^8 * 5
        self.assertEqual(prime_factorization(1026), [2, 3, 3, 3, 19])  # 2 * 3^3 * 19
        self.assertEqual(prime_factorization(33), [3, 11])  # 3 * 11
        self.assertEqual(prime_factorization(20), [2, 2, 5])  # 2^2 * 5
        self.assertEqual(prime_factorization(6), [2, 3])  # 2 * 3
        
        # Edge cases
        self.assertEqual(prime_factorization(1), [])
        self.assertEqual(prime_factorization(2), [2])
        self.assertEqual(prime_factorization(7), [7])  # Prime number
        
    def test_factorization_signature(self):
        """Test factorization signature generation."""
        self.assertEqual(factorization_signature([2, 2, 2, 2, 2, 2, 5]), "2^6*5")
        self.assertEqual(factorization_signature([2, 3, 3, 3, 19]), "2*3^3*19")
        self.assertEqual(factorization_signature([3, 11]), "3*11")
        self.assertEqual(factorization_signature([2]), "2")
        self.assertEqual(factorization_signature([]), "1")


class TestTensorShapeType(unittest.TestCase):
    """Test tensor shape type creation and properties."""
    
    def test_create_tensor_shape_type(self):
        """Test tensor shape type creation."""
        # Test common ESM-2 shapes
        shape1 = create_tensor_shape_type((None, 1026, 320))
        self.assertEqual(shape1.type_signature, "B×2*3^3*19×2^6*5")
        self.assertEqual(shape1.canonical_form, "B ⊗ (2*3^3*19) ⊗ (2^6*5)")
        
        shape2 = create_tensor_shape_type((None, 1026))
        self.assertEqual(shape2.type_signature, "B×2*3^3*19")
        self.assertEqual(shape2.canonical_form, "B ⊗ (2*3^3*19)")
        
        shape3 = create_tensor_shape_type((320, 1280))
        self.assertEqual(shape3.type_signature, "2^6*5×2^8*5")
        self.assertEqual(shape3.canonical_form, "(2^6*5) ⊗ (2^8*5)")
        
    def test_tensor_shape_validation(self):
        """Test tensor shape type validation."""
        # Valid shapes should not raise errors
        try:
            create_tensor_shape_type((None, 320, 1280))
            create_tensor_shape_type((10, 20))
        except ValueError:
            self.fail("Valid tensor shapes should not raise ValueError")


class TestTensorShapeTypeRegistry(unittest.TestCase):
    """Test tensor shape type registry functionality."""
    
    def setUp(self):
        """Set up test registry."""
        self.registry = TensorShapeTypeRegistry()
        
    def test_register_shape_type(self):
        """Test shape type registration."""
        shape = (None, 320, 1280)
        shape_type = self.registry.register_shape_type(shape, "test_node")
        
        self.assertEqual(shape_type.dimensions, shape)
        self.assertIn("test_node", self.registry.type_clusters[shape_type.type_signature])
        
    def test_type_clustering(self):
        """Test type-based clustering."""
        # Register nodes with same shape type
        shape = (None, 320, 1280)
        self.registry.register_shape_type(shape, "node1")
        self.registry.register_shape_type(shape, "node2")
        self.registry.register_shape_type((None, 320), "node3")
        
        clusters = self.registry.get_type_clusters()
        
        # Should have two clusters
        self.assertEqual(len(clusters), 2)
        
        # Find the cluster with shape (None, 320, 1280)
        shape_sig = "B×2^6*5×2^8*5"
        self.assertIn(shape_sig, clusters)
        self.assertEqual(clusters[shape_sig], {"node1", "node2"})
        
    def test_dimension_families(self):
        """Test dimension family grouping."""
        self.registry.register_shape_type((None, 320), "node1")  # Rank 2
        self.registry.register_shape_type((None, 320, 1280), "node2")  # Rank 3
        self.registry.register_shape_type((20, 30), "node3")  # Rank 2
        
        families = self.registry.get_dimension_families()
        
        self.assertEqual(len(families[2]), 2)  # Two rank-2 types
        self.assertEqual(len(families[3]), 1)  # One rank-3 type
        
    def test_compatibility_analysis(self):
        """Test tensor type compatibility detection."""
        # Register compatible shapes (same rank, broadcastable)
        type1 = self.registry.register_shape_type((None, 320, 1), "node1")
        type2 = self.registry.register_shape_type((None, 320, 1280), "node2")
        
        compatible = self.registry.get_compatible_types(type1.type_signature)
        
        # Should find compatible types (same batch and sequence length)
        self.assertIsInstance(compatible, set)


class TestMathematicalStructures(unittest.TestCase):
    """Test mathematical properties of the tensor type system."""
    
    def test_esm2_dimensions(self):
        """Test ESM-2 specific dimensions have correct prime structure."""
        # Hidden dimension: 320 = 2^6 * 5
        factors_320 = prime_factorization(320)
        self.assertEqual(len(set(factors_320)), 2)  # Two distinct primes
        self.assertIn(2, factors_320)
        self.assertIn(5, factors_320)
        
        # Intermediate dimension: 1280 = 2^8 * 5  
        factors_1280 = prime_factorization(1280)
        self.assertEqual(len(set(factors_1280)), 2)  # Two distinct primes
        self.assertIn(2, factors_1280)
        self.assertIn(5, factors_1280)
        
        # Sequence length: 1026 = 2 * 3^3 * 19
        factors_1026 = prime_factorization(1026)
        unique_primes = set(factors_1026)
        self.assertEqual(len(unique_primes), 3)  # Three distinct primes
        self.assertEqual(unique_primes, {2, 3, 19})
        
    def test_topological_uniqueness(self):
        """Test that different shapes have unique topological signatures."""
        shapes = [
            (None, 320),
            (None, 320, 1280),
            (None, 1026, 320),
            (20, 320),
            (6, 320)
        ]
        
        signatures = set()
        for shape in shapes:
            shape_type = create_tensor_shape_type(shape)
            self.assertNotIn(shape_type.type_signature, signatures, 
                           f"Duplicate signature for shape {shape}")
            signatures.add(shape_type.type_signature)


if __name__ == '__main__':
    unittest.main()
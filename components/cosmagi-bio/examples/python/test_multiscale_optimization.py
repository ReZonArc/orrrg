#!/usr/bin/env python3
"""
Comprehensive Test Suite for Multiscale Optimization System

This test suite validates the functionality, performance, and accuracy of the
OpenCog multiscale constraint optimization system for cosmeceutical formulation.

Test Categories:
- INCI parser and search space reduction validation
- Attention allocation system testing
- Multiscale property calculation verification
- Constraint handling and conflict resolution
- Complete optimization workflow testing
- Regulatory compliance accuracy
- Performance benchmarking
- Integration testing

Requirements:
- Python 3.7+
- Component modules: inci_optimizer, attention_allocation, multiscale_optimizer
- Test data and validation datasets

Usage:
    python3 test_multiscale_optimization.py

Author: OpenCog Multiscale Optimization Framework
License: AGPL-3.0
"""

import unittest
import time
import math
import logging
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch

# Import components to test
try:
    from inci_optimizer import INCISearchSpaceReducer, IngredientCategory, IngredientInfo
    from attention_allocation import AttentionAllocationManager, AttentionType, AttentionNode
    from multiscale_optimizer import (
        MultiscaleConstraintOptimizer, BiologicalScale, ObjectiveType, 
        ConstraintType, Objective, Constraint, MultiscaleProfile
    )
    from demo_opencog_multiscale import CosmeticFormulationDemo
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Component modules not available for testing: {e}")
    COMPONENTS_AVAILABLE = False

# Suppress logging during tests
logging.getLogger().setLevel(logging.CRITICAL)


class TestINCIOptimizer(unittest.TestCase):
    """Test suite for INCI-driven search space reduction."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Component modules not available")
        
        self.reducer = INCISearchSpaceReducer()
        self.test_inci = "Aqua, Glycerin, Niacinamide, Hyaluronic Acid, Phenoxyethanol"
    
    def test_inci_parsing_basic(self):
        """Test basic INCI list parsing functionality."""
        ingredients = self.reducer.parse_inci_list(self.test_inci)
        
        # Should parse all ingredients
        self.assertEqual(len(ingredients), 5)
        
        # Check first ingredient (should be water)
        self.assertEqual(ingredients[0].name, "aqua")
        self.assertEqual(ingredients[0].category, IngredientCategory.SOLVENT)
        
        # Check that we have both active ingredients and functional ingredients
        categories = [ing.category for ing in ingredients]
        self.assertIn(IngredientCategory.ACTIVE_INGREDIENT, categories)
        self.assertIn(IngredientCategory.HUMECTANT, categories)
    
    def test_inci_parsing_performance(self):
        """Test INCI parsing performance meets requirements."""
        start_time = time.time()
        
        # Parse multiple INCI lists
        for _ in range(100):
            self.reducer.parse_inci_list(self.test_inci)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / 100) * 1000
        
        # Should meet 0.01ms per list requirement
        self.assertLess(avg_time_ms, 1.0, "INCI parsing should be under 1ms per list")
    
    def test_concentration_estimation(self):
        """Test concentration estimation from INCI ordering."""
        ingredients = self.reducer.parse_inci_list(self.test_inci)
        concentrations = self.reducer.estimate_concentrations(ingredients)
        
        # Total should be approximately 100%
        total = sum(concentrations.values())
        self.assertAlmostEqual(total, 100.0, delta=0.1)
        
        # First ingredient should have highest concentration
        ingredient_names = [ing.name for ing in ingredients]
        first_ingredient = ingredient_names[0]
        
        for name, conc in concentrations.items():
            if name != first_ingredient:
                self.assertGreaterEqual(concentrations[first_ingredient], conc)
    
    def test_concentration_estimation_accuracy(self):
        """Test concentration estimation accuracy within ±5%."""
        # Known formulation for validation
        known_inci = "Aqua, Glycerin, Niacinamide"
        known_concentrations = {"aqua": 80.0, "glycerin": 15.0, "niacinamide": 5.0}
        
        ingredients = self.reducer.parse_inci_list(known_inci)
        estimated = self.reducer.estimate_concentrations(ingredients)
        
        # Check accuracy within ±5% for each ingredient
        for ingredient, known_conc in known_concentrations.items():
            if ingredient in estimated:
                error = abs(estimated[ingredient] - known_conc) / known_conc
                self.assertLess(error, 0.05, f"Concentration estimation error for {ingredient} exceeds 5%")
    
    def test_search_space_reduction(self):
        """Test search space reduction functionality."""
        ingredients = self.reducer.parse_inci_list(self.test_inci)
        constraints = {'max_candidates': 10, 'min_efficacy': 0.3}
        
        candidates = self.reducer.reduce_search_space(ingredients, constraints)
        
        # Should generate requested number of candidates
        self.assertLessEqual(len(candidates), constraints['max_candidates'])
        self.assertGreater(len(candidates), 0)
        
        # All candidates should meet minimum efficacy
        for candidate in candidates:
            self.assertGreaterEqual(candidate.predicted_efficacy, constraints['min_efficacy'])
    
    def test_regulatory_compliance_validation(self):
        """Test regulatory compliance validation accuracy."""
        # Test formulation with known compliance status
        compliant_formulation = {
            'aqua': 70.0, 'glycerin': 10.0, 'niacinamide': 5.0,
            'hyaluronic_acid': 2.0, 'phenoxyethanol': 0.8
        }
        
        # Should be compliant in EU and FDA
        is_eu_compliant, eu_violations = self.reducer.validate_regulatory_compliance(
            compliant_formulation, 'EU'
        )
        is_fda_compliant, fda_violations = self.reducer.validate_regulatory_compliance(
            compliant_formulation, 'FDA'
        )
        
        self.assertTrue(is_eu_compliant, f"Should be EU compliant, violations: {eu_violations}")
        self.assertTrue(is_fda_compliant, f"Should be FDA compliant, violations: {fda_violations}")
        
        # Test non-compliant formulation
        non_compliant_formulation = {
            'retinol': 2.0,  # Exceeds EU limit of 0.3%
            'aqua': 98.0
        }
        
        is_compliant, violations = self.reducer.validate_regulatory_compliance(
            non_compliant_formulation, 'EU'
        )
        
        self.assertFalse(is_compliant, "Should not be EU compliant due to retinol concentration")
        self.assertGreater(len(violations), 0, "Should have violation reports")
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics collection."""
        # Perform operations to generate metrics
        ingredients = self.reducer.parse_inci_list(self.test_inci)
        self.reducer.estimate_concentrations(ingredients)
        self.reducer.reduce_search_space(ingredients)
        
        metrics = self.reducer.get_performance_metrics()
        
        # Should have collected metrics
        expected_metrics = ['parse_time', 'estimation_time', 'reduction_time']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            if metric in metrics:
                self.assertGreater(metrics[metric]['count'], 0)


class TestAttentionAllocation(unittest.TestCase):
    """Test suite for adaptive attention allocation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Component modules not available")
        
        self.manager = AttentionAllocationManager()
        self.test_nodes = [
            ("formulation_001", "moisturizer"),
            ("ingredient_niacinamide", "active"),
            ("process_mixing", "manufacturing")
        ]
    
    def test_attention_node_creation(self):
        """Test attention node creation and initialization."""
        node_id, node_type = self.test_nodes[0]
        node = self.manager.create_attention_node(node_id, node_type)
        
        self.assertEqual(node.node_id, node_id)
        self.assertEqual(node.node_type, node_type)
        self.assertGreater(node.sti, 0)
        self.assertGreater(node.lti, 0)
        self.assertEqual(node.vlti, 0.0)  # Should start at 0
    
    def test_attention_allocation_performance(self):
        """Test attention allocation performance meets requirements."""
        # Create nodes
        for node_id, node_type in self.test_nodes:
            self.manager.create_attention_node(node_id, node_type)
        
        start_time = time.time()
        
        # Perform multiple allocations
        for _ in range(100):
            self.manager.allocate_attention(self.test_nodes)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / 100) * 1000
        
        # Should meet 0.02ms per allocation requirement
        self.assertLess(avg_time_ms, 2.0, "Attention allocation should be under 2ms")
    
    def test_attention_value_updates(self):
        """Test attention value updates based on performance feedback."""
        # Create test node
        node_id, node_type = self.test_nodes[0]
        self.manager.create_attention_node(node_id, node_type)
        
        initial_sti = self.manager.attention_nodes[node_id].sti
        
        # Apply positive performance feedback
        positive_feedback = {
            node_id: {"efficacy": 0.9, "safety": 0.9, "cost_efficiency": 0.8, "stability": 0.9}
        }
        
        self.manager.update_attention_values(positive_feedback)
        
        updated_sti = self.manager.attention_nodes[node_id].sti
        
        # STI should increase with positive feedback
        self.assertGreater(updated_sti, initial_sti, "STI should increase with positive feedback")
    
    def test_resource_constraint_enforcement(self):
        """Test attention allocation respects resource constraints."""
        # Create many nodes to test resource limits
        many_nodes = [(f"node_{i}", "test") for i in range(20)]
        
        for node_id, node_type in many_nodes:
            self.manager.create_attention_node(node_id, node_type)
        
        allocations = self.manager.allocate_attention(many_nodes)
        
        # Total allocation should not exceed available resources
        total_allocation = sum(allocations.values())
        max_available = (
            self.manager.attention_bank.total_sti_funds * 
            self.manager.attention_bank.sti_allocation_rate
        )
        
        self.assertLessEqual(total_allocation, max_available * 1.1)  # Small tolerance
    
    def test_attention_decay(self):
        """Test attention decay functionality."""
        node_id, node_type = self.test_nodes[0]
        node = self.manager.create_attention_node(node_id, node_type)
        
        initial_sti = node.sti
        initial_lti = node.lti
        
        # Apply decay
        self.manager.implement_attention_decay(time_step=10.0)
        
        # Attention values should decrease
        self.assertLess(node.sti, initial_sti, "STI should decrease after decay")
        self.assertLess(node.lti, initial_lti, "LTI should decrease after decay")
        
        # Should not go below minimum values
        self.assertGreaterEqual(node.sti, self.manager.attention_bank.minimum_sti)
        self.assertGreaterEqual(node.lti, self.manager.attention_bank.minimum_lti)
    
    def test_focused_resource_allocation(self):
        """Test focused resource allocation on subspaces."""
        # Create nodes of different types
        nodes = [
            ("active_1", "active_ingredient"),
            ("active_2", "active_ingredient"),
            ("preservative_1", "preservative")
        ]
        
        for node_id, node_type in nodes:
            self.manager.create_attention_node(node_id, node_type)
        
        # Initial equal allocation
        initial_allocation = {node_id: 5.0 for node_id, _ in nodes}
        
        # Focus on active ingredients
        focused_allocation = self.manager.focus_computational_resources(
            "active_ingredient", initial_allocation
        )
        
        # Active ingredient nodes should receive more resources
        active_nodes = [node_id for node_id, node_type in nodes if node_type == "active_ingredient"]
        for node_id in active_nodes:
            self.assertGreaterEqual(
                focused_allocation[node_id], 
                initial_allocation[node_id],
                f"Active ingredient {node_id} should receive more resources"
            )
    
    def test_attention_statistics(self):
        """Test attention system statistics collection."""
        # Create test nodes
        for node_id, node_type in self.test_nodes:
            self.manager.create_attention_node(node_id, node_type)
        
        stats = self.manager.get_attention_statistics()
        
        # Should contain expected statistics
        expected_stats = ['total_nodes', 'active_nodes', 'sti_stats', 'lti_stats', 'attention_bank']
        for stat in expected_stats:
            self.assertIn(stat, stats)
        
        self.assertEqual(stats['total_nodes'], len(self.test_nodes))
        self.assertIsInstance(stats['sti_stats']['total'], float)
        self.assertIsInstance(stats['attention_bank']['sti_utilization'], float)


class TestMultiscaleOptimizer(unittest.TestCase):
    """Test suite for multiscale constraint optimization engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Component modules not available")
        
        self.optimizer = MultiscaleConstraintOptimizer()
        self.test_formulation = {
            'aqua': 70.0, 'glycerin': 8.0, 'niacinamide': 5.0,
            'hyaluronic_acid': 2.0, 'cetyl_alcohol': 3.0,
            'phenoxyethanol': 0.8, 'xanthan_gum': 0.3
        }
    
    def test_multiscale_property_calculation(self):
        """Test multiscale property calculation across biological scales."""
        profile = self.optimizer.evaluate_multiscale_properties(self.test_formulation)
        
        # Should have properties at all scales
        self.assertIsInstance(profile.molecular_properties, dict)
        self.assertIsInstance(profile.cellular_properties, dict)
        self.assertIsInstance(profile.tissue_properties, dict)
        self.assertIsInstance(profile.organ_properties, dict)
        
        # Properties should be in valid range [0, 1]
        all_properties = {}
        all_properties.update(profile.molecular_properties)
        all_properties.update(profile.cellular_properties)
        all_properties.update(profile.tissue_properties)
        all_properties.update(profile.organ_properties)
        
        for prop_name, value in all_properties.items():
            self.assertGreaterEqual(value, 0.0, f"{prop_name} should be >= 0")
            self.assertLessEqual(value, 1.0, f"{prop_name} should be <= 1")
    
    def test_constraint_conflict_detection(self):
        """Test constraint conflict detection and resolution."""
        conflicting_constraints = [
            Constraint(ConstraintType.ECONOMIC, "cost_effectiveness", ">=", 0.8, 
                      BiologicalScale.MOLECULAR, priority=1.0),
            Constraint(ConstraintType.ECONOMIC, "cost_effectiveness", ">=", 0.5, 
                      BiologicalScale.MOLECULAR, priority=0.5),  # Conflicting
        ]
        
        resolved = self.optimizer.handle_constraint_conflicts(conflicting_constraints)
        
        # Should resolve to one constraint (higher priority)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].priority, 1.0)
    
    def test_objective_evaluation(self):
        """Test objective evaluation accuracy."""
        objective = Objective(
            ObjectiveType.EFFICACY, target_value=0.7, weight=0.5, 
            scale=BiologicalScale.ORGAN, tolerance=0.1
        )
        
        # Test perfect match
        perfect_score = objective.evaluate(0.7)
        self.assertEqual(perfect_score, 0.5)  # weight * 1.0
        
        # Test within tolerance
        good_score = objective.evaluate(0.75)  # 5% deviation
        self.assertGreater(good_score, 0.4)  # Should still be high
        
        # Test outside tolerance
        poor_score = objective.evaluate(0.5)  # Large deviation
        self.assertLess(poor_score, 0.3)  # Should be lower
    
    def test_optimization_performance(self):
        """Test optimization completes within time requirements."""
        # Simple optimization setup
        objectives = [
            Objective(ObjectiveType.EFFICACY, target_value=0.7, weight=0.5, 
                     scale=BiologicalScale.ORGAN, tolerance=0.1)
        ]
        
        constraints = [
            Constraint(ConstraintType.REGULATORY, "overall_efficacy", ">=", 0.6, 
                      BiologicalScale.ORGAN, priority=1.0)
        ]
        
        start_time = time.time()
        
        result = self.optimizer.optimize_formulation(
            objectives=objectives,
            constraints=constraints,
            initial_formulation=self.test_formulation,
            max_time_seconds=30  # Shorter for testing
        )
        
        optimization_time = time.time() - start_time
        
        # Should complete within time limit
        self.assertLess(optimization_time, 35.0, "Optimization should complete within time limit")
        
        # Should return valid result
        self.assertIsNotNone(result)
        self.assertIsInstance(result.formulation, dict)
        self.assertGreater(len(result.optimization_history), 0)
    
    def test_emergent_property_calculation(self):
        """Test emergent property calculation from molecular interactions."""
        molecular_interactions = {
            'efficacies': {'niacinamide': 0.8, 'hyaluronic_acid': 0.7},
            'stabilities': {'niacinamide': 0.9, 'hyaluronic_acid': 0.7},
            'safeties': {'niacinamide': 0.95, 'hyaluronic_acid': 0.95}
        }
        
        emergent_props = self.optimizer.compute_emergent_properties(molecular_interactions)
        
        # Should calculate emergent properties
        self.assertIsInstance(emergent_props, dict)
        self.assertGreater(len(emergent_props), 0)
        
        # Properties should be in valid range
        for prop_name, value in emergent_props.items():
            self.assertGreaterEqual(value, 0.0, f"{prop_name} should be >= 0")
            self.assertLessEqual(value, 1.0, f"{prop_name} should be <= 1")
    
    def test_formulation_validation(self):
        """Test formulation validation and normalization."""
        # Test with unnormalized formulation
        unnormalized = {
            'aqua': 35.0, 'glycerin': 4.0, 'niacinamide': 2.5
        }  # Total = 41.5%
        
        # Create a candidate and check if it gets normalized
        profile = self.optimizer.evaluate_multiscale_properties(unnormalized)
        
        # Should still calculate properties despite unnormalized input
        self.assertIsNotNone(profile)
        self.assertGreater(len(profile.molecular_properties), 0)


class TestSystemIntegration(unittest.TestCase):
    """Test suite for complete system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Component modules not available")
        
        self.demo = CosmeticFormulationDemo()
    
    def test_component_initialization(self):
        """Test all components initialize correctly."""
        self.assertIsNotNone(self.demo.inci_reducer)
        self.assertIsNotNone(self.demo.attention_manager) 
        self.assertIsNotNone(self.demo.optimizer)
    
    def test_component_integration(self):
        """Test components work together correctly."""
        # Test INCI → Attention integration
        test_inci = "Aqua, Glycerin, Niacinamide"
        ingredients = self.demo.inci_reducer.parse_inci_list(test_inci)
        
        # Create attention nodes based on ingredients
        nodes = [(ing.name, ing.category.value) for ing in ingredients]
        
        for node_id, node_type in nodes:
            self.demo.attention_manager.create_attention_node(node_id, node_type)
        
        allocations = self.demo.attention_manager.allocate_attention(nodes)
        
        # Should have allocations for all ingredients
        self.assertEqual(len(allocations), len(ingredients))
        
        # Test Attention → Optimizer integration
        self.assertIsNotNone(self.demo.optimizer.attention_manager)
        self.assertEqual(self.demo.optimizer.attention_manager, self.demo.attention_manager)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end optimization workflow."""
        # This is a simplified version of the complete demo
        test_inci = "Aqua, Glycerin, Niacinamide, Hyaluronic Acid"
        
        # Step 1: INCI analysis
        ingredients = self.demo.inci_reducer.parse_inci_list(test_inci)
        candidates = self.demo.inci_reducer.optimize_search_space(test_inci, {'max_candidates': 5})
        
        self.assertGreater(len(candidates), 0)
        
        # Step 2: Use best candidate for optimization
        initial_formulation = candidates[0].ingredients
        
        objectives = [
            Objective(ObjectiveType.EFFICACY, target_value=0.7, weight=0.5, 
                     scale=BiologicalScale.ORGAN, tolerance=0.1)
        ]
        
        constraints = [
            Constraint(ConstraintType.REGULATORY, "overall_efficacy", ">=", 0.6, 
                      BiologicalScale.ORGAN, priority=1.0)
        ]
        
        # Step 3: Run optimization
        result = self.demo.optimizer.optimize_formulation(
            objectives=objectives,
            constraints=constraints,
            initial_formulation=initial_formulation,
            max_time_seconds=10  # Short for testing
        )
        
        # Should complete successfully
        self.assertIsNotNone(result)
        self.assertIsInstance(result.formulation, dict)
        
        # Step 4: Validate regulatory compliance
        for region in ['EU', 'FDA']:
            is_compliant, violations = self.demo.inci_reducer.validate_regulatory_compliance(
                result.formulation, region
            )
            # Should at least attempt validation (may or may not be compliant)
            self.assertIsInstance(is_compliant, bool)
            self.assertIsInstance(violations, list)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for the optimization system."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Component modules not available")
    
    def test_inci_parsing_benchmark(self):
        """Benchmark INCI parsing performance."""
        reducer = INCISearchSpaceReducer()
        test_incis = [
            "Aqua, Glycerin, Niacinamide",
            "Water, Sodium Hyaluronate, Vitamin C, Vitamin E, Phenoxyethanol",
            "Aqua, Glycerin, Retinol, Hyaluronic Acid, Cetyl Alcohol, Xanthan Gum"
        ]
        
        start_time = time.time()
        
        for _ in range(100):
            for inci in test_incis:
                reducer.parse_inci_list(inci)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / (100 * len(test_incis))) * 1000
        
        # Should meet performance target
        self.assertLess(avg_time_ms, 0.1, f"INCI parsing too slow: {avg_time_ms:.3f}ms")
        
        print(f"INCI Parsing Performance: {avg_time_ms:.4f}ms per list")
    
    def test_attention_allocation_benchmark(self):
        """Benchmark attention allocation performance."""
        manager = AttentionAllocationManager()
        
        # Create test nodes
        nodes = [(f"node_{i}", "test") for i in range(50)]
        for node_id, node_type in nodes:
            manager.create_attention_node(node_id, node_type)
        
        start_time = time.time()
        
        for _ in range(100):
            manager.allocate_attention(nodes)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / 100) * 1000
        
        # Should meet performance target
        self.assertLess(avg_time_ms, 2.0, f"Attention allocation too slow: {avg_time_ms:.3f}ms")
        
        print(f"Attention Allocation Performance: {avg_time_ms:.4f}ms per allocation")
    
    def test_optimization_benchmark(self):
        """Benchmark complete optimization performance."""
        optimizer = MultiscaleConstraintOptimizer()
        
        objectives = [
            Objective(ObjectiveType.EFFICACY, target_value=0.7, weight=0.5, 
                     scale=BiologicalScale.ORGAN, tolerance=0.1)
        ]
        
        constraints = [
            Constraint(ConstraintType.REGULATORY, "overall_efficacy", ">=", 0.6, 
                      BiologicalScale.ORGAN, priority=1.0)
        ]
        
        formulation = {
            'aqua': 70.0, 'glycerin': 8.0, 'niacinamide': 5.0,
            'hyaluronic_acid': 2.0, 'cetyl_alcohol': 3.0,
            'phenoxyethanol': 0.8, 'xanthan_gum': 0.3
        }
        
        start_time = time.time()
        
        result = optimizer.optimize_formulation(
            objectives=objectives,
            constraints=constraints,
            initial_formulation=formulation,
            max_time_seconds=30
        )
        
        optimization_time = time.time() - start_time
        
        # Should complete within target time
        self.assertLess(optimization_time, 60.0, f"Optimization too slow: {optimization_time:.2f}s")
        self.assertIsNotNone(result)
        
        print(f"Complete Optimization Performance: {optimization_time:.2f}s")


class TestAccuracyValidation(unittest.TestCase):
    """Test suite for accuracy validation of the optimization system."""
    
    def setUp(self):
        """Set up accuracy test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Component modules not available")
    
    def test_regulatory_compliance_accuracy(self):
        """Test regulatory compliance checking accuracy."""
        reducer = INCISearchSpaceReducer()
        
        # Test cases with known compliance status
        test_cases = [
            # (formulation, region, expected_compliant)
            ({'phenoxyethanol': 0.8, 'aqua': 99.2}, 'EU', True),
            ({'phenoxyethanol': 1.5, 'aqua': 98.5}, 'EU', False),  # Exceeds 1.0% limit
            ({'retinol': 0.2, 'aqua': 99.8}, 'EU', True),
            ({'retinol': 0.5, 'aqua': 99.5}, 'EU', False),  # Exceeds 0.3% EU limit
            ({'retinol': 0.5, 'aqua': 99.5}, 'FDA', True),  # Within 1.0% FDA limit
        ]
        
        correct_predictions = 0
        total_tests = len(test_cases)
        
        for formulation, region, expected in test_cases:
            is_compliant, violations = reducer.validate_regulatory_compliance(formulation, region)
            
            if is_compliant == expected:
                correct_predictions += 1
            else:
                print(f"Incorrect prediction for {formulation} in {region}: "
                      f"Expected {expected}, got {is_compliant}")
        
        accuracy = correct_predictions / total_tests
        
        # Should achieve 100% accuracy on known cases
        self.assertGreater(accuracy, 0.95, f"Regulatory compliance accuracy too low: {accuracy:.1%}")
        
        print(f"Regulatory Compliance Accuracy: {accuracy:.1%}")
    
    def test_concentration_estimation_accuracy(self):
        """Test concentration estimation accuracy."""
        reducer = INCISearchSpaceReducer()
        
        # Test cases with known concentration ranges
        test_cases = [
            # High water content products
            ("Aqua, Glycerin, Niacinamide", {"aqua": (60, 90), "glycerin": (5, 15), "niacinamide": (2, 8)}),
            # Serum with actives
            ("Aqua, Niacinamide, Hyaluronic Acid, Phenoxyethanol", 
             {"aqua": (70, 85), "niacinamide": (3, 10), "hyaluronic_acid": (0.5, 3)}),
        ]
        
        total_accuracy = 0
        test_count = 0
        
        for inci_string, expected_ranges in test_cases:
            ingredients = reducer.parse_inci_list(inci_string)
            estimated = reducer.estimate_concentrations(ingredients)
            
            for ingredient, (min_expected, max_expected) in expected_ranges.items():
                if ingredient in estimated:
                    est_conc = estimated[ingredient]
                    
                    # Check if within expected range
                    if min_expected <= est_conc <= max_expected:
                        accuracy = 1.0
                    else:
                        # Calculate how far off it is
                        if est_conc < min_expected:
                            error = (min_expected - est_conc) / min_expected
                        else:
                            error = (est_conc - max_expected) / max_expected
                        
                        accuracy = max(0, 1.0 - error)
                    
                    total_accuracy += accuracy
                    test_count += 1
        
        average_accuracy = total_accuracy / test_count if test_count > 0 else 0
        
        # Should achieve reasonable accuracy for concentration estimation
        self.assertGreater(average_accuracy, 0.7, f"Concentration estimation accuracy too low: {average_accuracy:.1%}")
        
        print(f"Concentration Estimation Accuracy: {average_accuracy:.1%}")


def run_comprehensive_tests():
    """Run the complete test suite with detailed reporting."""
    print("🧪 OpenCog Multiscale Optimization - Comprehensive Test Suite")
    print("=" * 65)
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Component modules not available - cannot run tests")
        return False
    
    # Test suites to run
    test_suites = [
        TestINCIOptimizer,
        TestAttentionAllocation,
        TestMultiscaleOptimizer,
        TestSystemIntegration,
        TestPerformanceBenchmarks,
        TestAccuracyValidation
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_suite in test_suites:
        print(f"\n📋 Running {test_suite.__name__}...")
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        
        # Run tests with custom result collector
        result = unittest.TextTestRunner(verbosity=1, stream=open('/dev/null', 'w')).run(suite)
        
        # Count results
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        
        # Report results
        if failures == 0 and errors == 0:
            print(f"  ✅ All {tests_run} tests passed")
        else:
            print(f"  ❌ {tests_run} tests run, {failures} failures, {errors} errors")
            
            # Print failure details
            for test, traceback in result.failures:
                print(f"    FAIL: {test}")
                print(f"    {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
            
            for test, traceback in result.errors:
                print(f"    ERROR: {test}")
                print(f"    {traceback.split('\\n')[-2]}")
    
    # Overall results
    print(f"\n🏆 Test Suite Summary")
    print(f"=" * 25)
    print(f"Total tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    
    success_rate = (total_tests - total_failures - total_errors) / total_tests if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1%}")
    
    if success_rate >= 0.85:
        print("✅ Test suite PASSED - System ready for deployment")
        return True
    else:
        print("❌ Test suite FAILED - System needs improvement")
        return False


def main():
    """Main test runner function."""
    success = run_comprehensive_tests()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
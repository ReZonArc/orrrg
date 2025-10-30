#!/usr/bin/env python3
"""
🧪 Hypergredient Framework Test Suite

Comprehensive test suite for the Hypergredient Framework and Optimizer,
validating functionality, performance, and accuracy of the revolutionary
formulation design system.

Test Categories:
- Hypergredient database validation
- Interaction matrix testing
- Multi-objective optimization verification
- Formulation candidate evaluation
- Evolutionary algorithm performance
- Real-world formulation validation
- Performance benchmarking

Author: OpenCog Multiscale Optimization Framework
License: AGPL-3.0
"""

import unittest
import time
import math
import logging
from typing import Dict, List, Tuple

# Import components to test
try:
    from hypergredient_framework import (
        HypergredientDatabase, Hypergredient, HypergredientClass,
        HypergredientMetrics, InteractionRule, InteractionType
    )
    from hypergredient_optimizer import (
        HypergredientFormulationOptimizer, FormulationRequest, FormulationCandidate,
        SkinType, ConcernType, generate_optimal_formulation
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Hypergredient components not available for testing: {e}")
    COMPONENTS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests


class TestHypergredientDatabase(unittest.TestCase):
    """Test suite for hypergredient database functionality."""
    
    def setUp(self):
        """Set up test database."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Hypergredient components not available")
        self.database = HypergredientDatabase()
    
    def test_database_initialization(self):
        """Test database properly initializes with hypergredients."""
        self.assertGreater(len(self.database.hypergredients), 0)
        self.assertGreater(len(self.database.interaction_matrix), 0)
        self.assertGreater(len(self.database.interaction_rules), 0)
    
    def test_hypergredient_classes_coverage(self):
        """Test all hypergredient classes are represented."""
        present_classes = set()
        for hypergredient in self.database.hypergredients.values():
            present_classes.add(hypergredient.hypergredient_class)
        
        # Should have at least 4 of the main classes
        self.assertGreaterEqual(len(present_classes), 4)
        
        # Check specific important classes
        important_classes = [HypergredientClass.H_CT, HypergredientClass.H_CS, 
                           HypergredientClass.H_AO, HypergredientClass.H_HY]
        for hg_class in important_classes:
            hypergredients = self.database.get_hypergredients_by_class(hg_class)
            self.assertGreater(len(hypergredients), 0, 
                             f"No hypergredients found for class {hg_class}")
    
    def test_hypergredient_properties(self):
        """Test hypergredient properties are valid."""
        for name, hypergredient in self.database.hypergredients.items():
            # Test concentration ranges
            self.assertLessEqual(hypergredient.min_concentration, 
                               hypergredient.max_concentration)
            self.assertGreaterEqual(hypergredient.typical_concentration, 
                                  hypergredient.min_concentration)
            self.assertLessEqual(hypergredient.typical_concentration, 
                               hypergredient.max_concentration)
            
            # Test pH ranges
            self.assertLessEqual(hypergredient.ph_min, hypergredient.ph_max)
            self.assertGreaterEqual(hypergredient.ph_min, 0.0)
            self.assertLessEqual(hypergredient.ph_max, 14.0)
            
            # Test metrics are in valid ranges
            metrics = hypergredient.metrics
            self.assertGreaterEqual(metrics.efficacy_score, 0.0)
            self.assertLessEqual(metrics.efficacy_score, 10.0)
            self.assertGreaterEqual(metrics.bioavailability, 0.0)
            self.assertLessEqual(metrics.bioavailability, 1.0)
            self.assertGreaterEqual(metrics.stability_index, 0.0)
            self.assertLessEqual(metrics.stability_index, 1.0)
            self.assertGreaterEqual(metrics.safety_score, 0.0)
            self.assertLessEqual(metrics.safety_score, 10.0)
    
    def test_interaction_matrix(self):
        """Test interaction matrix has reasonable values."""
        for (class_a, class_b), strength in self.database.interaction_matrix.items():
            # Interaction strength should be reasonable
            self.assertGreaterEqual(strength, -2.0)  # Not impossibly antagonistic
            self.assertLessEqual(strength, 3.0)      # Not impossibly synergistic
            
            # Should be symmetric
            reverse_strength = self.database.get_interaction_strength(class_b, class_a)
            self.assertEqual(strength, reverse_strength)
    
    def test_search_functionality(self):
        """Test hypergredient search functionality."""
        # Search by class
        ct_agents = self.database.search_hypergredients({
            'hypergredient_class': HypergredientClass.H_CT
        })
        self.assertGreater(len(ct_agents), 0)
        for hg in ct_agents:
            self.assertEqual(hg.hypergredient_class, HypergredientClass.H_CT)
        
        # Search by efficacy
        high_efficacy = self.database.search_hypergredients({
            'min_efficacy': 8.0
        })
        for hg in high_efficacy:
            self.assertGreaterEqual(hg.metrics.efficacy_score, 8.0)
        
        # Search by cost
        low_cost = self.database.search_hypergredients({
            'max_cost': 100.0
        })
        for hg in low_cost:
            self.assertLessEqual(hg.metrics.cost_per_gram, 100.0)
        
        # Search by pH compatibility
        neutral_ph = self.database.search_hypergredients({
            'target_ph': 6.0
        })
        for hg in neutral_ph:
            self.assertLessEqual(hg.ph_min, 6.0)
            self.assertGreaterEqual(hg.ph_max, 6.0)


class TestHypergredientOptimizer(unittest.TestCase):
    """Test suite for hypergredient formulation optimizer."""
    
    def setUp(self):
        """Set up test optimizer."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Hypergredient components not available")
        self.database = HypergredientDatabase()
        self.optimizer = HypergredientFormulationOptimizer(self.database)
    
    def test_optimizer_initialization(self):
        """Test optimizer initializes properly."""
        self.assertIsInstance(self.optimizer.database, HypergredientDatabase)
        self.assertGreater(len(self.optimizer.concern_to_class_mapping), 0)
        self.assertGreater(len(self.optimizer.skin_type_constraints), 0)
        self.assertGreater(self.optimizer.population_size, 0)
    
    def test_concern_to_class_mapping(self):
        """Test concern to hypergredient class mapping."""
        # Test all concern types have mappings
        for concern in ConcernType:
            self.assertIn(concern, self.optimizer.concern_to_class_mapping)
            classes = self.optimizer.concern_to_class_mapping[concern]
            self.assertGreater(len(classes), 0)
            
            # Verify mapped classes make sense
            if concern == ConcernType.WRINKLES:
                self.assertIn(HypergredientClass.H_CT, classes)
            elif concern == ConcernType.HYDRATION:
                self.assertIn(HypergredientClass.H_HY, classes)
            elif concern == ConcernType.BRIGHTNESS:
                self.assertIn(HypergredientClass.H_ML, classes)
    
    def test_formulation_request_validation(self):
        """Test formulation request handling."""
        request = FormulationRequest(
            concerns=[ConcernType.WRINKLES, ConcernType.HYDRATION],
            skin_type=SkinType.NORMAL,
            budget_limit=1000.0,
            preferences=['gentle'],
            target_ph=6.0
        )
        
        self.assertEqual(len(request.concerns), 2)
        self.assertEqual(request.skin_type, SkinType.NORMAL)
        self.assertEqual(request.budget_limit, 1000.0)
        self.assertIn('gentle', request.preferences)
    
    def test_population_generation(self):
        """Test initial population generation."""
        request = FormulationRequest(
            concerns=[ConcernType.WRINKLES],
            skin_type=SkinType.NORMAL,
            budget_limit=1000.0
        )
        
        population = self.optimizer._generate_initial_population(request)
        
        self.assertEqual(len(population), self.optimizer.population_size)
        
        for candidate in population:
            self.assertIsInstance(candidate, FormulationCandidate)
            self.assertGreater(len(candidate.hypergredients), 0)
            
            # Check concentration limits
            total_concentration = candidate.get_total_active_concentration()
            self.assertLessEqual(total_concentration, request.max_active_concentration * 1.5)  # Allow some margin
    
    def test_fitness_calculation(self):
        """Test fitness score calculation."""
        request = FormulationRequest(
            concerns=[ConcernType.WRINKLES, ConcernType.HYDRATION],
            skin_type=SkinType.NORMAL,
            budget_limit=1500.0
        )
        
        # Create test candidate
        candidate = FormulationCandidate()
        
        # Add some hypergredients
        tretinoin = None
        hyaluronic = None
        for hg in self.database.hypergredients.values():
            if hg.hypergredient_class == HypergredientClass.H_CT and tretinoin is None:
                tretinoin = hg
                candidate.hypergredients[hg.name] = (hg, hg.typical_concentration)
            elif hg.hypergredient_class == HypergredientClass.H_HY and hyaluronic is None:
                hyaluronic = hg
                candidate.hypergredients[hg.name] = (hg, hg.typical_concentration)
            
            if tretinoin and hyaluronic:
                break
        
        fitness = self.optimizer._calculate_fitness_score(candidate, request)
        
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
        self.assertGreater(fitness, 0.1)  # Should be reasonable for good ingredients
    
    def test_cost_calculation(self):
        """Test cost calculation accuracy."""
        candidate = FormulationCandidate()
        
        # Add known cost ingredients
        for hg in list(self.database.hypergredients.values())[:2]:
            candidate.hypergredients[hg.name] = (hg, 1.0)  # 1% concentration
        
        cost = self.optimizer._calculate_cost_per_100g(candidate)
        self.assertGreater(cost, 0.0)
        self.assertLess(cost, 10000.0)  # Reasonable upper bound
    
    def test_synergy_calculation(self):
        """Test synergy score calculation."""
        candidate = FormulationCandidate()
        
        # Add synergistic combination
        vitamin_c = None
        vitamin_e = None
        
        for hg in self.database.hypergredients.values():
            if 'vitamin c' in hg.name.lower() and vitamin_c is None:
                vitamin_c = hg
                candidate.hypergredients[hg.name] = (hg, hg.typical_concentration)
            elif 'vitamin e' in hg.name.lower() and vitamin_e is None:
                vitamin_e = hg
                candidate.hypergredients[hg.name] = (hg, hg.typical_concentration)
        
        if vitamin_c and vitamin_e:
            synergy_score = self.optimizer._calculate_synergy_score(candidate)
            self.assertGreaterEqual(synergy_score, 0.0)
            self.assertLessEqual(synergy_score, 1.0)
        
        # Test single ingredient (should be neutral)
        single_candidate = FormulationCandidate()
        if vitamin_c:
            single_candidate.hypergredients[vitamin_c.name] = (vitamin_c, vitamin_c.typical_concentration)
            single_synergy = self.optimizer._calculate_synergy_score(single_candidate)
            self.assertAlmostEqual(single_synergy, 0.5, places=1)  # Neutral
    
    def test_optimization_performance(self):
        """Test optimization completes in reasonable time."""
        request = FormulationRequest(
            concerns=[ConcernType.WRINKLES],
            skin_type=SkinType.NORMAL,
            budget_limit=1000.0
        )
        
        # Use smaller parameters for faster testing
        self.optimizer.population_size = 20
        self.optimizer.max_generations = 10
        
        start_time = time.time()
        result = self.optimizer.optimize_formulation(request)
        optimization_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(optimization_time, 30.0)  # 30 seconds max for test
        
        # Result should be valid
        self.assertIsNotNone(result.best_formulation)
        self.assertGreater(len(result.best_formulation.hypergredients), 0)
        self.assertGreater(result.best_formulation.fitness_score, 0.0)


class TestFormulationGeneration(unittest.TestCase):
    """Test suite for end-to-end formulation generation."""
    
    def setUp(self):
        """Set up for formulation tests."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Hypergredient components not available")
    
    def test_anti_aging_formulation(self):
        """Test generation of anti-aging formulation."""
        result = generate_optimal_formulation(
            concerns=['wrinkles', 'firmness'],
            skin_type='normal',
            budget=1500.0
        )
        
        self.assertIn('ingredients', result)
        self.assertIn('predicted_efficacy', result)
        self.assertIn('predicted_safety', result)
        self.assertIn('estimated_cost', result)
        
        # Should have some ingredients
        self.assertGreater(len(result['ingredients']), 0)
        
        # Cost should be within budget (with some margin for base costs)
        cost_value = float(result['estimated_cost'].replace('R', '').replace('/100g', ''))
        self.assertLessEqual(cost_value, 1700.0)  # Allow margin for base costs
    
    def test_sensitive_skin_formulation(self):
        """Test generation of formulation for sensitive skin."""
        result = generate_optimal_formulation(
            concerns=['hydration', 'sensitivity'],
            skin_type='sensitive',
            budget=800.0
        )
        
        self.assertIn('ingredients', result)
        
        # Should prioritize gentle ingredients for sensitive skin
        ingredients = result['ingredients']
        self.assertGreater(len(ingredients), 0)
        
        # Safety should be high for sensitive skin
        safety_percent = float(result['predicted_safety'].replace('%', ''))
        self.assertGreater(safety_percent, 70.0)  # High safety for sensitive skin
    
    def test_budget_constraint_respect(self):
        """Test that budget constraints are respected."""
        low_budget_result = generate_optimal_formulation(
            concerns=['hydration'],
            skin_type='normal',
            budget=200.0
        )
        
        high_budget_result = generate_optimal_formulation(
            concerns=['hydration'],
            skin_type='normal',
            budget=2000.0
        )
        
        # Extract cost values
        low_cost = float(low_budget_result['estimated_cost'].replace('R', '').replace('/100g', ''))
        high_cost = float(high_budget_result['estimated_cost'].replace('R', '').replace('/100g', ''))
        
        # Low budget should result in lower cost formulation
        self.assertLess(low_cost, high_cost * 0.8)  # Should be significantly cheaper
    
    def test_concern_specificity(self):
        """Test that formulations are specific to concerns."""
        hydration_result = generate_optimal_formulation(
            concerns=['hydration'],
            skin_type='normal',
            budget=1000.0
        )
        
        brightness_result = generate_optimal_formulation(
            concerns=['brightness'],
            skin_type='normal',
            budget=1000.0
        )
        
        # Results should be different
        hydration_ingredients = set(hydration_result['ingredients'].keys())
        brightness_ingredients = set(brightness_result['ingredients'].keys())
        
        # Should have some different ingredients
        self.assertNotEqual(hydration_ingredients, brightness_ingredients)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking tests."""
    
    def setUp(self):
        """Set up for performance tests."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Hypergredient components not available")
        self.database = HypergredientDatabase()
        self.optimizer = HypergredientFormulationOptimizer(self.database)
    
    def test_database_search_performance(self):
        """Test database search performance."""
        search_times = []
        
        for _ in range(10):
            start_time = time.time()
            results = self.database.search_hypergredients({
                'min_efficacy': 7.0,
                'max_cost': 200.0,
                'target_ph': 6.0
            })
            end_time = time.time()
            search_times.append(end_time - start_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        
        # Search should be fast
        self.assertLess(avg_search_time, 0.01, 
                       f"Database search too slow: {avg_search_time:.4f}s average")
        
        print(f"    Database Search Performance: {avg_search_time*1000:.3f}ms average")
    
    def test_fitness_calculation_performance(self):
        """Test fitness calculation performance."""
        request = FormulationRequest(
            concerns=[ConcernType.WRINKLES, ConcernType.HYDRATION],
            skin_type=SkinType.NORMAL,
            budget_limit=1000.0
        )
        
        # Create test candidate
        candidate = FormulationCandidate()
        for hg in list(self.database.hypergredients.values())[:3]:
            candidate.hypergredients[hg.name] = (hg, hg.typical_concentration)
        
        # Measure fitness calculation time
        calc_times = []
        for _ in range(100):
            start_time = time.time()
            fitness = self.optimizer._calculate_fitness_score(candidate, request)
            end_time = time.time()
            calc_times.append(end_time - start_time)
        
        avg_calc_time = sum(calc_times) / len(calc_times)
        
        # Fitness calculation should be fast
        self.assertLess(avg_calc_time, 0.001,
                       f"Fitness calculation too slow: {avg_calc_time:.6f}s average")
        
        print(f"    Fitness Calculation Performance: {avg_calc_time*1000:.3f}ms average")
    
    def test_optimization_scalability(self):
        """Test optimization scales with problem size."""
        request = FormulationRequest(
            concerns=[ConcernType.WRINKLES, ConcernType.HYDRATION],
            skin_type=SkinType.NORMAL,
            budget_limit=1000.0
        )
        
        # Test different population sizes
        population_sizes = [10, 20, 30]
        optimization_times = []
        
        for pop_size in population_sizes:
            self.optimizer.population_size = pop_size
            self.optimizer.max_generations = 5  # Keep generations low for testing
            
            start_time = time.time()
            result = self.optimizer.optimize_formulation(request)
            end_time = time.time()
            
            optimization_times.append(end_time - start_time)
        
        # Optimization time should scale reasonably
        time_ratio = optimization_times[-1] / optimization_times[0]
        pop_ratio = population_sizes[-1] / population_sizes[0]
        
        # Should not scale worse than O(n^2)
        self.assertLess(time_ratio, pop_ratio ** 2,
                       f"Poor scalability: {time_ratio:.2f}x time for {pop_ratio:.2f}x population")
        
        print(f"    Optimization Scalability: {time_ratio:.2f}x time for {pop_ratio:.2f}x population")


def run_hypergredient_tests():
    """Run comprehensive hypergredient framework tests."""
    print("🧪 Hypergredient Framework - Comprehensive Test Suite")
    print("=" * 65)
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available - skipping tests")
        return False
    
    # Test suites to run
    test_suites = [
        ('TestHypergredientDatabase', TestHypergredientDatabase),
        ('TestHypergredientOptimizer', TestHypergredientOptimizer),
        ('TestFormulationGeneration', TestFormulationGeneration),
        ('TestPerformanceBenchmarks', TestPerformanceBenchmarks),
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for suite_name, test_class in test_suites:
        print(f"\n📋 Running {suite_name}...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
        result = runner.run(suite)
        
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        
        if failures == 0 and errors == 0:
            print(f"  ✅ All {tests_run} tests passed")
        else:
            print(f"  ❌ {tests_run} tests run, {failures} failures, {errors} errors")
            
            # Print failure details
            for test, traceback in result.failures:
                print(f"    FAIL: {test}")
                print(f"    {traceback.split('AssertionError: ')[-1].strip()}")
            
            for test, traceback in result.errors:
                print(f"    ERROR: {test}")
                print(f"    {traceback.split('Exception: ')[-1].strip()}")
    
    # Final summary
    print(f"\n🏆 Hypergredient Framework Test Summary")
    print("=" * 42)
    print(f"Total tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    success_rate = ((total_tests - total_failures - total_errors) / max(total_tests, 1)) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("✅ All tests PASSED - Hypergredient Framework is working correctly!")
        return True
    else:
        print("❌ Some tests FAILED - System needs improvement")
        return False


if __name__ == "__main__":
    success = run_hypergredient_tests()
    exit(0 if success else 1)
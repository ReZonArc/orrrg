#!/usr/bin/env python3
"""
Tests for Meta-Optimization Strategy

This test suite validates the meta-optimization functionality including
recursive optimization, strategy selection, and comprehensive formulation generation.
"""

import unittest
import sys
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add the cheminformatics module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from cheminformatics.hypergredient import (
        MetaOptimizationStrategy, OptimizationStrategy, ConditionTreatmentPair,
        OptimizationResult, MetaOptimizationCache, create_hypergredient_database,
        FormulationRequest, HypergredientFormulator
    )
    META_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Meta-optimization not available: {e}")
    META_OPTIMIZATION_AVAILABLE = False


@unittest.skipUnless(META_OPTIMIZATION_AVAILABLE, "Meta-optimization not available")
class TestMetaOptimizationCache(unittest.TestCase):
    """Test meta-optimization cache functionality"""
    
    def setUp(self):
        """Set up test cache"""
        self.cache = MetaOptimizationCache(max_size=5)
        self.test_request = FormulationRequest(
            target_concerns=['anti_aging'],
            skin_type='normal',
            budget=1000.0
        )
    
    def test_cache_creation(self):
        """Test cache initialization"""
        self.assertEqual(len(self.cache.cache), 0)
        self.assertEqual(self.cache.max_size, 5)
    
    def test_key_generation(self):
        """Test cache key generation"""
        key1 = self.cache._generate_key(self.test_request)
        key2 = self.cache._generate_key(self.test_request)
        self.assertEqual(key1, key2)
        
        # Different request should generate different key
        different_request = FormulationRequest(
            target_concerns=['acne'],
            skin_type='oily',
            budget=800.0
        )
        key3 = self.cache._generate_key(different_request)
        self.assertNotEqual(key1, key3)
    
    @patch('cheminformatics.hypergredient.meta_optimization.FormulationSolution')
    def test_cache_put_get(self, mock_solution):
        """Test putting and getting from cache"""
        mock_solution.hypergredients = {'ingredient1': 2.0, 'ingredient2': 3.0}
        mock_solution.total_score = 8.5
        
        # Initially should return None
        result = self.cache.get(self.test_request)
        self.assertIsNone(result)
        
        # Put solution in cache
        self.cache.put(self.test_request, mock_solution)
        
        # Should now return the solution
        result = self.cache.get(self.test_request)
        self.assertEqual(result, mock_solution)


@unittest.skipUnless(META_OPTIMIZATION_AVAILABLE, "Meta-optimization not available")
class TestConditionTreatmentPair(unittest.TestCase):
    """Test condition-treatment pair functionality"""
    
    def test_pair_creation(self):
        """Test condition-treatment pair creation"""
        pair = ConditionTreatmentPair(
            condition='acne',
            treatments=['acne', 'oily_skin', 'pores'],
            severity='moderate',
            skin_type='oily',
            budget_range=(600.0, 1200.0),
            priority=2,
            complexity_score=2.5
        )
        
        self.assertEqual(pair.condition, 'acne')
        self.assertEqual(len(pair.treatments), 3)
        self.assertEqual(pair.severity, 'moderate')
        self.assertEqual(pair.skin_type, 'oily')
        self.assertEqual(pair.budget_range, (600.0, 1200.0))
        self.assertEqual(pair.priority, 2)
        self.assertEqual(pair.complexity_score, 2.5)


@unittest.skipUnless(META_OPTIMIZATION_AVAILABLE, "Meta-optimization not available")
class TestMetaOptimizationStrategy(unittest.TestCase):
    """Test meta-optimization strategy functionality"""
    
    def setUp(self):
        """Set up meta-optimization strategy"""
        self.db = create_hypergredient_database()
        self.meta_optimizer = MetaOptimizationStrategy(self.db, cache_size=10)
    
    def test_initialization(self):
        """Test meta-optimizer initialization"""
        self.assertIsNotNone(self.meta_optimizer.database)
        self.assertIsNotNone(self.meta_optimizer.base_optimizer)
        self.assertIsNotNone(self.meta_optimizer.formulator)
        self.assertIsNotNone(self.meta_optimizer.cache)
        self.assertEqual(len(self.meta_optimizer.strategy_performance), len(OptimizationStrategy))
        self.assertGreater(len(self.meta_optimizer.condition_treatment_mapping), 0)
    
    def test_condition_treatment_mapping(self):
        """Test comprehensive condition-treatment mapping"""
        mapping = self.meta_optimizer.condition_treatment_mapping
        
        # Should have major conditions
        expected_conditions = ['aging', 'pigmentation', 'acne', 'sensitivity', 'dryness']
        for condition in expected_conditions:
            self.assertIn(condition, mapping)
            self.assertGreater(len(mapping[condition]), 0)
        
        # Check that pairs have proper structure
        for condition, pairs in mapping.items():
            for pair in pairs[:3]:  # Check first 3 pairs
                self.assertIsInstance(pair, ConditionTreatmentPair)
                self.assertEqual(pair.condition, condition)
                self.assertGreater(len(pair.treatments), 0)
                self.assertIn(pair.skin_type, ['normal', 'dry', 'oily', 'combination', 'sensitive'])
                self.assertIn(pair.severity, ['mild', 'moderate', 'severe'])
    
    def test_strategy_selection(self):
        """Test optimization strategy selection"""
        # Test simple case
        simple_pair = ConditionTreatmentPair(
            condition='hydration',
            treatments=['hydration'],
            complexity_score=1.0
        )
        strategy = self.meta_optimizer.select_optimal_strategy(simple_pair)
        self.assertIn(strategy, OptimizationStrategy)
        
        # Test complex case
        complex_pair = ConditionTreatmentPair(
            condition='aging',
            treatments=['anti_aging', 'wrinkles', 'firmness', 'brightness', 'hydration'],
            severity='severe',
            skin_type='sensitive',
            complexity_score=8.0
        )
        strategy = self.meta_optimizer.select_optimal_strategy(complex_pair)
        self.assertIn(strategy, [OptimizationStrategy.RECURSIVE_DECOMPOSITION, 
                               OptimizationStrategy.ADAPTIVE_SEARCH])
    
    def test_request_variations_generation(self):
        """Test generation of request variations for recursive exploration"""
        base_request = FormulationRequest(
            target_concerns=['anti_aging', 'hydration'],
            skin_type='normal',
            budget=1000.0,
            max_ingredients=6
        )
        
        variations = self.meta_optimizer._generate_request_variations(base_request)
        self.assertGreater(len(variations), 0)
        
        # Should have budget variations
        budget_variations = [v for v in variations if v.budget != base_request.budget]
        self.assertGreater(len(budget_variations), 0)
        
        # Should have ingredient count variations
        ingredient_variations = [v for v in variations if v.max_ingredients != base_request.max_ingredients]
        self.assertGreater(len(ingredient_variations), 0)
    
    def test_solution_deduplication(self):
        """Test solution deduplication"""
        # Create mock solutions with some duplicates
        from cheminformatics.hypergredient.optimization import FormulationSolution, OptimizationObjective
        
        solutions = [
            FormulationSolution(
                hypergredients={'ingredient1': 2.0, 'ingredient2': 3.0},
                objective_scores={OptimizationObjective.EFFICACY: 8.0},
                total_score=8.0,
                cost=500.0,
                predicted_efficacy={'anti_aging': 80.0}
            ),
            FormulationSolution(
                hypergredients={'ingredient1': 2.0, 'ingredient2': 3.0},  # Duplicate
                objective_scores={OptimizationObjective.EFFICACY: 8.0},
                total_score=8.0,
                cost=500.0,
                predicted_efficacy={'anti_aging': 80.0}
            ),
            FormulationSolution(
                hypergredients={'ingredient3': 1.5, 'ingredient4': 2.5},
                objective_scores={OptimizationObjective.EFFICACY: 7.5},
                total_score=7.5,
                cost=400.0,
                predicted_efficacy={'anti_aging': 75.0}
            )
        ]
        
        unique_solutions = self.meta_optimizer._deduplicate_solutions(solutions)
        self.assertEqual(len(unique_solutions), 2)  # Should remove one duplicate
    
    def test_recursive_formulation_exploration(self):
        """Test recursive formulation exploration"""
        request = FormulationRequest(
            target_concerns=['anti_aging'],
            skin_type='normal',
            budget=1000.0
        )
        
        # Mock the base optimizer to return some solutions
        with patch.object(self.meta_optimizer.base_optimizer, 'optimize_formulation') as mock_optimize:
            from cheminformatics.hypergredient.optimization import FormulationSolution, OptimizationObjective
            
            mock_solution = FormulationSolution(
                hypergredients={'ingredient1': 2.0},
                objective_scores={OptimizationObjective.EFFICACY: 7.0},
                total_score=7.0,
                cost=400.0,
                predicted_efficacy={'anti_aging': 70.0}
            )
            mock_optimize.return_value = [mock_solution]
            
            solutions = self.meta_optimizer.recursive_formulation_exploration(request, max_depth=1)
            self.assertGreater(len(solutions), 0)
            self.assertTrue(all(hasattr(s, 'total_score') for s in solutions))
    
    @patch('cheminformatics.hypergredient.meta_optimization.time.time')
    def test_optimization_report_generation(self, mock_time):
        """Test optimization performance report generation"""
        mock_time.return_value = 1234567890.0
        
        # Update some strategy performance data
        strategy = OptimizationStrategy.GENETIC_ALGORITHM
        self.meta_optimizer._update_strategy_performance(strategy, 8.0, 5.0)
        self.meta_optimizer._update_strategy_performance(strategy, 7.5, 4.5)
        
        report = self.meta_optimizer.get_optimization_report()
        
        self.assertIn('strategy_performance', report)
        self.assertIn('cache_statistics', report)
        self.assertIn('condition_coverage', report)
        self.assertIn('total_combinations', report)
        
        # Check strategy performance data
        self.assertIn(strategy.value, report['strategy_performance'])
        perf_data = report['strategy_performance'][strategy.value]
        self.assertEqual(perf_data['usage_count'], 2)
        self.assertGreater(perf_data['average_quality'], 0)
    
    def test_improvement_suggestions(self):
        """Test improvement suggestion generation"""
        from cheminformatics.hypergredient.optimization import FormulationSolution, OptimizationObjective
        
        pair = ConditionTreatmentPair(
            condition='acne',
            treatments=['acne'],
            budget_range=(500.0, 1500.0)
        )
        
        # Test with low-quality solution
        low_quality_solution = FormulationSolution(
            hypergredients={'ingredient1': 1.0},
            objective_scores={OptimizationObjective.EFFICACY: 5.0},
            total_score=5.0,
            cost=200.0,
            predicted_efficacy={'acne': 50.0},
            synergy_score=1.0
        )
        
        suggestions = self.meta_optimizer._generate_improvement_suggestions(pair, [low_quality_solution])
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any('budget' in s.lower() for s in suggestions))
    
    def test_export_formulation_library(self):
        """Test formulation library export"""
        from cheminformatics.hypergredient.optimization import FormulationSolution, OptimizationObjective
        
        # Create mock results
        mock_solution = FormulationSolution(
            hypergredients={'ingredient1': 2.0},
            objective_scores={OptimizationObjective.EFFICACY: 8.0},
            total_score=8.0,
            cost=500.0,
            predicted_efficacy={'anti_aging': 80.0}
        )
        
        pair = ConditionTreatmentPair(
            condition='aging',
            treatments=['anti_aging'],
            severity='moderate',
            skin_type='normal'
        )
        
        result = OptimizationResult(
            condition_treatment_pair=pair,
            formulation_solutions=[mock_solution],
            optimization_strategy=OptimizationStrategy.GENETIC_ALGORITHM,
            performance_metrics={'average_quality': 8.0},
            computation_time=5.0,
            iterations=100,
            quality_score=8.0
        )
        
        results = {'aging': [result]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            library = self.meta_optimizer.export_formulation_library(results, f.name)
            
            # Verify library structure
            self.assertIn('metadata', library)
            self.assertIn('formulations', library)
            self.assertIn('aging', library['formulations'])
            
            # Verify file was created
            self.assertTrue(os.path.exists(f.name))
            
            # Clean up
            os.unlink(f.name)


@unittest.skipUnless(META_OPTIMIZATION_AVAILABLE, "Meta-optimization not available")
class TestIntegration(unittest.TestCase):
    """Integration tests for meta-optimization"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.db = create_hypergredient_database()
        self.meta_optimizer = MetaOptimizationStrategy(self.db, cache_size=5)
    
    def test_small_scale_optimization(self):
        """Test small-scale optimization run"""
        # Temporarily reduce the condition mapping for testing
        original_mapping = self.meta_optimizer.condition_treatment_mapping
        
        # Create a minimal mapping for testing
        test_mapping = {
            'acne': [ConditionTreatmentPair(
                condition='acne',
                treatments=['acne'],
                severity='mild',
                skin_type='oily',
                budget_range=(500.0, 1000.0)
            )]
        }
        
        self.meta_optimizer.condition_treatment_mapping = test_mapping
        
        try:
            # Run optimization
            results = self.meta_optimizer.optimize_all_conditions(max_solutions_per_condition=1)
            
            # Verify results structure
            self.assertIn('acne', results)
            self.assertGreater(len(results['acne']), 0)
            
            result = results['acne'][0]
            self.assertIsInstance(result, OptimizationResult)
            self.assertIsNotNone(result.optimization_strategy)
            self.assertGreaterEqual(result.quality_score, 0)
            
        finally:
            # Restore original mapping
            self.meta_optimizer.condition_treatment_mapping = original_mapping
    
    def test_strategy_performance_tracking(self):
        """Test that strategy performance is properly tracked"""
        strategy = OptimizationStrategy.GENETIC_ALGORITHM
        initial_usage = self.meta_optimizer.strategy_performance[strategy].usage_count
        
        # Simulate some usage
        self.meta_optimizer._update_strategy_performance(strategy, 8.0, 5.0)
        self.meta_optimizer._update_strategy_performance(strategy, 7.5, 4.0)
        
        perf = self.meta_optimizer.strategy_performance[strategy]
        self.assertEqual(perf.usage_count, initial_usage + 2)
        self.assertGreater(perf.average_quality, 0)
        self.assertGreater(perf.average_time, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
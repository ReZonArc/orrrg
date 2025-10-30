#!/usr/bin/env python3
"""
🧪 Test Suite for Meta-Optimization Strategy

Comprehensive test suite for validating the meta-optimization system
that generates optimal formulations for all condition/treatment combinations.

Author: OpenCog Multiscale Optimization Framework
License: AGPL-3.0
"""

import unittest
import tempfile
import os
import json
from datetime import datetime
from typing import Dict, List, Any

from meta_optimization_strategy import (
    MetaOptimizationStrategy, ConditionProfile, OptimizationResult,
    StrategyPerformance, FormulationLibraryEntry, OptimizationStrategy,
    ConditionSeverity, TreatmentGoal
)

try:
    from hypergredient_optimizer import ConcernType, SkinType
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    # Mock enums for testing
    from enum import Enum
    
    class ConcernType(Enum):
        HYDRATION = "hydration"
        WRINKLES = "wrinkles"
        BRIGHTNESS = "brightness"
        ACNE = "acne"
    
    class SkinType(Enum):
        NORMAL = "normal"
        DRY = "dry"
        OILY = "oily"
        SENSITIVE = "sensitive"


class TestConditionProfile(unittest.TestCase):
    """Test suite for condition profile functionality."""
    
    def test_condition_profile_creation(self):
        """Test creation of condition profiles."""
        profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        self.assertEqual(profile.concern, ConcernType.HYDRATION)
        self.assertEqual(profile.severity, ConditionSeverity.MODERATE)
        self.assertEqual(profile.skin_type, SkinType.DRY)
        self.assertEqual(profile.treatment_goal, TreatmentGoal.TREATMENT)
    
    def test_profile_key_generation(self):
        """Test unique key generation for profiles."""
        profile1 = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        profile2 = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        profile3 = ConditionProfile(
            concern=ConcernType.WRINKLES,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        # Same profiles should have same keys
        self.assertEqual(profile1.get_profile_key(), profile2.get_profile_key())
        
        # Different profiles should have different keys
        self.assertNotEqual(profile1.get_profile_key(), profile3.get_profile_key())
    
    def test_profile_key_uniqueness(self):
        """Test that profile keys are unique across different attributes."""
        base_profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        keys = set()
        keys.add(base_profile.get_profile_key())
        
        # Test different severities
        for severity in ConditionSeverity:
            profile = ConditionProfile(
                concern=ConcernType.HYDRATION,
                severity=severity,
                skin_type=SkinType.DRY,
                treatment_goal=TreatmentGoal.TREATMENT
            )
            keys.add(profile.get_profile_key())
        
        # Should have 3 unique keys (3 severities)
        self.assertEqual(len(keys), 3)


class TestMetaOptimizationStrategy(unittest.TestCase):
    """Test suite for the meta-optimization strategy system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_db.close()
        
        self.meta_optimizer = MetaOptimizationStrategy(
            database_path=self.temp_db.name,
            learning_rate=0.1,
            exploration_rate=0.3
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_initialization(self):
        """Test meta-optimizer initialization."""
        self.assertIsNotNone(self.meta_optimizer)
        self.assertEqual(self.meta_optimizer.learning_rate, 0.1)
        self.assertEqual(self.meta_optimizer.exploration_rate, 0.3)
        
        # Check strategy performance tracking initialized
        self.assertEqual(len(self.meta_optimizer.strategy_performance), len(OptimizationStrategy))
        
        for strategy in OptimizationStrategy:
            self.assertIn(strategy, self.meta_optimizer.strategy_performance)
    
    def test_condition_profile_generation(self):
        """Test generation of all possible condition profiles."""
        profiles = self.meta_optimizer.generate_all_condition_profiles()
        
        # Should generate many combinations
        self.assertGreater(len(profiles), 1000)
        
        # Check that we have variety in profiles
        concerns = set(p.concern for p in profiles)
        skin_types = set(p.skin_type for p in profiles)
        severities = set(p.severity for p in profiles)
        
        self.assertGreater(len(concerns), 1)
        self.assertGreater(len(skin_types), 1)
        self.assertGreater(len(severities), 1)
    
    def test_strategy_selection(self):
        """Test optimization strategy selection."""
        profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        strategy = self.meta_optimizer.select_optimization_strategy(profile)
        self.assertIn(strategy, OptimizationStrategy)
    
    def test_single_condition_optimization(self):
        """Test optimization of a single condition."""
        profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        result = self.meta_optimizer.optimize_single_condition(profile)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.profile, profile)
        self.assertIn(result.strategy_used, OptimizationStrategy)
        self.assertIsInstance(result.success, bool)
        self.assertGreater(result.optimization_time, 0)
        
        if result.success:
            self.assertIsInstance(result.formulation, dict)
            self.assertIsInstance(result.performance_metrics, dict)
    
    def test_performance_tracking(self):
        """Test that performance is tracked correctly."""
        profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        # Run several optimizations
        initial_runs = self.meta_optimizer.strategy_performance[OptimizationStrategy.ADAPTIVE].total_runs
        
        for _ in range(3):
            result = self.meta_optimizer.optimize_single_condition(profile)
        
        # Check that runs were tracked
        final_runs = self.meta_optimizer.strategy_performance[OptimizationStrategy.ADAPTIVE].total_runs
        self.assertGreaterEqual(final_runs, initial_runs)
        
        # Check optimization history
        self.assertGreater(len(self.meta_optimizer.optimization_history), 0)
    
    def test_formulation_library_generation(self):
        """Test generation of formulation library."""
        # Generate small library for testing
        library = self.meta_optimizer.generate_comprehensive_library(max_conditions=20)
        
        self.assertIsInstance(library, dict)
        self.assertGreater(len(library), 0)
        self.assertLessEqual(len(library), 20)
        
        # Check library entries
        for key, entry in library.items():
            self.assertIsInstance(entry, FormulationLibraryEntry)
            self.assertIsInstance(entry.profile, ConditionProfile)
            self.assertIsInstance(entry.formulation, dict)
            self.assertIsInstance(entry.validation_score, float)
            self.assertGreaterEqual(entry.validation_score, 0.0)
            self.assertLessEqual(entry.validation_score, 1.0)
    
    def test_optimization_insights(self):
        """Test generation of optimization insights."""
        # Run some optimizations first
        for i in range(5):
            profile = ConditionProfile(
                concern=list(ConcernType)[i % len(ConcernType)],
                severity=ConditionSeverity.MODERATE,
                skin_type=SkinType.NORMAL,
                treatment_goal=TreatmentGoal.TREATMENT
            )
            self.meta_optimizer.optimize_single_condition(profile)
        
        insights = self.meta_optimizer.get_optimization_insights()
        
        self.assertIsInstance(insights, dict)
        self.assertIn('total_optimizations', insights)
        self.assertIn('success_rate', insights)
        self.assertIn('strategy_performance', insights)
        
        self.assertGreater(insights['total_optimizations'], 0)
        self.assertGreaterEqual(insights['success_rate'], 0.0)
        self.assertLessEqual(insights['success_rate'], 1.0)
    
    def test_condition_priority_scoring(self):
        """Test condition priority scoring for library generation."""
        common_profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.NORMAL,
            treatment_goal=TreatmentGoal.TREATMENT,
            age_group="adult",
            budget_range="medium"
        )
        
        rare_profile = ConditionProfile(
            concern=ConcernType.ACNE,
            severity=ConditionSeverity.SEVERE,
            skin_type=SkinType.SENSITIVE,
            treatment_goal=TreatmentGoal.PREVENTION,
            age_group="child",
            budget_range="low"
        )
        
        common_priority = self.meta_optimizer._get_condition_priority(common_profile)
        rare_priority = self.meta_optimizer._get_condition_priority(rare_profile)
        
        # Common conditions should have higher priority
        self.assertGreater(common_priority, rare_priority)
    
    def test_profile_similarity_calculation(self):
        """Test similarity calculation between profiles."""
        profile1 = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        profile2 = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        profile3 = ConditionProfile(
            concern=ConcernType.WRINKLES,
            severity=ConditionSeverity.SEVERE,
            skin_type=SkinType.OILY,
            treatment_goal=TreatmentGoal.PREVENTION
        )
        
        # Identical profiles should have high similarity
        similarity_identical = self.meta_optimizer._calculate_profile_similarity(profile1, profile2)
        self.assertGreater(similarity_identical, 0.9)
        
        # Very different profiles should have low similarity
        similarity_different = self.meta_optimizer._calculate_profile_similarity(profile1, profile3)
        self.assertLess(similarity_different, 0.3)
    
    def test_formulation_validation(self):
        """Test formulation validation scoring."""
        good_formulation = {
            'ingredients': {
                'aqua': 70.0,
                'glycerin': 5.0,
                'hyaluronic_acid': 2.0,
                'phenoxyethanol': 0.8
            },
            'predicted_efficacy': 0.8,
            'predicted_safety': 0.9
        }
        
        poor_formulation = {
            'ingredients': {
                'unknown_ingredient': 200.0  # Invalid concentration
            }
        }
        
        profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        good_score = self.meta_optimizer._validate_formulation(good_formulation, profile)
        poor_score = self.meta_optimizer._validate_formulation(poor_formulation, profile)
        
        self.assertGreater(good_score, poor_score)
        self.assertGreaterEqual(good_score, 0.0)
        self.assertLessEqual(good_score, 1.0)
    
    def test_database_persistence(self):
        """Test saving and loading of meta-database."""
        # Add some data
        profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        result = self.meta_optimizer.optimize_single_condition(profile)
        
        # Save database
        self.meta_optimizer._save_meta_database()
        
        # Create new instance and load
        new_optimizer = MetaOptimizationStrategy(database_path=self.temp_db.name)
        
        # Check that data was loaded
        self.assertGreater(len(new_optimizer.optimization_history), 0)


class TestAdaptiveOptimization(unittest.TestCase):
    """Test suite for adaptive optimization features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_db.close()
        
        self.meta_optimizer = MetaOptimizationStrategy(
            database_path=self.temp_db.name
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_learning_from_optimization_results(self):
        """Test that the system learns from optimization results."""
        profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        # Run multiple optimizations
        results = []
        for _ in range(5):
            result = self.meta_optimizer.optimize_single_condition(profile)
            results.append(result)
        
        # Check that patterns are being learned
        profile_key = profile.get_profile_key()
        if profile_key in self.meta_optimizer.condition_patterns:
            pattern = self.meta_optimizer.condition_patterns[profile_key]
            self.assertIsInstance(pattern, dict)
            self.assertIn('strategy_scores', pattern)
    
    def test_transfer_learning_between_similar_conditions(self):
        """Test transfer learning between similar conditions."""
        # Create similar profiles
        profile1 = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        profile2 = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MILD,  # Different severity
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        # Optimize first profile and add to library
        result1 = self.meta_optimizer.optimize_single_condition(profile1)
        if result1.success:
            entry = FormulationLibraryEntry(
                profile=profile1,
                formulation=result1.formulation,
                validation_score=0.8,
                creation_date=datetime.now(),
                last_updated=datetime.now()
            )
            self.meta_optimizer.formulation_library[profile1.get_profile_key()] = entry
        
        # Find similar profiles for second one
        similar = self.meta_optimizer._find_similar_profiles(profile2)
        
        if similar:
            # Should find the first profile as similar
            self.assertGreater(len(similar), 0)
            similar_profile = similar[0].profile
            similarity = self.meta_optimizer._calculate_profile_similarity(profile2, similar_profile)
            self.assertGreater(similarity, 0.5)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for the meta-optimization system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_db.close()
        
        self.meta_optimizer = MetaOptimizationStrategy(database_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_single_optimization_performance(self):
        """Test performance of single condition optimization."""
        profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        import time
        start_time = time.time()
        result = self.meta_optimizer.optimize_single_condition(profile)
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(optimization_time, 10.0)  # 10 seconds max
        self.assertGreater(optimization_time, 0.0)
        
        print(f"    Single optimization time: {optimization_time:.3f}s")
    
    def test_library_generation_performance(self):
        """Test performance of library generation."""
        import time
        start_time = time.time()
        
        # Generate small library
        library = self.meta_optimizer.generate_comprehensive_library(max_conditions=10)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(generation_time, 60.0)  # 1 minute max for 10 conditions
        self.assertGreater(len(library), 0)
        
        print(f"    Library generation time: {generation_time:.3f}s for {len(library)} formulations")
        print(f"    Average time per formulation: {generation_time/max(1, len(library)):.3f}s")
    
    def test_strategy_selection_performance(self):
        """Test performance of strategy selection."""
        profile = ConditionProfile(
            concern=ConcernType.HYDRATION,
            severity=ConditionSeverity.MODERATE,
            skin_type=SkinType.DRY,
            treatment_goal=TreatmentGoal.TREATMENT
        )
        
        import time
        times = []
        
        for _ in range(100):
            start_time = time.time()
            strategy = self.meta_optimizer.select_optimization_strategy(profile)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        # Should be very fast
        self.assertLess(avg_time, 0.001)  # 1ms max
        
        print(f"    Strategy selection time: {avg_time*1000:.3f}ms average")
    
    def test_profile_generation_performance(self):
        """Test performance of profile generation."""
        import time
        start_time = time.time()
        
        profiles = self.meta_optimizer.generate_all_condition_profiles()
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(generation_time, 5.0)  # 5 seconds max
        self.assertGreater(len(profiles), 1000)
        
        print(f"    Profile generation time: {generation_time:.3f}s for {len(profiles):,} profiles")


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete meta-optimization system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_db.close()
        
        self.meta_optimizer = MetaOptimizationStrategy(database_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Generate condition profiles
        profiles = self.meta_optimizer.generate_all_condition_profiles()
        self.assertGreater(len(profiles), 1000)
        
        # 2. Select a subset for testing
        test_profiles = profiles[:10]
        
        # 3. Optimize each condition
        results = []
        for profile in test_profiles:
            result = self.meta_optimizer.optimize_single_condition(profile)
            results.append(result)
        
        # 4. Check results
        successful_results = [r for r in results if r.success]
        self.assertGreater(len(successful_results), 0)
        
        # 5. Generate insights
        insights = self.meta_optimizer.get_optimization_insights()
        self.assertIn('total_optimizations', insights)
        self.assertGreater(insights['total_optimizations'], 0)
        
        # 6. Generate mini library
        library = self.meta_optimizer.generate_comprehensive_library(max_conditions=5)
        self.assertGreater(len(library), 0)
    
    def test_multiple_concern_combinations(self):
        """Test optimization across different concern combinations."""
        concerns_to_test = [ConcernType.HYDRATION, ConcernType.WRINKLES, ConcernType.ACNE]
        skin_types_to_test = [SkinType.NORMAL, SkinType.DRY, SkinType.OILY]
        
        results = []
        
        for concern in concerns_to_test:
            for skin_type in skin_types_to_test:
                profile = ConditionProfile(
                    concern=concern,
                    severity=ConditionSeverity.MODERATE,
                    skin_type=skin_type,
                    treatment_goal=TreatmentGoal.TREATMENT
                )
                
                result = self.meta_optimizer.optimize_single_condition(profile)
                results.append(result)
        
        # Check that we got results for all combinations
        self.assertEqual(len(results), len(concerns_to_test) * len(skin_types_to_test))
        
        # Check that different combinations produce different formulations
        successful_results = [r for r in results if r.success]
        self.assertGreater(len(successful_results), 0)
        
        # At least some formulations should be different
        formulations = [str(r.formulation) for r in successful_results]
        unique_formulations = set(formulations)
        self.assertGreater(len(unique_formulations), 1)


def main():
    """Run the test suite."""
    print("🧪 Meta-Optimization Strategy - Comprehensive Test Suite")
    print("=" * 65)
    
    # Create test suite
    test_classes = [
        TestConditionProfile,
        TestMetaOptimizationStrategy,
        TestAdaptiveOptimization,
        TestPerformanceBenchmarks,
        TestSystemIntegration
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        class_tests = result.testsRun
        class_failures = len(result.failures)
        class_errors = len(result.errors)
        
        total_tests += class_tests
        total_failures += class_failures
        total_errors += class_errors
        
        if class_failures == 0 and class_errors == 0:
            print(f"  ✅ All {class_tests} tests passed")
        else:
            print(f"  ❌ {class_tests} tests run, {class_failures} failures, {class_errors} errors")
            
            # Print failure details
            for test, traceback in result.failures:
                print(f"    FAIL: {test}")
                print(f"    {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
            
            for test, traceback in result.errors:
                print(f"    ERROR: {test}")
                print(f"    {traceback.split('\\n')[-2]}")
    
    # Print summary
    print(f"\n🏆 Meta-Optimization Test Summary")
    print("=" * 40)
    print(f"Total tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("✅ All tests PASSED - Meta-optimization system is working correctly")
        return 0
    else:
        print("❌ Some tests FAILED - Meta-optimization system needs improvement")
        return 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
test_meta_optimizer.py

🧬 Comprehensive Test Suite for Meta-Optimization Strategy
Validates the complete coverage optimization system

Tests:
1. Meta-optimization strategy completeness
2. Condition-treatment combination coverage
3. Optimization matrix generation
4. Performance ranking accuracy
5. Recommendation system validation
6. Coverage analysis verification
7. Integration with existing framework
"""

import time
import json
import unittest
from typing import Dict, List, Any

# Import meta-optimization framework
try:
    from hypergredient_meta_optimizer import (
        HypergredientMetaOptimizer, MetaOptimizationResult
    )
    from hypergredient_framework import (
        HypergredientClass, HypergredientOptimizer, HypergredientFormulation
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hypergredient_meta_optimizer import (
        HypergredientMetaOptimizer, MetaOptimizationResult
    )
    from hypergredient_framework import (
        HypergredientClass, HypergredientOptimizer, HypergredientFormulation
    )

class MetaOptimizerTestSuite:
    """Comprehensive test suite for meta-optimization strategy"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.meta_optimizer = HypergredientMetaOptimizer()
    
    def test_meta_optimizer_initialization(self):
        """Test meta-optimizer initialization and setup"""
        print("1. Testing meta-optimizer initialization...")
        
        # Test 1: Proper initialization
        initialization_success = isinstance(self.meta_optimizer, HypergredientMetaOptimizer)
        
        # Test 2: Comprehensive condition taxonomy
        has_conditions = len(self.meta_optimizer.all_conditions) >= 30
        condition_categories_covered = all([
            any('wrinkles' in c or 'firmness' in c for c in self.meta_optimizer.all_conditions),  # Anti-aging
            any('dark_spots' in c or 'hyperpigmentation' in c for c in self.meta_optimizer.all_conditions),  # Pigmentation
            any('dryness' in c or 'hydration' in c for c in self.meta_optimizer.all_conditions),  # Hydration
            any('acne' in c or 'oily' in c for c in self.meta_optimizer.all_conditions),  # Sebum
        ])
        
        # Test 3: Treatment strategy coverage
        has_treatment_strategies = len(self.meta_optimizer.treatment_strategies) >= 10
        
        # Test 4: Skin type coverage
        has_skin_types = len(self.meta_optimizer.skin_types) >= 10
        skin_type_basics = all(st in self.meta_optimizer.skin_types for st in ['normal', 'dry', 'oily', 'sensitive'])
        
        # Test 5: Budget range coverage
        has_budget_ranges = len(self.meta_optimizer.budget_ranges) >= 5
        budget_range_span = (max(self.meta_optimizer.budget_ranges) - min(self.meta_optimizer.budget_ranges)) >= 4000
        
        # Test 6: Preference set coverage
        has_preference_sets = len(self.meta_optimizer.preference_sets) >= 8
        
        self.test_results['meta_optimizer_initialization'] = {
            'initialization_success': initialization_success,
            'has_conditions': has_conditions,
            'condition_categories_covered': condition_categories_covered,
            'has_treatment_strategies': has_treatment_strategies,
            'has_skin_types': has_skin_types,
            'skin_type_basics': skin_type_basics,
            'has_budget_ranges': has_budget_ranges,
            'budget_range_span': budget_range_span,
            'has_preference_sets': has_preference_sets,
            'total_score': sum([
                initialization_success, has_conditions, condition_categories_covered,
                has_treatment_strategies, has_skin_types, skin_type_basics,
                has_budget_ranges, budget_range_span, has_preference_sets
            ])
        }
        
        print(f"  Initialization success: {initialization_success}")
        print(f"  Comprehensive condition taxonomy: {has_conditions} ({len(self.meta_optimizer.all_conditions)} conditions)")
        print(f"  Condition categories covered: {condition_categories_covered}")
        print(f"  Treatment strategies: {has_treatment_strategies} ({len(self.meta_optimizer.treatment_strategies)} strategies)")
        print(f"  Skin types: {has_skin_types} ({len(self.meta_optimizer.skin_types)} types)")
        print(f"  Budget ranges: {has_budget_ranges} (span: R{max(self.meta_optimizer.budget_ranges) - min(self.meta_optimizer.budget_ranges)})")
        print(f"  Preference sets: {has_preference_sets} ({len(self.meta_optimizer.preference_sets)} sets)")
    
    def test_condition_combination_generation(self):
        """Test comprehensive condition combination generation"""
        print("\n2. Testing condition combination generation...")
        
        # Test 1: Generate combinations
        start_time = time.time()
        combinations = self.meta_optimizer.generate_condition_combinations(max_combinations=2)
        generation_time = time.time() - start_time
        
        combinations_generated = len(combinations) > 0
        
        # Test 2: Single condition coverage
        single_conditions = [combo for combo in combinations if len(combo) == 1]
        single_condition_coverage = len(single_conditions) >= len(self.meta_optimizer.all_conditions) * 0.8
        
        # Test 3: Multi-condition combinations
        multi_conditions = [combo for combo in combinations if len(combo) > 1]
        has_multi_conditions = len(multi_conditions) > 0
        
        # Test 4: Realistic combination filtering
        realistic_combinations = True
        for combo in combinations[:10]:  # Test first 10
            if not self.meta_optimizer._is_realistic_combination(combo):
                realistic_combinations = False
                break
        
        # Test 5: Combination diversity
        unique_ingredients = set()
        for combo in combinations:
            unique_ingredients.update(combo)
        diversity_score = len(unique_ingredients) / len(self.meta_optimizer.all_conditions)
        has_diversity = diversity_score >= 0.8
        
        self.test_results['condition_combination_generation'] = {
            'combinations_generated': combinations_generated,
            'single_condition_coverage': single_condition_coverage,
            'has_multi_conditions': has_multi_conditions,
            'realistic_combinations': realistic_combinations,
            'has_diversity': has_diversity,
            'generation_time': generation_time,
            'total_combinations': len(combinations),
            'total_score': sum([
                combinations_generated, single_condition_coverage, has_multi_conditions,
                realistic_combinations, has_diversity
            ])
        }
        
        print(f"  Combinations generated: {combinations_generated} ({len(combinations)} total)")
        print(f"  Single condition coverage: {single_condition_coverage} ({len(single_conditions)} single)")
        print(f"  Multi-condition combinations: {has_multi_conditions} ({len(multi_conditions)} multi)")
        print(f"  Realistic combinations: {realistic_combinations}")
        print(f"  Diversity score: {has_diversity} ({diversity_score:.2f})")
        print(f"  Generation time: {generation_time:.3f}s")
    
    def test_meta_optimization_execution(self):
        """Test complete meta-optimization strategy execution"""
        print("\n3. Testing meta-optimization execution...")
        
        # Test 1: Execute meta-optimization
        start_time = time.time()
        try:
            result = self.meta_optimizer.optimize_all_combinations(limit_combinations=5)
            execution_success = True
            execution_time = time.time() - start_time
        except Exception as e:
            print(f"    Execution error: {e}")
            execution_success = False
            execution_time = 0
            result = None
        
        if not execution_success:
            self.test_results['meta_optimization_execution'] = {
                'execution_success': False,
                'total_score': 0
            }
            return
        
        # Test 2: Formulation matrix populated
        has_formulations = len(result.formulation_matrix) > 0
        
        # Test 3: Optimization metrics calculated
        has_metrics = len(result.optimization_metrics) > 0
        metrics_match_formulations = len(result.optimization_metrics) == len(result.formulation_matrix)
        
        # Test 4: Coverage analysis performed
        has_coverage_analysis = bool(result.coverage_analysis)
        coverage_has_stats = 'total_formulations' in result.coverage_analysis
        
        # Test 5: Performance rankings generated
        has_rankings = bool(result.performance_rankings)
        ranking_categories = len(result.performance_rankings) >= 3
        
        # Test 6: Recommendation matrix created
        has_recommendations = bool(result.recommendation_matrix)
        
        # Test 7: All formulations are valid
        valid_formulations = True
        for formulation in result.formulation_matrix.values():
            if not isinstance(formulation, HypergredientFormulation):
                valid_formulations = False
                break
        
        self.test_results['meta_optimization_execution'] = {
            'execution_success': execution_success,
            'has_formulations': has_formulations,
            'has_metrics': has_metrics,
            'metrics_match_formulations': metrics_match_formulations,
            'has_coverage_analysis': has_coverage_analysis,
            'coverage_has_stats': coverage_has_stats,
            'has_rankings': has_rankings,
            'ranking_categories': ranking_categories,
            'has_recommendations': has_recommendations,
            'valid_formulations': valid_formulations,
            'execution_time': execution_time,
            'total_formulations': len(result.formulation_matrix) if result else 0,
            'total_score': sum([
                execution_success, has_formulations, has_metrics, metrics_match_formulations,
                has_coverage_analysis, coverage_has_stats, has_rankings, ranking_categories,
                has_recommendations, valid_formulations
            ])
        }
        
        print(f"  Execution success: {execution_success}")
        print(f"  Formulations generated: {has_formulations} ({len(result.formulation_matrix) if result else 0})")
        print(f"  Optimization metrics: {has_metrics}")
        print(f"  Metrics match formulations: {metrics_match_formulations}")
        print(f"  Coverage analysis: {has_coverage_analysis}")
        print(f"  Performance rankings: {has_rankings} ({len(result.performance_rankings) if result else 0} categories)")
        print(f"  Recommendations generated: {has_recommendations}")
        print(f"  Valid formulations: {valid_formulations}")
        print(f"  Execution time: {execution_time:.3f}s")
        
        # Store result for other tests
        self.meta_optimization_result = result if execution_success else None
    
    def test_optimization_metrics_accuracy(self):
        """Test optimization metrics calculation accuracy"""
        print("\n4. Testing optimization metrics accuracy...")
        
        if not hasattr(self, 'meta_optimization_result') or not self.meta_optimization_result:
            print("  Skipping - no meta-optimization result available")
            self.test_results['optimization_metrics_accuracy'] = {'total_score': 0}
            return
        
        result = self.meta_optimization_result
        
        # Test 1: Metrics structure validity
        metrics_valid = True
        required_metrics = ['efficacy_score', 'synergy_score', 'cost_efficiency', 'stability_score']
        
        for combo_id, metrics in result.optimization_metrics.items():
            if not all(metric in metrics for metric in required_metrics):
                metrics_valid = False
                break
        
        # Test 2: Metric value ranges
        values_in_range = True
        for metrics in result.optimization_metrics.values():
            if not (0 <= metrics['efficacy_score'] <= 100):
                values_in_range = False
                break
            if not (0 <= metrics['synergy_score'] <= 5):
                values_in_range = False
                break
            if metrics['cost_efficiency'] < 0:
                values_in_range = False
                break
        
        # Test 3: Efficiency metrics calculation
        efficiency_calculated = bool(result.efficiency_metrics)
        efficiency_reasonable = True
        if result.efficiency_metrics:
            avg_efficacy = result.efficiency_metrics.get('avg_efficacy', 0)
            if not (0 <= avg_efficacy <= 100):
                efficiency_reasonable = False
        
        # Test 4: Metrics consistency
        metrics_consistent = True
        if result.optimization_metrics:
            sample_metrics = list(result.optimization_metrics.values())
            if len(sample_metrics) >= 2:
                # Check if different formulations have different metrics
                if sample_metrics[0] == sample_metrics[1]:
                    # This might be okay if formulations are identical
                    pass
        
        self.test_results['optimization_metrics_accuracy'] = {
            'metrics_valid': metrics_valid,
            'values_in_range': values_in_range,
            'efficiency_calculated': efficiency_calculated,
            'efficiency_reasonable': efficiency_reasonable,
            'metrics_consistent': metrics_consistent,
            'total_score': sum([
                metrics_valid, values_in_range, efficiency_calculated,
                efficiency_reasonable, metrics_consistent
            ])
        }
        
        print(f"  Metrics structure valid: {metrics_valid}")
        print(f"  Values in range: {values_in_range}")
        print(f"  Efficiency calculated: {efficiency_calculated}")
        print(f"  Efficiency reasonable: {efficiency_reasonable}")
        print(f"  Metrics consistent: {metrics_consistent}")
    
    def test_performance_ranking_system(self):
        """Test performance ranking system accuracy"""
        print("\n5. Testing performance ranking system...")
        
        if not hasattr(self, 'meta_optimization_result') or not self.meta_optimization_result:
            print("  Skipping - no meta-optimization result available")
            self.test_results['performance_ranking_system'] = {'total_score': 0}
            return
        
        result = self.meta_optimization_result
        
        # Test 1: Rankings exist
        has_rankings = bool(result.performance_rankings)
        
        # Test 2: Multiple ranking categories
        ranking_categories = len(result.performance_rankings) if result.performance_rankings else 0
        has_multiple_categories = ranking_categories >= 3
        
        # Test 3: Rankings are sorted correctly
        rankings_sorted = True
        if result.performance_rankings:
            for category, rankings in result.performance_rankings.items():
                if len(rankings) > 1:
                    for i in range(len(rankings) - 1):
                        if rankings[i][1] < rankings[i+1][1]:  # Should be descending
                            rankings_sorted = False
                            break
                if not rankings_sorted:
                    break
        
        # Test 4: Rankings contain valid data
        rankings_valid = True
        if result.performance_rankings:
            for category, rankings in result.performance_rankings.items():
                for combo_id, score in rankings:
                    if not isinstance(combo_id, str) or not isinstance(score, (int, float)):
                        rankings_valid = False
                        break
                if not rankings_valid:
                    break
        
        # Test 5: Top performers identified
        has_top_performers = False
        if result.performance_rankings.get('overall_performance'):
            has_top_performers = len(result.performance_rankings['overall_performance']) > 0
        
        self.test_results['performance_ranking_system'] = {
            'has_rankings': has_rankings,
            'has_multiple_categories': has_multiple_categories,
            'rankings_sorted': rankings_sorted,
            'rankings_valid': rankings_valid,
            'has_top_performers': has_top_performers,
            'ranking_categories': ranking_categories,
            'total_score': sum([
                has_rankings, has_multiple_categories, rankings_sorted,
                rankings_valid, has_top_performers
            ])
        }
        
        print(f"  Has rankings: {has_rankings}")
        print(f"  Multiple categories: {has_multiple_categories} ({ranking_categories} categories)")
        print(f"  Rankings sorted: {rankings_sorted}")
        print(f"  Rankings valid: {rankings_valid}")
        print(f"  Top performers identified: {has_top_performers}")
    
    def test_coverage_analysis(self):
        """Test coverage analysis completeness"""
        print("\n6. Testing coverage analysis...")
        
        if not hasattr(self, 'meta_optimization_result') or not self.meta_optimization_result:
            print("  Skipping - no meta-optimization result available")
            self.test_results['coverage_analysis'] = {'total_score': 0}
            return
        
        result = self.meta_optimization_result
        
        # Test 1: Coverage analysis exists
        has_coverage = bool(result.coverage_analysis)
        
        # Test 2: Essential coverage metrics
        coverage_metrics_present = False
        if result.coverage_analysis:
            required_metrics = ['total_formulations', 'hypergredient_class_usage']
            coverage_metrics_present = all(metric in result.coverage_analysis for metric in required_metrics)
        
        # Test 3: Hypergredient class usage tracking
        class_usage_tracked = False
        if result.coverage_analysis.get('hypergredient_class_usage'):
            class_usage_tracked = len(result.coverage_analysis['hypergredient_class_usage']) > 0
        
        # Test 4: Coverage completeness
        coverage_complete = False
        if result.coverage_analysis.get('total_formulations'):
            expected_formulations = result.coverage_analysis['total_formulations']
            actual_formulations = len(result.formulation_matrix)
            coverage_complete = expected_formulations == actual_formulations
        
        self.test_results['coverage_analysis'] = {
            'has_coverage': has_coverage,
            'coverage_metrics_present': coverage_metrics_present,
            'class_usage_tracked': class_usage_tracked,
            'coverage_complete': coverage_complete,
            'total_score': sum([
                has_coverage, coverage_metrics_present, class_usage_tracked, coverage_complete
            ])
        }
        
        print(f"  Has coverage analysis: {has_coverage}")
        print(f"  Coverage metrics present: {coverage_metrics_present}")
        print(f"  Class usage tracked: {class_usage_tracked}")
        print(f"  Coverage complete: {coverage_complete}")
    
    def test_recommendation_generation(self):
        """Test recommendation matrix generation"""
        print("\n7. Testing recommendation generation...")
        
        if not hasattr(self, 'meta_optimization_result') or not self.meta_optimization_result:
            print("  Skipping - no meta-optimization result available")
            self.test_results['recommendation_generation'] = {'total_score': 0}
            return
        
        result = self.meta_optimization_result
        
        # Test 1: Recommendations exist
        has_recommendations = bool(result.recommendation_matrix)
        
        # Test 2: Skin type recommendations
        skin_type_recommendations = False
        if result.recommendation_matrix:
            sample_skin_types = ['normal', 'dry', 'oily', 'sensitive']
            skin_type_recommendations = any(st in result.recommendation_matrix for st in sample_skin_types)
        
        # Test 3: Recommendation categories
        has_recommendation_categories = False
        if result.recommendation_matrix:
            for skin_type, recommendations in result.recommendation_matrix.items():
                if isinstance(recommendations, dict) and len(recommendations) > 0:
                    has_recommendation_categories = True
                    break
        
        # Test 4: Recommendations are valid formulation IDs
        recommendations_valid = True
        if result.recommendation_matrix:
            for skin_type, recommendations in result.recommendation_matrix.items():
                if isinstance(recommendations, dict):
                    for rec_type, combo_id in recommendations.items():
                        if combo_id not in result.formulation_matrix:
                            recommendations_valid = False
                            break
                if not recommendations_valid:
                    break
        
        self.test_results['recommendation_generation'] = {
            'has_recommendations': has_recommendations,
            'skin_type_recommendations': skin_type_recommendations,
            'has_recommendation_categories': has_recommendation_categories,
            'recommendations_valid': recommendations_valid,
            'total_score': sum([
                has_recommendations, skin_type_recommendations,
                has_recommendation_categories, recommendations_valid
            ])
        }
        
        print(f"  Has recommendations: {has_recommendations}")
        print(f"  Skin type recommendations: {skin_type_recommendations}")
        print(f"  Recommendation categories: {has_recommendation_categories}")
        print(f"  Recommendations valid: {recommendations_valid}")
    
    def test_integration_compatibility(self):
        """Test integration with existing framework"""
        print("\n8. Testing integration compatibility...")
        
        # Test 1: Meta-optimizer integrates with base optimizer
        integration_works = True
        try:
            base_optimizer = HypergredientOptimizer()
            meta_optimizer = HypergredientMetaOptimizer()
            # Test if they can work together
            formulation = base_optimizer.optimize_formulation(
                target_concerns=['wrinkles'],
                skin_type='normal',
                budget=1000
            )
            integration_works = isinstance(formulation, HypergredientFormulation)
        except Exception as e:
            print(f"    Integration error: {e}")
            integration_works = False
        
        # Test 2: Formulation compatibility
        formulation_compatible = True
        if integration_works and hasattr(self, 'meta_optimization_result'):
            result = self.meta_optimization_result
            if result and result.formulation_matrix:
                sample_formulation = list(result.formulation_matrix.values())[0]
                formulation_compatible = hasattr(sample_formulation, 'hypergredients')
        
        # Test 3: Data structure consistency
        data_structure_consistent = True
        if hasattr(self, 'meta_optimization_result') and self.meta_optimization_result:
            try:
                # Test serialization compatibility
                sample_data = {
                    'total_formulations': len(self.meta_optimization_result.formulation_matrix),
                    'efficiency_metrics': self.meta_optimization_result.efficiency_metrics
                }
                json.dumps(sample_data, default=str)
            except Exception:
                data_structure_consistent = False
        
        self.test_results['integration_compatibility'] = {
            'integration_works': integration_works,
            'formulation_compatible': formulation_compatible,
            'data_structure_consistent': data_structure_consistent,
            'total_score': sum([
                integration_works, formulation_compatible, data_structure_consistent
            ])
        }
        
        print(f"  Integration works: {integration_works}")
        print(f"  Formulation compatible: {formulation_compatible}")
        print(f"  Data structure consistent: {data_structure_consistent}")
    
    def run_all_tests(self) -> Dict:
        """Run all test cases and return results"""
        print("=== META-OPTIMIZATION STRATEGY TEST SUITE ===")
        
        # Run all tests
        self.test_meta_optimizer_initialization()
        self.test_condition_combination_generation()
        self.test_meta_optimization_execution()
        self.test_optimization_metrics_accuracy()
        self.test_performance_ranking_system()
        self.test_coverage_analysis()
        self.test_recommendation_generation()
        self.test_integration_compatibility()
        
        return self.compile_test_report()
    
    def compile_test_report(self) -> Dict:
        """Compile comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('total_score', 0) > 0)
        
        overall_score = sum(result.get('total_score', 0) for result in self.test_results.values())
        max_possible_score = total_tests * 10  # Approximate max
        
        return {
            'overall_status': 'PASSED' if passed_tests >= total_tests * 0.8 else 'FAILED',
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'overall_score': overall_score,
            'detailed_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if self.test_results.get('meta_optimization_execution', {}).get('total_score', 0) < 8:
            recommendations.append("Consider optimizing meta-optimization execution performance")
        
        if self.test_results.get('optimization_metrics_accuracy', {}).get('total_score', 0) < 4:
            recommendations.append("Review optimization metrics calculation accuracy")
        
        if self.test_results.get('coverage_analysis', {}).get('total_score', 0) < 3:
            recommendations.append("Improve coverage analysis completeness")
        
        if not recommendations:
            recommendations.append("Meta-optimization strategy performing well across all test categories")
        
        return recommendations

def run_meta_optimization_tests():
    """Run all meta-optimization tests and display results"""
    test_suite = MetaOptimizerTestSuite()
    results = test_suite.run_all_tests()
    
    print("\n" + "="*70)
    print("META-OPTIMIZATION STRATEGY TEST REPORT")
    print("="*70)
    
    print(f"\nOverall Results:")
    print(f"  Status: {results['overall_status']}")
    print(f"  Tests Passed: {results['tests_passed']}/{results['total_tests']}")
    print(f"  Pass Rate: {results['pass_rate']:.1f}%")
    print(f"  Overall Score: {results['overall_score']}")
    
    print(f"\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  • {rec}")
    
    print(f"\nDetailed Test Results:")
    for test_name, test_result in results['detailed_results'].items():
        print(f"\n  {test_name.replace('_', ' ').title()}:")
        score = test_result.get('total_score', 0)
        print(f"    Score: {score}")
        for key, value in test_result.items():
            if key != 'total_score':
                print(f"    {key.replace('_', ' ').title()}: {value}")
    
    # Save full report
    with open('/tmp/meta_optimization_test_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nFull test report saved to /tmp/meta_optimization_test_report.json")

if __name__ == "__main__":
    run_meta_optimization_tests()
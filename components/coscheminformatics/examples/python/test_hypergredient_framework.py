#!/usr/bin/env python3
"""
test_hypergredient_framework.py

Comprehensive Test Suite for Hypergredient Framework

This module provides comprehensive test cases for validating the hypergredient framework:
1. Hypergredient database functionality
2. Interaction matrix calculations 
3. Formulation optimization accuracy
4. Compatibility checking
5. Performance metrics
6. Integration with existing systems
"""

import time
import random
import json
from typing import Dict, List, Any

# Import hypergredient framework
try:
    from hypergredient_framework import (
        HypergredientClass, HypergredientDatabase, HypergredientInteractionMatrix,
        HypergredientOptimizer, HypergredientCompatibilityChecker, HypergredientFormulation
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hypergredient_framework import (
        HypergredientClass, HypergredientDatabase, HypergredientInteractionMatrix,
        HypergredientOptimizer, HypergredientCompatibilityChecker, HypergredientFormulation
    )

class HypergredientTestSuite:
    """Comprehensive test suite for hypergredient framework"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
    def run_all_tests(self) -> Dict:
        """Run all test cases and return results"""
        print("=== HYPERGREDIENT FRAMEWORK TEST SUITE ===")
        
        # Test 1: Database functionality
        print("\n1. Testing hypergredient database...")
        self.test_hypergredient_database()
        
        # Test 2: Interaction matrix
        print("\n2. Testing interaction matrix...")
        self.test_interaction_matrix()
        
        # Test 3: Formulation optimization
        print("\n3. Testing formulation optimization...")
        self.test_formulation_optimization()
        
        # Test 4: Compatibility checking
        print("\n4. Testing compatibility checker...")
        self.test_compatibility_checker()
        
        # Test 5: Concern mapping
        print("\n5. Testing concern mapping...")
        self.test_concern_mapping()
        
        # Test 6: Performance benchmarks
        print("\n6. Running performance benchmarks...")
        self.benchmark_performance()
        
        # Test 7: Integration with existing systems
        print("\n7. Testing integration...")
        self.test_integration()
        
        print("\n=== HYPERGREDIENT TEST SUITE COMPLETE ===")
        return self.compile_test_report()
    
    def test_hypergredient_database(self):
        """Test hypergredient database functionality"""
        database = HypergredientDatabase()
        
        # Test 1: Database initialization
        has_all_classes = all(
            database.get_ingredients_by_class(hg_class) 
            for hg_class in HypergredientClass
        )
        
        # Test 2: Ingredient retrieval
        ct_ingredients = database.get_ingredients_by_class(HypergredientClass.CT)
        ct_count_valid = len(ct_ingredients) >= 3  # Should have multiple CT agents
        
        # Test 3: Ingredient search
        retinol = database.find_ingredient_by_name("Retinol")
        bakuchiol = database.find_ingredient_by_name("Bakuchiol")
        search_works = retinol is not None and bakuchiol is not None
        
        # Test 4: Compatibility finding
        if retinol:
            compatible = database.get_compatible_ingredients(retinol)
            compatibility_works = len(compatible) > 0
        else:
            compatibility_works = False
        
        # Test 5: Data integrity
        data_valid = True
        total_ingredients = 0
        for hg_class in HypergredientClass:
            ingredients = database.get_ingredients_by_class(hg_class)
            total_ingredients += len(ingredients)
            for ingredient in ingredients:
                # Check required fields
                if not all([
                    ingredient.name,
                    ingredient.inci_name, 
                    ingredient.potency >= 1 and ingredient.potency <= 10,
                    ingredient.safety_score >= 1 and ingredient.safety_score <= 10,
                    ingredient.cost_per_gram > 0,
                    ingredient.bioavailability >= 0 and ingredient.bioavailability <= 100
                ]):
                    data_valid = False
                    break
        
        self.test_results['hypergredient_database'] = {
            'database_initialized': has_all_classes,
            'ingredient_retrieval_works': ct_count_valid,
            'ingredient_search_works': search_works,
            'compatibility_finding_works': compatibility_works,
            'data_integrity_valid': data_valid,
            'total_ingredients': total_ingredients,
            'total_score': sum([has_all_classes, ct_count_valid, search_works, 
                               compatibility_works, data_valid])
        }
        
        print(f"  Database initialized: {has_all_classes}")
        print(f"  Ingredient retrieval: {ct_count_valid}")
        print(f"  Ingredient search: {search_works}")
        print(f"  Compatibility finding: {compatibility_works}")
        print(f"  Data integrity: {data_valid}")
        print(f"  Total ingredients: {total_ingredients}")
    
    def test_interaction_matrix(self):
        """Test interaction matrix functionality"""
        matrix = HypergredientInteractionMatrix()
        
        # Test 1: Interaction coefficient retrieval
        ct_cs_coefficient = matrix.get_interaction_coefficient(
            HypergredientClass.CT, HypergredientClass.CS
        )
        coefficient_valid = ct_cs_coefficient > 0  # Should have positive synergy
        
        # Test 2: Bidirectional consistency
        cs_ct_coefficient = matrix.get_interaction_coefficient(
            HypergredientClass.CS, HypergredientClass.CT
        )
        bidirectional_consistent = ct_cs_coefficient == cs_ct_coefficient
        
        # Test 3: Default values
        uncommon_pair_coefficient = matrix.get_interaction_coefficient(
            HypergredientClass.CT, HypergredientClass.MB  # Not explicitly defined
        )
        default_works = uncommon_pair_coefficient == 1.0
        
        # Test 4: Network synergy calculation
        test_formulation = {
            HypergredientClass.CS: [None, None],  # 2 ingredients
            HypergredientClass.AO: [None],        # 1 ingredient  
            HypergredientClass.HY: [None]         # 1 ingredient
        }
        network_synergy = matrix.calculate_network_synergy(test_formulation)
        network_calculation_works = network_synergy > 0
        
        # Test 5: Expected synergies
        expected_synergies = [
            (HypergredientClass.CS, HypergredientClass.AO, 2.0),  # Strong synergy
            (HypergredientClass.BR, HypergredientClass.HY, 2.5),  # Excellent synergy
            (HypergredientClass.SE, HypergredientClass.CT, 0.6),  # Potential irritation
        ]
        
        synergy_accuracy = 0
        for class1, class2, expected in expected_synergies:
            actual = matrix.get_interaction_coefficient(class1, class2)
            if abs(actual - expected) < 0.1:
                synergy_accuracy += 1
        synergy_accurate = synergy_accuracy == len(expected_synergies)
        
        self.test_results['interaction_matrix'] = {
            'coefficient_retrieval_works': coefficient_valid,
            'bidirectional_consistent': bidirectional_consistent,
            'default_values_work': default_works,
            'network_calculation_works': network_calculation_works,
            'synergy_values_accurate': synergy_accurate,
            'network_synergy_value': network_synergy,
            'total_score': sum([coefficient_valid, bidirectional_consistent, default_works,
                               network_calculation_works, synergy_accurate])
        }
        
        print(f"  Coefficient retrieval: {coefficient_valid}")
        print(f"  Bidirectional consistency: {bidirectional_consistent}")
        print(f"  Default values: {default_works}")
        print(f"  Network calculation: {network_calculation_works}")
        print(f"  Synergy accuracy: {synergy_accurate}")
        print(f"  Network synergy value: {network_synergy:.2f}")
    
    def test_formulation_optimization(self):
        """Test formulation optimization accuracy"""
        optimizer = HypergredientOptimizer()
        
        # Test 1: Anti-aging formulation
        start_time = time.time()
        anti_aging = optimizer.optimize_formulation(
            target_concerns=['wrinkles', 'firmness'],
            skin_type='normal',
            budget=1000,
            preferences=['stable']
        )
        optimization_time = time.time() - start_time
        
        optimization_completed = isinstance(anti_aging, HypergredientFormulation)
        has_ingredients = len(anti_aging.hypergredients) > 0
        
        # Test 2: Budget compliance
        budget_compliant = anti_aging.cost_total <= 1000
        
        # Test 3: Synergy calculation
        synergy_calculated = anti_aging.synergy_score > 0
        
        # Test 4: Efficacy prediction
        efficacy_realistic = 0 <= anti_aging.efficacy_prediction <= 100
        
        # Test 5: Multiple concerns handling
        multi_concern = optimizer.optimize_formulation(
            target_concerns=['wrinkles', 'brightness', 'dryness'],
            skin_type='sensitive',
            budget=800,
            preferences=['gentle']
        )
        
        multi_concern_works = len(multi_concern.hypergredients) >= 2  # Should address multiple concerns
        
        # Test 6: Skin type adaptation
        sensitive_ingredients = []
        for hg_class, data in multi_concern.hypergredients.items():
            for ing_data in data['ingredients']:
                sensitive_ingredients.append(ing_data['ingredient'])
        
        skin_type_adapted = all(ing.safety_score >= 7.0 for ing in sensitive_ingredients)
        
        self.test_results['formulation_optimization'] = {
            'optimization_completed': optimization_completed,
            'has_ingredients': has_ingredients,  
            'budget_compliant': budget_compliant,
            'synergy_calculated': synergy_calculated,
            'efficacy_realistic': efficacy_realistic,
            'multi_concern_works': multi_concern_works,
            'skin_type_adapted': skin_type_adapted,
            'optimization_time': optimization_time,
            'total_score': sum([optimization_completed, has_ingredients, budget_compliant,
                               synergy_calculated, efficacy_realistic, multi_concern_works,
                               skin_type_adapted])
        }
        
        print(f"  Optimization completed: {optimization_completed}")
        print(f"  Has ingredients: {has_ingredients}")
        print(f"  Budget compliant: {budget_compliant}")
        print(f"  Synergy calculated: {synergy_calculated}")
        print(f"  Efficacy realistic: {efficacy_realistic}")
        print(f"  Multi-concern works: {multi_concern_works}")
        print(f"  Skin type adapted: {skin_type_adapted}")
        print(f"  Optimization time: {optimization_time:.3f}s")
    
    def test_compatibility_checker(self):
        """Test compatibility checker functionality"""
        database = HypergredientDatabase()
        matrix = HypergredientInteractionMatrix()
        checker = HypergredientCompatibilityChecker(database, matrix)
        
        # Test 1: Known incompatible pair
        retinol = database.find_ingredient_by_name("Retinol")
        vitamin_c = database.find_ingredient_by_name("Vitamin C (L-AA)")
        
        if retinol and vitamin_c:
            incompatible_result = checker.check_compatibility(retinol, vitamin_c)
            detects_incompatibility = incompatible_result['compatibility_score'] < 0.8
            has_recommendations = len(incompatible_result['recommendations']) > 0
        else:
            detects_incompatibility = False
            has_recommendations = False
        
        # Test 2: pH overlap calculation
        if retinol and vitamin_c:
            ph_overlap = incompatible_result['ph_overlap']
            ph_calculation_works = 0 <= ph_overlap <= 1.0
        else:
            ph_calculation_works = False
        
        # Test 3: Compatible pair
        hyaluronic = database.find_ingredient_by_name("Hyaluronic Acid (High MW)")
        glycerin = database.find_ingredient_by_name("Glycerin")
        
        if hyaluronic and glycerin:
            compatible_result = checker.check_compatibility(hyaluronic, glycerin)
            detects_compatibility = compatible_result['compatibility_score'] > 0.6
        else:
            detects_compatibility = False
        
        # Test 4: Stability assessment
        if retinol and vitamin_c:
            stability_impact = incompatible_result['stability_impact']
            stability_assessment_works = 0 <= stability_impact <= 1.0
        else:
            stability_assessment_works = False
        
        # Test 5: Alternative suggestions
        provides_alternatives = len(incompatible_result.get('alternatives', [])) > 0 if retinol and vitamin_c else False
        
        self.test_results['compatibility_checker'] = {
            'detects_incompatibility': detects_incompatibility,
            'has_recommendations': has_recommendations,
            'ph_calculation_works': ph_calculation_works,
            'detects_compatibility': detects_compatibility,
            'stability_assessment_works': stability_assessment_works,
            'provides_alternatives': provides_alternatives,
            'total_score': sum([detects_incompatibility, has_recommendations, ph_calculation_works,
                               detects_compatibility, stability_assessment_works, provides_alternatives])
        }
        
        print(f"  Detects incompatibility: {detects_incompatibility}")
        print(f"  Has recommendations: {has_recommendations}")
        print(f"  pH calculation works: {ph_calculation_works}")
        print(f"  Detects compatibility: {detects_compatibility}")
        print(f"  Stability assessment: {stability_assessment_works}")
        print(f"  Provides alternatives: {provides_alternatives}")
    
    def test_concern_mapping(self):
        """Test concern to hypergredient mapping"""
        optimizer = HypergredientOptimizer()
        
        # Test various concern mappings
        test_concerns = {
            'wrinkles': HypergredientClass.CT,
            'firmness': HypergredientClass.CS,
            'brightness': HypergredientClass.ML,
            'dryness': HypergredientClass.HY,
            'sensitivity': HypergredientClass.AI,
            'acne': HypergredientClass.SE,
            'dullness': HypergredientClass.AO
        }
        
        mapping_accuracy = 0
        for concern, expected_class in test_concerns.items():
            mapped_class = optimizer.map_concern_to_hypergredient(concern)
            if mapped_class == expected_class:
                mapping_accuracy += 1
        
        mapping_works = mapping_accuracy == len(test_concerns)
        
        # Test unknown concern handling
        unknown_mapped = optimizer.map_concern_to_hypergredient('unknown_concern')
        handles_unknown = unknown_mapped in HypergredientClass
        
        self.test_results['concern_mapping'] = {
            'mapping_accuracy': mapping_accuracy,
            'total_concerns_tested': len(test_concerns),
            'mapping_works': mapping_works,
            'handles_unknown_concerns': handles_unknown,
            'total_score': mapping_accuracy + (1 if handles_unknown else 0)
        }
        
        print(f"  Mapping accuracy: {mapping_accuracy}/{len(test_concerns)}")
        print(f"  All mappings correct: {mapping_works}")
        print(f"  Handles unknown concerns: {handles_unknown}")
    
    def benchmark_performance(self):
        """Benchmark hypergredient framework performance"""
        database = HypergredientDatabase()
        optimizer = HypergredientOptimizer()
        matrix = HypergredientInteractionMatrix()
        
        # Benchmark 1: Database ingredient search
        start_time = time.time()
        for _ in range(100):
            database.find_ingredient_by_name("Retinol")
        search_time = (time.time() - start_time) / 100
        
        # Benchmark 2: Formulation optimization
        start_time = time.time()
        for _ in range(10):
            optimizer.optimize_formulation(
                target_concerns=['wrinkles'],
                skin_type='normal',
                budget=500
            )
        optimization_time = (time.time() - start_time) / 10
        
        # Benchmark 3: Interaction matrix calculations
        start_time = time.time()
        for _ in range(1000):
            matrix.get_interaction_coefficient(HypergredientClass.CT, HypergredientClass.CS)
        interaction_time = (time.time() - start_time) / 1000
        
        # Benchmark 4: Network synergy calculation
        test_formulation = {
            HypergredientClass.CS: [None] * 3,
            HypergredientClass.AO: [None] * 2,  
            HypergredientClass.HY: [None] * 2
        }
        start_time = time.time()
        for _ in range(100):
            matrix.calculate_network_synergy(test_formulation)
        synergy_time = (time.time() - start_time) / 100
        
        self.performance_metrics = {
            'ingredient_search_time_ms': search_time * 1000,
            'formulation_optimization_time_ms': optimization_time * 1000,
            'interaction_calculation_time_ms': interaction_time * 1000,
            'network_synergy_time_ms': synergy_time * 1000
        }
        
        print(f"  Ingredient search: {search_time*1000:.2f} ms per search")
        print(f"  Formulation optimization: {optimization_time*1000:.0f} ms per formulation")
        print(f"  Interaction calculation: {interaction_time*1000:.3f} ms per calculation")
        print(f"  Network synergy: {synergy_time*1000:.2f} ms per calculation")
    
    def test_integration(self):
        """Test integration with existing systems"""
        optimizer = HypergredientOptimizer()
        
        # Test 1: Integration with INCI reducer (should not break)
        try:
            formulation = optimizer.optimize_formulation(
                target_concerns=['wrinkles', 'dryness'],
                skin_type='normal',
                budget=800
            )
            inci_integration_works = True
        except Exception as e:
            print(f"    INCI integration error: {e}")
            inci_integration_works = False
        
        # Test 2: Formulation data structure compatibility
        if inci_integration_works:
            has_required_fields = all([
                hasattr(formulation, 'id'),
                hasattr(formulation, 'hypergredients'),
                hasattr(formulation, 'synergy_score'),
                hasattr(formulation, 'efficacy_prediction'),
                hasattr(formulation, 'cost_total')
            ])
        else:
            has_required_fields = False
        
        # Test 3: Serialization capability
        try:
            if inci_integration_works:
                # Convert to dict for JSON serialization test
                formulation_dict = {
                    'id': formulation.id,
                    'target_concerns': formulation.target_concerns,
                    'skin_type': formulation.skin_type,
                    'budget': formulation.budget,
                    'synergy_score': formulation.synergy_score,
                    'efficacy_prediction': formulation.efficacy_prediction,
                    'cost_total': formulation.cost_total,
                    'stability_months': formulation.stability_months
                }
                json.dumps(formulation_dict)  # Test JSON serialization
                serialization_works = True
            else:
                serialization_works = False
        except Exception:
            serialization_works = False
        
        self.test_results['integration'] = {
            'inci_integration_works': inci_integration_works,
            'has_required_fields': has_required_fields,
            'serialization_works': serialization_works,
            'total_score': sum([inci_integration_works, has_required_fields, serialization_works])
        }
        
        print(f"  INCI integration: {inci_integration_works}")
        print(f"  Required fields: {has_required_fields}")
        print(f"  Serialization: {serialization_works}")
    
    def compile_test_report(self) -> Dict:
        """Compile comprehensive test report"""
        total_tests = 0
        passed_tests = 0
        
        report = {
            'summary': {},
            'detailed_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'recommendations': []
        }
        
        # Calculate overall scores
        for test_name, results in self.test_results.items():
            if 'total_score' in results:
                total_tests += 1
                max_possible = len([k for k in results.keys() if k != 'total_score' and isinstance(results[k], bool)])
                if max_possible > 0:
                    pass_threshold = max_possible * 0.7  # 70% pass rate
                    if results['total_score'] >= pass_threshold:
                        passed_tests += 1
        
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        report['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': pass_rate,
            'overall_status': 'PASSED' if pass_rate >= 0.8 else 'NEEDS_IMPROVEMENT'
        }
        
        # Generate recommendations
        if pass_rate < 1.0:
            report['recommendations'].append("Some test cases failed - review detailed results")
        
        if self.performance_metrics.get('formulation_optimization_time_ms', 0) > 1000:
            report['recommendations'].append("Formulation optimization performance could be improved")
        
        if not self.test_results.get('compatibility_checker', {}).get('detects_incompatibility', False):
            report['recommendations'].append("Compatibility detection may need improvement")
        
        return report

def run_hypergredient_tests():
    """Run all hypergredient framework tests and display results"""
    test_suite = HypergredientTestSuite()
    report = test_suite.run_all_tests()
    
    print("\n" + "="*60)
    print("HYPERGREDIENT FRAMEWORK TEST REPORT")
    print("="*60)
    
    print(f"\nOverall Results:")
    print(f"  Status: {report['summary']['overall_status']}")
    print(f"  Tests Passed: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")
    print(f"  Pass Rate: {report['summary']['pass_rate']:.1%}")
    
    print(f"\nPerformance Metrics:")
    for metric, value in report['performance_metrics'].items():
        if value < 1:
            print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    print(f"\nDetailed Test Results:")
    for test_name, results in report['detailed_results'].items():
        print(f"\n  {test_name.replace('_', ' ').title()}:")
        for key, value in results.items():
            if key != 'total_score':
                print(f"    {key.replace('_', ' ').title()}: {value}")
    
    # Save report to file
    with open('/tmp/hypergredient_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nFull test report saved to /tmp/hypergredient_test_report.json")
    
    return report

if __name__ == "__main__":
    run_hypergredient_tests()
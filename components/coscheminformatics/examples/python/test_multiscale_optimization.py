#!/usr/bin/env python3
"""
test_multiscale_optimization.py

Test Cases for Multiscale Constraint Optimization in Cosmeceutical Formulation

This module provides comprehensive test cases for validating:
1. INCI-based search space pruning accuracy
2. Attention allocation efficiency
3. Multiscale constraint satisfaction
4. Optimization accuracy and convergence
5. Regulatory compliance checking
"""

import time
import random
import math
from typing import List, Dict, Tuple
import json

# Import our modules
from inci_optimizer import INCISearchSpaceReducer, FormulationConstraint
from attention_allocation import AttentionAllocationManager
from multiscale_optimizer import MultiscaleConstraintOptimizer, OptimizationObjective, ObjectiveType, ScaleConstraint, OptimizationScale

class TestSuite:
    """Comprehensive test suite for multiscale optimization system"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
    
    def run_all_tests(self) -> Dict:
        """Run all test cases and return results"""
        print("=== Running Multiscale Optimization Test Suite ===")
        
        # Test 1: INCI Search Space Reduction
        print("\n1. Testing INCI-based search space reduction...")
        self.test_inci_search_space_reduction()
        
        # Test 2: Attention Allocation
        print("\n2. Testing attention allocation mechanisms...")
        self.test_attention_allocation()
        
        # Test 3: Multiscale Constraint Satisfaction
        print("\n3. Testing multiscale constraint satisfaction...")
        self.test_multiscale_constraints()
        
        # Test 4: Optimization Accuracy
        print("\n4. Testing optimization accuracy...")
        self.test_optimization_accuracy()
        
        # Test 5: Regulatory Compliance
        print("\n5. Testing regulatory compliance...")
        self.test_regulatory_compliance()
        
        # Test 6: Performance Benchmarking
        print("\n6. Running performance benchmarks...")
        self.benchmark_performance()
        
        # Test 7: Integration Test
        print("\n7. Running integration test...")
        self.test_full_integration()
        
        print("\n=== Test Suite Complete ===")
        return self.compile_test_report()
    
    def test_inci_search_space_reduction(self):
        """Test INCI-based search space pruning"""
        reducer = INCISearchSpaceReducer()
        
        # Test Case 1: INCI parsing accuracy
        test_inci = "Aqua, Sodium Hyaluronate, Niacinamide, Glycerin, Phenoxyethanol"
        parsed = reducer.parse_inci_list(test_inci)
        
        expected_ingredients = ['aqua', 'sodium hyaluronate', 'niacinamide', 'glycerin', 'phenoxyethanol']
        parsed_names = [ing['inci_name'].lower() for ing in parsed]
        
        parsing_accuracy = len(set(expected_ingredients) & set(parsed_names)) / len(expected_ingredients)
        
        # Test Case 2: Concentration estimation consistency
        concentrations = reducer.estimate_concentrations(parsed)
        total_concentration = sum(concentrations.values())
        concentration_consistency = abs(total_concentration - 100.0) < 10.0  # Within 10% of 100%
        
        # Test Case 3: Subset filtering
        constraints = FormulationConstraint(max_total_actives=15.0)
        filtered = reducer.filter_formulation_space(test_inci, constraints)
        subset_filtering_works = len(filtered) > 0
        
        self.test_results['inci_reduction'] = {
            'parsing_accuracy': parsing_accuracy,
            'concentration_consistency': concentration_consistency,
            'subset_filtering_works': subset_filtering_works,
            'total_score': (parsing_accuracy + concentration_consistency + subset_filtering_works) / 3
        }
        
        print(f"  INCI parsing accuracy: {parsing_accuracy:.2%}")
        print(f"  Concentration consistency: {concentration_consistency}")
        print(f"  Subset filtering works: {subset_filtering_works}")
    
    def test_attention_allocation(self):
        """Test attention allocation mechanisms"""
        manager = AttentionAllocationManager(max_active_nodes=20)
        
        # Test Case 1: Attention value updates
        formulations = [
            {'ingredients': {'retinol': 0.5, 'hyaluronic_acid': 1.0}, 'type': 'serum'},
            {'ingredients': {'vitamin_c': 10.0, 'vitamin_e': 0.5}, 'type': 'serum'},
            {'ingredients': {'niacinamide': 5.0, 'zinc_oxide': 2.0}, 'type': 'treatment'}
        ]
        
        node_ids = []
        for form in formulations:
            node_id = manager.add_formulation_node(form)
            node_ids.append(node_id)
        
        # Simulate optimization results
        results = []
        for node_id in node_ids:
            result = {
                'efficacy': random.uniform(0.4, 0.9),
                'cost': random.uniform(20, 80),
                'safety': random.uniform(0.7, 1.0)
            }
            manager.nodes[node_id].update_from_search_result(result)
            results.append((node_id, result))
        
        # Test attention spreading
        initial_attention = [manager.nodes[nid].get_total_attention() for nid in node_ids]
        manager._spread_attention()
        final_attention = [manager.nodes[nid].get_total_attention() for nid in node_ids]
        
        attention_changed = any(abs(a - b) > 0.01 for a, b in zip(initial_attention, final_attention))
        
        # Test resource allocation
        optimization_tasks = [lambda p: {'efficacy': 0.7, 'cost': 50}]
        allocation_results = manager.allocate_computational_resources(optimization_tasks, 10.0)
        
        resource_allocation_works = len(allocation_results) > 0
        
        self.test_results['attention_allocation'] = {
            'node_creation_works': len(node_ids) == 3,
            'attention_updates_work': attention_changed,
            'resource_allocation_works': resource_allocation_works,
            'total_score': (len(node_ids) == 3) + attention_changed + resource_allocation_works
        }
        
        print(f"  Node creation: {len(node_ids) == 3}")
        print(f"  Attention spreading: {attention_changed}")
        print(f"  Resource allocation: {resource_allocation_works}")
    
    def test_multiscale_constraints(self):
        """Test multiscale constraint satisfaction"""
        optimizer = MultiscaleConstraintOptimizer()
        
        # Add test constraints
        molecular_constraint = ScaleConstraint(
            scale=OptimizationScale.MOLECULAR,
            parameter="average_molecular_weight",
            max_value=500.0,
            weight=1.0
        )
        
        cellular_constraint = ScaleConstraint(
            scale=OptimizationScale.CELLULAR,
            parameter="fibroblast_stimulation",
            min_value=0.3,
            weight=0.8
        )
        
        optimizer.add_scale_constraint(molecular_constraint)
        optimizer.add_scale_constraint(cellular_constraint)
        
        # Test constraint evaluation
        from multiscale_optimizer import FormulationCandidate
        
        test_candidate = FormulationCandidate(
            id="test_candidate",
            ingredients={'retinol': 0.5, 'hyaluronic_acid': 1.0, 'glycerin': 3.0},
            formulation_type='serum'
        )
        
        optimizer._calculate_multiscale_properties(test_candidate)
        
        # Check if multiscale properties were calculated
        has_molecular_props = len(test_candidate.molecular_properties) > 0
        has_cellular_effects = len(test_candidate.cellular_effects) > 0
        has_tissue_responses = len(test_candidate.tissue_responses) > 0
        has_organ_outcomes = len(test_candidate.organ_outcomes) > 0
        
        multiscale_calculation_works = (has_molecular_props and has_cellular_effects and 
                                       has_tissue_responses and has_organ_outcomes)
        
        self.test_results['multiscale_constraints'] = {
            'molecular_properties_calculated': has_molecular_props,
            'cellular_effects_calculated': has_cellular_effects,
            'tissue_responses_calculated': has_tissue_responses,
            'organ_outcomes_calculated': has_organ_outcomes,
            'multiscale_calculation_works': multiscale_calculation_works,
            'total_score': sum([has_molecular_props, has_cellular_effects, 
                               has_tissue_responses, has_organ_outcomes])
        }
        
        print(f"  Molecular properties: {has_molecular_props}")
        print(f"  Cellular effects: {has_cellular_effects}")
        print(f"  Tissue responses: {has_tissue_responses}")
        print(f"  Organ outcomes: {has_organ_outcomes}")
    
    def test_optimization_accuracy(self):
        """Test optimization accuracy with known good formulations"""
        optimizer = MultiscaleConstraintOptimizer()
        optimizer.population_size = 20
        optimizer.max_generations = 10  # Short for testing
        
        # Add objectives
        optimizer.add_objective(OptimizationObjective(ObjectiveType.EFFICACY, 0.6))
        optimizer.add_objective(OptimizationObjective(ObjectiveType.SAFETY, 0.4))
        
        # Test with known effective anti-aging formulation
        target_inci = "Aqua, Retinol, Hyaluronic Acid, Niacinamide, Glycerin, Phenoxyethanol"
        constraints = FormulationConstraint(
            target_ph=(5.0, 7.0),
            max_total_actives=12.0
        )
        
        # Run short optimization
        start_time = time.time()
        results = optimizer.optimize_formulation(
            target_inci=target_inci,
            base_constraints=constraints,
            target_condition="anti_aging",
            max_time_minutes=2.0  # Very short for testing
        )
        optimization_time = time.time() - start_time
        
        # Check results quality
        optimization_completed = len(results) > 0
        results_have_ingredients = all(len(r.ingredients) > 0 for r in results) if results else False
        results_regulatory_compliant = all(r.regulatory_compliance for r in results) if results else False
        
        # Check fitness improvement over generations
        fitness_improved = False
        if optimizer.optimization_history and len(optimizer.optimization_history) > 1:
            initial_fitness = optimizer.optimization_history[0]['best_fitness']
            final_fitness = optimizer.optimization_history[-1]['best_fitness']
            fitness_improved = final_fitness > initial_fitness
        
        self.test_results['optimization_accuracy'] = {
            'optimization_completed': optimization_completed,
            'results_have_ingredients': results_have_ingredients,
            'results_regulatory_compliant': results_regulatory_compliant,
            'fitness_improved': fitness_improved,
            'optimization_time': optimization_time,
            'total_score': sum([optimization_completed, results_have_ingredients, 
                               results_regulatory_compliant, fitness_improved])
        }
        
        print(f"  Optimization completed: {optimization_completed}")
        print(f"  Results have ingredients: {results_have_ingredients}")
        print(f"  Results regulatory compliant: {results_regulatory_compliant}")
        print(f"  Fitness improved: {fitness_improved}")
        print(f"  Optimization time: {optimization_time:.2f}s")
    
    def test_regulatory_compliance(self):
        """Test regulatory compliance checking"""
        reducer = INCISearchSpaceReducer()
        
        # Test Case 1: Compliant formulation
        compliant_formulation = {
            'ingredients': {
                'water': 70.0,
                'hyaluronic_acid': 1.0,
                'niacinamide': 5.0,
                'phenoxyethanol': 0.8
            }
        }
        
        compliance_result1 = reducer.check_regulatory_compliance(compliant_formulation)
        
        # Test Case 2: Non-compliant formulation (over limits)
        non_compliant_formulation = {
            'ingredients': {
                'water': 60.0,
                'retinol': 2.0,  # Over 1% limit
                'vitamin_c': 25.0,  # Over 20% limit
                'phenoxyethanol': 0.5
            }
        }
        
        compliance_result2 = reducer.check_regulatory_compliance(non_compliant_formulation)
        
        # Test Case 3: Edge case formulation (at limits)
        edge_case_formulation = {
            'ingredients': {
                'water': 75.0,
                'retinol': 1.0,  # Exactly at limit
                'vitamin_c': 20.0,  # Exactly at limit
                'phenoxyethanol': 1.0  # Exactly at limit
            }
        }
        
        compliance_result3 = reducer.check_regulatory_compliance(edge_case_formulation)
        
        compliance_accuracy = (compliance_result1 and not compliance_result2 and compliance_result3)
        
        self.test_results['regulatory_compliance'] = {
            'compliant_detected_correctly': compliance_result1,
            'non_compliant_detected_correctly': not compliance_result2,
            'edge_case_handled_correctly': compliance_result3,
            'overall_accuracy': compliance_accuracy,
            'total_score': sum([compliance_result1, not compliance_result2, compliance_result3])
        }
        
        print(f"  Compliant formulation detected: {compliance_result1}")
        print(f"  Non-compliant formulation rejected: {not compliance_result2}")
        print(f"  Edge case handled correctly: {compliance_result3}")
    
    def benchmark_performance(self):
        """Benchmark system performance"""
        print("  Running performance benchmarks...")
        
        # Benchmark 1: INCI parsing speed
        reducer = INCISearchSpaceReducer()
        test_inci = "Aqua, Sodium Hyaluronate, Retinol, Niacinamide, Vitamin C, Glycerin, Cetyl Alcohol, Phenoxyethanol"
        
        start_time = time.time()
        for _ in range(100):
            reducer.parse_inci_list(test_inci)
        inci_parsing_time = (time.time() - start_time) / 100
        
        # Benchmark 2: Attention allocation speed
        manager = AttentionAllocationManager(max_active_nodes=50)
        
        # Add many formulations
        formulations = []
        for i in range(50):
            formulation = {
                'ingredients': {
                    f'ingredient_{j}': random.uniform(0.1, 10.0) 
                    for j in range(random.randint(3, 8))
                },
                'type': 'test_formulation'
            }
            formulations.append(formulation)
        
        start_time = time.time()
        for formulation in formulations:
            manager.add_formulation_node(formulation)
        attention_allocation_time = (time.time() - start_time) / len(formulations)
        
        # Benchmark 3: Multiscale property calculation speed
        optimizer = MultiscaleConstraintOptimizer()
        from multiscale_optimizer import FormulationCandidate
        
        test_candidate = FormulationCandidate(
            id="benchmark_candidate",
            ingredients={'retinol': 0.5, 'hyaluronic_acid': 1.0, 'vitamin_c': 10.0, 'niacinamide': 5.0},
            formulation_type='serum'
        )
        
        start_time = time.time()
        for _ in range(100):
            optimizer._calculate_multiscale_properties(test_candidate)
        multiscale_calculation_time = (time.time() - start_time) / 100
        
        self.performance_metrics = {
            'inci_parsing_time_ms': inci_parsing_time * 1000,
            'attention_allocation_time_ms': attention_allocation_time * 1000,
            'multiscale_calculation_time_ms': multiscale_calculation_time * 1000
        }
        
        print(f"  INCI parsing: {inci_parsing_time*1000:.2f} ms per parse")
        print(f"  Attention allocation: {attention_allocation_time*1000:.2f} ms per node")
        print(f"  Multiscale calculation: {multiscale_calculation_time*1000:.2f} ms per candidate")
    
    def test_full_integration(self):
        """Test full system integration"""
        print("  Running full system integration test...")
        
        # Initialize all components
        reducer = INCISearchSpaceReducer()
        attention_manager = AttentionAllocationManager()
        optimizer = MultiscaleConstraintOptimizer(reducer, attention_manager)
        
        # Add comprehensive objectives and constraints
        optimizer.add_objective(OptimizationObjective(ObjectiveType.EFFICACY, 0.3))
        optimizer.add_objective(OptimizationObjective(ObjectiveType.SAFETY, 0.3))
        optimizer.add_objective(OptimizationObjective(ObjectiveType.COST, 0.2, minimize=True))
        optimizer.add_objective(OptimizationObjective(ObjectiveType.STABILITY, 0.2))
        
        optimizer.add_scale_constraint(ScaleConstraint(
            scale=OptimizationScale.MOLECULAR,
            parameter="penetration_index",
            min_value=0.3,
            weight=0.8
        ))
        
        # Test with realistic formulation scenario
        target_inci = "Aqua, Retinol, Sodium Hyaluronate, Niacinamide, Tocopherol, Glycerin, Cetyl Alcohol, Phenoxyethanol"
        constraints = FormulationConstraint(
            target_ph=(5.5, 6.5),
            max_total_actives=10.0,
            required_ingredients=["water"],
            forbidden_ingredients=[]
        )
        
        try:
            start_time = time.time()
            results = optimizer.optimize_formulation(
                target_inci=target_inci,
                base_constraints=constraints,
                target_condition="anti_aging",
                max_time_minutes=3.0
            )
            integration_time = time.time() - start_time
            
            integration_success = len(results) > 0
            results_quality = False
            if results:
                results_quality = all(
                    r.efficacy_score > 0 and 
                    r.safety_score > 0 and
                    len(r.ingredients) > 0
                    for r in results
                )
            
        except Exception as e:
            print(f"  Integration test failed with error: {e}")
            integration_success = False
            results_quality = False
            integration_time = 0
        
        self.test_results['integration'] = {
            'integration_success': integration_success,
            'results_quality': results_quality,
            'integration_time': integration_time,
            'total_score': integration_success + results_quality
        }
        
        print(f"  Integration success: {integration_success}")
        print(f"  Results quality: {results_quality}")
        print(f"  Integration time: {integration_time:.2f}s")
    
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
                if isinstance(results['total_score'], (int, float)):
                    if results['total_score'] >= 0.7 * max(4, len(results) - 1):  # 70% pass rate
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
        
        if self.performance_metrics.get('multiscale_calculation_time_ms', 0) > 100:
            report['recommendations'].append("Multiscale calculation performance could be optimized")
        
        if self.test_results.get('optimization_accuracy', {}).get('fitness_improved', False) is False:
            report['recommendations'].append("Optimization algorithm may need tuning for better convergence")
        
        return report

def run_validation_tests():
    """Run all validation tests and display results"""
    test_suite = TestSuite()
    report = test_suite.run_all_tests()
    
    print("\n" + "="*60)
    print("VALIDATION TEST REPORT")
    print("="*60)
    
    print(f"\nOverall Results:")
    print(f"  Status: {report['summary']['overall_status']}")
    print(f"  Tests Passed: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")
    print(f"  Pass Rate: {report['summary']['pass_rate']:.1%}")
    
    print(f"\nPerformance Metrics:")
    for metric, value in report['performance_metrics'].items():
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
    with open('/tmp/test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nFull test report saved to /tmp/test_report.json")
    
    return report

if __name__ == "__main__":
    run_validation_tests()
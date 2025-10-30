#!/usr/bin/env python3
"""
Meta-Optimization Strategy Example

This example demonstrates the revolutionary meta-optimization strategy
that generates optimal formulations for every possible condition and
treatment combination using recursive optimization and adaptive learning.
"""

import sys
import os
import time
import json
from typing import Dict, List

# Add the cheminformatics module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cheminformatics.hypergredient import (
    MetaOptimizationStrategy, OptimizationStrategy, ConditionTreatmentPair,
    create_hypergredient_database, FormulationRequest
)


def demonstrate_meta_optimization_basics():
    """Demonstrate basic meta-optimization functionality"""
    print("🚀 META-OPTIMIZATION STRATEGY DEMONSTRATION")
    print("=" * 60)
    
    # Create database and meta-optimizer
    print("📊 Initializing Meta-Optimization System...")
    db = create_hypergredient_database()
    meta_optimizer = MetaOptimizationStrategy(db, cache_size=50, max_recursive_depth=2)
    
    print(f"✓ Database loaded with {len(db.hypergredients)} hypergredients")
    print(f"✓ Meta-optimizer initialized with {len(meta_optimizer.condition_treatment_mapping)} condition categories")
    print()
    
    # Show condition-treatment mapping
    print("🎯 COMPREHENSIVE CONDITION-TREATMENT MAPPING")
    print("-" * 50)
    
    total_pairs = 0
    for condition, pairs in meta_optimizer.condition_treatment_mapping.items():
        print(f"{condition.upper()}: {len(pairs)} treatment combinations")
        total_pairs += len(pairs)
        
        # Show example pairs
        for i, pair in enumerate(pairs[:2]):  # Show first 2 pairs
            treatments_str = ", ".join(pair.treatments[:3])  # Show first 3 treatments
            if len(pair.treatments) > 3:
                treatments_str += "..."
            print(f"  └─ {pair.severity.title()} {pair.skin_type} skin: {treatments_str}")
    
    print(f"\n📈 Total treatment combinations: {total_pairs}")
    print()


def demonstrate_strategy_selection():
    """Demonstrate adaptive strategy selection"""
    print("🧠 ADAPTIVE STRATEGY SELECTION")
    print("-" * 40)
    
    db = create_hypergredient_database()
    meta_optimizer = MetaOptimizationStrategy(db)
    
    # Test different condition complexities
    test_cases = [
        ConditionTreatmentPair(
            condition='hydration',
            treatments=['hydration'],
            severity='mild',
            skin_type='normal',
            complexity_score=1.0
        ),
        ConditionTreatmentPair(
            condition='aging',
            treatments=['anti_aging', 'wrinkles', 'firmness', 'brightness'],
            severity='moderate',
            skin_type='dry',
            complexity_score=4.5
        ),
        ConditionTreatmentPair(
            condition='comprehensive_treatment',
            treatments=['anti_aging', 'hyperpigmentation', 'acne', 'hydration', 'sensitivity'],
            severity='severe',
            skin_type='sensitive',
            complexity_score=8.0
        )
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case.condition.title()}")
        print(f"  Complexity: {test_case.complexity_score}")
        print(f"  Treatments: {len(test_case.treatments)}")
        print(f"  Severity: {test_case.severity}")
        print(f"  Skin Type: {test_case.skin_type}")
        
        strategy = meta_optimizer.select_optimal_strategy(test_case)
        print(f"  🎯 Selected Strategy: {strategy.value}")
        print()


def demonstrate_recursive_optimization():
    """Demonstrate recursive formulation exploration"""
    print("🔄 RECURSIVE FORMULATION EXPLORATION")
    print("-" * 45)
    
    db = create_hypergredient_database()
    meta_optimizer = MetaOptimizationStrategy(db, cache_size=10)
    
    # Create a challenging formulation request
    request = FormulationRequest(
        target_concerns=['anti_aging', 'hyperpigmentation'],
        skin_type='sensitive',
        budget=1200.0,
        preferences=['gentle', 'stable'],
        excluded_ingredients=['tretinoin', 'hydroquinone'],
        max_ingredients=6
    )
    
    print("📋 Formulation Request:")
    print(f"  Target Concerns: {request.target_concerns}")
    print(f"  Skin Type: {request.skin_type}")
    print(f"  Budget: R{request.budget}")
    print(f"  Preferences: {request.preferences}")
    print(f"  Excluded: {request.excluded_ingredients}")
    print(f"  Max Ingredients: {request.max_ingredients}")
    print()
    
    print("🔍 Starting Recursive Exploration...")
    start_time = time.time()
    
    solutions = meta_optimizer.recursive_formulation_exploration(request, max_depth=2)
    
    exploration_time = time.time() - start_time
    
    print(f"✓ Exploration completed in {exploration_time:.2f} seconds")
    print(f"✓ Found {len(solutions)} unique solutions")
    print()
    
    if solutions:
        best_solution = solutions[0]
        print("🏆 BEST SOLUTION:")
        print(f"  Total Score: {best_solution.total_score:.1f}/10")
        print(f"  Cost: R{best_solution.cost:.2f}")
        print(f"  Synergy Score: {best_solution.synergy_score:.2f}")
        print("  Hypergredients:")
        for ingredient, concentration in best_solution.hypergredients.items():
            print(f"    └─ {ingredient}: {concentration:.1f}%")
        print()


def demonstrate_small_scale_optimization():
    """Demonstrate small-scale comprehensive optimization"""
    print("⚡ SMALL-SCALE COMPREHENSIVE OPTIMIZATION")
    print("-" * 50)
    
    db = create_hypergredient_database()
    meta_optimizer = MetaOptimizationStrategy(db, cache_size=20)
    
    # Temporarily modify mapping for demonstration
    original_mapping = meta_optimizer.condition_treatment_mapping
    
    # Create focused test mapping
    demo_mapping = {
        'acne': [
            ConditionTreatmentPair(
                condition='acne',
                treatments=['acne', 'oily_skin'],
                severity='mild',
                skin_type='oily',
                budget_range=(600.0, 1000.0),
                complexity_score=2.0
            ),
            ConditionTreatmentPair(
                condition='acne',
                treatments=['acne', 'inflammation'],
                severity='moderate',
                skin_type='sensitive',
                budget_range=(800.0, 1200.0),
                complexity_score=3.0
            )
        ],
        'aging': [
            ConditionTreatmentPair(
                condition='aging',
                treatments=['anti_aging', 'hydration'],
                severity='moderate',
                skin_type='dry',
                budget_range=(1000.0, 1500.0),
                complexity_score=2.5
            )
        ]
    }
    
    meta_optimizer.condition_treatment_mapping = demo_mapping
    
    try:
        print(f"🎯 Optimizing {sum(len(pairs) for pairs in demo_mapping.values())} condition-treatment pairs...")
        print()
        
        start_time = time.time()
        results = meta_optimizer.optimize_all_conditions(max_solutions_per_condition=2)
        optimization_time = time.time() - start_time
        
        print(f"✓ Optimization completed in {optimization_time:.2f} seconds")
        print()
        
        # Display results
        for condition, condition_results in results.items():
            print(f"📊 CONDITION: {condition.upper()}")
            print("-" * 30)
            
            for result in condition_results:
                pair = result.condition_treatment_pair
                print(f"  Severity: {pair.severity.title()}")
                print(f"  Skin Type: {pair.skin_type.title()}")
                print(f"  Strategy: {result.optimization_strategy.value}")
                print(f"  Quality Score: {result.quality_score:.1f}/10")
                print(f"  Computation Time: {result.computation_time:.2f}s")
                print(f"  Solutions Found: {len(result.formulation_solutions)}")
                
                if result.formulation_solutions:
                    best_solution = result.formulation_solutions[0]
                    print(f"  Best Solution Score: {best_solution.total_score:.1f}/10")
                    print(f"  Best Solution Cost: R{best_solution.cost:.2f}")
                
                if result.improvement_suggestions:
                    print("  💡 Improvement Suggestions:")
                    for suggestion in result.improvement_suggestions[:2]:
                        print(f"    └─ {suggestion}")
                
                print()
        
        # Generate report
        report = meta_optimizer.get_optimization_report()
        print("📈 OPTIMIZATION PERFORMANCE REPORT")
        print("-" * 40)
        print(f"Cache Size: {report['cache_statistics']['size']}")
        print(f"Condition Coverage: {report['condition_coverage']}")
        print(f"Total Combinations: {report['total_combinations']}")
        print()
        
        print("Strategy Performance:")
        for strategy, perf in report['strategy_performance'].items():
            print(f"  {strategy}: {perf['usage_count']} uses, "
                  f"{perf['average_quality']:.1f} avg quality, "
                  f"{perf['success_rate']:.1%} success rate")
        print()
        
    finally:
        # Restore original mapping
        meta_optimizer.condition_treatment_mapping = original_mapping


def demonstrate_formulation_library_export():
    """Demonstrate formulation library export"""
    print("📚 FORMULATION LIBRARY EXPORT")
    print("-" * 35)
    
    db = create_hypergredient_database()
    meta_optimizer = MetaOptimizationStrategy(db)
    
    # Create some mock results for demonstration
    from cheminformatics.hypergredient import (
        OptimizationResult, FormulationSolution, OptimizationObjective
    )
    
    mock_solution = FormulationSolution(
        hypergredients={'vitamin_c': 10.0, 'niacinamide': 5.0, 'hyaluronic_acid': 2.0},
        objective_scores={
            OptimizationObjective.EFFICACY: 8.5,
            OptimizationObjective.SAFETY: 9.0,
            OptimizationObjective.STABILITY: 7.5,
            OptimizationObjective.COST: 7.0
        },
        total_score=8.0,
        cost=850.0,
        predicted_efficacy={'brightness': 85.0, 'hydration': 78.0},
        synergy_score=1.3,
        stability_months=18
    )
    
    test_pair = ConditionTreatmentPair(
        condition='brightness',
        treatments=['brightness', 'hydration'],
        severity='moderate',
        skin_type='normal'
    )
    
    mock_result = OptimizationResult(
        condition_treatment_pair=test_pair,
        formulation_solutions=[mock_solution],
        optimization_strategy=OptimizationStrategy.HYBRID_MULTI_OBJECTIVE,
        performance_metrics={'average_quality': 8.0, 'cost_efficiency': 0.94},
        computation_time=3.5,
        iterations=150,
        quality_score=8.0,
        improvement_suggestions=['Consider premium vitamin C derivative for enhanced stability']
    )
    
    results = {'brightness': [mock_result]}
    
    # Export library
    library_path = "/tmp/demo_formulation_library.json"
    library = meta_optimizer.export_formulation_library(results, library_path)
    
    print(f"✓ Formulation library exported to {library_path}")
    print(f"✓ Library contains {library['metadata']['total_formulations']} formulations")
    print(f"✓ Covers {library['metadata']['total_conditions']} conditions")
    print()
    
    # Show sample of exported data
    print("📋 SAMPLE LIBRARY ENTRY:")
    print("-" * 25)
    
    brightness_data = library['formulations']['brightness'][0]
    print(f"Condition: {brightness_data['condition']}")
    print(f"Treatments: {brightness_data['treatments']}")
    print(f"Quality Score: {brightness_data['quality_score']}")
    print(f"Strategy: {brightness_data['optimization_strategy']}")
    
    if brightness_data['solutions']:
        solution = brightness_data['solutions'][0]
        print("Best Solution:")
        print(f"  Score: {solution['total_score']}/10")
        print(f"  Cost: R{solution['cost']}")
        print("  Ingredients:")
        for ingredient, conc in solution['hypergredients'].items():
            print(f"    └─ {ingredient}: {conc}%")
    
    print()


def main():
    """Main demonstration function"""
    try:
        demonstrate_meta_optimization_basics()
        demonstrate_strategy_selection()
        demonstrate_recursive_optimization()
        demonstrate_small_scale_optimization()
        demonstrate_formulation_library_export()
        
        print("🎉 META-OPTIMIZATION DEMONSTRATION COMPLETE!")
        print("=" * 55)
        print("This revolutionary system transforms formulation design into")
        print("a comprehensive, adaptive, and self-improving process! 🚀🧬")
        
    except Exception as e:
        print(f"❌ Error running demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
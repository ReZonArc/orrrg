#!/usr/bin/env python3
"""
Hypergredient Framework Example

This example demonstrates the revolutionary hypergredient formulation
design system with real-world formulation optimization scenarios.
"""

import sys
import os

# Add the cheminformatics module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cheminformatics.hypergredient import (
    create_hypergredient_database, HypergredientFormulator,
    InteractionMatrix, DynamicScoringSystem
)
from cheminformatics.hypergredient.optimization import FormulationRequest


def demonstrate_hypergredient_database():
    """Demonstrate hypergredient database functionality"""
    print("🧬 HYPERGREDIENT FRAMEWORK DEMONSTRATION")
    print("=" * 50)
    
    # Create database
    db = create_hypergredient_database()
    stats = db.get_stats()
    
    print(f"📊 Database Statistics:")
    print(f"   Total Hypergredients: {stats['total_hypergredients']}")
    print(f"   Average Potency: {stats['avg_potency']:.1f}/10")
    print(f"   Cost Range: R{stats['cost_range'][0]:.0f} - R{stats['cost_range'][1]:.0f}/g")
    print()
    
    # Show hypergredient classes
    print("🔷 HYPERGREDIENT TAXONOMY:")
    for i, (class_code, description) in enumerate(db.class_index.items(), 1):
        count = len(db.class_index[class_code])
        print(f"   {i:2d}. {class_code}: {description} ({count} ingredients)")
    print()
    
    # Show top performers in each major class  
    print("🏆 TOP PERFORMERS BY CLASS:")
    major_classes = ['H.CT', 'H.CS', 'H.AO', 'H.HY']
    
    for class_code in major_classes:
        top_performers = db.get_top_performers(class_code, n=2)
        print(f"   {class_code}:")
        for performer in top_performers:
            score = performer.metrics.calculate_composite_score()
            print(f"     • {performer.name}: {score:.1f}/10 "
                  f"(Potency: {performer.potency}/10, Safety: {performer.safety_score}/10)")
    print()


def demonstrate_interaction_analysis():
    """Demonstrate interaction matrix analysis"""
    print("🔄 INTERACTION MATRIX ANALYSIS")
    print("=" * 50)
    
    db = create_hypergredient_database()
    matrix = InteractionMatrix()
    
    # Get some ingredients for analysis
    vitamin_c = db.hypergredients.get('vitamin_c_laa')
    vitamin_e = db.hypergredients.get('vitamin_e')
    retinol = db.hypergredients.get('retinol')
    
    if vitamin_c and vitamin_e:
        # Analyze synergistic pair
        score = matrix.calculate_interaction_score(vitamin_c, vitamin_e)
        print(f"✅ Vitamin C + Vitamin E Interaction Score: {score:.2f}")
        print(f"   Interpretation: {'Synergistic' if score > 1.2 else 'Neutral' if score > 0.8 else 'Antagonistic'}")
        print()
    
    if vitamin_c and retinol:
        # Analyze potentially problematic pair
        score = matrix.calculate_interaction_score(vitamin_c, retinol)
        print(f"⚠️  Vitamin C + Retinol Interaction Score: {score:.2f}")
        print(f"   Interpretation: {'Synergistic' if score > 1.2 else 'Neutral' if score > 0.8 else 'Antagonistic'}")
        print()
    
    # Analyze a complete formulation
    test_ingredients = []
    for name in ['niacinamide', 'hyaluronic_acid_hmw', 'vitamin_e']:
        if name in db.hypergredients:
            test_ingredients.append(db.hypergredients[name])
    
    if test_ingredients:
        analysis = matrix.analyze_formulation_interactions(test_ingredients)
        print("📊 FORMULATION INTERACTION ANALYSIS:")
        print(f"   Total Score: {analysis['total_score']:.1f}/10")
        print(f"   Synergistic Pairs: {len(analysis['synergistic_pairs'])}")
        print(f"   Antagonistic Pairs: {len(analysis['antagonistic_pairs'])}")
        
        if analysis['synergistic_pairs']:
            print("   Synergies Found:")
            for pair in analysis['synergistic_pairs']:
                print(f"     • {pair['ingredient1']} + {pair['ingredient2']}: {pair['score']:.2f}")
        print()


def demonstrate_formulation_optimization():
    """Demonstrate advanced formulation optimization"""
    print("🎯 FORMULATION OPTIMIZATION EXAMPLES")
    print("=" * 50)
    
    db = create_hypergredient_database()
    formulator = HypergredientFormulator(db)
    
    # Example 1: Anti-Aging Formulation
    print("Example 1: OPTIMAL ANTI-AGING FORMULATION")
    print("-" * 40)
    
    solution = formulator.generate_formulation(
        target='anti_aging',
        secondary=['hydration', 'brightness'],
        budget=1500,
        skin_type='normal_to_dry',
        exclude=['tretinoin']  # Exclude for sensitivity
    )
    
    if solution:
        print(solution.get_summary())
    else:
        print("❌ Could not generate suitable formulation with given constraints")
    
    print()
    
    # Example 2: Hyperpigmentation Treatment
    print("Example 2: BRIGHTENING SERUM FORMULATION")
    print("-" * 40)
    
    solution = formulator.generate_formulation(
        target='hyperpigmentation',
        secondary=['anti_aging', 'hydration'],
        budget=1000,
        exclude=['hydroquinone', 'kojic_acid'],
        skin_type='normal'
    )
    
    if solution:
        print(solution.get_summary())
    else:
        print("❌ Could not generate suitable formulation with given constraints")
    
    print()


def demonstrate_performance_scoring():
    """Demonstrate dynamic scoring system"""
    print("📈 DYNAMIC SCORING SYSTEM")
    print("=" * 50)
    
    db = create_hypergredient_database()
    scoring_system = DynamicScoringSystem()
    
    # Analyze individual hypergredient performance
    test_ingredient = db.hypergredients.get('bakuchiol')
    if test_ingredient:
        metrics = scoring_system.calculate_hypergredient_metrics(test_ingredient)
        
        print(f"📊 PERFORMANCE ANALYSIS: {test_ingredient.name.upper()}")
        print(f"   Efficacy Score: {metrics.efficacy_score:.1f}/10")
        print(f"   Safety Score: {metrics.safety_score:.1f}/10") 
        print(f"   Stability Score: {metrics.stability_score:.1f}/10")
        print(f"   Cost Efficiency: {metrics.cost_efficiency:.1f}/10")
        print(f"   Synergy Potential: {metrics.synergy_potential:.1f}/10")
        print()
        
        # Generate radar chart data
        radar_data = metrics.get_performance_radar()
        print("🎯 PERFORMANCE RADAR:")
        for metric, value in radar_data.items():
            bar_length = int(value)
            bar = "█" * bar_length + "░" * (10 - bar_length)
            print(f"   {metric:15s}: {bar} {value:.1f}")
        print()
    
    # Compare formulation metrics
    hypergredients = []
    concentrations = {}
    
    for name in ['niacinamide', 'hyaluronic_acid_hmw', 'vitamin_e']:
        if name in db.hypergredients:
            hypergredients.append(db.hypergredients[name])
            concentrations[name] = 3.0 if name == 'niacinamide' else 1.0
    
    if hypergredients:
        formulation_metrics = scoring_system.calculate_formulation_metrics(
            hypergredients, concentrations
        )
        
        print("📊 FORMULATION PERFORMANCE:")
        print(f"   Overall Efficacy: {formulation_metrics.efficacy_score:.1f}/10")
        print(f"   Overall Safety: {formulation_metrics.safety_score:.1f}/10")
        print(f"   Formulation Stability: {formulation_metrics.stability_score:.1f}/10")
        print()


def demonstrate_real_world_example():
    """Demonstrate real-world formulation example"""
    print("🚀 REAL-WORLD FORMULATION EXAMPLE")
    print("=" * 50)
    
    db = create_hypergredient_database()
    formulator = HypergredientFormulator(db)
    
    # Simulate customer request
    print("📝 CUSTOMER REQUEST:")
    print("   Target: Mature skin with wrinkles and dark spots")
    print("   Budget: R1200 for 50ml")
    print("   Skin Type: Dry, sensitive")
    print("   Preferences: Gentle, stable formulation")
    print("   Exclusions: No retinoids (too harsh)")
    print()
    
    # Generate optimized formulation
    solution = formulator.generate_formulation(
        target='anti_aging',
        secondary=['hyperpigmentation', 'hydration'],
        budget=1200,
        skin_type='dry',
        exclude=['tretinoin', 'retinol', 'glycolic_acid'],
        ph_range=(5.5, 6.5)
    )
    
    if solution:
        print("✅ OPTIMAL SOLUTION GENERATED:")
        print(solution.get_summary())
        
        # Show additional analysis
        if solution.warnings:
            print("⚠️  FORMULATION WARNINGS:")
            for warning in solution.warnings:
                print(f"   • {warning}")
            print()
        
        # Show ingredient details
        print("🔍 INGREDIENT DETAILS:")
        for name, percentage in solution.hypergredients.items():
            if name in db.hypergredients:
                ingredient = db.hypergredients[name]
                print(f"   • {ingredient.name} ({percentage:.1f}%)")
                print(f"     Class: {ingredient.hypergredient_class}")
                print(f"     Function: {ingredient.primary_function}")
                print(f"     Safety: {ingredient.safety_score}/10")
                print()
    else:
        print("❌ Could not generate suitable formulation")
        print("   Try adjusting constraints or budget")


def main():
    """Main demonstration function"""
    try:
        demonstrate_hypergredient_database()
        demonstrate_interaction_analysis()
        demonstrate_performance_scoring()
        demonstrate_formulation_optimization()
        demonstrate_real_world_example()
        
        print("🎉 HYPERGREDIENT FRAMEWORK DEMONSTRATION COMPLETE!")
        print("=" * 50)
        print("This revolutionary system transforms formulation from art to science! 🚀🧬")
        
    except Exception as e:
        print(f"❌ Error running demonstration: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_meta_optimization():
    """Demonstrate meta-optimization capabilities"""
    print("🚀 META-OPTIMIZATION STRATEGY")
    print("=" * 40)
    
    try:
        from cheminformatics.hypergredient import MetaOptimizationStrategy
        
        db = create_hypergredient_database()
        meta_optimizer = MetaOptimizationStrategy(db, cache_size=20)
        
        print(f"✓ Meta-optimizer initialized")
        print(f"✓ Condition categories: {len(meta_optimizer.condition_treatment_mapping)}")
        print(f"✓ Total combinations: {sum(len(pairs) for pairs in meta_optimizer.condition_treatment_mapping.values())}")
        print()
        
        # Quick demonstration
        print("🎯 QUICK META-OPTIMIZATION DEMO:")
        print("-" * 35)
        
        # Run small optimization
        original_mapping = meta_optimizer.condition_treatment_mapping
        demo_mapping = {'acne': original_mapping['acne'][:1]}  # Just one pair
        meta_optimizer.condition_treatment_mapping = demo_mapping
        
        results = meta_optimizer.optimize_all_conditions(max_solutions_per_condition=1)
        
        if results and 'acne' in results:
            result = results['acne'][0]
            print(f"✓ Generated formulation for {result.condition_treatment_pair.condition}")
            print(f"✓ Strategy used: {result.optimization_strategy.value}")
            print(f"✓ Quality score: {result.quality_score:.1f}/10")
            if result.formulation_solutions:
                best = result.formulation_solutions[0]
                print(f"✓ Best solution score: {best.total_score:.1f}/10")
                print(f"✓ Cost: R{best.cost:.2f}")
        
        # Restore mapping
        meta_optimizer.condition_treatment_mapping = original_mapping
        
        print()
        print("💡 For full meta-optimization demo, run: python examples/python/meta_optimization_example.py")
        print()
        
    except ImportError:
        print("❌ Meta-optimization not available - check installation")


if __name__ == "__main__":
    # Run main demonstrations
    main()
    
    # Run meta-optimization demo
    demonstrate_meta_optimization()
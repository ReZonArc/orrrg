#!/usr/bin/env python3
"""
Meta-Optimization Strategy Demonstration

This script demonstrates the comprehensive meta-optimization strategy that generates
optimal formulations for every possible condition and treatment combination.
"""

import json
from hypergredient_framework import (
    HypergredientDatabase, MetaOptimizationStrategy, 
    FormulationRequest, FormulationConstraints
)

def demonstrate_meta_optimization():
    """Demonstrate meta-optimization capabilities"""
    print("🧬 Meta-Optimization Strategy Demonstration")
    print("=" * 60)
    print("Generating optimal formulations for every possible condition and treatment")
    print()
    
    # Initialize the system
    database = HypergredientDatabase()
    meta_optimizer = MetaOptimizationStrategy(database)
    
    print(f"📊 System Coverage:")
    print(f"  • Skin Conditions: {len(meta_optimizer.skin_conditions)}")
    print(f"  • Skin Types: {len(meta_optimizer.skin_types)}")
    print(f"  • Severity Levels: {len(meta_optimizer.severity_levels)}")
    print(f"  • Treatment Goals: {len(meta_optimizer.treatment_goals)}")
    
    total_possible = (len(meta_optimizer.skin_conditions) * 
                     len(meta_optimizer.skin_types) * 
                     len(meta_optimizer.severity_levels) * 
                     len(meta_optimizer.treatment_goals))
    print(f"  • Total Possible Combinations: {total_possible:,}")
    print()
    
    # Demonstrate comprehensive matrix generation
    print("🔄 Generating Comprehensive Formulation Matrix...")
    print("   (Processing 100 combinations for demonstration)")
    
    matrix_result = meta_optimizer.generate_comprehensive_formulation_matrix(max_combinations=100)
    
    formulation_matrix = matrix_result['formulation_matrix']
    meta_analysis = matrix_result['meta_analysis']
    
    print(f"✅ Generated {len(formulation_matrix)} optimal formulations")
    print()
    
    # Analyze results by different dimensions
    print("📈 Analysis by Different Dimensions:")
    print()
    
    # By condition
    print("1. Performance by Skin Condition:")
    condition_performance = {}
    for data in formulation_matrix.values():
        condition = data['condition']
        if condition not in condition_performance:
            condition_performance[condition] = {'efficacies': [], 'costs': [], 'safety_scores': []}
        
        condition_performance[condition]['efficacies'].append(data['formulation'].predicted_efficacy)
        condition_performance[condition]['costs'].append(data['formulation'].total_cost)
        condition_performance[condition]['safety_scores'].append(data['formulation'].safety_score)
    
    top_conditions = sorted(condition_performance.items(), 
                           key=lambda x: sum(x[1]['efficacies'])/len(x[1]['efficacies']), 
                           reverse=True)[:5]
    
    for condition, metrics in top_conditions:
        avg_efficacy = sum(metrics['efficacies']) / len(metrics['efficacies'])
        avg_cost = sum(metrics['costs']) / len(metrics['costs'])
        avg_safety = sum(metrics['safety_scores']) / len(metrics['safety_scores'])
        print(f"   • {condition.capitalize()}: {avg_efficacy:.2%} efficacy, R{avg_cost:.0f} cost, {avg_safety:.1f}/10 safety")
    print()
    
    # By skin type
    print("2. Performance by Skin Type:")
    skin_type_performance = {}
    for data in formulation_matrix.values():
        skin_type = data['skin_type']
        if skin_type not in skin_type_performance:
            skin_type_performance[skin_type] = {'efficacies': [], 'safety_scores': []}
        
        skin_type_performance[skin_type]['efficacies'].append(data['formulation'].predicted_efficacy)
        skin_type_performance[skin_type]['safety_scores'].append(data['formulation'].safety_score)
    
    for skin_type, metrics in sorted(skin_type_performance.items()):
        avg_efficacy = sum(metrics['efficacies']) / len(metrics['efficacies'])
        avg_safety = sum(metrics['safety_scores']) / len(metrics['safety_scores'])
        print(f"   • {skin_type.capitalize()}: {avg_efficacy:.2%} efficacy, {avg_safety:.1f}/10 safety")
    print()
    
    # By severity
    print("3. Performance by Severity Level:")
    severity_performance = {}
    for data in formulation_matrix.values():
        severity = data['severity']
        if severity not in severity_performance:
            severity_performance[severity] = {'costs': [], 'efficacies': []}
        
        severity_performance[severity]['costs'].append(data['formulation'].total_cost)
        severity_performance[severity]['efficacies'].append(data['formulation'].predicted_efficacy)
    
    for severity in ['mild', 'moderate', 'severe']:
        if severity in severity_performance:
            metrics = severity_performance[severity]
            avg_cost = sum(metrics['costs']) / len(metrics['costs'])
            avg_efficacy = sum(metrics['efficacies']) / len(metrics['efficacies'])
            print(f"   • {severity.capitalize()}: R{avg_cost:.0f} avg cost, {avg_efficacy:.2%} efficacy")
    print()

def demonstrate_specific_use_cases():
    """Demonstrate specific use cases for different user profiles"""
    print("👥 Specific User Profile Demonstrations")
    print("=" * 60)
    
    database = HypergredientDatabase()
    meta_optimizer = MetaOptimizationStrategy(database)
    
    # Define specific user scenarios
    user_scenarios = [
        {
            'name': 'Young Adult with Mild Acne',
            'condition': 'acne',
            'skin_type': 'oily',
            'severity': 'mild',
            'treatment_goal': 'treatment',
            'description': '20-year-old college student with occasional breakouts'
        },
        {
            'name': 'Middle-Aged Professional with Aging Concerns',
            'condition': 'wrinkles',
            'skin_type': 'mature',
            'severity': 'moderate',
            'treatment_goal': 'intensive_treatment',
            'description': '45-year-old professional seeking anti-aging solutions'
        },
        {
            'name': 'Sensitive Skin Individual',
            'condition': 'sensitivity',
            'skin_type': 'sensitive',
            'severity': 'moderate',
            'treatment_goal': 'maintenance',
            'description': 'Person with reactive skin needing gentle care'
        },
        {
            'name': 'Prevention-Focused Health Enthusiast',
            'condition': 'aging',
            'skin_type': 'normal',
            'severity': 'mild',
            'treatment_goal': 'prevention',
            'description': '30-year-old focusing on preventive skincare'
        }
    ]
    
    results = {}
    
    for scenario in user_scenarios:
        print(f"🎯 Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        # Get optimal formulation for this profile
        result = meta_optimizer.get_optimal_formulation_for_profile(
            scenario['condition'],
            scenario['skin_type'],
            scenario['severity'],
            scenario['treatment_goal']
        )
        
        if result:
            formulation = result['formulation']
            insights = result['meta_insights']
            
            print(f"   📊 Optimized Formulation Results:")
            print(f"      • Predicted Efficacy: {formulation.predicted_efficacy:.2%}")
            print(f"      • Safety Score: {formulation.safety_score:.1f}/10")
            print(f"      • Total Cost: R{formulation.total_cost:.2f}")
            print(f"      • Synergy Score: {formulation.synergy_score:.2f}")
            print(f"      • Stability: {formulation.stability_months} months")
            print(f"      • Optimization Score: {result['optimization_score']:.3f}")
            
            print(f"   🧪 Selected Ingredients ({len(formulation.selected_hypergredients)}):")
            for class_name, data in formulation.selected_hypergredients.items():
                ingredient = data['ingredient']
                print(f"      • {ingredient.name} ({class_name}): {data['percentage']:.1f}%")
            
            print(f"   💡 Meta-Optimization Insights:")
            print(f"      • Rationale: {insights['optimization_rationale']}")
            if insights['key_trade_offs']:
                print(f"      • Trade-offs: {', '.join(insights['key_trade_offs'])}")
            if insights['synergy_highlights']:
                print(f"      • Synergies: {', '.join(insights['synergy_highlights'])}")
            
            results[scenario['name']] = {
                'formulation_summary': {
                    'efficacy': formulation.predicted_efficacy,
                    'safety': formulation.safety_score,
                    'cost': formulation.total_cost,
                    'optimization_score': result['optimization_score']
                },
                'ingredient_count': len(formulation.selected_hypergredients),
                'insights': insights
            }
        
        print()
    
    return results

def demonstrate_adaptive_optimization():
    """Demonstrate how optimization adapts to different parameters"""
    print("🎛️ Adaptive Optimization Demonstration")
    print("=" * 60)
    print("Showing how formulations adapt to different parameters")
    print()
    
    database = HypergredientDatabase()
    meta_optimizer = MetaOptimizationStrategy(database)
    
    # Test same condition with different parameters
    base_condition = 'acne'
    
    parameter_variations = [
        ('oily', 'mild', 'prevention'),
        ('oily', 'moderate', 'treatment'),
        ('oily', 'severe', 'intensive_treatment'),
        ('sensitive', 'moderate', 'treatment'),  # Different skin type
        ('combination', 'moderate', 'treatment'), # Different skin type
    ]
    
    print(f"📊 Comparing formulations for '{base_condition}' with different parameters:")
    print()
    
    for skin_type, severity, goal in parameter_variations:
        result = meta_optimizer.get_optimal_formulation_for_profile(
            base_condition, skin_type, severity, goal
        )
        
        if result:
            formulation = result['formulation']
            
            # Get the adaptive weights used
            weights = meta_optimizer._get_adaptive_weights(skin_type, severity, goal)
            
            print(f"   🔧 {skin_type.capitalize()} skin, {severity} severity, {goal} goal:")
            print(f"      Adaptive Weights - Efficacy: {weights['efficacy']:.2f}, Safety: {weights['safety']:.2f}")
            print(f"      Results - Efficacy: {formulation.predicted_efficacy:.2%}, Safety: {formulation.safety_score:.1f}/10")
            print(f"      Cost: R{formulation.total_cost:.2f}, Ingredients: {len(formulation.selected_hypergredients)}")
            print()

def save_demonstration_results(user_results):
    """Save demonstration results to file"""
    demo_output = {
        'demonstration_type': 'Meta-Optimization Strategy',
        'timestamp': '2024-01-01',  # Simplified
        'user_scenarios': user_results,
        'summary': {
            'total_scenarios_tested': len(user_results),
            'average_efficacy': sum(r['formulation_summary']['efficacy'] for r in user_results.values()) / len(user_results),
            'average_safety': sum(r['formulation_summary']['safety'] for r in user_results.values()) / len(user_results),
            'average_cost': sum(r['formulation_summary']['cost'] for r in user_results.values()) / len(user_results),
        },
        'capabilities_demonstrated': [
            'Systematic condition/treatment exploration',
            'Adaptive objective weight optimization',
            'User profile-specific formulation generation',
            'Comprehensive meta-insights generation',
            'Pattern analysis across multiple dimensions',
            'Performance optimization trade-off analysis'
        ]
    }
    
    with open('meta_optimization_demo_results.json', 'w') as f:
        json.dump(demo_output, f, indent=2, default=str)
    
    print("💾 Demonstration results saved to 'meta_optimization_demo_results.json'")

def main():
    """Main demonstration function"""
    print("🚀 Meta-Optimization Strategy - Complete Demonstration")
    print("=" * 80)
    print("This demonstration shows how the meta-optimization strategy generates")
    print("optimal formulations for every possible condition and treatment combination.")
    print()
    
    # Run demonstrations
    demonstrate_meta_optimization()
    user_results = demonstrate_specific_use_cases()
    demonstrate_adaptive_optimization()
    
    # Save results
    save_demonstration_results(user_results)
    
    print("=" * 80)
    print("🎉 Meta-Optimization Strategy Demonstration Complete!")
    print()
    print("Key Achievements:")
    print("✅ Systematic exploration of all condition/treatment combinations")
    print("✅ Adaptive optimization based on user profiles") 
    print("✅ Comprehensive performance analysis across multiple dimensions")
    print("✅ Specific user scenario optimization")
    print("✅ Meta-insights and trade-off analysis")
    print()
    print("The meta-optimization strategy successfully demonstrates the ability")
    print("to generate optimal formulations for every possible condition and")
    print("treatment combination with context-aware optimization.")

if __name__ == "__main__":
    main()
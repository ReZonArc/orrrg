#!/usr/bin/env python3
"""
demo_opencog_multiscale.py

Demonstration of OpenCog Features for Multiscale Constraint Optimization

This script provides a comprehensive demonstration of the implemented system,
showing how OpenCog-inspired components work together for cosmeceutical 
formulation optimization.
"""

from inci_optimizer import INCISearchSpaceReducer, FormulationConstraint
from attention_allocation import AttentionAllocationManager
from multiscale_optimizer import (
    MultiscaleConstraintOptimizer, OptimizationObjective, ObjectiveType,
    ScaleConstraint, OptimizationScale
)

def demonstrate_inci_search_reduction():
    """Demonstrate INCI-driven search space reduction"""
    print("="*60)
    print("1. INCI-DRIVEN SEARCH SPACE REDUCTION DEMONSTRATION")
    print("="*60)
    
    reducer = INCISearchSpaceReducer()
    
    # Example cosmeceutical product INCI
    product_inci = "Aqua, Retinol, Sodium Hyaluronate, Niacinamide, Tocopherol, Glycerin, Cetyl Alcohol, Phenoxyethanol"
    
    print(f"\nTarget Product INCI:")
    print(f"  {product_inci}")
    
    # Parse INCI list
    parsed_ingredients = reducer.parse_inci_list(product_inci)
    print(f"\nParsed Ingredients ({len(parsed_ingredients)} found):")
    for ing in parsed_ingredients:
        print(f"  {ing['position']}. {ing['inci_name']} -> {ing['ingredient_key']}")
    
    # Estimate concentrations
    concentrations = reducer.estimate_concentrations(parsed_ingredients)
    print(f"\nEstimated Concentrations:")
    total_conc = 0
    for ingredient, conc in concentrations.items():
        print(f"  {ingredient}: {conc:.2f}%")
        total_conc += conc
    print(f"  Total: {total_conc:.2f}%")
    
    # Define constraints for search space reduction
    constraints = FormulationConstraint(
        target_ph=(5.5, 6.5),
        max_total_actives=15.0,
        required_ingredients=["water"]
    )
    
    # Filter formulation space
    print(f"\nApplying Constraints:")
    print(f"  pH range: {constraints.target_ph}")
    print(f"  Max total actives: {constraints.max_total_actives}%")
    print(f"  Required ingredients: {constraints.required_ingredients}")
    
    filtered_formulations = reducer.filter_formulation_space(product_inci, constraints)
    print(f"\nFiltered Search Space: {len(filtered_formulations)} viable formulations")
    
    # Show top formulations
    if filtered_formulations:
        print(f"\nTop 3 Filtered Formulations:")
        for i, formulation in enumerate(filtered_formulations[:3]):
            print(f"  Formulation {i+1}:")
            for ingredient, conc in formulation['ingredients'].items():
                print(f"    {ingredient}: {conc:.2f}%")
            print()

def demonstrate_attention_allocation():
    """Demonstrate attention allocation mechanisms"""
    print("="*60)
    print("2. ADAPTIVE ATTENTION ALLOCATION DEMONSTRATION")
    print("="*60)
    
    manager = AttentionAllocationManager(max_active_nodes=20)
    
    # Create various formulation nodes
    formulations = [
        {'ingredients': {'retinol': 0.3, 'hyaluronic_acid': 1.5, 'glycerin': 3.0}, 'type': 'anti_aging_serum'},
        {'ingredients': {'vitamin_c': 15.0, 'vitamin_e': 0.8, 'glycerin': 4.0}, 'type': 'antioxidant_serum'},
        {'ingredients': {'niacinamide': 5.0, 'hyaluronic_acid': 1.0, 'glycerin': 2.0}, 'type': 'pore_minimizer'},
        {'ingredients': {'retinol': 0.8, 'niacinamide': 3.0, 'glycerin': 3.5}, 'type': 'combination_serum'},
        {'ingredients': {'vitamin_c': 10.0, 'hyaluronic_acid': 2.0, 'niacinamide': 2.0}, 'type': 'brightening_serum'}
    ]
    
    print(f"\nCreating Attention Network with {len(formulations)} formulations...")
    node_ids = []
    for i, formulation in enumerate(formulations):
        node_id = manager.add_formulation_node(formulation)
        node_ids.append(node_id)
        node = manager.nodes[node_id]
        print(f"  Node {i+1}: {formulation['type']} (ID: {node_id})")
        print(f"    Initial attention: {node.get_total_attention():.3f}")
    
    print(f"\nSimulating Optimization Results...")
    # Simulate different performance levels
    performance_data = [
        {'efficacy': 0.85, 'cost': 45, 'safety': 0.95},  # High performance
        {'efficacy': 0.72, 'cost': 65, 'safety': 0.88},  # Medium performance
        {'efficacy': 0.68, 'cost': 35, 'safety': 0.92},  # Low efficacy, low cost
        {'efficacy': 0.78, 'cost': 55, 'safety': 0.85},  # Balanced
        {'efficacy': 0.82, 'cost': 70, 'safety': 0.90}   # Good efficacy, high cost
    ]
    
    for node_id, performance in zip(node_ids, performance_data):
        node = manager.nodes[node_id]
        node.update_from_search_result(performance)
        print(f"  {node_id}: Updated attention = {node.get_total_attention():.3f}")
    
    print(f"\nAttention Network Statistics:")
    stats = manager.get_attention_statistics()
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Demonstrate attention spreading
    print(f"\nPerforming Attention Spreading...")
    manager._spread_attention()
    
    print(f"Post-spreading attention values:")
    for node_id in node_ids:
        node = manager.nodes[node_id]
        print(f"  {node_id}: {node.get_total_attention():.3f}")

def demonstrate_multiscale_optimization():
    """Demonstrate full multiscale optimization"""
    print("="*60) 
    print("3. MULTISCALE CONSTRAINT OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Initialize integrated system
    reducer = INCISearchSpaceReducer()
    attention_manager = AttentionAllocationManager()
    optimizer = MultiscaleConstraintOptimizer(reducer, attention_manager)
    
    # Configure multiscale constraints
    print(f"\nConfiguring Multiscale Constraints:")
    
    # Molecular scale constraints
    molecular_constraint = ScaleConstraint(
        scale=OptimizationScale.MOLECULAR,
        parameter="penetration_index",
        min_value=0.4,
        weight=0.8
    )
    optimizer.add_scale_constraint(molecular_constraint)
    print(f"  Molecular: Penetration index >= 0.4 (weight: 0.8)")
    
    # Cellular scale constraints
    cellular_constraint = ScaleConstraint(
        scale=OptimizationScale.CELLULAR,
        parameter="fibroblast_stimulation", 
        min_value=0.3,
        weight=1.0
    )
    optimizer.add_scale_constraint(cellular_constraint)
    print(f"  Cellular: Fibroblast stimulation >= 0.3 (weight: 1.0)")
    
    # Configure optimization objectives
    print(f"\nConfiguring Optimization Objectives:")
    objectives = [
        OptimizationObjective(ObjectiveType.EFFICACY, 0.35),
        OptimizationObjective(ObjectiveType.SAFETY, 0.25),
        OptimizationObjective(ObjectiveType.COST, 0.20, minimize=True),
        OptimizationObjective(ObjectiveType.STABILITY, 0.20)
    ]
    
    for obj in objectives:
        optimizer.add_objective(obj)
        direction = "minimize" if obj.minimize else "maximize"
        print(f"  {obj.objective_type.value.title()}: weight={obj.weight} ({direction})")
    
    # Define target formulation scenario
    target_inci = "Aqua, Retinol, Sodium Hyaluronate, Niacinamide, Tocopherol, Glycerin, Cetyl Alcohol, Phenoxyethanol"
    constraints = FormulationConstraint(
        target_ph=(5.5, 6.5),
        max_total_actives=12.0,
        required_ingredients=["water"]
    )
    
    print(f"\nTarget Scenario:")
    print(f"  Condition: Anti-aging")
    print(f"  INCI: {target_inci}")
    print(f"  pH Range: {constraints.target_ph}")
    print(f"  Max Actives: {constraints.max_total_actives}%")
    
    # Run optimization with smaller population for demo
    optimizer.population_size = 20
    optimizer.max_generations = 20
    
    print(f"\nRunning Multiscale Optimization...")
    print(f"  Population Size: {optimizer.population_size}")
    print(f"  Max Generations: {optimizer.max_generations}")
    
    results = optimizer.optimize_formulation(
        target_inci=target_inci,
        base_constraints=constraints,
        target_condition="anti_aging",
        max_time_minutes=2.0
    )
    
    # Display results
    print(f"\n" + "="*50)
    print(f"OPTIMIZATION RESULTS")
    print(f"="*50)
    
    if results:
        print(f"\nFound {len(results)} optimized formulations:")
        
        for i, candidate in enumerate(results[:3]):
            print(f"\nFormulation {i+1} (ID: {candidate.id}):")
            print(f"  Overall Fitness: {optimizer._calculate_fitness(candidate):.4f}")
            
            print(f"  Performance Scores:")
            print(f"    Efficacy: {candidate.efficacy_score:.3f}")
            print(f"    Safety: {candidate.safety_score:.3f}")
            print(f"    Cost: ${candidate.cost_estimate:.2f}")
            print(f"    Stability: {candidate.stability_score:.3f}")
            
            print(f"  Ingredients:")
            total_conc = 0
            for ingredient, conc in candidate.ingredients.items():
                print(f"    {ingredient}: {conc:.2f}%")
                total_conc += conc
            print(f"    Total: {total_conc:.2f}%")
            
            print(f"  Multiscale Properties:")
            if candidate.molecular_properties:
                print(f"    Molecular - Avg MW: {candidate.molecular_properties.get('average_molecular_weight', 0):.1f}")
                print(f"    Molecular - Penetration: {candidate.molecular_properties.get('penetration_index', 0):.3f}")
            if candidate.cellular_effects:
                print(f"    Cellular - Fibroblast: {candidate.cellular_effects.get('fibroblast_stimulation', 0):.3f}")
                print(f"    Cellular - Keratinocyte: {candidate.cellular_effects.get('keratinocyte_activation', 0):.3f}")
            if candidate.tissue_responses:
                print(f"    Tissue - Collagen: {candidate.tissue_responses.get('collagen_synthesis', 0):.3f}")
                print(f"    Tissue - Hydration: {candidate.tissue_responses.get('hydration_improvement', 0):.3f}")
            if candidate.organ_outcomes:
                print(f"    Organ - Health: {candidate.organ_outcomes.get('overall_skin_health', 0):.3f}")
    else:
        print(f"\nNo validated formulations found.")
        print(f"This may indicate overly restrictive constraints.")
    
    # Show optimization progress
    if optimizer.optimization_history:
        print(f"\nOptimization Progress:")
        initial_stats = optimizer.optimization_history[0]
        final_stats = optimizer.optimization_history[-1]
        
        print(f"  Initial -> Final Best Fitness: {initial_stats['best_fitness']:.4f} -> {final_stats['best_fitness']:.4f}")
        print(f"  Final Mean Fitness: {final_stats['mean_fitness']:.4f}")
        print(f"  Final Compliance Rate: {final_stats['regulatory_compliance_rate']:.1%}")
        print(f"  Generations Completed: {len(optimizer.optimization_history)}")

def demonstrate_regulatory_compliance():
    """Demonstrate regulatory compliance checking"""
    print("="*60)
    print("4. REGULATORY COMPLIANCE DEMONSTRATION")
    print("="*60)
    
    reducer = INCISearchSpaceReducer()
    
    # Test different formulation scenarios
    test_formulations = [
        {
            'name': 'Compliant Anti-Aging Serum',
            'formulation': {
                'ingredients': {
                    'water': 75.0,
                    'retinol': 0.5,
                    'hyaluronic_acid': 1.0,
                    'niacinamide': 5.0,
                    'glycerin': 3.0,
                    'phenoxyethanol': 0.8
                }
            }
        },
        {
            'name': 'Over-Limit Retinol Treatment',
            'formulation': {
                'ingredients': {
                    'water': 70.0,
                    'retinol': 1.5,  # Over 1% limit
                    'hyaluronic_acid': 1.0,
                    'glycerin': 3.0,
                    'phenoxyethanol': 0.8
                }
            }
        },
        {
            'name': 'High-Dose Vitamin C Serum',
            'formulation': {
                'ingredients': {
                    'water': 65.0,
                    'vitamin_c': 25.0,  # Over 20% limit
                    'hyaluronic_acid': 1.0,
                    'glycerin': 3.0,
                    'phenoxyethanol': 0.8
                }
            }
        },
        {
            'name': 'Edge-Case Formulation',
            'formulation': {
                'ingredients': {
                    'water': 73.2,
                    'retinol': 1.0,  # At limit
                    'vitamin_c': 20.0,  # At limit
                    'niacinamide': 10.0,  # At limit
                    'phenoxyethanol': 1.0  # At limit
                }
            }
        }
    ]
    
    print(f"\nTesting Regulatory Compliance:")
    
    for test_case in test_formulations:
        name = test_case['name']
        formulation = test_case['formulation']
        
        compliance = reducer.check_regulatory_compliance(formulation)
        status = "✓ COMPLIANT" if compliance else "✗ NON-COMPLIANT"
        
        print(f"\n  {name}: {status}")
        
        # Show ingredient details
        for ingredient, conc in formulation['ingredients'].items():
            limit_info = ""
            if ingredient in reducer.regulatory_limits:
                limit = reducer.regulatory_limits[ingredient]['max_concentration']
                if conc > limit:
                    limit_info = f" (EXCEEDS {limit}% LIMIT)"
                elif conc == limit:
                    limit_info = f" (AT {limit}% LIMIT)"
                else:
                    limit_info = f" (WITHIN {limit}% LIMIT)"
            
            print(f"    {ingredient}: {conc}%{limit_info}")

def main():
    """Main demonstration function"""
    print("OpenCog Features for Multiscale Constraint Optimization")
    print("in Cosmeceutical Formulation - Comprehensive Demonstration")
    print("="*80)
    
    try:
        demonstrate_inci_search_reduction()
        print("\n")
        
        demonstrate_attention_allocation()
        print("\n")
        
        demonstrate_regulatory_compliance()
        print("\n")
        
        demonstrate_multiscale_optimization()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        
        print(f"\nKey Achievements Demonstrated:")
        print(f"  ✓ INCI-driven search space reduction")
        print(f"  ✓ Adaptive attention allocation mechanisms")
        print(f"  ✓ Multiscale constraint optimization")
        print(f"  ✓ Regulatory compliance automation")
        print(f"  ✓ Integrated cognitive architecture approach")
        
        print(f"\nThis demonstration shows the successful integration of OpenCog-inspired")
        print(f"cognitive architecture components for next-generation cosmeceutical design.")
        
    except Exception as e:
        print(f"\nERROR in demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
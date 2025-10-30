#!/usr/bin/env python3
"""
OpenCog Integration Example for Cosmeceutical Formulation

This example demonstrates the complete OpenCog-inspired framework for
multiscale constraint optimization in cosmeceutical formulation.

Features demonstrated:
- AtomSpace hypergraph representation
- INCI-driven search space reduction
- Adaptive attention allocation
- PLN-inspired reasoning
- MOSES-inspired optimization
- Multiscale constraint satisfaction

Author: OpenCog Cheminformatics Team
License: MIT
"""

import sys
import os
from typing import List, Dict

# Add the cheminformatics module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cheminformatics.types.cosmetic_atoms import *
from cheminformatics.opencog_integration import *


def create_ingredient_knowledge_base(atomspace: CosmeceuticalAtomSpace) -> List[Atom]:
    """Create a knowledge base with cosmetic ingredients"""
    print("Creating ingredient knowledge base...")
    
    # Create ingredient atoms with properties
    ingredients_data = [
        {
            "name": "hyaluronic_acid",
            "category": "ACTIVE_INGREDIENT",
            "properties": {
                "molecular_weight": 1000000.0,
                "log_p": -3.0,
                "concentration_range": (0.1, 2.0),
                "functions": ["humectant", "anti_aging"],
                "cost_per_kg": 500.0
            }
        },
        {
            "name": "niacinamide", 
            "category": "ACTIVE_INGREDIENT",
            "properties": {
                "molecular_weight": 122.12,
                "log_p": -0.37,
                "concentration_range": (2.0, 10.0),
                "functions": ["anti_aging", "brightening"],
                "cost_per_kg": 80.0
            }
        },
        {
            "name": "vitamin_c",
            "category": "ACTIVE_INGREDIENT", 
            "properties": {
                "molecular_weight": 176.12,
                "log_p": -1.85,
                "concentration_range": (5.0, 20.0),
                "functions": ["antioxidant", "brightening"],
                "cost_per_kg": 150.0
            }
        },
        {
            "name": "retinol",
            "category": "ACTIVE_INGREDIENT",
            "properties": {
                "molecular_weight": 286.45,
                "log_p": 5.99,
                "concentration_range": (0.01, 1.0),
                "functions": ["anti_aging"],
                "cost_per_kg": 2000.0
            }
        },
        {
            "name": "glycerin",
            "category": "HUMECTANT",
            "properties": {
                "molecular_weight": 92.09,
                "log_p": -1.76,
                "concentration_range": (5.0, 30.0),
                "functions": ["humectant"],
                "cost_per_kg": 2.0
            }
        },
        {
            "name": "ceramides",
            "category": "EMOLLIENT", 
            "properties": {
                "molecular_weight": 600.0,
                "log_p": 8.0,
                "concentration_range": (0.1, 5.0),
                "functions": ["barrier_repair", "moisturizing"],
                "cost_per_kg": 800.0
            }
        }
    ]
    
    ingredient_atoms = []
    for data in ingredients_data:
        atom = atomspace.create_atom(
            AtomType.INGREDIENT_NODE,
            data["name"],
            properties=data["properties"]
        )
        ingredient_atoms.append(atom)
        print(f"  Created ingredient: {data['name']}")
    
    return ingredient_atoms


def establish_ingredient_relationships(atomspace: CosmeceuticalAtomSpace, 
                                     ingredients: List[Atom]) -> List[Atom]:
    """Establish compatibility and synergy relationships"""
    print("\nEstablishing ingredient relationships...")
    
    relationships = []
    
    # Get ingredients by name for easier reference
    ing_dict = {atom.name: atom for atom in ingredients}
    
    # Define compatibility relationships
    compatibility_pairs = [
        ("hyaluronic_acid", "niacinamide", 0.9),
        ("hyaluronic_acid", "glycerin", 0.95),
        ("niacinamide", "glycerin", 0.8),
        ("vitamin_c", "glycerin", 0.7),
        ("ceramides", "glycerin", 0.9),
        ("ceramides", "hyaluronic_acid", 0.85)
    ]
    
    for ing1_name, ing2_name, compatibility in compatibility_pairs:
        if ing1_name in ing_dict and ing2_name in ing_dict:
            link = atomspace.create_compatibility_link(
                ing_dict[ing1_name], ing_dict[ing2_name], compatibility
            )
            relationships.append(link)
            print(f"  Compatibility: {ing1_name} + {ing2_name} = {compatibility}")
    
    # Define synergy relationships
    synergy_pairs = [
        ("vitamin_c", "niacinamide", 0.8),  # Antioxidant + brightening synergy
        ("hyaluronic_acid", "ceramides", 0.9),  # Moisture retention synergy
        ("glycerin", "hyaluronic_acid", 0.7)  # Humectant synergy
    ]
    
    for ing1_name, ing2_name, synergy in synergy_pairs:
        if ing1_name in ing_dict and ing2_name in ing_dict:
            link = atomspace.create_synergy_link(
                ing_dict[ing1_name], ing_dict[ing2_name], synergy
            )
            relationships.append(link)
            print(f"  Synergy: {ing1_name} + {ing2_name} = {synergy}")
    
    # Define incompatibility (antagonism)
    incompatible_pairs = [
        ("vitamin_c", "retinol")  # pH incompatibility
    ]
    
    for ing1_name, ing2_name in incompatible_pairs:
        if ing1_name in ing_dict and ing2_name in ing_dict:
            link = atomspace.create_atom(
                AtomType.ANTAGONISM_LINK,
                f"incompatible_{ing1_name}_{ing2_name}",
                truth_value=0.9,
                outgoing=[ing_dict[ing1_name], ing_dict[ing2_name]]
            )
            relationships.append(link)
            print(f"  Incompatibility: {ing1_name} + {ing2_name}")
    
    return relationships


def demonstrate_inci_optimization(atomspace: CosmeceuticalAtomSpace):
    """Demonstrate INCI-driven search space optimization"""
    print("\n=== INCI-Driven Search Space Optimization ===")
    
    # Initialize INCI optimizer
    inci_optimizer = INCISearchOptimizer(atomspace)
    
    # Example product INCI list
    target_inci_list = [
        "AQUA", "GLYCERIN", "NIACINAMIDE", "HYALURONIC ACID", 
        "PHENOXYETHANOL", "CETYL ALCOHOL", "TOCOPHEROL"
    ]
    
    print(f"Target INCI list: {target_inci_list}")
    
    # Estimate concentrations from INCI ordering
    estimated_concentrations = inci_optimizer.estimate_concentrations_from_inci(target_inci_list)
    print("\nEstimated concentrations:")
    for ingredient, concentration in estimated_concentrations.items():
        print(f"  {ingredient}: {concentration:.2f}%")
    
    # Generate optimized combinations for anti-aging
    target_functions = ["anti_aging", "moisturizing", "antioxidant"]
    combinations = inci_optimizer.generate_optimized_inci_combinations(
        target_inci_list, target_functions, RegulationRegion.EU, max_ingredients=6
    )
    
    print(f"\nOptimized combinations for {target_functions}:")
    for i, combination in enumerate(combinations):
        print(f"  Combination {i+1}: {combination}")
    
    # Check regulatory compliance
    eu_compliant = inci_optimizer.filter_by_regulatory_compliance(target_inci_list, RegulationRegion.EU)
    print(f"\nEU compliant ingredients: {eu_compliant}")
    
    stats = inci_optimizer.get_statistics()
    print(f"\nINCI Optimizer Statistics: {stats}")


def demonstrate_attention_allocation(atomspace: CosmeceuticalAtomSpace, 
                                   ingredients: List[Atom]):
    """Demonstrate adaptive attention allocation"""
    print("\n=== Adaptive Attention Allocation ===")
    
    # Initialize attention allocator
    attention_allocator = AdaptiveAttentionAllocator(atomspace)
    
    # Allocate attention to high-value ingredients
    high_value_ingredients = [atom for atom in ingredients if "anti_aging" in atom.properties.get("functions", [])]
    
    print(f"Focusing attention on {len(high_value_ingredients)} anti-aging ingredients...")
    
    allocations = attention_allocator.allocate_attention(
        high_value_ingredients, 
        strategy="synergy_based",
        sti_budget=200.0,
        lti_budget=100.0
    )
    
    print("Attention allocations:")
    for atom_id, (sti, lti) in allocations.items():
        atom = atomspace.atoms[atom_id]
        print(f"  {atom.name}: STI={sti:.2f}, LTI={lti:.2f}")
    
    # Spread attention through the network
    attention_allocator.spread_attention(high_value_ingredients, max_spread_distance=2)
    
    # Get high attention atoms
    high_attention_atoms = attention_allocator.get_high_attention_atoms(
        attention_type=AttentionType.SHORT_TERM, count=5
    )
    
    print("\nTop 5 high-attention atoms:")
    for atom, attention_score in high_attention_atoms:
        print(f"  {atom.name} ({atom.atom_type.value}): {attention_score:.3f}")
    
    # Get promising ingredient combinations
    promising_combinations = attention_allocator.get_promising_ingredient_combinations(
        min_attention=0.1, max_combinations=3
    )
    
    print(f"\nPromising ingredient combinations ({len(promising_combinations)}):")
    for i, combination in enumerate(promising_combinations):
        combo_names = [atom.name for atom in combination]
        print(f"  Combination {i+1}: {combo_names}")
    
    stats = attention_allocator.get_attention_statistics()
    print(f"\nAttention Allocator Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demonstrate_reasoning_engine(atomspace: CosmeceuticalAtomSpace):
    """Demonstrate PLN-inspired reasoning"""
    print("\n=== PLN-Inspired Reasoning Engine ===")
    
    # Initialize reasoning engine
    reasoning_engine = IngredientReasoningEngine(atomspace)
    
    # Evaluate ingredient compatibility using reasoning
    test_pairs = [
        ("hyaluronic_acid", "niacinamide"),
        ("vitamin_c", "retinol"),
        ("glycerin", "ceramides"),
        ("niacinamide", "unknown_ingredient")  # Test inference
    ]
    
    print("Ingredient compatibility evaluation:")
    for ing1, ing2 in test_pairs:
        truth_value = reasoning_engine.evaluate_ingredient_compatibility(ing1, ing2)
        print(f"  {ing1} + {ing2}: strength={truth_value.strength:.3f}, confidence={truth_value.confidence:.3f}")
    
    # Run inference cycle to derive new knowledge
    print("\nRunning inference cycle...")
    new_atoms = reasoning_engine.run_inference_cycle(max_iterations=3)
    print(f"Derived {len(new_atoms)} new atoms from inference:")
    
    for atom in new_atoms[:5]:  # Show first 5
        print(f"  {atom.name} ({atom.atom_type.value}): truth={atom.truth_value:.3f}")
    
    # Add experimental feedback
    feedback_data = {
        "hyaluronic_acid_niacinamide": {"compatibility": 0.95, "confidence": 0.9},
        "vitamin_c_retinol": {"compatibility": 0.1, "confidence": 0.8}
    }
    
    print("\nUpdating with experimental feedback...")
    reasoning_engine.update_truth_values_from_feedback(feedback_data)
    
    # Re-evaluate after feedback
    for ing1, ing2 in [("hyaluronic_acid", "niacinamide"), ("vitamin_c", "retinol")]:
        truth_value = reasoning_engine.evaluate_ingredient_compatibility(ing1, ing2)
        print(f"  Updated {ing1} + {ing2}: strength={truth_value.strength:.3f}, confidence={truth_value.confidence:.3f}")
    
    stats = reasoning_engine.get_reasoning_statistics()
    print(f"\nReasoning Engine Statistics: {stats}")


def demonstrate_multiscale_optimization(atomspace: CosmeceuticalAtomSpace):
    """Demonstrate multiscale constraint optimization"""
    print("\n=== Multiscale Constraint Optimization ===")
    
    # Initialize components
    reasoning_engine = IngredientReasoningEngine(atomspace)
    optimizer = MultiscaleOptimizer(atomspace, reasoning_engine)
    skin_integrator = SkinModelIntegrator(atomspace, reasoning_engine, optimizer)
    
    # Define optimization targets
    target_therapeutic_vectors = ["anti_aging", "moisturizing", "brightening"]
    available_ingredients = ["hyaluronic_acid", "niacinamide", "vitamin_c", "glycerin", "ceramides"]
    
    print(f"Optimizing for therapeutic vectors: {target_therapeutic_vectors}")
    print(f"Available ingredients: {available_ingredients}")
    
    # Set up constraints
    constraints = {
        "max_total_concentration": 95.0,  # Leave 5% for water/preservatives
        "max_ingredients": 6,
        "required_ingredients": ["glycerin"],  # Base moisturizer
        "prohibited_ingredients": ["retinol"]  # Avoid incompatibility with vitamin C
    }
    
    print(f"Optimization constraints: {constraints}")
    
    # Run multiscale optimization
    print("\nRunning multiscale optimization...")
    optimal_solutions = skin_integrator.optimize_formulation_for_therapeutic_vectors(
        target_therapeutic_vectors, available_ingredients, constraints
    )
    
    print(f"\nFound {len(optimal_solutions)} optimal solutions:")
    
    for i, solution in enumerate(optimal_solutions[:3]):  # Show top 3 solutions
        print(f"\nSolution {i+1}:")
        print(f"  Genome ID: {solution.genome_id}")
        print(f"  Generation: {solution.generation}")
        print(f"  Total concentration: {solution.total_concentration:.2f}%")
        
        print("  Ingredients:")
        for ingredient, concentration in solution.ingredients.items():
            print(f"    {ingredient}: {concentration:.2f}%")
        
        print("  Fitness scores:")
        for objective, score in solution.fitness_scores.items():
            print(f"    {objective.value}: {score:.2f}")
        
        # Evaluate therapeutic vector achievement
        achievement = skin_integrator.evaluate_therapeutic_vector_achievement(
            solution, target_therapeutic_vectors
        )
        print("  Therapeutic vector achievement:")
        for vector, score in achievement.items():
            print(f"    {vector}: {score:.3f}")
        
        # Show penetration data if available
        if "penetration_profiles" in solution.properties:
            print("  Penetration profiles:")
            for ingredient, profile_data in solution.properties["penetration_profiles"].items():
                print(f"    {ingredient}: depth={profile_data['penetration_depth']:.1f}μm, "
                      f"bioavailability={profile_data['bioavailability']:.3f}")
    
    # Show optimization statistics
    opt_stats = optimizer.get_optimization_statistics()
    print(f"\nOptimization Statistics:")
    for key, value in opt_stats.items():
        print(f"  {key}: {value}")
    
    # Show multiscale system statistics
    multiscale_stats = skin_integrator.get_multiscale_statistics()
    print(f"\nMultiscale System Statistics:")
    for key, value in multiscale_stats.items():
        print(f"  {key}: {value}")


def demonstrate_penetration_modeling(atomspace: CosmeceuticalAtomSpace):
    """Demonstrate skin penetration modeling"""
    print("\n=== Skin Penetration Modeling ===")
    
    # Initialize skin model integrator
    reasoning_engine = IngredientReasoningEngine(atomspace)
    optimizer = MultiscaleOptimizer(atomspace, reasoning_engine)
    skin_integrator = SkinModelIntegrator(atomspace, reasoning_engine, optimizer)
    
    # Test ingredients with different properties
    test_ingredients = [
        ("hyaluronic_acid", 2.0),  # Large hydrophilic molecule
        ("niacinamide", 5.0),      # Small hydrophilic molecule
        ("vitamin_c", 10.0),       # Small hydrophilic antioxidant
        ("ceramides", 1.0)         # Large lipophilic molecule
    ]
    
    print("Calculating penetration profiles:")
    
    for ingredient_name, concentration in test_ingredients:
        print(f"\n{ingredient_name} ({concentration}%):")
        
        # Calculate penetration profile
        profile = skin_integrator.calculate_ingredient_penetration_profile(
            ingredient_name, concentration, DeliveryMechanism.PASSIVE_DIFFUSION
        )
        
        print(f"  Molecular weight: {profile.molecular_weight:.1f} Da")
        print(f"  Log P: {profile.log_p:.2f}")
        print(f"  Maximum penetration depth: {profile.penetration_depth:.1f} μm")
        print(f"  Overall bioavailability: {profile.bioavailability:.3f}")
        
        print("  Layer concentrations:")
        for layer, conc in profile.concentration_profile.items():
            print(f"    {layer.value}: {conc:.4f}%")
        
        # Compare with enhanced delivery
        enhanced_profile = skin_integrator.calculate_ingredient_penetration_profile(
            ingredient_name, concentration, DeliveryMechanism.LIPOSOMAL_DELIVERY
        )
        
        enhancement_factor = enhanced_profile.bioavailability / profile.bioavailability if profile.bioavailability > 0 else 1.0
        print(f"  Liposomal delivery enhancement factor: {enhancement_factor:.2f}x")


def main():
    """Main demonstration function"""
    print("=== OpenCog-Inspired Cosmeceutical Formulation Framework ===")
    print("Demonstrating multiscale constraint optimization with cognitive architectures\n")
    
    # Initialize AtomSpace
    atomspace = CosmeceuticalAtomSpace()
    print(f"Initialized AtomSpace with {atomspace.get_statistics()['total_atoms']} atoms")
    
    # Create knowledge base
    ingredients = create_ingredient_knowledge_base(atomspace)
    relationships = establish_ingredient_relationships(atomspace, ingredients)
    
    print(f"\nKnowledge base created:")
    print(f"  Ingredients: {len(ingredients)}")
    print(f"  Relationships: {len(relationships)}")
    
    # Demonstrate each component
    demonstrate_inci_optimization(atomspace)
    demonstrate_attention_allocation(atomspace, ingredients)
    demonstrate_reasoning_engine(atomspace)
    demonstrate_penetration_modeling(atomspace)
    demonstrate_multiscale_optimization(atomspace)
    
    # Final AtomSpace statistics
    final_stats = atomspace.get_statistics()
    print(f"\n=== Final AtomSpace Statistics ===")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("✓ AtomSpace hypergraph representation of cosmetic ingredients")
    print("✓ INCI-driven search space reduction and regulatory compliance")
    print("✓ ECAN-inspired adaptive attention allocation")
    print("✓ PLN-inspired probabilistic reasoning over ingredient interactions")
    print("✓ MOSES-inspired evolutionary optimization")
    print("✓ Multiscale constraint satisfaction across skin layers")
    print("✓ Skin penetration modeling and delivery optimization")
    print("✓ Therapeutic vector-driven formulation optimization")
    print("\nThe framework successfully integrates OpenCog's cognitive architecture")
    print("concepts with practical cosmeceutical formulation science!")


if __name__ == "__main__":
    main()
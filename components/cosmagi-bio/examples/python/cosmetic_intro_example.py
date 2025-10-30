#!/usr/bin/env python3
"""
Cosmetic Chemistry Introduction Example

This example demonstrates the basic usage of cosmetic chemistry atom types
in the OpenCog framework. It shows how to create and work with cosmetic
ingredients, define their properties, and model simple formulations.

Requirements:
- OpenCog AtomSpace with bioscience extensions
- Python OpenCog bindings

Usage:
    python3 cosmetic_intro_example.py

Author: OpenCog Cosmetic Chemistry Framework
License: AGPL-3.0
"""

from opencog.atomspace import AtomSpace, types
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog

def initialize_cosmetic_atomspace():
    """Initialize the AtomSpace with cosmetic chemistry types."""
    atomspace = AtomSpace()
    initialize_opencog(atomspace)
    
    # Load bioscience types including cosmetic chemistry extensions
    try:
        import opencog.bioscience
        print("✓ Bioscience module loaded successfully")
    except ImportError:
        print("⚠ Warning: Bioscience module not available")
        print("  Using standard atom types for demonstration")
    
    return atomspace

def demonstrate_ingredient_modeling(atomspace):
    """Demonstrate how to model cosmetic ingredients."""
    print("\n=== Ingredient Modeling ===")
    
    # Create basic cosmetic ingredients with functional classifications
    print("Creating cosmetic ingredients:")
    
    # Active ingredients
    hyaluronic_acid = ConceptNode("hyaluronic_acid")
    niacinamide = ConceptNode("niacinamide")
    retinol = ConceptNode("retinol")
    
    # Supporting ingredients
    glycerin = ConceptNode("glycerin")
    cetyl_alcohol = ConceptNode("cetyl_alcohol")
    phenoxyethanol = ConceptNode("phenoxyethanol")
    xanthan_gum = ConceptNode("xanthan_gum")
    
    print(f"  • Hyaluronic Acid: {hyaluronic_acid}")
    print(f"  • Niacinamide: {niacinamide}")
    print(f"  • Glycerin: {glycerin}")
    print(f"  • Cetyl Alcohol: {cetyl_alcohol}")
    print(f"  • Phenoxyethanol: {phenoxyethanol}")
    
    # Create functional classifications using inheritance links
    print("\nClassifying ingredients by function:")
    
    # Active ingredients
    InheritanceLink(hyaluronic_acid, ConceptNode("ACTIVE_INGREDIENT"))
    InheritanceLink(niacinamide, ConceptNode("ACTIVE_INGREDIENT"))
    InheritanceLink(retinol, ConceptNode("ACTIVE_INGREDIENT"))
    
    # Humectants
    InheritanceLink(glycerin, ConceptNode("HUMECTANT"))
    InheritanceLink(hyaluronic_acid, ConceptNode("HUMECTANT"))  # Dual function
    
    # Emulsifiers
    InheritanceLink(cetyl_alcohol, ConceptNode("EMULSIFIER"))
    
    # Preservatives
    InheritanceLink(phenoxyethanol, ConceptNode("PRESERVATIVE"))
    
    # Thickeners
    InheritanceLink(xanthan_gum, ConceptNode("THICKENER"))
    
    print("  ✓ Active ingredients classified")
    print("  ✓ Functional ingredients classified")
    
    return {
        'hyaluronic_acid': hyaluronic_acid,
        'niacinamide': niacinamide,
        'retinol': retinol,
        'glycerin': glycerin,
        'cetyl_alcohol': cetyl_alcohol,
        'phenoxyethanol': phenoxyethanol,
        'xanthan_gum': xanthan_gum
    }

def demonstrate_formulation_creation(atomspace, ingredients):
    """Demonstrate how to create cosmetic formulations."""
    print("\n=== Formulation Creation ===")
    
    # Create a simple moisturizer formulation
    moisturizer = ConceptNode("hydrating_moisturizer")
    
    print("Creating hydrating moisturizer formulation:")
    print("  Components:")
    
    # Define formulation components with concentrations
    components = [
        (ingredients['hyaluronic_acid'], 2.0, "hydrating active"),
        (ingredients['niacinamide'], 5.0, "barrier support active"),
        (ingredients['glycerin'], 10.0, "humectant"),
        (ingredients['cetyl_alcohol'], 3.0, "emulsifier"),
        (ingredients['phenoxyethanol'], 0.8, "preservative"),
        (ingredients['xanthan_gum'], 0.3, "thickener")
    ]
    
    for ingredient, concentration, function in components:
        # Create evaluation links for concentration
        EvaluationLink(
            PredicateNode("concentration"),
            ListLink(
                moisturizer,
                ingredient,
                NumberNode(str(concentration))
            )
        )
        
        # Create evaluation links for function
        EvaluationLink(
            PredicateNode("ingredient_function"),
            ListLink(
                moisturizer,
                ingredient,
                ConceptNode(function)
            )
        )
        
        print(f"    - {ingredient.name}: {concentration}% ({function})")
    
    # Classify the formulation
    InheritanceLink(moisturizer, ConceptNode("SKINCARE_FORMULATION"))
    
    print(f"  ✓ Formulation created: {moisturizer}")
    
    return moisturizer

def demonstrate_ingredient_interactions(atomspace, ingredients):
    """Demonstrate modeling of ingredient interactions."""
    print("\n=== Ingredient Interactions ===")
    
    # Define compatible combinations
    print("Compatible ingredient pairs:")
    
    compatible_pairs = [
        (ingredients['hyaluronic_acid'], ingredients['niacinamide'], "Enhanced hydration and barrier function"),
        (ingredients['niacinamide'], ingredients['glycerin'], "Complementary hydrating effects"),
        (ingredients['cetyl_alcohol'], ingredients['glycerin'], "Stable emulsion formation")
    ]
    
    for ing1, ing2, description in compatible_pairs:
        compatibility = EvaluationLink(
            PredicateNode("compatible_with"),
            ListLink(ing1, ing2)
        )
        
        # Add description
        EvaluationLink(
            PredicateNode("interaction_description"),
            ListLink(compatibility, ConceptNode(description))
        )
        
        print(f"  ✓ {ing1.name} + {ing2.name}: {description}")
    
    # Define incompatible combinations
    print("\nIncompatible ingredient pairs:")
    
    vitamin_c = ConceptNode("vitamin_c")
    InheritanceLink(vitamin_c, ConceptNode("ACTIVE_INGREDIENT"))
    
    incompatible_pairs = [
        (vitamin_c, ingredients['retinol'], "pH incompatibility and potential irritation"),
        (ConceptNode("benzoyl_peroxide"), ingredients['retinol'], "Oxidative degradation")
    ]
    
    for ing1, ing2, reason in incompatible_pairs:
        incompatibility = EvaluationLink(
            PredicateNode("incompatible_with"),
            ListLink(ing1, ing2)
        )
        
        # Add reason
        EvaluationLink(
            PredicateNode("incompatibility_reason"),
            ListLink(incompatibility, ConceptNode(reason))
        )
        
        print(f"  ⚠ {ing1.name} + {ing2.name}: {reason}")
    
    # Define synergistic combinations
    print("\nSynergistic ingredient pairs:")
    
    vitamin_e = ConceptNode("vitamin_e")
    InheritanceLink(vitamin_e, ConceptNode("ANTIOXIDANT"))
    
    synergistic_pairs = [
        (vitamin_c, vitamin_e, "Enhanced antioxidant stability and efficacy")
    ]
    
    for ing1, ing2, benefit in synergistic_pairs:
        synergy = EvaluationLink(
            PredicateNode("synergistic_with"),
            ListLink(ing1, ing2)
        )
        
        # Add benefit description
        EvaluationLink(
            PredicateNode("synergy_benefit"),
            ListLink(synergy, ConceptNode(benefit))
        )
        
        print(f"  ⚡ {ing1.name} + {ing2.name}: {benefit}")

def demonstrate_property_modeling(atomspace, formulation):
    """Demonstrate modeling of formulation properties."""
    print("\n=== Property Modeling ===")
    
    print("Defining formulation properties:")
    
    # pH properties
    ph_property = EvaluationLink(
        PredicateNode("pH"),
        ListLink(formulation, NumberNode("5.5"))
    )
    print("  • pH: 5.5 (skin-compatible)")
    
    # Viscosity properties
    viscosity_property = EvaluationLink(
        PredicateNode("viscosity"),
        ListLink(formulation, ConceptNode("medium"))
    )
    print("  • Viscosity: Medium (easy application)")
    
    # Stability properties
    stability_property = EvaluationLink(
        PredicateNode("stability_months"),
        ListLink(formulation, NumberNode("24"))
    )
    print("  • Stability: 24 months (long shelf life)")
    
    # Texture properties
    texture_property = EvaluationLink(
        PredicateNode("texture"),
        ListLink(formulation, ConceptNode("lightweight_cream"))
    )
    print("  • Texture: Lightweight cream (pleasant feel)")
    
    # Skin type suitability
    suitability = EvaluationLink(
        PredicateNode("suitable_for"),
        ListLink(formulation, ConceptNode("all_skin_types"))
    )
    print("  • Suitable for: All skin types")
    
    print("  ✓ Properties defined successfully")

def demonstrate_safety_assessment(atomspace, ingredients):
    """Demonstrate safety and regulatory assessment."""
    print("\n=== Safety Assessment ===")
    
    print("Evaluating ingredient safety:")
    
    # Define safety profiles for ingredients
    safety_data = [
        (ingredients['hyaluronic_acid'], "GRAS", "No known allergens", "Up to 5%"),
        (ingredients['niacinamide'], "GRAS", "No known allergens", "Up to 10%"),
        (ingredients['glycerin'], "GRAS", "No known allergens", "Up to 20%"),
        (ingredients['phenoxyethanol'], "Safe_in_cosmetics", "Rare sensitization", "Up to 1%")
    ]
    
    for ingredient, safety_status, allergen_info, max_concentration in safety_data:
        # Safety status
        EvaluationLink(
            PredicateNode("safety_status"),
            ListLink(ingredient, ConceptNode(safety_status))
        )
        
        # Allergen information
        EvaluationLink(
            PredicateNode("allergen_status"),
            ListLink(ingredient, ConceptNode(allergen_info))
        )
        
        # Maximum safe concentration
        EvaluationLink(
            PredicateNode("max_concentration"),
            ListLink(ingredient, ConceptNode(max_concentration))
        )
        
        print(f"  • {ingredient.name}: {safety_status}, {allergen_info}, Max: {max_concentration}")
    
    print("  ✓ Safety profiles established")

def print_atomspace_summary(atomspace):
    """Print a summary of the atomspace contents."""
    print("\n=== AtomSpace Summary ===")
    print(f"Total atoms created: {len(atomspace)}")
    
    # Count atoms by type
    type_counts = {}
    for atom in atomspace:
        atom_type = atom.type_name
        type_counts[atom_type] = type_counts.get(atom_type, 0) + 1
    
    print("Atom type distribution:")
    for atom_type, count in sorted(type_counts.items()):
        print(f"  • {atom_type}: {count}")

def main():
    """Main function demonstrating cosmetic chemistry concepts."""
    print("🧴 Cosmetic Chemistry Introduction Example")
    print("==========================================")
    
    # Initialize the atomspace
    atomspace = initialize_cosmetic_atomspace()
    
    # Demonstrate core concepts
    ingredients = demonstrate_ingredient_modeling(atomspace)
    formulation = demonstrate_formulation_creation(atomspace, ingredients)
    demonstrate_ingredient_interactions(atomspace, ingredients)
    demonstrate_property_modeling(atomspace, formulation)
    demonstrate_safety_assessment(atomspace, ingredients)
    
    # Print summary
    print_atomspace_summary(atomspace)
    
    print("\n✓ Example completed successfully!")
    print("\nNext steps:")
    print("  • Run cosmetic_chemistry_example.py for advanced features")
    print("  • Explore Scheme examples in examples/scheme/")
    print("  • Read docs/COSMETIC_CHEMISTRY.md for comprehensive reference")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Cosmetic Chemistry Introduction Example

This script demonstrates basic usage of the OpenCog cheminformatics framework
for cosmetic chemistry applications, including ingredient modeling, formulation
creation, and compatibility analysis.

Prerequisites:
- OpenCog Python bindings
- OpenCog AtomSpace
- Cosmetic chemistry atom types loaded
"""

from opencog.type_constructors import *
from opencog.atomspace import AtomSpace
from opencog.utilities import initialize_opencog

def setup_atomspace():
    """Initialize AtomSpace and load cosmetic chemistry types."""
    atomspace = AtomSpace()
    initialize_opencog(atomspace)
    
    # Load cosmetic chemistry atom types
    # In practice, this would load from the atom_types.script file
    print("Loading cosmetic chemistry atom types...")
    
    return atomspace

def create_basic_ingredients(atomspace):
    """Create basic cosmetic ingredients using specialized atom types."""
    print("\n=== Creating Basic Cosmetic Ingredients ===")
    
    # Active ingredients
    hyaluronic_acid = ConceptNode("hyaluronic_acid")
    niacinamide = ConceptNode("niacinamide")  
    retinol = ConceptNode("retinol")
    vitamin_c = ConceptNode("vitamin_c")
    
    # Supporting ingredients
    glycerin = ConceptNode("glycerin")
    cetyl_alcohol = ConceptNode("cetyl_alcohol")
    phenoxyethanol = ConceptNode("phenoxyethanol")
    xanthan_gum = ConceptNode("xanthan_gum")
    
    # Define ingredient classifications
    # Note: In actual implementation, these would use the specialized atom types
    InheritanceLink(hyaluronic_acid, ConceptNode("ACTIVE_INGREDIENT"))
    InheritanceLink(niacinamide, ConceptNode("ACTIVE_INGREDIENT"))
    InheritanceLink(retinol, ConceptNode("ACTIVE_INGREDIENT"))
    InheritanceLink(vitamin_c, ConceptNode("ACTIVE_INGREDIENT"))
    
    InheritanceLink(glycerin, ConceptNode("HUMECTANT"))
    InheritanceLink(cetyl_alcohol, ConceptNode("EMULSIFIER"))
    InheritanceLink(phenoxyethanol, ConceptNode("PRESERVATIVE"))
    InheritanceLink(xanthan_gum, ConceptNode("THICKENER"))
    
    print("✓ Created basic cosmetic ingredients")
    print("✓ Classified ingredients by functional category")
    
    return {
        'hyaluronic_acid': hyaluronic_acid,
        'niacinamide': niacinamide,
        'retinol': retinol,
        'vitamin_c': vitamin_c,
        'glycerin': glycerin,
        'cetyl_alcohol': cetyl_alcohol,
        'phenoxyethanol': phenoxyethanol,
        'xanthan_gum': xanthan_gum
    }

def create_formulation(atomspace, ingredients):
    """Create a basic moisturizing formulation."""
    print("\n=== Creating Moisturizing Formulation ===")
    
    # Create formulation concept
    moisturizer = ConceptNode("basic_moisturizer")
    InheritanceLink(moisturizer, ConceptNode("SKINCARE_FORMULATION"))
    
    # Add ingredients to formulation with concentrations
    formulation_ingredients = [
        (ingredients['hyaluronic_acid'], 1.0),  # 1% hyaluronic acid
        (ingredients['niacinamide'], 5.0),      # 5% niacinamide  
        (ingredients['glycerin'], 8.0),         # 8% glycerin
        (ingredients['cetyl_alcohol'], 3.0),    # 3% cetyl alcohol
        (ingredients['phenoxyethanol'], 0.5),   # 0.5% preservative
        (ingredients['xanthan_gum'], 0.2)       # 0.2% thickener
    ]
    
    for ingredient, concentration in formulation_ingredients:
        # Link ingredient to formulation
        MemberLink(ingredient, moisturizer)
        
        # Specify concentration
        ExecutionLink(
            SchemaNode("concentration"),
            ListLink(moisturizer, ingredient),
            NumberNode(str(concentration))
        )
    
    print("✓ Created basic moisturizer formulation")
    print("✓ Added ingredients with specified concentrations")
    
    # Add formulation properties
    properties = [
        ConceptNode("pH_5.5"),
        ConceptNode("non_greasy"),
        ConceptNode("quick_absorption"),
        ConceptNode("suitable_all_skin_types")
    ]
    
    for prop in properties:
        EvaluationLink(
            PredicateNode("has_property"),
            ListLink(moisturizer, prop)
        )
    
    print("✓ Added formulation properties")
    
    return moisturizer

def analyze_compatibility(atomspace, ingredients):
    """Analyze ingredient compatibility relationships."""
    print("\n=== Analyzing Ingredient Compatibility ===")
    
    # Define compatible ingredient pairs
    compatible_pairs = [
        (ingredients['hyaluronic_acid'], ingredients['niacinamide']),
        (ingredients['hyaluronic_acid'], ingredients['glycerin']),
        (ingredients['niacinamide'], ingredients['glycerin']),
        (ingredients['cetyl_alcohol'], ingredients['glycerin'])
    ]
    
    for ingredient1, ingredient2 in compatible_pairs:
        EvaluationLink(
            PredicateNode("compatible"),
            ListLink(ingredient1, ingredient2)
        )
    
    print("✓ Defined compatible ingredient relationships")
    
    # Define incompatible ingredient pairs
    incompatible_pairs = [
        (ingredients['vitamin_c'], ingredients['retinol']),  # pH and stability issues
    ]
    
    for ingredient1, ingredient2 in incompatible_pairs:
        EvaluationLink(
            PredicateNode("incompatible"),
            ListLink(ingredient1, ingredient2)
        )
    
    print("✓ Defined incompatible ingredient relationships")
    
    # Define synergistic relationships
    synergistic_pairs = [
        # Note: vitamin_e would be needed for this synergy
        # (ingredients['vitamin_c'], vitamin_e),  # Antioxidant synergy
    ]
    
    print("✓ Analyzed ingredient interactions")

def demonstrate_queries(atomspace, ingredients, formulation):
    """Demonstrate various queries on the cosmetic chemistry knowledge base."""
    print("\n=== Demonstrating Knowledge Base Queries ===")
    
    # Query 1: Find all active ingredients
    print("\n1. Finding all active ingredients:")
    active_ingredients = atomspace.get_atoms_by_type(ConceptNode)
    actives = []
    
    for atom in active_ingredients:
        inheritance_links = atomspace.get_incoming(atom)
        for link in inheritance_links:
            if (link.type == InheritanceLink and 
                len(link.out) == 2 and 
                link.out[1].name == "ACTIVE_INGREDIENT"):
                actives.append(atom.name)
                break
    
    print(f"   Active ingredients found: {actives}")
    
    # Query 2: Find ingredients in formulation
    print("\n2. Finding ingredients in moisturizer formulation:")
    formulation_members = atomspace.get_incoming(formulation)
    formulation_ingredients = []
    
    for link in formulation_members:
        if link.type == MemberLink and len(link.out) == 2:
            ingredient = link.out[0]
            formulation_ingredients.append(ingredient.name)
    
    print(f"   Formulation ingredients: {formulation_ingredients}")
    
    # Query 3: Find compatible ingredient pairs
    print("\n3. Finding compatible ingredient pairs:")
    evaluation_links = atomspace.get_atoms_by_type(EvaluationLink)
    compatible_pairs = []
    
    for link in evaluation_links:
        if (len(link.out) == 2 and 
            link.out[0].name == "compatible" and
            link.out[1].type == ListLink):
            pair = link.out[1].out
            if len(pair) == 2:
                compatible_pairs.append((pair[0].name, pair[1].name))
    
    print(f"   Compatible pairs found: {compatible_pairs}")

def main():
    """Main demonstration function."""
    print("Cosmetic Chemistry Framework - Introduction Example")
    print("=" * 55)
    
    # Setup
    atomspace = setup_atomspace()
    
    # Create ingredients
    ingredients = create_basic_ingredients(atomspace)
    
    # Create formulation
    formulation = create_formulation(atomspace, ingredients)
    
    # Analyze compatibility
    analyze_compatibility(atomspace, ingredients)
    
    # Demonstrate queries
    demonstrate_queries(atomspace, ingredients, formulation)
    
    # Summary
    print("\n=== Summary ===")
    print("✓ Successfully demonstrated basic cosmetic chemistry framework usage")
    print("✓ Created ingredient classifications and formulations")  
    print("✓ Analyzed ingredient compatibility relationships")
    print("✓ Performed knowledge base queries")
    print("\nNext steps:")
    print("- Explore advanced formulation optimization")
    print("- Add more complex ingredient interaction modeling")
    print("- Implement regulatory compliance checking")
    print("- See cosmetic_chemistry_example.py for advanced features")

if __name__ == "__main__":
    main()
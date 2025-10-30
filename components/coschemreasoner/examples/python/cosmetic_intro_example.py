#!/usr/bin/env python3
"""
Cosmetic Chemistry Introduction Example

This example demonstrates the basic usage of cosmetic-specific atom types
in the OpenCog cheminformatics framework. It covers ingredient modeling,
basic formulation creation, and simple compatibility checking.

Author: OpenCog Cheminformatics Team
License: MIT
"""

import sys
import os

# Add the cheminformatics module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class CosmeticAtom:
    """Base class for cosmetic atoms"""
    def __init__(self, name, atom_type):
        self.name = name
        self.atom_type = atom_type
        self.properties = {}
    
    def __str__(self):
        return f"{self.atom_type}('{self.name}')"
    
    def __repr__(self):
        return self.__str__()

# Ingredient Category Classes
class ACTIVE_INGREDIENT(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "ACTIVE_INGREDIENT")

class PRESERVATIVE(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "PRESERVATIVE")

class EMULSIFIER(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "EMULSIFIER")

class HUMECTANT(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "HUMECTANT")

class SURFACTANT(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "SURFACTANT")

class THICKENER(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "THICKENER")

class EMOLLIENT(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "EMOLLIENT")

class ANTIOXIDANT(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "ANTIOXIDANT")

class UV_FILTER(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "UV_FILTER")

class FRAGRANCE(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "FRAGRANCE")

class COLORANT(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "COLORANT")

class PH_ADJUSTER(CosmeticAtom):
    def __init__(self, name):
        super().__init__(name, "PH_ADJUSTER")

# Formulation Classes
class SKINCARE_FORMULATION:
    def __init__(self, *ingredients):
        self.ingredients = list(ingredients)
        self.properties = {}
    
    def add_ingredient(self, ingredient):
        self.ingredients.append(ingredient)
    
    def get_ingredients_by_type(self, atom_type):
        return [ing for ing in self.ingredients if ing.atom_type == atom_type]
    
    def __str__(self):
        return f"SKINCARE_FORMULATION({len(self.ingredients)} ingredients)"

class HAIRCARE_FORMULATION:
    def __init__(self, *ingredients):
        self.ingredients = list(ingredients)
        self.properties = {}

class MAKEUP_FORMULATION:
    def __init__(self, *ingredients):
        self.ingredients = list(ingredients)
        self.properties = {}

class FRAGRANCE_FORMULATION:
    def __init__(self, *ingredients):
        self.ingredients = list(ingredients)
        self.properties = {}

# Link Classes
class COMPATIBILITY_LINK:
    def __init__(self, ingredient1, ingredient2):
        self.ingredient1 = ingredient1
        self.ingredient2 = ingredient2
    
    def __str__(self):
        return f"COMPATIBLE: {self.ingredient1.name} <-> {self.ingredient2.name}"

class INCOMPATIBILITY_LINK:
    def __init__(self, ingredient1, ingredient2):
        self.ingredient1 = ingredient1
        self.ingredient2 = ingredient2
    
    def __str__(self):
        return f"INCOMPATIBLE: {self.ingredient1.name} <-> {self.ingredient2.name}"

class SYNERGY_LINK:
    def __init__(self, ingredient1, ingredient2):
        self.ingredient1 = ingredient1
        self.ingredient2 = ingredient2
    
    def __str__(self):
        return f"SYNERGY: {self.ingredient1.name} <-> {self.ingredient2.name}"

class ANTAGONISM_LINK:
    def __init__(self, ingredient1, ingredient2):
        self.ingredient1 = ingredient1
        self.ingredient2 = ingredient2
    
    def __str__(self):
        return f"ANTAGONISM: {self.ingredient1.name} <-> {self.ingredient2.name}"

def main():
    """
    Main function demonstrating basic cosmetic chemistry atom types usage
    """
    print("=== Cosmetic Chemistry Framework - Basic Introduction ===\n")
    
    # 1. Define basic cosmetic ingredients
    print("1. Defining Cosmetic Ingredients:")
    print("-" * 40)
    
    # Active ingredients
    hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
    niacinamide = ACTIVE_INGREDIENT('niacinamide')
    vitamin_c = ACTIVE_INGREDIENT('vitamin_c')
    retinol = ACTIVE_INGREDIENT('retinol')
    
    print(f"   {hyaluronic_acid}")
    print(f"   {niacinamide}")
    print(f"   {vitamin_c}")
    print(f"   {retinol}")
    
    # Supporting ingredients
    glycerin = HUMECTANT('glycerin')
    cetyl_alcohol = EMULSIFIER('cetyl_alcohol')
    phenoxyethanol = PRESERVATIVE('phenoxyethanol')
    vitamin_e = ANTIOXIDANT('vitamin_e')
    
    print(f"   {glycerin}")
    print(f"   {cetyl_alcohol}")
    print(f"   {phenoxyethanol}")
    print(f"   {vitamin_e}")
    
    # 2. Create a simple formulation
    print(f"\n2. Creating a Simple Skincare Formulation:")
    print("-" * 50)
    
    simple_serum = SKINCARE_FORMULATION(
        hyaluronic_acid,    # Hydrating active
        glycerin,           # Humectant
        phenoxyethanol      # Preservative
    )
    
    print(f"   {simple_serum}")
    print("   Ingredients:")
    for i, ingredient in enumerate(simple_serum.ingredients, 1):
        print(f"     {i}. {ingredient}")
    
    # 3. Create a more complex formulation
    print(f"\n3. Creating a Complex Moisturizer Formulation:")
    print("-" * 55)
    
    moisturizer = SKINCARE_FORMULATION(
        niacinamide,        # Brightening active
        hyaluronic_acid,    # Hydrating active
        cetyl_alcohol,      # Emulsifier
        glycerin,           # Humectant
        vitamin_e,          # Antioxidant
        phenoxyethanol      # Preservative
    )
    
    print(f"   {moisturizer}")
    print("   Ingredients by Category:")
    
    actives = moisturizer.get_ingredients_by_type("ACTIVE_INGREDIENT")
    humectants = moisturizer.get_ingredients_by_type("HUMECTANT")
    emulsifiers = moisturizer.get_ingredients_by_type("EMULSIFIER")
    antioxidants = moisturizer.get_ingredients_by_type("ANTIOXIDANT")
    preservatives = moisturizer.get_ingredients_by_type("PRESERVATIVE")
    
    print(f"     Active Ingredients: {[ing.name for ing in actives]}")
    print(f"     Humectants: {[ing.name for ing in humectants]}")
    print(f"     Emulsifiers: {[ing.name for ing in emulsifiers]}")
    print(f"     Antioxidants: {[ing.name for ing in antioxidants]}")
    print(f"     Preservatives: {[ing.name for ing in preservatives]}")
    
    # 4. Define ingredient interactions
    print(f"\n4. Defining Ingredient Interactions:")
    print("-" * 45)
    
    # Compatible ingredients
    compatible1 = COMPATIBILITY_LINK(hyaluronic_acid, niacinamide)
    compatible2 = COMPATIBILITY_LINK(hyaluronic_acid, glycerin)
    
    print(f"   {compatible1}")
    print(f"   {compatible2}")
    
    # Incompatible ingredients
    incompatible1 = INCOMPATIBILITY_LINK(vitamin_c, retinol)
    
    print(f"   {incompatible1}")
    print("   Note: Vitamin C and Retinol should not be used together")
    print("         due to potential irritation and reduced efficacy.")
    
    # Synergistic ingredients
    synergy1 = SYNERGY_LINK(vitamin_c, vitamin_e)
    
    print(f"   {synergy1}")
    print("   Note: Vitamin C and E work together to provide")
    print("         enhanced antioxidant protection.")
    
    # 5. Demonstrate basic safety checking
    print(f"\n5. Basic Safety Analysis:")
    print("-" * 35)
    
    def basic_safety_check(formulation):
        """Simple safety check for demonstration"""
        print(f"   Analyzing: {formulation}")
        
        # Check for known incompatible pairs
        known_incompatible = [
            (vitamin_c, retinol),
        ]
        
        ingredients = formulation.ingredients
        issues = []
        
        for ing1 in ingredients:
            for ing2 in ingredients:
                if ing1 != ing2:
                    for incompatible_pair in known_incompatible:
                        if (ing1.name == incompatible_pair[0].name and 
                            ing2.name == incompatible_pair[1].name) or \
                           (ing1.name == incompatible_pair[1].name and 
                            ing2.name == incompatible_pair[0].name):
                            issues.append(f"{ing1.name} + {ing2.name}")
        
        if issues:
            print(f"   ⚠️  Safety Issues Found:")
            for issue in issues:
                print(f"      - Incompatible combination: {issue}")
        else:
            print(f"   ✅ No known safety issues detected")
        
        return len(issues) == 0
    
    # Test safe formulation
    basic_safety_check(moisturizer)
    
    # Test potentially problematic formulation
    print()
    problematic_serum = SKINCARE_FORMULATION(
        vitamin_c,
        retinol,
        glycerin,
        phenoxyethanol
    )
    basic_safety_check(problematic_serum)
    
    # 6. Summary
    print(f"\n6. Summary:")
    print("-" * 20)
    print("   This example demonstrated:")
    print("   ✓ Basic ingredient modeling with functional classifications")
    print("   ✓ Simple and complex formulation creation")
    print("   ✓ Ingredient categorization and analysis")
    print("   ✓ Interaction modeling (compatibility, incompatibility, synergy)")
    print("   ✓ Basic safety checking for formulations")
    print()
    print("   Next Steps:")
    print("   → Explore advanced formulation optimization")
    print("   → Learn about pH and stability considerations")
    print("   → Study regulatory compliance checking")
    print("   → Practice with real-world formulation examples")

if __name__ == "__main__":
    main()
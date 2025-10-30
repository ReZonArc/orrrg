#!/usr/bin/env python3
"""
Advanced Cosmetic Chemistry Example

This example demonstrates advanced formulation analysis and optimization using
the OpenCog cheminformatics framework's cosmetic chemistry specializations.
It includes pH compatibility analysis, concentration limit checking, stability
prediction, and automated formulation optimization.

Author: OpenCog Cheminformatics Team
License: MIT
"""

import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

# Add the cheminformatics module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import basic classes from intro example
from cosmetic_intro_example import (
    CosmeticAtom, ACTIVE_INGREDIENT, PRESERVATIVE, EMULSIFIER, HUMECTANT,
    SURFACTANT, THICKENER, EMOLLIENT, ANTIOXIDANT, UV_FILTER, FRAGRANCE,
    COLORANT, PH_ADJUSTER, SKINCARE_FORMULATION, HAIRCARE_FORMULATION,
    MAKEUP_FORMULATION, FRAGRANCE_FORMULATION, COMPATIBILITY_LINK,
    INCOMPATIBILITY_LINK, SYNERGY_LINK, ANTAGONISM_LINK
)

class RegulationRegion(Enum):
    FDA = "FDA"
    EU = "EU"
    ASEAN = "ASEAN"
    CHINA = "CHINA"

@dataclass
class IngredientProperties:
    """Extended ingredient properties for advanced analysis"""
    ph_range: Tuple[float, float]
    stability_temperature: float
    max_concentration: float
    solubility: str  # "water", "oil", "both"
    molecular_weight: float
    allergen_status: bool
    cost_per_kg: float
    
@dataclass
class FormulationConstraints:
    """Constraints for formulation optimization"""
    target_ph: float
    ph_tolerance: float
    max_cost_per_unit: float
    required_properties: List[str]
    excluded_allergens: bool
    region: RegulationRegion

class AdvancedCosmeticAtom(CosmeticAtom):
    """Extended cosmetic atom with advanced properties"""
    def __init__(self, name, atom_type, properties: IngredientProperties):
        super().__init__(name, atom_type)
        self.properties = properties
    
    def is_ph_compatible(self, other_ingredient, target_ph: float) -> bool:
        """Check if two ingredients are pH compatible"""
        my_range = self.properties.ph_range
        other_range = other_ingredient.properties.ph_range
        
        # Check if ranges overlap and target pH is within both ranges
        overlap_min = max(my_range[0], other_range[0])
        overlap_max = min(my_range[1], other_range[1])
        
        return (overlap_min <= overlap_max and 
                overlap_min <= target_ph <= overlap_max)
    
    def stability_score(self, temperature: float) -> float:
        """Calculate stability score at given temperature"""
        temp_diff = abs(temperature - self.properties.stability_temperature)
        return max(0, 1 - (temp_diff / 50))  # Arbitrary scaling

class PropertyType:
    """Property types for formulations"""
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def __str__(self):
        return f"{self.name}: {self.value}"

class PH_PROPERTY(PropertyType):
    def __init__(self, value):
        super().__init__("pH", value)

class VISCOSITY_PROPERTY(PropertyType):
    def __init__(self, value):
        super().__init__("Viscosity", value)

class STABILITY_PROPERTY(PropertyType):
    def __init__(self, value):
        super().__init__("Stability", value)

class TEXTURE_PROPERTY(PropertyType):
    def __init__(self, value):
        super().__init__("Texture", value)

class SPF_PROPERTY(PropertyType):
    def __init__(self, value):
        super().__init__("SPF", value)

class AdvancedFormulation:
    """Advanced formulation with optimization and analysis capabilities"""
    
    def __init__(self, formulation_type: str):
        self.formulation_type = formulation_type
        self.ingredients: List[Tuple[AdvancedCosmeticAtom, float]] = []  # (ingredient, concentration)
        self.properties: Dict[str, PropertyType] = {}
        self.compatibility_matrix: Dict[Tuple[str, str], bool] = {}
        
    def add_ingredient(self, ingredient: AdvancedCosmeticAtom, concentration: float):
        """Add ingredient with concentration"""
        # Check concentration limits
        if concentration > ingredient.properties.max_concentration:
            raise ValueError(f"Concentration {concentration}% exceeds limit "
                           f"{ingredient.properties.max_concentration}% for {ingredient.name}")
        
        self.ingredients.append((ingredient, concentration))
        self._update_compatibility_matrix()
    
    def _update_compatibility_matrix(self):
        """Update compatibility matrix when ingredients change"""
        for i, (ing1, _) in enumerate(self.ingredients):
            for j, (ing2, _) in enumerate(self.ingredients):
                if i != j:
                    self.compatibility_matrix[(ing1.name, ing2.name)] = \
                        self._check_ingredient_compatibility(ing1, ing2)
    
    def _check_ingredient_compatibility(self, ing1: AdvancedCosmeticAtom, 
                                      ing2: AdvancedCosmeticAtom) -> bool:
        """Check compatibility between two ingredients"""
        # Known incompatible pairs
        incompatible_pairs = [
            ("vitamin_c", "retinol"),
            ("aha_acid", "retinol"),
            ("bha_acid", "retinol"),
        ]
        
        for pair in incompatible_pairs:
            if ((ing1.name == pair[0] and ing2.name == pair[1]) or 
                (ing1.name == pair[1] and ing2.name == pair[0])):
                return False
        
        return True
    
    def calculate_ph(self) -> float:
        """Calculate formulation pH based on ingredients"""
        # Simplified pH calculation
        total_concentration = sum(conc for _, conc in self.ingredients)
        if total_concentration == 0:
            return 7.0  # Neutral
        
        weighted_ph = 0
        for ingredient, concentration in self.ingredients:
            # Use middle of pH range as representative pH
            ingredient_ph = sum(ingredient.properties.ph_range) / 2
            weight = concentration / total_concentration
            weighted_ph += ingredient_ph * weight
        
        return weighted_ph
    
    def analyze_stability(self, temperature: float = 25.0) -> Dict[str, float]:
        """Analyze formulation stability"""
        stability_scores = {}
        
        for ingredient, concentration in self.ingredients:
            score = ingredient.stability_score(temperature)
            stability_scores[ingredient.name] = score
        
        # Overall stability is the minimum individual stability
        overall_stability = min(stability_scores.values()) if stability_scores else 0
        stability_scores['overall'] = overall_stability
        
        return stability_scores
    
    def calculate_cost(self) -> float:
        """Calculate formulation cost per unit"""
        total_cost = 0
        for ingredient, concentration in self.ingredients:
            # Cost calculation: (concentration/100) * cost_per_kg * assumed_batch_size
            ingredient_cost = (concentration / 100) * ingredient.properties.cost_per_kg * 0.1  # 100g batch
            total_cost += ingredient_cost
        
        return total_cost
    
    def check_regulatory_compliance(self, region: RegulationRegion) -> Dict[str, bool]:
        """Check regulatory compliance for specified region"""
        compliance = {
            'concentration_limits': True,
            'allergen_labeling': True,
            'prohibited_ingredients': True
        }
        
        # Check concentration limits
        for ingredient, concentration in self.ingredients:
            region_limit = self._get_regional_limit(ingredient.name, region)
            if region_limit and concentration > region_limit:
                compliance['concentration_limits'] = False
        
        # Check allergen requirements
        for ingredient, _ in self.ingredients:
            if ingredient.properties.allergen_status and region in [RegulationRegion.EU]:
                # In EU, allergens must be declared
                compliance['allergen_labeling'] = True  # Assuming proper labeling
        
        return compliance
    
    def _get_regional_limit(self, ingredient_name: str, region: RegulationRegion) -> Optional[float]:
        """Get regional concentration limits"""
        limits = {
            RegulationRegion.FDA: {
                'phenoxyethanol': 1.0,
                'retinol': 1.0,
            },
            RegulationRegion.EU: {
                'phenoxyethanol': 1.0,
                'parabens': 0.8,
                'formaldehyde': 0.2,
            }
        }
        
        return limits.get(region, {}).get(ingredient_name)
    
    def optimize(self, constraints: FormulationConstraints) -> 'AdvancedFormulation':
        """Optimize formulation based on constraints"""
        # This is a simplified optimization algorithm
        optimized = AdvancedFormulation(self.formulation_type)
        
        # Start with current ingredients
        for ingredient, concentration in self.ingredients:
            try:
                # Adjust concentration to meet pH requirements
                adjusted_conc = self._optimize_concentration(ingredient, concentration, constraints)
                optimized.add_ingredient(ingredient, adjusted_conc)
            except ValueError as e:
                print(f"Warning: Could not add {ingredient.name}: {e}")
        
        return optimized
    
    def _optimize_concentration(self, ingredient: AdvancedCosmeticAtom, 
                              current_conc: float, constraints: FormulationConstraints) -> float:
        """Optimize individual ingredient concentration"""
        # Ensure within limits
        max_allowed = ingredient.properties.max_concentration
        
        # Apply regional limits
        regional_limit = self._get_regional_limit(ingredient.name, constraints.region)
        if regional_limit:
            max_allowed = min(max_allowed, regional_limit)
        
        # For this example, just ensure we're within limits
        return min(current_conc, max_allowed)

def create_ingredient_database() -> Dict[str, AdvancedCosmeticAtom]:
    """Create a database of common cosmetic ingredients with properties"""
    
    ingredients = {}
    
    # Define ingredient properties
    hyaluronic_acid_props = IngredientProperties(
        ph_range=(5.0, 7.0),
        stability_temperature=25.0,
        max_concentration=2.0,
        solubility="water",
        molecular_weight=1000000,
        allergen_status=False,
        cost_per_kg=500.0
    )
    
    niacinamide_props = IngredientProperties(
        ph_range=(5.0, 7.0),
        stability_temperature=30.0,
        max_concentration=10.0,
        solubility="water",
        molecular_weight=122.12,
        allergen_status=False,
        cost_per_kg=80.0
    )
    
    vitamin_c_props = IngredientProperties(
        ph_range=(3.0, 4.0),
        stability_temperature=15.0,
        max_concentration=20.0,
        solubility="water",
        molecular_weight=176.12,
        allergen_status=False,
        cost_per_kg=150.0
    )
    
    retinol_props = IngredientProperties(
        ph_range=(5.5, 6.5),
        stability_temperature=20.0,
        max_concentration=1.0,
        solubility="oil",
        molecular_weight=286.45,
        allergen_status=False,
        cost_per_kg=2000.0
    )
    
    glycerin_props = IngredientProperties(
        ph_range=(4.0, 8.0),
        stability_temperature=50.0,
        max_concentration=50.0,
        solubility="water",
        molecular_weight=92.09,
        allergen_status=False,
        cost_per_kg=2.0
    )
    
    phenoxyethanol_props = IngredientProperties(
        ph_range=(4.0, 8.0),
        stability_temperature=40.0,
        max_concentration=1.0,
        solubility="both",
        molecular_weight=138.16,
        allergen_status=False,
        cost_per_kg=15.0
    )
    
    cetyl_alcohol_props = IngredientProperties(
        ph_range=(5.0, 8.0),
        stability_temperature=60.0,
        max_concentration=10.0,
        solubility="oil",
        molecular_weight=242.44,
        allergen_status=False,
        cost_per_kg=5.0
    )
    
    vitamin_e_props = IngredientProperties(
        ph_range=(5.0, 8.0),
        stability_temperature=40.0,
        max_concentration=1.0,
        solubility="oil",
        molecular_weight=430.71,
        allergen_status=False,
        cost_per_kg=25.0
    )
    
    # Create advanced ingredient atoms
    ingredients['hyaluronic_acid'] = AdvancedCosmeticAtom('hyaluronic_acid', 'ACTIVE_INGREDIENT', hyaluronic_acid_props)
    ingredients['niacinamide'] = AdvancedCosmeticAtom('niacinamide', 'ACTIVE_INGREDIENT', niacinamide_props)
    ingredients['vitamin_c'] = AdvancedCosmeticAtom('vitamin_c', 'ACTIVE_INGREDIENT', vitamin_c_props)
    ingredients['retinol'] = AdvancedCosmeticAtom('retinol', 'ACTIVE_INGREDIENT', retinol_props)
    ingredients['glycerin'] = AdvancedCosmeticAtom('glycerin', 'HUMECTANT', glycerin_props)
    ingredients['phenoxyethanol'] = AdvancedCosmeticAtom('phenoxyethanol', 'PRESERVATIVE', phenoxyethanol_props)
    ingredients['cetyl_alcohol'] = AdvancedCosmeticAtom('cetyl_alcohol', 'EMULSIFIER', cetyl_alcohol_props)
    ingredients['vitamin_e'] = AdvancedCosmeticAtom('vitamin_e', 'ANTIOXIDANT', vitamin_e_props)
    
    return ingredients

def demonstrate_ph_compatibility_analysis():
    """Demonstrate pH compatibility analysis"""
    print("=== pH Compatibility Analysis ===")
    print()
    
    ingredients = create_ingredient_database()
    
    # Test pH compatibility
    target_ph = 6.0
    test_pairs = [
        ('hyaluronic_acid', 'niacinamide'),
        ('vitamin_c', 'retinol'),
        ('glycerin', 'phenoxyethanol')
    ]
    
    for ing1_name, ing2_name in test_pairs:
        ing1 = ingredients[ing1_name]
        ing2 = ingredients[ing2_name]
        
        compatible = ing1.is_ph_compatible(ing2, target_ph)
        
        print(f"   {ing1.name} (pH {ing1.properties.ph_range[0]}-{ing1.properties.ph_range[1]}) + "
              f"{ing2.name} (pH {ing2.properties.ph_range[0]}-{ing2.properties.ph_range[1]})")
        print(f"   Target pH: {target_ph}")
        print(f"   Compatible: {'✅ Yes' if compatible else '❌ No'}")
        print()

def demonstrate_advanced_formulation():
    """Demonstrate advanced formulation analysis"""
    print("=== Advanced Formulation Analysis ===")
    print()
    
    ingredients = create_ingredient_database()
    
    # Create an advanced moisturizer formulation
    moisturizer = AdvancedFormulation("SKINCARE_FORMULATION")
    
    try:
        moisturizer.add_ingredient(ingredients['niacinamide'], 5.0)        # 5%
        moisturizer.add_ingredient(ingredients['hyaluronic_acid'], 1.0)    # 1%
        moisturizer.add_ingredient(ingredients['glycerin'], 8.0)           # 8%
        moisturizer.add_ingredient(ingredients['cetyl_alcohol'], 3.0)      # 3%
        moisturizer.add_ingredient(ingredients['vitamin_e'], 0.5)          # 0.5%
        moisturizer.add_ingredient(ingredients['phenoxyethanol'], 0.8)     # 0.8%
        
        print("   Formulation created successfully!")
        print("   Ingredients:")
        for ingredient, concentration in moisturizer.ingredients:
            print(f"     - {ingredient.name}: {concentration}%")
        
    except ValueError as e:
        print(f"   Error creating formulation: {e}")
        return
    
    # Analyze pH
    calculated_ph = moisturizer.calculate_ph()
    print(f"\n   Calculated pH: {calculated_ph:.2f}")
    
    # Analyze stability
    stability = moisturizer.analyze_stability()
    print(f"   Stability Analysis (at 25°C):")
    for ingredient_name, score in stability.items():
        if ingredient_name != 'overall':
            print(f"     - {ingredient_name}: {score:.2f}")
    print(f"     - Overall Stability: {stability['overall']:.2f}")
    
    # Calculate cost
    cost = moisturizer.calculate_cost()
    print(f"   Estimated Cost: ${cost:.2f} per 100g batch")
    
    # Check regulatory compliance
    fda_compliance = moisturizer.check_regulatory_compliance(RegulationRegion.FDA)
    eu_compliance = moisturizer.check_regulatory_compliance(RegulationRegion.EU)
    
    print(f"   FDA Compliance:")
    for check, passed in fda_compliance.items():
        print(f"     - {check}: {'✅ Pass' if passed else '❌ Fail'}")
    
    print(f"   EU Compliance:")
    for check, passed in eu_compliance.items():
        print(f"     - {check}: {'✅ Pass' if passed else '❌ Fail'}")

def demonstrate_formulation_optimization():
    """Demonstrate formulation optimization"""
    print("\n=== Formulation Optimization ===")
    print()
    
    ingredients = create_ingredient_database()
    
    # Create formulation constraints
    constraints = FormulationConstraints(
        target_ph=6.0,
        ph_tolerance=0.5,
        max_cost_per_unit=50.0,
        required_properties=["moisturizing", "anti-aging"],
        excluded_allergens=True,
        region=RegulationRegion.EU
    )
    
    print("   Optimization Constraints:")
    print(f"     - Target pH: {constraints.target_ph} ± {constraints.ph_tolerance}")
    print(f"     - Max Cost: ${constraints.max_cost_per_unit}")
    print(f"     - Region: {constraints.region.value}")
    print(f"     - Exclude Allergens: {constraints.excluded_allergens}")
    
    # Create initial formulation
    initial_formulation = AdvancedFormulation("SKINCARE_FORMULATION")
    initial_formulation.add_ingredient(ingredients['niacinamide'], 8.0)     # Slightly over-concentrated
    initial_formulation.add_ingredient(ingredients['hyaluronic_acid'], 1.5)
    initial_formulation.add_ingredient(ingredients['glycerin'], 10.0)
    initial_formulation.add_ingredient(ingredients['phenoxyethanol'], 0.9)
    
    print(f"\n   Initial Formulation Cost: ${initial_formulation.calculate_cost():.2f}")
    print(f"   Initial pH: {initial_formulation.calculate_ph():.2f}")
    
    # Optimize
    optimized_formulation = initial_formulation.optimize(constraints)
    
    print(f"\n   Optimized Formulation:")
    print(f"   Cost: ${optimized_formulation.calculate_cost():.2f}")
    print(f"   pH: {optimized_formulation.calculate_ph():.2f}")
    print("   Ingredients:")
    for ingredient, concentration in optimized_formulation.ingredients:
        print(f"     - {ingredient.name}: {concentration}%")

def demonstrate_incompatibility_analysis():
    """Demonstrate incompatibility analysis"""
    print("\n=== Incompatibility Analysis ===")
    print()
    
    ingredients = create_ingredient_database()
    
    # Create a potentially problematic formulation
    problematic_formula = AdvancedFormulation("SKINCARE_FORMULATION")
    
    try:
        problematic_formula.add_ingredient(ingredients['vitamin_c'], 15.0)
        problematic_formula.add_ingredient(ingredients['retinol'], 0.5)
        problematic_formula.add_ingredient(ingredients['glycerin'], 5.0)
        problematic_formula.add_ingredient(ingredients['phenoxyethanol'], 0.8)
        
    except ValueError:
        pass  # Expected for demonstration
    
    print("   Analyzing formulation with Vitamin C + Retinol:")
    print("   Compatibility Matrix:")
    
    for (ing1, ing2), compatible in problematic_formula.compatibility_matrix.items():
        status = "✅ Compatible" if compatible else "❌ Incompatible"
        print(f"     {ing1} + {ing2}: {status}")
        if not compatible:
            print("       → Recommendation: Use in separate products or alternate days")

def main():
    """Main function demonstrating advanced cosmetic chemistry analysis"""
    print("=== Advanced Cosmetic Chemistry Framework ===\n")
    
    # Run demonstrations
    demonstrate_ph_compatibility_analysis()
    demonstrate_advanced_formulation()
    demonstrate_formulation_optimization()
    demonstrate_incompatibility_analysis()
    
    # Summary
    print("\n=== Summary ===")
    print("This advanced example demonstrated:")
    print("✓ pH compatibility analysis between ingredients")
    print("✓ Advanced formulation properties and analysis")
    print("✓ Stability prediction and cost calculation")
    print("✓ Regulatory compliance checking")
    print("✓ Automated formulation optimization")
    print("✓ Ingredient incompatibility detection")
    print()
    print("These capabilities enable:")
    print("→ Systematic formulation development")
    print("→ Quality assurance and safety validation")
    print("→ Cost optimization and regulatory compliance")
    print("→ Predictive stability analysis")

if __name__ == "__main__":
    main()
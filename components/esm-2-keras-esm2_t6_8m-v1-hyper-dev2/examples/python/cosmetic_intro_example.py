#!/usr/bin/env python3
"""
Cosmetic Chemistry Framework - Introduction Example

This example demonstrates the basic usage of the cosmetic chemistry atom types
for ingredient modeling, formulation creation, and compatibility analysis.

Based on the OpenCog Cheminformatics Framework atom type system.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class AtomType(Enum):
    """Cosmetic Chemistry Atom Types"""
    # Ingredient Categories
    ACTIVE_INGREDIENT = "ACTIVE_INGREDIENT"
    PRESERVATIVE = "PRESERVATIVE"
    EMULSIFIER = "EMULSIFIER"
    HUMECTANT = "HUMECTANT"
    SURFACTANT = "SURFACTANT"
    THICKENER = "THICKENER"
    EMOLLIENT = "EMOLLIENT"
    ANTIOXIDANT = "ANTIOXIDANT"
    UV_FILTER = "UV_FILTER"
    FRAGRANCE = "FRAGRANCE"
    COLORANT = "COLORANT"
    PH_ADJUSTER = "PH_ADJUSTER"
    
    # Formulation Types
    SKINCARE_FORMULATION = "SKINCARE_FORMULATION"
    HAIRCARE_FORMULATION = "HAIRCARE_FORMULATION"
    MAKEUP_FORMULATION = "MAKEUP_FORMULATION"
    FRAGRANCE_FORMULATION = "FRAGRANCE_FORMULATION"
    
    # Property Types
    PH_PROPERTY = "PH_PROPERTY"
    VISCOSITY_PROPERTY = "VISCOSITY_PROPERTY"
    STABILITY_PROPERTY = "STABILITY_PROPERTY"
    TEXTURE_PROPERTY = "TEXTURE_PROPERTY"
    SPF_PROPERTY = "SPF_PROPERTY"
    
    # Interaction Types
    COMPATIBILITY_LINK = "COMPATIBILITY_LINK"
    INCOMPATIBILITY_LINK = "INCOMPATIBILITY_LINK"
    SYNERGY_LINK = "SYNERGY_LINK"
    ANTAGONISM_LINK = "ANTAGONISM_LINK"
    
    # Safety and Regulatory
    SAFETY_ASSESSMENT = "SAFETY_ASSESSMENT"
    ALLERGEN_CLASSIFICATION = "ALLERGEN_CLASSIFICATION"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"


@dataclass
class CosmeticIngredient:
    """Represents a cosmetic ingredient with its properties"""
    name: str
    inci_name: str
    atom_type: AtomType
    concentration: Optional[float] = None
    ph_min: Optional[float] = None
    ph_max: Optional[float] = None
    max_concentration: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.name} ({self.atom_type.value})"


@dataclass
class CosmeticFormulation:
    """Represents a complete cosmetic formulation"""
    name: str
    formulation_type: AtomType
    ingredients: List[CosmeticIngredient] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def add_ingredient(self, ingredient: CosmeticIngredient):
        """Add an ingredient to the formulation"""
        self.ingredients.append(ingredient)
    
    def get_total_concentration(self) -> float:
        """Calculate total active concentration"""
        return sum(ing.concentration or 0 for ing in self.ingredients 
                  if ing.concentration)
    
    def __str__(self):
        return f"{self.name} ({self.formulation_type.value}) - {len(self.ingredients)} ingredients"


class CosmeticChemistryFramework:
    """Main framework for cosmetic chemistry analysis"""
    
    def __init__(self):
        self.ingredients_database = {}
        self.compatibility_rules = {}
        self.incompatibility_rules = {}
        self.synergy_rules = {}
        self._initialize_database()
        self._initialize_compatibility_rules()
    
    def _initialize_database(self):
        """Initialize the cosmetic ingredients database"""
        
        # Active Ingredients
        self.add_ingredient(CosmeticIngredient(
            name="Hyaluronic Acid",
            inci_name="Sodium Hyaluronate",
            atom_type=AtomType.ACTIVE_INGREDIENT,
            concentration=1.0,
            ph_min=4.0,
            ph_max=7.0,
            max_concentration=2.0,
            properties={
                "function": "hydration",
                "molecular_weight": 1000000,
                "skin_penetration": "surface"
            }
        ))
        
        self.add_ingredient(CosmeticIngredient(
            name="Retinol",
            inci_name="Retinol",
            atom_type=AtomType.ACTIVE_INGREDIENT,
            concentration=0.5,
            ph_min=5.5,
            ph_max=6.5,
            max_concentration=1.0,
            properties={
                "function": "anti-aging",
                "stability": "oxygen_sensitive",
                "storage": "cool_dark"
            }
        ))
        
        self.add_ingredient(CosmeticIngredient(
            name="Niacinamide",
            inci_name="Niacinamide",
            atom_type=AtomType.ACTIVE_INGREDIENT,
            concentration=5.0,
            ph_min=5.0,
            ph_max=7.0,
            max_concentration=10.0,
            properties={
                "function": "pore_minimizing",
                "stability": "excellent",
                "skin_type": "all"
            }
        ))
        
        self.add_ingredient(CosmeticIngredient(
            name="Vitamin C",
            inci_name="L-Ascorbic Acid",
            atom_type=AtomType.ACTIVE_INGREDIENT,
            concentration=15.0,
            ph_min=3.5,
            ph_max=4.0,
            max_concentration=20.0,
            properties={
                "function": "antioxidant_brightening",
                "stability": "poor",
                "storage": "cool_dark"
            }
        ))
        
        # Humectants
        self.add_ingredient(CosmeticIngredient(
            name="Glycerin",
            inci_name="Glycerin",
            atom_type=AtomType.HUMECTANT,
            concentration=3.0,
            ph_min=4.0,
            ph_max=8.0,
            max_concentration=10.0,
            properties={
                "function": "moisture_retention",
                "feel": "sticky_at_high_concentration",
                "origin": "natural_synthetic"
            }
        ))
        
        # Emulsifiers
        self.add_ingredient(CosmeticIngredient(
            name="Cetyl Alcohol",
            inci_name="Cetyl Alcohol",
            atom_type=AtomType.EMULSIFIER,
            concentration=2.0,
            ph_min=4.0,
            ph_max=8.0,
            max_concentration=5.0,
            properties={
                "function": "emulsification_thickening",
                "hlb_value": 15.5,
                "system": "oil_in_water"
            }
        ))
        
        # Preservatives
        self.add_ingredient(CosmeticIngredient(
            name="Phenoxyethanol",
            inci_name="Phenoxyethanol",
            atom_type=AtomType.PRESERVATIVE,
            concentration=0.8,
            ph_min=4.0,
            ph_max=8.0,
            max_concentration=1.0,
            properties={
                "function": "broad_spectrum_preservation",
                "spectrum": "bacteria_fungi_yeast",
                "regulatory": "globally_accepted"
            }
        ))
        
        # Antioxidants
        self.add_ingredient(CosmeticIngredient(
            name="Vitamin E",
            inci_name="Tocopherol",
            atom_type=AtomType.ANTIOXIDANT,
            concentration=0.5,
            ph_min=4.0,
            ph_max=8.0,
            max_concentration=1.0,
            properties={
                "function": "antioxidant_stabilizer",
                "solubility": "lipid_soluble",
                "synergies": ["vitamin_c"]
            }
        ))
    
    def _initialize_compatibility_rules(self):
        """Initialize ingredient compatibility rules"""
        
        # Compatible combinations
        self.compatibility_rules.update({
            ("Hyaluronic Acid", "Niacinamide"): "Both stable in neutral pH range",
            ("Hyaluronic Acid", "Vitamin E"): "No chemical interaction",
            ("Niacinamide", "Glycerin"): "Complementary hydration effects",
            ("Vitamin E", "Glycerin"): "No interaction, stable combination"
        })
        
        # Incompatible combinations
        self.incompatibility_rules.update({
            ("Vitamin C", "Retinol"): "pH incompatibility and potential irritation",
            ("Vitamin C", "Niacinamide"): "pH incompatibility in some formulations"
        })
        
        # Synergistic combinations
        self.synergy_rules.update({
            ("Vitamin C", "Vitamin E"): "Enhanced antioxidant protection",
            ("Hyaluronic Acid", "Glycerin"): "Improved hydration effectiveness"
        })
    
    def add_ingredient(self, ingredient: CosmeticIngredient):
        """Add ingredient to database"""
        self.ingredients_database[ingredient.name] = ingredient
    
    def get_ingredient(self, name: str) -> Optional[CosmeticIngredient]:
        """Retrieve ingredient from database"""
        return self.ingredients_database.get(name)
    
    def create_formulation(self, name: str, formulation_type: AtomType) -> CosmeticFormulation:
        """Create a new formulation"""
        return CosmeticFormulation(name, formulation_type)
    
    def check_compatibility(self, ingredient1: str, ingredient2: str) -> Dict[str, Any]:
        """Check compatibility between two ingredients"""
        key1 = (ingredient1, ingredient2)
        key2 = (ingredient2, ingredient1)
        
        result = {
            "compatible": True,
            "synergistic": False,
            "incompatible": False,
            "reason": "",
            "recommendation": ""
        }
        
        # Check for incompatibilities
        if key1 in self.incompatibility_rules or key2 in self.incompatibility_rules:
            result["compatible"] = False
            result["incompatible"] = True
            result["reason"] = self.incompatibility_rules.get(key1) or self.incompatibility_rules.get(key2)
            result["recommendation"] = "Avoid combining these ingredients"
        
        # Check for synergies
        elif key1 in self.synergy_rules or key2 in self.synergy_rules:
            result["synergistic"] = True
            result["reason"] = self.synergy_rules.get(key1) or self.synergy_rules.get(key2)
            result["recommendation"] = "Excellent combination for enhanced benefits"
        
        # Basic compatibility
        elif key1 in self.compatibility_rules or key2 in self.compatibility_rules:
            result["reason"] = self.compatibility_rules.get(key1) or self.compatibility_rules.get(key2)
            result["recommendation"] = "Safe to combine"
        
        return result
    
    def validate_formulation(self, formulation: CosmeticFormulation) -> Dict[str, Any]:
        """Validate a complete formulation"""
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check for incompatible ingredients
        ingredients = [ing.name for ing in formulation.ingredients]
        for i, ing1 in enumerate(ingredients):
            for ing2 in ingredients[i+1:]:
                compatibility = self.check_compatibility(ing1, ing2)
                if compatibility["incompatible"]:
                    validation["valid"] = False
                    validation["errors"].append(
                        f"Incompatible ingredients: {ing1} + {ing2} - {compatibility['reason']}"
                    )
                elif compatibility["synergistic"]:
                    validation["recommendations"].append(
                        f"Synergistic combination: {ing1} + {ing2} - {compatibility['reason']}"
                    )
        
        # Check total active concentration
        total_actives = sum(ing.concentration or 0 for ing in formulation.ingredients 
                           if ing.atom_type == AtomType.ACTIVE_INGREDIENT)
        if total_actives > 25.0:
            validation["warnings"].append(
                f"High total active concentration: {total_actives:.1f}% - Consider reducing"
            )
        
        # Check for preservative
        has_preservative = any(ing.atom_type == AtomType.PRESERVATIVE 
                              for ing in formulation.ingredients)
        if not has_preservative:
            validation["warnings"].append("No preservative detected - Add preservation system")
        
        return validation
    
    def get_ingredient_profile(self, name: str) -> Dict[str, Any]:
        """Get detailed ingredient profile"""
        ingredient = self.get_ingredient(name)
        if not ingredient:
            return {"error": f"Ingredient '{name}' not found in database"}
        
        return {
            "name": ingredient.name,
            "inci_name": ingredient.inci_name,
            "type": ingredient.atom_type.value,
            "typical_concentration": ingredient.concentration,
            "max_concentration": ingredient.max_concentration,
            "ph_range": f"{ingredient.ph_min}-{ingredient.ph_max}" if ingredient.ph_min else "Not specified",
            "properties": ingredient.properties,
            "compatible_with": self._find_compatible_ingredients(name),
            "incompatible_with": self._find_incompatible_ingredients(name)
        }
    
    def _find_compatible_ingredients(self, ingredient_name: str) -> List[str]:
        """Find ingredients compatible with the given ingredient"""
        compatible = []
        for key, _ in self.compatibility_rules.items():
            if ingredient_name in key:
                other = key[0] if key[1] == ingredient_name else key[1]
                compatible.append(other)
        return compatible
    
    def _find_incompatible_ingredients(self, ingredient_name: str) -> List[str]:
        """Find ingredients incompatible with the given ingredient"""
        incompatible = []
        for key, _ in self.incompatibility_rules.items():
            if ingredient_name in key:
                other = key[0] if key[1] == ingredient_name else key[1]
                incompatible.append(other)
        return incompatible


def main():
    """Demonstrate basic cosmetic chemistry framework usage"""
    print("🧴 Cosmetic Chemistry Framework - Introduction Example")
    print("=" * 60)
    
    # Initialize framework
    framework = CosmeticChemistryFramework()
    
    print(f"\n📚 Initialized database with {len(framework.ingredients_database)} ingredients")
    
    # Display available ingredients
    print("\n🧪 Available Ingredients:")
    for name, ingredient in framework.ingredients_database.items():
        print(f"  • {ingredient}")
    
    # Example 1: Basic ingredient modeling
    print("\n1️⃣ Basic Ingredient Modeling")
    print("-" * 30)
    
    # Get ingredient profile
    ingredient_profile = framework.get_ingredient_profile("Hyaluronic Acid")
    print(f"Ingredient Profile: {ingredient_profile['name']}")
    print(f"  INCI Name: {ingredient_profile['inci_name']}")
    print(f"  Type: {ingredient_profile['type']}")
    print(f"  Typical Concentration: {ingredient_profile['typical_concentration']}%")
    print(f"  pH Range: {ingredient_profile['ph_range']}")
    print(f"  Function: {ingredient_profile['properties'].get('function', 'N/A')}")
    
    # Example 2: Formulation creation
    print("\n2️⃣ Formulation Creation")
    print("-" * 25)
    
    # Create a moisturizer formulation
    moisturizer = framework.create_formulation(
        "Daily Moisturizer", 
        AtomType.SKINCARE_FORMULATION
    )
    
    # Add ingredients
    moisturizer.add_ingredient(framework.get_ingredient("Hyaluronic Acid"))
    moisturizer.add_ingredient(framework.get_ingredient("Glycerin"))
    moisturizer.add_ingredient(framework.get_ingredient("Cetyl Alcohol"))
    moisturizer.add_ingredient(framework.get_ingredient("Phenoxyethanol"))
    
    print(f"Created formulation: {moisturizer}")
    print("Ingredients:")
    for ingredient in moisturizer.ingredients:
        print(f"  • {ingredient.name} ({ingredient.concentration}%)")
    
    print(f"Total Active Concentration: {moisturizer.get_total_concentration():.1f}%")
    
    # Example 3: Compatibility analysis
    print("\n3️⃣ Compatibility Analysis")
    print("-" * 27)
    
    # Test compatible combination
    compatibility1 = framework.check_compatibility("Hyaluronic Acid", "Niacinamide")
    print(f"Hyaluronic Acid + Niacinamide:")
    print(f"  Compatible: {compatibility1['compatible']}")
    print(f"  Reason: {compatibility1['reason']}")
    
    # Test incompatible combination
    compatibility2 = framework.check_compatibility("Vitamin C", "Retinol")
    print(f"\nVitamin C + Retinol:")
    print(f"  Compatible: {compatibility2['compatible']}")
    print(f"  Incompatible: {compatibility2['incompatible']}")
    print(f"  Reason: {compatibility2['reason']}")
    print(f"  Recommendation: {compatibility2['recommendation']}")
    
    # Test synergistic combination
    compatibility3 = framework.check_compatibility("Vitamin C", "Vitamin E")
    print(f"\nVitamin C + Vitamin E:")
    print(f"  Compatible: {compatibility3['compatible']}")
    print(f"  Synergistic: {compatibility3['synergistic']}")
    print(f"  Reason: {compatibility3['reason']}")
    
    # Example 4: Formulation validation
    print("\n4️⃣ Formulation Validation")
    print("-" * 26)
    
    validation = framework.validate_formulation(moisturizer)
    print(f"Formulation Valid: {validation['valid']}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️ {warning}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  ❌ {error}")
    
    if validation['recommendations']:
        print("Recommendations:")
        for rec in validation['recommendations']:
            print(f"  💡 {rec}")
    
    # Example 5: Create problematic formulation for demonstration
    print("\n5️⃣ Problematic Formulation Example")
    print("-" * 35)
    
    problem_serum = framework.create_formulation(
        "Problem Serum", 
        AtomType.SKINCARE_FORMULATION
    )
    
    # Add incompatible ingredients
    problem_serum.add_ingredient(framework.get_ingredient("Vitamin C"))
    problem_serum.add_ingredient(framework.get_ingredient("Retinol"))
    problem_serum.add_ingredient(framework.get_ingredient("Niacinamide"))
    
    print(f"Created problematic formulation: {problem_serum}")
    
    validation = framework.validate_formulation(problem_serum)
    print(f"Formulation Valid: {validation['valid']}")
    
    if validation['errors']:
        print("Detected Issues:")
        for error in validation['errors']:
            print(f"  ❌ {error}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️ {warning}")
    
    print("\n✅ Introduction example completed successfully!")
    print("\nNext steps:")
    print("- Explore advanced formulation analysis in cosmetic_chemistry_example.py")
    print("- Check out Scheme examples for OpenCog integration")
    print("- Review comprehensive documentation in docs/COSMETIC_CHEMISTRY.md")


if __name__ == "__main__":
    main()
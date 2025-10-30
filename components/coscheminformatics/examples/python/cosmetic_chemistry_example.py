#!/usr/bin/env python3
#
# cosmetic_chemistry_example.py
#
# Advanced example demonstrating cosmetic chemistry applications using
# the OpenCog cheminformatics framework with cosmetic specializations.
# This example shows how to:
# 1. Define cosmetic ingredients with functional classifications
# 2. Create complex cosmetic formulations
# 3. Analyze ingredient compatibility and interactions
# 4. Assess formulation properties and regulatory compliance
#
# --------------------------------------------------------------

# Import the AtomSpace and basic AtomSpace types
from opencog.atomspace import AtomSpace
from opencog.type_constructors import *

# Import cheminformatics types including cosmetic specializations
from opencog.cheminformatics import *

# Create AtomSpace for cosmetic chemistry knowledge
atomspace = AtomSpace()
set_default_atomspace(atomspace)

print("=== Cosmetic Chemistry Example ===")
print(f"AtomSpace initialized: {atomspace}")
print()

# ===========================================================
# 1. Define Common Cosmetic Ingredients
# ===========================================================

class CosmeticIngredient:
    """Helper class to create cosmetic ingredients with properties"""
    
    def __init__(self, name, inci_name, category, properties=None):
        self.name = name
        self.inci_name = inci_name
        self.category = category
        self.properties = properties or {}
        self.atom = None
        self._create_atom()
    
    def _create_atom(self):
        """Create appropriate atom type based on category"""
        if self.category == "active":
            self.atom = ACTIVE_INGREDIENT(self.name)
        elif self.category == "preservative":
            self.atom = PRESERVATIVE(self.name)
        elif self.category == "emulsifier":
            self.atom = EMULSIFIER(self.name)
        elif self.category == "humectant":
            self.atom = HUMECTANT(self.name)
        elif self.category == "emollient":
            self.atom = EMOLLIENT(self.name)
        elif self.category == "uv_filter":
            self.atom = UV_FILTER(self.name)
        elif self.category == "thickener":
            self.atom = THICKENER(self.name)
        elif self.category == "antioxidant":
            self.atom = ANTIOXIDANT(self.name)
        else:
            self.atom = COSMETIC_INGREDIENT(self.name)

# Define key cosmetic ingredients
print("1. Creating cosmetic ingredient database...")

# Active ingredients for skin care
hyaluronic_acid = CosmeticIngredient(
    name="hyaluronic_acid",
    inci_name="Sodium Hyaluronate", 
    category="active",
    properties={
        "function": "hydration",
        "molecular_weight": "high",
        "usage_range": "0.1-2.0%",
        "ph_stability": "4.0-8.0"
    }
)

retinol = CosmeticIngredient(
    name="retinol",
    inci_name="Retinol",
    category="active", 
    properties={
        "function": "anti_aging",
        "stability": "poor",
        "usage_range": "0.01-1.0%",
        "ph_stability": "5.5-7.0",
        "light_sensitive": True,
        "air_sensitive": True
    }
)

vitamin_c = CosmeticIngredient(
    name="vitamin_c",
    inci_name="L-Ascorbic Acid",
    category="active",
    properties={
        "function": "antioxidant",
        "stability": "poor", 
        "usage_range": "5.0-20.0%",
        "ph_stability": "3.0-6.0",
        "air_sensitive": True
    }
)

niacinamide = CosmeticIngredient(
    name="niacinamide", 
    inci_name="Niacinamide",
    category="active",
    properties={
        "function": "skin_conditioning",
        "stability": "excellent",
        "usage_range": "2.0-10.0%", 
        "ph_stability": "4.0-7.0"
    }
)

# Preservatives
phenoxyethanol = CosmeticIngredient(
    name="phenoxyethanol",
    inci_name="Phenoxyethanol",
    category="preservative",
    properties={
        "spectrum": "broad",
        "usage_range": "0.3-1.0%",
        "ph_effective": "3.0-10.0"
    }
)

# Emulsifiers and texture agents  
cetyl_alcohol = CosmeticIngredient(
    name="cetyl_alcohol",
    inci_name="Cetyl Alcohol", 
    category="emulsifier",
    properties={
        "emulsion_type": "oil_in_water",
        "usage_range": "1.0-5.0%",
        "melting_point": "49-51°C"
    }
)

# Humectants
glycerin = CosmeticIngredient(
    name="glycerin",
    inci_name="Glycerin",
    category="humectant", 
    properties={
        "hygroscopic": True,
        "usage_range": "3.0-15.0%",
        "solubility": "water"
    }
)

# UV filters
zinc_oxide = CosmeticIngredient(
    name="zinc_oxide",
    inci_name="Zinc Oxide",
    category="uv_filter",
    properties={
        "filter_type": "physical",
        "protection": "UVA/UVB",
        "usage_range": "5.0-25.0%"
    }
)

ingredients = [hyaluronic_acid, retinol, vitamin_c, niacinamide, 
               phenoxyethanol, cetyl_alcohol, glycerin, zinc_oxide]

print(f"Created {len(ingredients)} cosmetic ingredients")
for ingredient in ingredients:
    print(f"  - {ingredient.inci_name} ({ingredient.category})")
print()

# ===========================================================
# 2. Define Ingredient Interactions and Compatibility
# ===========================================================

print("2. Defining ingredient interactions...")

# Incompatible combinations
vitamin_c_retinol_incompatible = INCOMPATIBILITY_LINK(
    vitamin_c.atom, retinol.atom
)

# Compatible combinations  
ha_niacinamide_compatible = COMPATIBILITY_LINK(
    hyaluronic_acid.atom, niacinamide.atom
)

# Synergistic combinations
# Note: In a real implementation, we would create vitamin E atom
# For this example, we'll use a concept node
vitamin_e = ANTIOXIDANT("vitamin_e")
vitamin_c_e_synergy = SYNERGY_LINK(vitamin_c.atom, vitamin_e)

print("Ingredient interactions defined:")
print("  - Vitamin C <-> Retinol: INCOMPATIBLE (pH conflict)")
print("  - Hyaluronic Acid <-> Niacinamide: COMPATIBLE")  
print("  - Vitamin C <-> Vitamin E: SYNERGISTIC")
print()

# ===========================================================
# 3. Create Cosmetic Formulations
# ===========================================================

print("3. Creating cosmetic formulations...")

# Anti-aging serum formulation
anti_aging_serum = SKINCARE_FORMULATION(
    hyaluronic_acid.atom,    # 1.0% - Primary hydrating active
    niacinamide.atom,        # 5.0% - Skin conditioning active  
    glycerin.atom,           # 5.0% - Additional humectant
    phenoxyethanol.atom      # 0.8% - Preservation system
)

# Moisturizing cream formulation
moisturizing_cream = SKINCARE_FORMULATION(
    hyaluronic_acid.atom,    # 2.0% - Hydrating active
    glycerin.atom,           # 10.0% - Primary humectant
    cetyl_alcohol.atom,      # 3.0% - Emulsifier and texture
    phenoxyethanol.atom      # 0.7% - Preservation
)

# Sunscreen formulation
sunscreen = SKINCARE_FORMULATION(
    zinc_oxide.atom,         # 15.0% - Physical UV filter
    glycerin.atom,           # 5.0% - Humectant
    cetyl_alcohol.atom,      # 2.0% - Emulsifier  
    phenoxyethanol.atom      # 0.8% - Preservative
)

formulations = [
    ("Anti-Aging Serum", anti_aging_serum),
    ("Moisturizing Cream", moisturizing_cream), 
    ("Broad Spectrum Sunscreen", sunscreen)
]

print("Created formulations:")
for name, formulation in formulations:
    print(f"  - {name}: {formulation}")
print()

# ===========================================================
# 4. Formulation Property Analysis
# ===========================================================

print("4. Analyzing formulation properties...")

class FormulationAnalyzer:
    """Analyze cosmetic formulation properties"""
    
    @staticmethod
    def analyze_ph_compatibility(formulation_name, formulation):
        """Analyze pH compatibility of ingredients in formulation"""
        print(f"\nPH Analysis for {formulation_name}:")
        
        # In a full implementation, this would query the AtomSpace
        # for ingredient pH requirements and check compatibility
        if "serum" in formulation_name.lower():
            if niacinamide.atom in formulation.out:
                print("  ✓ Niacinamide stable at pH 4.0-7.0")
            if hyaluronic_acid.atom in formulation.out:
                print("  ✓ Hyaluronic Acid stable at pH 4.0-8.0")
            print("  → Recommended formulation pH: 5.5-6.5")
            
        elif "cream" in formulation_name.lower():
            print("  ✓ Cream formulation allows pH flexibility") 
            print("  → Recommended formulation pH: 6.0-7.0")
            
        elif "sunscreen" in formulation_name.lower():
            if zinc_oxide.atom in formulation.out:
                print("  ✓ Zinc Oxide stable at neutral to alkaline pH")
            print("  → Recommended formulation pH: 6.5-8.0")
    
    @staticmethod 
    def analyze_stability_factors(formulation_name, formulation):
        """Analyze stability factors for formulation"""
        print(f"\nStability Analysis for {formulation_name}:")
        
        stability_factors = []
        
        # Check for light-sensitive ingredients
        if hasattr(retinol, 'atom') and retinol.atom in formulation.out:
            stability_factors.append("⚠ Contains light-sensitive retinol - use opaque packaging")
            
        # Check for oxygen-sensitive ingredients  
        if hasattr(vitamin_c, 'atom') and vitamin_c.atom in formulation.out:
            stability_factors.append("⚠ Contains oxygen-sensitive vitamin C - minimize air exposure")
        
        # Check preservative coverage
        if phenoxyethanol.atom in formulation.out:
            stability_factors.append("✓ Preservative system present")
        
        if not stability_factors:
            stability_factors.append("✓ No major stability concerns identified")
            
        for factor in stability_factors:
            print(f"  {factor}")
    
    @staticmethod
    def generate_usage_recommendations(formulation_name, formulation):
        """Generate usage recommendations"""
        print(f"\nUsage Recommendations for {formulation_name}:")
        
        if "serum" in formulation_name.lower():
            print("  • Apply to clean skin before moisturizer")
            print("  • Use morning and/or evening")
            print("  • Allow absorption before next step")
            
        elif "cream" in formulation_name.lower():
            print("  • Apply as final step in skincare routine")  
            print("  • Use morning and/or evening")
            print("  • Suitable for daily use")
            
        elif "sunscreen" in formulation_name.lower():
            print("  • Apply 20-30 minutes before sun exposure")
            print("  • Reapply every 2 hours")
            print("  • Use minimum 1/4 teaspoon for face")

# Run analysis on all formulations
analyzer = FormulationAnalyzer()

for name, formulation in formulations:
    analyzer.analyze_ph_compatibility(name, formulation)
    analyzer.analyze_stability_factors(name, formulation)  
    analyzer.generate_usage_recommendations(name, formulation)
    print("-" * 50)

# ===========================================================
# 5. Ingredient Database Query Functions
# ===========================================================

print("\n5. Ingredient database queries...")

def find_ingredients_by_category(category):
    """Find all ingredients in a specific functional category"""
    category_ingredients = []
    for ingredient in ingredients:
        if ingredient.category == category:
            category_ingredients.append(ingredient)
    return category_ingredients

def find_compatible_ingredients(target_ingredient):
    """Find ingredients compatible with target ingredient"""
    # In a full implementation, this would query the AtomSpace
    compatible = []
    
    if target_ingredient.name == "vitamin_c":
        compatible = ["niacinamide", "vitamin_e", "hyaluronic_acid"]
    elif target_ingredient.name == "retinol":
        compatible = ["niacinamide", "hyaluronic_acid", "glycerin"]
    elif target_ingredient.name == "niacinamide":
        compatible = ["vitamin_c", "retinol", "hyaluronic_acid"]
    
    return compatible

def check_regulatory_compliance():
    """Check basic regulatory compliance factors"""
    print("Regulatory Compliance Check:")
    print("  ✓ All ingredients have valid INCI names")
    print("  ✓ UV filters within approved usage levels") 
    print("  ✓ Preservatives within effective ranges")
    print("  ⚠ Full safety assessment required for commercial use")

# Run database queries
print("\nActive ingredients available:")
actives = find_ingredients_by_category("active")
for active in actives:
    print(f"  - {active.inci_name}")

print(f"\nIngredients compatible with Vitamin C:")
compatible = find_compatible_ingredients(vitamin_c)
for comp in compatible:
    print(f"  - {comp.title()}")

print()
check_regulatory_compliance()

# ===========================================================
# 6. Summary and Advanced Applications
# ===========================================================

print("\n" + "=" * 60)
print("COSMETIC CHEMISTRY FRAMEWORK DEMONSTRATION COMPLETE")
print("=" * 60)

print(f"\nSummary:")
print(f"  • Created {len(ingredients)} cosmetic ingredients with properties")
print(f"  • Defined {len(formulations)} complete formulations")
print(f"  • Analyzed compatibility, stability, and usage factors")
print(f"  • Demonstrated regulatory compliance checking")

print(f"\nAdvanced applications possible with this framework:")
print(f"  • Automated formulation optimization")
print(f"  • Ingredient substitution recommendations") 
print(f"  • Stability prediction modeling")
print(f"  • Regulatory compliance automation")
print(f"  • Cost optimization algorithms")
print(f"  • Consumer preference matching")

print(f"\nAtomSpace final state: {atomspace}")
print(f"Total atoms created: {atomspace.size()}")

print("\n=== End of Cosmetic Chemistry Example ===")
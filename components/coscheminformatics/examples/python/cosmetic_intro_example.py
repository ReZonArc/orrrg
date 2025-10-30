#!/usr/bin/env python3
#
# cosmetic_intro_example.py
#
# An introductory example for using cosmetic chemistry specializations 
# in the OpenCog cheminformatics framework. This shows basic usage of
# cosmetic-specific atom types and simple formulation creation.
#
# --------------------------------------------------------------

# Import the AtomSpace and basic AtomSpace types  
from opencog.atomspace import AtomSpace
from opencog.type_constructors import *

# Import cheminformatics types including cosmetic specializations
from opencog.cheminformatics import *

# Create AtomSpace for cosmetic knowledge
atomspace = AtomSpace()
set_default_atomspace(atomspace)

print("=== Cosmetic Chemistry Intro Example ===")
print(f"AtomSpace: {atomspace}")
print()

# ===========================================================
# 1. Create Basic Cosmetic Ingredients
# ===========================================================

print("1. Creating basic cosmetic ingredients...")

# Active ingredients
retinol = ACTIVE_INGREDIENT('retinol')
hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
vitamin_c = ACTIVE_INGREDIENT('vitamin_c')

print(f"Active ingredient - Retinol: {retinol}")
print(f"Active ingredient - Hyaluronic Acid: {hyaluronic_acid}")
print(f"Active ingredient - Vitamin C: {vitamin_c}")

# Functional ingredients
glycerin = HUMECTANT('glycerin')
phenoxyethanol = PRESERVATIVE('phenoxyethanol')
cetyl_alcohol = EMULSIFIER('cetyl_alcohol')

print(f"Humectant - Glycerin: {glycerin}")
print(f"Preservative - Phenoxyethanol: {phenoxyethanol}")
print(f"Emulsifier - Cetyl Alcohol: {cetyl_alcohol}")
print()

# ===========================================================
# 2. Create Simple Cosmetic Formulations
# ===========================================================

print("2. Creating cosmetic formulations...")

# Simple hydrating serum
hydrating_serum = SKINCARE_FORMULATION(
    hyaluronic_acid,    # Primary active for hydration
    glycerin,           # Additional humectant  
    phenoxyethanol      # Preservative system
)

print(f"Hydrating serum formulation: {hydrating_serum}")

# Basic moisturizing cream
moisturizer = SKINCARE_FORMULATION(
    hyaluronic_acid,    # Hydrating active
    cetyl_alcohol,      # Emulsifier for texture
    glycerin,           # Humectant
    phenoxyethanol      # Preservative
)

print(f"Moisturizer formulation: {moisturizer}")
print()

# ===========================================================
# 3. Define Ingredient Properties
# ===========================================================

print("3. Defining cosmetic properties...")

# pH properties for ingredients
serum_ph = PH_PROPERTY('serum_optimal_ph')
cream_ph = PH_PROPERTY('cream_optimal_ph')

print(f"Serum pH property: {serum_ph}")
print(f"Cream pH property: {cream_ph}")

# Texture properties
lightweight_texture = TEXTURE_PROPERTY('lightweight')
rich_texture = TEXTURE_PROPERTY('rich_cream')

print(f"Lightweight texture: {lightweight_texture}")
print(f"Rich texture: {rich_texture}")
print()

# ===========================================================
# 4. Create Ingredient Interactions
# ===========================================================

print("4. Creating ingredient interactions...")

# Compatible combination
ha_glycerin_compatible = COMPATIBILITY_LINK(hyaluronic_acid, glycerin)
print(f"Hyaluronic Acid + Glycerin compatibility: {ha_glycerin_compatible}")

# Incompatible combination (vitamin C and retinol due to pH)
vitamin_c_retinol_conflict = INCOMPATIBILITY_LINK(vitamin_c, retinol)
print(f"Vitamin C + Retinol incompatibility: {vitamin_c_retinol_conflict}")

# Synergistic combination
vitamin_e = ANTIOXIDANT('vitamin_e')
antioxidant_synergy = SYNERGY_LINK(vitamin_c, vitamin_e)
print(f"Vitamin C + Vitamin E synergy: {antioxidant_synergy}")
print()

# ===========================================================
# 5. Create Specialized Formulation Types
# ===========================================================

print("5. Creating specialized formulation types...")

# Makeup formulation
foundation = MAKEUP_FORMULATION(
    COLORANT('iron_oxides'),
    EMULSIFIER('dimethicone'),
    UV_FILTER('zinc_oxide'),
    PRESERVATIVE('phenoxyethanol')
)
print(f"Foundation makeup: {foundation}")

# Hair care formulation  
shampoo = HAIRCARE_FORMULATION(
    SURFACTANT('sodium_lauryl_sulfate'),
    THICKENER('cocamidopropyl_betaine'),
    PRESERVATIVE('methylisothiazolinone')
)
print(f"Shampoo formulation: {shampoo}")

# Fragrance formulation
perfume = FRAGRANCE_FORMULATION(
    FRAGRANCE('linalool'),
    FRAGRANCE('limonene'), 
    EMOLLIENT('dipropylene_glycol')
)
print(f"Perfume formulation: {perfume}")
print()

# ===========================================================
# 6. Safety and Regulatory Information
# ===========================================================

print("6. Adding safety and regulatory information...")

# Allergen classification
linalool_allergen = ALLERGEN_CLASSIFICATION('linalool_allergen')
limonene_allergen = ALLERGEN_CLASSIFICATION('limonene_allergen')

print(f"Linalool allergen classification: {linalool_allergen}")
print(f"Limonene allergen classification: {limonene_allergen}")

# Concentration limits
retinol_limit = CONCENTRATION_LIMIT('retinol_max_1_percent')
uv_filter_limit = CONCENTRATION_LIMIT('zinc_oxide_max_25_percent')

print(f"Retinol concentration limit: {retinol_limit}")
print(f"UV filter concentration limit: {uv_filter_limit}")

# Safety assessment
formulation_safety = SAFETY_ASSESSMENT('moisturizer_safety_passed')
print(f"Safety assessment: {formulation_safety}")
print()

# ===========================================================
# 7. Summary
# ===========================================================

print("=== Summary ===")
print(f"AtomSpace now contains {atomspace.size()} atoms")
print()

print("Created cosmetic ingredients:")
print("  • Active ingredients: retinol, hyaluronic acid, vitamin C")  
print("  • Functional ingredients: glycerin, phenoxyethanol, cetyl alcohol")
print()

print("Created formulations:")
print("  • Skincare: hydrating serum, moisturizer")
print("  • Makeup: foundation")  
print("  • Haircare: shampoo")
print("  • Fragrance: perfume")
print()

print("Defined interactions:")
print("  • Compatibility: hyaluronic acid + glycerin")
print("  • Incompatibility: vitamin C + retinol")
print("  • Synergy: vitamin C + vitamin E")
print()

print("Added regulatory elements:")  
print("  • Allergen classifications")
print("  • Concentration limits")
print("  • Safety assessments")

print("\n=== End of Cosmetic Chemistry Intro ===")
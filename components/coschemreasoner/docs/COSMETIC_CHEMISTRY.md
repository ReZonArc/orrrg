# Cosmetic Chemistry Framework Documentation

This document provides comprehensive documentation for the OpenCog cheminformatics framework's cosmetic chemistry specializations, enabling systematic analysis and optimization of cosmetic formulations through knowledge representation and reasoning.

## Table of Contents

1. [Overview](#overview)
2. [Atom Types Reference](#atom-types-reference)
3. [Common Cosmetic Ingredients Database](#common-cosmetic-ingredients-database)
4. [Formulation Guidelines](#formulation-guidelines)
5. [Regulatory Compliance](#regulatory-compliance)
6. [Advanced Applications](#advanced-applications)
7. [Usage Examples](#usage-examples)

## Overview

The cosmetic chemistry framework extends the OpenCog cheminformatics system with 35+ specialized atom types designed for:

- **Ingredient Modeling**: Systematic representation of cosmetic ingredients with functional classifications
- **Formulation Creation**: Complex cosmetic formulations with compatibility analysis
- **Safety Assessment**: Regulatory compliance checking and allergen management
- **Property Analysis**: pH, viscosity, SPF, and sensory property modeling
- **Interaction Prediction**: Compatibility, synergy, and antagonism analysis

## Atom Types Reference

### Ingredient Categories

#### ACTIVE_INGREDIENT
Primary functional ingredients that provide the main benefit of the product.

**Examples:**
- Hyaluronic acid (hydration)
- Retinol (anti-aging)
- Niacinamide (brightening)
- Vitamin C (antioxidant)

**Usage:**
```scheme
(define hyaluronic-acid (ACTIVE_INGREDIENT "hyaluronic_acid"))
```

```python
hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
```

#### PRESERVATIVE
Antimicrobial agents that prevent bacterial and fungal growth.

**Examples:**
- Phenoxyethanol
- Parabens
- Benzyl alcohol
- Potassium sorbate

**Typical Concentrations:** 0.1% - 1.0%

#### EMULSIFIER
Ingredients that enable the blending of oil and water phases.

**Examples:**
- Cetyl alcohol
- Lecithin
- Polysorbate 80
- Stearic acid

#### HUMECTANT
Moisture-retaining agents that attract and hold water.

**Examples:**
- Glycerin
- Propylene glycol
- Sodium hyaluronate
- Aloe vera

#### SURFACTANT
Surface tension reducers used in cleansing products.

**Examples:**
- Sodium lauryl sulfate
- Cocamidopropyl betaine
- Decyl glucoside

#### THICKENER
Viscosity modifiers that provide desired texture and consistency.

**Examples:**
- Xanthan gum
- Carbomer
- Cellulose derivatives

#### EMOLLIENT
Skin conditioning agents that provide softness and smoothness.

**Examples:**
- Squalane
- Jojoba oil
- Shea butter
- Dimethicone

#### ANTIOXIDANT
Oxidation inhibitors that prevent product degradation and provide skin benefits.

**Examples:**
- Vitamin E (tocopherol)
- Vitamin C
- BHT
- Green tea extract

#### UV_FILTER
Sun protection agents that absorb or reflect UV radiation.

**Examples:**
- Zinc oxide (physical)
- Titanium dioxide (physical)
- Avobenzone (chemical)
- Octyl methoxycinnamate (chemical)

#### FRAGRANCE
Scenting agents that provide pleasant odor.

**Examples:**
- Essential oils
- Synthetic fragrances
- Parfum

#### COLORANT
Coloring agents for aesthetic appeal.

**Examples:**
- Iron oxides
- Mica
- FD&C dyes
- Natural colorants

#### PH_ADJUSTER
pH balancing agents to maintain optimal formulation pH.

**Examples:**
- Citric acid (lower pH)
- Sodium hydroxide (raise pH)
- Triethanolamine

### Formulation Types

#### SKINCARE_FORMULATION
Complete skincare products including serums, moisturizers, cleansers.

#### HAIRCARE_FORMULATION
Hair care products including shampoos, conditioners, treatments.

#### MAKEUP_FORMULATION
Cosmetic products including foundations, lipsticks, mascaras.

#### FRAGRANCE_FORMULATION
Fragrance products including perfumes, colognes, body sprays.

### Property Types

#### PH_PROPERTY
pH characteristics of ingredients and formulations.
- Typical cosmetic pH range: 4.5 - 7.0
- Skin-compatible pH: 5.5 - 6.5

#### VISCOSITY_PROPERTY
Flow characteristics and texture properties.

#### STABILITY_PROPERTY
Formulation stability over time and under various conditions.

#### TEXTURE_PROPERTY
Sensory characteristics like smoothness, greasiness, absorption.

#### SPF_PROPERTY
Sun protection factor measurements and claims.

### Interaction Types

#### COMPATIBILITY_LINK
Represents ingredients that work well together without adverse reactions.

**Example:**
```scheme
(COMPATIBILITY_LINK hyaluronic-acid niacinamide)
```

#### INCOMPATIBILITY_LINK
Represents ingredients that should not be combined due to adverse reactions.

**Example:**
```scheme
(INCOMPATIBILITY_LINK vitamin-c retinol)
```

#### SYNERGY_LINK
Represents ingredients that enhance each other's effectiveness.

**Example:**
```scheme
(SYNERGY_LINK vitamin-c vitamin-e)
```

#### ANTAGONISM_LINK
Represents ingredients that reduce each other's effectiveness.

### Safety and Regulatory Types

#### SAFETY_ASSESSMENT
Safety evaluation data and testing results.

#### ALLERGEN_CLASSIFICATION
Allergen identification and labeling requirements.

#### CONCENTRATION_LIMIT
Maximum allowed concentrations for specific ingredients.

## Common Cosmetic Ingredients Database

### Hydrating Ingredients
| Ingredient | Type | Typical Concentration | pH Range |
|------------|------|----------------------|----------|
| Hyaluronic Acid | Active | 0.1% - 2% | 5.0 - 7.0 |
| Glycerin | Humectant | 1% - 10% | 4.0 - 8.0 |
| Sodium PCA | Humectant | 0.2% - 2% | 5.0 - 7.0 |
| Aloe Vera | Humectant | 1% - 100% | 4.0 - 6.0 |

### Anti-Aging Ingredients
| Ingredient | Type | Typical Concentration | pH Range |
|------------|------|----------------------|----------|
| Retinol | Active | 0.01% - 1% | 5.5 - 6.5 |
| Niacinamide | Active | 2% - 10% | 5.0 - 7.0 |
| Vitamin C | Active | 5% - 20% | 3.5 - 4.0 |
| Alpha Hydroxy Acids | Active | 5% - 10% | 3.0 - 4.0 |

### Preservatives
| Ingredient | Type | Typical Concentration | Restrictions |
|------------|------|----------------------|--------------|
| Phenoxyethanol | Preservative | 0.5% - 1% | EU: Max 1% |
| Parabens | Preservative | 0.1% - 0.8% | Some restricted |
| Benzyl Alcohol | Preservative | 0.5% - 1% | Allergen labeling |

## Formulation Guidelines

### pH Considerations

**Optimal pH Ranges:**
- Cleansers: 5.5 - 6.5
- Toners: 4.0 - 6.0
- Serums: 5.0 - 6.5
- Moisturizers: 5.5 - 7.0
- Sunscreens: 6.0 - 8.0

**pH Compatibility Rules:**
1. Vitamin C requires pH < 4.0 for stability
2. Retinol is stable at pH 5.5 - 6.5
3. AHAs/BHAs require pH < 4.0 for efficacy
4. Niacinamide is stable at pH 5.0 - 7.0

### Stability Factors

**Heat Stability:**
- Store vitamin C formulations cool
- Retinol degrades in heat and light
- Natural extracts may require refrigeration

**Light Stability:**
- Use amber or opaque packaging for light-sensitive ingredients
- Add UV filters to protect formulation

**Oxidation Prevention:**
- Include antioxidants (vitamin E, BHT)
- Use airless packaging
- Minimize exposure to air during manufacturing

### Concentration Guidelines

**Maximum Safe Concentrations:**
- Retinol: 1% (over-the-counter)
- Vitamin C: 20%
- Niacinamide: 10%
- AHA: 10% (daily use)
- BHA: 2% (daily use)

## Regulatory Compliance

### FDA Regulations (USA)

**Cosmetic vs. Drug Classification:**
- Cosmetics: Cleanse, beautify, promote attractiveness
- Drugs: Affect body structure/function, treat/prevent disease

**Labeling Requirements:**
- INCI names required
- Allergen declarations
- Concentration limits for certain ingredients

### EU Regulations

**EU Cosmetics Regulation 1223/2009:**
- Comprehensive safety assessment required
- Prohibited and restricted ingredients list
- Allergen labeling mandatory for 26 substances

**Common EU Restrictions:**
- Formaldehyde: Max 0.2%
- Parabens: Various limits by type
- UV filters: Positive list only

### International Guidelines

**ASEAN Cosmetic Directive:**
- Similar to EU regulations
- Regional adaptations

**China Regulations:**
- Registration required for certain products
- Animal testing considerations

## Advanced Applications

### Formulation Optimization

**Systematic Ingredient Selection:**
```python
def optimize_formulation(target_properties, constraints):
    compatible_ingredients = find_compatible_ingredients(target_properties)
    filtered_ingredients = apply_constraints(compatible_ingredients, constraints)
    return rank_by_efficacy(filtered_ingredients)
```

**Compatibility Matrix Generation:**
```python
def generate_compatibility_matrix(ingredient_list):
    matrix = {}
    for i in ingredient_list:
        for j in ingredient_list:
            matrix[(i,j)] = check_compatibility(i, j)
    return matrix
```

### Stability Prediction

**pH Stability Analysis:**
```python
def predict_stability(formulation):
    ph_conflicts = check_ph_compatibility(formulation)
    chemical_conflicts = check_chemical_interactions(formulation)
    return combine_stability_scores(ph_conflicts, chemical_conflicts)
```

### Regulatory Compliance Checking

**Automated Compliance Verification:**
```python
def check_regulatory_compliance(formulation, region):
    concentration_check = verify_concentration_limits(formulation, region)
    allergen_check = verify_allergen_labeling(formulation, region)
    prohibited_check = check_prohibited_ingredients(formulation, region)
    return all([concentration_check, allergen_check, prohibited_check])
```

### Ingredient Substitution

**Compatible Alternative Finding:**
```python
def find_substitutes(target_ingredient, formulation_context):
    functional_equivalents = get_functional_equivalents(target_ingredient)
    compatible_substitutes = filter_compatible(functional_equivalents, formulation_context)
    return rank_by_performance(compatible_substitutes)
```

## Usage Examples

### Basic Ingredient Definition
```python
# Define ingredients
hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
glycerin = HUMECTANT('glycerin')
phenoxyethanol = PRESERVATIVE('phenoxyethanol')

# Create formulation
serum = SKINCARE_FORMULATION(
    hyaluronic_acid,
    glycerin,
    phenoxyethanol
)
```

### Compatibility Analysis
```python
# Check ingredient compatibility
compatible = COMPATIBILITY_LINK(hyaluronic_acid, niacinamide)
incompatible = INCOMPATIBILITY_LINK(vitamin_c, retinol)
synergy = SYNERGY_LINK(vitamin_c, vitamin_e)

# Analyze formulation compatibility
def analyze_formulation_safety(formulation):
    for ingredient1 in formulation:
        for ingredient2 in formulation:
            if incompatible(ingredient1, ingredient2):
                return False, f"Incompatible: {ingredient1} + {ingredient2}"
    return True, "Formulation is compatible"
```

### Advanced Formulation Design
```python
# Complex moisturizer formulation
moisturizer = SKINCARE_FORMULATION(
    # Active ingredients
    ACTIVE_INGREDIENT('niacinamide'),           # 5%
    ACTIVE_INGREDIENT('hyaluronic_acid'),       # 1%
    
    # Emulsion system
    EMULSIFIER('cetyl_alcohol'),                # 2%
    EMULSIFIER('stearic_acid'),                 # 1%
    
    # Humectants
    HUMECTANT('glycerin'),                      # 5%
    HUMECTANT('propylene_glycol'),              # 3%
    
    # Emollients
    EMOLLIENT('squalane'),                      # 3%
    EMOLLIENT('jojoba_oil'),                    # 2%
    
    # Preservation
    PRESERVATIVE('phenoxyethanol'),             # 0.8%
    PRESERVATIVE('ethylhexylglycerin'),         # 0.2%
    
    # pH adjustment
    PH_ADJUSTER('citric_acid')                  # q.s. to pH 6.0
)

# Add properties
pH_property = PH_PROPERTY(6.0)
viscosity_property = VISCOSITY_PROPERTY('medium')
texture_property = TEXTURE_PROPERTY('lightweight')

# Link properties to formulation
PROPERTY_LINK(moisturizer, pH_property)
PROPERTY_LINK(moisturizer, viscosity_property)
PROPERTY_LINK(moisturizer, texture_property)
```

This framework enables systematic cosmetic formulation development with built-in safety checking, regulatory compliance, and optimization capabilities.
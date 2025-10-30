# Cosmetic Chemistry Framework Documentation

## Overview

The OpenCog Cheminformatics Framework provides a comprehensive atom type system specifically designed for cosmetic chemistry applications. This framework enables systematic ingredient analysis, formulation modeling, compatibility checking, and regulatory compliance through knowledge representation and reasoning.

## Table of Contents

1. [Atom Type Reference](#atom-type-reference)
2. [Common Cosmetic Ingredients Database](#common-cosmetic-ingredients-database)
3. [Formulation Guidelines](#formulation-guidelines)
4. [Regulatory Compliance](#regulatory-compliance)
5. [Advanced Applications](#advanced-applications)
6. [Usage Examples](#usage-examples)

## Atom Type Reference

### Ingredient Category Atom Types

#### ACTIVE_INGREDIENT
**Purpose**: Represents active cosmetic ingredients that provide primary skincare benefits  
**Parent Type**: Concept  
**Usage**: Core functional ingredients that address specific skin concerns

```scheme
;; Example: Retinol as an active ingredient
(ACTIVE_INGREDIENT
  (ConceptNode "retinol")
  (ListLink
    (ConceptNode "anti-aging")
    (ConceptNode "cellular-turnover")))
```

#### PRESERVATIVE
**Purpose**: Represents preservative ingredients that prevent microbial growth  
**Parent Type**: Concept  
**Usage**: Essential for product safety and shelf life

```scheme
;; Example: Phenoxyethanol as preservative
(PRESERVATIVE
  (ConceptNode "phenoxyethanol")
  (NumberNode 1.0)) ; Maximum concentration 1.0%
```

#### EMULSIFIER
**Purpose**: Represents emulsifying agents that stabilize oil-water systems  
**Parent Type**: Concept  
**Usage**: Critical for cream and lotion stability

```scheme
;; Example: Cetyl alcohol as emulsifier
(EMULSIFIER
  (ConceptNode "cetyl_alcohol")
  (ListLink
    (ConceptNode "oil-in-water")
    (ConceptNode "water-in-oil")))
```

#### HUMECTANT
**Purpose**: Represents humectant ingredients that attract and retain moisture  
**Parent Type**: Concept  
**Usage**: Hydration and moisture retention

```scheme
;; Example: Hyaluronic acid as humectant
(HUMECTANT
  (ConceptNode "hyaluronic_acid")
  (NumberNode 1000)) ; Molecular weight affects penetration
```

#### SURFACTANT
**Purpose**: Represents surface-active agents for cleansing and emulsification  
**Parent Type**: Concept  
**Usage**: Cleansing products and foam generation

```scheme
;; Example: Sodium lauryl sulfate as surfactant
(SURFACTANT
  (ConceptNode "sodium_lauryl_sulfate")
  (ConceptNode "anionic"))
```

#### THICKENER
**Purpose**: Represents thickening agents that modify product viscosity and texture  
**Parent Type**: Concept  
**Usage**: Texture modification and product feel

```scheme
;; Example: Carbomer as thickener
(THICKENER
  (ConceptNode "carbomer")
  (NumberNode 0.5)) ; Typical usage concentration
```

#### EMOLLIENT
**Purpose**: Represents emollient ingredients that soften and smooth skin  
**Parent Type**: Concept  
**Usage**: Skin conditioning and barrier function

```scheme
;; Example: Squalane as emollient
(EMOLLIENT
  (ConceptNode "squalane")
  (ConceptNode "non-comedogenic"))
```

#### ANTIOXIDANT
**Purpose**: Represents antioxidant compounds that prevent oxidative damage  
**Parent Type**: Concept  
**Usage**: Product stability and skin protection

```scheme
;; Example: Vitamin E as antioxidant
(ANTIOXIDANT
  (ConceptNode "tocopherol")
  (ListLink
    (ConceptNode "lipid-soluble")
    (ConceptNode "free-radical-scavenger")))
```

#### UV_FILTER
**Purpose**: Represents UV filtering compounds that provide sun protection  
**Parent Type**: Concept  
**Usage**: Sunscreen and UV protection products

```scheme
;; Example: Zinc oxide as UV filter
(UV_FILTER
  (ConceptNode "zinc_oxide")
  (ListLink
    (ConceptNode "mineral")
    (ConceptNode "broad-spectrum")))
```

#### FRAGRANCE
**Purpose**: Represents fragrance ingredients that provide scent  
**Parent Type**: Concept  
**Usage**: Sensory enhancement and masking

```scheme
;; Example: Linalool as fragrance
(FRAGRANCE
  (ConceptNode "linalool")
  (ConceptNode "allergen-declaration-required"))
```

#### COLORANT
**Purpose**: Represents coloring agents for aesthetic enhancement  
**Parent Type**: Concept  
**Usage**: Product appearance and makeup applications

```scheme
;; Example: Iron oxide as colorant
(COLORANT
  (ConceptNode "iron_oxide_red")
  (ConceptNode "mineral-pigment"))
```

#### PH_ADJUSTER
**Purpose**: Represents pH adjusting agents for formulation stability  
**Parent Type**: Concept  
**Usage**: pH optimization and stability

```scheme
;; Example: Citric acid as pH adjuster
(PH_ADJUSTER
  (ConceptNode "citric_acid")
  (NumberNode 3.0)) ; Target pH
```

### Formulation Type Atom Types

#### SKINCARE_FORMULATION
**Purpose**: Represents complete skincare product formulations  
**Parent Type**: Concept  
**Usage**: Complex product modeling

```scheme
;; Example: Anti-aging serum formulation
(SKINCARE_FORMULATION
  (ConceptNode "anti_aging_serum")
  (ListLink
    (ACTIVE_INGREDIENT (ConceptNode "retinol"))
    (HUMECTANT (ConceptNode "hyaluronic_acid"))
    (ANTIOXIDANT (ConceptNode "vitamin_c"))
    (PRESERVATIVE (ConceptNode "phenoxyethanol"))))
```

#### HAIRCARE_FORMULATION
**Purpose**: Represents complete haircare product formulations  
**Parent Type**: Concept  
**Usage**: Hair product development

```scheme
;; Example: Moisturizing shampoo
(HAIRCARE_FORMULATION
  (ConceptNode "moisturizing_shampoo")
  (ListLink
    (SURFACTANT (ConceptNode "sodium_laureth_sulfate"))
    (CONDITIONING_AGENT (ConceptNode "cetrimonium_chloride"))
    (HUMECTANT (ConceptNode "glycerin"))))
```

#### MAKEUP_FORMULATION
**Purpose**: Represents complete makeup product formulations  
**Parent Type**: Concept  
**Usage**: Color cosmetic development

```scheme
;; Example: Foundation formulation
(MAKEUP_FORMULATION
  (ConceptNode "liquid_foundation")
  (ListLink
    (COLORANT (ConceptNode "titanium_dioxide"))
    (EMOLLIENT (ConceptNode "dimethicone"))
    (UV_FILTER (ConceptNode "octinoxate"))))
```

#### FRAGRANCE_FORMULATION
**Purpose**: Represents complete fragrance product formulations  
**Parent Type**: Concept  
**Usage**: Fragrance development

```scheme
;; Example: Floral perfume
(FRAGRANCE_FORMULATION
  (ConceptNode "floral_perfume")
  (ListLink
    (FRAGRANCE (ConceptNode "rose_essential_oil"))
    (FRAGRANCE (ConceptNode "jasmine_absolute"))
    (FRAGRANCE (ConceptNode "benzyl_acetate"))))
```

### Property Type Atom Types

#### PH_PROPERTY
**Purpose**: Represents pH measurements and ranges for formulations  
**Parent Type**: Predicate  
**Usage**: pH optimization and compatibility

```scheme
;; Example: pH range for vitamin C serum
(PH_PROPERTY
  (ConceptNode "vitamin_c_serum")
  (NumberNode 3.5)
  (NumberNode 4.0))
```

#### VISCOSITY_PROPERTY
**Purpose**: Represents viscosity measurements and characteristics  
**Parent Type**: Predicate  
**Usage**: Texture and application properties

```scheme
;; Example: Viscosity of moisturizer
(VISCOSITY_PROPERTY
  (ConceptNode "daily_moisturizer")
  (NumberNode 15000)) ; cP at 25°C
```

#### STABILITY_PROPERTY
**Purpose**: Represents formulation stability characteristics and testing  
**Parent Type**: Predicate  
**Usage**: Shelf life and storage conditions

```scheme
;; Example: Stability assessment
(STABILITY_PROPERTY
  (ConceptNode "anti_aging_cream")
  (ListLink
    (ConceptNode "36_months")
    (ConceptNode "room_temperature")))
```

#### TEXTURE_PROPERTY
**Purpose**: Represents sensory texture properties and feel  
**Parent Type**: Predicate  
**Usage**: Consumer experience optimization

```scheme
;; Example: Texture characteristics
(TEXTURE_PROPERTY
  (ConceptNode "night_cream")
  (ListLink
    (ConceptNode "rich")
    (ConceptNode "non-greasy")
    (ConceptNode "fast-absorbing")))
```

#### SPF_PROPERTY
**Purpose**: Represents sun protection factor measurements  
**Parent Type**: Predicate  
**Usage**: UV protection assessment

```scheme
;; Example: SPF rating
(SPF_PROPERTY
  (ConceptNode "daily_sunscreen")
  (NumberNode 30))
```

### Interaction Type Atom Types

#### COMPATIBILITY_LINK
**Purpose**: Represents compatible ingredient combinations  
**Parent Type**: Link  
**Usage**: Safe ingredient pairing

```scheme
;; Example: Compatible combination
(COMPATIBILITY_LINK
  (ACTIVE_INGREDIENT (ConceptNode "hyaluronic_acid"))
  (ACTIVE_INGREDIENT (ConceptNode "niacinamide")))
```

#### INCOMPATIBILITY_LINK
**Purpose**: Represents incompatible ingredient combinations that should be avoided  
**Parent Type**: Link  
**Usage**: Formulation risk management

```scheme
;; Example: Incompatible combination
(INCOMPATIBILITY_LINK
  (ACTIVE_INGREDIENT (ConceptNode "vitamin_c"))
  (ACTIVE_INGREDIENT (ConceptNode "retinol")))
```

#### SYNERGY_LINK
**Purpose**: Represents synergistic ingredient combinations with enhanced benefits  
**Parent Type**: Link  
**Usage**: Efficacy optimization

```scheme
;; Example: Synergistic combination
(SYNERGY_LINK
  (ANTIOXIDANT (ConceptNode "vitamin_c"))
  (ANTIOXIDANT (ConceptNode "vitamin_e")))
```

#### ANTAGONISM_LINK
**Purpose**: Represents antagonistic ingredient combinations with reduced benefits  
**Parent Type**: Link  
**Usage**: Avoiding counteractive effects

```scheme
;; Example: Antagonistic interaction
(ANTAGONISM_LINK
  (ACTIVE_INGREDIENT (ConceptNode "benzoyl_peroxide"))
  (ACTIVE_INGREDIENT (ConceptNode "tretinoin")))
```

### Safety and Regulatory Atom Types

#### SAFETY_ASSESSMENT
**Purpose**: Represents safety evaluation data for ingredients  
**Parent Type**: Predicate  
**Usage**: Safety database management

```scheme
;; Example: Safety profile
(SAFETY_ASSESSMENT
  (ConceptNode "parabens")
  (ListLink
    (ConceptNode "generally_recognized_as_safe")
    (ConceptNode "concentration_dependent")))
```

#### ALLERGEN_CLASSIFICATION
**Purpose**: Represents allergen potential and classification data  
**Parent Type**: Predicate  
**Usage**: Regulatory compliance and labeling

```scheme
;; Example: Allergen classification
(ALLERGEN_CLASSIFICATION
  (FRAGRANCE (ConceptNode "limonene"))
  (ConceptNode "eu_allergen_list"))
```

#### CONCENTRATION_LIMIT
**Purpose**: Represents regulatory concentration limits for ingredients  
**Parent Type**: Predicate  
**Usage**: Regulatory compliance

```scheme
;; Example: Concentration limit
(CONCENTRATION_LIMIT
  (PRESERVATIVE (ConceptNode "formaldehyde_donors"))
  (NumberNode 0.2)) ; 0.2% maximum
```

## Common Cosmetic Ingredients Database

### Active Ingredients

| Ingredient | INCI Name | Function | Typical Concentration | pH Range |
|------------|-----------|----------|---------------------|----------|
| Retinol | Retinol | Anti-aging, cellular turnover | 0.25-1.0% | 5.5-6.5 |
| Vitamin C | L-Ascorbic Acid | Antioxidant, brightening | 10-20% | 3.5-4.0 |
| Niacinamide | Niacinamide | Pore minimizing, oil control | 2-10% | 5.0-7.0 |
| Hyaluronic Acid | Sodium Hyaluronate | Hydration, plumping | 0.1-2.0% | 4.0-7.0 |
| Salicylic Acid | Salicylic Acid | Exfoliation, acne treatment | 0.5-2.0% | 3.0-4.0 |
| Alpha Arbutin | Alpha Arbutin | Pigmentation, brightening | 1-2% | 4.0-6.0 |

### Preservatives

| Ingredient | INCI Name | Maximum Concentration | pH Range | Spectrum |
|------------|-----------|---------------------|----------|----------|
| Phenoxyethanol | Phenoxyethanol | 1.0% | 4.0-8.0 | Broad |
| Benzyl Alcohol | Benzyl Alcohol | 1.0% | 3.0-8.0 | Limited |
| Potassium Sorbate | Potassium Sorbate | 0.6% | <6.5 | Fungi/Yeast |
| Sodium Benzoate | Sodium Benzoate | 0.5% | <4.5 | Bacteria |

### Emulsifiers

| Ingredient | INCI Name | Type | HLB Value | Application |
|------------|-----------|------|-----------|-------------|
| Cetyl Alcohol | Cetyl Alcohol | Co-emulsifier | 15.5 | Creams, lotions |
| Polysorbate 80 | Polysorbate 80 | Non-ionic | 15.0 | O/W emulsions |
| Lecithin | Lecithin | Natural | 9.0 | Natural formulations |
| Glyceryl Stearate | Glyceryl Stearate | Non-ionic | 3.8 | Rich creams |

### Humectants

| Ingredient | INCI Name | Water Binding | Penetration | Feel |
|------------|-----------|---------------|-------------|------|
| Glycerin | Glycerin | High | Surface | Sticky at high % |
| Propylene Glycol | Propylene Glycol | High | Good | Light feel |
| Butylene Glycol | Butylene Glycol | Medium | Excellent | Non-sticky |
| Sodium PCA | Sodium PCA | Very High | Good | Natural moisturizing |

## Formulation Guidelines

### pH Considerations

**Critical pH Ranges for Stability:**
- Vitamin C (L-Ascorbic Acid): pH 3.5-4.0
- Retinol: pH 5.5-6.5
- AHA/BHA: pH 3.0-4.0
- Niacinamide: pH 5.0-7.0
- Peptides: pH 4.5-7.0

**pH Compatibility Matrix:**
```
Ingredient 1    | Ingredient 2    | Compatible pH | Status
----------------|-----------------|---------------|--------
Vitamin C       | Niacinamide     | None         | Incompatible
Retinol         | AHA/BHA         | None         | Incompatible
Hyaluronic Acid | Most actives    | 4.0-7.0      | Compatible
Vitamin E       | Vitamin C       | 3.5-6.0      | Synergistic
```

### Stability Factors

**Temperature Stability:**
- Heat-sensitive: Retinol, Vitamin C, Peptides
- Heat-stable: Niacinamide, Ceramides, Hyaluronic Acid
- Requires cool storage: Vitamin C serums, Retinol products

**Light Stability:**
- Photosensitive: Retinol, Vitamin C, AHA/BHA
- Light-stable: Niacinamide, Ceramides, Peptides
- Requires opaque packaging: Most active ingredients

**Oxidation Sensitivity:**
- Highly sensitive: L-Ascorbic Acid, Retinol
- Moderately sensitive: Vitamin E, Unsaturated oils
- Stable: Magnesium Ascorbyl Phosphate, Bakuchiol

### Concentration Guidelines

**Safe Starting Concentrations:**
- First-time retinol users: 0.25%
- Vitamin C beginners: 10%
- Niacinamide: 5%
- Hyaluronic Acid: 1%
- AHA: 5%
- BHA: 0.5%

**Maximum Recommended Concentrations:**
- Retinol: 1.0% (prescription higher)
- Vitamin C: 20%
- Niacinamide: 10%
- Hyaluronic Acid: 2%
- Glycolic Acid: 10% (leave-on)
- Salicylic Acid: 2%

## Regulatory Compliance

### EU Regulations

**Restricted Ingredients:**
```scheme
;; Hydroquinone - banned in EU cosmetics
(CONCENTRATION_LIMIT
  (ACTIVE_INGREDIENT (ConceptNode "hydroquinone"))
  (NumberNode 0.0))

;; Retinol - concentration limits
(CONCENTRATION_LIMIT
  (ACTIVE_INGREDIENT (ConceptNode "retinol"))
  (NumberNode 0.3)) ; 0.3% maximum in EU
```

**Allergen Declaration Requirements:**
- Must declare if >0.001% in leave-on products
- Must declare if >0.01% in rinse-off products
- List of 26 fragrance allergens must be monitored

### FDA Guidelines

**Generally Recognized as Safe (GRAS):**
- Glycerin, Hyaluronic Acid, Niacinamide
- Most plant extracts at typical use levels
- Traditional emulsifiers and preservatives

**Drug vs Cosmetic Classification:**
- Sunscreen actives: Drug classification
- Anti-acne actives: Drug classification
- Anti-aging claims: Cosmetic (structure/function)

### Global Harmonization

**International Nomenclature of Cosmetic Ingredients (INCI):**
- Standardized naming system
- Required for international commerce
- Enables global ingredient recognition

## Advanced Applications

### Formulation Optimization

```scheme
;; Multi-objective optimization example
(define optimal-anti-aging-serum
  (SKINCARE_FORMULATION
    (ConceptNode "optimized_anti_aging_serum")
    (ListLink
      ;; Primary actives
      (ACTIVE_INGREDIENT 
        (ConceptNode "retinol")
        (NumberNode 0.5)) ; 0.5% concentration
      
      ;; Synergistic support
      (ANTIOXIDANT
        (ConceptNode "vitamin_e")
        (NumberNode 0.1))
      
      ;; Hydration support
      (HUMECTANT
        (ConceptNode "hyaluronic_acid")
        (NumberNode 1.0))
      
      ;; Barrier support
      (EMOLLIENT
        (ConceptNode "squalane")
        (NumberNode 2.0))
      
      ;; Preservation
      (PRESERVATIVE
        (ConceptNode "phenoxyethanol")
        (NumberNode 0.8)))))

;; Compatibility validation
(COMPATIBILITY_LINK
  (ConceptNode "retinol")
  (ConceptNode "vitamin_e"))

;; pH optimization
(PH_PROPERTY
  optimal-anti-aging-serum
  (NumberNode 5.5)
  (NumberNode 6.0))
```

### Stability Prediction

```scheme
;; Stability assessment framework
(define stability-model
  (lambda (formulation environmental-conditions)
    (cond
      ;; Check pH compatibility
      ((incompatible-ph? formulation) 
       (ConceptNode "unstable_ph"))
      
      ;; Check temperature sensitivity
      ((heat-sensitive? formulation environmental-conditions)
       (ConceptNode "temperature_sensitive"))
      
      ;; Check oxidation potential
      ((oxidation-prone? formulation)
       (ConceptNode "oxidation_risk"))
      
      ;; Stable formulation
      (else (ConceptNode "stable")))))
```

### Regulatory Compliance Checking

```scheme
;; Automated compliance verification
(define check-compliance
  (lambda (formulation region)
    (let ((violations '()))
      ;; Check concentration limits
      (for-each
        (lambda (ingredient)
          (when (exceeds-limit? ingredient region)
            (set! violations 
              (cons ingredient violations))))
        (get-ingredients formulation))
      
      ;; Check banned substances
      (for-each
        (lambda (ingredient)
          (when (banned? ingredient region)
            (set! violations 
              (cons ingredient violations))))
        (get-ingredients formulation))
      
      violations)))
```

### Ingredient Substitution

```scheme
;; Intelligent ingredient substitution system
(define find-substitutes
  (lambda (original-ingredient constraints)
    (filter
      (lambda (candidate)
        (and
          ;; Similar function
          (similar-function? original-ingredient candidate)
          ;; Meets constraints
          (meets-constraints? candidate constraints)
          ;; Compatible with formulation
          (compatible-with-formulation? candidate)))
      ingredient-database)))
```

## Usage Examples

### Basic Ingredient Modeling

```python
# Python interface example
from cosmetic_chemistry import *

# Create ingredient instances
hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
glycerin = HUMECTANT('glycerin') 
phenoxyethanol = PRESERVATIVE('phenoxyethanol')

# Set properties
set_property(hyaluronic_acid, 'concentration', 1.0)
set_property(hyaluronic_acid, 'molecular_weight', 1000000)
set_property(phenoxyethanol, 'max_concentration', 1.0)
```

### Formulation Creation

```python
# Create moisturizer formulation
moisturizer = SKINCARE_FORMULATION(
    'daily_moisturizer',
    ingredients=[
        hyaluronic_acid,    # Hydrating active
        cetyl_alcohol,      # Emulsifier  
        glycerin,           # Humectant
        phenoxyethanol      # Preservative
    ]
)

# Validate formulation
validation_result = validate_formulation(moisturizer)
print(f"Formulation valid: {validation_result.is_valid}")
print(f"Warnings: {validation_result.warnings}")
```

### Compatibility Analysis

```python
# Check ingredient compatibility
compatibility = check_compatibility([
    ACTIVE_INGREDIENT('vitamin_c'),
    ACTIVE_INGREDIENT('retinol')
])

if compatibility.incompatible:
    print("Warning: Vitamin C and Retinol are incompatible")
    print(f"Reason: {compatibility.reason}")
    print(f"Recommendation: {compatibility.recommendation}")
```

### Advanced Querying

```scheme
;; Find all antioxidants compatible with vitamin C
(cog-execute!
  (GetLink
    (VariableNode "$antioxidant")
    (AndLink
      (InheritanceLink
        (VariableNode "$antioxidant")
        (ConceptNode "ANTIOXIDANT"))
      (COMPATIBILITY_LINK
        (ConceptNode "vitamin_c")
        (VariableNode "$antioxidant")))))

;; Find optimal pH range for multi-active formulation
(cog-execute!
  (GetLink
    (VariableNode "$ph_range")
    (AndLink
      (SKINCARE_FORMULATION
        (ConceptNode "multi_active_serum")
        (ListLink
          (ConceptNode "vitamin_c")
          (ConceptNode "niacinamide")
          (ConceptNode "hyaluronic_acid")))
      (PH_PROPERTY
        (ConceptNode "multi_active_serum")
        (VariableNode "$ph_range")))))
```

## Conclusion

The OpenCog Cheminformatics Framework for Cosmetic Chemistry provides a comprehensive foundation for systematic cosmetic ingredient analysis and formulation development. Through its extensive atom type system, regulatory compliance features, and advanced reasoning capabilities, it enables both novice and expert formulators to create safe, effective, and compliant cosmetic products.

For additional examples and advanced use cases, see the `examples/` directory in this repository.
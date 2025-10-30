# Cosmetic Chemistry Framework for OpenCog

This document provides comprehensive guidance for using the OpenCog cheminformatics framework for cosmetic chemistry applications, including formulation modeling, ingredient analysis, and regulatory compliance.

## Overview

The cosmetic chemistry framework extends the OpenCog AtomSpace with specialized atom types designed for modeling cosmetic formulations, analyzing ingredient interactions, and ensuring regulatory compliance. The framework supports systematic analysis and optimization of cosmetic products through knowledge representation and reasoning.

## Atom Type Reference

### Base Chemical Types

#### MOLECULE
Base type for all chemical compounds and ingredients used in cosmetic formulations.

```scheme
(define hyaluronic_acid (MOLECULE "hyaluronic_acid"))
```

#### CHEMICAL_ELEMENT  
Individual chemical elements that comprise molecular structures.

```scheme
(define oxygen (CHEMICAL_ELEMENT "O"))
(define carbon (CHEMICAL_ELEMENT "C"))
```

#### FUNCTIONAL_GROUP
Chemical functional groups that determine molecular properties and behavior.

```scheme
(define hydroxyl (FUNCTIONAL_GROUP "OH"))
(define carboxyl (FUNCTIONAL_GROUP "COOH"))
```

### Cosmetic Ingredient Categories

#### ACTIVE_INGREDIENT
Primary components providing specific cosmetic benefits like anti-aging, moisturizing, or cleansing effects.

```scheme
(define retinol (ACTIVE_INGREDIENT "retinol"))
(define niacinamide (ACTIVE_INGREDIENT "niacinamide"))
(define salicylic_acid (ACTIVE_INGREDIENT "salicylic_acid"))
```

#### PRESERVATIVE
Components preventing microbial growth and extending product shelf life.

```scheme
(define phenoxyethanol (PRESERVATIVE "phenoxyethanol"))
(define potassium_sorbate (PRESERVATIVE "potassium_sorbate"))
(define benzyl_alcohol (PRESERVATIVE "benzyl_alcohol"))
```

#### EMULSIFIER
Components enabling stable mixing of oil and water phases in formulations.

```scheme
(define cetyl_alcohol (EMULSIFIER "cetyl_alcohol"))
(define polysorbate_80 (EMULSIFIER "polysorbate_80"))
(define lecithin (EMULSIFIER "lecithin"))
```

#### HUMECTANT
Components attracting and retaining moisture from the environment.

```scheme
(define glycerin (HUMECTANT "glycerin"))
(define propylene_glycol (HUMECTANT "propylene_glycol"))
(define sodium_hyaluronate (HUMECTANT "sodium_hyaluronate"))
```

#### SURFACTANT
Surface-active agents providing cleansing, foaming, or emulsifying properties.

```scheme
(define sodium_lauryl_sulfate (SURFACTANT "sodium_lauryl_sulfate"))
(define cocamidopropyl_betaine (SURFACTANT "cocamidopropyl_betaine"))
(define decyl_glucoside (SURFACTANT "decyl_glucoside"))
```

#### THICKENER
Components providing viscosity modification and texture enhancement.

```scheme
(define xanthan_gum (THICKENER "xanthan_gum"))
(define carbomer (THICKENER "carbomer"))
(define cellulose_gum (THICKENER "cellulose_gum"))
```

#### EMOLLIENT
Components providing softening, smoothing, and protective barrier properties.

```scheme
(define shea_butter (EMOLLIENT "shea_butter"))
(define jojoba_oil (EMOLLIENT "jojoba_oil"))
(define squalane (EMOLLIENT "squalane"))
```

#### ANTIOXIDANT
Components preventing oxidation, rancidity, and product degradation.

```scheme
(define vitamin_e (ANTIOXIDANT "vitamin_e"))
(define ascorbic_acid (ANTIOXIDANT "ascorbic_acid"))
(define green_tea_extract (ANTIOXIDANT "green_tea_extract"))
```

#### UV_FILTER
Components providing protection from harmful UV radiation.

```scheme
(define zinc_oxide (UV_FILTER "zinc_oxide"))
(define titanium_dioxide (UV_FILTER "titanium_dioxide"))
(define avobenzone (UV_FILTER "avobenzone"))
```

#### FRAGRANCE
Components providing pleasant scent and aromatherapeutic benefits.

```scheme
(define lavender_oil (FRAGRANCE "lavender_oil"))
(define rose_absolute (FRAGRANCE "rose_absolute"))
(define vanilla_extract (FRAGRANCE "vanilla_extract"))
```

#### COLORANT
Components providing color and visual appeal to cosmetic products.

```scheme
(define iron_oxide_red (COLORANT "iron_oxide_red"))
(define ultramarine_blue (COLORANT "ultramarine_blue"))  
(define mica (COLORANT "mica"))
```

#### PH_ADJUSTER
Components for adjusting and buffering pH levels to optimal ranges.

```scheme
(define citric_acid (PH_ADJUSTER "citric_acid"))
(define sodium_hydroxide (PH_ADJUSTER "sodium_hydroxide"))
(define triethanolamine (PH_ADJUSTER "triethanolamine"))
```

### Specialized Ingredient Subtypes

#### NATURAL_EXTRACT
Active ingredients derived from plant, mineral, or marine sources.

```scheme
(define aloe_vera_extract (NATURAL_EXTRACT "aloe_vera_extract"))
(define chamomile_extract (NATURAL_EXTRACT "chamomile_extract"))
(define sea_buckthorn_oil (NATURAL_EXTRACT "sea_buckthorn_oil"))
```

#### SYNTHETIC_ACTIVE
Synthetically produced active ingredients with precise molecular structures.

```scheme
(define synthetic_retinol (SYNTHETIC_ACTIVE "synthetic_retinol"))
(define synthetic_ceramides (SYNTHETIC_ACTIVE "synthetic_ceramides"))
```

#### PEPTIDE
Protein-based active ingredients for anti-aging and skin repair.

```scheme
(define palmitoyl_pentapeptide (PEPTIDE "palmitoyl_pentapeptide"))
(define copper_peptides (PEPTIDE "copper_peptides"))
```

#### VITAMIN
Vitamin-based active ingredients providing essential nutrients.

```scheme
(define vitamin_c (VITAMIN "vitamin_c"))
(define vitamin_a (VITAMIN "vitamin_a"))
(define vitamin_b3 (VITAMIN "vitamin_b3"))
```

#### MINERAL
Mineral-based active ingredients providing therapeutic benefits.

```scheme
(define zinc_pyrithione (MINERAL "zinc_pyrithione"))
(define kaolin_clay (MINERAL "kaolin_clay"))
(define dead_sea_salt (MINERAL "dead_sea_salt"))
```

### Formulation Types

#### SKINCARE_FORMULATION
Complete formulations designed for skin care applications.

```scheme
(define anti_aging_serum (SKINCARE_FORMULATION "anti_aging_serum"))
(define moisturizing_cream (SKINCARE_FORMULATION "moisturizing_cream"))
(define cleansing_gel (SKINCARE_FORMULATION "cleansing_gel"))
```

#### HAIRCARE_FORMULATION
Formulations specifically designed for hair and scalp care.

```scheme
(define strengthening_shampoo (HAIRCARE_FORMULATION "strengthening_shampoo"))
(define conditioning_mask (HAIRCARE_FORMULATION "conditioning_mask"))
(define scalp_treatment (HAIRCARE_FORMULATION "scalp_treatment"))
```

#### MAKEUP_FORMULATION
Formulations for decorative cosmetics and color products.

```scheme
(define liquid_foundation (MAKEUP_FORMULATION "liquid_foundation"))
(define lipstick (MAKEUP_FORMULATION "lipstick"))
(define eyeshadow_palette (MAKEUP_FORMULATION "eyeshadow_palette"))
```

#### FRAGRANCE_FORMULATION
Formulations for perfumes and scented products.

```scheme
(define eau_de_parfum (FRAGRANCE_FORMULATION "eau_de_parfum"))
(define body_mist (FRAGRANCE_FORMULATION "body_mist"))
(define scented_lotion (FRAGRANCE_FORMULATION "scented_lotion"))
```

### Property Types

#### PH_PROPERTY
pH level and buffering capacity properties.

```scheme
(define ph_5_5 (PH_PROPERTY "pH_5.5"))
(define buffered_system (PH_PROPERTY "buffered_system"))
```

#### VISCOSITY_PROPERTY
Flow characteristics and texture properties.

```scheme
(define low_viscosity (VISCOSITY_PROPERTY "low_viscosity"))
(define thixotropic (VISCOSITY_PROPERTY "thixotropic"))
(define gel_texture (VISCOSITY_PROPERTY "gel_texture"))
```

#### STABILITY_PROPERTY
Chemical and physical stability characteristics.

```scheme
(define heat_stable (STABILITY_PROPERTY "heat_stable"))
(define light_sensitive (STABILITY_PROPERTY "light_sensitive"))
(define oxidation_resistant (STABILITY_PROPERTY "oxidation_resistant"))
```

#### TEXTURE_PROPERTY
Sensory and tactile characteristics.

```scheme
(define silky_feel (TEXTURE_PROPERTY "silky_feel"))
(define quick_absorption (TEXTURE_PROPERTY "quick_absorption"))
(define non_greasy (TEXTURE_PROPERTY "non_greasy"))
```

#### SPF_PROPERTY
Sun protection factor for UV-filtering ingredients.

```scheme
(define spf_30 (SPF_PROPERTY "SPF_30"))
(define broad_spectrum (SPF_PROPERTY "broad_spectrum"))
```

### Interaction Types

#### COMPATIBILITY_LINK
Indicates ingredients that work well together without adverse interactions.

```scheme
(COMPATIBILITY_LINK
  (LIST hyaluronic_acid niacinamide))
```

#### INCOMPATIBILITY_LINK
Indicates ingredients that should not be combined due to adverse interactions.

```scheme
(INCOMPATIBILITY_LINK
  (LIST vitamin_c retinol))
```

#### SYNERGY_LINK
Indicates ingredients that enhance each other's beneficial effects.

```scheme
(SYNERGY_LINK
  (LIST vitamin_c vitamin_e))
```

#### ANTAGONISM_LINK
Indicates ingredients that counteract each other's effects.

```scheme
(ANTAGONISM_LINK
  (LIST benzoyl_peroxide retinol))
```

## Common Cosmetic Ingredients Database

### Moisturizing Agents
- **Hyaluronic Acid**: Powerful humectant capable of holding 1000x its weight in water
- **Glycerin**: Versatile humectant suitable for all skin types
- **Ceramides**: Lipid molecules that restore skin barrier function
- **Squalane**: Lightweight emollient derived from olives or sugarcane

### Anti-Aging Actives
- **Retinol**: Vitamin A derivative stimulating collagen production
- **Niacinamide**: Vitamin B3 improving skin texture and tone
- **Peptides**: Protein fragments signaling skin repair processes
- **Alpha Hydroxy Acids (AHAs)**: Chemical exfoliants promoting cell turnover

### Cleansing Ingredients
- **Sodium Lauryl Sulfate**: Strong anionic surfactant for deep cleansing
- **Cocamidopropyl Betaine**: Mild amphoteric surfactant for sensitive skin
- **Decyl Glucoside**: Plant-derived non-ionic surfactant

### Preservatives
- **Phenoxyethanol**: Broad-spectrum preservative effective against bacteria and fungi
- **Benzyl Alcohol**: Natural preservative with antimicrobial properties
- **Potassium Sorbate**: Food-grade preservative for natural formulations

## Formulation Guidelines

### pH Considerations

Most cosmetic formulations require careful pH management:

- **Skin pH**: Natural skin pH ranges from 4.5-6.5 (slightly acidic)
- **Product pH**: Should complement skin's natural pH for optimal compatibility
- **Stability pH**: Some actives require specific pH ranges for stability

```scheme
;; pH compatibility example
(HAS_PROPERTY moisturizing_cream ph_5_5)
(COMPATIBILITY_LINK
  (LIST vitamin_c citric_acid)) ; Vitamin C stable in acidic conditions
```

### Stability Factors

Key factors affecting cosmetic product stability:

1. **Temperature**: Heat can degrade active ingredients
2. **Light**: UV exposure can cause oxidation
3. **Oxygen**: Air exposure leads to rancidity
4. **pH**: Extreme pH can denature proteins and actives
5. **Microbial Growth**: Contamination reduces shelf life

```scheme
;; Stability relationships
(HAS_PROPERTY vitamin_c light_sensitive)
(HAS_PROPERTY vitamin_e oxidation_resistant)
(SYNERGY_LINK (LIST vitamin_c vitamin_e)) ; Vitamin E protects Vitamin C
```

### Concentration Guidelines

Typical concentration ranges for common ingredients:

- **Hyaluronic Acid**: 0.1-2%
- **Niacinamide**: 2-10%
- **Retinol**: 0.01-1%
- **Vitamin C**: 5-20%
- **Preservatives**: 0.1-1%

```scheme
(CONCENTRATION_LINK
  (LIST moisturizing_cream hyaluronic_acid)
  (NUMBER 1.0)) ; 1% concentration
```

## Regulatory Compliance

### Concentration Limits

Many jurisdictions impose maximum concentration limits:

```scheme
(CONCENTRATION_LIMIT retinol (NUMBER 0.3)) ; EU limit for retinol
(CONCENTRATION_LIMIT salicylic_acid (NUMBER 2.0)) ; Maximum BHA concentration
```

### Allergen Classification

Common cosmetic allergens requiring declaration:

```scheme
(ALLERGEN_CLASSIFICATION limonene "fragrance_allergen")
(ALLERGEN_CLASSIFICATION linalool "fragrance_allergen")
(ALLERGEN_CLASSIFICATION benzyl_alcohol "preservative_allergen")
```

### Safety Assessments

All cosmetic ingredients require safety evaluation:

```scheme
(SAFETY_ASSESSMENT phenoxyethanol "SCCS_approved")
(SAFETY_ASSESSMENT zinc_oxide "GRAS_status")
```

## Advanced Applications

### Formulation Optimization

The framework enables systematic formulation optimization:

```python
# Example: Finding compatible moisturizing ingredients
compatible_humectants = query_compatible_ingredients(
    base_formulation=moisturizer,
    ingredient_type=HUMECTANT,
    exclude_allergens=True
)
```

### Ingredient Substitution

Finding suitable replacements for problematic ingredients:

```python
# Example: Replacing sensitizing preservative
alternatives = find_ingredient_alternatives(
    original=parabens,
    requirements=[PRESERVATIVE, "broad_spectrum", "low_allergenicity"]
)
```

### Stability Prediction

Predicting formulation stability based on ingredient interactions:

```python
# Example: Analyzing potential stability issues
stability_analysis = evaluate_formulation_stability(
    ingredients=[vitamin_c, iron_oxide, water],
    conditions={"temperature": 25, "pH": 3.5, "oxygen_exposure": True}
)
```

### Property Modeling

Modeling physical and sensory properties:

```python
# Example: Predicting texture properties
texture_profile = predict_texture_properties(
    formulation=face_cream,
    rheology_modifiers=[xanthan_gum, carbomer],
    emollient_content=15.0
)
```

## Example Formulations

### Basic Moisturizing Cream

```scheme
(define moisturizing_cream
  (SKINCARE_FORMULATION "basic_moisturizer"))

;; Ingredients with concentrations
(CONTAINS_INGREDIENT moisturizing_cream hyaluronic_acid)
(CONCENTRATION_LINK (LIST moisturizing_cream hyaluronic_acid) (NUMBER 1.0))

(CONTAINS_INGREDIENT moisturizing_cream glycerin)
(CONCENTRATION_LINK (LIST moisturizing_cream glycerin) (NUMBER 5.0))

(CONTAINS_INGREDIENT moisturizing_cream cetyl_alcohol)
(CONCENTRATION_LINK (LIST moisturizing_cream cetyl_alcohol) (NUMBER 3.0))

(CONTAINS_INGREDIENT moisturizing_cream phenoxyethanol)
(CONCENTRATION_LINK (LIST moisturizing_cream phenoxyethanol) (NUMBER 0.5))

;; Properties
(HAS_PROPERTY moisturizing_cream ph_5_5)
(HAS_PROPERTY moisturizing_cream non_greasy)
(HAS_PROPERTY moisturizing_cream quick_absorption)
```

### Anti-Aging Serum

```scheme
(define anti_aging_serum
  (SKINCARE_FORMULATION "vitamin_c_serum"))

;; Active ingredients  
(CONTAINS_INGREDIENT anti_aging_serum vitamin_c)
(CONCENTRATION_LINK (LIST anti_aging_serum vitamin_c) (NUMBER 15.0))

(CONTAINS_INGREDIENT anti_aging_serum vitamin_e)
(CONCENTRATION_LINK (LIST anti_aging_serum vitamin_e) (NUMBER 0.5))

(CONTAINS_INGREDIENT anti_aging_serum niacinamide)
(CONCENTRATION_LINK (LIST anti_aging_serum niacinamide) (NUMBER 5.0))

;; Supporting ingredients
(CONTAINS_INGREDIENT anti_aging_serum hyaluronic_acid)
(CONTAINS_INGREDIENT anti_aging_serum citric_acid)

;; Synergistic relationships
(SYNERGY_LINK (LIST vitamin_c vitamin_e))
(COMPATIBILITY_LINK (LIST vitamin_c niacinamide))
```

### Gentle Cleanser

```scheme
(define gentle_cleanser
  (SKINCARE_FORMULATION "mild_face_wash"))

;; Primary surfactants
(CONTAINS_INGREDIENT gentle_cleanser cocamidopropyl_betaine)
(CONCENTRATION_LINK (LIST gentle_cleanser cocamidopropyl_betaine) (NUMBER 12.0))

(CONTAINS_INGREDIENT gentle_cleanser decyl_glucoside)
(CONCENTRATION_LINK (LIST gentle_cleanser decyl_glucoside) (NUMBER 8.0))

;; Conditioning agents
(CONTAINS_INGREDIENT gentle_cleanser glycerin)
(CONTAINS_INGREDIENT gentle_cleanser aloe_vera_extract)

;; Properties
(HAS_PROPERTY gentle_cleanser ph_5_5)
(HAS_PROPERTY gentle_cleanser low_irritation)
```

## Best Practices

### Formulation Development Process

1. **Define Target Properties**: Establish desired sensory and performance characteristics
2. **Select Core Actives**: Choose primary ingredients based on intended benefits
3. **Check Compatibility**: Verify ingredient interactions using compatibility links
4. **Optimize Concentrations**: Balance efficacy with safety and sensory properties
5. **Validate Stability**: Test formulation under various storage conditions
6. **Ensure Compliance**: Verify regulatory requirements are met

### Common Pitfalls to Avoid

1. **pH Incompatibility**: Mixing acids and bases without proper buffering
2. **Overloading Actives**: Using too many active ingredients causing irritation
3. **Ignoring Synergies**: Missing opportunities to enhance ingredient performance
4. **Inadequate Preservation**: Insufficient antimicrobial protection
5. **Regulatory Oversights**: Exceeding concentration limits or missing allergen declarations

### Quality Assurance

- **Ingredient Verification**: Confirm identity and purity of all components
- **Stability Testing**: Conduct accelerated aging studies
- **Microbiological Testing**: Ensure adequate preservation
- **Sensory Evaluation**: Assess consumer acceptability
- **Safety Assessment**: Complete toxicological evaluation

This framework provides a comprehensive foundation for computational cosmetic chemistry, enabling systematic analysis, optimization, and innovation in cosmetic product development through the power of knowledge representation and reasoning.
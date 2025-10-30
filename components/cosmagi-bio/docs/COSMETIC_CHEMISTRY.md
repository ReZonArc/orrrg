# Cosmetic Chemistry in OpenCog Framework

This document provides comprehensive guidance for using the OpenCog bioscience framework's cosmetic chemistry specializations. The framework enables systematic analysis, optimization, and knowledge representation for cosmetic formulations through specialized atom types and reasoning capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Atom Type Reference](#atom-type-reference)
3. [Common Cosmetic Ingredients Database](#common-cosmetic-ingredients-database)
4. [Formulation Guidelines](#formulation-guidelines)
5. [Regulatory Compliance](#regulatory-compliance)
6. [Advanced Applications](#advanced-applications)
7. [Examples](#examples)

## Overview

The cosmetic chemistry extension adds 35+ specialized atom types to the OpenCog framework, enabling:

- **Ingredient Modeling**: Systematic classification of cosmetic ingredients by function
- **Formulation Analysis**: Complex multi-ingredient formulation representation
- **Compatibility Assessment**: Modeling ingredient interactions and synergies
- **Property Prediction**: pH, viscosity, stability, and sensory analysis
- **Regulatory Compliance**: Automated checking of concentration limits and allergen requirements

## Atom Type Reference

### Ingredient Categories

#### ACTIVE_INGREDIENT
**Description**: Primary benefit-providing compounds in cosmetic formulations
**Parent**: MOLECULE_NODE
**Examples**: 
- Hyaluronic acid (hydration)
- Niacinamide (skin barrier)
- Retinol (anti-aging)
- Salicylic acid (exfoliation)

#### PRESERVATIVE
**Description**: Antimicrobial agents preventing contamination and spoilage
**Parent**: MOLECULE_NODE
**Examples**:
- Phenoxyethanol
- Methylparaben
- Benzyl alcohol
- Potassium sorbate

#### EMULSIFIER
**Description**: Compounds enabling stable oil-water phase mixing
**Parent**: MOLECULE_NODE
**Examples**:
- Cetyl alcohol
- Lecithin
- Polysorbate 80
- Glyceryl stearate

#### HUMECTANT
**Description**: Moisture-attracting and retaining compounds
**Parent**: MOLECULE_NODE
**Examples**:
- Glycerin
- Propylene glycol
- Sodium hyaluronate
- Betaine

#### SURFACTANT
**Description**: Surface-active agents for cleansing and foaming
**Parent**: MOLECULE_NODE
**Examples**:
- Sodium lauryl sulfate
- Cocamidopropyl betaine
- Decyl glucoside
- Coco-glucoside

#### THICKENER
**Description**: Viscosity-modifying compounds for texture control
**Parent**: MOLECULE_NODE
**Examples**:
- Xanthan gum
- Carbomer
- Hydroxyethylcellulose
- Acrylates copolymer

#### EMOLLIENT
**Description**: Skin-softening and conditioning agents
**Parent**: MOLECULE_NODE
**Examples**:
- Squalane
- Jojoba oil
- Isopropyl myristate
- Dimethicone

#### ANTIOXIDANT
**Description**: Compounds preventing oxidative degradation
**Parent**: MOLECULE_NODE
**Examples**:
- Tocopherol (Vitamin E)
- Ascorbic acid (Vitamin C)
- BHT
- Rosemary extract

#### UV_FILTER
**Description**: UV-blocking compounds for sun protection
**Parent**: MOLECULE_NODE
**Examples**:
- Zinc oxide
- Titanium dioxide
- Avobenzone
- Octinoxate

#### FRAGRANCE
**Description**: Scent-providing compounds
**Parent**: MOLECULE_NODE
**Examples**:
- Linalool
- Limonene
- Geraniol
- Citronellol

#### COLORANT
**Description**: Pigments and dyes for color effects
**Parent**: MOLECULE_NODE
**Examples**:
- Iron oxides
- Titanium dioxide
- Ultramarine blue
- Carmine

#### PH_ADJUSTER
**Description**: pH-modifying compounds for optimal formulation pH
**Parent**: MOLECULE_NODE
**Examples**:
- Triethanolamine
- Sodium hydroxide
- Citric acid
- Lactic acid

### Formulation Types

#### SKINCARE_FORMULATION
**Description**: Complete skincare product formulations
**Parent**: CONCEPT_NODE
**Usage**: Contains multiple ingredients with specific ratios and interactions

#### HAIRCARE_FORMULATION
**Description**: Complete haircare product formulations
**Parent**: CONCEPT_NODE
**Usage**: Specialized for hair treatment and styling products

#### MAKEUP_FORMULATION
**Description**: Complete makeup product formulations
**Parent**: CONCEPT_NODE
**Usage**: Color cosmetics with pigment, binder, and modifier components

#### FRAGRANCE_FORMULATION
**Description**: Complete fragrance product formulations
**Parent**: CONCEPT_NODE
**Usage**: Complex scent compositions with top, middle, and base notes

### Property Types

#### PH_PROPERTY
**Description**: pH characteristics of formulations (typically 4.5-7.0 for skin compatibility)
**Parent**: CONCEPT_NODE

#### VISCOSITY_PROPERTY
**Description**: Flow and texture characteristics
**Parent**: CONCEPT_NODE

#### STABILITY_PROPERTY
**Description**: Formulation stability over time and conditions
**Parent**: CONCEPT_NODE

#### TEXTURE_PROPERTY
**Description**: Sensory and feel characteristics
**Parent**: CONCEPT_NODE

#### SPF_PROPERTY
**Description**: Sun protection factor characteristics
**Parent**: CONCEPT_NODE

### Interaction Types

#### COMPATIBILITY_LINK
**Description**: Positive ingredient interactions
**Parent**: LINK
**Usage**: Links ingredients that work well together

#### INCOMPATIBILITY_LINK
**Description**: Negative ingredient interactions
**Parent**: LINK
**Usage**: Links ingredients that should not be combined

#### SYNERGY_LINK
**Description**: Beneficial ingredient interactions
**Parent**: LINK
**Usage**: Links ingredients that enhance each other's effects

#### ANTAGONISM_LINK
**Description**: Counteractive ingredient interactions
**Parent**: LINK
**Usage**: Links ingredients that reduce each other's effectiveness

### Safety and Regulatory

#### SAFETY_ASSESSMENT
**Description**: Safety evaluation data
**Parent**: CONCEPT_NODE

#### ALLERGEN_CLASSIFICATION
**Description**: Allergenicity classifications
**Parent**: CONCEPT_NODE

#### CONCENTRATION_LIMIT
**Description**: Maximum allowed concentrations
**Parent**: CONCEPT_NODE

## Common Cosmetic Ingredients Database

### Hydrating Ingredients
- **Hyaluronic Acid**: Powerful humectant, can hold 1000x its weight in water
- **Glycerin**: Multi-functional humectant and solvent
- **Sodium PCA**: Natural moisturizing factor component
- **Ceramides**: Skin barrier lipids

### Anti-Aging Actives
- **Retinol**: Vitamin A derivative for cell turnover
- **Niacinamide**: Vitamin B3 for barrier function and texture
- **Peptides**: Collagen-stimulating compounds
- **Alpha Hydroxy Acids**: Chemical exfoliants

### Cleansing Agents
- **Sodium Lauryl Sulfate**: Strong anionic surfactant
- **Cocamidopropyl Betaine**: Gentle amphoteric surfactant
- **Decyl Glucoside**: Mild non-ionic surfactant
- **Sodium Cocoyl Isethionate**: Gentle synthetic detergent

## Formulation Guidelines

### pH Considerations
- **Skin compatibility**: pH 4.5-6.5 optimal for most skin types
- **Ingredient stability**: Some actives require specific pH ranges
- **Preservative efficacy**: Most preservatives work best at pH < 6.0

### Stability Factors
- **Temperature**: Heat accelerates degradation
- **Light exposure**: UV light degrades many actives
- **Oxygen exposure**: Antioxidants prevent oxidation
- **Microbial contamination**: Preservatives prevent spoilage

### Concentration Guidelines
- **Active ingredients**: Usually 0.1-10% depending on potency
- **Preservatives**: Typically 0.1-1.0%
- **Emulsifiers**: Usually 1-5%
- **Thickeners**: Typically 0.1-2%

## Regulatory Compliance

### EU Regulations
- **Cosmetic Regulation (EC) No 1223/2009**: Comprehensive cosmetic safety requirements
- **INCI naming**: International nomenclature for ingredient listing
- **Allergen declaration**: 26 fragrance allergens must be declared if >0.001% (leave-on) or >0.01% (rinse-off)

### FDA Guidelines
- **GRAS ingredients**: Generally recognized as safe substances
- **Color additive approval**: Colors require specific FDA approval
- **SPF testing**: Standardized testing protocols for sun protection claims

### Common Restrictions
- **Parabens**: Some restricted in children's products
- **Formaldehyde releasers**: Concentration limits apply
- **Heavy metals**: Strict limits on lead, mercury, arsenic
- **Prohibited substances**: Complete ban on certain ingredients

## Advanced Applications

### Formulation Optimization
Use the framework to:
- Systematically test ingredient combinations
- Predict formulation stability
- Optimize texture and sensory properties
- Balance efficacy with safety

### Ingredient Substitution
- Find compatible alternatives for restricted ingredients
- Optimize cost while maintaining performance
- Address supply chain constraints
- Develop allergen-free formulations

### Property Modeling
- Predict pH changes during formulation
- Model viscosity based on thickener combinations
- Calculate SPF from UV filter concentrations
- Assess sensory characteristics

### Regulatory Compliance Checking
- Automated concentration limit verification
- Allergen declaration requirements
- INCI name validation
- Regional regulation compliance

## Examples

### Basic Ingredient Modeling
```python
# Define basic cosmetic ingredients
hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
glycerin = HUMECTANT('glycerin')
phenoxyethanol = PRESERVATIVE('phenoxyethanol')
```

### Complex Formulation
```python
# Create a complete moisturizer formulation
moisturizer = SKINCARE_FORMULATION(
    ACTIVE_INGREDIENT('hyaluronic_acid'),
    EMULSIFIER('cetyl_alcohol'), 
    HUMECTANT('glycerin'),
    EMOLLIENT('squalane'),
    PRESERVATIVE('phenoxyethanol'),
    THICKENER('xanthan_gum')
)
```

### Compatibility Analysis
```scheme
; Define ingredient interactions
(define compatible-pair
  (COMPATIBILITY_LINK
    (ACTIVE_INGREDIENT "hyaluronic_acid")
    (ACTIVE_INGREDIENT "niacinamide")))

(define incompatible-pair
  (INCOMPATIBILITY_LINK
    (ACTIVE_INGREDIENT "vitamin_c")
    (ACTIVE_INGREDIENT "retinol")))

(define synergistic-pair
  (SYNERGY_LINK
    (ACTIVE_INGREDIENT "vitamin_c")
    (ANTIOXIDANT "vitamin_e")))
```

### Property Specification
```python
# Define formulation properties
ph_spec = PH_PROPERTY('pH_5.5')
viscosity_spec = VISCOSITY_PROPERTY('medium_viscosity')
stability_spec = STABILITY_PROPERTY('12_month_stable')
```

For complete working examples, see the `/examples` directory with both Python and Scheme implementations demonstrating practical usage of the cosmetic chemistry framework.
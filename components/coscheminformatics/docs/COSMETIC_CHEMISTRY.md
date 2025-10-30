# Cosmetic Chemistry Informatics

This module extends the OpenCog cheminformatics framework with specialized support for cosmetic chemistry applications. It provides atom types, properties, and relationships specific to cosmetic formulation, ingredient analysis, and product development.

## Overview

The cosmetic chemistry specializations enable representation and analysis of:
- Cosmetic ingredients and their functional categories
- Formulation types and compositions  
- Ingredient interactions and compatibility
- Product properties and performance characteristics
- Safety and regulatory information

## Atom Types

### Cosmetic Ingredient Types

**Base Types:**
- `COSMETIC_INGREDIENT`: Base class for all cosmetic ingredients
- `ACTIVE_INGREDIENT`: Ingredients providing the primary functional benefit
- `PRESERVATIVE`: Antimicrobial agents preventing contamination
- `EMULSIFIER`: Agents enabling oil and water phase mixing
- `HUMECTANT`: Moisture-attracting and retaining ingredients
- `SURFACTANT`: Surface-active agents for cleansing and foaming
- `THICKENER`: Viscosity-modifying agents
- `EMOLLIENT`: Skin-softening and conditioning agents
- `ANTIOXIDANT`: Ingredients preventing oxidative degradation
- `UV_FILTER`: Sun protection ingredients
- `FRAGRANCE`: Scent-providing ingredients
- `COLORANT`: Color-providing ingredients
- `PH_ADJUSTER`: pH modification agents

### Formulation Types

- `COSMETIC_FORMULATION`: Base formulation type
- `SKINCARE_FORMULATION`: Skin care products (creams, serums, etc.)
- `HAIRCARE_FORMULATION`: Hair care products (shampoos, conditioners, etc.)
- `MAKEUP_FORMULATION`: Color cosmetics (foundations, lipsticks, etc.)
- `FRAGRANCE_FORMULATION`: Perfumes and fragrances

### Property Types

- `PH_PROPERTY`: pH level characteristics
- `VISCOSITY_PROPERTY`: Flow and texture characteristics
- `STABILITY_PROPERTY`: Product stability over time
- `TEXTURE_PROPERTY`: Sensorial feel characteristics
- `COLOR_PROPERTY`: Visual appearance characteristics
- `SCENT_PROPERTY`: Olfactory characteristics
- `SPF_PROPERTY`: Sun protection factor

### Interaction Types

- `COMPATIBILITY_LINK`: Compatible ingredient combinations
- `INCOMPATIBILITY_LINK`: Incompatible ingredient combinations
- `SYNERGY_LINK`: Ingredients with enhanced combined effects
- `ANTAGONISM_LINK`: Ingredients with opposing effects

### Safety and Regulatory Types

- `SAFETY_ASSESSMENT`: Safety evaluation information
- `ALLERGEN_CLASSIFICATION`: Allergen categorization
- `CONCENTRATION_LIMIT`: Usage level restrictions

## Usage Examples

### Basic Ingredient Definition

```python
# Define a moisturizing ingredient
hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
humectant_prop = HUMECTANT('hyaluronic_acid')

# Define a preservative
phenoxyethanol = PRESERVATIVE('phenoxyethanol')
```

### Formulation Creation

```python
# Create a skincare formulation
moisturizer = SKINCARE_FORMULATION(
    ACTIVE_INGREDIENT('hyaluronic_acid'),
    EMULSIFIER('cetyl_alcohol'),
    PRESERVATIVE('phenoxyethanol'),
    HUMECTANT('glycerin')
)
```

### Property Assignment

```python
# Assign pH property to formulation
pH_neutral = PH_PROPERTY('neutral', 7.0)
moisturizer_pH = COMPATIBILITY_LINK(moisturizer, pH_neutral)
```

### Ingredient Interactions

```python
# Define ingredient compatibility
vitamin_c = ACTIVE_INGREDIENT('ascorbic_acid')
vitamin_e = ANTIOXIDANT('tocopherol')
antioxidant_synergy = SYNERGY_LINK(vitamin_c, vitamin_e)

# Define incompatibility
retinol = ACTIVE_INGREDIENT('retinol')
incompatible = INCOMPATIBILITY_LINK(vitamin_c, retinol)
```

## Common Cosmetic Ingredients

### Active Ingredients
- Retinol (anti-aging)
- Hyaluronic Acid (moisturizing)
- Salicylic Acid (exfoliating)
- Niacinamide (skin conditioning)
- Vitamin C (antioxidant)

### Preservatives
- Phenoxyethanol
- Methylparaben
- Potassium Sorbate
- Benzyl Alcohol

### Emulsifiers
- Cetyl Alcohol
- Stearic Acid
- Polysorbate 20
- Lecithin

### UV Filters
- Zinc Oxide (physical)
- Titanium Dioxide (physical) 
- Avobenzone (chemical)
- Octinoxate (chemical)

## Formulation Guidelines

### pH Considerations
- Most skin care products: pH 4.5-7.0
- Cleansers: pH 5.0-7.0
- Exfoliants: pH 3.0-4.0

### Stability Factors
- Temperature stability
- Light sensitivity
- Oxidation susceptibility
- Microbial stability

### Regulatory Compliance
- INCI (International Nomenclature of Cosmetic Ingredients)
- FDA regulations (US)
- EU Cosmetics Regulation
- Allergen declarations

## Advanced Applications

### Ingredient Optimization
Use the framework to:
- Analyze ingredient compatibility matrices
- Optimize formulation stability
- Predict product performance
- Assess regulatory compliance

### Safety Assessment
- Evaluate allergen potential
- Check concentration limits
- Assess ingredient interactions
- Monitor regulatory changes

This framework provides a foundation for computational cosmetic chemistry, enabling systematic analysis and optimization of cosmetic formulations through knowledge representation and reasoning.
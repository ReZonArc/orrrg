# OpenCog Cheminformatics Framework - Cosmetic Chemistry Specializations

This directory contains the cosmetic chemistry specializations for the OpenCog cheminformatics framework, providing comprehensive support for cosmetic formulation modeling, ingredient analysis, and regulatory compliance.

## Directory Structure

```
cheminformatics/
├── __init__.py                    # Main package initialization
├── README.md                      # This file
└── types/
    ├── __init__.py               # Types package initialization
    ├── atom_types.script         # OpenCog atom type definitions (Scheme)
    └── cosmetic_atoms.py         # Python implementations for standalone use
```

## Features

### 🧪 Extended Atom Type System
- **35+ cosmetic-specific atom types** covering all major ingredient categories
- **Ingredient classifications**: Active ingredients, preservatives, emulsifiers, humectants, surfactants, thickeners, emollients, antioxidants, UV filters, fragrances, colorants, pH adjusters
- **Formulation types**: Skincare, haircare, makeup, and fragrance formulations
- **Property modeling**: pH, viscosity, stability, texture, and SPF properties
- **Interaction types**: Compatibility, incompatibility, synergy, and antagonism links
- **Safety & regulatory**: Safety assessments, allergen classifications, concentration limits

### 🔬 Core Capabilities
- **Ingredient Modeling**: Systematic representation with functional classifications
- **Formulation Creation**: Complex cosmetic formulations with compatibility analysis
- **Safety Assessment**: Automated regulatory compliance checking
- **Property Analysis**: pH, stability, and sensory property modeling
- **Interaction Prediction**: Compatibility and synergy analysis

## Quick Start

### Using Python Classes (Standalone)

```python
from cheminformatics.types.cosmetic_atoms import (
    ACTIVE_INGREDIENT, HUMECTANT, PRESERVATIVE, 
    SKINCARE_FORMULATION, COMPATIBILITY_LINK
)

# Create ingredients
hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
glycerin = HUMECTANT('glycerin')
phenoxyethanol = PRESERVATIVE('phenoxyethanol')

# Create formulation
serum = SKINCARE_FORMULATION(
    hyaluronic_acid,
    glycerin,
    phenoxyethanol
)

# Define compatibility
compatible = COMPATIBILITY_LINK(hyaluronic_acid, glycerin)
```

### Using OpenCog Scheme (Full Framework)

```scheme
;; Load atom types
(load "cheminformatics/types/atom_types.script")

;; Define ingredients
(define hyaluronic-acid (ACTIVE_INGREDIENT "hyaluronic_acid"))
(define glycerin (HUMECTANT "glycerin"))

;; Create compatibility relationship
(COMPATIBILITY_LINK hyaluronic-acid glycerin)
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- **Python Examples**:
  - `cosmetic_intro_example.py` - Basic ingredient modeling and formulation
  - `cosmetic_chemistry_example.py` - Advanced analysis and optimization

- **Scheme Examples**:
  - `cosmetic_formulation.scm` - Complex formulation modeling
  - `cosmetic_compatibility.scm` - Simple compatibility checking

## Testing

Run the unit tests to validate functionality:

```bash
python -m unittest test.cosmetic_chemistry.test_cosmetic_atoms -v
```

## Documentation

For complete documentation, see `docs/COSMETIC_CHEMISTRY.md` which includes:
- Complete atom type reference
- Common cosmetic ingredients database
- Formulation guidelines and pH considerations
- Regulatory compliance information
- Advanced applications and use cases

## Integration with Main Framework

This cosmetic chemistry framework integrates seamlessly with the main ChemReasoner system, providing specialized capabilities for cosmetic formulation discovery and optimization alongside the existing catalyst research capabilities.

## License

This implementation follows the same license as the main ChemReasoner project.
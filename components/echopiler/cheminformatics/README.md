# Cosmetic Chemistry Cheminformatics Framework

This directory contains the implementation of a comprehensive cosmetic chemistry specialization for the OpenCog cheminformatics framework, integrated into the Compiler Explorer ecosystem.

## Overview

The cosmetic chemistry framework provides specialized atom types, formulation modeling, and ingredient analysis capabilities specifically designed for cosmetic product development and analysis.

**NEW: Meta-Optimization Engine** - A comprehensive strategy for generating optimal formulations for every possible condition and treatment combination.

## Structure

```
cheminformatics/
├── README.md                    # This file
├── types/
│   └── atom_types.script       # OpenCog atom type definitions
├── docs/
│   └── COSMETIC_CHEMISTRY.md   # Comprehensive documentation
├── examples/
│   ├── python/
│   │   ├── cosmetic_intro_example.py      # Basic introduction
│   │   └── cosmetic_chemistry_example.py   # Advanced analysis
│   └── scheme/
│       ├── cosmetic_formulation.scm       # Complex modeling
│       └── cosmetic_compatibility.scm     # Compatibility checking
└── types/
    └── cheminformatics/
        └── cosmetic-chemistry.interfaces.ts  # TypeScript interfaces
```

## Key Features

### 1. Extended Atom Type System
- **35+ specialized atom types** for cosmetic chemistry
- **Ingredient categories**: ACTIVE_INGREDIENT, PRESERVATIVE, EMULSIFIER, etc.
- **Formulation types**: SKINCARE_FORMULATION, HAIRCARE_FORMULATION, etc.
- **Property types**: PH_PROPERTY, VISCOSITY_PROPERTY, STABILITY_PROPERTY, etc.
- **Interaction types**: COMPATIBILITY_LINK, SYNERGY_LINK, etc.

### 2. Comprehensive Documentation
- **Complete atom type reference** with usage examples
- **Common ingredients database** with properties and interactions
- **Formulation guidelines** including pH considerations and stability factors
- **Regulatory compliance information** for different jurisdictions
- **Best practices** and practical applications

### 3. Example Implementations

#### Python Examples
- **Basic Introduction** (`cosmetic_intro_example.py`)
  - Ingredient classification and formulation creation
  - Compatibility analysis and knowledge base queries
  - 170+ lines of commented code

- **Advanced Analysis** (`cosmetic_chemistry_example.py`)
  - Formulation optimization using constraint satisfaction
  - Stability analysis and regulatory compliance checking
  - Ingredient substitution and alternative finding
  - 650+ lines of production-ready code

#### Scheme Examples
- **Complex Formulation** (`cosmetic_formulation.scm`)
  - Advanced ingredient modeling with detailed properties
  - Multi-formulation compatibility analysis
  - Query functions and stability assessment
  - 470+ lines of functional programming

- **Compatibility Checking** (`cosmetic_compatibility.scm`)
  - Simple ingredient interaction modeling
  - Automated compatibility matrix generation
  - Formulation testing and validation
  - 420+ lines with demonstration functions

### 4. TypeScript Integration
- **Comprehensive type definitions** for cosmetic chemistry
- **Full integration** with Compiler Explorer's type system
- **290+ lines** of strictly typed interfaces
- **Test coverage** with Vitest framework

## Core Concepts

### Ingredient Modeling
```typescript
interface CosmeticIngredient {
    category: IngredientCategory;
    functions: string[];
    solubility: 'water_soluble' | 'oil_soluble' | 'both' | 'insoluble';
    ph_stability_range?: { min: number; max: number };
    allergenicity: 'very_low' | 'low' | 'medium' | 'high';
    // ... additional properties
}
```

### Formulation Analysis
```typescript
interface FormulationAnalysis {
    compatibility_matrix: CompatibilityMatrix;
    stability_assessment: StabilityData;
    regulatory_status: RegulatoryCompliance;
    optimization_suggestions: OptimizationSuggestion[];
    quality_score: number;
}
```

### Ingredient Interactions
```scheme
;; Compatible ingredients
(EvaluationLink
  (PredicateNode "compatible")
  (ListLink hyaluronic-acid niacinamide))

;; Synergistic relationships
(EvaluationLink
  (PredicateNode "synergistic")
  (ListLink vitamin-c vitamin-e))
```

## Usage Examples

### Basic Ingredient Creation (Python)
```python
from cheminformatics.cosmetic import CosmeticIngredient

hyaluronic_acid = CosmeticIngredient(
    name="hyaluronic_acid",
    category="ACTIVE_INGREDIENT",
    functions=["moisturizing", "anti_aging"],
    solubility="water_soluble"
)
```

### Formulation Optimization (Python)
```python
framework = CosmeticChemistryFramework()
optimized = framework.optimize_formulation(
    formulation=anti_aging_serum,
    constraints={'max_total_actives': 20.0}
)
```

### Compatibility Analysis (Scheme)
```scheme
(define compatibility-result
  (check-compatibility vitamin-c retinol))
```

### Meta-Optimization Strategy (TypeScript) - **NEW**
```typescript
import { MetaOptimizationEngine } from '../lib/cheminformatics/meta-optimization-engine.js';

// Initialize meta-optimization engine
const metaEngine = new MetaOptimizationEngine({
    max_combinations: 100,
    enable_caching: true,
    performance_tracking: true
});

// Get comprehensive condition-treatment matrix
const matrix = metaEngine.getConditionTreatmentMatrix();
console.log(`Generated ${matrix.combinations.length} combinations`);

// Optimize for specific combination
const result = await metaEngine.optimizeForCombination(
    'combo_0', // Anti-aging combination
    'mature',  // Skin type
    { budget_limit: 150 }
);

// Run comprehensive optimization for all combinations
const summary = await metaEngine.optimizeAllCombinations('normal');
console.log(`Optimized ${summary.successful_optimizations} formulations`);
console.log(`Best score: ${summary.performance_analytics.best_overall_score}`);
```

## Practical Applications

1. **Formulation Development**
   - Systematic ingredient selection
   - Concentration optimization
   - Compatibility verification

2. **Stability Analysis**
   - pH compatibility checking
   - Oxidation risk assessment
   - Storage condition recommendations

3. **Regulatory Compliance**
   - Concentration limit enforcement
   - Allergen declaration requirements
   - Regional regulation compliance

4. **Ingredient Substitution**
   - Alternative ingredient identification
   - Property-matched replacements
   - Cost and regulatory impact analysis

5. **Meta-Optimization Strategy (NEW)**
   - Automatic condition-treatment matrix generation (112+ combinations)
   - Intelligent optimization strategy selection based on complexity
   - Performance analytics and result caching
   - Contextual recommendations for each formulation

## Technical Implementation

### OpenCog Integration
- Uses OpenCog AtomSpace for knowledge representation
- Leverages reasoning capabilities for inference
- Maintains compatibility with existing cheminformatics tools

### Compiler Explorer Integration
- TypeScript interfaces for type safety
- Full integration with existing build system
- Comprehensive test coverage

### Performance Considerations
- Efficient atom type hierarchy
- Optimized query functions
- Scalable knowledge base design

## Testing

The framework includes comprehensive tests covering:
- Type safety and validation
- Ingredient modeling accuracy
- Formulation analysis correctness
- Regulatory compliance checking
- Integration with Compiler Explorer

```bash
npm run test -- --run test/cheminformatics/cosmetic-chemistry-tests.ts
npm run test -- --run test/cheminformatics/hypergredient-framework-tests.ts  
npm run test -- --run test/cheminformatics/multiscale-optimization-tests.ts
npm run test -- --run test/cheminformatics/meta-optimization-tests.ts  # NEW
```

## Future Enhancements

1. **Machine Learning Integration**
   - Predictive modeling for formulation properties
   - Automated optimization algorithms
   - Pattern recognition for ingredient interactions

2. **Extended Regulatory Support**
   - Additional jurisdictions (FDA, Health Canada, etc.)
   - Real-time regulation updates
   - Automated compliance reporting

3. **Advanced Analytics**
   - Consumer preference modeling
   - Market trend analysis
   - Sustainability metrics integration

4. **Web Interface**
   - Interactive formulation builder
   - Visual compatibility matrices
   - Real-time analysis dashboard

## Contributing

When contributing to the cosmetic chemistry framework:

1. Follow existing code patterns and documentation standards
2. Ensure TypeScript type safety
3. Add comprehensive tests for new features
4. Update documentation for any API changes
5. Consider regulatory implications of new features

## License

This implementation is part of the Compiler Explorer project and follows the same BSD-2-Clause license terms.

## References

- OpenCog Cheminformatics Documentation
- International Nomenclature of Cosmetic Ingredients (INCI)
- EU Cosmetic Regulation 1223/2009
- FDA Cosmetic Guidelines
- Cosmetic Chemistry Literature and Standards
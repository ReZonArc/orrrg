"""
OpenCog Integration for Cosmeceutical Formulation

This package provides OpenCog-inspired features for multiscale constraint optimization
in cosmeceutical formulation, including:

- AtomSpace hypergraph representation
- INCI-driven search space reduction
- Adaptive attention allocation (ECAN-inspired)
- PLN-based reasoning for ingredient interactions
- MOSES-inspired optimization routines
- Multiscale constraint satisfaction

Modules:
    atomspace: AtomSpace-inspired hypergraph representation
    inci_optimization: INCI-based search space reduction
    attention: Adaptive attention allocation mechanisms
    reasoning: PLN-inspired probabilistic reasoning
    optimization: MOSES-inspired optimization algorithms
    multiscale: Multiscale constraint optimization framework
"""

__version__ = "1.0.0"
__author__ = "OpenCog Cheminformatics Team"

# Import main classes for convenience
try:
    from .atomspace import CosmeceuticalAtomSpace, Atom, AtomType
    from .inci_optimization import INCISearchOptimizer, INCIEntry, RegulationRegion
    from .attention import AdaptiveAttentionAllocator, AttentionValue, AttentionType
    from .reasoning import IngredientReasoningEngine, TruthValue, InferenceRule
    from .optimization import MultiscaleOptimizer, FormulationGenome, OptimizationObjective
    from .multiscale import SkinModelIntegrator, SkinLayer, DeliveryMechanism, TherapeuticVector
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Some OpenCog integration features may not be available: {e}")
    # Define minimal fallback classes if needed
    pass
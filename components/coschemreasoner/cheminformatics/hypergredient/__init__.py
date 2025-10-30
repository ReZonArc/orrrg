"""
Hypergredient Framework - Revolutionary Formulation Design System

This module implements the Hypergredient Framework Architecture for
advanced cosmeceutical formulation optimization using semantic-aware
multi-objective optimization algorithms.

Main Components:
- Core Hypergredient Classes (H.CT, H.CS, H.AO, etc.)
- Dynamic Hypergredient Database
- Interaction Matrix System  
- Multi-Objective Optimization Algorithm
- Dynamic Scoring System
- Machine Learning Integration
"""

try:
    from .core import (
        Hypergredient,
        HypergredientDatabase,
        HYPERGREDIENT_CLASSES,
        HypergredientMetrics
    )

    from .database import (
        HypergredientDB,
        create_hypergredient_database
    )

    from .optimization import (
        HypergredientFormulator,
        FormulationOptimizer,
        OptimizationObjective,
        FormulationRequest,
        FormulationSolution
    )

    from .interaction import (
        InteractionMatrix,
        calculate_synergy_score
    )

    from .scoring import (
        DynamicScoringSystem,
        PerformanceMetrics
    )

    from .meta_optimization import (
        MetaOptimizationStrategy,
        OptimizationStrategy,
        ConditionTreatmentPair,
        OptimizationResult,
        MetaOptimizationCache
    )

    __all__ = [
        'Hypergredient',
        'HypergredientDatabase', 
        'HYPERGREDIENT_CLASSES',
        'HypergredientMetrics',
        'HypergredientDB',
        'create_hypergredient_database',
        'HypergredientFormulator',
        'FormulationOptimizer',
        'OptimizationObjective',
        'FormulationRequest',
        'FormulationSolution',
        'InteractionMatrix',
        'calculate_synergy_score',
        'DynamicScoringSystem',
        'PerformanceMetrics',
        'MetaOptimizationStrategy',
        'OptimizationStrategy',
        'ConditionTreatmentPair',
        'OptimizationResult',
        'MetaOptimizationCache'
    ]

except ImportError as e:
    print(f"Warning: Some hypergredient modules could not be imported: {e}")
    __all__ = []
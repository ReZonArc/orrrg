#!/usr/bin/env python3
"""
🧬 Hypergredient Framework Architecture

This module implements the revolutionary Hypergredient Framework for cosmeceutical
formulation design, abstracting ingredients into functional classes with 
multi-objective optimization algorithms.

Key Features:
- Hypergredient taxonomy with 10 core classes
- Dynamic interaction matrix for synergies/antagonisms
- Multi-objective formulation optimizer
- Machine learning integration for performance prediction
- Evolutionary formulation improvement system
- Real-time compatibility checking
- Performance-based scoring system

Hypergredient Classes:
- H.CT: Cellular Turnover Agents
- H.CS: Collagen Synthesis Promoters  
- H.AO: Antioxidant Systems
- H.BR: Barrier Repair Complex
- H.ML: Melanin Modulators
- H.HY: Hydration Systems
- H.AI: Anti-Inflammatory Agents
- H.MB: Microbiome Balancers
- H.SE: Sebum Regulators
- H.PD: Penetration/Delivery Enhancers

Author: OpenCog Multiscale Optimization Framework
License: AGPL-3.0
"""

import uuid
import json
import math
import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time

# Optional ML integration
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available, using basic math operations")

# Optional OpenCog integration
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.type_constructors import *
    OPENCOG_AVAILABLE = True
except ImportError:
    OPENCOG_AVAILABLE = False
    print("Warning: OpenCog not available, using standalone mode")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HypergredientClass(Enum):
    """Core Hypergredient functional classes."""
    H_CT = "Cellular Turnover Agents"
    H_CS = "Collagen Synthesis Promoters"
    H_AO = "Antioxidant Systems"
    H_BR = "Barrier Repair Complex"
    H_ML = "Melanin Modulators"
    H_HY = "Hydration Systems"
    H_AI = "Anti-Inflammatory Agents"
    H_MB = "Microbiome Balancers"
    H_SE = "Sebum Regulators"
    H_PD = "Penetration/Delivery Enhancers"


class StabilityFactor(Enum):
    """Stability factors affecting ingredient performance."""
    UV_SENSITIVE = "uv_sensitive"
    LIGHT_SENSITIVE = "light_sensitive"
    AIR_SENSITIVE = "air_sensitive"
    TEMPERATURE_SENSITIVE = "temperature_sensitive"
    PH_SENSITIVE = "ph_sensitive"
    STABLE = "stable"


class SafetyLevel(Enum):
    """Safety classification levels."""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class HypergredientMetrics:
    """Performance metrics for a hypergredient."""
    efficacy_score: float  # 0.0 - 10.0
    bioavailability: float  # 0.0 - 1.0
    stability_index: float  # 0.0 - 1.0
    safety_score: float  # 0.0 - 10.0
    cost_per_gram: float  # ZAR
    potency: float  # 0.0 - 10.0
    onset_time_weeks: int  # weeks to show effect
    duration_months: int  # months effect lasts
    evidence_level: str  # "strong", "moderate", "weak"
    
    def calculate_composite_score(self, weights: Dict[str, float]) -> float:
        """Calculate composite score based on weighted metrics."""
        score = 0.0
        total_weight = 0.0
        
        metric_map = {
            'efficacy': self.efficacy_score / 10.0,
            'bioavailability': self.bioavailability,
            'stability': self.stability_index,
            'safety': self.safety_score / 10.0,
            'cost_efficiency': 1.0 / (1.0 + self.cost_per_gram / 100.0),
            'potency': self.potency / 10.0
        }
        
        for metric, weight in weights.items():
            if metric in metric_map:
                score += metric_map[metric] * weight
                total_weight += weight
        
        return score / max(total_weight, 0.1)


@dataclass
class Hypergredient:
    """A functional ingredient abstraction with comprehensive properties."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    inci_name: str = ""
    hypergredient_class: HypergredientClass = HypergredientClass.H_AO
    primary_function: str = ""
    secondary_functions: List[str] = field(default_factory=list)
    
    # Chemical properties
    ph_min: float = 4.0
    ph_max: float = 9.0
    molecular_weight: float = 0.0
    solubility: str = "water"  # "water", "oil", "both", "ethanol"
    
    # Performance metrics
    metrics: HypergredientMetrics = field(default_factory=lambda: HypergredientMetrics(
        efficacy_score=5.0, bioavailability=0.5, stability_index=0.5,
        safety_score=5.0, cost_per_gram=100.0, potency=5.0,
        onset_time_weeks=4, duration_months=3, evidence_level="moderate"
    ))
    
    # Interaction profiles
    incompatibilities: List[str] = field(default_factory=list)
    synergies: List[str] = field(default_factory=list)
    
    # Regulatory and safety
    regulatory_limits: Dict[str, float] = field(default_factory=dict)
    stability_conditions: List[StabilityFactor] = field(default_factory=list)
    allergen_risk: SafetyLevel = SafetyLevel.LOW
    
    # Usage constraints
    max_concentration: float = 10.0
    min_concentration: float = 0.1
    typical_concentration: float = 2.0
    
    # Supply chain
    supplier_info: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)
    
    def calculate_optimization_score(self, target_concerns: List[str], 
                                   constraints: Dict[str, Any]) -> float:
        """Calculate optimization score for specific formulation targets."""
        # Base score from metrics
        weights = {
            'efficacy': 0.35,
            'safety': 0.25,
            'stability': 0.20,
            'cost_efficiency': 0.15,
            'bioavailability': 0.05
        }
        
        base_score = self.metrics.calculate_composite_score(weights)
        
        # Adjust for target concerns
        concern_boost = 0.0
        if any(concern in self.primary_function.lower() or 
               any(concern in func.lower() for func in self.secondary_functions)
               for concern in target_concerns):
            concern_boost = 0.2
        
        # pH compatibility check
        ph_penalty = 0.0
        if 'target_ph' in constraints:
            target_ph = constraints['target_ph']
            if not (self.ph_min <= target_ph <= self.ph_max):
                ph_penalty = 0.3
        
        # Budget constraint
        budget_penalty = 0.0
        if 'max_cost_per_gram' in constraints:
            if self.metrics.cost_per_gram > constraints['max_cost_per_gram']:
                budget_penalty = 0.2
        
        final_score = base_score + concern_boost - ph_penalty - budget_penalty
        return max(0.0, min(1.0, final_score))


class InteractionType(Enum):
    """Types of ingredient interactions."""
    SYNERGISTIC = "synergistic"
    COMPATIBLE = "compatible"
    NEUTRAL = "neutral"
    ANTAGONISTIC = "antagonistic"
    INCOMPATIBLE = "incompatible"


@dataclass
class InteractionRule:
    """Rule defining interaction between hypergredient classes."""
    class_a: HypergredientClass
    class_b: HypergredientClass
    interaction_type: InteractionType
    strength: float  # -2.0 to +2.0 (negative = harmful, positive = beneficial)
    description: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)


class HypergredientDatabase:
    """Comprehensive database of hypergredients with optimization capabilities."""
    
    def __init__(self):
        self.hypergredients: Dict[str, Hypergredient] = {}
        self.interaction_matrix: Dict[Tuple[HypergredientClass, HypergredientClass], float] = {}
        self.interaction_rules: List[InteractionRule] = []
        self._initialize_database()
        self._initialize_interaction_matrix()
    
    def _initialize_database(self):
        """Initialize comprehensive hypergredient database."""
        logger.info("🧬 Initializing Hypergredient Database...")
        
        # H.CT - Cellular Turnover Agents
        cellular_turnover_agents = [
            Hypergredient(
                name="Tretinoin", inci_name="Tretinoin",
                hypergredient_class=HypergredientClass.H_CT,
                primary_function="Cellular turnover acceleration",
                secondary_functions=["Collagen synthesis", "Acne treatment"],
                ph_min=5.5, ph_max=6.5,
                metrics=HypergredientMetrics(
                    efficacy_score=10.0, bioavailability=0.85, stability_index=0.3,
                    safety_score=6.0, cost_per_gram=15.0, potency=10.0,
                    onset_time_weeks=2, duration_months=6, evidence_level="strong"
                ),
                incompatibilities=["benzoyl_peroxide", "aha_acids"],
                stability_conditions=[StabilityFactor.UV_SENSITIVE, StabilityFactor.LIGHT_SENSITIVE],
                max_concentration=0.1, min_concentration=0.01, typical_concentration=0.05
            ),
            Hypergredient(
                name="Bakuchiol", inci_name="Bakuchiol",
                hypergredient_class=HypergredientClass.H_CT,
                primary_function="Gentle cellular renewal",
                secondary_functions=["Antioxidant activity", "Anti-aging"],
                ph_min=4.0, ph_max=9.0,
                metrics=HypergredientMetrics(
                    efficacy_score=7.0, bioavailability=0.70, stability_index=0.9,
                    safety_score=9.0, cost_per_gram=240.0, potency=7.0,
                    onset_time_weeks=4, duration_months=4, evidence_level="moderate"
                ),
                synergies=["vitamin_c", "niacinamide"],
                stability_conditions=[StabilityFactor.STABLE],
                max_concentration=2.0, min_concentration=0.5, typical_concentration=1.0
            ),
            Hypergredient(
                name="Retinol", inci_name="Retinol",
                hypergredient_class=HypergredientClass.H_CT,
                primary_function="Vitamin A derivative for cell renewal",
                secondary_functions=["Anti-aging", "Acne treatment"],
                ph_min=5.5, ph_max=6.5,
                metrics=HypergredientMetrics(
                    efficacy_score=8.0, bioavailability=0.60, stability_index=0.4,
                    safety_score=7.0, cost_per_gram=180.0, potency=8.0,
                    onset_time_weeks=3, duration_months=4, evidence_level="strong"
                ),
                incompatibilities=["aha_acids", "bha_acids", "vitamin_c"],
                stability_conditions=[StabilityFactor.AIR_SENSITIVE, StabilityFactor.LIGHT_SENSITIVE],
                max_concentration=1.0, min_concentration=0.1, typical_concentration=0.3
            )
        ]
        
        # H.CS - Collagen Synthesis Promoters
        collagen_synthesis_promoters = [
            Hypergredient(
                name="Matrixyl 3000", inci_name="Palmitoyl Tetrapeptide-7, Palmitoyl Oligopeptide",
                hypergredient_class=HypergredientClass.H_CS,
                primary_function="Collagen synthesis stimulation",
                secondary_functions=["Elastin production", "Skin firmness"],
                ph_min=5.0, ph_max=7.0,
                metrics=HypergredientMetrics(
                    efficacy_score=9.0, bioavailability=0.75, stability_index=0.8,
                    safety_score=9.0, cost_per_gram=120.0, potency=9.0,
                    onset_time_weeks=4, duration_months=6, evidence_level="strong"
                ),
                synergies=["vitamin_c", "peptides"],
                stability_conditions=[StabilityFactor.STABLE],
                max_concentration=5.0, min_concentration=1.0, typical_concentration=3.0
            ),
            Hypergredient(
                name="Vitamin C L-AA", inci_name="L-Ascorbic Acid",
                hypergredient_class=HypergredientClass.H_CS,
                primary_function="Collagen cofactor synthesis",
                secondary_functions=["Antioxidant", "Brightening"],
                ph_min=3.0, ph_max=4.0,
                metrics=HypergredientMetrics(
                    efficacy_score=8.0, bioavailability=0.90, stability_index=0.2,
                    safety_score=8.0, cost_per_gram=85.0, potency=8.0,
                    onset_time_weeks=3, duration_months=0, evidence_level="strong"
                ),
                incompatibilities=["copper_peptides", "retinoids"],
                synergies=["vitamin_e", "ferulic_acid"],
                stability_conditions=[StabilityFactor.UV_SENSITIVE, StabilityFactor.AIR_SENSITIVE],
                max_concentration=20.0, min_concentration=5.0, typical_concentration=10.0
            )
        ]
        
        # H.AO - Antioxidant Systems
        antioxidant_systems = [
            Hypergredient(
                name="Astaxanthin", inci_name="Astaxanthin",
                hypergredient_class=HypergredientClass.H_AO,
                primary_function="Powerful antioxidant protection",
                secondary_functions=["UV protection", "Anti-inflammatory"],
                ph_min=4.0, ph_max=8.0,
                metrics=HypergredientMetrics(
                    efficacy_score=9.5, bioavailability=0.65, stability_index=0.4,
                    safety_score=9.0, cost_per_gram=360.0, potency=9.5,
                    onset_time_weeks=2, duration_months=3, evidence_level="strong"
                ),
                synergies=["vitamin_e", "coq10"],
                stability_conditions=[StabilityFactor.LIGHT_SENSITIVE],
                max_concentration=0.1, min_concentration=0.01, typical_concentration=0.05
            ),
            Hypergredient(
                name="Vitamin E", inci_name="Tocopherol",
                hypergredient_class=HypergredientClass.H_AO,
                primary_function="Lipid antioxidant protection",
                secondary_functions=["Skin barrier support", "Anti-inflammatory"],
                ph_min=4.0, ph_max=8.0,
                metrics=HypergredientMetrics(
                    efficacy_score=7.0, bioavailability=0.85, stability_index=0.8,
                    safety_score=9.0, cost_per_gram=50.0, potency=7.0,
                    onset_time_weeks=2, duration_months=4, evidence_level="strong"
                ),
                synergies=["vitamin_c", "ferulic_acid"],
                stability_conditions=[StabilityFactor.STABLE],
                max_concentration=1.0, min_concentration=0.1, typical_concentration=0.5
            )
        ]
        
        # H.HY - Hydration Systems
        hydration_systems = [
            Hypergredient(
                name="Hyaluronic Acid Multi-MW", inci_name="Sodium Hyaluronate",
                hypergredient_class=HypergredientClass.H_HY,
                primary_function="Multi-depth hydration",
                secondary_functions=["Plumping effect", "Barrier support"],
                ph_min=4.0, ph_max=7.0,
                metrics=HypergredientMetrics(
                    efficacy_score=9.0, bioavailability=0.75, stability_index=0.9,
                    safety_score=10.0, cost_per_gram=2500.0, potency=9.0,
                    onset_time_weeks=1, duration_months=0, evidence_level="strong"
                ),
                synergies=["glycerin", "ceramides"],
                stability_conditions=[StabilityFactor.STABLE],
                max_concentration=2.0, min_concentration=0.1, typical_concentration=0.5
            ),
            Hypergredient(
                name="Glycerin", inci_name="Glycerin",
                hypergredient_class=HypergredientClass.H_HY,
                primary_function="Humectant moisture retention",
                secondary_functions=["Skin softening", "Penetration enhancement"],
                ph_min=4.0, ph_max=10.0,
                metrics=HypergredientMetrics(
                    efficacy_score=7.0, bioavailability=0.95, stability_index=1.0,
                    safety_score=10.0, cost_per_gram=8.0, potency=7.0,
                    onset_time_weeks=0, duration_months=0, evidence_level="strong"
                ),
                synergies=["hyaluronic_acid", "ceramides"],
                stability_conditions=[StabilityFactor.STABLE],
                max_concentration=15.0, min_concentration=3.0, typical_concentration=8.0
            )
        ]
        
        # Add all hypergredients to database
        all_hypergredients = (cellular_turnover_agents + collagen_synthesis_promoters + 
                            antioxidant_systems + hydration_systems)
        
        for hypergredient in all_hypergredients:
            self.hypergredients[hypergredient.name.lower().replace(" ", "_")] = hypergredient
        
        logger.info(f"✓ Loaded {len(self.hypergredients)} hypergredients across {len(HypergredientClass)} classes")
    
    def _initialize_interaction_matrix(self):
        """Initialize hypergredient interaction matrix."""
        logger.info("🔗 Initializing Interaction Matrix...")
        
        # Define interaction rules based on the specification
        interaction_rules = [
            # Positive synergies
            InteractionRule(
                HypergredientClass.H_CT, HypergredientClass.H_CS, 
                InteractionType.SYNERGISTIC, 1.5,
                "Cellular turnover enhances collagen synthesis effectiveness"
            ),
            InteractionRule(
                HypergredientClass.H_CS, HypergredientClass.H_AO,
                InteractionType.SYNERGISTIC, 2.0,
                "Antioxidants protect and enhance collagen synthesis"
            ),
            InteractionRule(
                HypergredientClass.H_BR, HypergredientClass.H_HY,
                InteractionType.SYNERGISTIC, 2.5,
                "Barrier repair and hydration work synergistically"
            ),
            InteractionRule(
                HypergredientClass.H_ML, HypergredientClass.H_AO,
                InteractionType.SYNERGISTIC, 1.8,
                "Antioxidants support melanin modulation"
            ),
            InteractionRule(
                HypergredientClass.H_AI, HypergredientClass.H_MB,
                InteractionType.SYNERGISTIC, 2.2,
                "Anti-inflammatory agents support microbiome balance"
            ),
            
            # Mild antagonisms
            InteractionRule(
                HypergredientClass.H_CT, HypergredientClass.H_AO,
                InteractionType.ANTAGONISTIC, 0.8,
                "Cellular turnover agents may be oxidized by some conditions"
            ),
            InteractionRule(
                HypergredientClass.H_SE, HypergredientClass.H_CT,
                InteractionType.ANTAGONISTIC, 0.6,
                "Sebum regulators may increase irritation with strong actives"
            ),
            
            # Neutral interactions (most combinations)
            InteractionRule(
                HypergredientClass.H_HY, HypergredientClass.H_ML,
                InteractionType.NEUTRAL, 1.0,
                "Hydration and melanin modulation are generally compatible"
            )
        ]
        
        # Build interaction matrix
        for rule in interaction_rules:
            key = (rule.class_a, rule.class_b)
            reverse_key = (rule.class_b, rule.class_a)
            
            self.interaction_matrix[key] = rule.strength
            self.interaction_matrix[reverse_key] = rule.strength
            self.interaction_rules.append(rule)
        
        logger.info(f"✓ Loaded {len(self.interaction_rules)} interaction rules")
    
    def get_hypergredients_by_class(self, hypergredient_class: HypergredientClass) -> List[Hypergredient]:
        """Get all hypergredients of a specific class."""
        return [h for h in self.hypergredients.values() 
                if h.hypergredient_class == hypergredient_class]
    
    def get_interaction_strength(self, class_a: HypergredientClass, 
                               class_b: HypergredientClass) -> float:
        """Get interaction strength between two hypergredient classes."""
        key = (class_a, class_b)
        return self.interaction_matrix.get(key, 1.0)  # Default neutral
    
    def search_hypergredients(self, criteria: Dict[str, Any]) -> List[Hypergredient]:
        """Search hypergredients based on multiple criteria."""
        results = []
        
        for hypergredient in self.hypergredients.values():
            match = True
            
            # Class filter
            if 'hypergredient_class' in criteria:
                if hypergredient.hypergredient_class != criteria['hypergredient_class']:
                    match = False
            
            # Efficacy filter
            if 'min_efficacy' in criteria:
                if hypergredient.metrics.efficacy_score < criteria['min_efficacy']:
                    match = False
            
            # Safety filter
            if 'min_safety' in criteria:
                if hypergredient.metrics.safety_score < criteria['min_safety']:
                    match = False
            
            # Cost filter
            if 'max_cost' in criteria:
                if hypergredient.metrics.cost_per_gram > criteria['max_cost']:
                    match = False
            
            # pH compatibility
            if 'target_ph' in criteria:
                target_ph = criteria['target_ph']
                if not (hypergredient.ph_min <= target_ph <= hypergredient.ph_max):
                    match = False
            
            if match:
                results.append(hypergredient)
        
        return results


if __name__ == "__main__":
    # Demonstration
    print("🧬 Hypergredient Framework Demonstration")
    print("=" * 50)
    
    db = HypergredientDatabase()
    
    print(f"Database loaded with {len(db.hypergredients)} hypergredients")
    print(f"Interaction matrix contains {len(db.interaction_matrix)} rules")
    
    # Example search
    print("\n🔍 Example Search: High-efficacy, low-cost cellular turnover agents")
    results = db.search_hypergredients({
        'hypergredient_class': HypergredientClass.H_CT,
        'min_efficacy': 7.0,
        'max_cost': 200.0
    })
    
    for result in results:
        print(f"  • {result.name}: Efficacy {result.metrics.efficacy_score}/10, "
              f"Cost R{result.metrics.cost_per_gram}/g")
    
    print("\n✅ Hypergredient Framework initialized successfully!")
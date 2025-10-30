"""
Core Hypergredient Framework Classes

This module defines the core hypergredient classes, taxonomy, and metrics
for the revolutionary formulation design system.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# Core Hypergredient Taxonomy
HYPERGREDIENT_CLASSES = {
    "H.CT": "Cellular Turnover Agents",
    "H.CS": "Collagen Synthesis Promoters", 
    "H.AO": "Antioxidant Systems",
    "H.BR": "Barrier Repair Complex",
    "H.ML": "Melanin Modulators",
    "H.HY": "Hydration Systems",
    "H.AI": "Anti-Inflammatory Agents",
    "H.MB": "Microbiome Balancers",
    "H.SE": "Sebum Regulators",
    "H.PD": "Penetration/Delivery Enhancers"
}


@dataclass
class HypergredientMetrics:
    """Performance metrics for hypergredients"""
    efficacy_score: float  # 0-10 scale
    bioavailability: float  # 0-100% 
    safety_score: float  # 0-10 scale
    stability_index: float  # 0-1 scale
    cost_efficiency: float  # calculated metric
    penetration_score: float = 0.0  # 0-10 scale
    synergy_potential: float = 0.0  # 0-10 scale
    
    def calculate_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted composite performance score"""
        if weights is None:
            weights = {
                'efficacy': 0.35,
                'safety': 0.25,
                'stability': 0.20,
                'cost': 0.15,
                'synergy': 0.05
            }
        
        score = (
            self.efficacy_score * weights.get('efficacy', 0.35) +
            self.safety_score * weights.get('safety', 0.25) +
            self.stability_index * 10 * weights.get('stability', 0.20) +
            (10 - self.cost_efficiency) * weights.get('cost', 0.15) +  # Invert cost
            self.synergy_potential * weights.get('synergy', 0.05)
        )
        return min(10.0, max(0.0, score))


@dataclass
class Hypergredient:
    """
    Hypergredient Definition:
    Hypergredient(*) := {ingredient_i | function(*) ∈ F_i, 
                         constraints ∈ C_i, 
                         performance ∈ P_i}
    """
    name: str
    inci_name: str
    hypergredient_class: str
    primary_function: str
    secondary_functions: List[str] = field(default_factory=list)
    
    # Core Properties
    potency: float = 0.0  # 0-10 scale
    ph_range: Tuple[float, float] = (5.0, 7.0)
    stability: str = "moderate"  # stable, moderate, unstable, uv-sensitive, o2-sensitive
    interactions: Dict[str, str] = field(default_factory=dict)  # ingredient_name: interaction_type
    cost_per_gram: float = 0.0  # ZAR
    bioavailability: float = 50.0  # 0-100%
    safety_score: float = 5.0  # 0-10 scale
    
    # Advanced Properties
    molecular_weight: float = 0.0
    solubility: str = "water"  # water, oil, both
    penetration_enhancer: bool = False
    regulatory_status: Dict[str, bool] = field(default_factory=dict)
    clinical_evidence: str = "moderate"  # strong, moderate, limited, none
    
    # Performance Metrics
    metrics: Optional[HypergredientMetrics] = None
    
    def __post_init__(self):
        """Initialize metrics if not provided"""
        if self.metrics is None:
            self.metrics = HypergredientMetrics(
                efficacy_score=self.potency,
                bioavailability=self.bioavailability,
                safety_score=self.safety_score,
                stability_index=self._calculate_stability_index(),
                cost_efficiency=self._calculate_cost_efficiency()
            )
    
    def _calculate_stability_index(self) -> float:
        """Convert stability string to numeric index"""
        stability_map = {
            "very_stable": 1.0,
            "stable": 0.8,
            "moderate": 0.6,
            "unstable": 0.4,
            "uv-sensitive": 0.3,
            "o2-sensitive": 0.2
        }
        return stability_map.get(self.stability.lower(), 0.5)
    
    def _calculate_cost_efficiency(self) -> float:
        """Calculate cost efficiency score (0-10, lower is better)"""
        if self.cost_per_gram <= 0:
            return 5.0
        
        # Logarithmic scale for cost efficiency
        # High-end ingredients (R500+) = 8-10, Budget ingredients (R1-50) = 1-3
        cost_score = np.log10(max(1, self.cost_per_gram)) * 2
        return min(10.0, max(1.0, cost_score))
    
    def check_compatibility(self, other_ingredient_name: str) -> str:
        """Check compatibility with another ingredient"""
        return self.interactions.get(other_ingredient_name, "unknown")
    
    def calculate_formulation_score(self, context: Dict[str, Any]) -> float:
        """Calculate score in formulation context"""
        base_score = self.metrics.calculate_composite_score()
        
        # Apply context-specific modifiers
        ph_penalty = self._calculate_ph_penalty(context.get('target_ph', 6.0))
        budget_penalty = self._calculate_budget_penalty(context.get('budget', 1000))
        
        final_score = base_score * (1 - ph_penalty) * (1 - budget_penalty)
        return max(0.0, min(10.0, final_score))
    
    def _calculate_ph_penalty(self, target_ph: float) -> float:
        """Calculate pH compatibility penalty"""
        ph_min, ph_max = self.ph_range
        if ph_min <= target_ph <= ph_max:
            return 0.0
        
        # Distance from optimal range
        if target_ph < ph_min:
            distance = ph_min - target_ph
        else:
            distance = target_ph - ph_max
            
        # Penalty increases with distance
        penalty = min(0.5, distance * 0.1)
        return penalty
    
    def _calculate_budget_penalty(self, budget: float) -> float:
        """Calculate budget constraint penalty"""
        if self.cost_per_gram * 100 <= budget:  # Assuming 100g formulation
            return 0.0
        
        over_budget_ratio = (self.cost_per_gram * 100) / budget - 1
        penalty = min(0.8, over_budget_ratio * 0.3)
        return penalty
    
    def predict_performance(self, formulation_context: Dict[str, Any]) -> Dict[str, float]:
        """Predict performance in specific formulation context"""
        base_performance = {
            'efficacy': self.metrics.efficacy_score,
            'safety': self.metrics.safety_score,
            'stability': self.metrics.stability_index * 10,
            'cost_effectiveness': 10 - self.metrics.cost_efficiency
        }
        
        # Apply synergistic effects if other ingredients present
        other_ingredients = formulation_context.get('other_ingredients', [])
        synergy_boost = self._calculate_synergy_boost(other_ingredients)
        
        for key in base_performance:
            if key != 'cost_effectiveness':
                base_performance[key] *= (1 + synergy_boost)
                base_performance[key] = min(10.0, base_performance[key])
        
        return base_performance
    
    def _calculate_synergy_boost(self, other_ingredients: List[str]) -> float:
        """Calculate synergy boost from other ingredients"""
        synergy_boost = 0.0
        
        for ingredient in other_ingredients:
            interaction = self.check_compatibility(ingredient)
            if interaction == "synergy":
                synergy_boost += 0.1
            elif interaction == "strong_synergy":
                synergy_boost += 0.2
        
        return min(0.5, synergy_boost)  # Cap at 50% boost


class HypergredientDatabase:
    """Database for managing hypergredients"""
    
    def __init__(self):
        self.hypergredients: Dict[str, Hypergredient] = {}
        self.class_index: Dict[str, List[str]] = {class_code: [] for class_code in HYPERGREDIENT_CLASSES}
        
    def add_hypergredient(self, hypergredient: Hypergredient):
        """Add hypergredient to database"""
        self.hypergredients[hypergredient.name] = hypergredient
        
        if hypergredient.hypergredient_class in self.class_index:
            self.class_index[hypergredient.hypergredient_class].append(hypergredient.name)
    
    def get_by_class(self, hypergredient_class: str) -> List[Hypergredient]:
        """Get all hypergredients of a specific class"""
        ingredient_names = self.class_index.get(hypergredient_class, [])
        return [self.hypergredients[name] for name in ingredient_names if name in self.hypergredients]
    
    def get_by_function(self, function: str) -> List[Hypergredient]:
        """Get hypergredients by primary or secondary function"""
        results = []
        for hypergredient in self.hypergredients.values():
            if (function.lower() in hypergredient.primary_function.lower() or
                any(function.lower() in func.lower() for func in hypergredient.secondary_functions)):
                results.append(hypergredient)
        return results
    
    def search(self, criteria: Dict[str, Any]) -> List[Hypergredient]:
        """Search hypergredients by multiple criteria"""
        results = list(self.hypergredients.values())
        
        if 'class' in criteria:
            results = [h for h in results if h.hypergredient_class == criteria['class']]
        
        if 'min_potency' in criteria:
            results = [h for h in results if h.potency >= criteria['min_potency']]
        
        if 'max_cost' in criteria:
            results = [h for h in results if h.cost_per_gram <= criteria['max_cost']]
        
        if 'ph_range' in criteria:
            target_ph = criteria['ph_range']
            results = [h for h in results if h.ph_range[0] <= target_ph <= h.ph_range[1]]
        
        if 'safety_min' in criteria:
            results = [h for h in results if h.safety_score >= criteria['safety_min']]
        
        return results
    
    def get_top_performers(self, hypergredient_class: str, n: int = 5, 
                          metric: str = 'composite') -> List[Hypergredient]:
        """Get top performing hypergredients in a class"""
        candidates = self.get_by_class(hypergredient_class)
        
        if metric == 'composite':
            candidates.sort(key=lambda h: h.metrics.calculate_composite_score(), reverse=True)
        elif metric == 'efficacy':
            candidates.sort(key=lambda h: h.metrics.efficacy_score, reverse=True)
        elif metric == 'safety':
            candidates.sort(key=lambda h: h.metrics.safety_score, reverse=True)
        elif metric == 'cost':
            candidates.sort(key=lambda h: h.metrics.cost_efficiency)
        
        return candidates[:n]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            'total_hypergredients': len(self.hypergredients),
            'by_class': {class_code: len(ingredients) 
                        for class_code, ingredients in self.class_index.items()},
            'avg_potency': np.mean([h.potency for h in self.hypergredients.values()]),
            'avg_safety': np.mean([h.safety_score for h in self.hypergredients.values()]),
            'cost_range': (
                min(h.cost_per_gram for h in self.hypergredients.values() if h.cost_per_gram > 0),
                max(h.cost_per_gram for h in self.hypergredients.values())
            ) if self.hypergredients else (0, 0)
        }
        return stats
"""
Dynamic Scoring System

This module implements the dynamic scoring and performance metrics
system for hypergredients and formulations.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from abc import ABC, abstractmethod

from .core import Hypergredient, HypergredientMetrics


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for hypergredients and formulations"""
    
    # Core Performance Metrics
    efficacy_score: float = 0.0  # 0-10 scale
    safety_score: float = 0.0    # 0-10 scale
    stability_score: float = 0.0  # 0-10 scale
    cost_efficiency: float = 0.0  # 0-10 scale (higher = more efficient)
    
    # Advanced Metrics
    bioavailability: float = 0.0  # 0-100%
    penetration_score: float = 0.0  # 0-10 scale
    synergy_potential: float = 0.0  # 0-10 scale
    sustainability_score: float = 0.0  # 0-10 scale
    
    # Formulation-specific Metrics
    ph_stability: float = 0.0  # 0-10 scale
    texture_score: float = 0.0  # 0-10 scale
    user_experience: float = 0.0  # 0-10 scale
    
    # Regulatory and Market Metrics
    regulatory_compliance: float = 0.0  # 0-10 scale
    market_acceptance: float = 0.0  # 0-10 scale
    innovation_score: float = 0.0  # 0-10 scale
    
    def calculate_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        total_score = 0.0
        total_weight = 0.0
        
        metric_values = {
            'efficacy': self.efficacy_score,
            'safety': self.safety_score,
            'stability': self.stability_score,
            'cost': self.cost_efficiency,
            'bioavailability': self.bioavailability / 10,  # Normalize to 0-10
            'penetration': self.penetration_score,
            'synergy': self.synergy_potential,
            'sustainability': self.sustainability_score,
            'ph_stability': self.ph_stability,
            'texture': self.texture_score,
            'user_experience': self.user_experience,
            'regulatory': self.regulatory_compliance,
            'market': self.market_acceptance,
            'innovation': self.innovation_score
        }
        
        for metric, weight in weights.items():
            if metric in metric_values:
                total_score += metric_values[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def get_performance_radar(self) -> Dict[str, float]:
        """Get data for radar chart visualization"""
        return {
            'Efficacy': self.efficacy_score,
            'Safety': self.safety_score,
            'Stability': self.stability_score,
            'Cost Efficiency': self.cost_efficiency,
            'Bioavailability': self.bioavailability / 10,
            'Synergy': self.synergy_potential
        }


class MetricCalculator(ABC):
    """Abstract base class for metric calculators"""
    
    @abstractmethod
    def calculate(self, data: Any) -> float:
        """Calculate metric score"""
        pass


class EfficacyCalculator(MetricCalculator):
    """Calculate efficacy scores based on multiple factors"""
    
    def __init__(self):
        self.clinical_evidence_weights = {
            'strong': 1.0,
            'moderate': 0.8,
            'limited': 0.5,
            'none': 0.2
        }
    
    def calculate(self, hypergredient: Hypergredient) -> float:
        """Calculate efficacy score for hypergredient"""
        base_potency = hypergredient.potency
        
        # Apply clinical evidence modifier
        evidence_modifier = self.clinical_evidence_weights.get(
            hypergredient.clinical_evidence, 0.5
        )
        
        # Apply bioavailability modifier
        bioavailability_modifier = hypergredient.bioavailability / 100
        
        # Calculate final efficacy
        efficacy_score = base_potency * evidence_modifier * bioavailability_modifier
        
        return min(10.0, max(0.0, efficacy_score))


class SafetyCalculator(MetricCalculator):
    """Calculate safety scores with multiple considerations"""
    
    def __init__(self):
        self.allergen_penalty = 2.0
        self.irritation_risk_map = {
            'H.CT': 0.3,  # Cellular turnover can be irritating
            'H.SE': 0.2,  # Sebum regulators can be drying
            'H.ML': 0.1,  # Melanin modulators are generally gentler
            'H.HY': 0.0,  # Hydration is very safe
            'H.BR': 0.0,  # Barrier repair is safe
            'H.AO': 0.05, # Antioxidants are generally safe
            'H.CS': 0.05, # Collagen synthesis is safe
            'H.AI': 0.0,  # Anti-inflammatory is safe
            'H.MB': 0.0,  # Microbiome balancers are safe
            'H.PD': 0.1   # Penetration enhancers can irritate
        }
    
    def calculate(self, hypergredient: Hypergredient) -> float:
        """Calculate safety score for hypergredient"""
        base_safety = hypergredient.safety_score
        
        # Apply allergen penalty
        if hasattr(hypergredient, 'allergen_status') and hypergredient.allergen_status:
            base_safety -= self.allergen_penalty
        
        # Apply class-based irritation risk
        irritation_risk = self.irritation_risk_map.get(
            hypergredient.hypergredient_class, 0.1
        )
        
        # Higher potency can increase irritation risk
        potency_risk = (hypergredient.potency - 5.0) * 0.1 if hypergredient.potency > 5.0 else 0.0
        
        final_safety = base_safety - irritation_risk - potency_risk
        
        return min(10.0, max(0.0, final_safety))


class StabilityCalculator(MetricCalculator):
    """Calculate stability scores considering multiple factors"""
    
    def __init__(self):
        self.stability_scores = {
            'very_stable': 10.0,
            'stable': 8.0,
            'moderate': 6.0,
            'unstable': 4.0,
            'uv-sensitive': 3.0,
            'o2-sensitive': 3.5,
            'light-sensitive': 3.0,
            'ph-dependent': 5.0
        }
    
    def calculate(self, hypergredient: Hypergredient) -> float:
        """Calculate stability score for hypergredient"""
        base_stability = self.stability_scores.get(hypergredient.stability, 5.0)
        
        # Apply temperature sensitivity
        if hasattr(hypergredient, 'stability_temperature'):
            if hypergredient.stability_temperature < 25.0:
                base_stability -= 1.0
            elif hypergredient.stability_temperature > 40.0:
                base_stability += 0.5
        
        # pH range impact on stability
        ph_min, ph_max = hypergredient.ph_range
        ph_range_width = ph_max - ph_min
        
        if ph_range_width < 1.0:  # Very narrow pH range = less stable
            base_stability -= 1.0
        elif ph_range_width > 4.0:  # Wide pH range = more stable
            base_stability += 0.5
        
        return min(10.0, max(0.0, base_stability))


class CostEfficiencyCalculator(MetricCalculator):
    """Calculate cost efficiency considering performance vs cost"""
    
    def calculate(self, hypergredient: Hypergredient) -> float:
        """Calculate cost efficiency score"""
        if hypergredient.cost_per_gram <= 0:
            return 5.0
        
        # Calculate performance per unit cost
        performance_score = (
            hypergredient.potency * 0.4 +
            hypergredient.safety_score * 0.3 +
            hypergredient.bioavailability / 10 * 0.3
        )
        
        # Logarithmic cost scaling
        cost_factor = np.log10(max(1.0, hypergredient.cost_per_gram))
        
        # Higher performance per cost = higher efficiency
        efficiency = performance_score / cost_factor
        
        # Normalize to 0-10 scale
        normalized_efficiency = min(10.0, max(0.0, efficiency * 1.5))
        
        return normalized_efficiency


class SynergyCalculator(MetricCalculator):
    """Calculate synergy potential of hypergredients"""
    
    def __init__(self):
        # Map hypergredient classes to their synergy potential
        self.synergy_potential_map = {
            'H.AO': 9.0,  # Antioxidants have high synergy potential
            'H.CS': 8.0,  # Collagen synthesis works well with others
            'H.BR': 8.5,  # Barrier repair synergizes well
            'H.HY': 8.0,  # Hydration supports everything
            'H.AI': 7.5,  # Anti-inflammatory is supportive
            'H.CT': 6.0,  # Cell turnover can conflict
            'H.ML': 7.0,  # Melanin modulators are moderately synergistic
            'H.SE': 5.5,  # Sebum regulators can conflict
            'H.MB': 8.0,  # Microbiome balancers are supportive
            'H.PD': 7.5   # Penetration enhancers help others
        }
    
    def calculate(self, hypergredient: Hypergredient) -> float:
        """Calculate synergy potential score"""
        base_synergy = self.synergy_potential_map.get(
            hypergredient.hypergredient_class, 6.0
        )
        
        # Higher safety ingredients tend to have better synergy
        safety_bonus = (hypergredient.safety_score - 5.0) * 0.2
        
        # Stable ingredients synergize better
        stability_bonus = hypergredient._calculate_stability_index() * 1.0
        
        final_synergy = base_synergy + safety_bonus + stability_bonus
        
        return min(10.0, max(0.0, final_synergy))


class DynamicScoringSystem:
    """Dynamic scoring system for hypergredients and formulations"""
    
    def __init__(self):
        self.efficacy_calc = EfficacyCalculator()
        self.safety_calc = SafetyCalculator()
        self.stability_calc = StabilityCalculator()
        self.cost_calc = CostEfficiencyCalculator()
        self.synergy_calc = SynergyCalculator()
        
        # Feedback data for ML model
        self.feedback_data = []
        
        # Default scoring weights
        self.default_weights = {
            'efficacy': 0.25,
            'safety': 0.20,
            'stability': 0.15,
            'cost': 0.15,
            'bioavailability': 0.10,
            'synergy': 0.10,
            'sustainability': 0.05
        }
    
    def calculate_hypergredient_metrics(self, hypergredient: Hypergredient) -> PerformanceMetrics:
        """Calculate comprehensive metrics for a hypergredient"""
        metrics = PerformanceMetrics()
        
        # Core metrics
        metrics.efficacy_score = self.efficacy_calc.calculate(hypergredient)
        metrics.safety_score = self.safety_calc.calculate(hypergredient)
        metrics.stability_score = self.stability_calc.calculate(hypergredient)
        metrics.cost_efficiency = self.cost_calc.calculate(hypergredient)
        
        # Advanced metrics
        metrics.bioavailability = hypergredient.bioavailability
        metrics.synergy_potential = self.synergy_calc.calculate(hypergredient)
        
        # Calculated metrics
        metrics.penetration_score = self._calculate_penetration_score(hypergredient)
        metrics.sustainability_score = self._calculate_sustainability_score(hypergredient)
        metrics.ph_stability = self._calculate_ph_stability(hypergredient)
        metrics.regulatory_compliance = self._calculate_regulatory_compliance(hypergredient)
        
        return metrics
    
    def calculate_formulation_metrics(self, hypergredients: List[Hypergredient],
                                    concentrations: Dict[str, float]) -> PerformanceMetrics:
        """Calculate metrics for complete formulation"""
        if not hypergredients:
            return PerformanceMetrics()
        
        # Calculate weighted average of individual metrics
        total_concentration = sum(concentrations.values())
        metrics = PerformanceMetrics()
        
        for hypergredient in hypergredients:
            conc = concentrations.get(hypergredient.name, 0.0)
            weight = conc / total_concentration if total_concentration > 0 else 1.0 / len(hypergredients)
            
            h_metrics = self.calculate_hypergredient_metrics(hypergredient)
            
            # Weighted contribution to formulation metrics
            metrics.efficacy_score += h_metrics.efficacy_score * weight
            metrics.safety_score += h_metrics.safety_score * weight
            metrics.stability_score += h_metrics.stability_score * weight
            metrics.cost_efficiency += h_metrics.cost_efficiency * weight
            metrics.bioavailability += h_metrics.bioavailability * weight
            metrics.synergy_potential += h_metrics.synergy_potential * weight
        
        # Apply formulation-specific adjustments
        metrics = self._apply_formulation_adjustments(metrics, hypergredients, concentrations)
        
        return metrics
    
    def _calculate_penetration_score(self, hypergredient: Hypergredient) -> float:
        """Calculate penetration score based on molecular properties"""
        if hasattr(hypergredient, 'molecular_weight'):
            mw = hypergredient.molecular_weight
            
            # Optimal molecular weight for skin penetration is 500-1000 Da
            if 500 <= mw <= 1000:
                base_score = 9.0
            elif 200 <= mw < 500:
                base_score = 8.0
            elif 1000 < mw <= 5000:
                base_score = 6.0
            elif mw > 5000:
                base_score = 3.0
            else:  # < 200 Da
                base_score = 7.0
        else:
            base_score = 5.0
        
        # Penetration enhancer bonus
        if hypergredient.hypergredient_class == 'H.PD':
            base_score += 2.0
        
        # Lipophilicity considerations
        if hasattr(hypergredient, 'solubility'):
            if hypergredient.solubility == 'both':
                base_score += 1.0
            elif hypergredient.solubility == 'oil':
                base_score += 0.5
        
        return min(10.0, max(0.0, base_score))
    
    def _calculate_sustainability_score(self, hypergredient: Hypergredient) -> float:
        """Calculate sustainability score (placeholder for future implementation)"""
        # This would be based on:
        # - Source sustainability (renewable vs non-renewable)
        # - Manufacturing environmental impact
        # - Biodegradability
        # - Packaging considerations
        
        # For now, return a moderate score
        base_score = 6.0
        
        # Natural/plant-derived ingredients get bonus
        natural_keywords = ['centella', 'green_tea', 'bakuchiol']
        if any(keyword in hypergredient.name.lower() for keyword in natural_keywords):
            base_score += 2.0
        
        return min(10.0, max(0.0, base_score))
    
    def _calculate_ph_stability(self, hypergredient: Hypergredient) -> float:
        """Calculate pH stability score"""
        ph_min, ph_max = hypergredient.ph_range
        ph_range_width = ph_max - ph_min
        
        # Wider pH range = more stable
        if ph_range_width >= 4.0:
            return 9.0
        elif ph_range_width >= 2.0:
            return 7.0
        elif ph_range_width >= 1.0:
            return 5.0
        else:
            return 3.0
    
    def _calculate_regulatory_compliance(self, hypergredient: Hypergredient) -> float:
        """Calculate regulatory compliance score"""
        # This would check against various regulatory databases
        # For now, return high compliance for common ingredients
        
        common_safe_ingredients = [
            'niacinamide', 'hyaluronic_acid', 'glycerin', 'vitamin_e',
            'ceramide', 'allantoin', 'centella_asiatica'
        ]
        
        if hypergredient.name.lower() in common_safe_ingredients:
            return 9.0
        elif hypergredient.safety_score >= 8.0:
            return 8.0
        else:
            return 6.0
    
    def _apply_formulation_adjustments(self, metrics: PerformanceMetrics,
                                     hypergredients: List[Hypergredient],
                                     concentrations: Dict[str, float]) -> PerformanceMetrics:
        """Apply formulation-specific adjustments to metrics"""
        
        # Synergy adjustments
        from .interaction import InteractionMatrix
        interaction_matrix = InteractionMatrix()
        interaction_analysis = interaction_matrix.analyze_formulation_interactions(hypergredients)
        
        # Boost efficacy for synergistic combinations
        synergy_boost = len(interaction_analysis['synergistic_pairs']) * 0.3
        metrics.efficacy_score = min(10.0, metrics.efficacy_score + synergy_boost)
        
        # Reduce safety for antagonistic combinations
        safety_penalty = len(interaction_analysis['antagonistic_pairs']) * 0.5
        metrics.safety_score = max(0.0, metrics.safety_score - safety_penalty)
        
        # Total concentration effects
        total_actives = sum(concentrations.values())
        if total_actives > 20.0:  # High total active concentration
            metrics.safety_score *= 0.9  # Slightly reduce safety
            metrics.stability_score *= 0.9  # Reduce stability
        elif total_actives < 5.0:  # Very low active concentration
            metrics.efficacy_score *= 0.8  # Reduce efficacy
        
        return metrics
    
    def update_from_feedback(self, hypergredient_name: str, 
                           actual_performance: Dict[str, float]):
        """Update scoring model from real-world feedback"""
        feedback_entry = {
            'hypergredient': hypergredient_name,
            'actual_performance': actual_performance,
            'timestamp': np.datetime64('now')
        }
        
        self.feedback_data.append(feedback_entry)
        
        # Simple learning: adjust weights based on feedback
        # In a full implementation, this would use ML models
        if len(self.feedback_data) > 10:
            self._adjust_weights_from_feedback()
    
    def _adjust_weights_from_feedback(self):
        """Adjust scoring weights based on accumulated feedback"""
        # Placeholder for ML-based weight adjustment
        # This would analyze feedback data to improve predictions
        pass
    
    def generate_improvement_suggestions(self, current_metrics: PerformanceMetrics,
                                       target_metrics: PerformanceMetrics) -> List[str]:
        """Generate suggestions for improving formulation metrics"""
        suggestions = []
        
        if current_metrics.efficacy_score < target_metrics.efficacy_score:
            suggestions.append(
                "Consider adding higher potency ingredients or increasing concentrations"
            )
        
        if current_metrics.safety_score < target_metrics.safety_score:
            suggestions.append(
                "Replace high-risk ingredients with safer alternatives"
            )
        
        if current_metrics.stability_score < target_metrics.stability_score:
            suggestions.append(
                "Add stabilizing ingredients or adjust pH range"
            )
        
        if current_metrics.cost_efficiency < target_metrics.cost_efficiency:
            suggestions.append(
                "Consider more cost-effective ingredient alternatives"
            )
        
        if current_metrics.synergy_potential < target_metrics.synergy_potential:
            suggestions.append(
                "Add ingredients with high synergy potential"
            )
        
        return suggestions
    
    def benchmark_against_market(self, metrics: PerformanceMetrics,
                               category: str = "anti_aging") -> Dict[str, str]:
        """Compare metrics against market benchmarks"""
        # Market benchmark data (would be from real market analysis)
        benchmarks = {
            'anti_aging': {
                'efficacy_score': 7.5,
                'safety_score': 8.0,
                'stability_score': 7.0,
                'cost_efficiency': 6.0
            },
            'hydration': {
                'efficacy_score': 8.0,
                'safety_score': 9.0,
                'stability_score': 8.5,
                'cost_efficiency': 7.0
            }
        }
        
        benchmark = benchmarks.get(category, benchmarks['anti_aging'])
        comparison = {}
        
        for metric, benchmark_value in benchmark.items():
            current_value = getattr(metrics, metric, 0.0)
            
            if current_value >= benchmark_value + 1.0:
                comparison[metric] = "Excellent"
            elif current_value >= benchmark_value:
                comparison[metric] = "Above Average"
            elif current_value >= benchmark_value - 1.0:
                comparison[metric] = "Average"
            else:
                comparison[metric] = "Below Average"
        
        return comparison
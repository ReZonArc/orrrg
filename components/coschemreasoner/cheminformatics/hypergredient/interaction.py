"""
Hypergredient Interaction Matrix System

This module implements the interaction scoring and synergy calculation
system for hypergredient combinations.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from .core import Hypergredient


# Hypergredient Interaction Matrix
# Values represent interaction coefficients: 
# > 1.0 = Synergy, < 1.0 = Antagonism, 1.0 = Neutral
INTERACTION_MATRIX = {
    ("H.CT", "H.CS"): 1.5,  # Positive synergy - turnover + collagen
    ("H.CT", "H.AO"): 0.8,  # Mild antagonism - oxidation concerns
    ("H.CT", "H.BR"): 1.2,  # Mild synergy - turnover supports barrier
    ("H.CT", "H.ML"): 1.3,  # Good synergy - exfoliation + brightening
    ("H.CT", "H.HY"): 1.1,  # Mild synergy - turnover can be drying
    ("H.CT", "H.AI"): 0.9,  # Mild antagonism - turnover can irritate
    
    ("H.CS", "H.AO"): 2.0,  # Strong synergy - collagen + antioxidants
    ("H.CS", "H.BR"): 1.6,  # Good synergy - collagen + barrier
    ("H.CS", "H.ML"): 1.4,  # Moderate synergy
    ("H.CS", "H.HY"): 1.7,  # Good synergy - collagen needs hydration
    ("H.CS", "H.AI"): 1.5,  # Good synergy - reduces inflammation
    
    ("H.AO", "H.BR"): 1.8,  # Good synergy - protection + repair
    ("H.AO", "H.ML"): 1.8,  # Good synergy - protection + brightening
    ("H.AO", "H.HY"): 1.4,  # Moderate synergy
    ("H.AO", "H.AI"): 2.2,  # Strong synergy - both fight damage
    ("H.AO", "H.MB"): 1.6,  # Good synergy - antioxidants support microbiome
    
    ("H.BR", "H.HY"): 2.5,  # Excellent synergy - barrier + hydration
    ("H.BR", "H.ML"): 1.3,  # Moderate synergy
    ("H.BR", "H.AI"): 1.8,  # Good synergy - barrier + soothing
    ("H.BR", "H.MB"): 2.0,  # Strong synergy - barrier + microbiome
    
    ("H.ML", "H.AO"): 1.8,  # Good synergy - brightening + protection
    ("H.ML", "H.HY"): 1.2,  # Mild synergy
    ("H.ML", "H.AI"): 1.6,  # Good synergy - brightening can irritate
    
    ("H.HY", "H.AI"): 1.4,  # Moderate synergy - hydration soothes
    ("H.HY", "H.MB"): 1.7,  # Good synergy - hydration supports microbiome
    ("H.HY", "H.SE"): 0.7,  # Mild antagonism - hydration vs oil control
    ("H.HY", "H.PD"): 1.8,  # Good synergy - hydration + delivery
    
    ("H.AI", "H.MB"): 2.2,  # Strong synergy - anti-inflammatory + microbiome
    ("H.AI", "H.SE"): 1.5,  # Good synergy - soothing + sebum control
    
    ("H.MB", "H.SE"): 1.4,  # Moderate synergy - microbiome + sebum
    
    ("H.SE", "H.CT"): 0.6,  # Potential irritation - both can be harsh
    ("H.SE", "H.PD"): 1.3,  # Moderate synergy - delivery of actives
    
    ("H.PD", "H.CT"): 1.4,  # Good synergy - enhances active delivery
    ("H.PD", "H.CS"): 1.5,  # Good synergy - enhances peptide delivery
    ("H.PD", "H.AO"): 1.3,  # Moderate synergy - enhances antioxidant delivery
    ("H.PD", "H.ML"): 1.6,  # Good synergy - enhances brightening delivery
}


class InteractionMatrix:
    """Manages hypergredient interaction scoring and analysis"""
    
    def __init__(self):
        self.matrix = INTERACTION_MATRIX.copy()
        self.ingredient_interactions = {}  # Specific ingredient overrides
    
    def get_class_interaction(self, class1: str, class2: str) -> float:
        """Get interaction coefficient between two hypergredient classes"""
        # Try both directions
        key = (class1, class2)
        reverse_key = (class2, class1)
        
        return self.matrix.get(key, self.matrix.get(reverse_key, 1.0))
    
    def get_ingredient_interaction(self, ingredient1: str, ingredient2: str) -> float:
        """Get specific ingredient interaction (overrides class interaction)"""
        key = (ingredient1, ingredient2)
        reverse_key = (ingredient2, ingredient1)
        
        return self.ingredient_interactions.get(
            key, self.ingredient_interactions.get(reverse_key, None)
        )
    
    def add_ingredient_interaction(self, ingredient1: str, ingredient2: str, 
                                 coefficient: float):
        """Add or update specific ingredient interaction"""
        self.ingredient_interactions[(ingredient1, ingredient2)] = coefficient
    
    def calculate_interaction_score(self, hypergredient1: Hypergredient, 
                                  hypergredient2: Hypergredient) -> float:
        """Calculate interaction score between two hypergredients"""
        # Check for specific ingredient interaction first
        specific_interaction = self.get_ingredient_interaction(
            hypergredient1.name, hypergredient2.name
        )
        
        if specific_interaction is not None:
            return specific_interaction
        
        # Use class-based interaction
        class_interaction = self.get_class_interaction(
            hypergredient1.hypergredient_class,
            hypergredient2.hypergredient_class
        )
        
        # Apply additional modifiers
        ph_modifier = self._calculate_ph_compatibility_modifier(
            hypergredient1, hypergredient2
        )
        
        stability_modifier = self._calculate_stability_modifier(
            hypergredient1, hypergredient2
        )
        
        final_score = class_interaction * ph_modifier * stability_modifier
        return max(0.1, min(3.0, final_score))  # Clamp to reasonable range
    
    def _calculate_ph_compatibility_modifier(self, h1: Hypergredient, 
                                           h2: Hypergredient) -> float:
        """Calculate pH compatibility modifier"""
        ph1_min, ph1_max = h1.ph_range
        ph2_min, ph2_max = h2.ph_range
        
        # Check for overlap
        overlap_min = max(ph1_min, ph2_min)
        overlap_max = min(ph1_max, ph2_max)
        
        if overlap_max <= overlap_min:
            # No overlap - major incompatibility
            return 0.3
        
        # Calculate overlap percentage
        h1_range = ph1_max - ph1_min
        h2_range = ph2_max - ph2_min
        overlap_range = overlap_max - overlap_min
        
        avg_range = (h1_range + h2_range) / 2
        overlap_ratio = overlap_range / avg_range if avg_range > 0 else 1.0
        
        # Strong overlap = no penalty, weak overlap = some penalty
        modifier = 0.7 + (0.3 * overlap_ratio)
        return max(0.3, min(1.0, modifier))
    
    def _calculate_stability_modifier(self, h1: Hypergredient, 
                                    h2: Hypergredient) -> float:
        """Calculate stability interaction modifier"""
        stability_conflicts = [
            ("uv-sensitive", "light-sensitive"),
            ("o2-sensitive", "unstable"),
            ("unstable", "unstable")
        ]
        
        for conflict in stability_conflicts:
            if (h1.stability in conflict and h2.stability in conflict):
                return 0.8  # Reduced interaction due to stability issues
        
        return 1.0
    
    def analyze_formulation_interactions(self, hypergredients: List[Hypergredient]) -> Dict:
        """Analyze all interactions in a formulation"""
        if len(hypergredients) < 2:
            return {
                'total_score': 10.0,
                'average_interaction': 1.0,
                'synergistic_pairs': [],
                'antagonistic_pairs': [],
                'neutral_pairs': []
            }
        
        interactions = []
        synergistic_pairs = []
        antagonistic_pairs = []
        neutral_pairs = []
        
        for i in range(len(hypergredients)):
            for j in range(i + 1, len(hypergredients)):
                h1, h2 = hypergredients[i], hypergredients[j]
                score = self.calculate_interaction_score(h1, h2)
                interactions.append(score)
                
                pair_info = {
                    'ingredient1': h1.name,
                    'ingredient2': h2.name,
                    'score': score
                }
                
                if score > 1.3:
                    synergistic_pairs.append(pair_info)
                elif score < 0.8:
                    antagonistic_pairs.append(pair_info)
                else:
                    neutral_pairs.append(pair_info)
        
        avg_interaction = np.mean(interactions) if interactions else 1.0
        
        # Calculate total formulation score
        # Start with base score and apply interaction effects
        base_score = 7.0
        synergy_bonus = len(synergistic_pairs) * 0.5
        antagonism_penalty = len(antagonistic_pairs) * 0.8
        
        total_score = base_score + synergy_bonus - antagonism_penalty
        total_score = max(0.0, min(10.0, total_score))
        
        return {
            'total_score': total_score,
            'average_interaction': avg_interaction,
            'synergistic_pairs': synergistic_pairs,
            'antagonistic_pairs': antagonistic_pairs,
            'neutral_pairs': neutral_pairs,
            'interaction_matrix': interactions
        }
    
    def suggest_complementary_hypergredients(self, base_hypergredient: Hypergredient,
                                           available_hypergredients: List[Hypergredient],
                                           top_n: int = 5) -> List[Tuple[Hypergredient, float]]:
        """Suggest hypergredients that work well with the base ingredient"""
        scored_suggestions = []
        
        for candidate in available_hypergredients:
            if candidate.name == base_hypergredient.name:
                continue
                
            interaction_score = self.calculate_interaction_score(
                base_hypergredient, candidate
            )
            
            # Consider both interaction and individual performance
            combined_score = (
                interaction_score * 0.6 + 
                candidate.metrics.calculate_composite_score() / 10 * 0.4
            )
            
            scored_suggestions.append((candidate, combined_score))
        
        # Sort by combined score and return top N
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        return scored_suggestions[:top_n]
    
    def get_interaction_warnings(self, hypergredients: List[Hypergredient]) -> List[Dict]:
        """Get warnings about problematic interactions"""
        warnings = []
        
        for i in range(len(hypergredients)):
            for j in range(i + 1, len(hypergredients)):
                h1, h2 = hypergredients[i], hypergredients[j]
                score = self.calculate_interaction_score(h1, h2)
                
                if score < 0.7:
                    warnings.append({
                        'type': 'strong_antagonism',
                        'ingredient1': h1.name,
                        'ingredient2': h2.name,
                        'score': score,
                        'message': f"Strong negative interaction between {h1.name} and {h2.name}",
                        'recommendation': "Consider using these ingredients in separate products or at different times"
                    })
                elif score < 0.9:
                    warnings.append({
                        'type': 'mild_antagonism',
                        'ingredient1': h1.name,
                        'ingredient2': h2.name,
                        'score': score,
                        'message': f"Mild negative interaction between {h1.name} and {h2.name}",
                        'recommendation': "Monitor formulation stability and consider pH adjustments"
                    })
        
        return warnings


def calculate_synergy_score(hypergredients: List[Hypergredient]) -> float:
    """Calculate overall synergy score for a list of hypergredients"""
    if len(hypergredients) < 2:
        return 1.0
    
    matrix = InteractionMatrix()
    analysis = matrix.analyze_formulation_interactions(hypergredients)
    
    # Convert total score to synergy multiplier
    # Score 7 = 1.0 (neutral), higher scores increase multiplier
    synergy_multiplier = 1.0 + (analysis['total_score'] - 7.0) * 0.1
    return max(0.5, min(2.0, synergy_multiplier))


def create_interaction_network_data(hypergredients: List[Hypergredient]) -> Dict:
    """Create network data for visualization of interactions"""
    matrix = InteractionMatrix()
    
    nodes = []
    edges = []
    
    # Create nodes
    for h in hypergredients:
        nodes.append({
            'id': h.name,
            'label': h.name,
            'class': h.hypergredient_class,
            'potency': h.potency,
            'safety': h.safety_score,
            'cost': h.cost_per_gram
        })
    
    # Create edges
    for i in range(len(hypergredients)):
        for j in range(i + 1, len(hypergredients)):
            h1, h2 = hypergredients[i], hypergredients[j]
            score = matrix.calculate_interaction_score(h1, h2)
            
            # Only include significant interactions
            if abs(score - 1.0) > 0.2:
                edge_type = 'synergy' if score > 1.0 else 'antagonism'
                edges.append({
                    'source': h1.name,
                    'target': h2.name,
                    'weight': score,
                    'type': edge_type,
                    'strength': abs(score - 1.0)
                })
    
    return {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'avg_interaction': np.mean([e['weight'] for e in edges]) if edges else 1.0
        }
    }
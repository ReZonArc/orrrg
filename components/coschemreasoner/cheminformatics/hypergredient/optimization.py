"""
Multi-Objective Formulation Optimizer

This module implements the advanced optimization algorithms for
hypergredient formulation design using multi-objective optimization.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
from collections import defaultdict

from .core import Hypergredient, HypergredientDatabase
from .interaction import InteractionMatrix, calculate_synergy_score


class OptimizationObjective(Enum):
    """Optimization objectives for formulation"""
    EFFICACY = "efficacy"
    SAFETY = "safety"
    STABILITY = "stability"
    COST = "cost"
    SYNERGY = "synergy"
    BIOAVAILABILITY = "bioavailability"
    REGULATORY = "regulatory"


@dataclass
class FormulationRequest:
    """Request for formulation optimization"""
    target_concerns: List[str]  # e.g., ['wrinkles', 'firmness', 'brightness']
    skin_type: str = "normal"  # normal, dry, oily, sensitive, combination
    budget: float = 1000.0  # ZAR budget limit
    preferences: List[str] = field(default_factory=list)  # ['gentle', 'stable', 'natural']
    excluded_ingredients: List[str] = field(default_factory=list)
    required_ingredients: List[str] = field(default_factory=list)
    ph_range: Tuple[float, float] = (4.5, 7.0)
    max_ingredients: int = 8
    regulatory_region: str = "EU"


@dataclass
class FormulationSolution:
    """Optimized formulation solution"""
    hypergredients: Dict[str, float]  # ingredient_name: percentage
    objective_scores: Dict[OptimizationObjective, float]
    total_score: float
    cost: float
    predicted_efficacy: Dict[str, float]  # concern: improvement_percentage
    warnings: List[str] = field(default_factory=list)
    synergy_score: float = 1.0
    stability_months: int = 24
    
    def get_summary(self) -> str:
        """Get formatted summary of the solution"""
        ingredients_list = [f"{name}: {pct:.1f}%" 
                          for name, pct in self.hypergredients.items()]
        
        return f"""
OPTIMAL FORMULATION
==================
Total Score: {self.total_score:.1f}/10
Cost: R{self.cost:.0f}/50ml
Stability: {self.stability_months} months
Synergy Score: {self.synergy_score:.1f}x

Ingredients:
{chr(10).join('• ' + ing for ing in ingredients_list)}

Predicted Results:
{chr(10).join(f'• {concern}: {improvement:.0f}% improvement' 
              for concern, improvement in self.predicted_efficacy.items())}
"""


class FormulationOptimizer:
    """Multi-objective optimizer for hypergredient formulations"""
    
    def __init__(self, database: HypergredientDatabase):
        self.database = database
        self.interaction_matrix = InteractionMatrix()
        
        # Optimization parameters
        self.population_size = 20  # Reduced for faster demo
        self.generations = 10      # Reduced for faster demo
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        # Objective weights
        self.default_weights = {
            OptimizationObjective.EFFICACY: 0.35,
            OptimizationObjective.SAFETY: 0.25,
            OptimizationObjective.STABILITY: 0.20,
            OptimizationObjective.COST: 0.15,
            OptimizationObjective.SYNERGY: 0.05
        }
    
    def optimize_formulation(self, request: FormulationRequest,
                           custom_weights: Optional[Dict[OptimizationObjective, float]] = None
                           ) -> List[FormulationSolution]:
        """Run multi-objective optimization"""
        weights = custom_weights or self.default_weights
        
        # Map concerns to hypergredient classes
        relevant_classes = self._map_concerns_to_classes(request.target_concerns)
        
        # Get candidate hypergredients
        candidates = self._get_candidate_hypergredients(
            relevant_classes, request
        )
        
        if len(candidates) < 2:
            return []  # Return empty list instead of raising exception
        
        # Initialize population
        population = self._initialize_population(candidates, request)
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            for solution in population:
                solution['total_score'] = self._evaluate_solution(
                    solution, request, weights
                )
            
            # Selection and reproduction
            population = self._evolve_population(population, candidates, request)
        
        # Final evaluation and ranking
        final_solutions = []
        for solution in population:
            final_solution = self._create_final_solution(solution, request, weights)
            final_solutions.append(final_solution)
        
        # Sort by total score and return top solutions
        final_solutions.sort(key=lambda x: x.total_score, reverse=True)
        return final_solutions[:5]
    
    def _map_concerns_to_classes(self, concerns: List[str]) -> List[str]:
        """Map user concerns to hypergredient classes"""
        concern_mapping = {
            'wrinkles': ['H.CS', 'H.CT'],
            'fine_lines': ['H.CS', 'H.CT'],
            'firmness': ['H.CS', 'H.BR'],
            'brightness': ['H.ML', 'H.AO'],
            'dark_spots': ['H.ML', 'H.CT'],
            'hyperpigmentation': ['H.ML', 'H.AO'],
            'hydration': ['H.HY', 'H.BR'],
            'dryness': ['H.HY', 'H.BR'],
            'anti_aging': ['H.CS', 'H.AO', 'H.CT'],
            'acne': ['H.SE', 'H.AI'],
            'oily_skin': ['H.SE'],
            'sensitive_skin': ['H.AI', 'H.BR'],
            'dullness': ['H.ML', 'H.CT'],
            'texture': ['H.CT', 'H.HY'],
            'pores': ['H.SE', 'H.CT'],
            'redness': ['H.AI'],
            'inflammation': ['H.AI', 'H.AO']
        }
        
        relevant_classes = set()
        for concern in concerns:
            classes = concern_mapping.get(concern.lower(), [])
            relevant_classes.update(classes)
        
        return list(relevant_classes)
    
    def _get_candidate_hypergredients(self, classes: List[str], 
                                    request: FormulationRequest) -> List[Hypergredient]:
        """Get candidate hypergredients based on classes and constraints"""
        candidates = []
        
        for class_code in classes:
            class_hypergredients = self.database.get_by_class(class_code)
            candidates.extend(class_hypergredients)
        
        # Apply filters
        filtered_candidates = []
        for h in candidates:
            # Budget filter
            if h.cost_per_gram * 5 > request.budget:  # Assume 5g per 50ml
                continue
            
            # Excluded ingredients filter
            if h.name in request.excluded_ingredients:
                continue
            
            # pH compatibility filter
            if (h.ph_range[1] < request.ph_range[0] or 
                h.ph_range[0] > request.ph_range[1]):
                continue
            
            # Skin type compatibility
            if not self._is_suitable_for_skin_type(h, request.skin_type):
                continue
            
            filtered_candidates.append(h)
        
        # Always include required ingredients
        for req_name in request.required_ingredients:
            if req_name in self.database.hypergredients:
                req_h = self.database.hypergredients[req_name]
                if req_h not in filtered_candidates:
                    filtered_candidates.append(req_h)
        
        return filtered_candidates
    
    def _is_suitable_for_skin_type(self, hypergredient: Hypergredient, 
                                 skin_type: str) -> bool:
        """Check if hypergredient is suitable for skin type"""
        sensitive_ingredients = ['tretinoin', 'glycolic_acid', 'salicylic_acid']
        drying_ingredients = ['salicylic_acid', 'glycolic_acid']
        
        if skin_type == "sensitive":
            if hypergredient.name in sensitive_ingredients:
                return hypergredient.safety_score >= 8.0
        
        if skin_type == "dry":
            if hypergredient.name in drying_ingredients:
                return False
        
        return True
    
    def _initialize_population(self, candidates: List[Hypergredient],
                             request: FormulationRequest) -> List[Dict]:
        """Initialize population of potential formulations"""
        population = []
        
        for _ in range(self.population_size):
            # Select random subset of candidates
            n_ingredients = random.randint(3, min(request.max_ingredients, len(candidates)))
            selected = random.sample(candidates, n_ingredients)
            
            # Assign random concentrations
            formulation = {}
            total_concentration = 0
            
            for h in selected:
                # Random concentration within reasonable bounds
                if h.hypergredient_class in ['H.CT', 'H.CS']:
                    max_conc = min(5.0, h.cost_per_gram / 20)  # Cost-based limit
                elif h.hypergredient_class == 'H.HY':
                    max_conc = 15.0
                else:
                    max_conc = 8.0
                
                concentration = random.uniform(0.5, max_conc)
                formulation[h.name] = concentration
                total_concentration += concentration
            
            # Normalize if total is too high
            if total_concentration > 25.0:  # Max 25% active ingredients
                scale_factor = 25.0 / total_concentration
                for name in formulation:
                    formulation[name] *= scale_factor
            
            population.append({
                'hypergredients': formulation,
                'candidates': selected,
                'total_score': 0.0
            })
        
        return population
    
    def _evaluate_solution(self, solution: Dict, request: FormulationRequest,
                          weights: Dict[OptimizationObjective, float]) -> float:
        """Evaluate a formulation solution"""
        hypergredients = [self.database.hypergredients[name] 
                         for name in solution['hypergredients']]
        
        # Calculate individual objective scores
        efficacy_score = self._calculate_efficacy_score(
            hypergredients, solution['hypergredients'], request.target_concerns
        )
        
        safety_score = self._calculate_safety_score(hypergredients)
        
        stability_score = self._calculate_stability_score(hypergredients)
        
        cost_score = self._calculate_cost_score(
            hypergredients, solution['hypergredients'], request.budget
        )
        
        synergy_score = calculate_synergy_score(hypergredients)
        
        # Weighted combination
        total_score = (
            efficacy_score * weights[OptimizationObjective.EFFICACY] +
            safety_score * weights[OptimizationObjective.SAFETY] +
            stability_score * weights[OptimizationObjective.STABILITY] +
            cost_score * weights[OptimizationObjective.COST] +
            (synergy_score - 1.0) * 10 * weights[OptimizationObjective.SYNERGY]
        )
        
        # Apply penalties
        penalty = self._calculate_penalties(hypergredients, request)
        total_score *= (1.0 - penalty)
        
        return max(0.0, min(10.0, total_score))
    
    def _calculate_efficacy_score(self, hypergredients: List[Hypergredient],
                                 concentrations: Dict[str, float],
                                 concerns: List[str]) -> float:
        """Calculate efficacy score for target concerns"""
        concern_scores = []
        
        for concern in concerns:
            relevant_classes = self._map_concerns_to_classes([concern])
            concern_score = 0.0
            
            for h in hypergredients:
                if h.hypergredient_class in relevant_classes:
                    concentration = concentrations[h.name]
                    # Efficacy increases with concentration but with diminishing returns
                    contrib = h.potency * (1 - np.exp(-concentration / 2.0))
                    concern_score += contrib
            
            concern_scores.append(min(10.0, concern_score))
        
        return np.mean(concern_scores) if concern_scores else 0.0
    
    def _calculate_safety_score(self, hypergredients: List[Hypergredient]) -> float:
        """Calculate overall safety score"""
        if not hypergredients:
            return 10.0
        
        # Weighted average of safety scores
        total_weight = 0
        weighted_sum = 0
        
        for h in hypergredients:
            weight = h.potency  # More potent ingredients have more impact
            weighted_sum += h.safety_score * weight
            total_weight += weight
        
        avg_safety = weighted_sum / total_weight if total_weight > 0 else 10.0
        
        # Check for interaction warnings
        interaction_analysis = self.interaction_matrix.analyze_formulation_interactions(
            hypergredients
        )
        
        # Penalize for antagonistic interactions
        antagonistic_penalty = len(interaction_analysis['antagonistic_pairs']) * 0.5
        
        return max(0.0, avg_safety - antagonistic_penalty)
    
    def _calculate_stability_score(self, hypergredients: List[Hypergredient]) -> float:
        """Calculate formulation stability score"""
        if not hypergredients:
            return 10.0
        
        stability_indices = [h._calculate_stability_index() for h in hypergredients]
        
        # Minimum stability determines overall stability
        min_stability = min(stability_indices)
        
        # Check for stability conflicts
        sensitive_count = sum(1 for h in hypergredients 
                            if 'sensitive' in h.stability or h.stability == 'unstable')
        
        stability_penalty = sensitive_count * 0.3
        
        final_score = (min_stability * 10) - stability_penalty
        return max(0.0, min(10.0, final_score))
    
    def _calculate_cost_score(self, hypergredients: List[Hypergredient],
                            concentrations: Dict[str, float], 
                            budget: float) -> float:
        """Calculate cost efficiency score"""
        total_cost = 0.0
        
        for h in hypergredients:
            concentration = concentrations[h.name]
            # Estimate cost per 50ml formulation
            ingredient_cost = h.cost_per_gram * concentration * 0.5  # 50ml = 50g approx
            total_cost += ingredient_cost
        
        if total_cost <= budget * 0.5:
            return 10.0  # Excellent cost efficiency
        elif total_cost <= budget:
            return 8.0 - (total_cost / budget - 0.5) * 8.0  # Linear decrease
        else:
            return max(0.0, 3.0 - (total_cost / budget - 1.0) * 2.0)  # Heavy penalty
    
    def _calculate_penalties(self, hypergredients: List[Hypergredient],
                           request: FormulationRequest) -> float:
        """Calculate various constraint penalties"""
        penalty = 0.0
        
        # pH incompatibility penalty
        for h in hypergredients:
            if (h.ph_range[1] < request.ph_range[0] or 
                h.ph_range[0] > request.ph_range[1]):
                penalty += 0.2
        
        # Too many ingredients penalty
        if len(hypergredients) > request.max_ingredients:
            penalty += (len(hypergredients) - request.max_ingredients) * 0.1
        
        # Missing required ingredients penalty
        present_names = {h.name for h in hypergredients}
        missing_required = set(request.required_ingredients) - present_names
        penalty += len(missing_required) * 0.3
        
        return min(0.8, penalty)  # Cap penalty at 80%
    
    def _evolve_population(self, population: List[Dict], 
                          candidates: List[Hypergredient],
                          request: FormulationRequest) -> List[Dict]:
        """Evolve population using genetic operators"""
        # Sort by fitness
        population.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Keep top performers (elitism)
        new_population = population[:self.population_size // 4]
        
        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                offspring = self._crossover(parent1, parent2, candidates)
            else:
                # Mutation only
                parent = self._tournament_selection(population)
                offspring = self._mutate(parent, candidates, request)
            
            new_population.append(offspring)
        
        return new_population
    
    def _tournament_selection(self, population: List[Dict], 
                            tournament_size: int = 3) -> Dict:
        """Tournament selection for genetic algorithm"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x['total_score'])
    
    def _crossover(self, parent1: Dict, parent2: Dict,
                  candidates: List[Hypergredient]) -> Dict:
        """Crossover operation between two formulations"""
        # Combine ingredients from both parents
        combined_ingredients = {}
        combined_ingredients.update(parent1['hypergredients'])
        
        for name, conc in parent2['hypergredients'].items():
            if name in combined_ingredients:
                # Average concentrations
                combined_ingredients[name] = (combined_ingredients[name] + conc) / 2
            else:
                combined_ingredients[name] = conc
        
        # Select subset if too many ingredients
        if len(combined_ingredients) > 8:
            # Keep highest concentration ingredients
            sorted_ingredients = sorted(combined_ingredients.items(), 
                                      key=lambda x: x[1], reverse=True)
            combined_ingredients = dict(sorted_ingredients[:8])
        
        selected_candidates = [self.database.hypergredients[name] 
                             for name in combined_ingredients]
        
        return {
            'hypergredients': combined_ingredients,
            'candidates': selected_candidates,
            'total_score': 0.0
        }
    
    def _mutate(self, individual: Dict, candidates: List[Hypergredient],
               request: FormulationRequest) -> Dict:
        """Mutation operation on a formulation"""
        mutated = individual.copy()
        mutated_ingredients = individual['hypergredients'].copy()
        
        if random.random() < self.mutation_rate:
            # Add new ingredient
            available = [c for c in candidates 
                        if c.name not in mutated_ingredients]
            if available and len(mutated_ingredients) < request.max_ingredients:
                new_ingredient = random.choice(available)
                mutated_ingredients[new_ingredient.name] = random.uniform(0.5, 3.0)
        
        if random.random() < self.mutation_rate and len(mutated_ingredients) > 2:
            # Remove ingredient
            to_remove = random.choice(list(mutated_ingredients.keys()))
            if to_remove not in request.required_ingredients:
                del mutated_ingredients[to_remove]
        
        # Mutate concentrations
        for name in mutated_ingredients:
            if random.random() < self.mutation_rate:
                current = mutated_ingredients[name]
                # Add random variation (±20%)
                variation = random.uniform(-0.2, 0.2) * current
                mutated_ingredients[name] = max(0.1, current + variation)
        
        selected_candidates = [self.database.hypergredients[name] 
                             for name in mutated_ingredients]
        
        return {
            'hypergredients': mutated_ingredients,
            'candidates': selected_candidates,
            'total_score': 0.0
        }
    
    def _create_final_solution(self, solution: Dict, request: FormulationRequest,
                             weights: Dict[OptimizationObjective, float]
                             ) -> FormulationSolution:
        """Create final FormulationSolution from optimization result"""
        hypergredients = [self.database.hypergredients[name] 
                         for name in solution['hypergredients']]
        
        # Calculate individual objective scores
        objective_scores = {
            OptimizationObjective.EFFICACY: self._calculate_efficacy_score(
                hypergredients, solution['hypergredients'], request.target_concerns
            ),
            OptimizationObjective.SAFETY: self._calculate_safety_score(hypergredients),
            OptimizationObjective.STABILITY: self._calculate_stability_score(hypergredients),
            OptimizationObjective.SYNERGY: calculate_synergy_score(hypergredients)
        }
        
        # Calculate cost
        total_cost = sum(
            self.database.hypergredients[name].cost_per_gram * conc * 0.5
            for name, conc in solution['hypergredients'].items()
        )
        
        objective_scores[OptimizationObjective.COST] = self._calculate_cost_score(
            hypergredients, solution['hypergredients'], request.budget
        )
        
        # Predict efficacy for each concern
        predicted_efficacy = {}
        for concern in request.target_concerns:
            # Simple prediction model based on ingredient potency and concentration
            efficacy = 0
            for h in hypergredients:
                if h.hypergredient_class in self._map_concerns_to_classes([concern]):
                    conc = solution['hypergredients'][h.name]
                    efficacy += h.potency * conc * 2  # Simple scaling
            
            predicted_efficacy[concern] = min(80, efficacy)  # Cap at 80% improvement
        
        # Generate warnings
        warnings = []
        interaction_analysis = self.interaction_matrix.analyze_formulation_interactions(
            hypergredients
        )
        
        for pair in interaction_analysis['antagonistic_pairs']:
            warnings.append(
                f"Potential incompatibility: {pair['ingredient1']} + {pair['ingredient2']}"
            )
        
        return FormulationSolution(
            hypergredients=solution['hypergredients'],
            objective_scores=objective_scores,
            total_score=solution['total_score'],
            cost=total_cost,
            predicted_efficacy=predicted_efficacy,
            warnings=warnings,
            synergy_score=calculate_synergy_score(hypergredients),
            stability_months=max(6, int(objective_scores[OptimizationObjective.STABILITY] * 3))
        )


class HypergredientFormulator:
    """High-level interface for hypergredient formulation"""
    
    def __init__(self, database: HypergredientDatabase):
        self.database = database
        self.optimizer = FormulationOptimizer(database)
    
    def generate_formulation(self, target: str, secondary: List[str] = None,
                           budget: float = 1000, exclude: List[str] = None,
                           **kwargs) -> FormulationSolution:
        """Generate optimal formulation for target concern"""
        request = FormulationRequest(
            target_concerns=[target] + (secondary or []),
            budget=budget,
            excluded_ingredients=exclude or [],
            **kwargs
        )
        
        solutions = self.optimizer.optimize_formulation(request)
        return solutions[0] if solutions else None
    
    def compare_formulations(self, requests: List[FormulationRequest]) -> List[FormulationSolution]:
        """Compare multiple formulation requests"""
        results = []
        
        for request in requests:
            solutions = self.optimizer.optimize_formulation(request)
            if solutions:
                results.append(solutions[0])
        
        return results
    
    def suggest_improvements(self, current_formulation: Dict[str, float],
                           target_score: float = 8.0) -> List[str]:
        """Suggest improvements to existing formulation"""
        suggestions = []
        
        # Analyze current formulation
        hypergredients = [self.database.hypergredients[name] 
                         for name in current_formulation if name in self.database.hypergredients]
        
        interaction_analysis = self.optimizer.interaction_matrix.analyze_formulation_interactions(
            hypergredients
        )
        
        current_score = interaction_analysis['total_score']
        
        if current_score < target_score:
            # Suggest removing antagonistic ingredients
            for pair in interaction_analysis['antagonistic_pairs']:
                suggestions.append(
                    f"Consider removing {pair['ingredient1']} or {pair['ingredient2']} due to negative interaction"
                )
            
            # Suggest adding synergistic ingredients
            for h in hypergredients:
                complementary = self.optimizer.interaction_matrix.suggest_complementary_hypergredients(
                    h, list(self.database.hypergredients.values()), 3
                )
                
                for comp_h, score in complementary:
                    if comp_h.name not in current_formulation:
                        suggestions.append(
                            f"Consider adding {comp_h.name} for synergy with {h.name} (score: {score:.2f})"
                        )
        
        return suggestions[:10]  # Limit to top 10 suggestions
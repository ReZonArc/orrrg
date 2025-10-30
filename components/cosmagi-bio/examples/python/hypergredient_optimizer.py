#!/usr/bin/env python3
"""
🧮 Hypergredient Formulation Optimizer

This module implements the multi-objective optimization algorithm for the
Hypergredient Framework, generating optimal cosmeceutical formulations
based on target concerns, constraints, and performance objectives.

Key Features:
- Multi-objective formulation optimization
- Dynamic scoring and ranking system
- Real-time compatibility checking
- Economic constraint handling
- Performance prediction and validation
- Evolutionary formulation improvement

Author: OpenCog Multiscale Optimization Framework
License: AGPL-3.0
"""

import uuid
import json
import math
import logging
import random
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time

from hypergredient_framework import (
    HypergredientDatabase, Hypergredient, HypergredientClass,
    InteractionType, HypergredientMetrics
)

# Optional ML integration
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available, using basic math operations")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinType(Enum):
    """Skin type classifications."""
    NORMAL = "normal"
    DRY = "dry"
    OILY = "oily"
    COMBINATION = "combination"
    SENSITIVE = "sensitive"
    MATURE = "mature"


class ConcernType(Enum):
    """Skin concern classifications."""
    WRINKLES = "wrinkles"
    FIRMNESS = "firmness" 
    BRIGHTNESS = "brightness"
    HYDRATION = "hydration"
    ACNE = "acne"
    HYPERPIGMENTATION = "hyperpigmentation"
    SENSITIVITY = "sensitivity"
    BARRIER_DAMAGE = "barrier_damage"
    UNEVEN_TEXTURE = "uneven_texture"
    SEBUM_CONTROL = "sebum_control"


@dataclass
class FormulationRequest:
    """Request for formulation optimization."""
    concerns: List[ConcernType]
    skin_type: SkinType
    budget_limit: float  # ZAR per 100g
    preferences: List[str] = field(default_factory=list)  # e.g., ['gentle', 'stable']
    exclude_ingredients: List[str] = field(default_factory=list)
    target_ph: Optional[float] = None
    max_active_concentration: float = 25.0
    texture_preference: str = "lightweight"  # lightweight, rich, gel, cream


@dataclass 
class FormulationCandidate:
    """A candidate formulation with predicted properties."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hypergredients: Dict[str, Tuple[Hypergredient, float]] = field(default_factory=dict)  # name -> (hypergredient, concentration)
    
    # Predicted properties
    efficacy_score: float = 0.0
    safety_score: float = 0.0
    stability_score: float = 0.0
    cost_per_100g: float = 0.0
    synergy_score: float = 0.0
    
    # Overall fitness
    fitness_score: float = 0.0
    
    # Metadata
    generation: int = 0
    created_at: float = field(default_factory=time.time)
    
    def get_total_active_concentration(self) -> float:
        """Calculate total active ingredient concentration."""
        return sum(conc for _, conc in self.hypergredients.values())
    
    def get_ingredient_summary(self) -> Dict[str, float]:
        """Get summary of ingredients and concentrations."""
        return {name: conc for name, (_, conc) in self.hypergredients.items()}


@dataclass
class OptimizationResult:
    """Result of formulation optimization."""
    best_formulation: FormulationCandidate
    all_candidates: List[FormulationCandidate]
    optimization_time: float
    generations: int
    convergence_metrics: Dict[str, float]
    
    # Performance summary
    predicted_efficacy: float
    predicted_safety: float
    estimated_cost: float
    stability_months: int
    
    # Detailed analysis
    synergy_analysis: Dict[str, Any]
    compatibility_issues: List[str]
    improvement_suggestions: List[str]


class HypergredientFormulationOptimizer:
    """Multi-objective optimizer for hypergredient-based formulations."""
    
    def __init__(self, database: HypergredientDatabase):
        self.database = database
        self.concern_to_class_mapping = self._initialize_concern_mapping()
        self.skin_type_constraints = self._initialize_skin_type_constraints()
        
        # Optimization parameters
        self.population_size = 50
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        # Objective weights
        self.default_weights = {
            'efficacy': 0.35,
            'safety': 0.25,
            'stability': 0.20,
            'cost': 0.15,
            'synergy': 0.05
        }
    
    def _initialize_concern_mapping(self) -> Dict[ConcernType, List[HypergredientClass]]:
        """Map skin concerns to relevant hypergredient classes."""
        return {
            ConcernType.WRINKLES: [HypergredientClass.H_CT, HypergredientClass.H_CS],
            ConcernType.FIRMNESS: [HypergredientClass.H_CS, HypergredientClass.H_CT],
            ConcernType.BRIGHTNESS: [HypergredientClass.H_ML, HypergredientClass.H_AO],
            ConcernType.HYDRATION: [HypergredientClass.H_HY, HypergredientClass.H_BR],
            ConcernType.ACNE: [HypergredientClass.H_CT, HypergredientClass.H_SE, HypergredientClass.H_AI],
            ConcernType.HYPERPIGMENTATION: [HypergredientClass.H_ML, HypergredientClass.H_AO],
            ConcernType.SENSITIVITY: [HypergredientClass.H_AI, HypergredientClass.H_BR],
            ConcernType.BARRIER_DAMAGE: [HypergredientClass.H_BR, HypergredientClass.H_HY],
            ConcernType.UNEVEN_TEXTURE: [HypergredientClass.H_CT, HypergredientClass.H_AO],
            ConcernType.SEBUM_CONTROL: [HypergredientClass.H_SE, HypergredientClass.H_AI]
        }
    
    def _initialize_skin_type_constraints(self) -> Dict[SkinType, Dict[str, Any]]:
        """Initialize constraints based on skin type."""
        return {
            SkinType.NORMAL: {
                'max_irritation_risk': 0.7,
                'preferred_ph_range': (5.0, 7.0),
                'max_active_concentration': 25.0
            },
            SkinType.DRY: {
                'max_irritation_risk': 0.5,
                'preferred_ph_range': (5.5, 7.5),
                'max_active_concentration': 20.0,
                'boost_hydration': 1.5
            },
            SkinType.OILY: {
                'max_irritation_risk': 0.8,
                'preferred_ph_range': (4.5, 6.5),
                'max_active_concentration': 30.0,
                'boost_sebum_control': 1.5
            },
            SkinType.COMBINATION: {
                'max_irritation_risk': 0.6,
                'preferred_ph_range': (5.0, 7.0),
                'max_active_concentration': 22.0
            },
            SkinType.SENSITIVE: {
                'max_irritation_risk': 0.3,
                'preferred_ph_range': (5.5, 7.0),
                'max_active_concentration': 15.0,
                'prefer_gentle': True
            },
            SkinType.MATURE: {
                'max_irritation_risk': 0.4,
                'preferred_ph_range': (5.0, 6.5),
                'max_active_concentration': 20.0,
                'boost_anti_aging': 1.5
            }
        }
    
    def optimize_formulation(self, request: FormulationRequest) -> OptimizationResult:
        """Generate optimal formulation using multi-objective optimization."""
        logger.info(f"🚀 Starting formulation optimization for {len(request.concerns)} concerns")
        start_time = time.time()
        
        # Generate initial population
        population = self._generate_initial_population(request)
        
        # Evolutionary optimization
        best_candidate = None 
        generations = 0
        convergence_history = []
        
        for generation in range(self.max_generations):
            # Evaluate fitness for all candidates
            self._evaluate_population_fitness(population, request)
            
            # Track best candidate
            population.sort(key=lambda x: x.fitness_score, reverse=True)
            current_best = population[0]
            
            if best_candidate is None or current_best.fitness_score > best_candidate.fitness_score:
                best_candidate = current_best
                best_candidate.generation = generation
            
            convergence_history.append(current_best.fitness_score)
            
            # Check convergence
            if generation > 10:
                recent_improvement = max(convergence_history[-10:]) - min(convergence_history[-10:])
                if recent_improvement < 0.001:
                    logger.info(f"Converged after {generation} generations")
                    break
            
            # Generate next generation
            population = self._evolve_population(population, request)
            generations = generation + 1
        
        optimization_time = time.time() - start_time
        
        # Analyze final result
        synergy_analysis = self._analyze_synergies(best_candidate)
        compatibility_issues = self._check_compatibility_issues(best_candidate)
        improvement_suggestions = self._generate_improvement_suggestions(best_candidate, request)
        
        result = OptimizationResult(
            best_formulation=best_candidate,
            all_candidates=population[:10],  # Top 10 candidates
            optimization_time=optimization_time,
            generations=generations,
            convergence_metrics={'fitness_history': convergence_history},
            predicted_efficacy=best_candidate.efficacy_score,
            predicted_safety=best_candidate.safety_score,
            estimated_cost=best_candidate.cost_per_100g,
            stability_months=int(best_candidate.stability_score * 24),
            synergy_analysis=synergy_analysis,
            compatibility_issues=compatibility_issues,
            improvement_suggestions=improvement_suggestions
        )
        
        logger.info(f"✅ Optimization completed in {optimization_time:.2f}s")
        return result
    
    def _generate_initial_population(self, request: FormulationRequest) -> List[FormulationCandidate]:
        """Generate initial population of candidate formulations."""
        population = []
        
        for i in range(self.population_size):
            candidate = FormulationCandidate()
            
            # Use different strategies for diversity
            strategy = i % 3  # 0: conservative, 1: balanced, 2: aggressive
            
            # Select hypergredients for each concern
            added_classes = set()
            for concern in request.concerns:
                relevant_classes = self.concern_to_class_mapping.get(concern, [])
                
                for hg_class in relevant_classes[:2]:  # Max 2 classes per concern
                    if hg_class in added_classes:
                        continue  # Avoid duplicates
                    
                    hypergredients = self.database.get_hypergredients_by_class(hg_class)
                    
                    if hypergredients:
                        # Filter out excluded ingredients
                        available = [h for h in hypergredients 
                                   if h.name.lower() not in [e.lower() for e in request.exclude_ingredients]]
                        
                        if available:
                            # Strategy-based selection
                            if strategy == 0:  # Conservative - lowest cost
                                selected = min(available, key=lambda x: x.metrics.cost_per_gram)
                            elif strategy == 1:  # Balanced - best efficacy/cost ratio
                                selected = max(available, key=lambda x: x.metrics.efficacy_score / max(x.metrics.cost_per_gram, 1.0))
                            else:  # Aggressive - highest efficacy
                                selected = max(available, key=lambda x: x.metrics.efficacy_score)
                            
                            # Strategy-based concentration
                            if strategy == 0:
                                concentration = selected.min_concentration + 0.1
                            elif strategy == 1:
                                concentration = selected.typical_concentration
                            else:
                                concentration = min(selected.max_concentration, 
                                                  request.max_active_concentration / 4)
                            
                            candidate.hypergredients[selected.name] = (selected, concentration)
                            added_classes.add(hg_class)
            
            # Add base ingredients if needed (with some randomness)
            if not any('hydration' in h.primary_function.lower() 
                      for h, _ in candidate.hypergredients.values()):
                # Add basic hydration
                hydration_hgs = self.database.get_hypergredients_by_class(HypergredientClass.H_HY)
                if hydration_hgs and random.random() > 0.3:  # 70% chance to add
                    selected = random.choice(hydration_hgs)
                    concentration = random.uniform(
                        selected.min_concentration, 
                        min(selected.max_concentration, selected.typical_concentration * 1.5)
                    )
                    candidate.hypergredients[selected.name] = (selected, concentration)
            
            population.append(candidate)
        
        return population
    
    def _evaluate_population_fitness(self, population: List[FormulationCandidate], 
                                   request: FormulationRequest):
        """Evaluate fitness scores for all candidates in population."""
        for candidate in population:
            candidate.fitness_score = self._calculate_fitness_score(candidate, request)
    
    def _calculate_fitness_score(self, candidate: FormulationCandidate, 
                               request: FormulationRequest) -> float:
        """Calculate comprehensive fitness score for a candidate."""
        
        # Calculate component scores
        efficacy_score = self._calculate_efficacy_score(candidate, request)
        safety_score = self._calculate_safety_score(candidate, request)
        stability_score = self._calculate_stability_score(candidate)
        cost_score = self._calculate_cost_score(candidate, request)
        synergy_score = self._calculate_synergy_score(candidate)
        
        # Store individual scores
        candidate.efficacy_score = efficacy_score
        candidate.safety_score = safety_score
        candidate.stability_score = stability_score
        candidate.cost_per_100g = self._calculate_cost_per_100g(candidate)
        candidate.synergy_score = synergy_score
        
        # Weight and combine scores
        weights = self.default_weights.copy()
        
        # Adjust weights based on preferences
        if 'gentle' in request.preferences:
            weights['safety'] *= 1.5
            weights['efficacy'] *= 0.9
        if 'stable' in request.preferences:
            weights['stability'] *= 1.3
        if 'budget' in request.preferences:
            weights['cost'] *= 1.4
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        fitness = (efficacy_score * weights['efficacy'] +
                  safety_score * weights['safety'] +
                  stability_score * weights['stability'] +
                  cost_score * weights['cost'] +
                  synergy_score * weights['synergy'])
        
        # Apply penalties
        fitness = self._apply_constraint_penalties(fitness, candidate, request)
        
        return max(0.0, min(1.0, fitness))
    
    def _calculate_efficacy_score(self, candidate: FormulationCandidate, 
                                request: FormulationRequest) -> float:
        """Calculate predicted efficacy score."""
        if not candidate.hypergredients:
            return 0.0
        
        total_efficacy = 0.0
        total_weight = 0.0
        
        for name, (hypergredient, concentration) in candidate.hypergredients.items():
            # Base efficacy from hypergredient
            base_efficacy = hypergredient.metrics.efficacy_score / 10.0
            
            # Concentration effectiveness curve (not linear)
            conc_factor = min(1.0, concentration / hypergredient.typical_concentration)
            conc_effectiveness = math.sqrt(conc_factor)  # Diminishing returns
            
            # Bioavailability factor
            bio_factor = hypergredient.metrics.bioavailability
            
            # Concern relevance boost
            concern_boost = 1.0
            for concern in request.concerns:
                relevant_classes = self.concern_to_class_mapping.get(concern, [])
                if hypergredient.hypergredient_class in relevant_classes:
                    concern_boost = 1.3
                    break
            
            ingredient_efficacy = base_efficacy * conc_effectiveness * bio_factor * concern_boost
            weight = concentration
            
            total_efficacy += ingredient_efficacy * weight
            total_weight += weight
        
        return total_efficacy / max(total_weight, 0.1)
    
    def _calculate_safety_score(self, candidate: FormulationCandidate, 
                              request: FormulationRequest) -> float:
        """Calculate predicted safety score."""
        if not candidate.hypergredients:
            return 1.0
        
        safety_scores = []
        irritation_risk = 0.0
        
        for name, (hypergredient, concentration) in candidate.hypergredients.items():
            # Base safety
            base_safety = hypergredient.metrics.safety_score / 10.0
            safety_scores.append(base_safety)
            
            # Concentration-dependent irritation risk
            conc_risk = (concentration / hypergredient.max_concentration) * 0.2
            irritation_risk += conc_risk
        
        # Average safety, penalized by irritation risk
        avg_safety = sum(safety_scores) / len(safety_scores)
        skin_constraints = self.skin_type_constraints.get(request.skin_type, {})
        max_irritation = skin_constraints.get('max_irritation_risk', 0.7)
        
        if irritation_risk > max_irritation:
            penalty = (irritation_risk - max_irritation) * 2.0
            avg_safety *= (1.0 - penalty)
        
        return max(0.0, min(1.0, avg_safety))
    
    def _calculate_stability_score(self, candidate: FormulationCandidate) -> float:
        """Calculate predicted stability score."""
        if not candidate.hypergredients:
            return 1.0
        
        stability_scores = []
        
        for name, (hypergredient, concentration) in candidate.hypergredients.items():
            stability_scores.append(hypergredient.metrics.stability_index)
        
        # Weakest link approach for stability
        return min(stability_scores) if stability_scores else 1.0
    
    def _calculate_cost_score(self, candidate: FormulationCandidate, 
                            request: FormulationRequest) -> float:
        """Calculate cost efficiency score."""
        cost_per_100g = self._calculate_cost_per_100g(candidate)
        
        # Budget constraint with strong penalty for exceeding
        if cost_per_100g <= request.budget_limit:
            # Reward being under budget
            efficiency = 1.0 - (cost_per_100g / request.budget_limit) * 0.3
            return max(0.7, efficiency)
        else:
            # Strong penalty for exceeding budget
            excess = (cost_per_100g - request.budget_limit) / request.budget_limit
            return max(0.0, 1.0 - excess * 5.0)  # Increased penalty factor
    
    def _calculate_cost_per_100g(self, candidate: FormulationCandidate) -> float:
        """Calculate estimated cost per 100g of formulation."""
        total_cost = 0.0
        
        for name, (hypergredient, concentration) in candidate.hypergredients.items():
            ingredient_cost = (concentration / 100.0) * hypergredient.metrics.cost_per_gram
            total_cost += ingredient_cost
        
        # Add base formulation costs (water, preservatives, etc.)
        base_cost = 20.0  # ZAR per 100g for base
        
        return total_cost + base_cost
    
    def _calculate_synergy_score(self, candidate: FormulationCandidate) -> float:
        """Calculate synergy score based on ingredient interactions."""
        if len(candidate.hypergredients) < 2:
            return 0.5  # Neutral for single ingredients
        
        synergy_effects = []
        hypergredient_classes = [h.hypergredient_class for h, _ in candidate.hypergredients.values()]
        
        # Check all pairs
        for i, class_a in enumerate(hypergredient_classes):
            for j, class_b in enumerate(hypergredient_classes[i+1:], i+1):
                interaction_strength = self.database.get_interaction_strength(class_a, class_b)
                synergy_effects.append(interaction_strength)
        
        if synergy_effects:
            # Average interaction strength, normalized to 0-1
            avg_synergy = sum(synergy_effects) / len(synergy_effects)
            # Map interaction strength to 0-1 range with proper bounds
            normalized = (avg_synergy - 0.5) / 1.5 + 0.5  # Map [0.5-1.5, 0.5+1.5] to [0, 1]
            return max(0.0, min(1.0, normalized))
        
        return 0.5
    
    def _apply_constraint_penalties(self, fitness: float, candidate: FormulationCandidate,
                                  request: FormulationRequest) -> float:
        """Apply constraint penalties to fitness score."""
        
        # Total active concentration penalty
        total_actives = candidate.get_total_active_concentration()
        if total_actives > request.max_active_concentration:
            excess = (total_actives - request.max_active_concentration) / request.max_active_concentration
            fitness *= (1.0 - excess * 0.5)
        
        # pH compatibility penalty
        if request.target_ph:
            incompatible_count = 0
            total_ingredients = len(candidate.hypergredients)
            
            for name, (hypergredient, _) in candidate.hypergredients.items():
                if not (hypergredient.ph_min <= request.target_ph <= hypergredient.ph_max):
                    incompatible_count += 1
            
            if incompatible_count > 0 and total_ingredients > 0:
                ph_penalty = (incompatible_count / total_ingredients) * 0.3
                fitness *= (1.0 - ph_penalty)
        
        return fitness
    
    def _evolve_population(self, population: List[FormulationCandidate], 
                          request: FormulationRequest) -> List[FormulationCandidate]:
        """Evolve population using genetic algorithm operations."""
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        new_population = []
        
        # Elite selection (keep top 20%)
        elite_size = int(self.population_size * 0.2)
        new_population.extend(population[:elite_size])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                offspring = self._crossover(parent1, parent2, request)
            else:
                # Mutation only
                parent = self._tournament_selection(population)
                offspring = self._mutate(parent, request)
            
            new_population.append(offspring)
        
        return new_population
    
    def _tournament_selection(self, population: List[FormulationCandidate],
                            tournament_size: int = 3) -> FormulationCandidate:
        """Select candidate using tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    def _crossover(self, parent1: FormulationCandidate, parent2: FormulationCandidate,
                  request: FormulationRequest) -> FormulationCandidate:
        """Create offspring through crossover of two parents."""
        offspring = FormulationCandidate()
        
        # Combine ingredients from both parents
        all_ingredients = {}
        all_ingredients.update(parent1.hypergredients)
        all_ingredients.update(parent2.hypergredients)
        
        # Randomly select subset of ingredients
        for name, (hypergredient, concentration) in all_ingredients.items():
            if random.random() < 0.7:  # 70% chance to include each ingredient
                # Blend concentrations if present in both parents
                if (name in parent1.hypergredients and name in parent2.hypergredients):
                    conc1 = parent1.hypergredients[name][1]
                    conc2 = parent2.hypergredients[name][1]
                    new_concentration = (conc1 + conc2) / 2.0
                else:
                    new_concentration = concentration
                
                # Clamp to valid range
                new_concentration = max(hypergredient.min_concentration,
                                      min(hypergredient.max_concentration, new_concentration))
                
                offspring.hypergredients[name] = (hypergredient, new_concentration)
        
        return offspring
    
    def _mutate(self, candidate: FormulationCandidate, 
               request: FormulationRequest) -> FormulationCandidate:
        """Create mutated copy of candidate."""
        mutated = FormulationCandidate()
        mutated.hypergredients = candidate.hypergredients.copy()
        
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['add', 'remove', 'modify'])
            
            if mutation_type == 'add' and len(mutated.hypergredients) < 8:
                # Add new ingredient
                all_hypergredients = list(self.database.hypergredients.values())
                available = [h for h in all_hypergredients 
                           if h.name not in mutated.hypergredients and
                              h.name.lower() not in [e.lower() for e in request.exclude_ingredients]]
                
                if available:
                    new_hg = random.choice(available)
                    concentration = random.uniform(new_hg.min_concentration, new_hg.max_concentration)
                    mutated.hypergredients[new_hg.name] = (new_hg, concentration)
            
            elif mutation_type == 'remove' and len(mutated.hypergredients) > 2:
                # Remove ingredient
                name_to_remove = random.choice(list(mutated.hypergredients.keys()))
                del mutated.hypergredients[name_to_remove]
            
            elif mutation_type == 'modify' and mutated.hypergredients:
                # Modify concentration
                name_to_modify = random.choice(list(mutated.hypergredients.keys()))
                hypergredient, old_concentration = mutated.hypergredients[name_to_modify]
                
                # Random walk in concentration
                delta = random.uniform(-0.5, 0.5)
                new_concentration = old_concentration + delta
                new_concentration = max(hypergredient.min_concentration,
                                      min(hypergredient.max_concentration, new_concentration))
                
                mutated.hypergredients[name_to_modify] = (hypergredient, new_concentration)
        
        return mutated
    
    def _analyze_synergies(self, candidate: FormulationCandidate) -> Dict[str, Any]:
        """Analyze synergistic effects in the formulation."""
        synergies = []
        antagonisms = []
        
        hypergredients = list(candidate.hypergredients.values())
        
        for i, (hg_a, conc_a) in enumerate(hypergredients):
            for hg_b, conc_b in hypergredients[i+1:]:
                interaction = self.database.get_interaction_strength(
                    hg_a.hypergredient_class, hg_b.hypergredient_class
                )
                
                if interaction > 1.2:
                    synergies.append({
                        'ingredients': [hg_a.name, hg_b.name],
                        'strength': interaction,
                        'classes': [hg_a.hypergredient_class.value, hg_b.hypergredient_class.value]
                    })
                elif interaction < 0.8:
                    antagonisms.append({
                        'ingredients': [hg_a.name, hg_b.name],
                        'strength': interaction,
                        'classes': [hg_a.hypergredient_class.value, hg_b.hypergredient_class.value]
                    })
        
        return {
            'synergies': synergies,
            'antagonisms': antagonisms,
            'synergy_count': len(synergies),
            'antagonism_count': len(antagonisms)
        }
    
    def _check_compatibility_issues(self, candidate: FormulationCandidate) -> List[str]:
        """Check for compatibility issues in the formulation."""
        issues = []
        
        for name, (hypergredient, concentration) in candidate.hypergredients.items():
            # Check incompatibilities
            for other_name, (other_hg, _) in candidate.hypergredients.items():
                if name != other_name:
                    if other_hg.name.lower() in [inc.lower() for inc in hypergredient.incompatibilities]:
                        issues.append(f"Incompatibility: {hypergredient.name} and {other_hg.name}")
            
            # Check concentration limits
            if concentration > hypergredient.max_concentration:
                issues.append(f"Concentration too high: {hypergredient.name} at {concentration:.1f}% (max: {hypergredient.max_concentration:.1f}%)")
        
        return issues
    
    def _generate_improvement_suggestions(self, candidate: FormulationCandidate,
                                        request: FormulationRequest) -> List[str]:
        """Generate suggestions for formulation improvement."""
        suggestions = []
        
        # Cost optimization
        if candidate.cost_per_100g > request.budget_limit:
            suggestions.append("Consider replacing expensive ingredients with cost-effective alternatives")
        
        # Efficacy improvement
        if candidate.efficacy_score < 0.7:
            suggestions.append("Add more potent active ingredients or increase concentrations")
        
        # Safety improvement
        if candidate.safety_score < 0.8:
            suggestions.append("Reduce concentrations of potentially irritating ingredients")
        
        # Stability improvement
        if candidate.stability_score < 0.6:
            suggestions.append("Add stabilizing agents or replace unstable ingredients")
        
        # Synergy optimization
        if candidate.synergy_score < 0.6:
            suggestions.append("Consider adding synergistic ingredient combinations")
        
        return suggestions


def generate_optimal_formulation(concerns: List[str], skin_type: str = "normal", 
                               budget: float = 1000.0) -> Dict[str, Any]:
    """Convenience function to generate optimal formulation."""
    
    # Convert string inputs to enums
    concern_enums = []
    for concern in concerns:
        try:
            concern_enums.append(ConcernType(concern.lower()))
        except ValueError:
            logger.warning(f"Unknown concern: {concern}")
    
    try:
        skin_type_enum = SkinType(skin_type.lower())
    except ValueError:
        logger.warning(f"Unknown skin type: {skin_type}, using normal")
        skin_type_enum = SkinType.NORMAL
    
    # Create request
    request = FormulationRequest(
        concerns=concern_enums,
        skin_type=skin_type_enum,
        budget_limit=budget,
        preferences=['gentle', 'stable']
    )
    
    # Optimize
    db = HypergredientDatabase()
    optimizer = HypergredientFormulationOptimizer(db)
    result = optimizer.optimize_formulation(request)
    
    # Format output
    formulation_summary = {
        'ingredients': result.best_formulation.get_ingredient_summary(),
        'predicted_efficacy': f"{result.predicted_efficacy:.1%}",
        'predicted_safety': f"{result.predicted_safety:.1%}",
        'estimated_cost': f"R{result.estimated_cost:.0f}/100g",
        'stability_months': result.stability_months,
        'synergies': result.synergy_analysis['synergy_count'],
        'optimization_time': f"{result.optimization_time:.2f}s",
        'generations': result.generations
    }
    
    return formulation_summary


if __name__ == "__main__":
    # Demonstration
    print("🧮 Hypergredient Formulation Optimizer Demonstration")
    print("=" * 60)
    
    # Example: Generate optimal anti-aging formulation
    print("\n🎯 Generating Optimal Anti-Aging Formulation...")
    result = generate_optimal_formulation(
        concerns=['wrinkles', 'firmness', 'brightness'],
        skin_type='normal',
        budget=1500.0
    )
    
    print("\n📋 OPTIMAL FORMULATION RESULTS:")
    print(f"  🧪 Ingredients:")
    for ingredient, concentration in result['ingredients'].items():
        print(f"    • {ingredient}: {concentration:.2f}%")
    
    print(f"\n  📊 Performance Metrics:")
    print(f"    • Predicted Efficacy: {result['predicted_efficacy']}")
    print(f"    • Predicted Safety: {result['predicted_safety']}")
    print(f"    • Estimated Cost: {result['estimated_cost']}")
    print(f"    • Stability: {result['stability_months']} months")
    print(f"    • Synergistic Combinations: {result['synergies']}")
    
    print(f"\n  ⚡ Optimization Performance:")
    print(f"    • Time: {result['optimization_time']}")
    print(f"    • Generations: {result['generations']}")
    
    print("\n✅ Hypergredient Optimizer demonstration completed!")
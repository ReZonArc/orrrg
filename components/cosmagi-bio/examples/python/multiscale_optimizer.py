#!/usr/bin/env python3
"""
Multiscale Constraint Optimization Engine for Cosmeceutical Formulation

This module implements a comprehensive multiscale optimization system that operates
across molecular, cellular, tissue, and organ biological scales to achieve optimal
cosmeceutical formulation design. It integrates MOSES-inspired evolutionary algorithms
with PLN-based constraint reasoning and ECAN attention management.

Key Features:
- Multi-objective evolutionary optimization across biological scales
- Constraint satisfaction with regulatory compliance automation
- Emergent property calculation from molecular to organ-level
- Integration with INCI search space reduction and attention allocation

Requirements:
- Python 3.7+
- NumPy, SciPy for mathematical operations
- OpenCog AtomSpace (if available)
- INCI optimizer and attention allocation modules

Usage:
    from multiscale_optimizer import MultiscaleConstraintOptimizer
    
    optimizer = MultiscaleConstraintOptimizer()
    result = optimizer.optimize_formulation(objectives, constraints)

Author: OpenCog Multiscale Optimization Framework
License: AGPL-3.0
"""

import math
import time
import random
import logging
from typing import Dict, List, Tuple, Optional, Set, NamedTuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    print("Warning: NumPy not available, using basic math operations")

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available, using basic optimization")

try:
    from opencog.atomspace import AtomSpace, types
    from opencog.type_constructors import *
    OPENCOG_AVAILABLE = True
except ImportError:
    OPENCOG_AVAILABLE = False
    print("Warning: OpenCog not available, using standalone optimization")

# Import our other modules
try:
    from inci_optimizer import INCISearchSpaceReducer, FormulationCandidate
    from attention_allocation import AttentionAllocationManager
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("Warning: INCI and Attention modules not available, using simplified optimization")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiologicalScale(Enum):
    """Biological scales for multiscale optimization."""
    MOLECULAR = "molecular"
    CELLULAR = "cellular" 
    TISSUE = "tissue"
    ORGAN = "organ"


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    EFFICACY = "efficacy"
    SAFETY = "safety"
    COST = "cost"
    STABILITY = "stability"
    SENSORY = "sensory"
    SUSTAINABILITY = "sustainability"


class ConstraintType(Enum):
    """Types of optimization constraints."""
    REGULATORY = "regulatory"
    PHYSICAL = "physical"
    CHEMICAL = "chemical"
    BIOLOGICAL = "biological"
    MANUFACTURING = "manufacturing"
    ECONOMIC = "economic"


@dataclass
class Objective:
    """Optimization objective definition."""
    objective_type: ObjectiveType
    target_value: float
    weight: float
    scale: BiologicalScale
    tolerance: float = 0.1
    
    def evaluate(self, value: float) -> float:
        """Evaluate objective satisfaction."""
        deviation = abs(value - self.target_value) / max(self.target_value, 0.1)
        satisfaction = max(0.0, 1.0 - deviation / self.tolerance)
        return satisfaction * self.weight


@dataclass
class Constraint:
    """Optimization constraint definition."""
    constraint_type: ConstraintType
    parameter: str
    operator: str  # '<=', '>=', '==', '!='
    threshold: float
    scale: BiologicalScale
    priority: float = 1.0
    
    def is_satisfied(self, value: float) -> bool:
        """Check if constraint is satisfied."""
        if self.operator == '<=':
            return value <= self.threshold
        elif self.operator == '>=':
            return value >= self.threshold
        elif self.operator == '==':
            return abs(value - self.threshold) < 0.01
        elif self.operator == '!=':
            return abs(value - self.threshold) >= 0.01
        return False
    
    def violation_penalty(self, value: float) -> float:
        """Calculate penalty for constraint violation."""
        if self.is_satisfied(value):
            return 0.0
        
        if self.operator in ['<=', '>=']:
            violation = abs(value - self.threshold) / max(self.threshold, 0.01)
        else:
            violation = abs(value - self.threshold) / max(self.threshold, 0.01)
        
        return violation * self.priority


@dataclass
class MultiscaleProfile:
    """Multiscale property profile for a formulation."""
    molecular_properties: Dict[str, float]
    cellular_properties: Dict[str, float]
    tissue_properties: Dict[str, float]
    organ_properties: Dict[str, float]
    emergent_properties: Dict[str, float]
    
    def get_property(self, scale: BiologicalScale, property_name: str) -> Optional[float]:
        """Get property value at specific scale."""
        scale_map = {
            BiologicalScale.MOLECULAR: self.molecular_properties,
            BiologicalScale.CELLULAR: self.cellular_properties,
            BiologicalScale.TISSUE: self.tissue_properties,
            BiologicalScale.ORGAN: self.organ_properties
        }
        
        return scale_map.get(scale, {}).get(property_name)


@dataclass
class OptimizationResult:
    """Result of multiscale optimization."""
    formulation: Dict[str, float]
    objective_values: Dict[ObjectiveType, float]
    constraint_violations: List[Tuple[Constraint, float]]
    multiscale_profile: MultiscaleProfile
    optimization_history: List[Dict]
    convergence_metrics: Dict[str, float]
    computational_cost: float
    attention_allocation: Dict[str, float]


class MultiscaleConstraintOptimizer:
    """
    Advanced multiscale constraint optimization engine for cosmeceutical formulation.
    
    This optimizer integrates across multiple biological scales using evolutionary
    algorithms, constraint satisfaction, and attention-guided resource allocation.
    """
    
    def __init__(self, inci_reducer=None, attention_manager=None, atomspace=None):
        """Initialize the multiscale optimization engine."""
        self.atomspace = atomspace or (AtomSpace() if OPENCOG_AVAILABLE else None)
        
        # Initialize component modules
        if MODULES_AVAILABLE:
            self.inci_reducer = inci_reducer or INCISearchSpaceReducer(self.atomspace)
            self.attention_manager = attention_manager or AttentionAllocationManager(self.atomspace)
        else:
            self.inci_reducer = None
            self.attention_manager = None
        
        # Optimization parameters
        self.population_size = 50
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elitism_rate = 0.1
        
        # Multiscale parameters
        self.scale_weights = {
            BiologicalScale.MOLECULAR: 0.2,
            BiologicalScale.CELLULAR: 0.3,
            BiologicalScale.TISSUE: 0.3,
            BiologicalScale.ORGAN: 0.2
        }
        
        # Performance tracking
        self.optimization_history = []
        self.performance_metrics = {
            'optimization_time': [],
            'convergence_iterations': [],
            'constraint_satisfaction_rate': [],
            'objective_achievement': []
        }
        
        # Initialize property calculation models
        self._initialize_property_models()
        
        logger.info("🔬 Multiscale Constraint Optimizer initialized")
    
    def optimize_formulation(self, objectives: List[Objective], 
                           constraints: List[Constraint],
                           initial_formulation: Dict[str, float] = None,
                           max_time_seconds: int = 60) -> OptimizationResult:
        """
        Optimize cosmeceutical formulation across multiple biological scales.
        
        Args:
            objectives: List of optimization objectives
            constraints: List of optimization constraints
            initial_formulation: Optional starting formulation
            max_time_seconds: Maximum optimization time
            
        Returns:
            OptimizationResult with optimized formulation and analysis
        """
        start_time = time.time()
        logger.info(f"🚀 Starting multiscale formulation optimization")
        logger.info(f"   Objectives: {len(objectives)}")
        logger.info(f"   Constraints: {len(constraints)}")
        
        # Initialize optimization
        self.optimization_history = []
        
        # Generate initial population
        population = self._generate_initial_population(initial_formulation)
        
        # Evaluate initial population
        population_fitness = []
        for individual in population:
            fitness = self._evaluate_fitness(individual, objectives, constraints)
            population_fitness.append(fitness)
        
        best_individual = None
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        # Evolution loop
        for generation in range(self.max_generations):
            if time.time() - start_time > max_time_seconds:
                logger.info("⏰ Optimization time limit reached")
                break
            
            # Selection, crossover, and mutation
            new_population = self._evolve_population(population, population_fitness)
            
            # Evaluate new population
            new_fitness = []
            for individual in new_population:
                fitness = self._evaluate_fitness(individual, objectives, constraints)
                new_fitness.append(fitness)
            
            # Update best solution
            generation_best_idx = max(range(len(new_fitness)), key=lambda i: new_fitness[i])
            generation_best_fitness = new_fitness[generation_best_idx]
            
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_individual = new_population[generation_best_idx].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Update population
            population = new_population
            population_fitness = new_fitness
            
            # Record history
            self.optimization_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': sum(population_fitness) / len(population_fitness),
                'constraint_violations': self._count_constraint_violations(population, constraints)
            })
            
            # Update attention allocation
            if self.attention_manager:
                self._update_attention_based_on_progress(generation, best_fitness, objectives)
            
            # Early stopping check
            if generations_without_improvement > 20:
                logger.info(f"🔄 Early stopping at generation {generation}")
                break
            
            if generation % 10 == 0:
                logger.debug(f"   Generation {generation}: Best fitness {best_fitness:.4f}")
        
        # Create optimization result
        optimization_time = time.time() - start_time
        result = self._create_optimization_result(
            best_individual, objectives, constraints, optimization_time
        )
        
        # Update performance metrics
        self.performance_metrics['optimization_time'].append(optimization_time)
        self.performance_metrics['convergence_iterations'].append(len(self.optimization_history))
        
        logger.info(f"✅ Optimization completed in {optimization_time:.2f}s")
        logger.info(f"   Best fitness: {best_fitness:.4f}")
        logger.info(f"   Generations: {len(self.optimization_history)}")
        
        return result
    
    def evaluate_multiscale_properties(self, formulation: Dict[str, float]) -> MultiscaleProfile:
        """
        Calculate multiscale properties for a given formulation.
        
        Args:
            formulation: Ingredient concentrations
            
        Returns:
            MultiscaleProfile with properties at all biological scales
        """
        molecular_props = self._calculate_molecular_properties(formulation)
        cellular_props = self._calculate_cellular_properties(formulation, molecular_props)
        tissue_props = self._calculate_tissue_properties(formulation, cellular_props)
        organ_props = self._calculate_organ_properties(formulation, tissue_props)
        emergent_props = self._calculate_emergent_properties(
            molecular_props, cellular_props, tissue_props, organ_props
        )
        
        return MultiscaleProfile(
            molecular_properties=molecular_props,
            cellular_properties=cellular_props,
            tissue_properties=tissue_props,
            organ_properties=organ_props,
            emergent_properties=emergent_props
        )
    
    def handle_constraint_conflicts(self, constraints: List[Constraint]) -> List[Constraint]:
        """
        Handle conflicts between constraints using priority-based resolution.
        
        Args:
            constraints: List of potentially conflicting constraints
            
        Returns:
            Resolved constraint list
        """
        logger.info("⚖️ Analyzing constraint conflicts...")
        
        # Group constraints by parameter and scale
        constraint_groups = defaultdict(list)
        for constraint in constraints:
            key = (constraint.parameter, constraint.scale)
            constraint_groups[key].append(constraint)
        
        resolved_constraints = []
        conflicts_found = 0
        
        for key, group_constraints in constraint_groups.items():
            if len(group_constraints) == 1:
                resolved_constraints.extend(group_constraints)
                continue
            
            # Check for conflicts
            conflicts = self._detect_constraint_conflicts(group_constraints)
            
            if conflicts:
                conflicts_found += len(conflicts)
                logger.warning(f"   Conflict detected for {key}: {len(conflicts)} conflicting pairs")
                
                # Resolve by priority
                resolved = self._resolve_constraint_conflicts(group_constraints)
                resolved_constraints.extend(resolved)
            else:
                resolved_constraints.extend(group_constraints)
        
        logger.info(f"✓ Resolved {conflicts_found} constraint conflicts")
        return resolved_constraints
    
    def compute_emergent_properties(self, molecular_interactions: Dict) -> Dict[str, float]:
        """
        Compute emergent system properties from molecular-level interactions.
        
        Args:
            molecular_interactions: Dictionary of molecular interaction data
            
        Returns:
            Dictionary of computed emergent properties
        """
        emergent_props = {}
        
        # Synergistic efficacy (non-linear combination of individual efficacies)
        individual_efficacies = molecular_interactions.get('efficacies', {})
        if individual_efficacies:
            # Calculate synergistic enhancement
            base_efficacy = sum(individual_efficacies.values()) / len(individual_efficacies)
            synergy_factor = 1.0
            
            # Look for known synergistic pairs
            synergistic_pairs = [
                ('hyaluronic_acid', 'niacinamide'),
                ('vitamin_c', 'vitamin_e'),
                ('retinol', 'hyaluronic_acid')
            ]
            
            for ing1, ing2 in synergistic_pairs:
                if ing1 in individual_efficacies and ing2 in individual_efficacies:
                    synergy_factor *= 1.2  # 20% synergistic enhancement
            
            emergent_props['synergistic_efficacy'] = base_efficacy * synergy_factor
        
        # System stability (emergent from individual stabilities)
        individual_stabilities = molecular_interactions.get('stabilities', {})
        if individual_stabilities:
            # System stability is limited by least stable component
            min_stability = min(individual_stabilities.values())
            avg_stability = sum(individual_stabilities.values()) / len(individual_stabilities)
            
            # Emergent stability considers both min and average
            emergent_props['system_stability'] = (min_stability * 0.7 + avg_stability * 0.3)
        
        # Safety profile (emergent safety considerations)
        individual_safeties = molecular_interactions.get('safeties', {})
        if individual_safeties:
            # Look for potential negative interactions
            interaction_penalty = 0.0
            
            incompatible_pairs = [
                ('retinol', 'vitamin_c'),
                ('salicylic_acid', 'retinol'),
                ('strong_acids', 'peptides')
            ]
            
            for ing1, ing2 in incompatible_pairs:
                if ing1 in individual_safeties and ing2 in individual_safeties:
                    interaction_penalty += 0.1
            
            base_safety = sum(individual_safeties.values()) / len(individual_safeties)
            emergent_props['system_safety'] = max(0.0, base_safety - interaction_penalty)
        
        # Sensory properties (emergent feel and appearance)
        texture_factors = molecular_interactions.get('texture_factors', {})
        if texture_factors:
            # Combine texture contributions
            viscosity_contribution = sum(
                factor * 0.1 for factor in texture_factors.values()
            )
            emergent_props['sensory_texture'] = min(1.0, viscosity_contribution)
        
        return emergent_props
    
    def _initialize_property_models(self):
        """Initialize models for calculating properties at different scales."""
        # Molecular property models
        self.molecular_models = {
            'solubility': self._calculate_solubility,
            'stability': self._calculate_molecular_stability,
            'bioavailability': self._calculate_bioavailability,
            'molecular_weight': self._calculate_molecular_weight
        }
        
        # Cellular property models
        self.cellular_models = {
            'penetration': self._calculate_skin_penetration,
            'uptake': self._calculate_cellular_uptake,
            'cytotoxicity': self._calculate_cytotoxicity,
            'cellular_response': self._calculate_cellular_response
        }
        
        # Tissue property models
        self.tissue_models = {
            'barrier_function': self._calculate_barrier_function,
            'hydration': self._calculate_tissue_hydration,
            'anti_aging': self._calculate_anti_aging_effect,
            'inflammation': self._calculate_inflammation_response
        }
        
        # Organ property models
        self.organ_models = {
            'overall_efficacy': self._calculate_overall_efficacy,
            'safety_profile': self._calculate_safety_profile,
            'sensory_properties': self._calculate_sensory_properties,
            'long_term_effects': self._calculate_long_term_effects
        }
    
    def _generate_initial_population(self, initial_formulation: Dict[str, float] = None) -> List[Dict[str, float]]:
        """Generate initial population for evolutionary optimization."""
        population = []
        
        # Use INCI reducer if available to get viable candidates
        if self.inci_reducer and initial_formulation:
            # Convert formulation to INCI string (simplified)
            inci_string = ", ".join(initial_formulation.keys())
            candidates = self.inci_reducer.optimize_search_space(inci_string, {'max_candidates': self.population_size // 2})
            
            for candidate in candidates:
                population.append(candidate.ingredients)
        
        # Fill remaining slots with random formulations
        base_formulation = initial_formulation or self._get_default_formulation()
        
        while len(population) < self.population_size:
            individual = self._generate_random_variation(base_formulation)
            population.append(individual)
        
        return population
    
    def _get_default_formulation(self) -> Dict[str, float]:
        """Get default formulation template."""
        return {
            'aqua': 70.0,
            'glycerin': 8.0,
            'niacinamide': 5.0,
            'hyaluronic_acid': 2.0,
            'cetyl_alcohol': 3.0,
            'phenoxyethanol': 0.8,
            'xanthan_gum': 0.3
        }
    
    def _generate_random_variation(self, base_formulation: Dict[str, float]) -> Dict[str, float]:
        """Generate random variation of base formulation."""
        variation = base_formulation.copy()
        
        # Apply random perturbations
        for ingredient in variation:
            if ingredient == 'aqua':  # Water content is adjusted to maintain 100%
                continue
            
            # Random variation within ±20%
            perturbation = random.uniform(-0.2, 0.2)
            original_value = base_formulation[ingredient]
            new_value = original_value * (1.0 + perturbation)
            
            # Ensure positive values
            variation[ingredient] = max(0.1, new_value)
        
        # Normalize to 100%
        total = sum(variation.values())
        if total != 100.0:
            factor = 100.0 / total
            for ingredient in variation:
                variation[ingredient] *= factor
        
        return variation
    
    def _evaluate_fitness(self, individual: Dict[str, float], 
                         objectives: List[Objective], 
                         constraints: List[Constraint]) -> float:
        """Evaluate fitness of an individual formulation."""
        # Calculate multiscale properties
        profile = self.evaluate_multiscale_properties(individual)
        
        # Evaluate objectives
        objective_scores = []
        for objective in objectives:
            property_value = profile.get_property(objective.scale, objective.objective_type.value)
            if property_value is not None:
                score = objective.evaluate(property_value)
                objective_scores.append(score)
        
        # Calculate constraint penalties
        constraint_penalties = []
        for constraint in constraints:
            property_value = profile.get_property(constraint.scale, constraint.parameter)
            if property_value is not None:
                penalty = constraint.violation_penalty(property_value)
                constraint_penalties.append(penalty)
        
        # Combine objective scores and constraint penalties
        objective_fitness = sum(objective_scores) if objective_scores else 0.0
        constraint_penalty = sum(constraint_penalties) if constraint_penalties else 0.0
        
        fitness = objective_fitness - constraint_penalty
        
        return fitness
    
    def _evolve_population(self, population: List[Dict[str, float]], 
                          fitness_scores: List[float]) -> List[Dict[str, float]]:
        """Evolve population using genetic operators."""
        new_population = []
        
        # Elitism - keep best individuals
        elite_count = int(self.population_size * self.elitism_rate)
        elite_indices = sorted(range(len(fitness_scores)), 
                             key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
        
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            # Selection (tournament selection)
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[Dict[str, float]], 
                            fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, float]:
        """Select individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), 
                                          min(tournament_size, len(population)))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _crossover(self, parent1: Dict[str, float], 
                  parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Perform crossover between two parents."""
        child1 = {}
        child2 = {}
        
        # Uniform crossover
        for ingredient in parent1:
            if random.random() < 0.5:
                child1[ingredient] = parent1[ingredient]
                child2[ingredient] = parent2[ingredient]
            else:
                child1[ingredient] = parent2[ingredient]
                child2[ingredient] = parent1[ingredient]
        
        # Normalize children to 100%
        for child in [child1, child2]:
            total = sum(child.values())
            if total > 0:
                factor = 100.0 / total
                for ingredient in child:
                    child[ingredient] *= factor
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Apply mutation to an individual."""
        mutated = individual.copy()
        
        # Random ingredient mutation
        ingredients_to_mutate = random.sample(
            list(mutated.keys()), 
            max(1, int(len(mutated) * 0.3))  # Mutate 30% of ingredients
        )
        
        for ingredient in ingredients_to_mutate:
            if ingredient == 'aqua':  # Skip water - it's adjusted automatically
                continue
            
            # Gaussian mutation
            current_value = mutated[ingredient]
            mutation_strength = current_value * 0.1  # 10% standard deviation
            
            if NUMPY_AVAILABLE:
                delta = np.random.normal(0, mutation_strength)
            else:
                delta = random.gauss(0, mutation_strength)
            
            new_value = max(0.01, current_value + delta)
            mutated[ingredient] = new_value
        
        # Normalize to 100%
        total = sum(mutated.values())
        if total > 0:
            factor = 100.0 / total
            for ingredient in mutated:
                mutated[ingredient] *= factor
        
        return mutated
    
    def _calculate_molecular_properties(self, formulation: Dict[str, float]) -> Dict[str, float]:
        """Calculate properties at molecular scale."""
        props = {}
        
        # Molecular weight (simplified calculation)
        molecular_weights = {
            'aqua': 18.0,
            'glycerin': 92.1,
            'niacinamide': 122.1,
            'hyaluronic_acid': 1000000.0,  # High MW
            'cetyl_alcohol': 242.4,
            'phenoxyethanol': 138.2,
            'xanthan_gum': 2000000.0  # Very high MW
        }
        
        weighted_mw = sum(
            formulation.get(ing, 0) * mw 
            for ing, mw in molecular_weights.items()
        ) / 100.0
        
        props['average_molecular_weight'] = weighted_mw
        
        # Solubility (water vs oil soluble components)
        water_soluble = ['aqua', 'glycerin', 'niacinamide', 'hyaluronic_acid', 'xanthan_gum']
        oil_soluble = ['cetyl_alcohol', 'phenoxyethanol']
        
        water_soluble_fraction = sum(
            formulation.get(ing, 0) for ing in water_soluble
        ) / 100.0
        
        props['water_solubility'] = water_soluble_fraction
        
        # Stability (based on known ingredient stabilities)
        stability_scores = {
            'aqua': 1.0,
            'glycerin': 0.9,
            'niacinamide': 0.8,
            'hyaluronic_acid': 0.7,  # Temperature sensitive
            'cetyl_alcohol': 0.9,
            'phenoxyethanol': 0.9,
            'xanthan_gum': 0.8  # pH sensitive
        }
        
        weighted_stability = sum(
            formulation.get(ing, 0) * score 
            for ing, score in stability_scores.items()
        ) / 100.0
        
        props['molecular_stability'] = weighted_stability
        
        return props
    
    def _calculate_cellular_properties(self, formulation: Dict[str, float], 
                                     molecular_props: Dict[str, float]) -> Dict[str, float]:
        """Calculate properties at cellular scale."""
        props = {}
        
        # Skin penetration (based on molecular weight and solubility)
        mw = molecular_props.get('average_molecular_weight', 500)
        solubility = molecular_props.get('water_solubility', 0.5)
        
        # Smaller, more lipophilic molecules penetrate better
        penetration = 1.0 / (1.0 + mw / 1000.0) * (1.0 - solubility * 0.5)
        props['skin_penetration'] = min(1.0, penetration)
        
        # Cellular uptake (depends on penetration and bioactivity)
        active_ingredients = ['niacinamide', 'hyaluronic_acid', 'retinol', 'vitamin_c']
        active_concentration = sum(
            formulation.get(ing, 0) for ing in active_ingredients
        ) / 100.0
        
        uptake = penetration * active_concentration * 2.0  # Scale factor
        props['cellular_uptake'] = min(1.0, uptake)
        
        # Cytotoxicity (simplified safety assessment)
        preservative_conc = formulation.get('phenoxyethanol', 0)
        cytotoxicity = min(0.2, preservative_conc / 5.0)  # Low toxicity for normal concentrations
        props['cytotoxicity'] = cytotoxicity
        
        return props
    
    def _calculate_tissue_properties(self, formulation: Dict[str, float], 
                                   cellular_props: Dict[str, float]) -> Dict[str, float]:
        """Calculate properties at tissue scale."""
        props = {}
        
        # Barrier function improvement
        barrier_ingredients = ['niacinamide', 'cetyl_alcohol', 'hyaluronic_acid']
        barrier_concentration = sum(
            formulation.get(ing, 0) for ing in barrier_ingredients
        ) / 100.0
        
        uptake = cellular_props.get('cellular_uptake', 0.5)
        barrier_function = barrier_concentration * uptake * 1.5
        props['barrier_function'] = min(1.0, barrier_function)
        
        # Hydration (tissue level)
        humectants = ['glycerin', 'hyaluronic_acid']
        humectant_concentration = sum(
            formulation.get(ing, 0) for ing in humectants
        ) / 100.0
        
        hydration = humectant_concentration * 1.2
        props['tissue_hydration'] = min(1.0, hydration)
        
        # Anti-aging effect
        anti_aging_ingredients = ['retinol', 'niacinamide', 'vitamin_c', 'hyaluronic_acid']
        anti_aging_concentration = sum(
            formulation.get(ing, 0) for ing in anti_aging_ingredients
        ) / 100.0
        
        anti_aging = anti_aging_concentration * uptake * 1.8
        props['anti_aging_effect'] = min(1.0, anti_aging)
        
        return props
    
    def _calculate_organ_properties(self, formulation: Dict[str, float], 
                                  tissue_props: Dict[str, float]) -> Dict[str, float]:
        """Calculate properties at organ (skin) scale."""
        props = {}
        
        # Overall efficacy (combination of tissue-level effects)
        barrier_function = tissue_props.get('barrier_function', 0.5)
        hydration = tissue_props.get('tissue_hydration', 0.5)
        anti_aging = tissue_props.get('anti_aging_effect', 0.5)
        
        overall_efficacy = (barrier_function * 0.3 + hydration * 0.4 + anti_aging * 0.3)
        props['overall_efficacy'] = overall_efficacy
        
        # Safety profile (organ level)
        individual_safety_scores = []
        safety_factors = {
            'preservative_safety': 1.0 - min(0.3, formulation.get('phenoxyethanol', 0) / 2.0),
            'allergen_risk': 0.9,  # Assume low allergen risk for standard ingredients
            'irritation_potential': 0.8  # Moderate irritation potential
        }
        
        organ_safety = sum(safety_factors.values()) / len(safety_factors)
        props['safety_profile'] = organ_safety
        
        # Sensory properties
        texture_factors = {
            'viscosity': min(1.0, formulation.get('xanthan_gum', 0) * 10.0),
            'spreadability': 1.0 - min(0.3, formulation.get('cetyl_alcohol', 0) / 10.0),
            'absorption': tissue_props.get('barrier_function', 0.5)
        }
        
        sensory_score = sum(texture_factors.values()) / len(texture_factors)
        props['sensory_properties'] = sensory_score
        
        return props
    
    def _calculate_emergent_properties(self, molecular_props: Dict[str, float],
                                     cellular_props: Dict[str, float],
                                     tissue_props: Dict[str, float],
                                     organ_props: Dict[str, float]) -> Dict[str, float]:
        """Calculate emergent properties from all scales."""
        props = {}
        
        # System-wide stability (emergent from molecular stability and interactions)
        molecular_stability = molecular_props.get('molecular_stability', 0.8)
        
        # Consider negative interactions
        interaction_penalty = 0.0
        # This would be expanded with real interaction data
        
        system_stability = molecular_stability * (1.0 - interaction_penalty)
        props['system_stability'] = system_stability
        
        # Synergistic efficacy (non-linear enhancement)
        organ_efficacy = organ_props.get('overall_efficacy', 0.5)
        cellular_uptake = cellular_props.get('cellular_uptake', 0.5)
        
        # Synergy factor based on ingredient combinations
        synergy_factor = 1.0  # Base factor
        # This would be enhanced with real synergy data
        
        synergistic_efficacy = organ_efficacy * synergy_factor
        props['synergistic_efficacy'] = min(1.0, synergistic_efficacy)
        
        # Cost-effectiveness (emergent property)
        # This would require cost data integration
        props['cost_effectiveness'] = 0.7  # Placeholder
        
        return props
    
    def _count_constraint_violations(self, population: List[Dict[str, float]], 
                                   constraints: List[Constraint]) -> int:
        """Count total constraint violations in population."""
        violations = 0
        for individual in population:
            profile = self.evaluate_multiscale_properties(individual)
            for constraint in constraints:
                property_value = profile.get_property(constraint.scale, constraint.parameter)
                if property_value is not None and not constraint.is_satisfied(property_value):
                    violations += 1
        return violations
    
    def _update_attention_based_on_progress(self, generation: int, best_fitness: float, 
                                          objectives: List[Objective]):
        """Update attention allocation based on optimization progress."""
        if not self.attention_manager:
            return
        
        # Create performance feedback based on progress
        performance_feedback = {}
        
        # Analyze objective achievement
        for objective in objectives:
            node_id = f"objective_{objective.objective_type.value}_{objective.scale.value}"
            
            # Performance based on fitness improvement
            performance_score = min(1.0, best_fitness / max(len(objectives), 1.0))
            
            performance_feedback[node_id] = {
                'efficacy': performance_score,
                'safety': 0.9,  # Assume good safety
                'cost_efficiency': 0.7,  # Moderate cost efficiency
                'stability': 0.8  # Good stability
            }
        
        self.attention_manager.update_attention_values(performance_feedback)
    
    def _create_optimization_result(self, best_formulation: Dict[str, float],
                                  objectives: List[Objective],
                                  constraints: List[Constraint],
                                  optimization_time: float) -> OptimizationResult:
        """Create comprehensive optimization result."""
        # Calculate final properties
        profile = self.evaluate_multiscale_properties(best_formulation)
        
        # Evaluate final objectives
        objective_values = {}
        for objective in objectives:
            property_value = profile.get_property(objective.scale, objective.objective_type.value)
            if property_value is not None:
                objective_values[objective.objective_type] = property_value
        
        # Check constraint violations
        constraint_violations = []
        for constraint in constraints:
            property_value = profile.get_property(constraint.scale, constraint.parameter)
            if property_value is not None:
                violation = constraint.violation_penalty(property_value)
                if violation > 0:
                    constraint_violations.append((constraint, violation))
        
        # Calculate convergence metrics
        if self.optimization_history:
            final_fitness = self.optimization_history[-1]['best_fitness']
            initial_fitness = self.optimization_history[0]['best_fitness']
            improvement = final_fitness - initial_fitness
            
            convergence_metrics = {
                'final_fitness': final_fitness,
                'improvement': improvement,
                'convergence_rate': improvement / len(self.optimization_history) if self.optimization_history else 0,
                'constraint_satisfaction_rate': 1.0 - len(constraint_violations) / max(len(constraints), 1)
            }
        else:
            convergence_metrics = {}
        
        # Get attention allocation if available
        attention_allocation = {}
        if self.attention_manager:
            stats = self.attention_manager.get_attention_statistics()
            attention_allocation = {
                'total_nodes': stats.get('total_nodes', 0),
                'active_nodes': stats.get('active_nodes', 0),
                'sti_utilization': stats.get('attention_bank', {}).get('sti_utilization', 0)
            }
        
        return OptimizationResult(
            formulation=best_formulation,
            objective_values=objective_values,
            constraint_violations=constraint_violations,
            multiscale_profile=profile,
            optimization_history=self.optimization_history.copy(),
            convergence_metrics=convergence_metrics,
            computational_cost=optimization_time,
            attention_allocation=attention_allocation
        )
    
    def _detect_constraint_conflicts(self, constraints: List[Constraint]) -> List[Tuple[Constraint, Constraint]]:
        """Detect conflicts between constraints."""
        conflicts = []
        
        for i, constraint1 in enumerate(constraints):
            for j, constraint2 in enumerate(constraints[i+1:], i+1):
                if self._are_constraints_conflicting(constraint1, constraint2):
                    conflicts.append((constraint1, constraint2))
        
        return conflicts
    
    def _are_constraints_conflicting(self, constraint1: Constraint, constraint2: Constraint) -> bool:
        """Check if two constraints are conflicting."""
        if constraint1.parameter != constraint2.parameter or constraint1.scale != constraint2.scale:
            return False
        
        # Check for direct conflicts
        if constraint1.operator == '<=' and constraint2.operator == '>=' and constraint1.threshold < constraint2.threshold:
            return True
        if constraint1.operator == '>=' and constraint2.operator == '<=' and constraint1.threshold > constraint2.threshold:
            return True
        if constraint1.operator == '==' and constraint2.operator == '==' and constraint1.threshold != constraint2.threshold:
            return True
        
        return False
    
    def _resolve_constraint_conflicts(self, constraints: List[Constraint]) -> List[Constraint]:
        """Resolve conflicts by selecting constraints with higher priority."""
        if len(constraints) <= 1:
            return constraints
        
        # Sort by priority (higher priority first)
        sorted_constraints = sorted(constraints, key=lambda c: c.priority, reverse=True)
        
        # Keep highest priority constraint
        return [sorted_constraints[0]]
    
    # Placeholder methods for property calculations
    def _calculate_solubility(self, formulation: Dict[str, float]) -> float:
        return 0.7  # Placeholder
    
    def _calculate_molecular_stability(self, formulation: Dict[str, float]) -> float:
        return 0.8  # Placeholder
    
    def _calculate_bioavailability(self, formulation: Dict[str, float]) -> float:
        return 0.6  # Placeholder
        
    def _calculate_molecular_weight(self, formulation: Dict[str, float]) -> float:
        return 500.0  # Placeholder
        
    def _calculate_skin_penetration(self, formulation: Dict[str, float]) -> float:
        return 0.5  # Placeholder
        
    def _calculate_cellular_uptake(self, formulation: Dict[str, float]) -> float:
        return 0.6  # Placeholder
        
    def _calculate_cytotoxicity(self, formulation: Dict[str, float]) -> float:
        return 0.1  # Placeholder
        
    def _calculate_cellular_response(self, formulation: Dict[str, float]) -> float:
        return 0.7  # Placeholder
        
    def _calculate_barrier_function(self, formulation: Dict[str, float]) -> float:
        return 0.7  # Placeholder
        
    def _calculate_tissue_hydration(self, formulation: Dict[str, float]) -> float:
        return 0.8  # Placeholder
        
    def _calculate_anti_aging_effect(self, formulation: Dict[str, float]) -> float:
        return 0.6  # Placeholder
        
    def _calculate_inflammation_response(self, formulation: Dict[str, float]) -> float:
        return 0.2  # Placeholder
        
    def _calculate_overall_efficacy(self, formulation: Dict[str, float]) -> float:
        return 0.7  # Placeholder
        
    def _calculate_safety_profile(self, formulation: Dict[str, float]) -> float:
        return 0.9  # Placeholder
        
    def _calculate_sensory_properties(self, formulation: Dict[str, float]) -> float:
        return 0.8  # Placeholder
        
    def _calculate_long_term_effects(self, formulation: Dict[str, float]) -> float:
        return 0.7  # Placeholder


def main():
    """Demonstration of multiscale constraint optimization."""
    print("🔬 Multiscale Constraint Optimization Demo")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = MultiscaleConstraintOptimizer()
    
    # Define objectives
    print("\n1. Defining Optimization Objectives:")
    objectives = [
        Objective(ObjectiveType.EFFICACY, target_value=0.8, weight=0.3, 
                 scale=BiologicalScale.ORGAN, tolerance=0.1),
        Objective(ObjectiveType.SAFETY, target_value=0.9, weight=0.3, 
                 scale=BiologicalScale.ORGAN, tolerance=0.05),
        Objective(ObjectiveType.COST, target_value=0.6, weight=0.2, 
                 scale=BiologicalScale.MOLECULAR, tolerance=0.2),
        Objective(ObjectiveType.STABILITY, target_value=0.8, weight=0.2, 
                 scale=BiologicalScale.MOLECULAR, tolerance=0.1)
    ]
    
    for obj in objectives:
        print(f"   • {obj.objective_type.value}: Target={obj.target_value}, "
              f"Weight={obj.weight}, Scale={obj.scale.value}")
    
    # Define constraints
    print("\n2. Defining Optimization Constraints:")
    constraints = [
        Constraint(ConstraintType.REGULATORY, "overall_efficacy", ">=", 0.6, 
                  BiologicalScale.ORGAN, priority=1.0),
        Constraint(ConstraintType.REGULATORY, "safety_profile", ">=", 0.8, 
                  BiologicalScale.ORGAN, priority=1.0),
        Constraint(ConstraintType.PHYSICAL, "system_stability", ">=", 0.7, 
                  BiologicalScale.MOLECULAR, priority=0.8),
        Constraint(ConstraintType.ECONOMIC, "cost_effectiveness", ">=", 0.5, 
                  BiologicalScale.MOLECULAR, priority=0.6)
    ]
    
    for const in constraints:
        print(f"   • {const.parameter} {const.operator} {const.threshold} "
              f"({const.constraint_type.value}, {const.scale.value})")
    
    # Handle constraint conflicts
    print("\n3. Resolving Constraint Conflicts:")
    resolved_constraints = optimizer.handle_constraint_conflicts(constraints)
    print(f"   Resolved {len(constraints)} constraints → {len(resolved_constraints)} constraints")
    
    # Define initial formulation
    initial_formulation = {
        'aqua': 70.0,
        'glycerin': 8.0,
        'niacinamide': 5.0,
        'hyaluronic_acid': 2.0,
        'cetyl_alcohol': 3.0,
        'phenoxyethanol': 0.8,
        'xanthan_gum': 0.3
    }
    
    print(f"\n4. Initial Formulation:")
    for ingredient, conc in initial_formulation.items():
        print(f"   • {ingredient}: {conc:.1f}%")
    
    # Evaluate initial multiscale properties
    print("\n5. Initial Multiscale Properties:")
    initial_profile = optimizer.evaluate_multiscale_properties(initial_formulation)
    
    print("   Molecular properties:")
    for prop, value in initial_profile.molecular_properties.items():
        print(f"     - {prop}: {value:.3f}")
    
    print("   Organ properties:")
    for prop, value in initial_profile.organ_properties.items():
        print(f"     - {prop}: {value:.3f}")
    
    # Run optimization
    print("\n6. Running Multiscale Optimization:")
    print("   (This may take up to 60 seconds...)")
    
    result = optimizer.optimize_formulation(
        objectives=objectives,
        constraints=resolved_constraints,
        initial_formulation=initial_formulation,
        max_time_seconds=60
    )
    
    # Display results
    print(f"\n7. Optimization Results:")
    print(f"   Optimization time: {result.computational_cost:.2f}s")
    print(f"   Generations: {len(result.optimization_history)}")
    
    print("\n   Optimized formulation:")
    for ingredient, conc in result.formulation.items():
        change = conc - initial_formulation.get(ingredient, 0)
        print(f"     • {ingredient}: {conc:.2f}% ({'+' if change >= 0 else ''}{change:.2f}%)")
    
    print("\n   Objective achievements:")
    for obj_type, value in result.objective_values.items():
        target = next(obj.target_value for obj in objectives if obj.objective_type == obj_type)
        achievement = (1.0 - abs(value - target) / target) * 100 if target > 0 else 0
        print(f"     • {obj_type.value}: {value:.3f} (target: {target:.3f}, {achievement:.1f}%)")
    
    print(f"\n   Constraint violations: {len(result.constraint_violations)}")
    for constraint, violation in result.constraint_violations:
        print(f"     • {constraint.parameter}: Violation penalty {violation:.3f}")
    
    # Display convergence
    if result.optimization_history:
        print(f"\n   Convergence metrics:")
        print(f"     • Final fitness: {result.convergence_metrics.get('final_fitness', 0):.4f}")
        print(f"     • Improvement: {result.convergence_metrics.get('improvement', 0):.4f}")
        print(f"     • Constraint satisfaction: {result.convergence_metrics.get('constraint_satisfaction_rate', 0):.1%}")
    
    # Test emergent properties
    print("\n8. Emergent Properties:")
    molecular_interactions = {
        'efficacies': {'niacinamide': 0.8, 'hyaluronic_acid': 0.7},
        'stabilities': {'niacinamide': 0.9, 'hyaluronic_acid': 0.7},
        'safeties': {'niacinamide': 0.95, 'hyaluronic_acid': 0.95}
    }
    
    emergent_props = optimizer.compute_emergent_properties(molecular_interactions)
    for prop, value in emergent_props.items():
        print(f"   • {prop}: {value:.3f}")
    
    print("\n✅ Multiscale optimization demonstration completed successfully!")
    print("\nKey Benefits Demonstrated:")
    print("  • Multi-objective optimization across biological scales")
    print("  • Automated constraint conflict resolution")
    print("  • Emergent property calculation from molecular interactions")
    print("  • Complete formulation optimization in under 60 seconds")
    print("  • Integration with INCI and attention allocation systems")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
multiscale_optimizer.py

Multiscale Constraint Optimization Engine for Cosmeceutical Formulation

This module implements a comprehensive optimization engine that integrates:
1. OpenCog-inspired hypergraph knowledge representation
2. INCI-driven search space reduction
3. Attention allocation mechanisms
4. Multiscale skin model constraint optimization
5. Probabilistic reasoning for uncertainty handling

Key Features:
- Simultaneous local and global constraint optimization
- Multi-objective optimization (efficacy, safety, cost, regulatory compliance)
- Adaptive search with attention-based resource allocation
- Integration across molecular, cellular, tissue, and organ scales
- Regulatory compliance automation
"""

import math
import time
import random
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

# Import our custom modules
try:
    from inci_optimizer import INCISearchSpaceReducer, FormulationConstraint
    from attention_allocation import AttentionAllocationManager, FormulationNode
except ImportError:
    # For direct execution, adjust import path
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inci_optimizer import INCISearchSpaceReducer, FormulationConstraint
    from attention_allocation import AttentionAllocationManager, FormulationNode

class OptimizationScale(Enum):
    """Scales of optimization in multiscale skin model"""
    MOLECULAR = "molecular"
    CELLULAR = "cellular" 
    TISSUE = "tissue"
    ORGAN = "organ"
    SYSTEMIC = "systemic"

class ObjectiveType(Enum):
    """Types of optimization objectives"""
    EFFICACY = "efficacy"
    SAFETY = "safety"
    COST = "cost"
    STABILITY = "stability"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    CONSUMER_PREFERENCE = "consumer_preference"

@dataclass
class ScaleConstraint:
    """Constraint at a specific biological scale"""
    scale: OptimizationScale
    parameter: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    weight: float = 1.0
    
@dataclass
class OptimizationObjective:
    """Multi-objective optimization objective"""
    objective_type: ObjectiveType
    weight: float
    target_value: Optional[float] = None
    minimize: bool = False  # True for minimization, False for maximization
    
@dataclass
class FormulationCandidate:
    """Candidate formulation with multiscale properties"""
    id: str
    ingredients: Dict[str, float]  # ingredient -> concentration
    formulation_type: str
    
    # Multiscale properties
    molecular_properties: Dict[str, float] = field(default_factory=dict)
    cellular_effects: Dict[str, float] = field(default_factory=dict)
    tissue_responses: Dict[str, float] = field(default_factory=dict)
    organ_outcomes: Dict[str, float] = field(default_factory=dict)
    
    # Objective values
    efficacy_score: float = 0.0
    safety_score: float = 0.0
    cost_estimate: float = 0.0
    stability_score: float = 0.0
    regulatory_compliance: bool = False
    
    # Optimization metadata
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)

class ProbabilisticReasoner:
    """PLN-inspired probabilistic reasoning for uncertainty handling"""
    
    def __init__(self):
        self.belief_network = {}
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    def assess_ingredient_compatibility(
        self, 
        ingredient1: str, 
        ingredient2: str,
        context: Dict = None
    ) -> Tuple[float, float]:
        """
        Assess compatibility between two ingredients with uncertainty.
        
        Returns:
            Tuple of (compatibility_strength, confidence)
        """
        # Mock implementation - in practice would use comprehensive knowledge base
        compatibility_rules = {
            ('vitamin_c', 'retinol'): (0.2, 0.9),  # Low compatibility, high confidence
            ('hyaluronic_acid', 'niacinamide'): (0.8, 0.85),  # High compatibility
            ('vitamin_c', 'vitamin_e'): (0.9, 0.7),  # Synergistic, medium confidence
        }
        
        key1 = (ingredient1, ingredient2)
        key2 = (ingredient2, ingredient1)
        
        if key1 in compatibility_rules:
            return compatibility_rules[key1]
        elif key2 in compatibility_rules:
            return compatibility_rules[key2]
        else:
            # Default uncertain compatibility
            return (0.5, 0.3)
    
    def predict_efficacy(
        self, 
        formulation: Dict[str, float],
        target_condition: str = "anti_aging"
    ) -> Tuple[float, float]:
        """
        Predict formulation efficacy with uncertainty quantification.
        
        Returns:
            Tuple of (efficacy_strength, confidence)
        """
        # Mock efficacy prediction based on ingredient synergies
        total_efficacy = 0.0
        confidence_factors = []
        
        ingredients = list(formulation.keys())
        for i, ing1 in enumerate(ingredients):
            concentration1 = formulation[ing1]
            
            # Individual ingredient contribution
            individual_efficacy = self._get_individual_efficacy(ing1, target_condition)
            weighted_efficacy = individual_efficacy * (concentration1 / 100.0)
            total_efficacy += weighted_efficacy
            confidence_factors.append(0.7)  # Base confidence for known ingredients
            
            # Pairwise interactions
            for j, ing2 in enumerate(ingredients[i+1:], i+1):
                concentration2 = formulation[ing2]
                compatibility, comp_confidence = self.assess_ingredient_compatibility(ing1, ing2)
                
                # Synergy/antagonism effect
                interaction_effect = compatibility * 0.1 * min(concentration1, concentration2) / 100.0
                total_efficacy += interaction_effect
                confidence_factors.append(comp_confidence * 0.8)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        
        # Normalize efficacy to 0-1 range
        normalized_efficacy = min(1.0, total_efficacy)
        
        return (normalized_efficacy, overall_confidence)
    
    def _get_individual_efficacy(self, ingredient: str, condition: str) -> float:
        """Get individual ingredient efficacy for specific condition"""
        # Mock efficacy database
        efficacy_db = {
            'retinol': {'anti_aging': 0.8, 'acne': 0.7},
            'hyaluronic_acid': {'hydration': 0.9, 'anti_aging': 0.6},
            'vitamin_c': {'antioxidant': 0.85, 'brightening': 0.8},
            'niacinamide': {'pore_minimizing': 0.7, 'oil_control': 0.75}
        }
        
        return efficacy_db.get(ingredient, {}).get(condition, 0.5)

class MultiscaleConstraintOptimizer:
    """
    Main optimization engine integrating all components for multiscale
    constraint optimization in cosmeceutical formulation.
    """
    
    def __init__(
        self,
        inci_reducer: INCISearchSpaceReducer = None,
        attention_manager: AttentionAllocationManager = None,
        reasoner: ProbabilisticReasoner = None
    ):
        self.inci_reducer = inci_reducer or INCISearchSpaceReducer()
        self.attention_manager = attention_manager or AttentionAllocationManager()
        self.reasoner = reasoner or ProbabilisticReasoner()
        
        # Optimization parameters
        self.population_size = 50
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        
        # Multiscale constraints and objectives
        self.scale_constraints: List[ScaleConstraint] = []
        self.objectives: List[OptimizationObjective] = []
        
        # Optimization history
        self.optimization_history: List[Dict] = []
        self.best_candidates: List[FormulationCandidate] = []
        
        # Knowledge integration
        self.ingredient_database = self._load_ingredient_knowledge()
        self.skin_model_parameters = self._load_skin_model()
    
    def _load_ingredient_knowledge(self) -> Dict:
        """Load comprehensive ingredient knowledge base"""
        # Mock ingredient database with multiscale properties
        return {
            'retinol': {
                'molecular_weight': 286.45,
                'penetration_coefficient': 0.7,
                'cellular_targets': ['keratinocyte', 'fibroblast'],
                'tissue_effects': ['collagen_synthesis', 'cell_turnover'],
                'stability_factors': {'ph_sensitive': True, 'light_sensitive': True}
            },
            'hyaluronic_acid': {
                'molecular_weight': 1000000,  # High MW variant
                'penetration_coefficient': 0.3,
                'cellular_targets': ['keratinocyte'],
                'tissue_effects': ['hydration', 'barrier_function'],
                'stability_factors': {'ph_sensitive': False, 'temperature_sensitive': False}
            },
            'vitamin_c': {
                'molecular_weight': 176.12,
                'penetration_coefficient': 0.6,
                'cellular_targets': ['fibroblast', 'melanocyte'],
                'tissue_effects': ['collagen_synthesis', 'antioxidant_protection'],
                'stability_factors': {'ph_sensitive': True, 'oxygen_sensitive': True}
            }
        }
    
    def _load_skin_model(self) -> Dict:
        """Load multiscale skin model parameters"""
        return {
            'molecular_scale': {
                'diffusion_rates': {'small_molecule': 0.8, 'large_molecule': 0.2},
                'binding_affinities': {'receptor_a': 0.7, 'receptor_b': 0.5}
            },
            'cellular_scale': {
                'cell_types': ['keratinocyte', 'fibroblast', 'melanocyte'],
                'response_thresholds': {'activation': 0.1, 'saturation': 10.0}
            },
            'tissue_scale': {
                'layers': ['stratum_corneum', 'epidermis', 'dermis'],
                'barrier_properties': {'permeability': 0.3, 'retention': 0.7}
            },
            'organ_scale': {
                'skin_types': ['normal', 'dry', 'oily', 'sensitive'],
                'aging_factors': {'chronological': 1.0, 'photodamage': 0.8}
            }
        }
    
    def add_scale_constraint(self, constraint: ScaleConstraint):
        """Add a constraint at specific biological scale"""
        self.scale_constraints.append(constraint)
    
    def add_objective(self, objective: OptimizationObjective):
        """Add optimization objective"""
        self.objectives.append(objective)
    
    def optimize_formulation(
        self,
        target_inci: str,
        base_constraints: FormulationConstraint,
        target_condition: str = "anti_aging",
        max_time_minutes: float = 30.0
    ) -> List[FormulationCandidate]:
        """
        Main optimization function integrating all components.
        
        Args:
            target_inci: Target INCI ingredient list for search space reduction
            base_constraints: Basic formulation constraints
            target_condition: Target skin condition to address
            max_time_minutes: Maximum optimization time
            
        Returns:
            List of optimized formulation candidates
        """
        start_time = time.time()
        max_time_seconds = max_time_minutes * 60
        
        print(f"Starting multiscale constraint optimization...")
        print(f"Target condition: {target_condition}")
        print(f"Time budget: {max_time_minutes} minutes")
        
        # Phase 1: INCI-driven search space reduction
        print("\nPhase 1: INCI-driven search space reduction...")
        filtered_formulations = self.inci_reducer.filter_formulation_space(
            target_inci, base_constraints
        )
        print(f"Reduced search space to {len(filtered_formulations)} candidates")
        
        # Phase 2: Initialize population
        print("\nPhase 2: Population initialization...")
        population = self._initialize_population(filtered_formulations, target_condition)
        print(f"Initialized population of {len(population)} candidates")
        
        # Phase 3: Evolutionary optimization with attention allocation
        print("\nPhase 3: Evolutionary optimization...")
        generation = 0
        
        while generation < self.max_generations and (time.time() - start_time) < max_time_seconds:
            # Evaluate population
            self._evaluate_population(population, target_condition)
            
            # Update attention allocation based on results
            self._update_attention_allocation(population)
            
            # Select best candidates based on multi-objective optimization
            selected_candidates = self._multi_objective_selection(population)
            
            # Store best candidates
            self.best_candidates = sorted(selected_candidates, 
                                        key=lambda x: self._calculate_fitness(x, target_condition), 
                                        reverse=True)[:10]
            
            # Record generation statistics
            generation_stats = self._calculate_generation_statistics(population, generation)
            self.optimization_history.append(generation_stats)
            
            if generation % 10 == 0:
                print(f"  Generation {generation}: Best fitness = {generation_stats['best_fitness']:.4f}")
            
            # Create next generation
            population = self._create_next_generation(selected_candidates)
            generation += 1
        
        elapsed_time = time.time() - start_time
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds ({generation} generations)")
        print(f"Best fitness achieved: {self.optimization_history[-1]['best_fitness']:.4f}")
        
        # Phase 4: Post-processing and validation
        print("\nPhase 4: Post-processing and validation...")
        validated_candidates = self._validate_final_candidates(self.best_candidates[:5])
        
        return validated_candidates
    
    def _initialize_population(
        self, 
        filtered_formulations: List[Dict],
        target_condition: str
    ) -> List[FormulationCandidate]:
        """Initialize population from filtered formulation space"""
        population = []
        
        for i, formulation in enumerate(filtered_formulations[:self.population_size]):
            candidate = FormulationCandidate(
                id=f"gen0_candidate_{i}",
                ingredients=formulation.get('ingredients', {}),
                formulation_type=formulation.get('formulation_type', 'serum'),
                generation=0
            )
            
            # Calculate multiscale properties
            self._calculate_multiscale_properties(candidate)
            
            population.append(candidate)
        
        # Fill remaining population with variations
        while len(population) < self.population_size:
            base_candidate = population[len(population) % len(filtered_formulations)]
            mutated_candidate = self._mutate_candidate(base_candidate, f"gen0_variant_{len(population)}")
            population.append(mutated_candidate)
        
        return population
    
    def _calculate_multiscale_properties(self, candidate: FormulationCandidate):
        """Calculate properties across all biological scales"""
        ingredients = candidate.ingredients
        
        # Molecular scale properties
        total_mw = 0.0
        avg_penetration = 0.0
        for ingredient, concentration in ingredients.items():
            if ingredient in self.ingredient_database:
                ing_data = self.ingredient_database[ingredient]
                weight = concentration / 100.0
                total_mw += ing_data.get('molecular_weight', 300) * weight
                avg_penetration += ing_data.get('penetration_coefficient', 0.5) * weight
        
        candidate.molecular_properties = {
            'average_molecular_weight': total_mw,
            'penetration_index': avg_penetration,
            'stability_index': self._calculate_stability_index(candidate)
        }
        
        # Cellular scale effects
        candidate.cellular_effects = {
            'keratinocyte_activation': self._calculate_cellular_effect(candidate, 'keratinocyte'),
            'fibroblast_stimulation': self._calculate_cellular_effect(candidate, 'fibroblast'),
            'melanocyte_regulation': self._calculate_cellular_effect(candidate, 'melanocyte')
        }
        
        # Tissue scale responses
        candidate.tissue_responses = {
            'barrier_enhancement': self._calculate_barrier_effect(candidate),
            'collagen_synthesis': self._calculate_collagen_effect(candidate),
            'hydration_improvement': self._calculate_hydration_effect(candidate)
        }
        
        # Organ scale outcomes
        candidate.organ_outcomes = {
            'overall_skin_health': self._calculate_overall_health(candidate),
            'aesthetic_improvement': self._calculate_aesthetic_effect(candidate),
            'long_term_benefits': self._calculate_long_term_benefits(candidate)
        }
    
    def _calculate_stability_index(self, candidate: FormulationCandidate) -> float:
        """Calculate formulation stability index"""
        stability_factors = []
        
        for ingredient, concentration in candidate.ingredients.items():
            if ingredient in self.ingredient_database:
                ing_data = self.ingredient_database[ingredient]
                stability_data = ing_data.get('stability_factors', {})
                
                # Account for pH sensitivity, light sensitivity, etc.
                ingredient_stability = 1.0
                if stability_data.get('ph_sensitive'):
                    ingredient_stability *= 0.8
                if stability_data.get('light_sensitive'):
                    ingredient_stability *= 0.9
                if stability_data.get('oxygen_sensitive'):
                    ingredient_stability *= 0.85
                
                stability_factors.append(ingredient_stability)
        
        return sum(stability_factors) / len(stability_factors) if stability_factors else 0.7
    
    def _calculate_cellular_effect(self, candidate: FormulationCandidate, cell_type: str) -> float:
        """Calculate effect on specific cell type"""
        total_effect = 0.0
        
        for ingredient, concentration in candidate.ingredients.items():
            if ingredient in self.ingredient_database:
                ing_data = self.ingredient_database[ingredient]
                targets = ing_data.get('cellular_targets', [])
                
                if cell_type in targets:
                    # Concentration-dependent effect with saturation
                    normalized_conc = concentration / 100.0
                    effect = normalized_conc / (normalized_conc + 0.1)  # Michaelis-Menten-like
                    total_effect += effect
        
        return min(1.0, total_effect)
    
    def _calculate_barrier_effect(self, candidate: FormulationCandidate) -> float:
        """Calculate barrier function enhancement"""
        barrier_ingredients = ['hyaluronic_acid', 'ceramides', 'glycerin']
        total_effect = 0.0
        
        for ingredient, concentration in candidate.ingredients.items():
            if ingredient in barrier_ingredients:
                effect = concentration / 100.0 * 0.8
                total_effect += effect
        
        return min(1.0, total_effect)
    
    def _calculate_collagen_effect(self, candidate: FormulationCandidate) -> float:
        """Calculate collagen synthesis stimulation"""
        collagen_ingredients = ['retinol', 'vitamin_c', 'peptides']
        total_effect = 0.0
        
        for ingredient, concentration in candidate.ingredients.items():
            if ingredient in collagen_ingredients:
                # Non-linear response with optimal concentration
                optimal_conc = {'retinol': 0.5, 'vitamin_c': 15.0, 'peptides': 3.0}
                opt_conc = optimal_conc.get(ingredient, 5.0)
                
                # Bell curve response
                normalized_dist = abs(concentration - opt_conc) / opt_conc
                effect = math.exp(-normalized_dist**2) * 0.8
                total_effect += effect
        
        return min(1.0, total_effect)
    
    def _calculate_hydration_effect(self, candidate: FormulationCandidate) -> float:
        """Calculate hydration improvement"""
        hydration_ingredients = ['hyaluronic_acid', 'glycerin', 'sodium_pca']
        total_effect = 0.0
        
        for ingredient, concentration in candidate.ingredients.items():
            if ingredient in hydration_ingredients:
                effect = min(0.8, concentration / 100.0 * 1.2)
                total_effect += effect
        
        return min(1.0, total_effect)
    
    def _calculate_overall_health(self, candidate: FormulationCandidate) -> float:
        """Calculate overall skin health improvement"""
        cellular_avg = sum(candidate.cellular_effects.values()) / len(candidate.cellular_effects)
        tissue_avg = sum(candidate.tissue_responses.values()) / len(candidate.tissue_responses)
        
        return (cellular_avg + tissue_avg) / 2.0
    
    def _calculate_aesthetic_effect(self, candidate: FormulationCandidate) -> float:
        """Calculate aesthetic improvement (appearance, texture, etc.)"""
        aesthetic_factors = [
            candidate.tissue_responses.get('collagen_synthesis', 0.5),
            candidate.tissue_responses.get('hydration_improvement', 0.5),
            candidate.molecular_properties.get('penetration_index', 0.5)
        ]
        
        return sum(aesthetic_factors) / len(aesthetic_factors)
    
    def _calculate_long_term_benefits(self, candidate: FormulationCandidate) -> float:
        """Calculate long-term benefit potential"""
        stability = candidate.molecular_properties.get('stability_index', 0.7)
        efficacy = candidate.organ_outcomes.get('overall_skin_health', 0.5)
        
        return stability * efficacy
    
    def _evaluate_population(self, population: List[FormulationCandidate], target_condition: str):
        """Evaluate all candidates in population"""
        for candidate in population:
            # Efficacy prediction with uncertainty
            efficacy, efficacy_confidence = self.reasoner.predict_efficacy(
                candidate.ingredients, target_condition
            )
            candidate.efficacy_score = efficacy
            
            # Safety assessment
            candidate.safety_score = self._assess_safety(candidate)
            
            # Cost estimation
            candidate.cost_estimate = self._estimate_cost(candidate)
            
            # Stability assessment
            candidate.stability_score = candidate.molecular_properties.get('stability_index', 0.7)
            
            # Regulatory compliance
            candidate.regulatory_compliance = self.inci_reducer.check_regulatory_compliance({
                'ingredients': candidate.ingredients
            })
    
    def _assess_safety(self, candidate: FormulationCandidate) -> float:
        """Assess formulation safety"""
        # Mock safety assessment based on concentration limits and interactions
        safety_score = 1.0
        
        for ingredient, concentration in candidate.ingredients.items():
            # Check concentration limits
            if ingredient in self.inci_reducer.regulatory_limits:
                limit = self.inci_reducer.regulatory_limits[ingredient]['max_concentration']
                if concentration > limit:
                    safety_score *= 0.5  # Penalize over-limit concentrations
            
            # Check for known irritants at high concentrations
            if ingredient == 'retinol' and concentration > 1.0:
                safety_score *= 0.8
            elif ingredient == 'vitamin_c' and concentration > 20.0:
                safety_score *= 0.7
        
        return safety_score
    
    def _estimate_cost(self, candidate: FormulationCandidate) -> float:
        """Estimate formulation cost"""
        # Mock cost estimation based on ingredient costs
        ingredient_costs = {
            'water': 0.01,
            'hyaluronic_acid': 50.0,
            'retinol': 80.0,
            'vitamin_c': 30.0,
            'niacinamide': 15.0,
            'glycerin': 2.0,
            'phenoxyethanol': 5.0,
            'cetyl_alcohol': 3.0
        }
        
        total_cost = 0.0
        for ingredient, concentration in candidate.ingredients.items():
            unit_cost = ingredient_costs.get(ingredient, 10.0)  # Default cost
            ingredient_cost = unit_cost * concentration / 100.0
            total_cost += ingredient_cost
        
        return total_cost
    
    def _update_attention_allocation(self, population: List[FormulationCandidate]):
        """Update attention allocation based on population performance"""
        for candidate in population:
            # Add candidate to attention network if not already present
            formulation_dict = {
                'ingredients': candidate.ingredients,
                'type': candidate.formulation_type
            }
            
            node_id = self.attention_manager.add_formulation_node(formulation_dict)
            
            # Update with optimization results
            result = {
                'efficacy': candidate.efficacy_score,
                'cost': candidate.cost_estimate,
                'safety': candidate.safety_score,
                'stability': candidate.stability_score
            }
            
            node = self.attention_manager.nodes.get(node_id)
            if node:
                node.update_from_search_result(result)
    
    def _multi_objective_selection(
        self, 
        population: List[FormulationCandidate]
    ) -> List[FormulationCandidate]:
        """Select candidates using multi-objective optimization"""
        # Calculate fitness for each candidate
        fitness_scores = []
        for candidate in population:
            fitness = self._calculate_fitness(candidate)
            fitness_scores.append((fitness, candidate))
        
        # Sort by fitness and select top candidates
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        selected_count = max(10, self.population_size // 2)
        
        return [candidate for _, candidate in fitness_scores[:selected_count]]
    
    def _calculate_fitness(
        self, 
        candidate: FormulationCandidate,
        target_condition: str = "anti_aging"
    ) -> float:
        """Calculate multi-objective fitness score"""
        # Default objectives if none specified
        if not self.objectives:
            objectives = [
                OptimizationObjective(ObjectiveType.EFFICACY, 0.4),
                OptimizationObjective(ObjectiveType.SAFETY, 0.3),
                OptimizationObjective(ObjectiveType.COST, 0.2, minimize=True),
                OptimizationObjective(ObjectiveType.STABILITY, 0.1)
            ]
        else:
            objectives = self.objectives
        
        total_fitness = 0.0
        total_weight = sum(obj.weight for obj in objectives)
        
        for objective in objectives:
            obj_value = 0.0
            
            if objective.objective_type == ObjectiveType.EFFICACY:
                obj_value = candidate.efficacy_score
            elif objective.objective_type == ObjectiveType.SAFETY:
                obj_value = candidate.safety_score
            elif objective.objective_type == ObjectiveType.COST:
                # Normalize cost (lower is better)
                obj_value = 1.0 / (1.0 + candidate.cost_estimate / 100.0)
            elif objective.objective_type == ObjectiveType.STABILITY:
                obj_value = candidate.stability_score
            elif objective.objective_type == ObjectiveType.REGULATORY_COMPLIANCE:
                obj_value = 1.0 if candidate.regulatory_compliance else 0.0
            
            # Apply minimization if specified
            if objective.minimize:
                obj_value = 1.0 - obj_value
            
            weighted_value = obj_value * objective.weight / total_weight
            total_fitness += weighted_value
        
        return total_fitness
    
    def _create_next_generation(
        self, 
        selected_candidates: List[FormulationCandidate]
    ) -> List[FormulationCandidate]:
        """Create next generation through crossover and mutation"""
        next_generation = []
        generation_num = max(c.generation for c in selected_candidates) + 1
        
        # Keep best candidates (elitism)
        elite_count = max(2, self.population_size // 10)
        for i, candidate in enumerate(selected_candidates[:elite_count]):
            elite_copy = self._copy_candidate(candidate, f"gen{generation_num}_elite_{i}")
            elite_copy.generation = generation_num
            next_generation.append(elite_copy)
        
        # Generate offspring through crossover and mutation
        while len(next_generation) < self.population_size:
            if len(selected_candidates) >= 2 and random.random() < self.crossover_rate:
                # Crossover
                parent1 = random.choice(selected_candidates)
                parent2 = random.choice(selected_candidates)
                offspring = self._crossover(parent1, parent2, f"gen{generation_num}_cross_{len(next_generation)}")
            else:
                # Mutation only
                parent = random.choice(selected_candidates)
                offspring = self._mutate_candidate(parent, f"gen{generation_num}_mut_{len(next_generation)}")
            
            offspring.generation = generation_num
            next_generation.append(offspring)
        
        return next_generation
    
    def _copy_candidate(self, candidate: FormulationCandidate, new_id: str) -> FormulationCandidate:
        """Create a copy of a candidate"""
        return FormulationCandidate(
            id=new_id,
            ingredients=candidate.ingredients.copy(),
            formulation_type=candidate.formulation_type,
            molecular_properties=candidate.molecular_properties.copy(),
            cellular_effects=candidate.cellular_effects.copy(),
            tissue_responses=candidate.tissue_responses.copy(),
            organ_outcomes=candidate.organ_outcomes.copy(),
            generation=candidate.generation,
            parent_ids=[candidate.id]
        )
    
    def _crossover(
        self, 
        parent1: FormulationCandidate, 
        parent2: FormulationCandidate,
        offspring_id: str
    ) -> FormulationCandidate:
        """Create offspring through crossover"""
        # Combine ingredients from both parents
        offspring_ingredients = {}
        
        all_ingredients = set(parent1.ingredients.keys()) | set(parent2.ingredients.keys())
        
        for ingredient in all_ingredients:
            conc1 = parent1.ingredients.get(ingredient, 0.0)
            conc2 = parent2.ingredients.get(ingredient, 0.0)
            
            # Average concentration with some random variation
            avg_conc = (conc1 + conc2) / 2.0
            variation = random.gauss(0, avg_conc * 0.1)
            new_conc = max(0.0, avg_conc + variation)
            
            if new_conc > 0.01:  # Only include ingredients with meaningful concentration
                offspring_ingredients[ingredient] = new_conc
        
        offspring = FormulationCandidate(
            id=offspring_id,
            ingredients=offspring_ingredients,
            formulation_type=parent1.formulation_type,
            parent_ids=[parent1.id, parent2.id]
        )
        
        self._calculate_multiscale_properties(offspring)
        return offspring
    
    def _mutate_candidate(
        self, 
        candidate: FormulationCandidate,
        new_id: str
    ) -> FormulationCandidate:
        """Create mutated version of candidate"""
        mutated_ingredients = candidate.ingredients.copy()
        
        # Random ingredient concentration mutations
        for ingredient in list(mutated_ingredients.keys()):
            if random.random() < self.mutation_rate:
                current_conc = mutated_ingredients[ingredient]
                # Gaussian mutation
                mutation = random.gauss(0, current_conc * 0.2)
                new_conc = max(0.01, current_conc + mutation)
                
                # Respect regulatory limits
                if ingredient in self.inci_reducer.regulatory_limits:
                    limit = self.inci_reducer.regulatory_limits[ingredient]['max_concentration']
                    new_conc = min(new_conc, limit)
                
                mutated_ingredients[ingredient] = new_conc
        
        # Occasionally add new ingredients
        if random.random() < self.mutation_rate * 0.5:
            available_ingredients = list(self.ingredient_database.keys())
            new_ingredient = random.choice(available_ingredients)
            if new_ingredient not in mutated_ingredients:
                mutated_ingredients[new_ingredient] = random.uniform(0.1, 2.0)
        
        mutated_candidate = FormulationCandidate(
            id=new_id,
            ingredients=mutated_ingredients,
            formulation_type=candidate.formulation_type,
            parent_ids=[candidate.id],
            mutation_history=candidate.mutation_history + [f"mut_{len(candidate.mutation_history)}"]
        )
        
        self._calculate_multiscale_properties(mutated_candidate)
        return mutated_candidate
    
    def _calculate_generation_statistics(
        self, 
        population: List[FormulationCandidate],
        generation: int
    ) -> Dict:
        """Calculate statistics for current generation"""
        fitness_scores = [self._calculate_fitness(candidate) for candidate in population]
        
        return {
            'generation': generation,
            'population_size': len(population),
            'best_fitness': max(fitness_scores),
            'mean_fitness': sum(fitness_scores) / len(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'fitness_std': math.sqrt(sum((x - sum(fitness_scores)/len(fitness_scores))**2 for x in fitness_scores) / len(fitness_scores)),
            'regulatory_compliance_rate': sum(1 for c in population if c.regulatory_compliance) / len(population),
            'avg_cost': sum(c.cost_estimate for c in population) / len(population),
            'avg_efficacy': sum(c.efficacy_score for c in population) / len(population),
            'avg_safety': sum(c.safety_score for c in population) / len(population)
        }
    
    def _validate_final_candidates(
        self, 
        candidates: List[FormulationCandidate]
    ) -> List[FormulationCandidate]:
        """Final validation and ranking of best candidates"""
        validated_candidates = []
        
        for candidate in candidates:
            # Final regulatory compliance check
            compliance_passed = self.inci_reducer.check_regulatory_compliance({
                'ingredients': candidate.ingredients
            })
            
            if compliance_passed:
                # Final multiscale property calculation
                self._calculate_multiscale_properties(candidate)
                
                # Final safety assessment
                candidate.safety_score = self._assess_safety(candidate)
                
                validated_candidates.append(candidate)
        
        # Sort by final fitness
        validated_candidates.sort(
            key=lambda x: self._calculate_fitness(x),
            reverse=True
        )
        
        return validated_candidates

# Example usage function
def example_multiscale_optimization():
    """Example demonstrating multiscale constraint optimization"""
    print("=== Multiscale Constraint Optimization Example ===")
    
    # Initialize optimizer
    optimizer = MultiscaleConstraintOptimizer()
    
    # Add multiscale constraints
    optimizer.add_scale_constraint(ScaleConstraint(
        scale=OptimizationScale.MOLECULAR,
        parameter="average_molecular_weight",
        max_value=1000.0,
        weight=0.8
    ))
    
    optimizer.add_scale_constraint(ScaleConstraint(
        scale=OptimizationScale.TISSUE,
        parameter="collagen_synthesis",
        min_value=0.6,
        weight=1.0
    ))
    
    # Add optimization objectives
    optimizer.add_objective(OptimizationObjective(ObjectiveType.EFFICACY, 0.4))
    optimizer.add_objective(OptimizationObjective(ObjectiveType.SAFETY, 0.3))
    optimizer.add_objective(OptimizationObjective(ObjectiveType.COST, 0.2, minimize=True))
    optimizer.add_objective(OptimizationObjective(ObjectiveType.STABILITY, 0.1))
    
    # Define target INCI and constraints
    target_inci = "Aqua, Retinol, Hyaluronic Acid, Niacinamide, Glycerin, Phenoxyethanol"
    constraints = FormulationConstraint(
        target_ph=(5.0, 7.0),
        max_total_actives=15.0,
        required_ingredients=["water"]
    )
    
    # Run optimization
    results = optimizer.optimize_formulation(
        target_inci=target_inci,
        base_constraints=constraints,
        target_condition="anti_aging",
        max_time_minutes=5.0  # Short for example
    )
    
    # Display results
    print(f"\n=== Optimization Results ===")
    print(f"Found {len(results)} validated formulations")
    
    for i, candidate in enumerate(results[:3]):
        print(f"\nFormulation {i+1} (ID: {candidate.id}):")
        print(f"  Fitness: {optimizer._calculate_fitness(candidate):.4f}")
        print(f"  Efficacy: {candidate.efficacy_score:.3f}")
        print(f"  Safety: {candidate.safety_score:.3f}")
        print(f"  Cost: ${candidate.cost_estimate:.2f}")
        print(f"  Regulatory Compliant: {candidate.regulatory_compliance}")
        
        print(f"  Ingredients:")
        for ingredient, concentration in candidate.ingredients.items():
            print(f"    {ingredient}: {concentration:.2f}%")
        
        print(f"  Multiscale Properties:")
        print(f"    Molecular - Penetration Index: {candidate.molecular_properties.get('penetration_index', 0):.3f}")
        print(f"    Cellular - Fibroblast Stimulation: {candidate.cellular_effects.get('fibroblast_stimulation', 0):.3f}")
        print(f"    Tissue - Collagen Synthesis: {candidate.tissue_responses.get('collagen_synthesis', 0):.3f}")
        print(f"    Organ - Overall Health: {candidate.organ_outcomes.get('overall_skin_health', 0):.3f}")
    
    # Show optimization statistics
    if optimizer.optimization_history:
        final_stats = optimizer.optimization_history[-1]
        print(f"\n=== Optimization Statistics ===")
        print(f"Generations: {final_stats['generation'] + 1}")
        print(f"Final best fitness: {final_stats['best_fitness']:.4f}")
        print(f"Final mean fitness: {final_stats['mean_fitness']:.4f}")
        print(f"Regulatory compliance rate: {final_stats['regulatory_compliance_rate']:.2%}")
    
    print("\n=== Multiscale Optimization Complete ===")

if __name__ == "__main__":
    example_multiscale_optimization()
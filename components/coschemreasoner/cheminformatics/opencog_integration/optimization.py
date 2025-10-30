"""
MOSES-Inspired Optimization for Cosmeceutical Formulation

This module implements MOSES (Meta-Optimizing Semantic Evolutionary Search) inspired
algorithms for multiscale constraint optimization in cosmeceutical formulation.

Features:
- Evolutionary formulation optimization
- Semantic-aware genetic operations
- Multi-objective optimization for clinical effectiveness
- Population-based search with diversity maintenance
- Adaptive parameter tuning
"""

from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
from collections import defaultdict
import math
from abc import ABC, abstractmethod

from .atomspace import CosmeceuticalAtomSpace, Atom, AtomType
from .reasoning import IngredientReasoningEngine, TruthValue


class OptimizationObjective(Enum):
    """Optimization objectives for cosmeceutical formulation"""
    CLINICAL_EFFECTIVENESS = "clinical_effectiveness"
    COST_MINIMIZATION = "cost_minimization"
    STABILITY_MAXIMIZATION = "stability_maximization"
    SAFETY_MAXIMIZATION = "safety_maximization"
    SYNERGY_MAXIMIZATION = "synergy_maximization"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


@dataclass
class FormulationGenome:
    """Represents a formulation as a genome for evolutionary optimization"""
    ingredients: Dict[str, float] = field(default_factory=dict)  # ingredient_name -> concentration
    properties: Dict[str, Any] = field(default_factory=dict)
    fitness_scores: Dict[OptimizationObjective, float] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    genome_id: str = field(default_factory=lambda: f"genome_{random.randint(1000, 9999)}")
    
    @property
    def total_concentration(self) -> float:
        """Total concentration of all ingredients"""
        return sum(self.ingredients.values())
    
    @property
    def ingredient_count(self) -> int:
        """Number of ingredients in formulation"""
        return len(self.ingredients)
    
    def normalize_concentrations(self, target_total: float = 100.0):
        """Normalize concentrations to sum to target total"""
        current_total = self.total_concentration
        if current_total > 0:
            scale_factor = target_total / current_total
            self.ingredients = {name: conc * scale_factor 
                              for name, conc in self.ingredients.items()}
    
    def clone(self) -> 'FormulationGenome':
        """Create a copy of this genome"""
        return FormulationGenome(
            ingredients=self.ingredients.copy(),
            properties=self.properties.copy(),
            fitness_scores=self.fitness_scores.copy(),
            generation=self.generation,
            parent_ids=self.parent_ids.copy(),
            genome_id=f"genome_{random.randint(1000, 9999)}"
        )


class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation"""
    
    @abstractmethod
    def evaluate(self, genome: FormulationGenome, 
                atomspace: CosmeceuticalAtomSpace) -> float:
        """Evaluate fitness score for a genome"""
        pass


class ClinicalEffectivenessFitness(FitnessEvaluator):
    """Evaluates clinical effectiveness based on ingredient synergies and concentrations"""
    
    def evaluate(self, genome: FormulationGenome, 
                atomspace: CosmeceuticalAtomSpace) -> float:
        effectiveness_score = 0.0
        ingredient_atoms = []
        
        # Get ingredient atoms
        for ingredient_name in genome.ingredients:
            atom = atomspace.get_atom_by_name(ingredient_name)
            if atom:
                ingredient_atoms.append(atom)
        
        if not ingredient_atoms:
            return 0.0
        
        # Evaluate individual ingredient effectiveness
        for ingredient_name, concentration in genome.ingredients.items():
            atom = atomspace.get_atom_by_name(ingredient_name)
            if atom and atom.atom_type == AtomType.INGREDIENT_NODE:
                # Base effectiveness from concentration (with diminishing returns)
                if "concentration_range" in atom.properties:
                    min_conc, max_conc = atom.properties["concentration_range"]
                    normalized_conc = min(1.0, max(0.0, 
                        (concentration - min_conc) / (max_conc - min_conc)))
                    effectiveness_score += math.sqrt(normalized_conc) * 10.0
                else:
                    # Default effectiveness scoring
                    effectiveness_score += min(10.0, concentration * 0.5)
        
        # Evaluate synergistic effects
        synergy_bonus = 0.0
        for i, atom1 in enumerate(ingredient_atoms):
            for atom2 in ingredient_atoms[i+1:]:
                synergies = atomspace.get_ingredient_synergies(atom1.name)
                for synergy_partner, strength in synergies:
                    if synergy_partner == atom2.name:
                        # Synergy bonus based on concentrations and strength
                        conc1 = genome.ingredients.get(atom1.name, 0.0)
                        conc2 = genome.ingredients.get(atom2.name, 0.0)
                        synergy_effect = strength * math.sqrt(conc1 * conc2) * 0.1
                        synergy_bonus += synergy_effect
        
        return effectiveness_score + synergy_bonus


class SafetyFitness(FitnessEvaluator):
    """Evaluates safety based on ingredient interactions and concentration limits"""
    
    def evaluate(self, genome: FormulationGenome, 
                atomspace: CosmeceuticalAtomSpace) -> float:
        safety_score = 100.0  # Start with perfect safety
        
        # Check concentration limits
        for ingredient_name, concentration in genome.ingredients.items():
            atom = atomspace.get_atom_by_name(ingredient_name)
            if atom and "max_concentration" in atom.properties:
                max_allowed = atom.properties["max_concentration"]
                if concentration > max_allowed:
                    # Penalty for exceeding limits
                    excess_penalty = (concentration - max_allowed) / max_allowed * 20.0
                    safety_score -= excess_penalty
        
        # Check for incompatible ingredient pairs
        ingredient_names = list(genome.ingredients.keys())
        for i, ingredient1 in enumerate(ingredient_names):
            for ingredient2 in ingredient_names[i+1:]:
                compatibility = atomspace.get_ingredient_compatibility(ingredient1, ingredient2)
                if compatibility is not None and compatibility < 0.3:
                    # Penalty for incompatible ingredients
                    incompatibility_penalty = (0.3 - compatibility) * 15.0
                    safety_score -= incompatibility_penalty
        
        return max(0.0, safety_score)


class CostFitness(FitnessEvaluator):
    """Evaluates cost efficiency (lower cost = higher fitness)"""
    
    def evaluate(self, genome: FormulationGenome, 
                atomspace: CosmeceuticalAtomSpace) -> float:
        total_cost = 0.0
        
        for ingredient_name, concentration in genome.ingredients.items():
            atom = atomspace.get_atom_by_name(ingredient_name)
            if atom and "cost_per_kg" in atom.properties:
                cost_per_kg = atom.properties["cost_per_kg"]
                ingredient_cost = (concentration / 100.0) * cost_per_kg * 0.1  # Cost per 100g batch
                total_cost += ingredient_cost
        
        # Return inverse cost as fitness (higher fitness for lower cost)
        if total_cost > 0:
            return 100.0 / (1.0 + total_cost)
        else:
            return 100.0


class GeneticOperator(ABC):
    """Abstract base class for genetic operators"""
    
    @abstractmethod
    def apply(self, parents: List[FormulationGenome], 
             atomspace: CosmeceuticalAtomSpace) -> List[FormulationGenome]:
        """Apply genetic operator to create offspring"""
        pass


class CrossoverOperator(GeneticOperator):
    """Crossover operator for combining formulations"""
    
    def __init__(self, crossover_rate: float = 0.8):
        self.crossover_rate = crossover_rate
    
    def apply(self, parents: List[FormulationGenome], 
             atomspace: CosmeceuticalAtomSpace) -> List[FormulationGenome]:
        if len(parents) < 2 or random.random() > self.crossover_rate:
            return [parent.clone() for parent in parents]
        
        parent1, parent2 = parents[0], parents[1]
        offspring = []
        
        # Semantic crossover: combine ingredients based on functional similarity
        child1_ingredients = {}
        child2_ingredients = {}
        
        all_ingredients = set(parent1.ingredients.keys()) | set(parent2.ingredients.keys())
        
        for ingredient in all_ingredients:
            conc1 = parent1.ingredients.get(ingredient, 0.0)
            conc2 = parent2.ingredients.get(ingredient, 0.0)
            
            # Blend concentrations with some randomness
            blend_factor = random.uniform(0.3, 0.7)
            
            child1_conc = blend_factor * conc1 + (1 - blend_factor) * conc2
            child2_conc = (1 - blend_factor) * conc1 + blend_factor * conc2
            
            if child1_conc > 0.01:  # Minimum threshold
                child1_ingredients[ingredient] = child1_conc
            if child2_conc > 0.01:
                child2_ingredients[ingredient] = child2_conc
        
        # Create offspring
        child1 = FormulationGenome(
            ingredients=child1_ingredients,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )
        child2 = FormulationGenome(
            ingredients=child2_ingredients,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )
        
        # Normalize concentrations
        child1.normalize_concentrations()
        child2.normalize_concentrations()
        
        offspring.extend([child1, child2])
        return offspring


class MutationOperator(GeneticOperator):
    """Mutation operator for introducing variations"""
    
    def __init__(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
    
    def apply(self, parents: List[FormulationGenome], 
             atomspace: CosmeceuticalAtomSpace) -> List[FormulationGenome]:
        offspring = []
        
        for parent in parents:
            if random.random() > self.mutation_rate:
                offspring.append(parent.clone())
                continue
            
            mutated = parent.clone()
            mutated.generation += 1
            mutated.parent_ids = [parent.genome_id]
            
            # Concentration mutation
            for ingredient in list(mutated.ingredients.keys()):
                if random.random() < 0.3:  # 30% chance to mutate each ingredient
                    current_conc = mutated.ingredients[ingredient]
                    
                    # Get valid concentration range
                    atom = atomspace.get_atom_by_name(ingredient)
                    if atom and "concentration_range" in atom.properties:
                        min_conc, max_conc = atom.properties["concentration_range"]
                    else:
                        min_conc, max_conc = 0.01, 20.0
                    
                    # Apply gaussian mutation
                    mutation_delta = random.gauss(0, self.mutation_strength * current_conc)
                    new_conc = current_conc + mutation_delta
                    
                    # Clamp to valid range
                    new_conc = max(min_conc, min(max_conc, new_conc))
                    
                    # Remove ingredient if concentration becomes too low
                    if new_conc < 0.01:
                        del mutated.ingredients[ingredient]
                    else:
                        mutated.ingredients[ingredient] = new_conc
            
            # Ingredient addition/removal mutation
            if random.random() < 0.2:  # 20% chance to add/remove ingredient
                available_ingredients = [
                    atom.name for atom in atomspace.get_atoms_by_type(AtomType.INGREDIENT_NODE)
                    if atom.name not in mutated.ingredients
                ]
                
                if available_ingredients and random.random() < 0.5:
                    # Add new ingredient
                    new_ingredient = random.choice(available_ingredients)
                    atom = atomspace.get_atom_by_name(new_ingredient)
                    if atom and "concentration_range" in atom.properties:
                        min_conc, max_conc = atom.properties["concentration_range"]
                        initial_conc = random.uniform(min_conc, min_conc + (max_conc - min_conc) * 0.3)
                        mutated.ingredients[new_ingredient] = initial_conc
                
                elif mutated.ingredients and random.random() < 0.3:
                    # Remove random ingredient (but keep at least 2 ingredients)
                    if len(mutated.ingredients) > 2:
                        ingredient_to_remove = random.choice(list(mutated.ingredients.keys()))
                        del mutated.ingredients[ingredient_to_remove]
            
            # Normalize concentrations
            mutated.normalize_concentrations()
            offspring.append(mutated)
        
        return offspring


class MultiscaleOptimizer:
    """
    MOSES-inspired evolutionary optimizer for cosmeceutical formulation.
    
    This class implements multi-objective evolutionary optimization with semantic
    awareness and adaptive parameter tuning for complex formulation problems.
    """
    
    def __init__(self, atomspace: CosmeceuticalAtomSpace, 
                 reasoning_engine: IngredientReasoningEngine):
        self.atomspace = atomspace
        self.reasoning_engine = reasoning_engine
        
        # Optimization parameters
        self.population_size = 50
        self.max_generations = 100
        self.elite_size = 5
        self.tournament_size = 3
        
        # Fitness evaluators
        self.fitness_evaluators = {
            OptimizationObjective.CLINICAL_EFFECTIVENESS: ClinicalEffectivenessFitness(),
            OptimizationObjective.SAFETY_MAXIMIZATION: SafetyFitness(),
            OptimizationObjective.COST_MINIMIZATION: CostFitness()
        }
        
        # Genetic operators
        self.genetic_operators = [
            CrossoverOperator(crossover_rate=0.8),
            MutationOperator(mutation_rate=0.2, mutation_strength=0.1)
        ]
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_solutions: List[FormulationGenome] = []
    
    def initialize_population(self, seed_formulations: Optional[List[Dict[str, float]]] = None,
                            target_ingredients: Optional[List[str]] = None) -> List[FormulationGenome]:
        """Initialize population with random or seeded formulations"""
        population = []
        
        # Available ingredients
        ingredient_atoms = self.atomspace.get_atoms_by_type(AtomType.INGREDIENT_NODE)
        available_ingredients = [atom.name for atom in ingredient_atoms]
        
        if target_ingredients:
            available_ingredients = [ing for ing in available_ingredients if ing in target_ingredients]
        
        # Add seed formulations if provided
        if seed_formulations:
            for seed in seed_formulations[:self.population_size // 2]:
                genome = FormulationGenome(ingredients=seed.copy())
                genome.normalize_concentrations()
                population.append(genome)
        
        # Generate random formulations to fill population
        while len(population) < self.population_size:
            # Random number of ingredients (3-8)
            num_ingredients = random.randint(3, min(8, len(available_ingredients)))
            selected_ingredients = random.sample(available_ingredients, num_ingredients)
            
            ingredients = {}
            for ingredient_name in selected_ingredients:
                atom = self.atomspace.get_atom_by_name(ingredient_name)
                if atom and "concentration_range" in atom.properties:
                    min_conc, max_conc = atom.properties["concentration_range"]
                    concentration = random.uniform(min_conc, max_conc)
                else:
                    concentration = random.uniform(0.1, 10.0)
                
                ingredients[ingredient_name] = concentration
            
            genome = FormulationGenome(ingredients=ingredients)
            genome.normalize_concentrations()
            population.append(genome)
        
        return population
    
    def evaluate_fitness(self, genome: FormulationGenome, 
                        objectives: List[OptimizationObjective]) -> Dict[OptimizationObjective, float]:
        """Evaluate fitness for multiple objectives"""
        fitness_scores = {}
        
        for objective in objectives:
            if objective in self.fitness_evaluators:
                score = self.fitness_evaluators[objective].evaluate(genome, self.atomspace)
                fitness_scores[objective] = score
            else:
                fitness_scores[objective] = 0.0
        
        genome.fitness_scores = fitness_scores
        return fitness_scores
    
    def tournament_selection(self, population: List[FormulationGenome], 
                           objective: OptimizationObjective) -> FormulationGenome:
        """Tournament selection for a specific objective"""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        best = tournament[0]
        best_fitness = best.fitness_scores.get(objective, 0.0)
        
        for individual in tournament[1:]:
            fitness = individual.fitness_scores.get(objective, 0.0)
            if fitness > best_fitness:
                best = individual
                best_fitness = fitness
        
        return best
    
    def pareto_ranking(self, population: List[FormulationGenome], 
                      objectives: List[OptimizationObjective]) -> List[List[FormulationGenome]]:
        """Pareto ranking for multi-objective optimization"""
        fronts = []
        remaining = population.copy()
        
        while remaining:
            current_front = []
            
            for individual in remaining:
                is_dominated = False
                
                for other in remaining:
                    if other == individual:
                        continue
                    
                    # Check if 'other' dominates 'individual'
                    dominates = True
                    for objective in objectives:
                        ind_score = individual.fitness_scores.get(objective, 0.0)
                        other_score = other.fitness_scores.get(objective, 0.0)
                        
                        if ind_score > other_score:
                            dominates = False
                            break
                    
                    if dominates:
                        # Check if 'other' is actually better in at least one objective
                        better_in_one = False
                        for objective in objectives:
                            ind_score = individual.fitness_scores.get(objective, 0.0)
                            other_score = other.fitness_scores.get(objective, 0.0)
                            if other_score > ind_score:
                                better_in_one = True
                                break
                        
                        if better_in_one:
                            is_dominated = True
                            break
                
                if not is_dominated:
                    current_front.append(individual)
            
            if current_front:
                fronts.append(current_front)
                for individual in current_front:
                    remaining.remove(individual)
            else:
                # Avoid infinite loop
                break
        
        return fronts
    
    def optimize(self, objectives: List[OptimizationObjective],
                seed_formulations: Optional[List[Dict[str, float]]] = None,
                target_ingredients: Optional[List[str]] = None,
                constraints: Optional[Dict[str, Any]] = None) -> List[FormulationGenome]:
        """
        Run multi-objective evolutionary optimization.
        
        Returns list of Pareto-optimal solutions.
        """
        # Initialize population
        population = self.initialize_population(seed_formulations, target_ingredients)
        
        # Evaluate initial population
        for genome in population:
            self.evaluate_fitness(genome, objectives)
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Create offspring
            offspring = []
            
            while len(offspring) < self.population_size:
                # Select parents using tournament selection
                parents = []
                for objective in objectives:
                    parent = self.tournament_selection(population, objective)
                    parents.append(parent)
                
                # Apply genetic operators
                for operator in self.genetic_operators:
                    new_offspring = operator.apply(parents[:2], self.atomspace)
                    offspring.extend(new_offspring)
            
            # Limit offspring size
            offspring = offspring[:self.population_size]
            
            # Evaluate offspring
            for genome in offspring:
                genome.generation = generation + 1
                self.evaluate_fitness(genome, objectives)
            
            # Combine population and offspring
            combined_population = population + offspring
            
            # Apply constraints if specified
            if constraints:
                combined_population = self._apply_constraints(combined_population, constraints)
            
            # Select next generation using Pareto ranking
            fronts = self.pareto_ranking(combined_population, objectives)
            
            next_population = []
            for front in fronts:
                if len(next_population) + len(front) <= self.population_size:
                    next_population.extend(front)
                else:
                    # Select best individuals from current front
                    remaining_slots = self.population_size - len(next_population)
                    # Sort by crowding distance or first objective
                    front.sort(key=lambda x: x.fitness_scores.get(objectives[0], 0.0), reverse=True)
                    next_population.extend(front[:remaining_slots])
                    break
            
            population = next_population
            
            # Store best solutions
            if fronts:
                self.best_solutions.extend(fronts[0])
            
            # Log generation statistics
            avg_fitness = {
                obj.value: np.mean([genome.fitness_scores.get(obj, 0.0) for genome in population])
                for obj in objectives
            }
            
            self.optimization_history.append({
                "generation": generation,
                "population_size": len(population),
                "average_fitness": avg_fitness,
                "pareto_front_size": len(fronts[0]) if fronts else 0
            })
        
        # Return Pareto front
        final_fronts = self.pareto_ranking(population, objectives)
        return final_fronts[0] if final_fronts else []
    
    def _apply_constraints(self, population: List[FormulationGenome], 
                          constraints: Dict[str, Any]) -> List[FormulationGenome]:
        """Apply hard constraints to filter population"""
        filtered_population = []
        
        for genome in population:
            satisfies_constraints = True
            
            # Check total concentration constraint
            if "max_total_concentration" in constraints:
                max_total = constraints["max_total_concentration"]
                if genome.total_concentration > max_total:
                    satisfies_constraints = False
            
            # Check ingredient count constraint
            if "max_ingredients" in constraints:
                max_ingredients = constraints["max_ingredients"]
                if genome.ingredient_count > max_ingredients:
                    satisfies_constraints = False
            
            # Check required ingredients
            if "required_ingredients" in constraints:
                required = set(constraints["required_ingredients"])
                if not required.issubset(set(genome.ingredients.keys())):
                    satisfies_constraints = False
            
            # Check prohibited ingredients
            if "prohibited_ingredients" in constraints:
                prohibited = set(constraints["prohibited_ingredients"])
                if prohibited.intersection(set(genome.ingredients.keys())):
                    satisfies_constraints = False
            
            if satisfies_constraints:
                filtered_population.append(genome)
        
        return filtered_population
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about the optimization process"""
        if not self.optimization_history:
            return {"status": "not_run"}
        
        final_stats = self.optimization_history[-1]
        
        return {
            "total_generations": len(self.optimization_history),
            "final_population_size": final_stats["population_size"],
            "final_pareto_front_size": final_stats["pareto_front_size"],
            "best_solutions_found": len(self.best_solutions),
            "objectives_optimized": list(final_stats["average_fitness"].keys()),
            "convergence_trend": [
                stats["average_fitness"] for stats in self.optimization_history[-10:]
            ]
        }
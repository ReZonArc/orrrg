#!/usr/bin/env python3
"""
hypergredient_evolution.py

🧬 Hypergredient Evolutionary Optimization System
Advanced Machine Learning and Evolutionary Algorithms for Formulation Improvement

This module implements:
1. Evolutionary formulation improvement algorithms
2. Machine learning integration for performance prediction
3. Real-time formulation adaptation based on feedback
4. Genetic algorithm optimization for hypergredient combinations
5. Performance feedback loops and continuous learning
"""

import time
import random
import math
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Import hypergredient framework
try:
    from hypergredient_framework import (
        HypergredientClass, HypergredientDatabase, HypergredientInteractionMatrix,
        HypergredientOptimizer, HypergredientFormulation, HypergredientIngredient
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hypergredient_framework import (
        HypergredientClass, HypergredientDatabase, HypergredientInteractionMatrix,
        HypergredientOptimizer, HypergredientFormulation, HypergredientIngredient
    )

@dataclass
class PerformanceFeedback:
    """Real-world performance feedback for formulation"""
    formulation_id: str
    efficacy_rating: float  # 0-100 user rating
    safety_rating: float    # 0-100 user rating
    texture_rating: float   # 0-100 user rating
    results_timeline: Dict[str, float]  # week -> improvement %
    side_effects: List[str]
    user_satisfaction: float  # 0-100
    weeks_used: int
    skin_type: str
    age_group: str
    timestamp: float = field(default_factory=time.time)

@dataclass 
class FormulationGene:
    """Individual gene in formulation genetic algorithm"""
    hypergredient_class: HypergredientClass
    ingredient_name: str
    concentration: float
    mutation_rate: float = 0.1

@dataclass
class FormulationGenome:
    """Complete genome representing a formulation"""
    id: str
    genes: List[FormulationGene]
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)

class EvolutionaryStrategy(Enum):
    """Different evolutionary strategies for optimization"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution" 
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"

class HypergredientAI:
    """Machine learning integration for hypergredient performance prediction"""
    
    def __init__(self):
        self.feedback_database: List[PerformanceFeedback] = []
        self.model_weights = self._initialize_ml_weights()
        self.learning_rate = 0.01
        self.confidence_threshold = 0.7
        
    def _initialize_ml_weights(self) -> Dict[str, float]:
        """Initialize machine learning model weights"""
        return {
            'potency_weight': 0.3,
            'safety_weight': 0.25,
            'synergy_weight': 0.2,
            'bioavailability_weight': 0.15,
            'stability_weight': 0.1,
            'user_feedback_weight': 0.4,
            'clinical_evidence_weight': 0.6
        }
    
    def predict_optimal_combination(self, requirements: Dict[str, Any]) -> List[Tuple[HypergredientIngredient, float]]:
        """Predict optimal hypergredient combination for given requirements"""
        
        database = HypergredientDatabase()
        target_concerns = requirements.get('concerns', [])
        skin_type = requirements.get('skin_type', 'normal')
        budget = requirements.get('budget', 1000)
        
        # Feature extraction from requirements
        features = self._extract_features(requirements)
        
        # Get candidate ingredients for each concern
        candidates = []
        for concern in target_concerns:
            optimizer = HypergredientOptimizer()
            hg_class = optimizer.map_concern_to_hypergredient(concern)
            class_ingredients = database.get_ingredients_by_class(hg_class)
            
            for ingredient in class_ingredients:
                # Predict performance score using ML model
                performance_score = self._predict_ingredient_performance(
                    ingredient, features, concern
                )
                
                # Calculate confidence based on available data
                confidence = self._calculate_prediction_confidence(ingredient)
                
                if confidence >= self.confidence_threshold:
                    candidates.append((ingredient, performance_score))
        
        # Sort by predicted performance and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:10]  # Top 10 candidates
    
    def _extract_features(self, requirements: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from requirements for ML prediction"""
        features = {
            'age_factor': self._map_age_to_factor(requirements.get('age_group', 'adult')),
            'skin_sensitivity': self._map_skin_type_sensitivity(requirements.get('skin_type', 'normal')),
            'budget_tier': min(1.0, requirements.get('budget', 500) / 2000),
            'concern_severity': len(requirements.get('concerns', [])) / 5.0,  # Normalize by max concerns
            'preference_gentle': 1.0 if 'gentle' in requirements.get('preferences', []) else 0.0,
            'preference_natural': 1.0 if 'natural' in requirements.get('preferences', []) else 0.0
        }
        return features
    
    def _predict_ingredient_performance(self, ingredient: HypergredientIngredient, 
                                      features: Dict[str, float], concern: str) -> float:
        """Predict ingredient performance using ML model"""
        
        # Base performance from ingredient properties
        base_score = (
            ingredient.potency * self.model_weights['potency_weight'] +
            ingredient.safety_score * self.model_weights['safety_weight'] +
            ingredient.bioavailability * 0.01 * self.model_weights['bioavailability_weight']
        )
        
        # Adjust for skin type compatibility
        skin_adjustment = self._calculate_skin_type_adjustment(ingredient, features)
        
        # Adjust for concern specificity
        concern_adjustment = self._calculate_concern_adjustment(ingredient, concern)
        
        # Include historical feedback if available
        feedback_adjustment = self._calculate_feedback_adjustment(ingredient)
        
        final_score = (base_score * skin_adjustment * concern_adjustment + 
                      feedback_adjustment * self.model_weights['user_feedback_weight'])
        
        return min(10.0, max(0.0, final_score))
    
    def _calculate_skin_type_adjustment(self, ingredient: HypergredientIngredient, 
                                     features: Dict[str, float]) -> float:
        """Calculate adjustment factor based on skin type compatibility"""
        sensitivity = features.get('skin_sensitivity', 0.5)
        
        # High sensitivity users need gentler ingredients
        if sensitivity > 0.7 and ingredient.safety_score < 8.0:
            return 0.7  # Penalize harsh ingredients for sensitive skin
        elif sensitivity < 0.3 and ingredient.potency > 8.0:
            return 1.2  # Boost powerful ingredients for resilient skin
        
        return 1.0
    
    def _calculate_concern_adjustment(self, ingredient: HypergredientIngredient, concern: str) -> float:
        """Calculate adjustment based on ingredient's suitability for specific concern"""
        
        # Concern-specific ingredient preferences
        concern_preferences = {
            'wrinkles': {'potency': 1.3, 'evidence': 1.2},
            'sensitivity': {'safety': 1.5, 'potency': 0.8},
            'acne': {'potency': 1.2, 'safety': 1.1},
            'brightness': {'evidence': 1.3, 'bioavailability': 1.1},
            'dryness': {'safety': 1.2, 'bioavailability': 1.1}
        }
        
        preferences = concern_preferences.get(concern, {})
        adjustment = 1.0
        
        if 'potency' in preferences:
            adjustment *= preferences['potency'] if ingredient.potency >= 7.0 else 1.0
        if 'safety' in preferences:
            adjustment *= preferences['safety'] if ingredient.safety_score >= 8.0 else 1.0
        if 'evidence' in preferences:
            adjustment *= preferences['evidence'] if ingredient.evidence_level == "Strong" else 1.0
        
        return adjustment
    
    def _calculate_feedback_adjustment(self, ingredient: HypergredientIngredient) -> float:
        """Calculate adjustment based on historical user feedback"""
        
        # Find feedback for formulations containing this ingredient
        relevant_feedback = []
        for feedback in self.feedback_database:
            # In a real implementation, we'd track which ingredients were in each formulation
            # For now, use a simplified approach
            if ingredient.name.lower() in feedback.formulation_id.lower():
                relevant_feedback.append(feedback)
        
        if not relevant_feedback:
            return 0.0  # No adjustment if no feedback available
        
        # Calculate average satisfaction
        avg_satisfaction = sum(fb.user_satisfaction for fb in relevant_feedback) / len(relevant_feedback)
        avg_efficacy = sum(fb.efficacy_rating for fb in relevant_feedback) / len(relevant_feedback)
        
        # Convert to adjustment factor (0.5 - 1.5 range)
        satisfaction_factor = 0.5 + (avg_satisfaction / 100.0)
        efficacy_factor = 0.5 + (avg_efficacy / 100.0)
        
        return (satisfaction_factor + efficacy_factor) / 2.0
    
    def _calculate_prediction_confidence(self, ingredient: HypergredientIngredient) -> float:
        """Calculate confidence in prediction for this ingredient"""
        
        confidence_factors = []
        
        # Evidence level confidence
        if ingredient.evidence_level == "Strong":
            confidence_factors.append(0.9)
        elif ingredient.evidence_level == "Moderate":
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Data completeness confidence
        completeness = sum([
            1 if ingredient.potency > 0 else 0,
            1 if ingredient.safety_score > 0 else 0,
            1 if ingredient.bioavailability > 0 else 0,
            1 if ingredient.mechanism else 0,
            1 if ingredient.onset_time else 0
        ]) / 5.0
        confidence_factors.append(completeness)
        
        # Historical feedback availability
        feedback_count = len([fb for fb in self.feedback_database 
                            if ingredient.name.lower() in fb.formulation_id.lower()])
        feedback_confidence = min(1.0, feedback_count / 10.0)  # Max confidence at 10+ feedback points
        confidence_factors.append(feedback_confidence)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _map_age_to_factor(self, age_group: str) -> float:
        """Map age group to numerical factor"""
        age_map = {
            'teen': 0.2,
            'young_adult': 0.4,
            'adult': 0.6,
            'mature': 0.8,
            'senior': 1.0
        }
        return age_map.get(age_group.lower(), 0.6)
    
    def _map_skin_type_sensitivity(self, skin_type: str) -> float:
        """Map skin type to sensitivity factor"""
        sensitivity_map = {
            'sensitive': 0.9,
            'dry': 0.6,
            'normal': 0.5,
            'combination': 0.4,
            'oily': 0.3,
            'resilient': 0.1
        }
        return sensitivity_map.get(skin_type.lower(), 0.5)
    
    def add_feedback(self, feedback: PerformanceFeedback):
        """Add new performance feedback to the learning system"""
        self.feedback_database.append(feedback)
        
        # Update model weights based on feedback
        self._update_model_weights(feedback)
    
    def _update_model_weights(self, feedback: PerformanceFeedback):
        """Update ML model weights based on new feedback"""
        
        # Simple online learning update
        prediction_error = abs(feedback.user_satisfaction - 75.0) / 100.0  # Assume 75% expected satisfaction
        
        # Adjust weights based on error (simplified gradient descent)
        adjustment = self.learning_rate * prediction_error
        
        if feedback.user_satisfaction > 75.0:
            # Successful formulation - strengthen current preferences
            self.model_weights['user_feedback_weight'] += adjustment
        else:
            # Unsuccessful formulation - adjust other factors
            self.model_weights['safety_weight'] += adjustment * 0.5
            self.model_weights['potency_weight'] -= adjustment * 0.3

class FormulationEvolution:
    """Evolutionary formulation improvement system"""
    
    def __init__(self, strategy: EvolutionaryStrategy = EvolutionaryStrategy.GENETIC_ALGORITHM):
        self.strategy = strategy
        self.database = HypergredientDatabase()
        self.ai_predictor = HypergredientAI()
        self.population_size = 50
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_ratio = 0.1
        
        self.generation_history: List[Dict[str, Any]] = []
        self.best_genome: Optional[FormulationGenome] = None
        
    def evolve_formulation(self, base_formulation: HypergredientFormulation,
                          target_improvements: Dict[str, float],
                          constraints: Dict[str, Any] = None) -> HypergredientFormulation:
        """Evolve a formulation to meet target improvements"""
        
        print(f"Starting evolutionary optimization using {self.strategy.value}...")
        print(f"Target improvements: {target_improvements}")
        
        # Convert formulation to genome
        base_genome = self._formulation_to_genome(base_formulation)
        
        # Initialize population around base genome
        population = self._initialize_population(base_genome)
        
        for generation in range(self.max_generations):
            # Evaluate fitness for all genomes
            self._evaluate_population(population, target_improvements)
            
            # Track generation statistics
            gen_stats = self._calculate_generation_stats(population, generation)
            self.generation_history.append(gen_stats)
            
            if generation % 10 == 0:
                print(f"  Generation {generation}: Best fitness = {gen_stats['best_fitness']:.3f}")
            
            # Check convergence
            if gen_stats['best_fitness'] > 0.95:  # 95% of maximum possible fitness
                print(f"  Converged at generation {generation}")
                break
            
            # Create next generation
            population = self._create_next_generation(population)
        
        # Convert best genome back to formulation
        self.best_genome = max(population, key=lambda g: g.fitness_score)
        optimized_formulation = self._genome_to_formulation(self.best_genome, base_formulation)
        
        print(f"Evolution complete. Final fitness: {self.best_genome.fitness_score:.3f}")
        return optimized_formulation
    
    def _formulation_to_genome(self, formulation: HypergredientFormulation) -> FormulationGenome:
        """Convert hypergredient formulation to genetic representation"""
        
        genes = []
        for hg_class, data in formulation.hypergredients.items():
            for ing_data in data['ingredients']:
                ingredient = ing_data['ingredient']
                concentration = ing_data['percentage']
                
                gene = FormulationGene(
                    hypergredient_class=hg_class,
                    ingredient_name=ingredient.name,
                    concentration=concentration
                )
                genes.append(gene)
        
        return FormulationGenome(
            id=f"genome_{formulation.id}",
            genes=genes
        )
    
    def _initialize_population(self, base_genome: FormulationGenome) -> List[FormulationGenome]:
        """Initialize population around base genome with variations"""
        
        population = [base_genome]  # Include original
        
        for i in range(self.population_size - 1):
            # Create variant by mutating base genome
            variant = FormulationGenome(
                id=f"genome_init_{i}",
                genes=[]
            )
            
            for gene in base_genome.genes:
                # Create mutated copy of gene
                new_concentration = gene.concentration
                if random.random() < 0.3:  # 30% chance of mutation during initialization
                    mutation_factor = random.uniform(0.7, 1.3)  # ±30% variation
                    new_concentration = max(0.1, min(20.0, gene.concentration * mutation_factor))
                
                variant_gene = FormulationGene(
                    hypergredient_class=gene.hypergredient_class,
                    ingredient_name=gene.ingredient_name,
                    concentration=new_concentration
                )
                variant.genes.append(variant_gene)
            
            # Potentially add new genes (ingredients)
            if random.random() < 0.2:  # 20% chance of adding new ingredient
                self._add_random_ingredient(variant)
            
            population.append(variant)
        
        return population
    
    def _add_random_ingredient(self, genome: FormulationGenome):
        """Add a random ingredient to genome"""
        
        # Choose random hypergredient class
        available_classes = list(HypergredientClass)
        random_class = random.choice(available_classes)
        
        # Get ingredients from that class
        ingredients = self.database.get_ingredients_by_class(random_class)
        if ingredients:
            random_ingredient = random.choice(ingredients)
            
            # Check if ingredient already exists
            existing_names = [gene.ingredient_name for gene in genome.genes]
            if random_ingredient.name not in existing_names:
                new_gene = FormulationGene(
                    hypergredient_class=random_class,
                    ingredient_name=random_ingredient.name,
                    concentration=random.uniform(0.5, 5.0)  # Conservative concentration
                )
                genome.genes.append(new_gene)
    
    def _evaluate_population(self, population: List[FormulationGenome], 
                           target_improvements: Dict[str, float]):
        """Evaluate fitness for entire population"""
        
        for genome in population:
            fitness_components = []
            
            # Calculate formulation properties
            total_cost = 0.0
            total_potency = 0.0
            total_safety = 0.0
            concentration_sum = 0.0
            
            for gene in genome.genes:
                ingredient = self.database.find_ingredient_by_name(gene.ingredient_name)
                if ingredient:
                    weight = gene.concentration / 100.0
                    total_cost += ingredient.cost_per_gram * weight * 50  # 50g product
                    total_potency += ingredient.potency * weight
                    total_safety += ingredient.safety_score * weight
                    concentration_sum += gene.concentration
            
            # Normalize scores
            avg_potency = total_potency / len(genome.genes) if genome.genes else 0
            avg_safety = total_safety / len(genome.genes) if genome.genes else 0
            
            # Evaluate against target improvements
            efficacy_fitness = min(1.0, avg_potency / 10.0)  # Normalize to 0-1
            safety_fitness = min(1.0, avg_safety / 10.0)
            
            # Cost fitness (lower cost = higher fitness)
            cost_fitness = max(0.0, 1.0 - (total_cost / 2000.0))  # Assume R2000 max budget
            
            # Concentration constraint fitness
            concentration_fitness = 1.0 if concentration_sum <= 25.0 else 0.5  # Penalty for over-concentration
            
            # Synergy fitness (simplified)
            synergy_fitness = self._calculate_synergy_fitness(genome)
            
            # Target improvement fitness
            target_fitness = self._calculate_target_fitness(genome, target_improvements)
            
            # Combine fitness components
            genome.fitness_score = (
                efficacy_fitness * 0.25 +
                safety_fitness * 0.2 +
                cost_fitness * 0.15 +
                concentration_fitness * 0.15 +
                synergy_fitness * 0.1 +
                target_fitness * 0.15
            )
    
    def _calculate_synergy_fitness(self, genome: FormulationGenome) -> float:
        """Calculate synergy fitness for genome"""
        
        if len(genome.genes) < 2:
            return 0.5  # Neutral score for single ingredient
        
        interaction_matrix = HypergredientInteractionMatrix()
        synergy_scores = []
        
        for i, gene1 in enumerate(genome.genes):
            for gene2 in genome.genes[i+1:]:
                coefficient = interaction_matrix.get_interaction_coefficient(
                    gene1.hypergredient_class, gene2.hypergredient_class
                )
                # Normalize coefficient to 0-1 fitness scale
                synergy_score = max(0.0, min(1.0, (coefficient - 0.5) / 2.0))
                synergy_scores.append(synergy_score)
        
        return sum(synergy_scores) / len(synergy_scores) if synergy_scores else 0.5
    
    def _calculate_target_fitness(self, genome: FormulationGenome, 
                                target_improvements: Dict[str, float]) -> float:
        """Calculate how well genome meets target improvements"""
        
        # Simplified target fitness calculation
        target_scores = []
        
        for target, improvement_needed in target_improvements.items():
            if target == 'efficacy':
                # Calculate efficacy potential
                efficacy_potential = 0.0
                for gene in genome.genes:
                    ingredient = self.database.find_ingredient_by_name(gene.ingredient_name)
                    if ingredient:
                        efficacy_potential += ingredient.potency * (gene.concentration / 100.0)
                
                efficacy_score = min(1.0, efficacy_potential / (improvement_needed * 10))
                target_scores.append(efficacy_score)
            
            elif target == 'safety':
                # Calculate safety score
                safety_potential = 0.0
                for gene in genome.genes:
                    ingredient = self.database.find_ingredient_by_name(gene.ingredient_name)
                    if ingredient:
                        safety_potential += ingredient.safety_score * (gene.concentration / 100.0)
                
                safety_score = min(1.0, safety_potential / (improvement_needed * 10))
                target_scores.append(safety_score)
        
        return sum(target_scores) / len(target_scores) if target_scores else 0.5
    
    def _calculate_generation_stats(self, population: List[FormulationGenome], generation: int) -> Dict[str, Any]:
        """Calculate statistics for current generation"""
        
        fitness_scores = [genome.fitness_score for genome in population]
        
        return {
            'generation': generation,
            'best_fitness': max(fitness_scores),
            'avg_fitness': sum(fitness_scores) / len(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'population_size': len(population),
            'diversity': self._calculate_population_diversity(population)
        }
    
    def _calculate_population_diversity(self, population: List[FormulationGenome]) -> float:
        """Calculate genetic diversity in population"""
        
        # Count unique ingredient combinations
        unique_combinations = set()
        for genome in population:
            ingredient_signature = tuple(sorted(gene.ingredient_name for gene in genome.genes))
            unique_combinations.add(ingredient_signature)
        
        diversity = len(unique_combinations) / len(population)
        return diversity
    
    def _create_next_generation(self, population: List[FormulationGenome]) -> List[FormulationGenome]:
        """Create next generation using genetic operators"""
        
        # Sort by fitness
        population.sort(key=lambda g: g.fitness_score, reverse=True)
        
        next_generation = []
        
        # Elitism - keep best genomes
        elite_count = max(1, int(self.population_size * self.elite_ratio))
        for i in range(elite_count):
            elite_copy = self._copy_genome(population[i], f"gen_elite_{i}")
            next_generation.append(elite_copy)
        
        # Generate offspring through crossover and mutation
        while len(next_generation) < self.population_size:
            
            if random.random() < self.crossover_rate and len(population) >= 2:
                # Crossover
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                offspring = self._crossover(parent1, parent2)
            else:
                # Mutation only
                parent = self._tournament_selection(population)
                offspring = self._copy_genome(parent, f"gen_mutation_{len(next_generation)}")
            
            # Apply mutation
            if random.random() < self.mutation_rate:
                self._mutate(offspring)
            
            next_generation.append(offspring)
        
        return next_generation[:self.population_size]
    
    def _tournament_selection(self, population: List[FormulationGenome], 
                            tournament_size: int = 3) -> FormulationGenome:
        """Select parent using tournament selection"""
        
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda g: g.fitness_score)
    
    def _crossover(self, parent1: FormulationGenome, parent2: FormulationGenome) -> FormulationGenome:
        """Create offspring through crossover"""
        
        offspring = FormulationGenome(
            id=f"offspring_{parent1.id}_{parent2.id}",
            genes=[],
            parent_ids=[parent1.id, parent2.id]
        )
        
        # Combine genes from both parents
        all_ingredients = {}
        
        # Add genes from parent1
        for gene in parent1.genes:
            all_ingredients[gene.ingredient_name] = gene
        
        # Add/merge genes from parent2
        for gene in parent2.genes:
            if gene.ingredient_name in all_ingredients:
                # Average concentrations
                existing_gene = all_ingredients[gene.ingredient_name]
                avg_concentration = (existing_gene.concentration + gene.concentration) / 2.0
                existing_gene.concentration = avg_concentration
            else:
                all_ingredients[gene.ingredient_name] = gene
        
        offspring.genes = list(all_ingredients.values())
        return offspring
    
    def _mutate(self, genome: FormulationGenome):
        """Apply mutation to genome"""
        
        for gene in genome.genes:
            if random.random() < gene.mutation_rate:
                # Concentration mutation
                mutation_factor = random.uniform(0.8, 1.2)  # ±20% variation
                gene.concentration = max(0.1, min(20.0, gene.concentration * mutation_factor))
                genome.mutation_history.append(f"concentration_{gene.ingredient_name}")
        
        # Gene addition/deletion mutations
        if random.random() < 0.1:  # 10% chance
            if len(genome.genes) > 2:  # Don't go below 2 ingredients
                # Remove random gene
                removed_gene = random.choice(genome.genes)
                genome.genes.remove(removed_gene)
                genome.mutation_history.append(f"removed_{removed_gene.ingredient_name}")
            else:
                # Add random gene
                self._add_random_ingredient(genome)
                genome.mutation_history.append("added_random_ingredient")
    
    def _copy_genome(self, genome: FormulationGenome, new_id: str) -> FormulationGenome:
        """Create deep copy of genome"""
        
        new_genes = []
        for gene in genome.genes:
            new_gene = FormulationGene(
                hypergredient_class=gene.hypergredient_class,
                ingredient_name=gene.ingredient_name,
                concentration=gene.concentration,
                mutation_rate=gene.mutation_rate
            )
            new_genes.append(new_gene)
        
        return FormulationGenome(
            id=new_id,
            genes=new_genes,
            generation=genome.generation + 1,
            parent_ids=[genome.id]
        )
    
    def _genome_to_formulation(self, genome: FormulationGenome, 
                             base_formulation: HypergredientFormulation) -> HypergredientFormulation:
        """Convert genome back to hypergredient formulation"""
        
        # Create new formulation based on evolved genome
        evolved_formulation = HypergredientFormulation(
            id=f"evolved_{genome.id}",
            target_concerns=base_formulation.target_concerns,
            skin_type=base_formulation.skin_type,
            budget=base_formulation.budget,
            preferences=base_formulation.preferences
        )
        
        # Group genes by hypergredient class
        for gene in genome.genes:
            if gene.hypergredient_class not in evolved_formulation.hypergredients:
                evolved_formulation.hypergredients[gene.hypergredient_class] = {
                    'ingredients': [],
                    'total_percentage': 0.0
                }
            
            ingredient = self.database.find_ingredient_by_name(gene.ingredient_name)
            if ingredient:
                evolved_formulation.hypergredients[gene.hypergredient_class]['ingredients'].append({
                    'ingredient': ingredient,
                    'percentage': gene.concentration,
                    'reasoning': f'Evolved solution (fitness: {genome.fitness_score:.3f})'
                })
                evolved_formulation.hypergredients[gene.hypergredient_class]['total_percentage'] += gene.concentration
        
        # Recalculate formulation metrics
        optimizer = HypergredientOptimizer()
        optimizer._calculate_formulation_metrics(evolved_formulation)
        
        return evolved_formulation

def demonstrate_evolutionary_optimization():
    """Demonstrate evolutionary optimization capabilities"""
    
    print("=== HYPERGREDIENT EVOLUTIONARY OPTIMIZATION DEMO ===\n")
    
    # Create base formulation
    optimizer = HypergredientOptimizer()
    base_formulation = optimizer.optimize_formulation(
        target_concerns=['wrinkles', 'dryness'],
        skin_type='mature',
        budget=1500,
        preferences=['gentle']
    )
    
    print("1. Base Formulation:")
    print(f"   ID: {base_formulation.id}")
    print(f"   Efficacy Prediction: {base_formulation.efficacy_prediction:.0f}%")
    print(f"   Cost: R{base_formulation.cost_total:.2f}")
    print(f"   Synergy Score: {base_formulation.synergy_score:.2f}")
    
    # Define improvement targets
    target_improvements = {
        'efficacy': 1.2,  # 20% improvement in efficacy
        'safety': 1.1     # 10% improvement in safety
    }
    
    # Run evolutionary optimization
    evolution = FormulationEvolution(EvolutionaryStrategy.GENETIC_ALGORITHM)
    evolution.population_size = 20  # Smaller for demo
    evolution.max_generations = 30
    
    print(f"\n2. Evolutionary Optimization:")
    evolved_formulation = evolution.evolve_formulation(
        base_formulation, 
        target_improvements
    )
    
    print(f"\n3. Evolved Formulation:")
    print(f"   ID: {evolved_formulation.id}")
    print(f"   Efficacy Prediction: {evolved_formulation.efficacy_prediction:.0f}%")
    print(f"   Cost: R{evolved_formulation.cost_total:.2f}")
    print(f"   Synergy Score: {evolved_formulation.synergy_score:.2f}")
    
    # Show evolution statistics
    print(f"\n4. Evolution Statistics:")
    if evolution.generation_history:
        final_gen = evolution.generation_history[-1]
        print(f"   Generations: {final_gen['generation'] + 1}")
        print(f"   Final Fitness: {final_gen['best_fitness']:.3f}")
        print(f"   Population Diversity: {final_gen['diversity']:.2f}")
        
        # Show improvement over generations
        initial_fitness = evolution.generation_history[0]['best_fitness']
        final_fitness = final_gen['best_fitness']
        improvement = ((final_fitness - initial_fitness) / initial_fitness) * 100
        print(f"   Fitness Improvement: {improvement:.1f}%")
    
    # Demonstrate AI prediction
    print(f"\n5. AI-Powered Prediction Demo:")
    ai = HypergredientAI()
    
    requirements = {
        'concerns': ['wrinkles', 'brightness'],
        'skin_type': 'mature',
        'age_group': 'senior',
        'budget': 1200,
        'preferences': ['gentle', 'proven']
    }
    
    predictions = ai.predict_optimal_combination(requirements)
    print(f"   Top 3 AI-Recommended Ingredients:")
    for i, (ingredient, score) in enumerate(predictions[:3]):
        print(f"     {i+1}. {ingredient.name} (Score: {score:.2f})")
        print(f"        Class: {ingredient.hypergredient_class.value}")
        print(f"        Potency: {ingredient.potency}/10, Safety: {ingredient.safety_score}/10")
    
    # Simulate feedback learning
    print(f"\n6. Feedback Learning Demo:")
    feedback = PerformanceFeedback(
        formulation_id=evolved_formulation.id,
        efficacy_rating=85.0,
        safety_rating=92.0,
        texture_rating=78.0,
        results_timeline={
            '2_weeks': 15.0,
            '4_weeks': 35.0,
            '8_weeks': 65.0,
            '12_weeks': 85.0
        },
        side_effects=[],
        user_satisfaction=88.0,
        weeks_used=12,
        skin_type='mature',
        age_group='senior'
    )
    
    ai.add_feedback(feedback)
    print(f"   Added feedback: {feedback.user_satisfaction}% satisfaction")
    print(f"   Updated ML weights: User feedback weight = {ai.model_weights['user_feedback_weight']:.3f}")
    print(f"   Total feedback entries: {len(ai.feedback_database)}")

if __name__ == "__main__":
    demonstrate_evolutionary_optimization()
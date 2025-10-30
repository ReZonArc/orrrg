"""
Meta-Optimization Strategy for Comprehensive Formulation Generation

This module implements a meta-optimization strategy that generates optimal formulations
for every possible condition and treatment combination using recursive optimization
techniques, adaptive strategy selection, and continuous learning.
"""

from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json
import hashlib
from collections import defaultdict, Counter
from itertools import combinations, product
import time
import logging

from .core import Hypergredient, HypergredientDatabase, HYPERGREDIENT_CLASSES
from .optimization import (
    FormulationOptimizer, OptimizationObjective, FormulationRequest, 
    FormulationSolution, HypergredientFormulator
)
from .interaction import calculate_synergy_score


class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    GENETIC_ALGORITHM = "genetic"
    SIMULATED_ANNEALING = "annealing"
    PARTICLE_SWARM = "pso"
    DIFFERENTIAL_EVOLUTION = "de"
    HYBRID_MULTI_OBJECTIVE = "hybrid"
    RECURSIVE_DECOMPOSITION = "recursive"
    ADAPTIVE_SEARCH = "adaptive"


@dataclass
class ConditionTreatmentPair:
    """Represents a specific condition-treatment combination"""
    condition: str
    treatments: List[str]
    severity: str = "moderate"  # mild, moderate, severe
    skin_type: str = "normal"
    budget_range: Tuple[float, float] = (500.0, 2000.0)
    priority: int = 1  # 1=low, 2=medium, 3=high
    complexity_score: float = 1.0  # Calculated based on condition interactions


@dataclass
class OptimizationResult:
    """Results from meta-optimization process"""
    condition_treatment_pair: ConditionTreatmentPair
    formulation_solutions: List[FormulationSolution]
    optimization_strategy: OptimizationStrategy
    performance_metrics: Dict[str, float]
    computation_time: float
    iterations: int
    quality_score: float
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class StrategyPerformance:
    """Performance tracking for optimization strategies"""
    strategy: OptimizationStrategy
    success_rate: float = 0.0
    average_quality: float = 0.0
    average_time: float = 0.0
    usage_count: int = 0
    best_conditions: List[str] = field(default_factory=list)
    performance_trend: List[float] = field(default_factory=list)


class MetaOptimizationCache:
    """Advanced caching system for formulation patterns"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, FormulationSolution] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.pattern_signatures: Dict[str, Set[str]] = {}
        
    def _generate_key(self, request: FormulationRequest) -> str:
        """Generate cache key from formulation request"""
        key_data = {
            'concerns': sorted(request.target_concerns),
            'skin_type': request.skin_type,
            'budget': round(request.budget, -1),  # Round to nearest 10
            'excluded': sorted(request.excluded_ingredients),
            'required': sorted(request.required_ingredients)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, request: FormulationRequest) -> Optional[FormulationSolution]:
        """Get cached solution if available"""
        key = self._generate_key(request)
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, request: FormulationRequest, solution: FormulationSolution):
        """Cache formulation solution"""
        if len(self.cache) >= self.max_size:
            # Remove least accessed items
            least_used = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_used]
            del self.access_count[least_used]
            if least_used in self.pattern_signatures:
                del self.pattern_signatures[least_used]
        
        key = self._generate_key(request)
        self.cache[key] = solution
        self.access_count[key] += 1
        
        # Store pattern signature for similarity matching
        pattern = set(solution.hypergredients.keys())
        self.pattern_signatures[key] = pattern
    
    def find_similar_patterns(self, solution: FormulationSolution, 
                            similarity_threshold: float = 0.7) -> List[FormulationSolution]:
        """Find similar formulation patterns in cache"""
        current_pattern = set(solution.hypergredients.keys())
        similar_solutions = []
        
        for key, cached_pattern in self.pattern_signatures.items():
            # Calculate Jaccard similarity
            intersection = len(current_pattern & cached_pattern)
            union = len(current_pattern | cached_pattern)
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= similarity_threshold:
                similar_solutions.append(self.cache[key])
        
        return similar_solutions


class MetaOptimizationStrategy:
    """Meta-optimization strategy for comprehensive formulation generation"""
    
    def __init__(self, database: HypergredientDatabase, 
                 cache_size: int = 10000,
                 max_recursive_depth: int = 3):
        self.database = database
        self.base_optimizer = FormulationOptimizer(database)
        self.formulator = HypergredientFormulator(database)
        self.cache = MetaOptimizationCache(cache_size)
        self.max_recursive_depth = max_recursive_depth
        
        # Strategy performance tracking
        self.strategy_performance: Dict[OptimizationStrategy, StrategyPerformance] = {}
        self._initialize_strategy_tracking()
        
        # Comprehensive condition-treatment mapping
        self.condition_treatment_mapping = self._build_comprehensive_mapping()
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.05
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_strategy_tracking(self):
        """Initialize strategy performance tracking"""
        for strategy in OptimizationStrategy:
            self.strategy_performance[strategy] = StrategyPerformance(strategy=strategy)
    
    def _build_comprehensive_mapping(self) -> Dict[str, List[ConditionTreatmentPair]]:
        """Build comprehensive condition-treatment mapping"""
        mapping = defaultdict(list)
        
        # Primary skin concerns and their treatment combinations
        primary_concerns = {
            'aging': ['anti_aging', 'wrinkles', 'fine_lines', 'firmness', 'elasticity'],
            'pigmentation': ['hyperpigmentation', 'dark_spots', 'melasma', 'age_spots', 'brightness'],
            'acne': ['acne', 'blackheads', 'whiteheads', 'oily_skin', 'pores'],
            'sensitivity': ['sensitive_skin', 'redness', 'inflammation', 'irritation'], 
            'dryness': ['dryness', 'hydration', 'barrier_repair', 'flaking'],
            'texture': ['texture', 'roughness', 'smoothness', 'refinement'],
            'dullness': ['dullness', 'radiance', 'glow', 'luminosity']
        }
        
        skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        severities = ['mild', 'moderate', 'severe']
        
        # Generate all combinations
        for condition, treatments in primary_concerns.items():
            for skin_type in skin_types:
                for severity in severities:
                    # Calculate complexity based on number of treatments and skin sensitivity
                    complexity = len(treatments) * (1.5 if skin_type == 'sensitive' else 1.0)
                    if severity == 'severe':
                        complexity *= 1.3
                    
                    # Set budget range based on complexity and severity
                    base_budget = 800 if severity == 'mild' else 1200 if severity == 'moderate' else 1800
                    budget_range = (base_budget * 0.7, base_budget * 1.5)
                    
                    pair = ConditionTreatmentPair(
                        condition=condition,
                        treatments=treatments,
                        severity=severity,
                        skin_type=skin_type,
                        budget_range=budget_range,
                        priority=2 if severity == 'moderate' else 3 if severity == 'severe' else 1,
                        complexity_score=complexity
                    )
                    mapping[condition].append(pair)
        
        return dict(mapping)
    
    def select_optimal_strategy(self, pair: ConditionTreatmentPair, 
                              recursive_depth: int = 0) -> OptimizationStrategy:
        """Recursively select optimal optimization strategy based on problem characteristics"""
        if recursive_depth >= self.max_recursive_depth:
            return OptimizationStrategy.HYBRID_MULTI_OBJECTIVE
        
        # Calculate problem characteristics
        complexity = pair.complexity_score
        treatment_count = len(pair.treatments)
        is_sensitive = pair.skin_type == 'sensitive'
        is_severe = pair.severity == 'severe'
        
        # Strategy selection logic with recursive fallback
        if complexity > 6 and treatment_count > 4:
            # High complexity - use recursive decomposition
            selected = OptimizationStrategy.RECURSIVE_DECOMPOSITION
        elif is_sensitive and is_severe:
            # Sensitive + severe - use careful adaptive search
            selected = OptimizationStrategy.ADAPTIVE_SEARCH
        elif treatment_count > 3:
            # Multiple treatments - use multi-objective hybrid
            selected = OptimizationStrategy.HYBRID_MULTI_OBJECTIVE
        elif complexity < 2:
            # Simple case - use genetic algorithm
            selected = OptimizationStrategy.GENETIC_ALGORITHM
        else:
            # Default to particle swarm optimization
            selected = OptimizationStrategy.PARTICLE_SWARM
        
        # Check strategy performance and potentially select alternative
        perf = self.strategy_performance[selected]
        if perf.usage_count > 5 and perf.average_quality < 6.0:
            # Strategy performing poorly, try recursive selection
            self.logger.info(f"Strategy {selected} performing poorly, using recursive selection")
            alternative_pair = ConditionTreatmentPair(
                condition=pair.condition,
                treatments=pair.treatments[:2],  # Simplify for recursive call
                severity=pair.severity,
                skin_type=pair.skin_type,
                complexity_score=pair.complexity_score * 0.8
            )
            return self.select_optimal_strategy(alternative_pair, recursive_depth + 1)
        
        return selected
    
    def recursive_formulation_exploration(self, request: FormulationRequest,
                                        depth: int = 0,
                                        max_depth: int = 3) -> List[FormulationSolution]:
        """Recursively explore formulation space for optimal solutions"""
        if depth >= max_depth:
            return []
        
        # Check cache first
        cached_solution = self.cache.get(request)
        if cached_solution and cached_solution.total_score >= 7.0:
            return [cached_solution]
        
        # Generate initial solutions
        base_solutions = self.base_optimizer.optimize_formulation(request)
        if not base_solutions:
            return []
        
        all_solutions = base_solutions.copy()
        best_solution = max(base_solutions, key=lambda x: x.total_score)
        
        # If solution quality is insufficient, recurse with variations
        if best_solution.total_score < 8.0 and depth < max_depth:
            # Generate variations by modifying request parameters
            variations = self._generate_request_variations(request)
            
            for variation in variations:
                if self._is_promising_variation(variation, best_solution):
                    sub_solutions = self.recursive_formulation_exploration(
                        variation, depth + 1, max_depth
                    )
                    all_solutions.extend(sub_solutions)
        
        # Remove duplicates and sort by quality
        unique_solutions = self._deduplicate_solutions(all_solutions)
        sorted_solutions = sorted(unique_solutions, key=lambda x: x.total_score, reverse=True)
        
        # Cache best solution
        if sorted_solutions:
            self.cache.put(request, sorted_solutions[0])
        
        return sorted_solutions[:5]  # Return top 5 solutions
    
    def _generate_request_variations(self, base_request: FormulationRequest) -> List[FormulationRequest]:
        """Generate variations of formulation request for recursive exploration"""
        variations = []
        
        # Budget variations
        budget_multipliers = [0.8, 1.2, 1.5]
        for multiplier in budget_multipliers:
            variation = FormulationRequest(
                target_concerns=base_request.target_concerns.copy(),
                skin_type=base_request.skin_type,
                budget=base_request.budget * multiplier,
                preferences=base_request.preferences.copy(),
                excluded_ingredients=base_request.excluded_ingredients.copy(),
                required_ingredients=base_request.required_ingredients.copy(),
                ph_range=base_request.ph_range,
                max_ingredients=base_request.max_ingredients,
                regulatory_region=base_request.regulatory_region
            )
            variations.append(variation)
        
        # Ingredient count variations
        if base_request.max_ingredients > 4:
            variation = FormulationRequest(
                target_concerns=base_request.target_concerns.copy(),
                skin_type=base_request.skin_type,
                budget=base_request.budget,
                preferences=base_request.preferences.copy(),
                excluded_ingredients=base_request.excluded_ingredients.copy(),
                required_ingredients=base_request.required_ingredients.copy(),
                ph_range=base_request.ph_range,
                max_ingredients=base_request.max_ingredients - 2,
                regulatory_region=base_request.regulatory_region
            )
            variations.append(variation)
        
        # pH range variations
        ph_variations = [(4.0, 6.5), (5.5, 7.5), (4.5, 8.0)]
        for ph_range in ph_variations:
            if ph_range != base_request.ph_range:
                variation = FormulationRequest(
                    target_concerns=base_request.target_concerns.copy(),
                    skin_type=base_request.skin_type,
                    budget=base_request.budget,
                    preferences=base_request.preferences.copy(),
                    excluded_ingredients=base_request.excluded_ingredients.copy(),
                    required_ingredients=base_request.required_ingredients.copy(),
                    ph_range=ph_range,
                    max_ingredients=base_request.max_ingredients,
                    regulatory_region=base_request.regulatory_region
                )
                variations.append(variation)
        
        return variations
    
    def _is_promising_variation(self, variation: FormulationRequest, 
                              baseline: FormulationSolution) -> bool:
        """Determine if a variation is worth exploring"""
        # Simple heuristic: explore if budget allows for potentially better ingredients
        if variation.budget > baseline.cost * 1.2:
            return True
        
        # Or if we're reducing complexity (fewer ingredients)
        if variation.max_ingredients < 6:
            return True
        
        # Or if we're adjusting pH for better compatibility
        return True  # For now, explore all variations
    
    def _deduplicate_solutions(self, solutions: List[FormulationSolution]) -> List[FormulationSolution]:
        """Remove duplicate solutions based on ingredient composition"""
        seen_compositions = set()
        unique_solutions = []
        
        for solution in solutions:
            # Create signature based on ingredients and their ratios
            composition = tuple(sorted([
                (name, round(concentration, 1)) 
                for name, concentration in solution.hypergredients.items()
            ]))
            
            if composition not in seen_compositions:
                seen_compositions.add(composition)
                unique_solutions.append(solution)
        
        return unique_solutions
    
    def optimize_all_conditions(self, 
                              max_solutions_per_condition: int = 3,
                              use_parallel: bool = False) -> Dict[str, List[OptimizationResult]]:
        """Generate optimal formulations for all possible condition-treatment combinations"""
        all_results = {}
        total_pairs = sum(len(pairs) for pairs in self.condition_treatment_mapping.values())
        processed = 0
        
        self.logger.info(f"Starting meta-optimization for {total_pairs} condition-treatment pairs")
        
        for condition, pairs in self.condition_treatment_mapping.items():
            condition_results = []
            
            # Sort pairs by priority and complexity for optimal processing order
            sorted_pairs = sorted(pairs, key=lambda x: (x.priority, x.complexity_score), reverse=True)
            
            for pair in sorted_pairs:
                start_time = time.time()
                
                # Select optimal strategy for this pair
                strategy = self.select_optimal_strategy(pair)
                
                # Create formulation requests for different budget points
                budget_points = [
                    pair.budget_range[0],
                    (pair.budget_range[0] + pair.budget_range[1]) / 2,
                    pair.budget_range[1]
                ]
                
                best_solutions = []
                for budget in budget_points:
                    request = FormulationRequest(
                        target_concerns=pair.treatments,
                        skin_type=pair.skin_type,
                        budget=budget,
                        preferences=['gentle'] if pair.skin_type == 'sensitive' else [],
                        excluded_ingredients=[],
                        required_ingredients=[],
                        ph_range=(5.0, 7.0) if pair.skin_type == 'sensitive' else (4.5, 7.5),
                        max_ingredients=6 if pair.severity == 'severe' else 8,
                        regulatory_region="EU"
                    )
                    
                    # Use recursive exploration
                    solutions = self.recursive_formulation_exploration(request)
                    best_solutions.extend(solutions[:max_solutions_per_condition])
                
                # Remove duplicates and select best solutions
                unique_solutions = self._deduplicate_solutions(best_solutions)
                top_solutions = sorted(unique_solutions, key=lambda x: x.total_score, reverse=True)[:max_solutions_per_condition]
                
                computation_time = time.time() - start_time
                
                # Calculate performance metrics
                if top_solutions:
                    quality_score = np.mean([s.total_score for s in top_solutions])
                    performance_metrics = {
                        'average_quality': quality_score,
                        'best_quality': max(s.total_score for s in top_solutions),
                        'cost_efficiency': np.mean([s.total_score / s.cost for s in top_solutions]),
                        'solution_diversity': len(set(tuple(sorted(s.hypergredients.keys())) for s in top_solutions))
                    }
                else:
                    quality_score = 0.0
                    performance_metrics = {'average_quality': 0.0, 'best_quality': 0.0, 'cost_efficiency': 0.0, 'solution_diversity': 0}
                
                # Update strategy performance
                self._update_strategy_performance(strategy, quality_score, computation_time)
                
                # Generate improvement suggestions
                improvement_suggestions = self._generate_improvement_suggestions(pair, top_solutions)
                
                result = OptimizationResult(
                    condition_treatment_pair=pair,
                    formulation_solutions=top_solutions,
                    optimization_strategy=strategy,
                    performance_metrics=performance_metrics,
                    computation_time=computation_time,
                    iterations=len(best_solutions),
                    quality_score=quality_score,
                    improvement_suggestions=improvement_suggestions
                )
                
                condition_results.append(result)
                processed += 1
                
                if processed % 10 == 0:
                    self.logger.info(f"Processed {processed}/{total_pairs} pairs ({processed/total_pairs*100:.1f}%)")
            
            all_results[condition] = condition_results
        
        self.logger.info(f"Meta-optimization complete. Processed {processed} condition-treatment pairs.")
        return all_results
    
    def _update_strategy_performance(self, strategy: OptimizationStrategy, 
                                   quality_score: float, computation_time: float):
        """Update performance tracking for optimization strategy"""
        perf = self.strategy_performance[strategy]
        perf.usage_count += 1
        
        # Update running averages
        alpha = self.learning_rate
        perf.average_quality = (1 - alpha) * perf.average_quality + alpha * quality_score
        perf.average_time = (1 - alpha) * perf.average_time + alpha * computation_time
        perf.success_rate = (1 - alpha) * perf.success_rate + alpha * (1.0 if quality_score >= 7.0 else 0.0)
        
        # Track performance trend
        perf.performance_trend.append(quality_score)
        if len(perf.performance_trend) > 100:  # Keep last 100 measurements
            perf.performance_trend.pop(0)
    
    def _generate_improvement_suggestions(self, pair: ConditionTreatmentPair, 
                                        solutions: List[FormulationSolution]) -> List[str]:
        """Generate suggestions for improving formulation quality"""
        suggestions = []
        
        if not solutions:
            suggestions.append("No viable solutions found - consider relaxing constraints")
            return suggestions
        
        best_solution = max(solutions, key=lambda x: x.total_score)
        
        if best_solution.total_score < 6.0:
            suggestions.append("Consider increasing budget for higher quality ingredients")
            suggestions.append("Evaluate alternative treatment approaches")
        
        if best_solution.cost < pair.budget_range[1] * 0.6:
            suggestions.append("Budget underutilized - consider premium ingredients")
        
        if len(best_solution.warnings) > 2:
            suggestions.append("Multiple compatibility warnings - review ingredient interactions")
        
        # Check for missing synergies
        if best_solution.synergy_score < 1.2:
            suggestions.append("Low synergy score - consider complementary ingredient combinations")
        
        return suggestions
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization performance report"""
        report = {
            'strategy_performance': {},
            'cache_statistics': {
                'size': len(self.cache.cache),
                'hit_rate': sum(self.cache.access_count.values()) / len(self.cache.cache) if self.cache.cache else 0,
                'pattern_diversity': len(self.cache.pattern_signatures)
            },
            'condition_coverage': len(self.condition_treatment_mapping),
            'total_combinations': sum(len(pairs) for pairs in self.condition_treatment_mapping.values())
        }
        
        for strategy, perf in self.strategy_performance.items():
            if perf.usage_count > 0:
                report['strategy_performance'][strategy.value] = {
                    'usage_count': perf.usage_count,
                    'success_rate': perf.success_rate,
                    'average_quality': perf.average_quality,
                    'average_time': perf.average_time,
                    'performance_trend': perf.performance_trend[-10:]  # Last 10 measurements
                }
        
        return report
    
    def export_formulation_library(self, results: Dict[str, List[OptimizationResult]], 
                                 output_path: str = "formulation_library.json"):
        """Export comprehensive formulation library to JSON"""
        library = {
            'metadata': {
                'generated_at': time.time(),
                'total_conditions': len(results),
                'total_formulations': sum(len(condition_results) for condition_results in results.values()),
                'optimization_report': self.get_optimization_report()
            },
            'formulations': {}
        }
        
        for condition, condition_results in results.items():
            library['formulations'][condition] = []
            
            for result in condition_results:
                formulation_data = {
                    'condition': result.condition_treatment_pair.condition,
                    'treatments': result.condition_treatment_pair.treatments,
                    'severity': result.condition_treatment_pair.severity,
                    'skin_type': result.condition_treatment_pair.skin_type,
                    'budget_range': result.condition_treatment_pair.budget_range,
                    'optimization_strategy': result.optimization_strategy.value,
                    'quality_score': result.quality_score,
                    'computation_time': result.computation_time,
                    'solutions': []
                }
                
                for solution in result.formulation_solutions:
                    solution_data = {
                        'hypergredients': solution.hypergredients,
                        'total_score': solution.total_score,
                        'cost': solution.cost,
                        'predicted_efficacy': solution.predicted_efficacy,
                        'synergy_score': solution.synergy_score,
                        'stability_months': solution.stability_months,
                        'warnings': solution.warnings
                    }
                    formulation_data['solutions'].append(solution_data)
                
                library['formulations'][condition].append(formulation_data)
        
        with open(output_path, 'w') as f:
            json.dump(library, f, indent=2)
        
        self.logger.info(f"Formulation library exported to {output_path}")
        return library
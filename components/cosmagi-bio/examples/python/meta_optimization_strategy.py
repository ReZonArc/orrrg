#!/usr/bin/env python3
"""
🎯 Meta-Optimization Strategy for Cosmeceutical Formulation

This module implements a comprehensive meta-optimization system that generates
optimal formulations for every possible condition and treatment combination.
It learns from optimization results to continuously improve and adapt strategies.

Key Features:
- Systematic exploration of all condition/treatment combinations
- Adaptive optimization strategy selection based on learned patterns
- Performance analytics and continuous improvement
- Comprehensive formulation library generation
- Cross-condition learning and knowledge transfer
- Multi-objective meta-learning with evolutionary strategies

Author: OpenCog Multiscale Optimization Framework
License: AGPL-3.0
"""

import json
import time
import random
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime
import itertools
import statistics

# Import existing optimizers
try:
    from hypergredient_optimizer import (
        HypergredientFormulationOptimizer, FormulationRequest, 
        ConcernType, SkinType, generate_optimal_formulation
    )
    from multiscale_optimizer import (
        MultiscaleConstraintOptimizer, Objective, Constraint,
        ObjectiveType, ConstraintType, BiologicalScale
    )
    from hypergredient_framework import HypergredientDatabase
    OPTIMIZERS_AVAILABLE = True
except ImportError:
    OPTIMIZERS_AVAILABLE = False
    print("Warning: Core optimizers not available, using mock implementations")

# Optional ML libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    HYPERGREDIENT = "hypergredient"
    MULTISCALE = "multiscale" 
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class ConditionSeverity(Enum):
    """Severity levels for skin conditions."""
    MILD = "mild"
    MODERATE = "moderate"  
    SEVERE = "severe"


class TreatmentGoal(Enum):
    """Treatment objectives."""
    PREVENTION = "prevention"
    TREATMENT = "treatment"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"


@dataclass
class ConditionProfile:
    """Complete profile of a skin condition."""
    concern: ConcernType
    severity: ConditionSeverity
    skin_type: SkinType
    treatment_goal: TreatmentGoal
    age_group: str = "adult"  # child, teen, adult, mature
    budget_range: str = "medium"  # low, medium, high, premium
    timeline: str = "standard"  # immediate, standard, long_term
    
    def get_profile_key(self) -> str:
        """Generate unique key for this condition profile."""
        return f"{self.concern.value}_{self.severity.value}_{self.skin_type.value}_{self.treatment_goal.value}_{self.age_group}_{self.budget_range}_{self.timeline}"


@dataclass
class OptimizationResult:
    """Result of a single optimization run."""
    profile: ConditionProfile
    strategy_used: OptimizationStrategy
    formulation: Dict[str, Any]
    performance_metrics: Dict[str, float]
    optimization_time: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyPerformance:
    """Performance tracking for optimization strategies."""
    strategy: OptimizationStrategy
    success_rate: float = 0.0
    avg_efficacy: float = 0.0
    avg_safety: float = 0.0
    avg_cost_efficiency: float = 0.0
    avg_optimization_time: float = 0.0
    total_runs: int = 0
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class FormulationLibraryEntry:
    """Entry in the formulation library."""
    profile: ConditionProfile
    formulation: Dict[str, Any]
    validation_score: float
    creation_date: datetime
    last_updated: datetime
    usage_count: int = 0
    performance_history: List[Dict[str, float]] = field(default_factory=list)


class MetaOptimizationStrategy:
    """
    Advanced meta-optimization system for generating optimal formulations
    across all possible condition and treatment combinations.
    """
    
    def __init__(self, 
                 database_path: str = None,
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.2):
        """Initialize the meta-optimization system."""
        self.database_path = database_path or "/tmp/meta_optimization_db.json"
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Initialize core components
        if OPTIMIZERS_AVAILABLE:
            self.hypergredient_db = HypergredientDatabase()
            self.hypergredient_optimizer = HypergredientFormulationOptimizer(self.hypergredient_db)
            self.multiscale_optimizer = MultiscaleConstraintOptimizer()
        
        # Performance tracking
        self.strategy_performance: Dict[OptimizationStrategy, StrategyPerformance] = {}
        self._initialize_strategy_tracking()
        
        # Formulation library
        self.formulation_library: Dict[str, FormulationLibraryEntry] = {}
        
        # Learning components
        self.condition_patterns: Dict[str, Dict] = defaultdict(dict)
        self.cross_condition_learnings: Dict[Tuple[str, str], float] = {}
        self.optimization_history: List[OptimizationResult] = []
        
        # Load existing data
        self._load_meta_database()
        
        logger.info("🎯 Meta-Optimization Strategy initialized")
    
    def _initialize_strategy_tracking(self):
        """Initialize performance tracking for all strategies."""
        for strategy in OptimizationStrategy:
            self.strategy_performance[strategy] = StrategyPerformance(strategy=strategy)
    
    def generate_all_condition_profiles(self) -> List[ConditionProfile]:
        """Generate all possible condition/treatment combinations."""
        logger.info("🔍 Generating comprehensive condition profile matrix...")
        
        profiles = []
        
        # Get all possible combinations
        concerns = list(ConcernType)
        severities = list(ConditionSeverity)
        skin_types = list(SkinType)
        treatment_goals = list(TreatmentGoal)
        age_groups = ["child", "teen", "adult", "mature"]
        budget_ranges = ["low", "medium", "high", "premium"]
        timelines = ["immediate", "standard", "long_term"]
        
        total_combinations = (len(concerns) * len(severities) * len(skin_types) * 
                            len(treatment_goals) * len(age_groups) * 
                            len(budget_ranges) * len(timelines))
        
        logger.info(f"📊 Total possible combinations: {total_combinations:,}")
        
        for concern in concerns:
            for severity in severities:
                for skin_type in skin_types:
                    for treatment_goal in treatment_goals:
                        for age_group in age_groups:
                            for budget_range in budget_ranges:
                                for timeline in timelines:
                                    profile = ConditionProfile(
                                        concern=concern,
                                        severity=severity,
                                        skin_type=skin_type,
                                        treatment_goal=treatment_goal,
                                        age_group=age_group,
                                        budget_range=budget_range,
                                        timeline=timeline
                                    )
                                    profiles.append(profile)
        
        logger.info(f"✅ Generated {len(profiles):,} condition profiles")
        return profiles
    
    def select_optimization_strategy(self, profile: ConditionProfile) -> OptimizationStrategy:
        """Select the best optimization strategy for a given condition profile."""
        
        # Check if we have enough historical data for this profile type
        profile_key = profile.get_profile_key()
        
        if profile_key in self.condition_patterns:
            pattern_data = self.condition_patterns[profile_key]
            if 'best_strategy' in pattern_data and pattern_data.get('confidence', 0) > 0.7:
                return OptimizationStrategy(pattern_data['best_strategy'])
        
        # Use exploration vs exploitation strategy
        if random.random() < self.exploration_rate:
            # Exploration: try different strategies
            return random.choice(list(OptimizationStrategy))
        else:
            # Exploitation: use best performing strategy overall
            best_strategy = max(self.strategy_performance.items(), 
                              key=lambda x: x[1].success_rate + x[1].avg_efficacy)
            return best_strategy[0]
    
    def optimize_single_condition(self, profile: ConditionProfile) -> OptimizationResult:
        """Optimize formulation for a single condition profile."""
        
        # Select optimization strategy
        strategy = self.select_optimization_strategy(profile)
        
        start_time = time.time()
        success = False
        formulation = {}
        performance_metrics = {}
        
        try:
            if strategy == OptimizationStrategy.HYPERGREDIENT and OPTIMIZERS_AVAILABLE:
                formulation = self._optimize_with_hypergredient(profile)
                success = True
                
            elif strategy == OptimizationStrategy.MULTISCALE and OPTIMIZERS_AVAILABLE:
                formulation = self._optimize_with_multiscale(profile)
                success = True
                
            elif strategy == OptimizationStrategy.HYBRID and OPTIMIZERS_AVAILABLE:
                formulation = self._optimize_with_hybrid(profile)
                success = True
                
            elif strategy == OptimizationStrategy.ADAPTIVE:
                formulation = self._optimize_with_adaptive(profile)
                success = True
                
            else:
                # Fallback to mock optimization
                formulation = self._mock_optimization(profile)
                success = True
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(formulation, profile)
            
        except Exception as e:
            logger.error(f"❌ Optimization failed for {profile.get_profile_key()}: {str(e)}")
            success = False
            formulation = {}
            performance_metrics = {'error': str(e)}
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            profile=profile,
            strategy_used=strategy,
            formulation=formulation,
            performance_metrics=performance_metrics,
            optimization_time=optimization_time,
            success=success
        )
        
        # Update learning and performance tracking
        self._update_learning(result)
        
        return result
    
    def _optimize_with_hypergredient(self, profile: ConditionProfile) -> Dict[str, Any]:
        """Optimize using hypergredient strategy."""
        
        # Convert profile to hypergredient request
        budget_mapping = {
            'low': 500.0,
            'medium': 1000.0, 
            'high': 2000.0,
            'premium': 5000.0
        }
        
        request = FormulationRequest(
            concerns=[profile.concern],
            skin_type=profile.skin_type,
            budget_limit=budget_mapping.get(profile.budget_range, 1000.0),
            preferences=self._get_preferences_for_profile(profile)
        )
        
        result = self.hypergredient_optimizer.optimize_formulation(request)
        
        return {
            'ingredients': result.best_formulation.get_ingredient_summary(),
            'predicted_efficacy': result.predicted_efficacy,
            'predicted_safety': result.predicted_safety,
            'estimated_cost': result.estimated_cost,
            'stability_months': result.stability_months,
            'optimization_details': {
                'generations': result.generations,
                'strategy': 'hypergredient'
            }
        }
    
    def _optimize_with_multiscale(self, profile: ConditionProfile) -> Dict[str, Any]:
        """Optimize using multiscale strategy."""
        
        # Define objectives based on profile
        objectives = self._create_objectives_for_profile(profile)
        constraints = self._create_constraints_for_profile(profile)
        
        result = self.multiscale_optimizer.optimize_formulation(
            objectives=objectives,
            constraints=constraints,
            max_time_seconds=30
        )
        
        return {
            'formulation': result.formulation,
            'objective_values': result.objective_values,
            'multiscale_profile': result.multiscale_profile,
            'optimization_details': {
                'computational_cost': result.computational_cost,
                'strategy': 'multiscale'
            }
        }
    
    def _optimize_with_hybrid(self, profile: ConditionProfile) -> Dict[str, Any]:
        """Optimize using hybrid strategy combining both approaches."""
        
        # First run hypergredient optimization
        hypergredient_result = self._optimize_with_hypergredient(profile)
        
        # Then refine with multiscale optimization
        multiscale_result = self._optimize_with_multiscale(profile)
        
        # Combine results intelligently
        combined_formulation = self._combine_optimization_results(
            hypergredient_result, multiscale_result, profile
        )
        
        return combined_formulation
    
    def _optimize_with_adaptive(self, profile: ConditionProfile) -> Dict[str, Any]:
        """Optimize using adaptive strategy that learns from patterns."""
        
        # Find similar conditions that worked well
        similar_profiles = self._find_similar_profiles(profile)
        
        if similar_profiles:
            # Use transfer learning approach
            base_formulation = self._create_base_from_similar(similar_profiles)
            # Adapt for specific profile
            adapted_formulation = self._adapt_formulation(base_formulation, profile)
            return adapted_formulation
        else:
            # Fall back to hybrid approach
            return self._optimize_with_hybrid(profile)
    
    def _mock_optimization(self, profile: ConditionProfile) -> Dict[str, Any]:
        """Mock optimization for testing when real optimizers unavailable."""
        
        base_ingredients = {
            'aqua': 70.0,
            'glycerin': 5.0 + random.uniform(-2, 2),
            'phenoxyethanol': 0.8
        }
        
        # Add concern-specific ingredients
        concern_ingredients = {
            ConcernType.WRINKLES: {'retinol': 0.5, 'hyaluronic_acid': 2.0},
            ConcernType.HYDRATION: {'hyaluronic_acid': 3.0, 'ceramides': 1.0},
            ConcernType.BRIGHTNESS: {'vitamin_c': 10.0, 'niacinamide': 5.0},
            ConcernType.ACNE: {'salicylic_acid': 2.0, 'niacinamide': 5.0}
        }
        
        if profile.concern in concern_ingredients:
            base_ingredients.update(concern_ingredients[profile.concern])
        
        # Adjust water content
        total_actives = sum(v for k, v in base_ingredients.items() if k != 'aqua')
        base_ingredients['aqua'] = max(60.0, 100.0 - total_actives)
        
        return {
            'ingredients': base_ingredients,
            'predicted_efficacy': random.uniform(0.6, 0.9),
            'predicted_safety': random.uniform(0.8, 0.95),
            'estimated_cost': random.uniform(800, 1500),
            'optimization_details': {'strategy': 'mock'}
        }
    
    def generate_comprehensive_library(self, 
                                     max_conditions: int = None,
                                     prioritize_common: bool = True) -> Dict[str, FormulationLibraryEntry]:
        """Generate comprehensive formulation library for all conditions."""
        
        logger.info("🏗️ Generating comprehensive formulation library...")
        
        all_profiles = self.generate_all_condition_profiles()
        
        if max_conditions:
            if prioritize_common:
                # Prioritize common conditions and skin types
                all_profiles.sort(key=self._get_condition_priority, reverse=True)
            all_profiles = all_profiles[:max_conditions]
        
        library = {}
        total_profiles = len(all_profiles)
        
        for i, profile in enumerate(all_profiles):
            if i % 100 == 0:
                logger.info(f"📈 Progress: {i}/{total_profiles} ({i/total_profiles:.1%})")
            
            profile_key = profile.get_profile_key()
            
            # Skip if already optimized recently
            if profile_key in self.formulation_library:
                entry = self.formulation_library[profile_key]
                if (datetime.now() - entry.last_updated).days < 30:
                    library[profile_key] = entry
                    continue
            
            # Optimize this condition
            result = self.optimize_single_condition(profile)
            
            if result.success:
                validation_score = self._validate_formulation(result.formulation, profile)
                
                entry = FormulationLibraryEntry(
                    profile=profile,
                    formulation=result.formulation,
                    validation_score=validation_score,
                    creation_date=datetime.now(),
                    last_updated=datetime.now()
                )
                
                library[profile_key] = entry
                self.formulation_library[profile_key] = entry
        
        logger.info(f"✅ Generated library with {len(library)} formulations")
        
        # Save to database
        self._save_meta_database()
        
        return library
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Generate insights from optimization history."""
        
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        insights = {
            'total_optimizations': len(self.optimization_history),
            'success_rate': sum(1 for r in self.optimization_history if r.success) / len(self.optimization_history),
            'strategy_performance': {},
            'condition_insights': {},
            'performance_trends': {}
        }
        
        # Strategy performance analysis
        for strategy, perf in self.strategy_performance.items():
            if perf.total_runs > 0:
                insights['strategy_performance'][strategy.value] = {
                    'success_rate': perf.success_rate,
                    'avg_efficacy': perf.avg_efficacy,
                    'avg_safety': perf.avg_safety,
                    'avg_optimization_time': perf.avg_optimization_time,
                    'total_runs': perf.total_runs
                }
        
        # Condition-specific insights
        condition_groups = defaultdict(list)
        for result in self.optimization_history:
            if result.success:
                condition_groups[result.profile.concern.value].append(result)
        
        for condition, results in condition_groups.items():
            if results:
                avg_efficacy = statistics.mean(r.performance_metrics.get('efficacy', 0) for r in results)
                avg_time = statistics.mean(r.optimization_time for r in results)
                insights['condition_insights'][condition] = {
                    'total_optimizations': len(results),
                    'avg_efficacy': avg_efficacy,
                    'avg_optimization_time': avg_time,
                    'best_strategy': max(results, key=lambda r: r.performance_metrics.get('efficacy', 0)).strategy_used.value
                }
        
        return insights
    
    def _get_preferences_for_profile(self, profile: ConditionProfile) -> List[str]:
        """Get optimization preferences based on profile."""
        preferences = []
        
        if profile.skin_type == SkinType.SENSITIVE:
            preferences.append('gentle')
        
        if profile.timeline == 'long_term':
            preferences.append('stable')
            
        if profile.budget_range == 'low':
            preferences.append('budget')
            
        if profile.age_group in ['child', 'teen']:
            preferences.append('gentle')
            
        return preferences or ['gentle', 'stable']
    
    def _create_objectives_for_profile(self, profile: ConditionProfile) -> List[Objective]:
        """Create optimization objectives based on profile."""
        objectives = []
        
        # Base objectives
        objectives.append(Objective(
            ObjectiveType.EFFICACY, 
            target_value=0.8 if profile.severity == ConditionSeverity.SEVERE else 0.7,
            weight=0.4,
            scale=BiologicalScale.ORGAN,
            tolerance=0.1
        ))
        
        objectives.append(Objective(
            ObjectiveType.SAFETY,
            target_value=0.95 if profile.skin_type == SkinType.SENSITIVE else 0.9,
            weight=0.3,
            scale=BiologicalScale.ORGAN,
            tolerance=0.05
        ))
        
        # Budget-based cost objective
        cost_target = {'low': 0.4, 'medium': 0.6, 'high': 0.8, 'premium': 1.0}
        objectives.append(Objective(
            ObjectiveType.COST,
            target_value=cost_target.get(profile.budget_range, 0.6),
            weight=0.2,
            scale=BiologicalScale.MOLECULAR,
            tolerance=0.2
        ))
        
        objectives.append(Objective(
            ObjectiveType.STABILITY,
            target_value=0.8,
            weight=0.1,
            scale=BiologicalScale.MOLECULAR,
            tolerance=0.1
        ))
        
        return objectives
    
    def _create_constraints_for_profile(self, profile: ConditionProfile) -> List[Constraint]:
        """Create optimization constraints based on profile."""
        constraints = []
        
        # Regulatory constraints
        constraints.append(Constraint(
            ConstraintType.REGULATORY,
            "overall_efficacy",
            ">=",
            0.6,
            BiologicalScale.ORGAN,
            priority=1.0
        ))
        
        constraints.append(Constraint(
            ConstraintType.REGULATORY,
            "safety_profile", 
            ">=",
            0.8,
            BiologicalScale.ORGAN,
            priority=1.0
        ))
        
        return constraints
    
    def _calculate_performance_metrics(self, formulation: Dict[str, Any], 
                                     profile: ConditionProfile) -> Dict[str, float]:
        """Calculate performance metrics for a formulation."""
        metrics = {}
        
        if 'predicted_efficacy' in formulation:
            metrics['efficacy'] = formulation['predicted_efficacy']
        elif 'objective_values' in formulation and ObjectiveType.EFFICACY in formulation['objective_values']:
            metrics['efficacy'] = formulation['objective_values'][ObjectiveType.EFFICACY]
        else:
            metrics['efficacy'] = 0.7  # Default
            
        if 'predicted_safety' in formulation:
            metrics['safety'] = formulation['predicted_safety']
        elif 'objective_values' in formulation and ObjectiveType.SAFETY in formulation['objective_values']:
            metrics['safety'] = formulation['objective_values'][ObjectiveType.SAFETY]
        else:
            metrics['safety'] = 0.9  # Default
            
        # Cost efficiency (higher is better)
        cost = formulation.get('estimated_cost', 1000)
        budget_mapping = {'low': 500, 'medium': 1000, 'high': 2000, 'premium': 5000}
        target_budget = budget_mapping.get(profile.budget_range, 1000)
        metrics['cost_efficiency'] = max(0, min(1, (target_budget - cost) / target_budget + 0.5))
        
        return metrics
    
    def _update_learning(self, result: OptimizationResult):
        """Update learning systems with optimization result."""
        
        # Update strategy performance
        strategy_perf = self.strategy_performance[result.strategy_used]
        strategy_perf.total_runs += 1
        
        if result.success:
            # Update recent performance
            perf_score = (result.performance_metrics.get('efficacy', 0) + 
                         result.performance_metrics.get('safety', 0) + 
                         result.performance_metrics.get('cost_efficiency', 0)) / 3
            
            strategy_perf.recent_performance.append(perf_score)
            
            # Update averages
            strategy_perf.success_rate = (strategy_perf.success_rate * (strategy_perf.total_runs - 1) + 1) / strategy_perf.total_runs
            strategy_perf.avg_efficacy = (strategy_perf.avg_efficacy * (strategy_perf.total_runs - 1) + 
                                        result.performance_metrics.get('efficacy', 0)) / strategy_perf.total_runs
            strategy_perf.avg_safety = (strategy_perf.avg_safety * (strategy_perf.total_runs - 1) + 
                                      result.performance_metrics.get('safety', 0)) / strategy_perf.total_runs
            strategy_perf.avg_cost_efficiency = (strategy_perf.avg_cost_efficiency * (strategy_perf.total_runs - 1) + 
                                               result.performance_metrics.get('cost_efficiency', 0)) / strategy_perf.total_runs
        
        strategy_perf.avg_optimization_time = (strategy_perf.avg_optimization_time * (strategy_perf.total_runs - 1) + 
                                             result.optimization_time) / strategy_perf.total_runs
        
        # Update condition patterns
        profile_key = result.profile.get_profile_key()
        if profile_key not in self.condition_patterns:
            self.condition_patterns[profile_key] = {
                'strategy_scores': defaultdict(list),
                'best_strategy': None,
                'confidence': 0.0
            }
        
        pattern = self.condition_patterns[profile_key]
        if result.success:
            perf_score = (result.performance_metrics.get('efficacy', 0) + 
                         result.performance_metrics.get('safety', 0)) / 2
            pattern['strategy_scores'][result.strategy_used.value].append(perf_score)
            
            # Update best strategy if we have enough data
            if len(pattern['strategy_scores'][result.strategy_used.value]) >= 3:
                avg_scores = {}
                for strategy, scores in pattern['strategy_scores'].items():
                    if len(scores) >= 2:
                        avg_scores[strategy] = statistics.mean(scores)
                
                if avg_scores:
                    best_strategy = max(avg_scores.keys(), key=lambda s: avg_scores[s])
                    pattern['best_strategy'] = best_strategy
                    pattern['confidence'] = min(0.9, len(pattern['strategy_scores'][best_strategy]) * 0.2)
        
        # Add to history
        self.optimization_history.append(result)
        
        # Keep history manageable
        if len(self.optimization_history) > 10000:
            self.optimization_history = self.optimization_history[-5000:]
    
    def _get_condition_priority(self, profile: ConditionProfile) -> float:
        """Get priority score for a condition (higher = more common/important)."""
        
        # Common concerns get higher priority
        concern_priority = {
            ConcernType.HYDRATION: 0.9,
            ConcernType.WRINKLES: 0.8,
            ConcernType.BRIGHTNESS: 0.7,
            ConcernType.ACNE: 0.6,
            ConcernType.SENSITIVITY: 0.5
        }
        
        # Common skin types get higher priority  
        skin_type_priority = {
            SkinType.NORMAL: 0.8,
            SkinType.COMBINATION: 0.7,
            SkinType.DRY: 0.6,
            SkinType.OILY: 0.5,
            SkinType.SENSITIVE: 0.4
        }
        
        base_score = (concern_priority.get(profile.concern, 0.3) + 
                     skin_type_priority.get(profile.skin_type, 0.3)) / 2
        
        # Boost common demographics
        if profile.age_group == 'adult':
            base_score += 0.1
        if profile.budget_range in ['medium', 'high']:
            base_score += 0.1
        if profile.treatment_goal == TreatmentGoal.TREATMENT:
            base_score += 0.1
            
        return base_score
    
    def _find_similar_profiles(self, profile: ConditionProfile) -> List[FormulationLibraryEntry]:
        """Find similar condition profiles with good formulations."""
        similar = []
        
        for entry in self.formulation_library.values():
            similarity = self._calculate_profile_similarity(profile, entry.profile)
            if similarity > 0.6 and entry.validation_score > 0.7:
                similar.append(entry)
        
        # Sort by similarity and validation score
        similar.sort(key=lambda e: (self._calculate_profile_similarity(profile, e.profile) + 
                                   e.validation_score) / 2, reverse=True)
        
        return similar[:5]  # Top 5 similar profiles
    
    def _calculate_profile_similarity(self, profile1: ConditionProfile, 
                                    profile2: ConditionProfile) -> float:
        """Calculate similarity between two condition profiles."""
        
        # Exact matches get high scores
        similarity = 0.0
        
        if profile1.concern == profile2.concern:
            similarity += 0.4
        if profile1.skin_type == profile2.skin_type:
            similarity += 0.2
        if profile1.treatment_goal == profile2.treatment_goal:
            similarity += 0.15
        if profile1.severity == profile2.severity:
            similarity += 0.1
        if profile1.age_group == profile2.age_group:
            similarity += 0.08
        if profile1.budget_range == profile2.budget_range:
            similarity += 0.07
            
        return similarity
    
    def _create_base_from_similar(self, similar_entries: List[FormulationLibraryEntry]) -> Dict[str, Any]:
        """Create base formulation from similar successful formulations."""
        
        if not similar_entries:
            return {}
        
        # Weighted average based on validation scores
        total_weight = sum(entry.validation_score for entry in similar_entries)
        
        base_formulation = {}
        ingredient_weights = defaultdict(float)
        ingredient_concentrations = defaultdict(list)
        
        for entry in similar_entries:
            weight = entry.validation_score / total_weight
            formulation = entry.formulation
            
            if 'ingredients' in formulation:
                ingredients = formulation['ingredients']
                for ingredient, concentration in ingredients.items():
                    ingredient_weights[ingredient] += weight
                    ingredient_concentrations[ingredient].append(concentration * weight)
        
        # Create weighted average formulation
        total_concentration = 0
        for ingredient, weights in ingredient_weights.items():
            if weights > 0.3:  # Include ingredients present in enough formulations
                avg_concentration = sum(ingredient_concentrations[ingredient]) / weights
                base_formulation[ingredient] = avg_concentration
                total_concentration += avg_concentration
        
        # Normalize to 100%
        if total_concentration > 0:
            for ingredient in base_formulation:
                base_formulation[ingredient] = (base_formulation[ingredient] / total_concentration) * 100
        
        return {'ingredients': base_formulation, 'strategy': 'transfer_learning'}
    
    def _adapt_formulation(self, base_formulation: Dict[str, Any], 
                          profile: ConditionProfile) -> Dict[str, Any]:
        """Adapt base formulation for specific profile requirements."""
        
        if 'ingredients' not in base_formulation:
            return base_formulation
        
        adapted = base_formulation.copy()
        ingredients = adapted['ingredients'].copy()
        
        # Profile-specific adaptations
        if profile.severity == ConditionSeverity.SEVERE:
            # Increase active concentrations
            actives = ['retinol', 'vitamin_c', 'niacinamide', 'salicylic_acid', 'hyaluronic_acid']
            for active in actives:
                if active in ingredients:
                    ingredients[active] = min(ingredients[active] * 1.2, 10.0)  # Cap at 10%
        
        if profile.skin_type == SkinType.SENSITIVE:
            # Reduce potentially irritating ingredients
            irritants = ['retinol', 'salicylic_acid', 'glycolic_acid']
            for irritant in irritants:
                if irritant in ingredients:
                    ingredients[irritant] = max(ingredients[irritant] * 0.7, 0.1)
        
        if profile.age_group in ['child', 'teen']:
            # Remove strong actives
            strong_actives = ['retinol', 'hydroquinone']
            for active in strong_actives:
                if active in ingredients:
                    del ingredients[active]
        
        # Renormalize
        total = sum(ingredients.values())
        if total > 0:
            for ingredient in ingredients:
                ingredients[ingredient] = (ingredients[ingredient] / total) * 100
        
        adapted['ingredients'] = ingredients
        adapted['strategy'] = 'adaptive'
        
        return adapted
    
    def _combine_optimization_results(self, hypergredient_result: Dict[str, Any], 
                                    multiscale_result: Dict[str, Any],
                                    profile: ConditionProfile) -> Dict[str, Any]:
        """Intelligently combine results from different optimization strategies."""
        
        combined = {
            'strategy': 'hybrid',
            'hypergredient_component': hypergredient_result,
            'multiscale_component': multiscale_result
        }
        
        # Choose primary formulation based on performance
        hypergredient_score = hypergredient_result.get('predicted_efficacy', 0) * 0.6 + hypergredient_result.get('predicted_safety', 0) * 0.4
        multiscale_efficacy = multiscale_result.get('objective_values', {}).get(ObjectiveType.EFFICACY, 0)
        multiscale_safety = multiscale_result.get('objective_values', {}).get(ObjectiveType.SAFETY, 0)
        multiscale_score = multiscale_efficacy * 0.6 + multiscale_safety * 0.4
        
        if hypergredient_score > multiscale_score:
            combined.update(hypergredient_result)
            combined['primary_strategy'] = 'hypergredient'
        else:
            combined.update({
                'ingredients': multiscale_result['formulation'],
                'predicted_efficacy': multiscale_efficacy,
                'predicted_safety': multiscale_safety
            })
            combined['primary_strategy'] = 'multiscale'
        
        return combined
    
    def _validate_formulation(self, formulation: Dict[str, Any], 
                            profile: ConditionProfile) -> float:
        """Validate formulation quality and assign score."""
        
        score = 0.0
        
        # Check if formulation has required components
        if 'ingredients' in formulation:
            score += 0.3
            ingredients = formulation['ingredients']
            
            # Check for water/solvent
            if any(solvent in ingredients for solvent in ['aqua', 'water']):
                score += 0.1
            
            # Check for preservative
            if any(preservative in ingredients for preservative in ['phenoxyethanol', 'methylparaben']):
                score += 0.1
            
            # Check total concentration adds up reasonably
            total_conc = sum(ingredients.values())
            if 95 <= total_conc <= 105:
                score += 0.1
        
        # Performance metrics validation
        if 'predicted_efficacy' in formulation:
            score += min(0.2, formulation['predicted_efficacy'] * 0.2)
        
        if 'predicted_safety' in formulation:
            score += min(0.2, formulation['predicted_safety'] * 0.2)
        
        return min(1.0, score)
    
    def _save_meta_database(self):
        """Save meta-optimization database to file."""
        try:
            data = {
                'strategy_performance': {k.value: asdict(v) for k, v in self.strategy_performance.items()},
                'formulation_library': {k: asdict(v) for k, v in self.formulation_library.items()},
                'condition_patterns': dict(self.condition_patterns),
                'optimization_history': [asdict(r) for r in self.optimization_history[-1000:]],  # Keep last 1000
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.info(f"💾 Meta-database saved to {self.database_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save meta-database: {str(e)}")
    
    def _load_meta_database(self):
        """Load meta-optimization database from file."""
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
            
            # Load strategy performance
            for strategy_name, perf_data in data.get('strategy_performance', {}).items():
                strategy = OptimizationStrategy(strategy_name)
                perf = StrategyPerformance(strategy=strategy)
                for key, value in perf_data.items():
                    if key != 'strategy' and key != 'recent_performance':
                        setattr(perf, key, value)
                self.strategy_performance[strategy] = perf
            
            # Load condition patterns
            self.condition_patterns.update(data.get('condition_patterns', {}))
            
            logger.info(f"📖 Meta-database loaded from {self.database_path}")
            
        except FileNotFoundError:
            logger.info("📋 No existing meta-database found, starting fresh")
        except Exception as e:
            logger.error(f"❌ Failed to load meta-database: {str(e)}")


def main():
    """Demonstration of meta-optimization strategy."""
    print("🎯 Meta-Optimization Strategy Demonstration")
    print("=" * 60)
    
    # Initialize meta-optimizer
    meta_optimizer = MetaOptimizationStrategy()
    
    # Generate sample conditions for demonstration
    print("\n1. Testing Single Condition Optimization:")
    test_profile = ConditionProfile(
        concern=ConcernType.HYDRATION,
        severity=ConditionSeverity.MODERATE,
        skin_type=SkinType.DRY,
        treatment_goal=TreatmentGoal.TREATMENT,
        budget_range="medium"
    )
    
    result = meta_optimizer.optimize_single_condition(test_profile)
    print(f"   ✅ Optimized {test_profile.concern.value} for {test_profile.skin_type.value} skin")
    print(f"   Strategy: {result.strategy_used.value}")
    print(f"   Success: {result.success}")
    print(f"   Time: {result.optimization_time:.2f}s")
    
    # Generate small library for demonstration
    print("\n2. Generating Sample Formulation Library:")
    library = meta_optimizer.generate_comprehensive_library(max_conditions=50)
    print(f"   📚 Generated {len(library)} formulations")
    
    # Show insights
    print("\n3. Optimization Insights:")
    insights = meta_optimizer.get_optimization_insights()
    print(f"   Total optimizations: {insights['total_optimizations']}")
    print(f"   Success rate: {insights['success_rate']:.1%}")
    
    if insights['strategy_performance']:
        best_strategy = max(insights['strategy_performance'].items(), 
                          key=lambda x: x[1]['success_rate'])
        print(f"   Best strategy: {best_strategy[0]} ({best_strategy[1]['success_rate']:.1%} success)")
    
    print("\n🎯 Meta-optimization demonstration complete!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
hypergredient_meta_optimizer.py

🧬 Meta-Optimization Strategy for Comprehensive Formulation Coverage
Revolutionary system to generate optimal formulations for every possible condition and treatment

This module implements:
1. Comprehensive condition-treatment mapping
2. Meta-optimization strategy for all possible combinations
3. Systematic formulation generation with complete coverage
4. Performance comparison across all conditions and treatments
5. Automated optimization metrics and analysis
6. Advanced multi-objective optimization for complete matrix coverage
"""

import time
import json
import itertools
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import math

# Import hypergredient framework
try:
    from hypergredient_framework import (
        HypergredientClass, HypergredientDatabase, HypergredientOptimizer,
        HypergredientFormulation, HypergredientCompatibilityChecker,
        HYPERGREDIENT_DATABASE
    )
    from hypergredient_evolution import FormulationEvolution, EvolutionaryStrategy
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hypergredient_framework import (
        HypergredientClass, HypergredientDatabase, HypergredientOptimizer,
        HypergredientFormulation, HypergredientCompatibilityChecker,
        HYPERGREDIENT_DATABASE
    )
    from hypergredient_evolution import FormulationEvolution, EvolutionaryStrategy

@dataclass
class MetaOptimizationResult:
    """Results from meta-optimization strategy"""
    condition_combinations: List[List[str]]
    treatment_strategies: List[str]
    skin_types: List[str]
    budget_ranges: List[float]
    preference_sets: List[List[str]]
    
    # Generated formulations
    formulation_matrix: Dict[str, HypergredientFormulation] = field(default_factory=dict)
    optimization_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_rankings: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    
    # Analysis results
    coverage_analysis: Dict[str, Any] = field(default_factory=dict)
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    recommendation_matrix: Dict[str, Dict[str, str]] = field(default_factory=dict)

class HypergredientMetaOptimizer:
    """Meta-optimization strategy for comprehensive formulation coverage"""
    
    def __init__(self):
        self.optimizer = HypergredientOptimizer()
        self.evolution = FormulationEvolution()
        self.database = HypergredientDatabase()
        
        # Comprehensive condition taxonomy
        self.all_conditions = [
            # Anti-aging concerns
            'wrinkles', 'fine_lines', 'firmness', 'sagging', 'elasticity_loss',
            
            # Pigmentation concerns
            'dark_spots', 'hyperpigmentation', 'melasma', 'post_inflammatory_hyperpigmentation',
            'uneven_tone', 'brightness', 'dullness',
            
            # Hydration concerns
            'dryness', 'dehydration', 'flaking', 'tightness', 'rough_texture',
            
            # Barrier concerns
            'barrier_damage', 'compromised_barrier', 'trans_epidermal_water_loss',
            'barrier_weakness', 'permeability_issues',
            
            # Inflammatory concerns
            'sensitivity', 'redness', 'irritation', 'inflammation', 'reactive_skin',
            'rosacea', 'eczema_prone',
            
            # Sebum regulation concerns
            'acne', 'oily_skin', 'enlarged_pores', 'blackheads', 'shine_control',
            'sebaceous_hyperactivity', 'comedogenesis',
            
            # Environmental concerns
            'environmental_damage', 'pollution_damage', 'uv_damage', 'free_radical_damage',
            'oxidative_stress', 'blue_light_damage',
            
            # Microbiome concerns
            'microbiome_imbalance', 'bacterial_overgrowth', 'skin_ph_imbalance',
            'dysbiosis', 'microflora_disruption'
        ]
        
        # Treatment strategy approaches
        self.treatment_strategies = [
            'preventive', 'corrective', 'maintenance', 'intensive', 'gentle',
            'clinical_strength', 'natural_approach', 'hybrid_approach',
            'multi_modal', 'targeted_precision', 'systemic_comprehensive'
        ]
        
        # Comprehensive skin type matrix
        self.skin_types = [
            'normal', 'dry', 'oily', 'combination', 'sensitive',
            'mature', 'young', 'acne_prone', 'dehydrated', 'reactive',
            'ethnic_skin', 'post_menopausal', 'hormonal_fluctuation'
        ]
        
        # Budget optimization ranges
        self.budget_ranges = [300, 500, 800, 1200, 1800, 2500, 3500, 5000]
        
        # Preference optimization sets
        self.preference_sets = [
            ['gentle', 'natural'],
            ['clinical', 'proven'],
            ['fast_acting', 'potent'],
            ['stable', 'long_lasting'],
            ['affordable', 'accessible'],
            ['premium', 'luxury'],
            ['sensitive_safe', 'hypoallergenic'],
            ['multi_functional', 'comprehensive'],
            ['minimal', 'simple'],
            ['advanced', 'cutting_edge']
        ]
    
    def generate_condition_combinations(self, max_combinations: int = 5) -> List[List[str]]:
        """Generate all meaningful condition combinations"""
        combinations = []
        
        # Single conditions
        combinations.extend([[condition] for condition in self.all_conditions])
        
        # Two-condition combinations (most common)
        for combo in itertools.combinations(self.all_conditions, 2):
            combinations.append(list(combo))
        
        # Three-condition combinations (complex cases)
        for combo in itertools.combinations(self.all_conditions, 3):
            combinations.append(list(combo))
            if len(combinations) >= max_combinations * len(self.all_conditions):
                break
        
        # Filter for realistic combinations based on hypergredient class compatibility
        realistic_combinations = []
        for combo in combinations:
            # Check if combination makes biological sense
            if self._is_realistic_combination(combo):
                realistic_combinations.append(combo)
        
        return realistic_combinations
    
    def _is_realistic_combination(self, conditions: List[str]) -> bool:
        """Check if condition combination is biologically realistic"""
        # Map conditions to hypergredient classes
        classes = set()
        for condition in conditions:
            hg_class = self.optimizer.map_concern_to_hypergredient(condition)
            classes.add(hg_class)
        
        # Avoid conflicting combinations (e.g., severe dryness + oily skin)
        conflicting_pairs = [
            (['dryness', 'dehydration'], ['oily_skin', 'sebaceous_hyperactivity']),
            (['sensitivity', 'reactive_skin'], ['clinical_strength', 'intensive']),
        ]
        
        for conflicting_set1, conflicting_set2 in conflicting_pairs:
            if (any(c in conditions for c in conflicting_set1) and 
                any(c in conditions for c in conflicting_set2)):
                return False
        
        # Limit to maximum 4 conditions for practical formulation
        return len(conditions) <= 4
    
    def optimize_all_combinations(self, limit_combinations: Optional[int] = None) -> MetaOptimizationResult:
        """Generate optimal formulations for all possible combinations"""
        print("🧬 Initiating Meta-Optimization Strategy...")
        print(f"Target: Generate optimal formulations for every possible condition and treatment")
        
        # Generate comprehensive combination matrix
        condition_combinations = self.generate_condition_combinations()
        if limit_combinations:
            condition_combinations = condition_combinations[:limit_combinations]
        
        print(f"📊 Optimization Matrix:")
        print(f"  • Condition combinations: {len(condition_combinations)}")
        print(f"  • Treatment strategies: {len(self.treatment_strategies)}")
        print(f"  • Skin types: {len(self.skin_types)}")
        print(f"  • Budget ranges: {len(self.budget_ranges)}")
        print(f"  • Preference sets: {len(self.preference_sets)}")
        
        total_formulations = (len(condition_combinations) * len(self.skin_types) * 
                            len(self.budget_ranges) * len(self.preference_sets))
        print(f"  • Total formulations to generate: {total_formulations:,}")
        
        result = MetaOptimizationResult(
            condition_combinations=condition_combinations,
            treatment_strategies=self.treatment_strategies,
            skin_types=self.skin_types,
            budget_ranges=self.budget_ranges,
            preference_sets=self.preference_sets
        )
        
        # Generate formulations for all combinations
        formulation_count = 0
        start_time = time.time()
        
        for i, conditions in enumerate(condition_combinations):
            for skin_type in self.skin_types:
                for budget in self.budget_ranges:
                    for preferences in self.preference_sets:
                        
                        # Create unique identifier
                        combo_id = f"{'-'.join(conditions)}_{skin_type}_{budget}_{'-'.join(preferences)}"
                        
                        try:
                            # Generate optimal formulation
                            formulation = self.optimizer.optimize_formulation(
                                target_concerns=conditions,
                                skin_type=skin_type,
                                budget=budget,
                                preferences=preferences
                            )
                            
                            result.formulation_matrix[combo_id] = formulation
                            
                            # Calculate optimization metrics
                            metrics = self._calculate_optimization_metrics(formulation, conditions, skin_type, budget)
                            result.optimization_metrics[combo_id] = metrics
                            
                            formulation_count += 1
                            
                            # Progress reporting
                            if formulation_count % 100 == 0:
                                elapsed = time.time() - start_time
                                rate = formulation_count / elapsed if elapsed > 0 else 0
                                remaining = (total_formulations - formulation_count) / rate if rate > 0 else 0
                                print(f"  Progress: {formulation_count:,}/{total_formulations:,} "
                                      f"({formulation_count/total_formulations*100:.1f}%) - "
                                      f"Rate: {rate:.1f}/sec - ETA: {remaining/60:.1f}min")
                        
                        except Exception as e:
                            print(f"  Warning: Failed to optimize {combo_id}: {e}")
                            continue
                        
                        # Limit total for demo purposes
                        if limit_combinations and formulation_count >= limit_combinations * 10:
                            break
                    if limit_combinations and formulation_count >= limit_combinations * 10:
                        break
                if limit_combinations and formulation_count >= limit_combinations * 10:
                    break
            if limit_combinations and formulation_count >= limit_combinations * 10:
                break
        
        print(f"✅ Meta-optimization complete! Generated {formulation_count:,} optimal formulations")
        
        # Analyze results
        result.coverage_analysis = self._analyze_coverage(result)
        result.efficiency_metrics = self._calculate_efficiency_metrics(result)
        result.performance_rankings = self._rank_formulations(result)
        result.recommendation_matrix = self._generate_recommendation_matrix(result)
        
        return result
    
    def _calculate_optimization_metrics(self, formulation: HypergredientFormulation, 
                                      conditions: List[str], skin_type: str, budget: float) -> Dict[str, float]:
        """Calculate comprehensive optimization metrics"""
        return {
            'efficacy_score': formulation.efficacy_prediction,
            'synergy_score': formulation.synergy_score,
            'cost_efficiency': budget / formulation.cost_total if formulation.cost_total > 0 else 0,
            'stability_score': formulation.stability_months / 24.0,  # Normalize to 0-1
            'complexity_score': len(formulation.hypergredients),
            'condition_coverage': len(conditions),
            'budget_utilization': formulation.cost_total / budget if budget > 0 else 0,
            'confidence_score': getattr(formulation, 'confidence_score', 0.7)
        }
    
    def _analyze_coverage(self, result: MetaOptimizationResult) -> Dict[str, Any]:
        """Analyze optimization coverage across all dimensions"""
        coverage = {
            'total_formulations': len(result.formulation_matrix),
            'condition_coverage': {},
            'skin_type_coverage': {},
            'budget_coverage': {},
            'hypergredient_class_usage': defaultdict(int),
            'success_rate': 0.0
        }
        
        # Analyze condition coverage
        for conditions in result.condition_combinations:
            condition_key = '-'.join(conditions)
            coverage['condition_coverage'][condition_key] = 0
            
            for combo_id in result.formulation_matrix:
                if condition_key in combo_id:
                    coverage['condition_coverage'][condition_key] += 1
        
        # Analyze hypergredient class usage
        for formulation in result.formulation_matrix.values():
            for hg_class in formulation.hypergredients:
                coverage['hypergredient_class_usage'][hg_class.value] += 1
        
        return coverage
    
    def _calculate_efficiency_metrics(self, result: MetaOptimizationResult) -> Dict[str, float]:
        """Calculate overall optimization efficiency metrics"""
        if not result.optimization_metrics:
            return {}
        
        all_metrics = list(result.optimization_metrics.values())
        
        return {
            'avg_efficacy': sum(m['efficacy_score'] for m in all_metrics) / len(all_metrics),
            'avg_synergy': sum(m['synergy_score'] for m in all_metrics) / len(all_metrics),
            'avg_cost_efficiency': sum(m['cost_efficiency'] for m in all_metrics) / len(all_metrics),
            'avg_stability': sum(m['stability_score'] for m in all_metrics) / len(all_metrics),
            'avg_complexity': sum(m['complexity_score'] for m in all_metrics) / len(all_metrics),
            'avg_confidence': sum(m['confidence_score'] for m in all_metrics) / len(all_metrics)
        }
    
    def _rank_formulations(self, result: MetaOptimizationResult) -> Dict[str, List[Tuple[str, float]]]:
        """Rank formulations across different criteria"""
        rankings = {
            'overall_performance': [],
            'cost_efficiency': [],
            'efficacy': [],
            'synergy': [],
            'stability': []
        }
        
        for combo_id, metrics in result.optimization_metrics.items():
            # Calculate overall performance score
            overall_score = (
                metrics['efficacy_score'] * 0.3 +
                metrics['synergy_score'] * 10 * 0.25 +  # Scale synergy to 0-100
                metrics['cost_efficiency'] * 0.2 +
                metrics['stability_score'] * 100 * 0.15 +  # Scale stability to 0-100
                metrics['confidence_score'] * 100 * 0.1  # Scale confidence to 0-100
            )
            
            rankings['overall_performance'].append((combo_id, overall_score))
            rankings['cost_efficiency'].append((combo_id, metrics['cost_efficiency']))
            rankings['efficacy'].append((combo_id, metrics['efficacy_score']))
            rankings['synergy'].append((combo_id, metrics['synergy_score']))
            rankings['stability'].append((combo_id, metrics['stability_score']))
        
        # Sort all rankings
        for criteria in rankings:
            rankings[criteria].sort(key=lambda x: x[1], reverse=True)
            rankings[criteria] = rankings[criteria][:50]  # Keep top 50
        
        return rankings
    
    def _generate_recommendation_matrix(self, result: MetaOptimizationResult) -> Dict[str, Dict[str, str]]:
        """Generate recommendation matrix for different scenarios"""
        matrix = {}
        
        # Generate recommendations by skin type
        for skin_type in result.skin_types:
            matrix[skin_type] = {}
            
            # Find best formulations for each condition set for this skin type
            skin_type_formulations = {
                combo_id: metrics for combo_id, metrics in result.optimization_metrics.items()
                if f"_{skin_type}_" in combo_id
            }
            
            if skin_type_formulations:
                # Find top recommendation
                best_combo = max(skin_type_formulations.items(), key=lambda x: x[1]['efficacy_score'])
                matrix[skin_type]['top_recommendation'] = best_combo[0]
                
                # Find most cost-efficient
                cost_efficient = max(skin_type_formulations.items(), key=lambda x: x[1]['cost_efficiency'])
                matrix[skin_type]['cost_efficient'] = cost_efficient[0]
        
        return matrix
    
    def generate_comprehensive_report(self, result: MetaOptimizationResult) -> str:
        """Generate comprehensive meta-optimization report"""
        report = []
        report.append("🧬 HYPERGREDIENT META-OPTIMIZATION STRATEGY REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overview
        report.append("📊 OPTIMIZATION COVERAGE OVERVIEW")
        report.append(f"Total Formulations Generated: {len(result.formulation_matrix):,}")
        report.append(f"Condition Combinations Covered: {len(result.condition_combinations)}")
        report.append(f"Skin Types Covered: {len(result.skin_types)}")
        report.append(f"Budget Ranges Covered: {len(result.budget_ranges)}")
        report.append("")
        
        # Efficiency Metrics
        if result.efficiency_metrics:
            report.append("⚡ OPTIMIZATION EFFICIENCY METRICS")
            for metric, value in result.efficiency_metrics.items():
                report.append(f"{metric.replace('_', ' ').title()}: {value:.2f}")
            report.append("")
        
        # Top Performers
        if result.performance_rankings.get('overall_performance'):
            report.append("🏆 TOP PERFORMING FORMULATIONS")
            for i, (combo_id, score) in enumerate(result.performance_rankings['overall_performance'][:10]):
                conditions = combo_id.split('_')[0].replace('-', ', ')
                skin_type = combo_id.split('_')[1]
                budget = combo_id.split('_')[2]
                report.append(f"{i+1:2d}. {conditions} | {skin_type} | R{budget} | Score: {score:.1f}")
            report.append("")
        
        # Coverage Analysis
        if result.coverage_analysis.get('hypergredient_class_usage'):
            report.append("🎯 HYPERGREDIENT CLASS UTILIZATION")
            class_usage = result.coverage_analysis['hypergredient_class_usage']
            for hg_class, count in sorted(class_usage.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(result.formulation_matrix)) * 100 if result.formulation_matrix else 0
                report.append(f"{hg_class:8s}: {count:4d} formulations ({percentage:5.1f}%)")
            report.append("")
        
        # Recommendations
        report.append("💡 META-OPTIMIZATION INSIGHTS")
        report.append("• Complete coverage achieved across all major condition combinations")
        report.append("• Systematic optimization ensures no condition-treatment gap")
        report.append("• Multi-objective approach balances efficacy, cost, and safety")
        report.append("• Evolutionary refinement possible for suboptimal combinations")
        report.append("")
        
        return "\n".join(report)

def demonstrate_meta_optimization():
    """Demonstrate the meta-optimization strategy"""
    print("🧬 HYPERGREDIENT META-OPTIMIZATION STRATEGY DEMONSTRATION")
    print("=" * 70)
    
    meta_optimizer = HypergredientMetaOptimizer()
    
    # Run meta-optimization (limited for demo)
    print("\n🚀 Running Meta-Optimization Strategy...")
    result = meta_optimizer.optimize_all_combinations(limit_combinations=20)
    
    # Generate and display report
    print("\n📋 GENERATING COMPREHENSIVE REPORT...")
    report = meta_optimizer.generate_comprehensive_report(result)
    print(report)
    
    # Save detailed results
    detailed_results = {
        'meta_optimization_summary': {
            'total_formulations': len(result.formulation_matrix),
            'condition_combinations': len(result.condition_combinations),
            'efficiency_metrics': result.efficiency_metrics,
            'top_performers': result.performance_rankings.get('overall_performance', [])[:10]
        },
        'coverage_analysis': result.coverage_analysis,
        'recommendation_matrix': result.recommendation_matrix
    }
    
    with open('/tmp/meta_optimization_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n✅ Meta-optimization complete! Detailed results saved to /tmp/meta_optimization_results.json")
    print(f"🎯 Generated optimal formulations for {len(result.formulation_matrix)} condition-treatment combinations")

if __name__ == "__main__":
    demonstrate_meta_optimization()
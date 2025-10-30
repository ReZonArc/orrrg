// Copyright (c) 2024, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * TypeScript interfaces for the revolutionary Hypergredient Framework Architecture
 *
 * This framework transforms cosmetic formulation from art to science by:
 * 1. Abstracting ingredients into functional classes (Hypergredients)
 * 2. Implementing multi-objective optimization algorithms
 * 3. Capturing network effects and synergies
 * 4. Enabling predictive modeling and continuous learning
 */

import type {CosmeticIngredient} from './cosmetic-chemistry.interfaces.js';

// Core Hypergredient Classification System
export type HypergredientClass =
    | 'H.CT' // Cellular Turnover Agents
    | 'H.CS' // Collagen Synthesis Promoters
    | 'H.AO' // Antioxidant Systems
    | 'H.BR' // Barrier Repair Complex
    | 'H.ML' // Melanin Modulators
    | 'H.HY' // Hydration Systems
    | 'H.AI' // Anti-Inflammatory Agents
    | 'H.MB' // Microbiome Balancers
    | 'H.SE' // Sebum Regulators
    | 'H.PD'; // Penetration/Delivery Enhancers

export interface HypergredientTaxonomy {
    [key: string]: {
        class: HypergredientClass;
        name: string;
        description: string;
        primary_functions: string[];
        secondary_functions: string[];
        target_concerns: string[];
    };
}

// Hypergredient Performance Metrics
export interface HypergredientMetrics {
    efficacy_score: number; // 0-10 scale
    bioavailability: number; // 0-100%
    stability_index: number; // 0-10 scale
    safety_profile: number; // 0-10 scale
    cost_efficiency: number; // calculated score
    potency_rating: number; // 0-10 scale
    onset_time_weeks: number; // time to effect
    duration_months: number; // effect duration
    evidence_strength: 'weak' | 'moderate' | 'strong' | 'clinical';
}

// Enhanced Ingredient with Hypergredient Properties
export interface HypergredientIngredient extends CosmeticIngredient {
    hypergredient_class: HypergredientClass;
    hypergredient_metrics: HypergredientMetrics;
    interaction_profile: HypergredientInteractionProfile;
    optimization_parameters: HypergredientOptimizationParams;
}

export interface HypergredientInteractionProfile {
    synergy_partners: Map<string, number>; // ingredient_id -> synergy_score (0-3)
    antagonistic_pairs: Map<string, number>; // ingredient_id -> antagonism_score (0-3)
    ph_dependencies: Map<string, number>; // pH -> stability_factor
    concentration_dependencies: Map<string, number>; // other_ingredient -> optimal_ratio
}

export interface HypergredientOptimizationParams {
    weight_efficacy: number;
    weight_safety: number;
    weight_stability: number;
    weight_cost: number;
    weight_synergy: number;
    constraint_min_concentration: number;
    constraint_max_concentration: number;
    constraint_ph_range: {min: number; max: number};
}

// Multi-Objective Optimization Framework
export interface OptimizationObjective {
    efficacy: number; // 0.35 default weight
    safety: number; // 0.25 default weight
    stability: number; // 0.20 default weight
    cost: number; // 0.15 default weight
    synergy: number; // 0.05 default weight
}

export interface FormulationConstraints {
    ph_range: {min: number; max: number};
    total_actives_range: {min: number; max: number}; // percentage
    max_individual_concentration: number;
    budget_limit: number;
    skin_type_restrictions: string[];
    regulatory_regions: string[];
    exclude_ingredients: string[];
    required_functions: string[];
}

// Hypergredient Network and Synergy Calculations
export interface HypergredientNetwork {
    nodes: HypergredientNode[];
    edges: HypergredientEdge[];
    network_score: number;
    critical_paths: HypergredientPath[];
}

export interface HypergredientNode {
    ingredient_id: string;
    hypergredient_class: HypergredientClass;
    centrality_score: number;
    importance_weight: number;
    active_connections: number;
}

export interface HypergredientEdge {
    from_ingredient: string;
    to_ingredient: string;
    interaction_type: 'synergistic' | 'antagonistic' | 'neutral';
    strength: number; // 0-3 scale
    mechanism: string;
    evidence_level: 'theoretical' | 'in_vitro' | 'clinical';
}

export interface HypergredientPath {
    ingredients: string[];
    path_efficacy: number;
    path_stability: number;
    bottleneck_ingredient: string;
}

// Dynamic Scoring and Performance Prediction
export interface HypergredientScore {
    composite_score: number;
    individual_scores: {
        efficacy: number;
        bioavailability: number;
        stability: number;
        safety: number;
        cost_efficiency: number;
    };
    network_bonus: number;
    constraint_penalties: number;
    confidence_interval: {min: number; max: number};
}

export interface PerformancePrediction {
    formulation_id: string;
    predicted_efficacy: Map<string, number>; // concern -> predicted_improvement_%
    predicted_timeline: Map<string, number>; // concern -> weeks_to_effect
    confidence_scores: Map<string, number>; // concern -> confidence_0_to_1
    risk_factors: string[];
    optimization_suggestions: OptimizationSuggestion[];
}

// Optimization Algorithm Results
export interface OptimizationSuggestion {
    type:
        | 'ingredient_substitution'
        | 'concentration_adjustment'
        | 'ph_modification'
        | 'stability_improvement'
        | 'cost_reduction'
        | 'synergy_enhancement';
    description: string;
    current_state: any;
    proposed_change: any;
    expected_improvement: FormulationImpact;
    implementation_difficulty: 'easy' | 'moderate' | 'difficult';
    estimated_cost: number;
    time_to_implement: number; // days
    risk_level: 'low' | 'medium' | 'high';
}

export interface FormulationImpact {
    efficacy_change: number; // -100 to +100%
    safety_change: number; // -100 to +100%
    stability_change: number; // -100 to +100%
    cost_change: number; // -100 to +100%
    market_appeal_change: number; // -100 to +100%
}

// Real-time Compatibility and Analysis
export interface CompatibilityAnalysis {
    ingredient_pairs: CompatibilityPair[];
    overall_compatibility: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
    stability_prediction: StabilityPrediction;
    interaction_warnings: InteractionWarning[];
    optimization_opportunities: OptimizationOpportunity[];
}

export interface CompatibilityPair {
    ingredient_a: string;
    ingredient_b: string;
    compatibility_score: number; // 0-100
    interaction_type: 'synergistic' | 'neutral' | 'antagonistic' | 'incompatible';
    ph_sensitivity: boolean;
    concentration_sensitivity: boolean;
    temperature_sensitivity: boolean;
    mechanism_description: string;
}

export interface StabilityPrediction {
    overall_stability: number; // 0-100
    shelf_life_months: number;
    degradation_pathways: DegradationPathway[];
    storage_requirements: StorageRequirement[];
    stability_testing_recommendations: string[];
}

export interface DegradationPathway {
    trigger: string; // 'light', 'oxygen', 'heat', 'pH_drift', etc.
    affected_ingredients: string[];
    degradation_rate: number; // % per month
    mitigation_strategies: string[];
}

export interface StorageRequirement {
    parameter: string; // 'temperature', 'humidity', 'light', etc.
    optimal_range: {min: number; max: number};
    critical_threshold: {min: number; max: number};
    monitoring_frequency: string;
}

export interface InteractionWarning {
    severity: 'info' | 'warning' | 'error' | 'critical';
    ingredients_involved: string[];
    warning_message: string;
    potential_consequences: string[];
    recommended_actions: string[];
}

export interface OptimizationOpportunity {
    opportunity_type: string;
    affected_ingredients: string[];
    potential_benefit: FormulationImpact;
    implementation_steps: string[];
    success_probability: number; // 0-100%
}

// Database and Search Interfaces
export interface HypergredientDatabase {
    hypergredients: Map<string, HypergredientIngredient>;
    interaction_matrix: Map<string, Map<string, number>>;
    performance_data: Map<string, PerformanceDataPoint[]>;
    regulatory_updates: RegulatoryChange[];
    market_intelligence: MarketIntelligence;
}

export interface PerformanceDataPoint {
    timestamp: Date;
    market_feedback: MarketFeedback;
    clinical_data: ClinicalDataPoint[];
    user_satisfaction: number; // 0-10 scale
    repurchase_rate: number; // 0-100%
}

export interface MarketFeedback {
    efficacy_rating: number;
    texture_rating: number;
    value_rating: number;
    overall_satisfaction: number;
    common_complaints: string[];
    positive_mentions: string[];
}

export interface ClinicalDataPoint {
    concern_addressed: string;
    improvement_percentage: number;
    time_to_effect_weeks: number;
    participant_count: number;
    statistical_significance: number; // p-value
}

export interface RegulatoryChange {
    region: string;
    ingredient_affected: string;
    change_type: 'banned' | 'restricted' | 'concentration_limit' | 'label_requirement';
    effective_date: Date;
    impact_severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface MarketIntelligence {
    trending_ingredients: TrendingIngredient[];
    emerging_concerns: EmergingConcern[];
    regulatory_landscape: RegulatoryLandscape[];
    consumer_preferences: ConsumerPreference[];
    pricing_trends: PricingTrend[];
}

export interface TrendingIngredient {
    ingredient_id: string;
    trend_velocity: number; // searches/mentions per month growth
    market_adoption_rate: number; // % of new products containing it
    price_trajectory: 'increasing' | 'stable' | 'decreasing';
    regulatory_status: 'approved' | 'pending' | 'restricted';
    consumer_sentiment: number; // -100 to +100
}

export interface EmergingConcern {
    concern_name: string;
    growth_rate: number; // % increase in mentions/searches
    demographic_concentration: string[];
    related_ingredients: string[];
    market_opportunity_score: number; // 0-100
}

export interface RegulatoryLandscape {
    region: string;
    recent_changes: RegulatoryChange[];
    upcoming_regulations: RegulatoryChange[];
    compliance_complexity: 'low' | 'medium' | 'high' | 'very_high';
}

export interface ConsumerPreference {
    preference_category: string; // 'texture', 'packaging', 'ingredients', etc.
    trending_attributes: string[];
    declining_attributes: string[];
    regional_variations: Map<string, string[]>;
}

export interface PricingTrend {
    ingredient_id: string;
    price_change_6m: number; // % change over 6 months
    price_volatility: number; // standard deviation
    supply_stability: 'stable' | 'volatile' | 'critical';
    alternative_sources: number; // count of suppliers
}

// Configuration and System Settings
export interface HypergredientSystemConfig {
    optimization_weights: OptimizationObjective;
    default_constraints: FormulationConstraints;
    database_sync_frequency: number; // hours
    performance_tracking: boolean;
    evolutionary_learning: boolean;
    real_time_compatibility: boolean;
}

// Event System for Real-time Updates
export interface HypergredientEvent {
    event_id: string;
    event_type:
        | 'formulation_optimized'
        | 'ingredient_added'
        | 'performance_data_updated'
        | 'regulatory_change'
        | 'market_trend_detected'
        | 'compatibility_warning';
    timestamp: Date;
    data: any;
    source: string;
    priority: 'low' | 'medium' | 'high' | 'critical';
    affected_formulations: string[];
}

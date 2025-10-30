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
 * TypeScript interfaces for cosmetic chemistry cheminformatics framework
 * Provides type definitions for cosmetic ingredient modeling, formulation analysis,
 * and compatibility checking within the OpenCog cheminformatics system.
 */

// Base Types
export interface Molecule {
    id: string;
    name: string;
    molecularWeight?: number;
    formula?: string;
    smiles?: string;
}

export interface ChemicalElement {
    symbol: string;
    atomicNumber: number;
    atomicWeight: number;
}

export interface FunctionalGroup {
    name: string;
    structure: string;
    properties: string[];
}

// Cosmetic Ingredient Categories
export type IngredientCategory =
    | 'ACTIVE_INGREDIENT'
    | 'PRESERVATIVE'
    | 'EMULSIFIER'
    | 'HUMECTANT'
    | 'SURFACTANT'
    | 'THICKENER'
    | 'EMOLLIENT'
    | 'ANTIOXIDANT'
    | 'UV_FILTER'
    | 'FRAGRANCE'
    | 'COLORANT'
    | 'PH_ADJUSTER';

export type IngredientSubtype =
    | 'NATURAL_EXTRACT'
    | 'SYNTHETIC_ACTIVE'
    | 'PEPTIDE'
    | 'VITAMIN'
    | 'MINERAL'
    | 'HUMECTANT';

export interface CosmeticIngredient extends Molecule {
    category: IngredientCategory;
    subtype?: IngredientSubtype;
    inci_name: string; // International Nomenclature of Cosmetic Ingredients
    cas_number?: string;
    functions: string[];
    solubility: 'water_soluble' | 'oil_soluble' | 'both' | 'insoluble';
    ph_stability_range?: {
        min: number;
        max: number;
    };
    concentration_range?: {
        min: number;
        max: number;
    };
    max_concentration?: number; // Percentage
    allergenicity: 'very_low' | 'low' | 'medium' | 'high';
    comedogenicity?: number; // 0-5 scale
    pregnancy_safe?: boolean;
    therapeutic_vectors?: string[];
    skin_penetration_depth?: string;
    onset_time_hours?: number;
    duration_hours?: number;
    stability_factors?: string[];
    regulatory_status?: Map<string, string>;
    evidence_level?: 'theoretical' | 'in_vitro' | 'in_vivo' | 'clinical';
    cost_per_gram?: number;
    sensitive_properties?: {
        light_sensitive?: boolean;
        oxygen_sensitive?: boolean;
        heat_sensitive?: boolean;
        oxidation_prone?: boolean;
    };
}

// Formulation Types
export type FormulationType =
    | 'SKINCARE_FORMULATION'
    | 'HAIRCARE_FORMULATION'
    | 'MAKEUP_FORMULATION'
    | 'FRAGRANCE_FORMULATION';

export interface FormulationIngredient {
    id: string;
    ingredient: CosmeticIngredient;
    concentration: number; // Percentage
    function_in_formula: string[];
}

export interface CosmeticFormulation {
    id: string;
    name: string;
    type: FormulationType;
    ingredients: CosmeticIngredient[];
    concentrations: Map<string, number>;
    total_cost: number;
    ph_target: number;
    target_properties: CosmeticProperty[];
    physical_properties?: PhysicalProperty[];
    target_ph?: number;
    stability_data?: StabilityData;
    regulatory_compliance?: RegulatoryCompliance;
    regulatory_approvals: Map<string, string>;
    creation_date: Date;
    last_modified: Date;
}

// Property Types
export interface CosmeticProperty {
    name: string;
    value: string | number;
    unit?: string;
    measurement_method?: string;
}

export interface PhysicalProperty extends CosmeticProperty {
    type:
        | 'PH_PROPERTY'
        | 'VISCOSITY_PROPERTY'
        | 'STABILITY_PROPERTY'
        | 'TEXTURE_PROPERTY'
        | 'SPF_PROPERTY'
        | 'SOLUBILITY_PROPERTY';
}

export interface SensoryProperty extends CosmeticProperty {
    type: 'VISUAL_PROPERTY' | 'TACTILE_PROPERTY' | 'OLFACTORY_PROPERTY';
    consumer_rating?: number; // 1-10 scale
}

// Interaction Types
export type InteractionType = 'COMPATIBLE' | 'INCOMPATIBLE' | 'SYNERGISTIC' | 'ANTAGONISTIC';

export interface IngredientInteraction {
    ingredient1: string; // Ingredient ID
    ingredient2: string; // Ingredient ID
    interaction_type: InteractionType;
    mechanism?: string;
    ph_dependent?: boolean;
    concentration_dependent?: boolean;
    evidence_level: 'theoretical' | 'in_vitro' | 'in_vivo' | 'clinical';
    references?: string[];
}

// Safety and Regulatory Types
export interface SafetyAssessment {
    ingredient_id: string;
    assessment_type:
        | 'acute_toxicity'
        | 'skin_irritation'
        | 'eye_irritation'
        | 'skin_sensitisation'
        | 'genotoxicity'
        | 'carcinogenicity';
    result: 'safe' | 'caution' | 'restricted' | 'prohibited';
    concentration_limit?: number;
    conditions?: string[];
    regulatory_body: string;
    assessment_date: Date;
}

export interface AllergenClassification {
    ingredient_id: string;
    allergen_type: 'fragrance_allergen' | 'preservative_allergen' | 'colorant_allergen' | 'other';
    requires_declaration: boolean;
    threshold_concentration?: number; // ppm or percentage
    regulations: string[]; // e.g., ['EU', 'FDA', 'HC']
}

export interface ConcentrationLimit {
    ingredient_id: string;
    max_concentration: number; // Percentage
    product_type?: string;
    regulatory_region: 'EU' | 'US' | 'JP' | 'CN' | 'BR' | 'global';
    restriction_reason?: string;
    effective_date?: Date;
}

export interface RegulatoryCompliance {
    formulation_id: string;
    compliant: boolean;
    violations: RegulatoryViolation[];
    warnings: RegulatoryWarning[];
    last_checked: Date;
}

export interface RegulatoryViolation {
    ingredient_id: string;
    violation_type:
        | 'concentration_exceeded'
        | 'prohibited_ingredient'
        | 'missing_declaration'
        | 'incompatible_combination';
    description: string;
    current_value?: number;
    limit_value?: number;
    regulation_reference: string;
}

export interface RegulatoryWarning {
    ingredient_id: string;
    warning_type: 'approaching_limit' | 'allergen_present' | 'pregnancy_caution';
    description: string;
    recommendation: string;
}

// Stability and Environmental Types
export interface StabilityData {
    formulation_id: string;
    stability_factors: StabilityFactor[];
    shelf_life_estimate?: number; // months
    storage_conditions: StorageCondition[];
    degradation_products?: string[];
    stability_rating: 'excellent' | 'good' | 'fair' | 'poor';
}

export interface StabilityFactor {
    factor: 'ph_compatibility' | 'oxidation_risk' | 'light_sensitivity' | 'temperature_stability' | 'microbial_growth';
    risk_level: 'low' | 'medium' | 'high';
    mitigation_strategies?: string[];
}

export interface StorageCondition {
    temperature_range?: {min: number; max: number}; // Celsius
    humidity_range?: {min: number; max: number}; // Percentage
    light_protection: boolean;
    atmosphere?: 'air' | 'nitrogen' | 'vacuum';
    container_type?: string;
}

export interface EnvironmentalImpact {
    ingredient_id: string;
    biodegradability: 'readily_biodegradable' | 'inherently_biodegradable' | 'not_readily_biodegradable' | 'persistent';
    aquatic_toxicity?: 'low' | 'medium' | 'high';
    bioaccumulation_potential?: 'low' | 'medium' | 'high';
    carbon_footprint?: number; // kg CO2 equivalent
    sustainability_rating?: number; // 1-10 scale
}

// Analysis and Optimization Types
export interface FormulationAnalysis {
    formulation: CosmeticFormulation;
    compatibility_matrix: CompatibilityMatrix;
    stability_assessment: StabilityData;
    regulatory_status: RegulatoryCompliance;
    optimization_suggestions: OptimizationSuggestion[];
    quality_score: number; // 0-100
}

export interface CompatibilityMatrix {
    ingredients: string[]; // Ingredient IDs
    interactions: IngredientInteraction[];
    overall_compatibility: 'excellent' | 'good' | 'caution' | 'problematic';
    critical_issues: string[];
}

export interface OptimizationSuggestion {
    type:
        | 'ingredient_substitution'
        | 'concentration_adjustment'
        | 'ph_modification'
        | 'stability_improvement'
        | 'cost_reduction';
    description: string;
    impact: 'low' | 'medium' | 'high';
    implementation_difficulty: 'easy' | 'moderate' | 'difficult';
    estimated_improvement: string;
}

export interface IngredientSubstitution {
    original_ingredient: string;
    alternative_ingredients: string[];
    substitution_ratio?: number;
    property_changes: PropertyChange[];
    cost_impact: 'lower' | 'similar' | 'higher';
    regulatory_impact: 'none' | 'minor' | 'significant';
}

export interface PropertyChange {
    property_name: string;
    current_value: string | number;
    new_value: string | number;
    impact: 'positive' | 'neutral' | 'negative';
}

// Query and Search Types
export interface IngredientSearchCriteria {
    category?: IngredientCategory;
    subtype?: IngredientSubtype;
    functions?: string[];
    solubility?: string;
    max_allergenicity?: string;
    pregnancy_safe?: boolean;
    max_concentration_range?: {min: number; max: number};
    ph_range?: {min: number; max: number};
    exclude_ingredients?: string[];
}

export interface FormulationSearchCriteria {
    type?: FormulationType;
    target_properties?: string[];
    max_ingredients?: number;
    budget_range?: {min: number; max: number};
    regulatory_regions?: string[];
    exclude_allergens?: boolean;
}

// Database and Repository Types
export interface CosmeticDatabase {
    ingredients: Map<string, CosmeticIngredient>;
    interactions: IngredientInteraction[];
    regulatory_data: Map<string, ConcentrationLimit[]>;
    safety_assessments: Map<string, SafetyAssessment[]>;
    formulations: Map<string, CosmeticFormulation>;
}

export interface CheminformaticsRepository {
    findIngredientById(id: string): Promise<CosmeticIngredient | null>;
    findIngredientsByCategory(category: IngredientCategory): Promise<CosmeticIngredient[]>;
    findIngredientInteractions(ingredientId: string): Promise<IngredientInteraction[]>;
    findFormulationById(id: string): Promise<CosmeticFormulation | null>;
    searchIngredients(criteria: IngredientSearchCriteria): Promise<CosmeticIngredient[]>;
    searchFormulations(criteria: FormulationSearchCriteria): Promise<CosmeticFormulation[]>;
    analyzeFormulation(formulation: CosmeticFormulation): Promise<FormulationAnalysis>;
    checkRegulatoryCompliance(formulation: CosmeticFormulation, region: string): Promise<RegulatoryCompliance>;
    findIngredientAlternatives(ingredientId: string, requirements: string[]): Promise<IngredientSubstitution[]>;
}

// Event Types for Real-time Updates
export interface CheminformaticsEvent {
    type:
        | 'ingredient_added'
        | 'formulation_created'
        | 'analysis_completed'
        | 'regulatory_update'
        | 'interaction_discovered';
    timestamp: Date;
    data: any;
    source: string;
}

// Configuration Types
export interface CheminformaticsConfig {
    database_url: string;
    regulatory_data_sources: string[];
    analysis_engine: 'basic' | 'advanced' | 'ml_enhanced';
    cache_ttl: number; // seconds
    max_formulation_ingredients: number;
    default_safety_margins: {
        concentration_buffer: number; // percentage
        ph_tolerance: number;
    };
}

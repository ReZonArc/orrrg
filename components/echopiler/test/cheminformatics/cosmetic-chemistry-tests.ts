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

import {describe, expect, it} from 'vitest';

import type {
    CosmeticFormulation,
    CosmeticIngredient,
    FormulationAnalysis,
    IngredientInteraction,
    RegulatoryCompliance,
} from '../../types/cheminformatics/cosmetic-chemistry.interfaces.js';

describe('Cosmetic Chemistry Framework', () => {
    describe('Ingredient Modeling', () => {
        it('should create a valid cosmetic ingredient', () => {
            const hyaluronicAcid: CosmeticIngredient = {
                id: 'ha_001',
                name: 'hyaluronic_acid',
                inci_name: 'Sodium Hyaluronate',
                category: 'ACTIVE_INGREDIENT',
                subtype: 'NATURAL_EXTRACT',
                molecularWeight: 1000000,
                functions: ['moisturizing', 'anti_aging', 'wound_healing'],
                solubility: 'water_soluble',
                ph_stability_range: {min: 3.0, max: 8.0},
                max_concentration: 2.0,
                allergenicity: 'very_low',
                comedogenicity: 0,
                pregnancy_safe: true,
            };

            expect(hyaluronicAcid.category).toBe('ACTIVE_INGREDIENT');
            expect(hyaluronicAcid.solubility).toBe('water_soluble');
            expect(hyaluronicAcid.functions).toContain('moisturizing');
            expect(hyaluronicAcid.ph_stability_range?.min).toBe(3.0);
            expect(hyaluronicAcid.pregnancy_safe).toBe(true);
        });

        it('should create ingredients with sensitive properties', () => {
            const vitaminC: CosmeticIngredient = {
                id: 'vc_001',
                name: 'vitamin_c',
                inci_name: 'Ascorbic Acid',
                category: 'ANTIOXIDANT',
                subtype: 'VITAMIN',
                functions: ['antioxidant', 'brightening'],
                solubility: 'water_soluble',
                ph_stability_range: {min: 2.0, max: 3.5},
                allergenicity: 'low',
                sensitive_properties: {
                    oxidation_prone: true,
                    light_sensitive: true,
                },
            };

            expect(vitaminC.sensitive_properties?.oxidation_prone).toBe(true);
            expect(vitaminC.sensitive_properties?.light_sensitive).toBe(true);
        });
    });

    describe('Formulation Creation', () => {
        it('should create a valid cosmetic formulation', () => {
            const moisturizer: CosmeticFormulation = {
                id: 'form_001',
                name: 'Basic Moisturizer',
                type: 'SKINCARE_FORMULATION',
                ingredients: [
                    {
                        id: 'ha_001',
                        name: 'hyaluronic_acid',
                        inci_name: 'Sodium Hyaluronate',
                        category: 'HUMECTANT',
                        functions: ['moisturizing'],
                        solubility: 'water_soluble',
                        allergenicity: 'very_low',
                    },
                    {
                        id: 'gly_001',
                        name: 'glycerin',
                        inci_name: 'Glycerin',
                        category: 'HUMECTANT',
                        functions: ['moisturizing'],
                        solubility: 'water_soluble',
                        allergenicity: 'very_low',
                    },
                ],
                concentrations: new Map([
                    ['ha_001', 1.0],
                    ['gly_001', 5.0],
                ]),
                total_cost: 15.5,
                ph_target: 5.5,
                target_properties: [
                    {name: 'hydration', value: 'high', unit: 'subjective'},
                    {name: 'texture', value: 'lightweight', unit: 'subjective'},
                ],
                physical_properties: [
                    {name: 'pH', value: 5.5, type: 'PH_PROPERTY', unit: 'pH units'},
                    {name: 'viscosity', value: 'medium', type: 'VISCOSITY_PROPERTY'},
                ],
                regulatory_approvals: new Map([
                    ['EU', 'approved'],
                    ['FDA', 'approved'],
                ]),
                creation_date: new Date(),
                last_modified: new Date(),
            };

            expect(moisturizer.type).toBe('SKINCARE_FORMULATION');
            expect(moisturizer.ingredients).toHaveLength(2);
            expect(moisturizer.ph_target).toBe(5.5);
            expect(moisturizer.concentrations.get('ha_001')).toBe(1.0);
        });
    });

    describe('Ingredient Interactions', () => {
        it('should model compatible ingredient interactions', () => {
            const compatibleInteraction: IngredientInteraction = {
                ingredient1: 'ha_001',
                ingredient2: 'nia_001',
                interaction_type: 'COMPATIBLE',
                mechanism: 'both_water_soluble_neutral_pH',
                ph_dependent: false,
                concentration_dependent: false,
                evidence_level: 'clinical',
            };

            expect(compatibleInteraction.interaction_type).toBe('COMPATIBLE');
            expect(compatibleInteraction.evidence_level).toBe('clinical');
        });

        it('should model incompatible ingredient interactions', () => {
            const incompatibleInteraction: IngredientInteraction = {
                ingredient1: 'vc_001',
                ingredient2: 'ret_001',
                interaction_type: 'INCOMPATIBLE',
                mechanism: 'pH_incompatibility',
                ph_dependent: true,
                evidence_level: 'theoretical',
                references: ['Cosmetic Chemistry Journal, 2023'],
            };

            expect(incompatibleInteraction.interaction_type).toBe('INCOMPATIBLE');
            expect(incompatibleInteraction.ph_dependent).toBe(true);
            expect(incompatibleInteraction.references).toContain('Cosmetic Chemistry Journal, 2023');
        });

        it('should model synergistic interactions', () => {
            const synergisticInteraction: IngredientInteraction = {
                ingredient1: 'vc_001',
                ingredient2: 've_001',
                interaction_type: 'SYNERGISTIC',
                mechanism: 'antioxidant_network',
                evidence_level: 'in_vivo',
            };

            expect(synergisticInteraction.interaction_type).toBe('SYNERGISTIC');
            expect(synergisticInteraction.mechanism).toBe('antioxidant_network');
        });
    });

    describe('Regulatory Compliance', () => {
        it('should track regulatory compliance status', () => {
            const compliance: RegulatoryCompliance = {
                formulation_id: 'form_001',
                compliant: true,
                violations: [],
                warnings: [
                    {
                        ingredient_id: 'ret_001',
                        warning_type: 'pregnancy_caution',
                        description: 'Retinol should be avoided during pregnancy',
                        recommendation: 'Use alternative anti-aging ingredients for pregnant consumers',
                    },
                ],
                last_checked: new Date(),
            };

            expect(compliance.compliant).toBe(true);
            expect(compliance.violations).toHaveLength(0);
            expect(compliance.warnings).toHaveLength(1);
            expect(compliance.warnings[0].warning_type).toBe('pregnancy_caution');
        });

        it('should handle regulatory violations', () => {
            const nonCompliant: RegulatoryCompliance = {
                formulation_id: 'form_002',
                compliant: false,
                violations: [
                    {
                        ingredient_id: 'ret_001',
                        violation_type: 'concentration_exceeded',
                        description: 'Retinol concentration exceeds EU limit',
                        current_value: 0.5,
                        limit_value: 0.3,
                        regulation_reference: 'EU Cosmetic Regulation 1223/2009',
                    },
                ],
                warnings: [],
                last_checked: new Date(),
            };

            expect(nonCompliant.compliant).toBe(false);
            expect(nonCompliant.violations).toHaveLength(1);
            expect(nonCompliant.violations[0].current_value).toBeGreaterThan(nonCompliant.violations[0].limit_value!);
        });
    });

    describe('Formulation Analysis', () => {
        it('should provide comprehensive formulation analysis', () => {
            const analysis: FormulationAnalysis = {
                formulation: {
                    id: 'form_001',
                    name: 'Test Formulation',
                    type: 'SKINCARE_FORMULATION',
                    ingredients: [],
                    concentrations: new Map(),
                    total_cost: 12.5,
                    ph_target: 5.5,
                    target_properties: [],
                    physical_properties: [],
                    regulatory_approvals: new Map([['EU', 'approved']]),
                    creation_date: new Date(),
                    last_modified: new Date(),
                },
                compatibility_matrix: {
                    ingredients: ['ha_001', 'nia_001'],
                    interactions: [],
                    overall_compatibility: 'excellent',
                    critical_issues: [],
                },
                stability_assessment: {
                    formulation_id: 'form_001',
                    stability_factors: [
                        {
                            factor: 'ph_compatibility',
                            risk_level: 'low',
                        },
                    ],
                    shelf_life_estimate: 24,
                    storage_conditions: [
                        {
                            temperature_range: {min: 15, max: 25},
                            humidity_range: {min: 40, max: 60},
                            light_protection: false,
                        },
                    ],
                    stability_rating: 'excellent',
                },
                regulatory_status: {
                    formulation_id: 'form_001',
                    compliant: true,
                    violations: [],
                    warnings: [],
                    last_checked: new Date(),
                },
                optimization_suggestions: [
                    {
                        type: 'stability_improvement',
                        description: 'Add antioxidant protection',
                        impact: 'medium',
                        implementation_difficulty: 'easy',
                        estimated_improvement: 'Extend shelf life by 6 months',
                    },
                ],
                quality_score: 85,
            };

            expect(analysis.compatibility_matrix.overall_compatibility).toBe('excellent');
            expect(analysis.stability_assessment.stability_rating).toBe('excellent');
            expect(analysis.regulatory_status.compliant).toBe(true);
            expect(analysis.quality_score).toBe(85);
            expect(analysis.optimization_suggestions).toHaveLength(1);
        });
    });

    describe('Type Safety and Validation', () => {
        it('should enforce ingredient category types', () => {
            const categories: Array<CosmeticIngredient['category']> = [
                'ACTIVE_INGREDIENT',
                'PRESERVATIVE',
                'EMULSIFIER',
                'HUMECTANT',
                'SURFACTANT',
                'THICKENER',
                'EMOLLIENT',
                'ANTIOXIDANT',
                'UV_FILTER',
                'FRAGRANCE',
                'COLORANT',
                'PH_ADJUSTER',
            ];

            expect(categories).toContain('ACTIVE_INGREDIENT');
            expect(categories).toContain('ANTIOXIDANT');
            expect(categories).toHaveLength(12);
        });

        it('should enforce solubility types', () => {
            const solubilities: Array<CosmeticIngredient['solubility']> = [
                'water_soluble',
                'oil_soluble',
                'both',
                'insoluble',
            ];

            expect(solubilities).toContain('water_soluble');
            expect(solubilities).toContain('oil_soluble');
            expect(solubilities).toHaveLength(4);
        });

        it('should enforce allergenicity levels', () => {
            const allergenicityLevels: Array<CosmeticIngredient['allergenicity']> = [
                'very_low',
                'low',
                'medium',
                'high',
            ];

            expect(allergenicityLevels).toContain('very_low');
            expect(allergenicityLevels).toContain('high');
            expect(allergenicityLevels).toHaveLength(4);
        });
    });
});

describe('Integration with Compiler Explorer', () => {
    it('should be compatible with existing type system', () => {
        // This test ensures our types integrate well with the existing Compiler Explorer codebase
        const ingredient: CosmeticIngredient = {
            id: 'test_001',
            name: 'test_ingredient',
            inci_name: 'Test Ingredient',
            category: 'ACTIVE_INGREDIENT',
            functions: ['testing'],
            solubility: 'water_soluble',
            allergenicity: 'low',
        };

        // Test that our types work with standard JavaScript operations
        expect(typeof ingredient.id).toBe('string');
        expect(Array.isArray(ingredient.functions)).toBe(true);
        expect(ingredient.functions.includes('testing')).toBe(true);
    });

    it('should handle optional properties correctly', () => {
        const minimalIngredient: CosmeticIngredient = {
            id: 'minimal_001',
            name: 'minimal_ingredient',
            inci_name: 'Minimal Ingredient',
            category: 'HUMECTANT',
            functions: ['moisturizing'],
            solubility: 'water_soluble',
            allergenicity: 'very_low',
        };

        expect(minimalIngredient.molecularWeight).toBeUndefined();
        expect(minimalIngredient.ph_stability_range).toBeUndefined();
        expect(minimalIngredient.sensitive_properties).toBeUndefined();
    });
});

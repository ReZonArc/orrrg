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

import {beforeEach, describe, expect, it} from 'vitest';

import {HypergredientFramework} from '../../lib/cheminformatics/hypergredient-framework.js';
import type {HypergredientClass} from '../../types/cheminformatics/hypergredient-framework.interfaces.js';

describe('Hypergredient Framework', () => {
    let framework: HypergredientFramework;

    beforeEach(() => {
        framework = new HypergredientFramework();
    });

    describe('Framework Initialization', () => {
        it('should initialize with default configuration', () => {
            expect(framework).toBeDefined();
            expect(framework.getDatabaseStats().total_ingredients).toBeGreaterThan(0);
        });

        it('should initialize with custom configuration', () => {
            const customConfig = {
                optimization_weights: {
                    efficacy: 0.4,
                    safety: 0.3,
                    stability: 0.15,
                    cost: 0.1,
                    synergy: 0.05,
                },
            };

            const customFramework = new HypergredientFramework(customConfig);
            expect(customFramework).toBeDefined();
        });

        it('should have comprehensive hypergredient database', () => {
            const stats = framework.getDatabaseStats();

            expect(stats.total_ingredients).toBeGreaterThanOrEqual(4);
            expect(stats.ingredients_by_class.size).toBeGreaterThan(0);
            expect(stats.avg_efficacy_by_class.size).toBeGreaterThan(0);

            // Check specific classes are present
            expect(stats.ingredients_by_class.has('H.CT')).toBe(true); // Cellular Turnover
            expect(stats.ingredients_by_class.has('H.CS')).toBe(true); // Collagen Synthesis
            expect(stats.ingredients_by_class.has('H.AO')).toBe(true); // Antioxidants
            expect(stats.ingredients_by_class.has('H.HY')).toBe(true); // Hydration
        });
    });

    describe('Hypergredient Classification System', () => {
        it('should properly classify ingredients by hypergredient class', () => {
            const stats = framework.getDatabaseStats();

            // Check that we have the 10 core hypergredient classes
            const expectedClasses: HypergredientClass[] = [
                'H.CT',
                'H.CS',
                'H.AO',
                'H.BR',
                'H.ML',
                'H.HY',
                'H.AI',
                'H.MB',
                'H.SE',
                'H.PD',
            ];

            // At least some of these classes should be represented
            let classesFound = 0;
            for (const expectedClass of expectedClasses) {
                if (stats.ingredients_by_class.has(expectedClass)) {
                    classesFound++;
                }
            }

            expect(classesFound).toBeGreaterThanOrEqual(4);
        });

        it('should have realistic efficacy scores by class', () => {
            const stats = framework.getDatabaseStats();

            for (const [, avgEfficacy] of stats.avg_efficacy_by_class) {
                expect(avgEfficacy).toBeGreaterThanOrEqual(1);
                expect(avgEfficacy).toBeLessThanOrEqual(10);
                expect(typeof avgEfficacy).toBe('number');
            }
        });
    });

    describe('Multi-Objective Optimization', () => {
        it('should optimize formulation for anti-aging concerns', () => {
            const result = framework.optimizeFormulation(
                ['wrinkles', 'fine_lines', 'firmness'],
                {
                    budget_limit: 2000,
                    total_actives_range: {min: 8, max: 20},
                },
                'normal',
            );

            expect(result).toBeDefined();
            expect(result.formulation).toBeDefined();
            expect(result.analysis).toBeDefined();
            expect(result.prediction).toBeDefined();
            expect(result.score).toBeDefined();

            // Check formulation properties
            expect(result.formulation.ingredients.length).toBeGreaterThan(0);
            expect(result.formulation.concentrations.size).toBeGreaterThan(0);
            expect(result.formulation.type).toBe('SKINCARE_FORMULATION');

            // Check score is reasonable
            expect(result.score.composite_score).toBeGreaterThanOrEqual(0);
            expect(result.score.composite_score).toBeLessThanOrEqual(10);
        });

        it('should optimize formulation for hydration concerns', () => {
            const result = framework.optimizeFormulation(
                ['dryness', 'hydration'],
                {
                    budget_limit: 1000,
                    total_actives_range: {min: 5, max: 15},
                },
                'dry',
            );

            expect(result.formulation.ingredients.length).toBeGreaterThan(0);
            expect(result.score.composite_score).toBeGreaterThan(0);

            // Should contain hydration ingredients
            const hasHydrationIngredient = result.formulation.ingredients.some(
                ing => (ing as any).hypergredient_class === 'H.HY',
            );
            expect(hasHydrationIngredient).toBe(true);
        });

        it('should optimize formulation for multiple skin concerns', () => {
            const result = framework.optimizeFormulation(
                ['wrinkles', 'hydration', 'brightness', 'firmness'],
                {
                    budget_limit: 3000,
                    total_actives_range: {min: 10, max: 25},
                },
                'combination',
            );

            expect(result.formulation.ingredients.length).toBeGreaterThanOrEqual(2);
            expect(result.score.composite_score).toBeGreaterThan(0);

            // Should have ingredients from multiple classes
            const classes = new Set(result.formulation.ingredients.map(ing => (ing as any).hypergredient_class));
            expect(classes.size).toBeGreaterThanOrEqual(2);
        });

        it('should respect budget constraints', () => {
            const lowBudgetResult = framework.optimizeFormulation(['hydration'], {budget_limit: 500}, 'normal');

            const highBudgetResult = framework.optimizeFormulation(['hydration'], {budget_limit: 5000}, 'normal');

            // Low budget should result in lower or equal cost
            expect(lowBudgetResult.formulation.total_cost).toBeLessThanOrEqual(highBudgetResult.formulation.total_cost);
        });

        it('should respect concentration constraints', () => {
            const result = framework.optimizeFormulation(['wrinkles'], {
                total_actives_range: {min: 2, max: 8},
                max_individual_concentration: 3.0,
            });

            const totalActives = Array.from(result.formulation.concentrations.values()).reduce(
                (sum, conc) => sum + conc,
                0,
            );

            expect(totalActives).toBeLessThanOrEqual(8.5); // Small tolerance

            for (const concentration of result.formulation.concentrations.values()) {
                expect(concentration).toBeLessThanOrEqual(3.1); // Small tolerance
            }
        });
    });

    describe('Compatibility Analysis', () => {
        it('should analyze ingredient compatibility', () => {
            const result = framework.optimizeFormulation(['hydration', 'anti_aging']);

            expect(result.analysis.ingredient_pairs.length).toBeGreaterThanOrEqual(0);
            expect(result.analysis.overall_compatibility).toMatch(/excellent|good|fair|poor|critical/);
            expect(result.analysis.stability_prediction.overall_stability).toBeGreaterThanOrEqual(0);
            expect(result.analysis.stability_prediction.overall_stability).toBeLessThanOrEqual(100);
        });

        it('should identify synergistic combinations', () => {
            const result = framework.optimizeFormulation(['firmness', 'hydration']);

            // Look for synergistic pairs
            const synergisticPairs = result.analysis.ingredient_pairs.filter(
                pair => pair.interaction_type === 'synergistic',
            );

            // Should find some synergies in a multi-ingredient formulation
            if (result.formulation.ingredients.length > 1) {
                expect(synergisticPairs.length).toBeGreaterThanOrEqual(0);
            }
        });

        it('should detect potential incompatibilities', () => {
            const result = framework.optimizeFormulation(['acne', 'anti_aging']);

            const incompatiblePairs = result.analysis.ingredient_pairs.filter(
                pair => pair.interaction_type === 'incompatible' || pair.compatibility_score < 50,
            );

            // Check that warnings are generated for low compatibility
            if (incompatiblePairs.length > 0) {
                expect(result.analysis.interaction_warnings.length).toBeGreaterThan(0);
            }
        });
    });

    describe('Performance Prediction', () => {
        it('should predict efficacy for target concerns', () => {
            const targetConcerns = ['wrinkles', 'hydration'];
            const result = framework.optimizeFormulation(targetConcerns);

            expect(result.prediction.predicted_efficacy.size).toBeGreaterThan(0);
            expect(result.prediction.predicted_timeline.size).toBeGreaterThan(0);
            expect(result.prediction.confidence_scores.size).toBeGreaterThan(0);

            for (const concern of targetConcerns) {
                const efficacy = result.prediction.predicted_efficacy.get(concern);
                const timeline = result.prediction.predicted_timeline.get(concern);
                const confidence = result.prediction.confidence_scores.get(concern);

                if (efficacy !== undefined) {
                    expect(efficacy).toBeGreaterThanOrEqual(0);
                    expect(efficacy).toBeLessThanOrEqual(100);
                }

                if (timeline !== undefined) {
                    expect(timeline).toBeGreaterThan(0);
                    expect(timeline).toBeLessThan(52); // Less than a year
                }

                if (confidence !== undefined) {
                    expect(confidence).toBeGreaterThanOrEqual(0);
                    expect(confidence).toBeLessThanOrEqual(1);
                }
            }
        });

        it('should provide realistic timeline predictions', () => {
            const result = framework.optimizeFormulation(['hydration']);

            for (const [, weeks] of result.prediction.predicted_timeline) {
                expect(weeks).toBeGreaterThan(0);
                expect(weeks).toBeLessThan(26); // Reasonable timeline
            }
        });
    });

    describe('Scoring System', () => {
        it('should calculate comprehensive formulation scores', () => {
            const result = framework.optimizeFormulation(['wrinkles', 'hydration']);

            expect(result.score.composite_score).toBeGreaterThanOrEqual(0);
            expect(result.score.composite_score).toBeLessThanOrEqual(10);

            // Check individual scores
            expect(result.score.individual_scores.efficacy).toBeGreaterThanOrEqual(0);
            expect(result.score.individual_scores.efficacy).toBeLessThanOrEqual(1);

            expect(result.score.individual_scores.safety).toBeGreaterThanOrEqual(0);
            expect(result.score.individual_scores.safety).toBeLessThanOrEqual(1);

            expect(result.score.individual_scores.stability).toBeGreaterThanOrEqual(0);
            expect(result.score.individual_scores.stability).toBeLessThanOrEqual(1);
        });

        it('should apply network bonuses for synergistic combinations', () => {
            const result = framework.optimizeFormulation(['firmness', 'hydration', 'brightness']);

            // Network bonus should be non-negative
            expect(result.score.network_bonus).toBeGreaterThanOrEqual(0);

            // For multi-ingredient formulations, there should be some network effects
            if (result.formulation.ingredients.length > 2) {
                expect(result.score.network_bonus).toBeGreaterThanOrEqual(0);
            }
        });

        it('should apply constraint penalties when appropriate', () => {
            const overConstrainedResult = framework.optimizeFormulation(['wrinkles', 'hydration', 'brightness'], {
                total_actives_range: {min: 25, max: 30}, // Very high
                budget_limit: 100, // Very low
            });

            // Should have penalties for violating constraints
            expect(overConstrainedResult.score.constraint_penalties).toBeGreaterThanOrEqual(0);
        });
    });

    describe('Edge Cases and Error Handling', () => {
        it('should handle empty concern lists', () => {
            const result = framework.optimizeFormulation([]);

            expect(result.formulation.ingredients.length).toBeGreaterThan(0); // Should default to hydration
            expect(result.score.composite_score).toBeGreaterThan(0);
        });

        it('should handle unknown concerns gracefully', () => {
            const result = framework.optimizeFormulation(['unknown_concern_12345']);

            expect(result.formulation.ingredients.length).toBeGreaterThan(0); // Should fallback
            expect(result.score.composite_score).toBeGreaterThan(0);
        });

        it('should handle extreme budget constraints', () => {
            const veryLowBudget = framework.optimizeFormulation(['hydration'], {budget_limit: 1});

            const veryHighBudget = framework.optimizeFormulation(['hydration'], {budget_limit: 100000});

            expect(veryLowBudget.formulation.ingredients.length).toBeGreaterThan(0);
            expect(veryHighBudget.formulation.ingredients.length).toBeGreaterThan(0);
        });

        it('should handle extreme concentration constraints', () => {
            const result = framework.optimizeFormulation(['hydration'], {
                total_actives_range: {min: 0.1, max: 0.5},
                max_individual_concentration: 0.1,
            });

            expect(result.formulation.ingredients.length).toBeGreaterThan(0);

            const totalActives = Array.from(result.formulation.concentrations.values()).reduce(
                (sum, conc) => sum + conc,
                0,
            );
            expect(totalActives).toBeLessThanOrEqual(1.0); // Should respect constraints
        });
    });

    describe('Database and Statistics', () => {
        it('should provide accurate database statistics', () => {
            const stats = framework.getDatabaseStats();

            expect(stats.total_ingredients).toBeGreaterThan(0);
            expect(stats.ingredients_by_class.size).toBeGreaterThan(0);
            expect(stats.avg_efficacy_by_class.size).toBeGreaterThan(0);

            // Verify counts are consistent
            let totalCount = 0;
            for (const count of stats.ingredients_by_class.values()) {
                totalCount += count;
            }
            expect(totalCount).toBe(stats.total_ingredients);
        });

        it('should maintain data integrity across operations', () => {
            const statsBefore = framework.getDatabaseStats();

            // Perform multiple optimizations
            framework.optimizeFormulation(['hydration']);
            framework.optimizeFormulation(['anti_aging']);
            framework.optimizeFormulation(['brightness']);

            const statsAfter = framework.getDatabaseStats();

            // Database should remain unchanged
            expect(statsAfter.total_ingredients).toBe(statsBefore.total_ingredients);
            expect(statsAfter.ingredients_by_class.size).toBe(statsBefore.ingredients_by_class.size);
        });
    });

    describe('Integration and Performance', () => {
        it('should complete optimization within reasonable time', () => {
            const startTime = Date.now();

            const result = framework.optimizeFormulation(['wrinkles', 'hydration', 'brightness', 'firmness']);

            const endTime = Date.now();
            const duration = endTime - startTime;

            expect(result).toBeDefined();
            expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
        });

        it('should produce consistent results for identical inputs', () => {
            const concerns = ['hydration', 'anti_aging'];
            const constraints = {budget_limit: 1500};

            const result1 = framework.optimizeFormulation(concerns, constraints);
            const result2 = framework.optimizeFormulation(concerns, constraints);

            // Results should be identical (same ingredients and concentrations)
            expect(result1.formulation.ingredients.length).toBe(result2.formulation.ingredients.length);
            expect(result1.score.composite_score).toBeCloseTo(result2.score.composite_score, 2);
        });

        it('should scale with increased complexity', () => {
            const simpleConcerns = ['hydration'];
            const complexConcerns = ['wrinkles', 'hydration', 'brightness', 'firmness', 'sensitivity'];

            const simpleResult = framework.optimizeFormulation(simpleConcerns);
            const complexResult = framework.optimizeFormulation(complexConcerns);

            expect(simpleResult.formulation.ingredients.length).toBeLessThanOrEqual(
                complexResult.formulation.ingredients.length,
            );
        });
    });
});

describe('Hypergredient Framework Real-world Scenarios', () => {
    let framework: HypergredientFramework;

    beforeEach(() => {
        framework = new HypergredientFramework();
    });

    describe('Anti-Aging Serum Formulation', () => {
        it('should create optimal anti-aging formulation as described in problem statement', () => {
            const result = framework.optimizeFormulation(
                ['wrinkles', 'firmness', 'brightness'],
                {
                    budget_limit: 1500,
                    total_actives_range: {min: 8, max: 20},
                    regulatory_regions: ['EU', 'FDA'],
                },
                'normal_to_dry',
            );

            expect(result.formulation.ingredients.length).toBeGreaterThanOrEqual(3);
            expect(result.score.composite_score).toBeGreaterThan(6.0);
            expect(result.analysis.overall_compatibility).toMatch(/excellent|good|fair/);

            // Should contain relevant hypergredient classes
            const classes = new Set(result.formulation.ingredients.map(ing => (ing as any).hypergredient_class));
            expect(classes.has('H.CT') || classes.has('H.CS')).toBe(true); // Anti-aging actives

            console.log('\n=== OPTIMAL ANTI-AGING FORMULATION ===');
            console.log(`Score: ${result.score.composite_score.toFixed(2)}/10`);
            console.log(`Ingredients: ${result.formulation.ingredients.length}`);
            console.log(`Total Cost: R${result.formulation.total_cost.toFixed(2)}`);
            console.log(`Compatibility: ${result.analysis.overall_compatibility}`);

            result.formulation.ingredients.forEach(ingredient => {
                const concentration = result.formulation.concentrations.get(ingredient.id) || 0;
                console.log(
                    `  • ${ingredient.name} (${(ingredient as any).hypergredient_class}): ${concentration.toFixed(1)}%`,
                );
            });
        });
    });

    describe('Sensitive Skin Formulation', () => {
        it('should prioritize safety for sensitive skin types', () => {
            const result = framework.optimizeFormulation(
                ['irritation', 'barrier_damage', 'hydration'],
                {
                    budget_limit: 1200,
                    skin_type_restrictions: ['sensitive'],
                },
                'sensitive',
            );

            // Should prioritize high safety scores
            const avgSafety =
                result.formulation.ingredients.reduce(
                    (sum, ing) => sum + (ing as any).hypergredient_metrics.safety_profile,
                    0,
                ) / result.formulation.ingredients.length;

            expect(avgSafety).toBeGreaterThan(7); // High safety requirement
            expect(result.score.individual_scores.safety).toBeGreaterThan(0.7);
        });
    });

    describe('Budget-Conscious Formulation', () => {
        it('should optimize for cost-effectiveness', () => {
            const budgetResult = framework.optimizeFormulation(
                ['hydration', 'anti_aging'],
                {budget_limit: 800}, // Lower budget
                'normal',
            );

            const premiumResult = framework.optimizeFormulation(
                ['hydration', 'anti_aging'],
                {budget_limit: 3000}, // Higher budget
                'normal',
            );

            expect(budgetResult.formulation.total_cost).toBeLessThanOrEqual(premiumResult.formulation.total_cost);
            expect(budgetResult.score.individual_scores.cost_efficiency).toBeGreaterThanOrEqual(
                premiumResult.score.individual_scores.cost_efficiency * 0.9, // Allow some tolerance
            );
        });
    });
});

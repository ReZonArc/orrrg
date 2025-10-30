/**
 * Test Cases for Meta-Optimization Engine
 *
 * This test suite validates the meta-optimization strategy implementation,
 * including condition-treatment matrix generation, strategy selection,
 * and comprehensive formulation optimization across all possible combinations.
 */

import {beforeEach, describe, expect, it, vi} from 'vitest';
import {MetaOptimizationConfig, MetaOptimizationEngine} from '../../lib/cheminformatics/meta-optimization-engine.js';

describe('Meta-Optimization Engine', () => {
    let metaEngine: MetaOptimizationEngine;
    let testConfig: Partial<MetaOptimizationConfig>;

    beforeEach(() => {
        testConfig = {
            max_combinations: 50, // Limit for testing
            enable_caching: true,
            cache_duration_hours: 1,
            performance_tracking: true,
            parallel_optimization: false, // Disable for testing consistency
            max_parallel_workers: 1,
            strategy_selection_weights: {
                complexity: 0.3,
                performance: 0.4,
                cost: 0.15,
                time: 0.15,
            },
        };

        metaEngine = new MetaOptimizationEngine(testConfig);
    });

    describe('Engine Initialization', () => {
        it('should initialize with default configuration', () => {
            const defaultEngine = new MetaOptimizationEngine();
            expect(defaultEngine).toBeDefined();

            const matrix = defaultEngine.getConditionTreatmentMatrix();
            expect(matrix.combinations.length).toBeGreaterThan(0);
            expect(matrix.conditions.length).toBeGreaterThan(20);
            expect(matrix.treatments.length).toBeGreaterThan(15);
        });

        it('should initialize with custom configuration', () => {
            expect(metaEngine).toBeDefined();

            const matrix = metaEngine.getConditionTreatmentMatrix();
            expect(matrix.combinations.length).toBeLessThanOrEqual(testConfig.max_combinations!);
        });

        it('should have comprehensive condition and treatment coverage', () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();

            // Check key conditions are included
            const expectedConditions = [
                'wrinkles',
                'aging',
                'dryness',
                'hyperpigmentation',
                'acne',
                'sensitive_skin',
                'oily_skin',
            ];
            expectedConditions.forEach(condition => {
                expect(matrix.conditions).toContain(condition);
            });

            // Check key treatments are included
            const expectedTreatments = [
                'hydration',
                'anti_aging',
                'brightening',
                'acne_treatment',
                'anti_inflammatory',
                'sebum_regulation',
            ];
            expectedTreatments.forEach(treatment => {
                expect(matrix.treatments).toContain(treatment);
            });
        });
    });

    describe('Condition-Treatment Matrix Generation', () => {
        it('should generate meaningful condition-treatment combinations', () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();

            expect(matrix.combinations.length).toBeGreaterThan(10);

            // Check combination structure
            matrix.combinations.forEach(combo => {
                expect(combo.id).toBeDefined();
                expect(combo.conditions.length).toBeGreaterThan(0);
                expect(combo.treatments.length).toBeGreaterThan(0);
                expect(combo.complexity_score).toBeGreaterThanOrEqual(0);
                expect(combo.complexity_score).toBeLessThanOrEqual(10);
                expect(['hypergredient', 'multiscale', 'hybrid', 'custom']).toContain(combo.recommended_strategy);
            });
        });

        it('should assign appropriate complexity scores', () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();

            // Simple combinations should have lower complexity
            const simpleCombos = matrix.combinations.filter(
                c => c.conditions.length === 1 && c.treatments.length === 1,
            );
            const complexCombos = matrix.combinations.filter(c => c.conditions.length > 1 || c.treatments.length > 2);

            if (simpleCombos.length > 0 && complexCombos.length > 0) {
                const avgSimpleComplexity =
                    simpleCombos.reduce((sum, c) => sum + c.complexity_score, 0) / simpleCombos.length;
                const avgComplexComplexity =
                    complexCombos.reduce((sum, c) => sum + c.complexity_score, 0) / complexCombos.length;

                expect(avgComplexComplexity).toBeGreaterThan(avgSimpleComplexity);
            }
        });

        it('should recommend appropriate strategies based on complexity', () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();

            // Simple cases should prefer hypergredient
            const lowComplexityCombos = matrix.combinations.filter(c => c.complexity_score <= 3);
            const highComplexityCombos = matrix.combinations.filter(c => c.complexity_score > 6);

            if (lowComplexityCombos.length > 0) {
                const hypergredientRatio =
                    lowComplexityCombos.filter(c => c.recommended_strategy === 'hypergredient').length /
                    lowComplexityCombos.length;
                expect(hypergredientRatio).toBeGreaterThan(0.5);
            }

            if (highComplexityCombos.length > 0) {
                const simpleStrategyRatio =
                    highComplexityCombos.filter(c => c.recommended_strategy === 'hypergredient').length /
                    highComplexityCombos.length;
                expect(simpleStrategyRatio).toBeLessThan(0.5);
            }
        });
    });

    describe('Single Combination Optimization', () => {
        it('should optimize a simple anti-aging combination', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const antiAgingCombo = matrix.combinations.find(
                c => c.conditions.includes('wrinkles') && c.treatments.includes('anti_aging'),
            );

            expect(antiAgingCombo).toBeDefined();

            const result = await metaEngine.optimizeForCombination(antiAgingCombo!.id, 'mature', {budget_limit: 100});

            expect(result).toBeDefined();
            expect(result.combination_id).toBe(antiAgingCombo!.id);
            expect(result.optimal_formulation).toBeDefined();
            expect(result.optimal_formulation.ingredients.length).toBeGreaterThan(0);
            expect(result.performance_metrics.optimization_score).toBeGreaterThan(0);
            expect(result.performance_metrics.execution_time_ms).toBeGreaterThan(0);
            expect(result.confidence_score).toBeGreaterThanOrEqual(0);
            expect(Array.isArray(result.recommendations)).toBe(true);
        });

        it('should optimize a complex multi-condition combination', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const complexCombo = matrix.combinations.find(c => c.conditions.length > 1 && c.treatments.length > 1);

            if (complexCombo) {
                const result = await metaEngine.optimizeForCombination(complexCombo.id, 'combination', {
                    budget_limit: 150,
                });

                expect(result).toBeDefined();
                expect(result.strategy_used).toBeDefined();
                expect(['hypergredient', 'multiscale', 'hybrid', 'custom']).toContain(result.strategy_used);
                expect(result.optimal_formulation.ingredients.length).toBeGreaterThan(2);
            }
        });

        it('should use appropriate strategy for different complexity levels', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();

            // Test simple case - should use hypergredient
            const simpleCombo = matrix.combinations.find(c => c.complexity_score <= 3);
            if (simpleCombo) {
                const simpleResult = await metaEngine.optimizeForCombination(simpleCombo.id);
                expect(['hypergredient', 'hybrid']).toContain(simpleResult.strategy_used);
            }

            // Test complex case - should use multiscale or hybrid
            const complexCombo = matrix.combinations.find(c => c.complexity_score > 6);
            if (complexCombo) {
                const complexResult = await metaEngine.optimizeForCombination(complexCombo.id);
                expect(['multiscale', 'hybrid', 'custom']).toContain(complexResult.strategy_used);
            }
        });

        it('should handle sensitive skin combinations appropriately', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const sensitiveCombo = matrix.combinations.find(c => c.conditions.includes('sensitive_skin'));

            if (sensitiveCombo) {
                const result = await metaEngine.optimizeForCombination(sensitiveCombo.id, 'sensitive');

                expect(result.recommendations).toContain('Use gentle, hypoallergenic base ingredients');
                expect(result.optimal_formulation.ingredients.length).toBeLessThan(10); // Should be simpler
            }
        });
    });

    describe('Caching and Performance', () => {
        it('should cache optimization results', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const testCombo = matrix.combinations[0];

            // First optimization
            const startTime = Date.now();
            const result1 = await metaEngine.optimizeForCombination(testCombo.id);
            const firstExecutionTime = Date.now() - startTime;

            // Second optimization (should be cached)
            const startTime2 = Date.now();
            const result2 = await metaEngine.optimizeForCombination(testCombo.id);
            const secondExecutionTime = Date.now() - startTime2;

            expect(result1.combination_id).toBe(result2.combination_id);
            expect(result1.performance_metrics.optimization_score).toBe(result2.performance_metrics.optimization_score);
            // Both results should be identical since the second one comes from cache
            expect(result1.optimal_formulation.id).toBe(result2.optimal_formulation.id);
        });

        it('should track performance by strategy', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();

            // Optimize a few combinations
            const testCombos = matrix.combinations.slice(0, 3);
            for (const combo of testCombos) {
                await metaEngine.optimizeForCombination(combo.id);
            }

            const analytics = metaEngine.getPerformanceAnalytics();
            expect(analytics.size).toBeGreaterThan(0);

            // Check that each strategy has recorded performance data
            analytics.forEach((scores, strategy) => {
                expect(scores.length).toBeGreaterThan(0);
                scores.forEach(score => {
                    expect(score).toBeGreaterThanOrEqual(0);
                    expect(score).toBeLessThanOrEqual(10);
                });
            });
        });

        it('should clear cache when requested', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const testCombo = matrix.combinations[0];

            // Optimize and cache
            await metaEngine.optimizeForCombination(testCombo.id);

            // Clear cache
            metaEngine.clearCache();

            // Should not throw error
            expect(() => metaEngine.clearCache()).not.toThrow();
        });
    });

    describe('Strategy Selection Logic', () => {
        it('should select hypergredient strategy for simple cases', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const simpleCombo = matrix.combinations.find(
                c => c.conditions.length === 1 && c.treatments.length === 1 && c.complexity_score <= 3,
            );

            if (simpleCombo) {
                expect(simpleCombo.recommended_strategy).toBe('hypergredient');
            }
        });

        it('should select multiscale strategy for medium complexity', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const mediumCombo = matrix.combinations.find(c => c.complexity_score > 3 && c.complexity_score <= 6);

            if (mediumCombo) {
                expect(['multiscale', 'hybrid']).toContain(mediumCombo.recommended_strategy);
            }
        });

        it('should select hybrid or custom strategy for high complexity', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const complexCombo = matrix.combinations.find(c => c.complexity_score > 6);

            if (complexCombo) {
                expect(['hybrid', 'custom']).toContain(complexCombo.recommended_strategy);
            }
        });
    });

    describe('Comprehensive Optimization', () => {
        it('should optimize all combinations efficiently', async () => {
            // Use a small subset for testing
            const limitedEngine = new MetaOptimizationEngine({
                max_combinations: 5,
                enable_caching: true,
                performance_tracking: true,
            });

            const summary = await limitedEngine.optimizeAllCombinations('normal', {
                budget_limit: 100,
            });

            expect(summary).toBeDefined();
            expect(summary.total_combinations).toBe(5);
            expect(summary.successful_optimizations).toBeGreaterThan(0);
            expect(summary.successful_optimizations).toBeLessThanOrEqual(5);

            expect(summary.strategy_distribution.size).toBeGreaterThan(0);
            expect(summary.average_performance_by_strategy.size).toBeGreaterThan(0);
            expect(summary.top_performing_combinations.length).toBeGreaterThan(0);

            expect(summary.performance_analytics.best_overall_score).toBeGreaterThanOrEqual(0);
            expect(summary.performance_analytics.average_score).toBeGreaterThanOrEqual(0);
            expect(summary.performance_analytics.strategy_efficiency.size).toBeGreaterThan(0);
        });

        it('should provide meaningful performance analytics', async () => {
            const limitedEngine = new MetaOptimizationEngine({
                max_combinations: 3,
                enable_caching: true,
                performance_tracking: true,
            });

            const summary = await limitedEngine.optimizeAllCombinations();

            // Check strategy distribution
            let totalStrategies = 0;
            summary.strategy_distribution.forEach(count => {
                totalStrategies += count;
                expect(count).toBeGreaterThan(0);
            });
            expect(totalStrategies).toBe(summary.successful_optimizations);

            // Check average performance makes sense
            summary.average_performance_by_strategy.forEach(avgScore => {
                expect(avgScore).toBeGreaterThanOrEqual(0);
                expect(avgScore).toBeLessThanOrEqual(10);
            });

            // Check top performing combinations
            expect(summary.top_performing_combinations.length).toBeLessThanOrEqual(summary.successful_optimizations);

            // Should be sorted by performance
            for (let i = 1; i < summary.top_performing_combinations.length; i++) {
                expect(
                    summary.top_performing_combinations[i - 1].performance_metrics.optimization_score,
                ).toBeGreaterThanOrEqual(summary.top_performing_combinations[i].performance_metrics.optimization_score);
            }
        });

        it('should handle optimization failures gracefully', async () => {
            // Mock a failing optimization to test error handling
            const spy = vi.spyOn(metaEngine, 'optimizeForCombination');
            spy.mockRejectedValueOnce(new Error('Simulated failure'));

            const limitedEngine = new MetaOptimizationEngine({
                max_combinations: 2,
                enable_caching: false,
            });

            // Should not throw, but handle gracefully
            const summary = await limitedEngine.optimizeAllCombinations();

            expect(summary.total_combinations).toBe(2);
            // Even with failures, should provide valid summary
            expect(summary).toBeDefined();
            expect(typeof summary.successful_optimizations).toBe('number');
            expect(summary.successful_optimizations).toBeLessThanOrEqual(summary.total_combinations);

            spy.mockRestore();
        });
    });

    describe('Integration with Existing Frameworks', () => {
        it('should properly integrate with HypergredientFramework', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const hypergredientCombo = matrix.combinations.find(c => c.recommended_strategy === 'hypergredient');

            if (hypergredientCombo) {
                const result = await metaEngine.optimizeForCombination(hypergredientCombo.id);

                expect(result.strategy_used).toBe('hypergredient');
                expect(result.optimal_formulation).toBeDefined();
                expect(result.optimal_formulation.ingredients.length).toBeGreaterThan(0);
                expect(result.performance_metrics.optimization_score).toBeGreaterThan(0);
            }
        });

        it('should properly integrate with MultiscaleOptimizer', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const multiscaleCombo = matrix.combinations.find(c => c.recommended_strategy === 'multiscale');

            if (multiscaleCombo) {
                const result = await metaEngine.optimizeForCombination(multiscaleCombo.id);

                expect(result.strategy_used).toBe('multiscale');
                expect(result.optimal_formulation).toBeDefined();
                expect(result.performance_metrics.iterations).toBeGreaterThan(0);
            }
        });

        it('should provide alternative formulations for hybrid strategy', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const hybridCombo = matrix.combinations.find(c => c.recommended_strategy === 'hybrid');

            if (hybridCombo) {
                const result = await metaEngine.optimizeForCombination(hybridCombo.id);

                expect(result.strategy_used).toBe('hybrid');
                expect(result.alternative_formulations).toBeDefined();
                expect(result.alternative_formulations!.length).toBe(2); // Both hypergredient and multiscale results
            }
        });
    });

    describe('Recommendation System', () => {
        it('should provide contextual recommendations', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();

            // Test sensitive skin recommendations
            const sensitiveCombo = matrix.combinations.find(c => c.conditions.includes('sensitive_skin'));

            if (sensitiveCombo) {
                const result = await metaEngine.optimizeForCombination(sensitiveCombo.id);
                expect(result.recommendations.some(r => r.includes('gentle'))).toBe(true);
            }

            // Test cellular turnover recommendations
            const cellularCombo = matrix.combinations.find(c => c.treatments.includes('cellular_turnover'));

            if (cellularCombo) {
                const result = await metaEngine.optimizeForCombination(cellularCombo.id);
                expect(result.recommendations.some(r => r.includes('gradually'))).toBe(true);
            }
        });

        it('should recommend patch testing for complex formulations', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();
            const complexCombo = matrix.combinations.find(c => c.complexity_score > 7);

            if (complexCombo) {
                const result = await metaEngine.optimizeForCombination(complexCombo.id);
                expect(result.recommendations.some(r => r.includes('patch testing'))).toBe(true);
            }
        });

        it('should recommend formulation simplification when appropriate', async () => {
            const matrix = metaEngine.getConditionTreatmentMatrix();

            // Find a combination that might result in many ingredients
            const multiTreatmentCombo = matrix.combinations.find(c => c.treatments.length > 2);

            if (multiTreatmentCombo) {
                const result = await metaEngine.optimizeForCombination(multiTreatmentCombo.id);

                if (result.optimal_formulation.ingredients.length > 8) {
                    expect(result.recommendations.some(r => r.includes('simplifying'))).toBe(true);
                }
            }
        });
    });
});

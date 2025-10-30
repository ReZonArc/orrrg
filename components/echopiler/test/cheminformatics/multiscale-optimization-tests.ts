/**
 * Test Cases for Multiscale Constraint Optimization in Cosmeceutical Formulation
 *
 * This test suite validates the implementation of OpenCog-inspired features for
 * multiscale constraint optimization, including INCI-driven search space reduction,
 * adaptive attention allocation, and recursive optimization pathways.
 */

import {beforeEach, describe, expect, it} from 'vitest';
import {AdaptiveAttentionAllocator, AttentionAtom} from '../../lib/cheminformatics/adaptive-attention-allocator.js';
import {
    INCISearchSpaceReducer,
    INCIUtilities,
    SearchSpaceReductionConfig,
} from '../../lib/cheminformatics/inci-search-space-reducer.js';
import {
    MultiscaleOptimizationConfig,
    MultiscaleOptimizer,
    OptimizationContext,
} from '../../lib/cheminformatics/multiscale-optimizer.js';
import {CosmeticFormulation, CosmeticIngredient} from '../../types/cheminformatics/cosmetic-chemistry.interfaces.js';

describe('INCI-Driven Search Space Reduction', () => {
    let reducer: INCISearchSpaceReducer;
    let testConfig: SearchSpaceReductionConfig;

    beforeEach(() => {
        reducer = new INCISearchSpaceReducer();
        testConfig = {
            max_ingredients: 8,
            max_total_actives_concentration: 20.0,
            target_therapeutic_vectors: ['anti_aging', 'hydration'],
            skin_penetration_requirements: ['stratum_corneum', 'epidermis'],
            stability_requirements: ['oxidation_resistant', 'ph_stable'],
            cost_constraints: {min: 0.5, max: 15.0},
            regulatory_regions: ['EU', 'FDA'],
        };
    });

    describe('Search Space Reduction Algorithm', () => {
        it('should reduce ingredient search space based on INCI constraints', async () => {
            const targetFormulation = {type: 'SKINCARE_FORMULATION' as const};

            const result = await reducer.reduceSearchSpace(targetFormulation, testConfig);

            expect(result.reduced_search_space).toBeDefined();
            expect(result.reduced_search_space.length).toBeLessThanOrEqual(testConfig.max_ingredients);
            expect(result.optimization_metrics.space_reduction_ratio).toBeGreaterThan(0);
        });

        it('should estimate concentrations from INCI ordering', async () => {
            const targetFormulation = {type: 'SKINCARE_FORMULATION' as const};

            const result = await reducer.reduceSearchSpace(targetFormulation, testConfig);

            expect(result.estimated_concentrations).toBeDefined();
            expect(result.estimated_concentrations.size).toBeGreaterThan(0);

            // Verify total concentration is within limits
            const totalConcentration = Array.from(result.estimated_concentrations.values()).reduce(
                (sum, conc) => sum + conc,
                0,
            );
            expect(totalConcentration).toBeLessThanOrEqual(testConfig.max_total_actives_concentration);
        });

        it('should calculate therapeutic vector coverage', async () => {
            const targetFormulation = {type: 'SKINCARE_FORMULATION' as const};

            const result = await reducer.reduceSearchSpace(targetFormulation, testConfig);

            expect(result.therapeutic_vector_coverage).toBeDefined();
            testConfig.target_therapeutic_vectors.forEach(vector => {
                expect(result.therapeutic_vector_coverage.has(vector)).toBe(true);
                expect(result.therapeutic_vector_coverage.get(vector)).toBeGreaterThanOrEqual(0);
                expect(result.therapeutic_vector_coverage.get(vector)!).toBeLessThanOrEqual(1);
            });
        });

        it('should generate synergy matrix for ingredient combinations', async () => {
            const targetFormulation = {type: 'SKINCARE_FORMULATION' as const};

            const result = await reducer.reduceSearchSpace(targetFormulation, testConfig);

            expect(result.synergy_matrix).toBeDefined();
            expect(result.synergy_matrix.size).toBeGreaterThan(0);

            // Verify matrix symmetry and valid values
            result.synergy_matrix.forEach((row, ingredient1) => {
                row.forEach((synergyValue, ingredient2) => {
                    expect(synergyValue).toBeGreaterThanOrEqual(0);
                    expect(synergyValue).toBeLessThanOrEqual(1);

                    // Check reciprocal relationship exists
                    const reciprocal = result.synergy_matrix.get(ingredient2)?.get(ingredient1);
                    if (reciprocal !== undefined) {
                        expect(Math.abs(synergyValue - reciprocal)).toBeLessThan(0.1);
                    }
                });
            });
        });

        it('should enforce regulatory compliance across regions', async () => {
            const targetFormulation = {type: 'SKINCARE_FORMULATION' as const};

            const result = await reducer.reduceSearchSpace(targetFormulation, testConfig);

            expect(result.regulatory_compliance_score).toBeGreaterThan(0.8);

            // All ingredients should be compliant with specified regions
            result.reduced_search_space.forEach(ingredient => {
                testConfig.regulatory_regions.forEach(region => {
                    const status = ingredient.regulatory_status?.get(region);
                    expect(status).toBe('approved');
                });
            });
        });
    });

    describe('INCI Utilities', () => {
        it('should parse INCI list from product labeling', () => {
            const inciString = 'Aqua, Glycerin, Niacinamide, Hyaluronic Acid, Phenoxyethanol';
            const parsed = INCIUtilities.parseINCIList(inciString);

            expect(parsed).toEqual(['aqua', 'glycerin', 'niacinamide', 'hyaluronic acid', 'phenoxyethanol']);
        });

        it('should validate INCI ordering based on concentration rules', () => {
            const inciList = ['aqua', 'glycerin', 'niacinamide', 'phenoxyethanol'];
            const concentrations = new Map([
                ['aqua', 70.0],
                ['glycerin', 10.0],
                ['niacinamide', 5.0],
                ['phenoxyethanol', 0.5],
            ]);

            const isValid = INCIUtilities.validateINCIOrdering(inciList, concentrations);
            expect(isValid).toBe(true);

            // Test invalid ordering
            const invalidConcentrations = new Map([
                ['aqua', 70.0],
                ['glycerin', 5.0], // Should be higher than niacinamide
                ['niacinamide', 10.0],
                ['phenoxyethanol', 0.5],
            ]);

            const isInvalid = INCIUtilities.validateINCIOrdering(inciList, invalidConcentrations);
            expect(isInvalid).toBe(false);
        });

        it('should estimate concentrations from INCI ordering using Zipf distribution', () => {
            const inciList = ['aqua', 'glycerin', 'niacinamide', 'retinol'];
            const estimated = INCIUtilities.estimateConcentrationsFromOrdering(inciList, 100);

            expect(estimated.size).toBe(4);

            // Verify descending order
            const concentrations = Array.from(estimated.values());
            for (let i = 0; i < concentrations.length - 1; i++) {
                expect(concentrations[i]).toBeGreaterThanOrEqual(concentrations[i + 1]);
            }

            // Verify total sums to 100 (approximately)
            const total = concentrations.reduce((sum, conc) => sum + conc, 0);
            expect(Math.abs(total - 100)).toBeLessThan(0.01);
        });
    });
});

describe('Adaptive Attention Allocation', () => {
    let allocator: AdaptiveAttentionAllocator;
    let testAtoms: AttentionAtom[];

    beforeEach(() => {
        allocator = new AdaptiveAttentionAllocator({
            max_attention_atoms: 100,
            sti_decay_rate: 0.1,
            lti_decay_rate: 0.01,
            vlti_decay_rate: 0.001,
            attention_threshold: 10,
            reinforcement_factor: 1.5,
            exploration_factor: 0.1,
            cost_penalty_factor: 0.2,
            market_weight: 0.3,
            regulatory_weight: 0.4,
        });

        testAtoms = [
            {
                id: 'hyaluronic_acid_optimization',
                type: 'ingredient',
                content: {ingredient_id: 'hyaluronic_acid'},
                short_term_importance: 200,
                long_term_importance: 150,
                very_long_term_importance: 100,
                attention_value: 0,
                last_accessed: new Date(),
                access_count: 5,
                creation_time: new Date(),
                confidence: 0.8,
                utility: 0.7,
                cost: 1.2,
                market_relevance: 0.9,
                regulatory_risk: 0.2,
            },
            {
                id: 'retinol_vitamin_c_synergy',
                type: 'combination',
                content: {ingredients: ['retinol', 'vitamin_c']},
                short_term_importance: 100,
                long_term_importance: 300,
                very_long_term_importance: 200,
                attention_value: 0,
                last_accessed: new Date(),
                access_count: 2,
                creation_time: new Date(),
                confidence: 0.6,
                utility: 0.9,
                cost: 2.5,
                market_relevance: 0.8,
                regulatory_risk: 0.4,
            },
        ];
    });

    describe('Attention Allocation System', () => {
        it('should compute attention values based on multiple factors', () => {
            testAtoms.forEach(atom => {
                allocator.addAttentionAtom(atom);
            });

            const distribution = allocator.allocateAttention();

            expect(distribution.high_attention.length).toBeGreaterThan(0);
            expect(distribution.focus_areas.length).toBeGreaterThan(0);
            expect(distribution.resource_allocation.size).toBeGreaterThan(0);
        });

        it('should prioritize atoms with high market relevance and low regulatory risk', () => {
            const highMarketAtom: AttentionAtom = {
                id: 'market_opportunity_test',
                type: 'market_opportunity',
                content: {opportunity: 'clean_beauty'},
                short_term_importance: 150,
                long_term_importance: 200,
                very_long_term_importance: 100,
                attention_value: 0,
                last_accessed: new Date(),
                access_count: 1,
                creation_time: new Date(),
                confidence: 0.9,
                utility: 0.8,
                cost: 1.0,
                market_relevance: 0.95,
                regulatory_risk: 0.1,
            };

            const lowMarketAtom: AttentionAtom = {
                ...highMarketAtom,
                id: 'low_market_test',
                market_relevance: 0.3,
                regulatory_risk: 0.8,
            };

            allocator.addAttentionAtom(highMarketAtom);
            allocator.addAttentionAtom(lowMarketAtom);

            const distribution = allocator.allocateAttention();
            const highAttentionIds = distribution.high_attention.map(atom => atom.id);

            expect(highAttentionIds).toContain('market_opportunity_test');
            expect(
                distribution.high_attention.find(atom => atom.id === 'market_opportunity_test')?.attention_value,
            ).toBeGreaterThan(
                distribution.high_attention.find(atom => atom.id === 'low_market_test')?.attention_value || 0,
            );
        });

        it('should implement attention decay over time', () => {
            const atom = testAtoms[0];
            atom.last_accessed = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24 hours ago

            allocator.addAttentionAtom(atom);
            const initialAttention = atom.attention_value;

            // Simulate time passage
            allocator.updateAttentionDecay();

            const currentAttention = atom.attention_value;
            expect(currentAttention).toBeLessThan(initialAttention);
        });

        it('should reinforce attention for successful computations', () => {
            const atom = testAtoms[0];
            allocator.addAttentionAtom(atom);

            const initialSTI = atom.short_term_importance;
            const initialConfidence = atom.confidence;

            allocator.reinforceAttention(atom.id, true, 1.0);

            expect(atom.short_term_importance).toBeGreaterThan(initialSTI);
            expect(atom.confidence).toBeGreaterThan(initialConfidence);
        });

        it('should penalize attention for failed computations', () => {
            const atom = testAtoms[0];
            allocator.addAttentionAtom(atom);

            const initialSTI = atom.short_term_importance;
            const initialConfidence = atom.confidence;

            allocator.reinforceAttention(atom.id, false, 1.0);

            expect(atom.short_term_importance).toBeLessThan(initialSTI);
            expect(atom.confidence).toBeLessThan(initialConfidence);
        });
    });

    describe('Market Opportunity Integration', () => {
        it('should update attention based on market opportunities', () => {
            allocator.updateMarketOpportunityAttention();

            const distribution = allocator.allocateAttention();
            const marketAtoms = distribution.high_attention.filter(
                atom =>
                    atom.type === 'market_opportunity' ||
                    (atom.type === 'ingredient' && atom.content.market_opportunity),
            );

            expect(marketAtoms.length).toBeGreaterThan(0);
        });

        it('should prioritize high-growth, low-competition opportunities', () => {
            allocator.updateMarketOpportunityAttention();

            const distribution = allocator.allocateAttention();
            const focusAreas = distribution.focus_areas;

            expect(focusAreas).toContain('market_innovation');
        });
    });

    describe('Regulatory Compliance Attention', () => {
        it('should allocate attention to regulatory compliance based on risk', () => {
            const testIngredients: CosmeticIngredient[] = [
                {
                    id: 'high_risk_ingredient',
                    name: 'High Risk Test',
                    inci_name: 'Test Chemical',
                    category: 'ACTIVE_INGREDIENT',
                    subtype: 'SYNTHETIC_ACTIVE',
                    functions: ['anti_aging'],
                    molecularWeight: 500,
                    solubility: 'oil_soluble',
                    ph_stability_range: {min: 5.0, max: 7.0},
                    concentration_range: {min: 0.1, max: 2.0},
                    allergenicity: 'high',
                    pregnancy_safe: false,
                    therapeutic_vectors: ['anti_aging'],
                    skin_penetration_depth: 'epidermis',
                    onset_time_hours: 24,
                    duration_hours: 48,
                    stability_factors: ['light_sensitive'],
                    regulatory_status: new Map([
                        ['EU', 'pending'],
                        ['FDA', 'restricted'],
                    ]),
                    evidence_level: 'theoretical',
                    cost_per_gram: 5.0,
                },
            ];

            allocator.updateRegulatoryAttention(testIngredients);

            const distribution = allocator.allocateAttention();
            const regulatoryAtoms = distribution.high_attention.filter(atom => atom.type === 'constraint');

            expect(regulatoryAtoms.length).toBeGreaterThan(0);
            expect(regulatoryAtoms.some(atom => atom.regulatory_risk > 0.5)).toBe(true);
        });
    });

    describe('Attention Statistics and Insights', () => {
        it('should provide comprehensive attention statistics', () => {
            testAtoms.forEach(atom => allocator.addAttentionAtom(atom));

            const stats = allocator.getAttentionStatistics();

            expect(stats.total_atoms).toBeGreaterThan(0);
            expect(stats.attention_distribution.high).toBeGreaterThanOrEqual(0);
            expect(stats.attention_distribution.medium).toBeGreaterThanOrEqual(0);
            expect(stats.attention_distribution.low).toBeGreaterThanOrEqual(0);
            expect(stats.top_focus_areas.length).toBeGreaterThan(0);
            expect(stats.computational_efficiency).toBeGreaterThanOrEqual(0);
            expect(stats.computational_efficiency).toBeLessThanOrEqual(1);
        });
    });
});

describe('Multiscale Optimization Engine', () => {
    let optimizer: MultiscaleOptimizer;
    let testContext: OptimizationContext;
    let testConfig: MultiscaleOptimizationConfig;

    beforeEach(() => {
        optimizer = new MultiscaleOptimizer();

        testContext = {
            target_skin_type: 'normal',
            environmental_conditions: new Map([
                ['temperature', 25],
                ['humidity', 60],
                ['uv_index', 5],
            ]),
            user_preferences: new Map([
                ['texture', 0.8],
                ['absorption', 0.9],
                ['fragrance', 0.3],
            ]),
            regulatory_regions: ['EU', 'FDA'],
            budget_constraints: {min: 2.0, max: 20.0},
            time_constraints: 180, // 6 months
            market_positioning: 'premium',
        };

        testConfig = {
            max_iterations: 50,
            convergence_threshold: 0.001,
            exploration_probability: 0.2,
            local_search_intensity: 0.7,
            global_search_scope: 0.3,
            constraint_penalty_weight: 2.0,
            synergy_reward_weight: 1.5,
            stability_weight: 1.0,
            cost_weight: 0.8,
            efficacy_weight: 2.0,
        };
    });

    describe('Multiscale Formulation Optimization', () => {
        it('should optimize formulation for anti-aging therapeutic outcomes', async () => {
            const targetOutcomes = ['anti_aging', 'barrier_enhancement'];

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            expect(result.optimized_formulation).toBeDefined();
            expect(result.optimization_score).toBeGreaterThan(0);
            expect(result.optimized_formulation.ingredients.length).toBeGreaterThan(0);
            expect(result.optimized_formulation.ingredients.length).toBeLessThanOrEqual(12);
        });

        it('should satisfy regulatory compliance constraints', async () => {
            const targetOutcomes = ['hydration', 'pigmentation_control'];

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            // Check regulatory compliance
            result.regulatory_compliance.forEach((compliance, region) => {
                expect(compliance).toBeGreaterThan(0.8); // At least 80% compliance
            });

            // Check constraint satisfaction
            const regulatoryConstraint = result.constraint_satisfaction.get('regulatory_compliance');
            expect(regulatoryConstraint?.satisfied).toBe(true);
        });

        it('should achieve therapeutic efficacy across target vectors', async () => {
            const targetOutcomes = ['collagen_synthesis_stimulation', 'barrier_enhancement'];

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            // Check therapeutic efficacy
            result.therapeutic_efficacy.forEach((efficacy, action) => {
                expect(efficacy).toBeGreaterThan(0.3); // Minimum therapeutic threshold
            });

            expect(result.predicted_stability).toBeGreaterThan(0.7);
        });

        it('should respect cost constraints', async () => {
            const targetOutcomes = ['anti_aging'];

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            expect(result.estimated_cost).toBeLessThanOrEqual(testContext.budget_constraints.max);
            expect(result.estimated_cost).toBeGreaterThanOrEqual(testContext.budget_constraints.min);

            const costConstraint = result.constraint_satisfaction.get('cost_effectiveness');
            expect(costConstraint?.satisfaction_degree).toBeGreaterThan(0.5);
        });

        it('should generate synergy matrix for ingredient interactions', async () => {
            const targetOutcomes = ['anti_aging', 'hydration'];

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            expect(result.synergy_matrix.size).toBeGreaterThan(0);

            // Verify synergy values are within valid range
            result.synergy_matrix.forEach((row, ingredient1) => {
                row.forEach((synergyValue, ingredient2) => {
                    expect(synergyValue).toBeGreaterThanOrEqual(0);
                    expect(synergyValue).toBeLessThanOrEqual(1);
                });
            });
        });

        it('should provide detailed optimization trace', async () => {
            const targetOutcomes = ['barrier_enhancement'];

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            expect(result.optimization_trace.length).toBeGreaterThan(0);
            expect(result.convergence_metrics.iterations_to_convergence).toBeGreaterThan(0);
            expect(result.convergence_metrics.final_score).toBe(result.optimization_score);

            // Verify trace contains valid optimization steps
            result.optimization_trace.forEach(step => {
                expect(step.iteration).toBeGreaterThanOrEqual(0);
                expect([
                    'add_ingredient',
                    'remove_ingredient',
                    'adjust_concentration',
                    'local_search',
                    'global_jump',
                ]).toContain(step.action);
                expect(step.reasoning).toBeTruthy();
            });
        });
    });

    describe('Constraint Satisfaction', () => {
        it('should handle incompatible ingredient combinations', async () => {
            const targetOutcomes = ['anti_aging']; // May include retinol + vitamin C

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            const compatibilityConstraint = result.constraint_satisfaction.get('ingredient_compatibility');
            expect(compatibilityConstraint?.satisfied).toBe(true);
        });

        it('should enforce total actives concentration limits', async () => {
            const targetOutcomes = ['anti_aging', 'hydration', 'pigmentation_control'];

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            const totalActives = Array.from(result.optimized_formulation.concentrations.values()).reduce(
                (sum, conc) => sum + conc,
                0,
            );

            expect(totalActives).toBeLessThanOrEqual(25.0); // Maximum safe limit

            const activesConstraint = result.constraint_satisfaction.get('total_actives_limit');
            expect(activesConstraint?.satisfied).toBe(true);
        });

        it('should maximize synergistic therapeutic effects', async () => {
            const targetOutcomes = ['anti_aging', 'barrier_enhancement'];

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            const synergyConstraint = result.constraint_satisfaction.get('therapeutic_synergy');
            expect(synergyConstraint?.satisfaction_degree).toBeGreaterThan(0.5);

            // Verify actual synergistic combinations exist
            let synergyFound = false;
            result.synergy_matrix.forEach(row => {
                row.forEach(synergyValue => {
                    if (synergyValue > 0.7) synergyFound = true;
                });
            });
            expect(synergyFound).toBe(true);
        });
    });

    describe('Multiscale Skin Model Integration', () => {
        it('should optimize for different skin layer targets', async () => {
            const targetOutcomes = ['collagen_synthesis_stimulation', 'melanin_inhibition'];

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            // Should contain ingredients targeting different skin layers
            const targetLayers = new Set<string>();
            result.optimized_formulation.ingredients.forEach(ingredient => {
                if (ingredient.skin_penetration_depth) {
                    targetLayers.add(ingredient.skin_penetration_depth);
                }
            });

            expect(targetLayers.size).toBeGreaterThan(1); // Multiple skin layers targeted
        });

        it('should consider penetration requirements for deep dermal targets', async () => {
            const targetOutcomes = ['collagen_synthesis_stimulation']; // Requires dermal penetration

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            const hasDeepPenetrating = result.optimized_formulation.ingredients.some(
                ingredient => ingredient.skin_penetration_depth === 'dermis' || ingredient.molecularWeight! < 1000,
            );

            expect(hasDeepPenetrating).toBe(true);
        });
    });

    // Performance and convergence tests
    describe('Optimization Performance', () => {
        it('should converge within reasonable iteration limits', async () => {
            const targetOutcomes = ['hydration'];
            const quickConfig = {...testConfig, max_iterations: 20, convergence_threshold: 0.01};

            const startTime = Date.now();
            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, quickConfig);
            const duration = Date.now() - startTime;

            expect(result.convergence_metrics.iterations_to_convergence).toBeLessThanOrEqual(20);
            expect(duration).toBeLessThan(10000); // Should complete within 10 seconds
        });

        it('should show improvement over optimization iterations', async () => {
            const targetOutcomes = ['anti_aging'];

            const result = await optimizer.optimizeFormulation(targetOutcomes, testContext, testConfig);

            // Verify score improvement trend
            const trace = result.optimization_trace;
            if (trace.length > 5) {
                const earlyScore = trace.slice(0, 5).reduce((sum, step) => sum + step.score_after, 0) / 5;
                const lateScore = trace.slice(-5).reduce((sum, step) => sum + step.score_after, 0) / 5;

                expect(lateScore).toBeGreaterThanOrEqual(earlyScore * 0.95); // Allow some variation
            }
        });
    });
});

// Integration tests combining all components
describe('Integrated Multiscale Optimization Pipeline', () => {
    let optimizer: MultiscaleOptimizer;

    beforeEach(() => {
        optimizer = new MultiscaleOptimizer();
    });

    it('should execute complete optimization pipeline for premium anti-aging formulation', async () => {
        // Step 1: Define comprehensive optimization target
        const targetOutcomes = ['anti_aging', 'barrier_enhancement', 'hydration'];
        const context: OptimizationContext = {
            target_skin_type: 'mature',
            environmental_conditions: new Map([
                ['temperature', 22],
                ['humidity', 55],
                ['uv_index', 6],
            ]),
            user_preferences: new Map([
                ['luxury_feel', 0.9],
                ['fast_absorption', 0.8],
                ['visible_results', 0.95],
            ]),
            regulatory_regions: ['EU', 'FDA', 'Health_Canada'],
            budget_constraints: {min: 5.0, max: 35.0},
            time_constraints: 365, // 1 year development
            market_positioning: 'luxury_premium',
        };

        const config: MultiscaleOptimizationConfig = {
            max_iterations: 100,
            convergence_threshold: 0.0005,
            exploration_probability: 0.15,
            local_search_intensity: 0.8,
            global_search_scope: 0.25,
            constraint_penalty_weight: 2.5,
            synergy_reward_weight: 2.0,
            stability_weight: 1.5,
            cost_weight: 0.6, // Less important for luxury positioning
            efficacy_weight: 2.5,
        };

        // Step 2: Execute optimization
        const result = await optimizer.optimizeFormulation(targetOutcomes, context, config);

        // Step 3: Comprehensive validation
        expect(result.optimization_score).toBeGreaterThan(0.7); // High-quality formulation
        expect(result.optimized_formulation.ingredients.length).toBeGreaterThan(5); // Complex formulation
        expect(result.optimized_formulation.ingredients.length).toBeLessThanOrEqual(12); // Not overcomplicated

        // Verify multi-objective optimization success
        expect(result.convergence_metrics.constraint_violations).toBeLessThanOrEqual(1);
        expect(result.estimated_cost).toBeLessThanOrEqual(context.budget_constraints.max);

        // Verify regulatory compliance across all regions
        context.regulatory_regions.forEach(region => {
            const compliance = result.regulatory_compliance.get(region) || 0;
            expect(compliance).toBeGreaterThan(0.9);
        });

        // Verify therapeutic coverage
        targetOutcomes.forEach(outcome => {
            const efficacy = result.therapeutic_efficacy.get(outcome) || 0;
            expect(efficacy).toBeGreaterThan(0.4);
        });

        // Verify formulation quality indicators
        expect(result.predicted_stability).toBeGreaterThan(0.8);
        expect(result.optimization_trace.length).toBeGreaterThan(10); // Substantial optimization effort

        console.log(`✅ Integrated optimization completed successfully:
        - Final Score: ${result.optimization_score.toFixed(3)}
        - Ingredients: ${result.optimized_formulation.ingredients.length}
        - Iterations: ${result.convergence_metrics.iterations_to_convergence}
        - Cost: $${result.estimated_cost.toFixed(2)}/100g
        - Stability: ${(result.predicted_stability * 100).toFixed(1)}%`);
    });

    it('should handle edge case with minimal ingredient requirements', async () => {
        const targetOutcomes = ['hydration'];
        const minimalContext: OptimizationContext = {
            target_skin_type: 'sensitive',
            environmental_conditions: new Map(),
            user_preferences: new Map([['gentle_formula', 1.0]]),
            regulatory_regions: ['EU'],
            budget_constraints: {min: 0.5, max: 8.0},
            time_constraints: 90,
            market_positioning: 'drugstore',
        };

        const quickConfig: MultiscaleOptimizationConfig = {
            max_iterations: 30,
            convergence_threshold: 0.01,
            exploration_probability: 0.1,
            local_search_intensity: 0.9,
            global_search_scope: 0.1,
            constraint_penalty_weight: 3.0,
            synergy_reward_weight: 1.0,
            stability_weight: 2.0,
            cost_weight: 2.0, // High importance for drugstore positioning
            efficacy_weight: 1.5,
        };

        const result = await optimizer.optimizeFormulation(targetOutcomes, minimalContext, quickConfig);

        expect(result.optimized_formulation.ingredients.length).toBeGreaterThanOrEqual(3);
        expect(result.optimized_formulation.ingredients.length).toBeLessThanOrEqual(8);
        expect(result.estimated_cost).toBeLessThanOrEqual(minimalContext.budget_constraints.max);

        // Should prioritize gentle, well-tolerated ingredients
        const gentleIngredients = result.optimized_formulation.ingredients.filter(
            ing => ing.allergenicity === 'very_low' || ing.allergenicity === 'low',
        );
        expect(gentleIngredients.length / result.optimized_formulation.ingredients.length).toBeGreaterThan(0.7);
    });
});

// Mock data and utilities for testing
export const createMockIngredient = (overrides: Partial<CosmeticIngredient> = {}): CosmeticIngredient => ({
    id: 'mock_ingredient',
    name: 'Mock Ingredient',
    inci_name: 'Mock INCI',
    category: 'ACTIVE_INGREDIENT',
    subtype: 'SYNTHETIC_ACTIVE',
    functions: ['moisturizing'],
    molecularWeight: 500,
    solubility: 'water_soluble',
    ph_stability_range: {min: 5.0, max: 7.0},
    concentration_range: {min: 0.1, max: 5.0},
    allergenicity: 'low',
    pregnancy_safe: true,
    therapeutic_vectors: ['hydration'],
    skin_penetration_depth: 'epidermis',
    onset_time_hours: 24,
    duration_hours: 48,
    stability_factors: ['ph_stable'],
    regulatory_status: new Map([
        ['EU', 'approved'],
        ['FDA', 'approved'],
    ]),
    evidence_level: 'clinical',
    cost_per_gram: 1.0,
    ...overrides,
});

export const createMockFormulation = (overrides: Partial<CosmeticFormulation> = {}): CosmeticFormulation => ({
    id: 'mock_formulation',
    name: 'Mock Formulation',
    type: 'SKINCARE_FORMULATION',
    ingredients: [],
    concentrations: new Map(),
    total_cost: 0,
    ph_target: 6.0,
    stability_data: {
        formulation_id: 'mock_formulation',
        stability_factors: [{factor: 'ph_compatibility', risk_level: 'low'}],
        shelf_life_estimate: 24,
        storage_conditions: [{light_protection: true}],
        stability_rating: 'good',
    },
    regulatory_approvals: new Map(),
    target_properties: [],
    creation_date: new Date(),
    last_modified: new Date(),
    ...overrides,
});

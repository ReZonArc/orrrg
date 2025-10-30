/**
 * Hypergredient Framework Demonstration
 * 
 * This demonstration showcases the revolutionary Hypergredient Framework 
 * that transforms cosmetic formulation from art to science.
 * 
 * Key Features Demonstrated:
 * 1. Multi-objective optimization with constraint satisfaction
 * 2. Real-time compatibility analysis
 * 3. Performance prediction with confidence intervals  
 * 4. Network synergy calculations
 * 5. Cost-effectiveness optimization
 */

import { HypergredientFramework } from '../../lib/cheminformatics/hypergredient-framework.js';

/**
 * Main demonstration function
 */
export function runHypergredientDemo(): void {
    console.log('\n🧬 HYPERGREDIENT FRAMEWORK DEMONSTRATION');
    console.log('=' .repeat(50));

    // Initialize the framework
    const framework = new HypergredientFramework({
        optimization_weights: {
            efficacy: 0.35,
            safety: 0.25,
            stability: 0.20,
            cost: 0.15,
            synergy: 0.05
        }
    });

    console.log('\n📊 DATABASE STATISTICS');
    console.log('-'.repeat(30));
    const stats = framework.getDatabaseStats();
    console.log(`Total Ingredients: ${stats.total_ingredients}`);
    console.log(`Hypergredient Classes: ${stats.ingredients_by_class.size}`);
    
    for (const [hClass, count] of stats.ingredients_by_class) {
        const avgEfficacy = stats.avg_efficacy_by_class.get(hClass) || 0;
        console.log(`  ${hClass}: ${count} ingredients (avg efficacy: ${avgEfficacy.toFixed(1)}/10)`);
    }

    // Demonstration 1: Premium Anti-Aging Serum
    console.log('\n🚀 DEMONSTRATION 1: PREMIUM ANTI-AGING SERUM');
    console.log('-'.repeat(50));
    
    const antiAgingResult = framework.optimizeFormulation(
        ['wrinkles', 'fine_lines', 'firmness', 'brightness'],
        {
            budget_limit: 2500,
            total_actives_range: { min: 10, max: 20 },
            regulatory_regions: ['EU', 'FDA']
        },
        'mature'
    );

    displayFormulationResults('Premium Anti-Aging Serum', antiAgingResult);

    // Demonstration 2: Sensitive Skin Hydration
    console.log('\n🌿 DEMONSTRATION 2: SENSITIVE SKIN HYDRATION');
    console.log('-'.repeat(50));
    
    const sensitiveResult = framework.optimizeFormulation(
        ['hydration', 'barrier_damage', 'irritation'],
        {
            budget_limit: 1200,
            total_actives_range: { min: 5, max: 15 },
            skin_type_restrictions: ['sensitive']
        },
        'sensitive'
    );

    displayFormulationResults('Sensitive Skin Hydration', sensitiveResult);

    // Demonstration 3: Budget-Friendly Daily Moisturizer
    console.log('\n💰 DEMONSTRATION 3: BUDGET-FRIENDLY DAILY MOISTURIZER');
    console.log('-'.repeat(50));
    
    const budgetResult = framework.optimizeFormulation(
        ['hydration', 'barrier_damage'],
        {
            budget_limit: 800,
            total_actives_range: { min: 3, max: 12 }
        },
        'normal'
    );

    displayFormulationResults('Budget-Friendly Daily Moisturizer', budgetResult);

    // Demonstration 4: Multi-Concern Complex Formulation
    console.log('\n🎯 DEMONSTRATION 4: MULTI-CONCERN COMPLEX FORMULATION');
    console.log('-'.repeat(50));
    
    const complexResult = framework.optimizeFormulation(
        ['wrinkles', 'hydration', 'brightness', 'oily_skin', 'acne'],
        {
            budget_limit: 3000,
            total_actives_range: { min: 15, max: 25 },
            regulatory_regions: ['EU', 'FDA', 'JP']
        },
        'combination'
    );

    displayFormulationResults('Multi-Concern Complex Formulation', complexResult);

    console.log('\n✨ HYPERGREDIENT FRAMEWORK SUMMARY');
    console.log('=' .repeat(50));
    console.log('🔬 Science-driven formulation design');
    console.log('⚖️  Multi-objective optimization');
    console.log('🔗 Network synergy effects');
    console.log('📈 Performance prediction');
    console.log('💡 Cost-effectiveness analysis');
    console.log('🛡️  Real-time compatibility checking');
    console.log('\nTransforming cosmetic formulation from art to science! 🚀');
}

/**
 * Display comprehensive formulation results
 */
function displayFormulationResults(
    title: string, 
    result: {
        formulation: any;
        analysis: any;
        prediction: any;
        score: any;
    }
): void {
    console.log(`\n📋 ${title.toUpperCase()}`);
    console.log(`Score: ${result.score.composite_score.toFixed(2)}/10`);
    console.log(`Total Cost: R${result.formulation.total_cost.toFixed(2)}/100g`);
    console.log(`Compatibility: ${result.analysis.overall_compatibility}`);
    console.log(`Stability: ${result.analysis.stability_prediction.overall_stability}/100`);
    
    console.log('\nIngredients:');
    result.formulation.ingredients.forEach((ingredient: any) => {
        const concentration = result.formulation.concentrations.get(ingredient.id) || 0;
        const hClass = ingredient.hypergredient_class || 'Unknown';
        console.log(`  • ${ingredient.name} (${hClass}): ${concentration.toFixed(1)}%`);
    });

    console.log('\nPerformance Breakdown:');
    console.log(`  Efficacy: ${(result.score.individual_scores.efficacy * 10).toFixed(1)}/10`);
    console.log(`  Safety: ${(result.score.individual_scores.safety * 10).toFixed(1)}/10`);
    console.log(`  Stability: ${(result.score.individual_scores.stability * 10).toFixed(1)}/10`);
    console.log(`  Cost Efficiency: ${(result.score.individual_scores.cost_efficiency * 10).toFixed(1)}/10`);
    
    if (result.score.network_bonus > 0) {
        console.log(`  Network Synergy Bonus: +${result.score.network_bonus.toFixed(2)}`);
    }

    console.log('\nPredicted Performance:');
    for (const [concern, efficacy] of result.prediction.predicted_efficacy) {
        const timeline = result.prediction.predicted_timeline.get(concern) || 0;
        const confidence = result.prediction.confidence_scores.get(concern) || 0;
        console.log(`  ${concern}: ${efficacy.toFixed(0)}% improvement in ${timeline} weeks (confidence: ${(confidence * 100).toFixed(0)}%)`);
    }

    if (result.analysis.interaction_warnings.length > 0) {
        console.log('\n⚠️  Compatibility Warnings:');
        result.analysis.interaction_warnings.forEach((warning: any) => {
            console.log(`  • ${warning.warning_message}`);
        });
    }
}

// Run the demo if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    runHypergredientDemo();
}
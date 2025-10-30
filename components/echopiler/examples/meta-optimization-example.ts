/**
 * Meta-Optimization Engine Usage Example
 * 
 * This example demonstrates how to use the new MetaOptimizationEngine to
 * generate optimal formulations for every possible condition and treatment
 * combination automatically.
 */

import { MetaOptimizationEngine } from '../lib/cheminformatics/meta-optimization-engine.js';

async function demonstrateMetaOptimization() {
    console.log('🚀 Meta-Optimization Engine Demo\n');

    // Initialize the meta-optimization engine
    const metaEngine = new MetaOptimizationEngine({
        max_combinations: 50,  // Limit for demo
        enable_caching: true,
        performance_tracking: true
    });

    console.log('📊 Getting condition-treatment matrix...');
    const matrix = metaEngine.getConditionTreatmentMatrix();
    console.log(`   Found ${matrix.combinations.length} meaningful combinations`);
    console.log(`   Conditions: ${matrix.conditions.slice(0, 5).join(', ')}...`);
    console.log(`   Treatments: ${matrix.treatments.slice(0, 5).join(', ')}...`);

    // Example 1: Optimize for a specific combination
    console.log('\n🎯 Example 1: Single Combination Optimization');
    const antiAgingCombo = matrix.combinations.find(c => 
        c.conditions.includes('wrinkles') && c.treatments.includes('anti_aging')
    );
    
    if (antiAgingCombo) {
        console.log(`   Optimizing: ${antiAgingCombo.conditions.join(', ')} with ${antiAgingCombo.treatments.join(', ')}`);
        console.log(`   Recommended strategy: ${antiAgingCombo.recommended_strategy}`);
        console.log(`   Complexity score: ${antiAgingCombo.complexity_score}/10`);
        
        const result = await metaEngine.optimizeForCombination(
            antiAgingCombo.id,
            'mature',
            { budget_limit: 150 }
        );
        
        console.log(`   ✅ Optimization complete!`);
        console.log(`   📈 Score: ${result.performance_metrics.optimization_score.toFixed(2)}/10`);
        console.log(`   🧪 Ingredients: ${result.optimal_formulation.ingredients.length}`);
        console.log(`   ⏱️  Time: ${result.performance_metrics.execution_time_ms}ms`);
        console.log(`   💡 Recommendations: ${result.recommendations.slice(0, 2).join(', ')}`);
    }

    // Example 2: Comprehensive optimization (limited subset)
    console.log('\n🌟 Example 2: Comprehensive Meta-Optimization (5 combinations)');
    const limitedEngine = new MetaOptimizationEngine({
        max_combinations: 5,
        enable_caching: true,
        performance_tracking: true
    });

    const summary = await limitedEngine.optimizeAllCombinations('normal', {
        budget_limit: 100
    });

    console.log(`   ✅ Optimized ${summary.successful_optimizations}/${summary.total_combinations} combinations`);
    console.log(`   📈 Best score: ${summary.performance_analytics.best_overall_score.toFixed(2)}`);
    console.log(`   📊 Average score: ${summary.performance_analytics.average_score.toFixed(2)}`);
    
    console.log('\n   🎯 Strategy Distribution:');
    summary.strategy_distribution.forEach((count, strategy) => {
        console.log(`     ${strategy}: ${count} combinations`);
    });

    console.log('\n   🏆 Top 3 Performing Combinations:');
    summary.top_performing_combinations.slice(0, 3).forEach((combo, i) => {
        console.log(`     ${i + 1}. ${combo.combination_id} - Score: ${combo.performance_metrics.optimization_score.toFixed(2)}`);
    });

    // Example 3: Performance Analytics
    console.log('\n📊 Example 3: Performance Analytics');
    const analytics = metaEngine.getPerformanceAnalytics();
    
    analytics.forEach((scores, strategy) => {
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        console.log(`   ${strategy}: avg ${avgScore.toFixed(2)} (${scores.length} runs)`);
    });

    console.log('\n🎉 Meta-optimization demo complete!');
    console.log('\nKey Benefits:');
    console.log('• Automatically generates formulations for ALL condition-treatment combinations');
    console.log('• Intelligently selects optimal optimization strategy based on complexity');
    console.log('• Provides performance analytics and caching for efficiency');
    console.log('• Offers contextual recommendations for each formulation');
    console.log('• Seamlessly integrates existing HypergredientFramework and MultiscaleOptimizer');
}

// Run the demo if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    demonstrateMetaOptimization().catch(console.error);
}

export { demonstrateMetaOptimization };
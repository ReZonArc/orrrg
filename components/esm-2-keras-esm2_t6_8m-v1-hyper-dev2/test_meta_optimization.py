#!/usr/bin/env python3
"""
Test suite for Meta-Optimization Strategy
Validates the comprehensive condition/treatment formulation generation system.
"""

import json
from hypergredient_framework import (
    HypergredientDatabase, MetaOptimizationStrategy, 
    FormulationRequest, FormulationConstraints
)

def test_meta_optimization_basic():
    """Test basic meta-optimization functionality"""
    print("🧪 Testing Meta-Optimization Strategy - Basic Functionality")
    print("=" * 60)
    
    # Initialize system
    database = HypergredientDatabase()
    meta_optimizer = MetaOptimizationStrategy(database)
    
    # Test 1: Verify initialization
    print("Test 1: Initialization")
    assert len(meta_optimizer.skin_conditions) > 15, "Should have comprehensive skin conditions"
    assert len(meta_optimizer.skin_types) == 6, "Should have 6 skin types"
    assert len(meta_optimizer.severity_levels) == 3, "Should have 3 severity levels"
    assert len(meta_optimizer.treatment_goals) == 6, "Should have 6 treatment goals"
    print("✓ Initialization test passed")
    
    # Test 2: Adaptive weights
    print("\nTest 2: Adaptive Weights")
    weights_sensitive = meta_optimizer._get_adaptive_weights('sensitive', 'mild', 'prevention')
    weights_oily = meta_optimizer._get_adaptive_weights('oily', 'severe', 'intensive_treatment')
    
    assert weights_sensitive['safety'] > weights_oily['safety'], "Sensitive skin should prioritize safety"
    assert weights_oily['efficacy'] > weights_sensitive['efficacy'], "Severe conditions should prioritize efficacy"
    assert abs(sum(weights_sensitive.values()) - 1.0) < 0.01, "Weights should sum to 1.0"
    print("✓ Adaptive weights test passed")
    
    # Test 3: Dynamic request creation
    print("\nTest 3: Dynamic Request Creation")
    request = meta_optimizer._create_dynamic_request('acne', 'oily', 'severe', 'treatment')
    
    assert 'acne' in request.target_concerns, "Should include primary condition"
    assert request.skin_type == 'oily', "Should match skin type"
    assert request.budget > 1000, "Severe conditions should have higher budget"
    print("✓ Dynamic request creation test passed")
    
    return True

def test_meta_optimization_formulation_generation():
    """Test formulation generation for specific combinations"""
    print("\n🧪 Testing Meta-Optimization Strategy - Formulation Generation")
    print("=" * 60)
    
    database = HypergredientDatabase()
    meta_optimizer = MetaOptimizationStrategy(database)
    
    # Test specific combinations
    test_combinations = [
        ('acne', 'oily', 'moderate', 'treatment'),
        ('wrinkles', 'mature', 'severe', 'intensive_treatment'),
        ('sensitivity', 'sensitive', 'mild', 'prevention'),
        ('dryness', 'dry', 'moderate', 'maintenance')
    ]
    
    for i, (condition, skin_type, severity, goal) in enumerate(test_combinations, 1):
        print(f"\nTest {i}: {condition} + {skin_type} + {severity} + {goal}")
        
        # Generate formulation for this combination
        result = meta_optimizer._optimize_for_combination(condition, skin_type, severity, goal)
        
        assert result is not None, f"Should generate formulation for {condition}/{skin_type}"
        assert result.predicted_efficacy > 0, "Should have positive efficacy"
        assert result.safety_score > 0, "Should have positive safety score"
        assert result.total_cost > 0, "Should have positive cost"
        assert len(result.selected_hypergredients) > 0, "Should select ingredients"
        
        # Verify optimization score
        score = meta_optimizer._calculate_combination_score(result, condition, skin_type, severity, goal)
        assert 0 <= score <= 1, "Optimization score should be between 0 and 1"
        
        # Verify meta insights
        insights = meta_optimizer._generate_meta_insights(result, condition, skin_type, severity, goal)
        assert 'optimization_rationale' in insights, "Should have optimization rationale"
        
        print(f"  ✓ Generated formulation: {result.predicted_efficacy:.2%} efficacy, "
              f"{result.safety_score:.1f}/10 safety, R{result.total_cost:.2f} cost")
    
    return True

def test_meta_optimization_comprehensive_matrix():
    """Test comprehensive matrix generation"""
    print("\n🧪 Testing Meta-Optimization Strategy - Comprehensive Matrix")
    print("=" * 60)
    
    database = HypergredientDatabase()
    meta_optimizer = MetaOptimizationStrategy(database)
    
    # Generate limited matrix for testing
    print("Generating matrix with 20 combinations...")
    matrix_result = meta_optimizer.generate_comprehensive_formulation_matrix(max_combinations=20)
    
    # Validate matrix structure
    assert 'formulation_matrix' in matrix_result, "Should have formulation matrix"
    assert 'meta_analysis' in matrix_result, "Should have meta analysis"
    assert 'optimization_statistics' in matrix_result, "Should have optimization statistics"
    
    formulation_matrix = matrix_result['formulation_matrix']
    assert len(formulation_matrix) > 0, "Should generate at least some formulations"
    assert len(formulation_matrix) <= 20, "Should respect max combinations limit"
    
    # Validate individual formulations
    for combo_key, data in formulation_matrix.items():
        assert 'condition' in data, "Should have condition"
        assert 'skin_type' in data, "Should have skin type"
        assert 'severity' in data, "Should have severity"
        assert 'treatment_goal' in data, "Should have treatment goal"
        assert 'formulation' in data, "Should have formulation"
        assert 'optimization_score' in data, "Should have optimization score"
        assert 'meta_insights' in data, "Should have meta insights"
        
        # Validate formulation
        formulation = data['formulation']
        assert formulation.predicted_efficacy >= 0, "Efficacy should be non-negative"
        assert formulation.safety_score > 0, "Safety should be positive"
        assert formulation.total_cost > 0, "Cost should be positive"
    
    print(f"✓ Generated {len(formulation_matrix)} formulations successfully")
    
    # Test pattern analysis
    meta_analysis = matrix_result['meta_analysis']
    if meta_analysis.get('efficacy_patterns'):
        print("✓ Pattern analysis generated successfully")
    
    return True

def test_meta_optimization_profile_specific():
    """Test profile-specific optimization"""
    print("\n🧪 Testing Meta-Optimization Strategy - Profile-Specific Optimization")
    print("=" * 60)
    
    database = HypergredientDatabase()
    meta_optimizer = MetaOptimizationStrategy(database)
    
    # Test specific profiles
    test_profiles = [
        ('acne', 'oily', 'moderate', 'treatment'),
        ('wrinkles', 'mature', 'severe', 'intensive_treatment'),
        ('sensitivity', 'sensitive', 'mild', 'prevention')
    ]
    
    for i, (condition, skin_type, severity, goal) in enumerate(test_profiles, 1):
        print(f"\nTest {i}: Profile-specific optimization")
        
        # Get optimal formulation for profile
        result = meta_optimizer.get_optimal_formulation_for_profile(
            condition, skin_type, severity, goal
        )
        
        assert result is not None, "Should return formulation for valid profile"
        assert result['condition'] == condition, "Should match requested condition"
        assert result['skin_type'] == skin_type, "Should match requested skin type"
        assert result['severity'] == severity, "Should match requested severity"
        assert result['treatment_goal'] == goal, "Should match requested goal"
        
        formulation = result['formulation']
        assert formulation.predicted_efficacy > 0, "Should have positive efficacy"
        assert formulation.safety_score > 0, "Should have positive safety"
        
        print(f"  ✓ Profile {condition}/{skin_type}: {formulation.predicted_efficacy:.2%} efficacy")
        
        # Test caching - second call should be faster and return cached result
        cached_result = meta_optimizer.get_optimal_formulation_for_profile(
            condition, skin_type, severity, goal
        )
        assert cached_result == result, "Should return cached result"
    
    return True

def test_meta_optimization_report_generation():
    """Test report generation functionality"""
    print("\n🧪 Testing Meta-Optimization Strategy - Report Generation")
    print("=" * 60)
    
    database = HypergredientDatabase()
    meta_optimizer = MetaOptimizationStrategy(database)
    
    # Generate some formulations first
    meta_optimizer.generate_comprehensive_formulation_matrix(max_combinations=10)
    
    # Generate report
    report = meta_optimizer.generate_meta_optimization_report()
    
    assert isinstance(report, str), "Report should be a string"
    assert len(report) > 100, "Report should be substantial"
    assert "Meta-Optimization Strategy Report" in report, "Should have proper title"
    assert "Total Cached Formulations" in report, "Should include cache statistics"
    assert "Performance Averages" in report, "Should include performance metrics"
    
    print("✓ Report generation test passed")
    print(f"  Report length: {len(report)} characters")
    
    return True

def run_all_tests():
    """Run all meta-optimization tests"""
    print("🧬 Meta-Optimization Strategy Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_meta_optimization_basic),
        ("Formulation Generation", test_meta_optimization_formulation_generation),
        ("Comprehensive Matrix", test_meta_optimization_comprehensive_matrix),
        ("Profile-Specific Optimization", test_meta_optimization_profile_specific),
        ("Report Generation", test_meta_optimization_report_generation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                print(f"✅ {test_name} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} - FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} - FAILED with error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"Success rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 All meta-optimization tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

def main():
    """Main test execution"""
    success = run_all_tests()
    
    # Save test results
    test_results = {
        "test_suite": "Meta-Optimization Strategy",
        "timestamp": "2024-01-01",  # Simplified timestamp
        "success": success,
        "description": "Comprehensive test of meta-optimization strategy for condition/treatment formulations"
    }
    
    with open("meta_optimization_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Test results saved to meta_optimization_test_results.json")

if __name__ == "__main__":
    main()
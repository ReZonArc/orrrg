#!/usr/bin/env python3
"""
Test suite for persona-based training functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hypergredient_framework import (
    PersonaTrainingSystem, PersonaProfile, HypergredientAI, HypergredientDatabase,
    FormulationRequest, FormulationResult
)


def test_persona_system_initialization():
    """Test persona system initialization"""
    print("Testing persona system initialization...")
    
    persona_system = PersonaTrainingSystem()
    
    # Check default personas are loaded
    assert len(persona_system.personas) >= 4, "Should have at least 4 default personas"
    
    # Check specific personas exist
    expected_personas = ['sensitive_skin', 'anti_aging', 'acne_prone', 'natural_beauty']
    for persona_id in expected_personas:
        assert persona_id in persona_system.personas, f"Missing persona: {persona_id}"
    
    print("✓ Persona system initialization test passed")


def test_persona_profile_creation():
    """Test creating custom persona profiles"""
    print("Testing custom persona profile creation...")
    
    custom_persona = PersonaProfile(
        persona_id="test_persona",
        name="Test User",
        description="Test persona for validation",
        skin_type="normal",
        primary_concerns=["test_concern"],
        sensitivity_level=0.5,
        budget_preference="mid-range"
    )
    
    # Test persona weights
    weights = custom_persona.get_persona_weights()
    assert 'efficacy' in weights, "Should include efficacy weight"
    assert 'safety' in weights, "Should include safety weight"
    assert weights['safety'] == custom_persona.safety_priority, "Safety weight should match persona"
    
    print("✓ Custom persona profile creation test passed")


def test_persona_aware_predictions():
    """Test that personas affect AI predictions"""
    print("Testing persona-aware predictions...")
    
    persona_system = PersonaTrainingSystem()
    ai_system = HypergredientAI(persona_system)
    
    # Create test request
    test_request = FormulationRequest(
        target_concerns=['sensitivity', 'wrinkles'],
        skin_type='sensitive',
        budget=500.0
    )
    
    # Test without persona
    baseline_prediction = ai_system.predict_optimal_combination(test_request)
    
    # Test with sensitive skin persona
    ai_system.persona_system.set_active_persona('sensitive_skin')
    sensitive_prediction = ai_system.predict_optimal_combination(test_request)
    
    # Test with anti-aging persona  
    ai_system.persona_system.set_active_persona('anti_aging')
    anti_aging_prediction = ai_system.predict_optimal_combination(test_request)
    
    # Verify predictions include persona information
    assert sensitive_prediction['active_persona'] == 'sensitive_skin', "Should track active persona"
    assert sensitive_prediction['persona_adjustments'] is not None, "Should include persona adjustments"
    
    # Verify different personas give different results
    sensitive_top = sensitive_prediction['predictions'][0]
    anti_aging_top = anti_aging_prediction['predictions'][0]
    
    # The predictions should be different due to persona adjustments
    assert (sensitive_top['ingredient_class'] != anti_aging_top['ingredient_class'] or 
            abs(sensitive_top['confidence'] - anti_aging_top['confidence']) > 0.01), \
           "Different personas should produce different predictions"
    
    print("✓ Persona-aware predictions test passed")


def test_persona_training():
    """Test persona training functionality"""
    print("Testing persona training...")
    
    persona_system = PersonaTrainingSystem()
    ai_system = HypergredientAI(persona_system)
    database = HypergredientDatabase()
    
    # Create training data
    training_requests = [
        FormulationRequest(['sensitivity'], skin_type='sensitive', budget=400),
        FormulationRequest(['redness'], skin_type='sensitive', budget=500)
    ]
    
    training_results = [
        FormulationResult(
            selected_hypergredients={'H.AI': {'ingredient': database.hypergredients['niacinamide'], 'percentage': 5.0, 'cost': 25.0, 'reasoning': 'Test'}},
            total_cost=300.0,
            predicted_efficacy=0.8,
            safety_score=9.0,
            stability_months=24,
            synergy_score=0.7,
            reasoning={'H.AI': 'Test reasoning'}
        ),
        FormulationResult(
            selected_hypergredients={'H.BR': {'ingredient': database.hypergredients['ceramide_np'], 'percentage': 3.0, 'cost': 30.0, 'reasoning': 'Test'}},
            total_cost=350.0,
            predicted_efficacy=0.7,
            safety_score=9.5,
            stability_months=18,
            synergy_score=0.8,
            reasoning={'H.BR': 'Test reasoning'}
        )
    ]
    
    training_feedback = [
        {'efficacy': 8.0, 'safety': 9.5, 'user_satisfaction': 8.5},
        {'efficacy': 7.5, 'safety': 9.8, 'user_satisfaction': 9.0}
    ]
    
    # Get initial training summary
    initial_summary = persona_system.get_training_summary()
    initial_samples = initial_summary['personas']['sensitive_skin']['training_samples']
    
    # Train the persona
    ai_system.train_with_persona('sensitive_skin', training_requests, training_results, training_feedback)
    
    # Verify training data was added
    updated_summary = persona_system.get_training_summary()
    updated_samples = updated_summary['personas']['sensitive_skin']['training_samples']
    
    assert updated_samples == initial_samples + len(training_requests), \
           f"Training samples should increase from {initial_samples} to {initial_samples + len(training_requests)}"
    
    print("✓ Persona training test passed")


def test_persona_switching():
    """Test switching between personas"""
    print("Testing persona switching...")
    
    persona_system = PersonaTrainingSystem()
    
    # Test setting active persona
    persona_system.set_active_persona('sensitive_skin')
    assert persona_system.active_persona == 'sensitive_skin', "Should set active persona"
    
    # Test switching personas
    persona_system.set_active_persona('anti_aging')
    assert persona_system.active_persona == 'anti_aging', "Should switch active persona"
    
    # Test invalid persona
    try:
        persona_system.set_active_persona('invalid_persona')
        assert False, "Should raise error for invalid persona"
    except ValueError:
        pass  # Expected
    
    print("✓ Persona switching test passed")


def run_all_tests():
    """Run all persona training tests"""
    print("🎭 Running Persona Training System Tests")
    print("=" * 50)
    
    try:
        test_persona_system_initialization()
        test_persona_profile_creation()
        test_persona_aware_predictions()
        test_persona_training()
        test_persona_switching()
        
        print("\n✅ All persona training tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
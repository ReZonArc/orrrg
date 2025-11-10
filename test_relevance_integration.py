#!/usr/bin/env python3
"""
Test script for Relevance Realization Ennead Integration
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the current directory to sys.path to import core modules
sys.path.append(str(Path(__file__).parent))

from core import (
    SelfOrganizingCore,
    RelevanceRealizationIntegrator,
    KnowingMode,
    UnderstandingOrder,
    WisdomPractice
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_relevance_realization_integration():
    """Test the full relevance realization integration with SOC."""
    print("=" * 70)
    print("🎯 Testing Relevance Realization Ennead Integration")
    print("=" * 70)
    
    # Initialize Self-Organizing Core
    print("\n1. Initializing Self-Organizing Core...")
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    print(f"   ✓ SOC initialized with {len(soc.components)} components")
    
    # Check relevance realization is integrated
    print("\n2. Checking Relevance Realization Integrator...")
    assert hasattr(soc, 'relevance_realization'), "SOC should have relevance_realization"
    assert soc.relevance_realization.running, "Relevance realization should be running"
    
    print(f"   ✓ Relevance Realization Integrator active")
    print(f"   ✓ {len(soc.relevance_realization.relevance_frames)} relevance frames initialized")
    
    # Test Ennead status
    print("\n3. Testing Ennead Status...")
    status = soc.relevance_realization.get_ennead_status()
    
    print(f"   Integration Score: {status['ennead_integration_score']:.3f}")
    print(f"   Triad Coherence:")
    for triad_name, coherence in status['triad_coherence'].items():
        print(f"     • {triad_name}: {coherence:.3f}")
    
    assert 'ennead_integration_score' in status
    assert 'triad_coherence' in status
    assert len(status['triad_coherence']) == 3  # Three triads
    
    print(f"   ✓ Ennead status retrieved successfully")
    
    # Test relevance optimization
    print("\n4. Testing Relevance Optimization...")
    context = {
        'task_type': 'multi_domain_integration',
        'domain': 'research',
        'requirements': ['analysis', 'reasoning', 'processing']
    }
    
    result = await soc.relevance_realization.optimize_relevance_realization(context)
    
    print(f"   Relevance Score: {result['relevance_score']:.3f}")
    print(f"   Integration: {result['ennead_integration']:.3f}")
    print(f"   Salient Components: {len(result['salient_components'])}")
    
    assert 'relevance_score' in result
    assert 'ennead_integration' in result
    assert 'salient_components' in result
    
    print(f"   ✓ Relevance optimization completed")
    
    # Test knowing modes integration
    print("\n5. Testing Ways of Knowing Integration...")
    integrated_knowledge = result['integrated_knowledge']
    
    for mode in ['propositional', 'procedural', 'perspectival', 'participatory']:
        assert mode in integrated_knowledge
        print(f"   • {mode.capitalize()}: {len(integrated_knowledge[mode])} components")
    
    print(f"   ✓ All four knowing modes integrated")
    
    # Test understanding orders
    print("\n6. Testing Orders of Understanding...")
    integrated_understanding = result['integrated_understanding']
    
    for order in ['nomological', 'normative', 'narrative']:
        assert order in integrated_understanding
        print(f"   • {order.capitalize()}: {integrated_understanding[order]}")
    
    print(f"   ✓ All three understanding orders active")
    
    # Test wisdom practices
    print("\n7. Testing Wisdom Practices...")
    wisdom_alignment = result['wisdom_alignment']
    
    for practice in ['morality', 'meaning', 'mastery']:
        assert practice in wisdom_alignment
        print(f"   • {practice.capitalize()}: {wisdom_alignment[practice]:.3f}")
    
    print(f"   ✓ All three wisdom practices aligned")
    
    # Test insight generation
    print("\n8. Testing Insight Generation...")
    insight = await soc.relevance_realization.generate_ennead_insight()
    
    print(f"   Insight: {insight}")
    assert insight is not None
    assert len(insight) > 0
    
    print(f"   ✓ Insight generated successfully")
    
    # Test perspective shift (if multiple components available)
    if len(soc.components) >= 2:
        print("\n9. Testing Perspective Shift...")
        component_names = list(soc.components.keys())[:2]
        
        shift_result = await soc.relevance_realization.realize_perspective_shift(
            component_names[0],
            component_names[1]
        )
        
        if shift_result['success']:
            print(f"   ✓ Perspective shift: {component_names[0]} → {component_names[1]}")
            print(f"   • Mode shifts: {len(shift_result['mode_shifts'])}")
            print(f"   • Gnostic: {shift_result['is_gnostic_transformation']}")
        else:
            print(f"   ⚠ Shift not applicable: {shift_result['reason']}")
    
    # Test integration patterns
    print("\n10. Testing Integration Patterns...")
    print(f"   Discovered patterns: {len(soc.relevance_realization.integration_patterns)}")
    
    for i, pattern in enumerate(soc.relevance_realization.integration_patterns[:3], 1):
        print(f"   {i}. {pattern['type']}: {pattern['components']}")
        print(f"      Score: {pattern['score']:.3f}")
    
    print(f"   ✓ Integration patterns identified")
    
    # Check SOC system status includes relevance realization
    print("\n11. Testing SOC System Status Integration...")
    system_status = soc.get_system_status()
    
    assert 'relevance_realization' in system_status
    rr_status = system_status['relevance_realization']
    
    print(f"   ✓ Relevance realization in system status")
    print(f"   • Optimizations: {rr_status['relevance_optimizations']}")
    print(f"   • Perspective shifts: {rr_status['perspective_shifts']}")
    print(f"   • Gnostic transformations: {rr_status['gnostic_transformations']}")
    
    # Cleanup
    print("\n12. Cleaning up...")
    await soc.shutdown()
    print(f"   ✓ SOC shutdown complete")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - Relevance Realization Integration Successful!")
    print("=" * 70)
    
    # Summary
    print("\n📊 Integration Summary:")
    print(f"   Components integrated: {len(soc.components)}")
    print(f"   Relevance frames: {len(soc.relevance_realization.relevance_frames)}")
    print(f"   Integration patterns: {len(soc.relevance_realization.integration_patterns)}")
    print(f"   Ennead integration score: {status['ennead_integration_score']:.3f}")
    print(f"   Final relevance score: {result['relevance_score']:.3f}")
    
    print("\n🎯 The Relevance Realization Ennead successfully integrates:")
    print("   • TRIAD I: Ways of Knowing (Propositional, Procedural, Perspectival, Participatory)")
    print("   • TRIAD II: Orders of Understanding (Nomological, Normative, Narrative)")
    print("   • TRIAD III: Practices of Wisdom (Morality, Meaning, Mastery)")
    print("\n   This enables optimal relevance realization across all ORRRG components!")


async def test_standalone_relevance_integrator():
    """Test standalone relevance integrator functionality."""
    print("\n" + "=" * 70)
    print("🧪 Testing Standalone Relevance Realization Integrator")
    print("=" * 70)
    
    integrator = RelevanceRealizationIntegrator()
    
    # Test Ennead state
    print("\n1. Testing Ennead State...")
    ennead_state = integrator.ennead_state
    
    assert hasattr(ennead_state, 'propositional_knowledge')
    assert hasattr(ennead_state, 'procedural_knowledge')
    assert hasattr(ennead_state, 'perspectival_knowledge')
    assert hasattr(ennead_state, 'participatory_knowledge')
    
    assert hasattr(ennead_state, 'nomological_understanding')
    assert hasattr(ennead_state, 'normative_understanding')
    assert hasattr(ennead_state, 'narrative_understanding')
    
    assert hasattr(ennead_state, 'morality_cultivation')
    assert hasattr(ennead_state, 'meaning_realization')
    assert hasattr(ennead_state, 'mastery_development')
    
    print("   ✓ All nine Ennead dimensions present")
    
    # Test integration calculation
    print("\n2. Testing Integration Calculation...")
    integration = ennead_state.calculate_integration()
    
    print(f"   Integration score: {integration:.3f}")
    print(f"   Triad coherence: {ennead_state.triad_coherence}")
    
    assert integration >= 0.0
    assert integration <= 1.0
    
    print("   ✓ Integration calculation working")
    
    print("\n✅ Standalone tests passed!")


async def main():
    """Run all tests."""
    try:
        # Test standalone integrator first
        await test_standalone_relevance_integrator()
        
        # Test full integration with SOC
        await test_relevance_realization_integration()
        
        print("\n🎉 All relevance realization integration tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Test script for Eric Schwarz's Holistic Metamodel implementation
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the current directory to sys.path to import core modules
sys.path.append(str(Path(__file__).parent))

from core.holistic_metamodel import (
    HolisticMetamodelOrchestrator, HieroglyphicMonad, DualComplementarity,
    TriadicSystem, SelfStabilizingCycle, OrganizationalDynamicsProcessor,
    OrganizationalDynamic, CyclePhase, DualMode, TriadPrimitive
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_hieroglyphic_monad():
    """Test the hieroglyphic monad implementation."""
    print("=" * 60)
    print("Testing Hieroglyphic Monad (The 1 Unity Principle)")
    print("=" * 60)
    
    monad = HieroglyphicMonad(
        essence="test_organizational_unity",
        integration_degree=0.9
    )
    
    for level in range(4):
        manifestation = monad.manifest_at_level(level)
        print(f"Level {level} Manifestation:")
        print(f"  Unity Degree: {manifestation['unity_degree']:.3f}")
        print(f"  Coherence: {manifestation['coherence']:.3f}")
        print(f"  Pattern Length: {len(manifestation['manifestation_pattern'])}")
        print()


async def test_dual_complementarity():
    """Test the dual complementarity implementation."""
    print("=" * 60)
    print("Testing Dual Complementarity (The 2 Modes)")
    print("=" * 60)
    
    actual_state = {
        'activity_level': 0.7,
        'structural_integrity': 0.8,
        'energy': 0.6
    }
    
    virtual_state = {
        'potential_activity_level': 0.3,
        'potential_structural_integrity': 0.2,
        'potential_energy': 0.4,
        'latent_activity_level': 0.21,
        'latent_structural_integrity': 0.24,
        'latent_energy': 0.18
    }
    
    dual_comp = DualComplementarity(
        actual_state=actual_state,
        virtual_state=virtual_state,
        complementarity_degree=0.8,
        tension_level=0.6,
        resolution_potential=0.75
    )
    
    resolution = dual_comp.resolve_tension()
    
    print("Actual State:", actual_state)
    print("Virtual State:", virtual_state)
    print("Resolution Quality:", resolution['resolution_quality'])
    print("Emergent Properties:", resolution['emergent_properties'])
    print("Synthesized State:", resolution['synthesized_state'])
    print()


async def test_triadic_system():
    """Test the triadic system implementation."""
    print("=" * 60)
    print("Testing Triadic System (The 3 Primitives)")
    print("=" * 60)
    
    being_comp = {
        'structural_presence': 0.8,
        'foundation_strength': 0.7,
        'stability': 0.75
    }
    
    becoming_comp = {
        'transformation_rate': 0.6,
        'process_intensity': 0.5,
        'evolution_momentum': 0.55
    }
    
    relation_comp = {
        'connectivity': 0.7,
        'interaction_frequency': 0.65,
        'coherence': 0.8
    }
    
    triad = TriadicSystem(
        being_component=being_comp,
        becoming_component=becoming_comp,
        relation_component=relation_comp,
        triad_coherence=0.75,
        dynamic_balance=0.7
    )
    
    state = triad.compute_triad_state()
    
    print("Being Strength:", state['being_strength'])
    print("Becoming Intensity:", state['becoming_intensity'])
    print("Relation Density:", state['relation_density'])
    print("Dynamic Equilibrium:", state['dynamic_equilibrium'])
    print("Transformation Potential:", state['transformation_potential'])
    print()


async def test_self_stabilizing_cycle():
    """Test the self-stabilizing cycle implementation."""
    print("=" * 60)
    print("Testing Self-Stabilizing Cycle (The 4 Phases)")
    print("=" * 60)
    
    # Create a dummy dual complementarity for the cycle
    dual_comp = DualComplementarity(
        actual_state={'energy': 0.6},
        virtual_state={'potential_energy': 0.4},
        complementarity_degree=0.7,
        tension_level=0.5,
        resolution_potential=0.8
    )
    
    cycle = SelfStabilizingCycle(
        current_phase=CyclePhase.EMERGENCE,
        phase_progression={phase: 0.0 for phase in CyclePhase},
        cycle_energy=0.8,
        stabilization_degree=0.6,
        actual_virtual_balance=dual_comp
    )
    
    print(f"Initial Phase: {cycle.current_phase.value}")
    
    # Simulate cycle progression
    for i in range(6):
        # Simulate some progress in current phase
        cycle.phase_progression[cycle.current_phase] = min(1.0, (i + 1) * 0.25)
        
        next_phase, transition_data = cycle.advance_cycle()
        print(f"Step {i+1}: Phase = {next_phase.value}")
        
        if 'deepening' in transition_data:
            print(f"  Deepening current phase, progress: {transition_data['progress']:.2f}")
        else:
            print(f"  Transitioned from {transition_data['from_phase']} to {transition_data['to_phase']}")
            print(f"  Emergent qualities: {transition_data['emergent_qualities']}")
    
    print()


async def test_organizational_dynamics():
    """Test the organizational dynamics processor."""
    print("=" * 60)
    print("Testing Organizational Dynamics (The 3 Streams)")
    print("=" * 60)
    
    processor = OrganizationalDynamicsProcessor()
    
    system_state = {
        'component_coordination': 0.7,
        'pattern_coherence': 0.6,
        'system_integration': 0.8,
        'structural_integrity': 0.75,
        'functional_coherence': 0.7,
        'equilibrium_maintenance': 0.65,
        'self_recognition': 0.6,
        'boundary_definition': 0.7,
        'identity_coherence': 0.65,
        'system_energy': 0.8,
        'cognitive_complexity': 0.5
    }
    
    results = await processor.process_dynamics(system_state)
    
    print("Entropic Stream (en-tropis → auto-vortis → auto-morphosis):")
    entropic = results['entropic']
    print(f"  En-tropis Level: {entropic['entropis_level']:.3f}")
    print(f"  Auto-vortis Intensity: {entropic['auto_vortis']['intensity']:.3f}")
    print(f"  Auto-morphosis Transformation Rate: {entropic['auto_morphosis']['transformation_rate']:.3f}")
    print(f"  Stream Energy: {entropic['stream_energy']:.3f}")
    print()
    
    print("Negnentropic Stream (negen-tropis → auto-stasis → auto-poiesis):")
    negnentropic = results['negnentropic']
    print(f"  Negen-tropis Level: {negnentropic['negnentropis_level']:.3f}")
    print(f"  Auto-stasis Stability: {negnentropic['auto_stasis']['stability']:.3f}")
    print(f"  Auto-poiesis Self-creation Rate: {negnentropic['auto_poiesis']['self_creation_rate']:.3f}")
    print(f"  Stream Stability: {negnentropic['stream_stability']:.3f}")
    print()
    
    print("Identity Stream (iden-tropis → auto-gnosis → auto-genesis):")
    identity = results['identity']
    print(f"  Iden-tropis Level: {identity['identropis_level']:.3f}")
    print(f"  Auto-gnosis Awareness: {identity['auto_gnosis']['awareness']:.3f}")
    print(f"  Auto-genesis Self-generation Rate: {identity['auto_genesis']['self_generation_rate']:.3f}")
    print(f"  Stream Coherence: {identity['stream_coherence']:.3f}")
    print()
    
    print("Integrated Dynamics:")
    integrated = results['integrated_dynamics']
    print(f"  Stream Coherence: {integrated['stream_coherence']:.3f}")
    print(f"  Dynamic Balance: {integrated['dynamic_balance']:.3f}")
    print(f"  Emergent Synergy: {integrated['emergent_synergy']:.3f}")
    print(f"  Evolution Rate: {integrated['organizational_evolution']['evolution_rate']:.3f}")
    print(f"  Evolution Direction: {integrated['organizational_evolution']['evolution_direction']}")
    print()


async def test_full_metamodel_orchestrator():
    """Test the complete holistic metamodel orchestrator."""
    print("=" * 60)
    print("Testing Complete Holistic Metamodel Orchestrator")
    print("=" * 60)
    
    orchestrator = HolisticMetamodelOrchestrator()
    
    # Initialize with test context
    system_context = {
        'total_components': 8,
        'active_components': 6,
        'system_energy': 0.8,
        'structural_integrity': 0.85,
        'foundation_strength': 0.75,
        'adaptation_rate': 0.6,
        'processing_load': 0.5,
        'evolution_rate': 0.4,
        'component_connectivity': 0.7,
        'interaction_rate': 0.6,
        'system_coherence': 0.8,
        'max_hierarchical_levels': 4,
        'initial_integration': 0.85
    }
    
    await orchestrator.initialize_metamodel(system_context)
    print("Metamodel initialized successfully")
    
    # Run a processing cycle
    system_state = {
        'component_coordination': 0.75,
        'pattern_coherence': 0.7,
        'system_integration': 0.8,
        'structural_integrity': 0.85,
        'functional_coherence': 0.75,
        'equilibrium_maintenance': 0.7,
        'self_recognition': 0.65,
        'boundary_definition': 0.75,
        'identity_coherence': 0.7,
        'system_energy': 0.8,
        'cognitive_complexity': 0.6
    }
    
    cycle_results = await orchestrator.process_metamodel_cycle(system_state)
    
    print(f"Metamodel Coherence: {cycle_results['metamodel_coherence']:.3f}")
    print(f"Monad Manifestations: {len(cycle_results['monad_manifestations'])} levels")
    print(f"Dual Resolutions: {len(cycle_results['dual_resolutions'])} levels")
    print(f"Triadic States: {len(cycle_results['triadic_states'])} levels")
    print(f"Cycle Progressions: {len(cycle_results['cycle_progressions'])} levels")
    
    # Get status
    status = orchestrator.get_metamodel_status()
    print("\nMetamodel Status:")
    print(f"  Coherence Level: {status['metamodel_state']['coherence_level']:.3f}")
    print(f"  Integration Depth: {status['metamodel_state']['integration_depth']}")
    print(f"  Active Levels: {status['active_levels']}")
    print(f"  Monad Essence: {status['monad_essence']}")
    print()


async def main():
    """Run all tests."""
    print("🧠⚡ Testing Eric Schwarz's Holistic Metamodel Implementation")
    print("=" * 80)
    
    try:
        await test_hieroglyphic_monad()
        await test_dual_complementarity()
        await test_triadic_system()
        await test_self_stabilizing_cycle()
        await test_organizational_dynamics()
        await test_full_metamodel_orchestrator()
        
        print("=" * 80)
        print("✅ All tests completed successfully!")
        print("🌟 Holistic Metamodel is operational and integrated")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
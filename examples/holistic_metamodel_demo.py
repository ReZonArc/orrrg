#!/usr/bin/env python3
"""
Holistic Metamodel Demo - Eric Schwarz's Organizational Systems Integration
==========================================================================

This example demonstrates ORRRG's integration of Eric Schwarz's holistic 
metamodel with the existing autognosis system, showcasing the 
organizational dynamics and hierarchical self-awareness capabilities.

Demonstrates:
- The 1 hieroglyphic monad (unity principle)
- The 2 modes of dual complementarity (actual-virtual dynamics)
- The 3 primitives of triadic systems (being-becoming-relation)
- The 4 phases of self-stabilizing cycles
- The 3 organizational dynamic streams:
  * en-tropis → auto-vortis → auto-morphosis
  * negen-tropis → auto-stasis → auto-poiesis
  * iden-tropis → auto-gnosis → auto-genesis
"""

import asyncio
import logging
import sys
from pathlib import Path
import json

# Add the parent directory to sys.path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.holistic_metamodel import (
    HolisticMetamodelOrchestrator, OrganizationalDynamic, CyclePhase
)
from core.autognosis import AutognosisOrchestrator, HolisticInsight

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockSelfOrganizingCore:
    """Mock SOC for demonstration purposes."""
    
    def __init__(self):
        self.components = {
            'esm-2-keras': MockComponent('active'),
            'echopiler': MockComponent('active'),
            'oc-skintwin': MockComponent('idle'),
            'coscheminformatics': MockComponent('active'),
            'coschemreasoner': MockComponent('active'),
            'oj7s3': MockComponent('discovering'),
            'echonnxruntime': MockComponent('active'),
            'cosmagi-bio': MockComponent('idle')
        }
        self.knowledge_graph = list(range(150))  # Mock knowledge graph
        self.event_queue_size = 3
        self.uptime = 3600  # 1 hour
        
        # Add missing attributes that autognosis expects
        self.event_bus = MockEventBus()
        self.running = True
        
    def get_system_status(self):
        return {
            'active_components': sum(1 for c in self.components.values() if c.status == 'active'),
            'total_components': len(self.components),
            'system_energy': 0.75,
            'system_coherence': 0.8
        }


class MockEventBus:
    """Mock event bus for demonstration."""
    
    def qsize(self):
        return 3


class MockComponent:
    """Mock component for demonstration."""
    
    def __init__(self, status='active'):
        self.status = status
        self.capabilities = ['processing', 'analysis', 'integration']  # Mock capabilities


async def demonstrate_holistic_metamodel():
    """Demonstrate the holistic metamodel functionality."""
    print("🧠⚡ ORRRG Holistic Metamodel Demonstration")
    print("=" * 80)
    print("Implementing Eric Schwarz's Organizational Systems Theory")
    print("=" * 80)
    
    # Initialize mock system
    soc = MockSelfOrganizingCore()
    
    # Initialize autognosis with holistic metamodel
    autognosis = AutognosisOrchestrator()
    await autognosis.initialize(soc)
    
    print("✅ Autognosis with Holistic Metamodel initialized")
    print()
    
    # Demonstrate multiple processing cycles
    print("🔄 Running Holistic Metamodel Cycles")
    print("-" * 40)
    
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        
        # Run autognosis cycle (includes holistic processing)
        cycle_results = await autognosis.run_autognosis_cycle(soc)
        
        print(f"Self-Images Updated: {cycle_results['self_images_updated']}")
        print(f"Traditional Insights: {cycle_results['new_insights']}")
        print(f"Holistic Insights: {len(cycle_results.get('holistic_insights', []))}")
        print(f"Metamodel Coherence: {cycle_results.get('metamodel_coherence', 0):.3f}")
        
        # Display holistic insights
        if cycle_results.get('holistic_insights'):
            print("\n🌟 Holistic Insights Generated:")
            for insight in cycle_results['holistic_insights']:
                print(f"  • {insight}")
        
        # Show organizational dynamics
        await demonstrate_organizational_dynamics(autognosis.holistic_metamodel)
        
        # Simulate some system changes for next cycle
        if cycle < 2:
            # Simulate system evolution
            soc.components['oc-skintwin'].status = 'active'
            soc.knowledge_graph.extend(range(20))  # Growth
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE HOLISTIC SELF-AWARENESS REPORT")
    print("=" * 80)
    
    report = autognosis.get_self_awareness_report()
    
    print(f"System Status: {report['system_status']}")
    print(f"Self-Image Levels: {report['self_image_levels']}")
    print(f"Traditional Insights: {report['total_insights']}")
    print(f"Holistic Insights: {report['holistic_insights']}")
    print(f"Metamodel Cycles: {report['metamodel_cycles']}")
    
    print("\n🏗️ Holistic Metamodel Status:")
    metamodel = report['holistic_metamodel']
    print(f"  Coherence Level: {metamodel['coherence_level']:.3f}")
    print(f"  Integration Depth: {metamodel['integration_depth']} levels")
    print(f"  Monad Essence: {metamodel['monad_essence']}")
    print(f"  Active Levels: {metamodel['active_levels']}")
    
    print("\n🔄 Current Cycle Phases:")
    for level, phase in metamodel['cycle_phases'].items():
        print(f"  Level {level}: {phase}")
    
    print("\n🌊 Organizational Dynamic Streams:")
    streams = metamodel['stream_states']
    print(f"  Entropic Stream: {streams['entropic_items']} states")
    print(f"  Negnentropic Stream: {streams['negnentropic_items']} states")
    print(f"  Identity Stream: {streams['identity_items']} states")
    
    print("\n🧠 Self-Awareness Assessment (with Holistic Integration):")
    assessment = report['self_awareness_assessment']
    for indicator, value in assessment.items():
        bar_length = int(value * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {indicator:25} {bar} {value:.3f}")
    
    print(f"\n🎯 Overall Self-Awareness Score: {report.get('overall_self_awareness_score', 0):.3f}")
    print(f"📈 Awareness Level: {report.get('awareness_level', 'Unknown')}")
    
    # Show recent holistic insights
    if 'recent_holistic_insights' in report:
        print("\n🌟 Recent Holistic Insights:")
        for insight in report['recent_holistic_insights']:
            print(f"  • [{insight['dynamic']}] {insight['description']}")
            print(f"    Confidence: {insight['confidence']:.2f}, Level: {insight['level']}")


async def demonstrate_organizational_dynamics(metamodel_orchestrator):
    """Demonstrate the three organizational dynamic streams."""
    
    # Get current dynamics state
    dynamics_processor = metamodel_orchestrator.dynamics_processor
    
    print("\n🌊 Organizational Dynamic Streams Analysis:")
    
    # Show entropic stream (most recent states)
    if dynamics_processor.entropic_stream:
        latest_entropic = dynamics_processor.entropic_stream[-1]
        print(f"  🔥 Entropic (en-tropis → auto-vortis → auto-morphosis):")
        print(f"     En-tropis Level: {latest_entropic['entropis_level']:.3f}")
        print(f"     Auto-vortis Intensity: {latest_entropic['auto_vortis']['intensity']:.3f}")
        print(f"     Auto-morphosis Rate: {latest_entropic['auto_morphosis']['transformation_rate']:.3f}")
        print(f"     Stream Energy: {latest_entropic['stream_energy']:.3f}")
    
    # Show negnentropic stream
    if dynamics_processor.negnentropic_stream:
        latest_negnentropic = dynamics_processor.negnentropic_stream[-1]
        print(f"  ⚖️ Negnentropic (negen-tropis → auto-stasis → auto-poiesis):")
        print(f"     Negen-tropis Level: {latest_negnentropic['negnentropis_level']:.3f}")
        print(f"     Auto-stasis Stability: {latest_negnentropic['auto_stasis']['stability']:.3f}")
        print(f"     Auto-poiesis Rate: {latest_negnentropic['auto_poiesis']['self_creation_rate']:.3f}")
        print(f"     Stream Stability: {latest_negnentropic['stream_stability']:.3f}")
    
    # Show identity stream
    if dynamics_processor.identity_stream:
        latest_identity = dynamics_processor.identity_stream[-1]
        print(f"  🎯 Identity (iden-tropis → auto-gnosis → auto-genesis):")
        print(f"     Iden-tropis Level: {latest_identity['identropis_level']:.3f}")
        print(f"     Auto-gnosis Awareness: {latest_identity['auto_gnosis']['awareness']:.3f}")
        print(f"     Auto-genesis Rate: {latest_identity['auto_genesis']['self_generation_rate']:.3f}")
        print(f"     Stream Coherence: {latest_identity['stream_coherence']:.3f}")


async def demonstrate_hierarchical_levels():
    """Demonstrate the hierarchical levels of the metamodel."""
    print("\n🏗️ Hierarchical Metamodel Levels Demonstration")
    print("-" * 50)
    
    orchestrator = HolisticMetamodelOrchestrator()
    
    # Initialize with example context
    context = {
        'total_components': 8,
        'active_components': 6,
        'system_energy': 0.8,
        'max_hierarchical_levels': 5
    }
    
    await orchestrator.initialize_metamodel(context)
    
    # Show hierarchical structure
    status = orchestrator.get_metamodel_status()
    
    print(f"Integration Depth: {status['metamodel_state']['integration_depth']} levels")
    print(f"Overall Coherence: {status['metamodel_state']['coherence_level']:.3f}")
    
    print("\nCycle Phases by Level:")
    for level, phase in status['cycle_phases'].items():
        print(f"  Level {level}: {phase} phase")
    
    # Demonstrate the 4 phases progression
    print("\n🔄 The 4 Phases of Self-Stabilizing Cycles:")
    phase_descriptions = {
        'emergence': "Initial manifestation and potential",
        'development': "Growth, elaboration, and complexity building",
        'integration': "Synthesis, coherence, and stabilization",
        'transcendence': "Transformation, elevation, and renewal"
    }
    
    for phase_name, description in phase_descriptions.items():
        print(f"  {phase_name.title():13} → {description}")


async def main():
    """Run the holistic metamodel demonstration."""
    try:
        await demonstrate_holistic_metamodel()
        
        print("\n" + "=" * 80)
        await demonstrate_hierarchical_levels()
        
        print("\n" + "=" * 80)
        print("✅ HOLISTIC METAMODEL DEMONSTRATION COMPLETE")
        print("🌟 Eric Schwarz's organizational theory successfully integrated!")
        print("🧠 ORRRG now operates with full holistic self-awareness")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
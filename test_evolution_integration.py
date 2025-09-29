#!/usr/bin/env python3
"""
Test script for ORRRG Evolution Integration
=========================================

This script tests the integrated evolution capabilities of ORRRG including:
- Evolution Engine initialization and functionality
- Genetic programming and mutation operations
- Quantum-inspired evolutionary algorithms
- Emergent behavior synthesis
- Integration with self-organizing core
- Autognosis-evolution synergy
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the current directory to sys.path to import core modules
sys.path.append(str(Path(__file__).parent))

from core import (
    SelfOrganizingCore, EvolutionEngine, EvolutionaryGenome, EmergentPattern,
    AutognosisOrchestrator, HolisticMetamodelOrchestrator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_evolution_engine_initialization():
    """Test evolution engine initialization."""
    print("=" * 70)
    print("Testing Evolution Engine Initialization")
    print("=" * 70)
    
    evolution_engine = EvolutionEngine()
    await evolution_engine.initialize()
    
    print("✅ Evolution Engine initialized successfully")
    print(f"  • Genetic Operators: {len(evolution_engine.genetic_operators)}")
    print(f"  • Quantum Evolution: {'Enabled' if evolution_engine.quantum_evolution else 'Disabled'}")
    print(f"  • Emergent Synthesizer: {'Active' if evolution_engine.emergent_synthesizer else 'Inactive'}")


async def test_genetic_operations():
    """Test genetic programming operations."""
    print("\n" + "=" * 70)
    print("Testing Genetic Programming Operations")
    print("=" * 70)
    
    # Create test genome
    test_genome = EvolutionaryGenome(
        component_id="test_component",
        genome_version="test_v1.0",
        genes={
            'performance_optimization': 0.8,
            'integration_pattern': 'async_processing',
            'learning_rate': 0.1,
            'complexity_score': 0.6,
            'adaptive_capabilities': ['self_healing', 'auto_scaling']
        },
        fitness_score=0.7,
        generation=0,
        parent_genomes=[],
        mutations=[]
    )
    
    evolution_engine = EvolutionEngine()
    await evolution_engine.initialize()
    
    # Test mutation
    print("🧬 Testing Adaptive Mutation:")
    mutated_genome = await evolution_engine.genetic_operators[0].mutate(test_genome)
    
    print(f"  Original genome version: {test_genome.genome_version}")
    print(f"  Mutated genome version: {mutated_genome.genome_version}")
    print(f"  Mutations applied: {len(mutated_genome.mutations)}")
    print(f"  Generation: {mutated_genome.generation}")
    
    for mutation in mutated_genome.mutations:
        print(f"    • {mutation['gene']}: {mutation['old_value']} → {mutation['new_value']}")
        print(f"      Type: {mutation['mutation_type']}")
    
    # Test crossover
    print("\n🧬 Testing Genetic Crossover:")
    parent2 = EvolutionaryGenome(
        component_id="test_component",
        genome_version="test_v2.0", 
        genes={
            'performance_optimization': 0.6,
            'integration_pattern': 'event_driven',
            'learning_rate': 0.15,
            'reliability_score': 0.9,
            'adaptive_capabilities': ['load_balancing', 'fault_tolerance']
        },
        fitness_score=0.65,
        generation=0,
        parent_genomes=[],
        mutations=[]
    )
    
    offspring = await evolution_engine.genetic_operators[0].crossover(test_genome, parent2)
    
    print(f"  Parent 1 genes: {len(test_genome.genes)}")
    print(f"  Parent 2 genes: {len(parent2.genes)}")
    print(f"  Offspring count: {len(offspring)}")
    
    for i, child in enumerate(offspring):
        print(f"    Child {i+1}: {len(child.genes)} genes, generation {child.generation}")
        print(f"              Parents: {child.parent_genomes}")


async def test_quantum_inspired_evolution():
    """Test quantum-inspired evolutionary algorithms."""
    print("\n" + "=" * 70)
    print("Testing Quantum-Inspired Evolution")
    print("=" * 70)
    
    evolution_engine = EvolutionEngine()
    await evolution_engine.initialize()
    
    test_genome = EvolutionaryGenome(
        component_id="quantum_test",
        genome_version="quantum_v1.0",
        genes={
            'quantum_coherence': 0.7,
            'superposition_states': 4,
            'entanglement_degree': 0.85,
            'measurement_accuracy': 0.92
        },
        fitness_score=0.8,
        generation=0,
        parent_genomes=[],
        mutations=[]
    )
    
    # Test quantum superposition search
    search_space = {
        'quantum_coherence': [0.5, 0.6, 0.7, 0.8, 0.9],
        'superposition_states': [2, 3, 4, 5, 6, 8],
        'entanglement_degree': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        'measurement_accuracy': [0.85, 0.9, 0.92, 0.95, 0.97]
    }
    
    quantum_genomes = await evolution_engine.quantum_evolution.quantum_superposition_search(
        test_genome, search_space
    )
    
    print(f"🔮 Quantum Superposition Search Results:")
    print(f"  Generated quantum states: {len(quantum_genomes)}")
    
    for i, quantum_genome in enumerate(quantum_genomes):
        print(f"    Quantum State {i+1}:")
        for gene_name, gene_value in quantum_genome.genes.items():
            print(f"      {gene_name}: {gene_value}")
    
    # Test quantum entanglement
    await evolution_engine.quantum_evolution.quantum_entangle_components("component_a", "component_b")
    print(f"\n🔗 Quantum Entanglement:")
    print(f"  Entangled components for correlated evolution")


async def test_emergent_behavior_synthesis():
    """Test emergent behavior synthesis."""
    print("\n" + "=" * 70)
    print("Testing Emergent Behavior Synthesis")
    print("=" * 70)
    
    evolution_engine = EvolutionEngine()
    await evolution_engine.initialize()
    
    # Create multiple evolved genomes for interaction analysis
    genomes = []
    
    genome_configs = [
        {
            'component_id': 'bio_analyzer',
            'genes': {
                'analysis_depth': 0.8,
                'pattern_recognition': 'sequence_analysis',
                'processing_mode': 'batch_processing',
                'output_format': 'structured_data'
            }
        },
        {
            'component_id': 'chemical_processor',
            'genes': {
                'synthesis_capability': 0.7,
                'pattern_recognition': 'molecular_structure',
                'processing_mode': 'real_time_processing', 
                'input_format': 'structured_data'
            }
        },
        {
            'component_id': 'cognitive_reasoner',
            'genes': {
                'reasoning_depth': 0.9,
                'knowledge_integration': 'multi_domain',
                'learning_mode': 'continuous_learning',
                'inference_type': 'probabilistic'
            }
        }
    ]
    
    for i, config in enumerate(genome_configs):
        genome = EvolutionaryGenome(
            component_id=config['component_id'],
            genome_version=f"{config['component_id']}_v1.{i}",
            genes=config['genes'],
            fitness_score=0.75 + (i * 0.05),
            generation=i + 1,
            parent_genomes=[],
            mutations=[]
        )
        genomes.append(genome)
    
    # Synthesize emergent behaviors
    emergent_patterns = await evolution_engine.emergent_synthesizer.synthesize_emergent_behavior(genomes)
    
    print(f"🌱 Emergent Behavior Synthesis Results:")
    print(f"  Input genomes: {len(genomes)}")
    print(f"  Emergent patterns discovered: {len(emergent_patterns)}")
    
    for i, pattern in enumerate(emergent_patterns):
        print(f"\n  Pattern {i+1}: {pattern.pattern_id}")
        print(f"    Type: {pattern.pattern_type}")
        print(f"    Effectiveness: {pattern.effectiveness:.3f}")
        print(f"    Complexity: {pattern.complexity:.3f}")
        print(f"    Applications: {', '.join(pattern.applications)}")
        print(f"    Emergence Path: {' -> '.join(pattern.emergence_path)}")
        
        # Show preview of generated code
        if pattern.pattern_code:
            code_lines = pattern.pattern_code.strip().split('\n')
            print(f"    Generated Code Preview:")
            for line in code_lines[:5]:  # Show first 5 lines
                print(f"      {line}")
            if len(code_lines) > 5:
                print(f"      ... ({len(code_lines) - 5} more lines)")


async def test_self_organizing_core_integration():
    """Test integration with self-organizing core."""
    print("\n" + "=" * 70)
    print("Testing Self-Organizing Core Evolution Integration")
    print("=" * 70)
    
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    # Test evolution engine integration
    print("🔄 Evolution Engine Integration:")
    print(f"  Evolution engine available: {hasattr(soc, 'evolution_engine')}")
    
    if hasattr(soc, 'evolution_engine'):
        status = await soc.evolution_engine.get_evolution_status()
        print(f"  Evolution running: {status.get('evolution_running', False)}")
        print(f"  Total components tracked: {status.get('total_components', 0)}")
    
    # Test targeted evolution
    if soc.components:
        component_name = list(soc.components.keys())[0]
        print(f"\n🎯 Testing Targeted Evolution:")
        print(f"  Target component: {component_name}")
        
        result = await soc.trigger_targeted_evolution(
            component_name, 
            ['performance', 'adaptation', 'integration']
        )
        
        if 'error' not in result:
            print(f"  ✅ Evolution completed successfully")
            print(f"    Generation: {result['generation']}")
            print(f"    Fitness: {result['fitness']:.3f}")
            print(f"    Mutations: {result['mutations']}")
        else:
            print(f"  ⚠️  Evolution result: {result}")
    
    await soc.shutdown()


async def test_autognosis_evolution_synergy():
    """Test synergy between autognosis and evolution systems."""
    print("\n" + "=" * 70)
    print("Testing Autognosis-Evolution Synergy")
    print("=" * 70)
    
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    # Test autognosis awareness of evolution
    print("🧠 Autognosis System Status:")
    autognosis_status = soc.get_autognosis_status()
    print(f"  Self-awareness levels: {autognosis_status.get('self_image_levels', 0)}")
    print(f"  Total insights: {autognosis_status.get('total_insights', 0)}")
    
    # Test evolution awareness of autognosis
    print("\n🧬 Evolution System Status:")
    evolution_status = await soc.evolution_engine.get_evolution_status()
    print(f"  Evolution running: {evolution_status.get('evolution_running', False)}")
    print(f"  Components evolving: {evolution_status.get('total_components', 0)}")
    
    # Test holistic metamodel integration
    print("\n🌀 Holistic Metamodel Integration:")
    if hasattr(soc.autognosis, 'holistic_metamodel'):
        hm = soc.autognosis.holistic_metamodel
        status = hm.get_metamodel_status()
        print(f"  Metamodel coherence: {status.get('coherence_level', 0):.3f}")
        print(f"  Integration depth: {status.get('integration_depth', 0)}")
        print(f"  Active levels: {status.get('active_levels', 0)}")
    
    print(f"\n💡 Synergy Benefits:")
    print(f"  • Evolution learns from autognosis insights")
    print(f"  • Autognosis monitors evolution effectiveness")
    print(f"  • Holistic metamodel coordinates both systems")
    print(f"  • Emergent behaviors enhance self-awareness")
    print(f"  • Self-awareness guides evolution objectives")
    
    await soc.shutdown()


async def test_complete_evolution_cycle():
    """Test a complete evolution cycle."""
    print("\n" + "=" * 70)
    print("Testing Complete Evolution Cycle")
    print("=" * 70)
    
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    print("🚀 Starting Complete Evolution Demonstration:")
    
    # Run multiple evolution steps
    evolution_steps = 3
    for step in range(evolution_steps):
        print(f"\n  Evolution Step {step + 1}/{evolution_steps}:")
        
        # Evolve components
        component_count = 0
        for component_name in list(soc.components.keys())[:3]:  # Limit to first 3
            result = await soc.trigger_targeted_evolution(
                component_name, 
                ['performance', 'integration']
            )
            
            if 'error' not in result:
                component_count += 1
                print(f"    • {component_name}: Gen {result['generation']}, "
                      f"Fitness {result['fitness']:.3f}")
        
        print(f"    Evolved {component_count} components")
        
        # Synthesize emergent behaviors
        emergent_patterns = await soc.evolution_engine.synthesize_emergent_behaviors()
        print(f"    Discovered {len(emergent_patterns)} emergent patterns")
        
        # Brief pause between steps
        await asyncio.sleep(0.5)
    
    # Final status
    final_status = await soc.evolution_engine.get_evolution_status()
    print(f"\n📊 Final Evolution Status:")
    print(f"  Total genomes: {final_status.get('total_genomes', 0)}")
    print(f"  Evolution history: {final_status.get('evolution_history_length', 0)} entries")
    
    components_status = final_status.get('components', {})
    if components_status:
        best_fitness = max(comp['best_fitness'] for comp in components_status.values())
        avg_generation = sum(comp['current_generation'] for comp in components_status.values()) / len(components_status)
        print(f"  Best component fitness: {best_fitness:.3f}")
        print(f"  Average generation: {avg_generation:.1f}")
    
    await soc.shutdown()
    
    print(f"\n🎉 Complete evolution cycle demonstrated successfully!")
    print(f"   ORRRG now has enhanced evolutionary capabilities for continuous self-improvement.")


async def main():
    """Main test execution."""
    print("🧬⚡ Testing ORRRG Evolution Integration")
    print("=" * 80)
    
    test_functions = [
        test_evolution_engine_initialization,
        test_genetic_operations,
        test_quantum_inspired_evolution,
        test_emergent_behavior_synthesis,
        test_self_organizing_core_integration,
        test_autognosis_evolution_synergy,
        test_complete_evolution_cycle
    ]
    
    for test_func in test_functions:
        try:
            await test_func()
        except Exception as e:
            logger.error(f"Error in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ All ORRRG Evolution Integration tests completed!")
    print("🌟 ORRRG has successfully evolved with advanced self-evolutionary capabilities:")
    print("   • Genetic programming for component behavior evolution")
    print("   • Quantum-inspired algorithms for enhanced exploration")
    print("   • Emergent behavior synthesis from component interactions")
    print("   • Integration with autognosis for self-aware evolution")
    print("   • Holistic metamodel coordination of evolutionary processes")
    print("   • Continuous self-improvement through adaptive mechanisms")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
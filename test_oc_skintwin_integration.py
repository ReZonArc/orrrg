#!/usr/bin/env python3
"""
Test suite for oc-skintwin (OpenCog AGI) component integration
================================================================

Tests the cognitive reasoning capabilities and integration with:
- ORRRG Self-Organizing Core
- Relevance Realization Ennead
- Evolution Engine
"""

import asyncio
import sys
import logging
from pathlib import Path
import pytest
import importlib

# Add the current directory to sys.path to import core modules
sys.path.append(str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'components'))

from core import SelfOrganizingCore

# Import oc-skintwin component using importlib (module name has hyphen)
oc_skintwin = importlib.import_module('oc-skintwin')
Oc_SkintwinComponent = oc_skintwin.Oc_SkintwinComponent
AtomType = oc_skintwin.AtomType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_oc_skintwin_initialization():
    """Test oc-skintwin component initialization."""
    print("\n" + "=" * 70)
    print("🧠 Test: oc-skintwin Initialization")
    print("=" * 70)
    
    # Import component
    
    print("\n1. Creating oc-skintwin component...")
    component = Oc_SkintwinComponent()
    assert component is not None
    print("   ✓ Component created")
    
    print("\n2. Initializing component...")
    result = await component.initialize({})
    assert result == True
    assert component.initialized
    print("   ✓ Component initialized")
    
    print("\n3. Getting component info...")
    info = component.get_info()
    assert info['name'] == 'oc-skintwin'
    assert 'AGI' in info['description'] or 'cognitive' in info['description']
    print(f"   ✓ Name: {info['name']}")
    print(f"   ✓ Version: {info['version']}")
    
    print("\n4. Getting capabilities...")
    capabilities = component.get_capabilities()
    assert 'cognitive_reasoning' in capabilities
    assert 'agi_reasoning' in capabilities
    print(f"   ✓ Capabilities: {len(capabilities)}")
    for cap in capabilities[:5]:
        print(f"     • {cap}")
    
    await component.cleanup()
    print("\n✅ oc-skintwin initialization test PASSED")


@pytest.mark.asyncio
async def test_atomspace_operations():
    """Test AtomSpace knowledge representation operations."""
    print("\n" + "=" * 70)
    print("🧬 Test: AtomSpace Operations")
    print("=" * 70)
    
    
    component = Oc_SkintwinComponent()
    await component.initialize({})
    
    print("\n1. Testing knowledge representation...")
    # Query initial knowledge
    result = await component.query_knowledge({
        'type': AtomType.CONCEPT.value
    })
    
    assert result['status'] == 'success'
    initial_concepts = result['matches']
    print(f"   ✓ Initial concepts: {initial_concepts}")
    
    print("\n2. Testing pattern learning...")
    patterns = [
        {'type': 'genomic', 'data': 'DNA_SEQUENCE'},
        {'type': 'chemical', 'data': 'MOLECULAR_STRUCTURE'},
        {'type': 'reasoning', 'data': 'LOGICAL_RULE'}
    ]
    
    learn_result = await component.learn_patterns(patterns)
    assert learn_result['status'] == 'success'
    print(f"   ✓ Patterns learned: {learn_result['patterns_learned']}")
    print(f"   ✓ Total patterns: {learn_result['total_learned_patterns']}")
    
    print("\n3. Verifying AtomSpace statistics...")
    info = component.get_info()
    stats = info.get('statistics', {}).get('atomspace', {})
    print(f"   ✓ Total atoms: {stats.get('total_atoms', 0)}")
    
    await component.cleanup()
    print("\n✅ AtomSpace operations test PASSED")


@pytest.mark.asyncio
async def test_cognitive_reasoning():
    """Test cognitive reasoning capabilities."""
    print("\n" + "=" * 70)
    print("🤔 Test: Cognitive Reasoning")
    print("=" * 70)
    
    
    component = Oc_SkintwinComponent()
    await component.initialize({})
    
    print("\n1. Performing cognitive reasoning...")
    result = await component.cognitive_reasoning(
        query="protein structure analysis",
        context={
            'domain': 'biology',
            'task': 'structure_prediction'
        }
    )
    
    assert result['status'] == 'success'
    assert 'reasoning_cycle' in result
    print(f"   ✓ Reasoning cycle: {result['reasoning_cycle']}")
    print(f"   ✓ Matches found: {result.get('matches', 0)}")
    
    print("\n2. Testing inference...")
    result2 = await component.cognitive_reasoning(
        query="chemical synthesis",
        context={'domain': 'chemistry'}
    )
    
    assert result2['reasoning_cycle'] == result['reasoning_cycle'] + 1
    print(f"   ✓ Reasoning cycle incremented: {result2['reasoning_cycle']}")
    
    print("\n3. Verifying inferences...")
    inferences = result2.get('inferences', {})
    print(f"   ✓ Inferences generated")
    
    await component.cleanup()
    print("\n✅ Cognitive reasoning test PASSED")


@pytest.mark.asyncio
async def test_relevance_integration():
    """Test integration with Relevance Realization Ennead."""
    print("\n" + "=" * 70)
    print("🎯 Test: Relevance Realization Integration")
    print("=" * 70)
    
    
    component = Oc_SkintwinComponent()
    await component.initialize({})
    
    # Add some knowledge first
    await component.learn_patterns([
        {'name': 'genomic_analysis', 'relevance': 'high'},
        {'name': 'protein_folding', 'relevance': 'high'},
        {'name': 'chemical_structure', 'relevance': 'medium'}
    ])
    
    print("\n1. Integrating with relevance context...")
    relevance_context = {
        'task_type': 'biological_analysis',
        'requirements': ['genomic', 'protein'],
        'domain': 'computational_biology'
    }
    
    result = await component.integrate_with_relevance(relevance_context)
    
    assert result['status'] == 'success'
    assert 'attention_allocated' in result
    print(f"   ✓ Attention allocated to: {result['attention_allocated']} atoms")
    
    print("\n2. Checking top attended atoms...")
    top_atoms = result.get('top_attended_atoms', [])
    print(f"   ✓ Top attended atoms: {len(top_atoms)}")
    
    for i, atom_info in enumerate(top_atoms[:3]):
        print(f"     #{i+1}: Attention={atom_info['attention']:.3f}")
    
    await component.cleanup()
    print("\n✅ Relevance integration test PASSED")


@pytest.mark.asyncio
async def test_evolutionary_reasoning():
    """Test evolutionary optimization of reasoning patterns."""
    print("\n" + "=" * 70)
    print("🧬 Test: Evolutionary Reasoning")
    print("=" * 70)
    
    
    component = Oc_SkintwinComponent()
    await component.initialize({})
    
    # Add initial patterns
    await component.learn_patterns([
        {'pattern': 'rule1'},
        {'pattern': 'rule2'},
        {'pattern': 'rule3'}
    ])
    
    print("\n1. Evolving reasoning patterns for efficiency...")
    result = await component.evolve_reasoning(['efficiency'])
    
    assert result['status'] == 'success'
    assert 'evolutions' in result
    print(f"   ✓ Evolutions applied: {len(result['evolutions'])}")
    
    print("\n2. Evolving for connectivity...")
    result2 = await component.evolve_reasoning(['connectivity'])
    
    assert result2['status'] == 'success'
    print(f"   ✓ Connectivity evolutions: {len(result2['evolutions'])}")
    
    print("\n3. Evolving for diversity...")
    result3 = await component.evolve_reasoning(['diversity'])
    
    assert result3['status'] == 'success'
    print(f"   ✓ Diversity evolutions: {len(result3['evolutions'])}")
    
    print("\n4. Verifying AtomSpace after evolution...")
    info = component.get_info()
    stats = info['statistics']
    print(f"   ✓ Reasoning cycles: {stats['reasoning_cycles']}")
    
    await component.cleanup()
    print("\n✅ Evolutionary reasoning test PASSED")


@pytest.mark.asyncio
async def test_soc_integration():
    """Test full integration with Self-Organizing Core."""
    print("\n" + "=" * 70)
    print("🌐 Test: Self-Organizing Core Integration")
    print("=" * 70)
    
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    print("\n1. Verifying oc-skintwin in components...")
    assert 'oc-skintwin' in soc.components
    component_info = soc.components['oc-skintwin']
    print(f"   ✓ Component status: {component_info.status}")
    print(f"   ✓ Capabilities: {len(component_info.capabilities)}")
    
    print("\n2. Checking relevance frame...")
    relevance_frames = soc.relevance_realization.relevance_frames
    assert 'oc-skintwin' in relevance_frames
    
    frame = relevance_frames['oc-skintwin']
    print(f"   ✓ Relevance frame created")
    print(f"   ✓ Frame coherence: {frame.coherence_score:.3f}")
    
    print("\n3. Testing cognitive reasoning pipeline...")
    # Queue cognitive reasoning event
    event = {
        'type': 'cognitive_reasoning',
        'component': 'oc-skintwin',
        'query': 'multi_domain_integration',
        'context': {
            'domains': ['biology', 'chemistry', 'reasoning']
        }
    }
    
    await soc.event_bus.put(event)
    print(f"   ✓ Cognitive reasoning event queued")
    
    print("\n4. Testing evolution integration...")
    if 'oc-skintwin' in soc.components:
        evolution_result = await soc.trigger_targeted_evolution(
            'oc-skintwin',
            ['cognitive_enhancement', 'efficiency', 'connectivity']
        )
        
        print(f"   ✓ Evolution triggered")
        print(f"   ✓ Fitness: {evolution_result.get('fitness', 0):.3f}")
    
    await soc.shutdown()
    print("\n✅ SOC integration test PASSED")


@pytest.mark.asyncio
async def test_end_to_end_agi_workflow():
    """Test complete AGI reasoning workflow."""
    print("\n" + "=" * 70)
    print("🚀 Test: End-to-End AGI Workflow")
    print("=" * 70)
    
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    print("\n1. Phase 1: Knowledge Acquisition")
    component = Oc_SkintwinComponent()
    await component.initialize({})
    
    # Learn domain knowledge
    patterns = [
        {'domain': 'genomics', 'concept': 'DNA_sequence'},
        {'domain': 'genomics', 'concept': 'gene_expression'},
        {'domain': 'chemistry', 'concept': 'molecular_structure'},
        {'domain': 'chemistry', 'concept': 'reaction_pathway'}
    ]
    
    learn_result = await component.learn_patterns(patterns)
    print(f"   ✓ Learned {learn_result['patterns_learned']} domain patterns")
    
    print("\n2. Phase 2: Relevance-Guided Reasoning")
    # Optimize for specific task
    context = {
        'task_type': 'protein_analysis',
        'requirements': ['genomics', 'chemistry', 'structure'],
        'domain': 'computational_biology'
    }
    
    relevance_result = await soc.relevance_realization.optimize_relevance_realization(context)
    print(f"   ✓ Relevance score: {relevance_result['relevance_score']:.3f}")
    print(f"   ✓ Salient components: {len(relevance_result['salient_components'])}")
    
    print("\n3. Phase 3: Cognitive Reasoning")
    # Perform reasoning with relevance context
    reasoning_result = await component.cognitive_reasoning(
        query="protein structure prediction",
        context=context
    )
    
    print(f"   ✓ Reasoning completed: cycle {reasoning_result['reasoning_cycle']}")
    print(f"   ✓ Knowledge matches: {reasoning_result.get('matches', 0)}")
    
    print("\n4. Phase 4: Attention Allocation")
    # Integrate with relevance for attention
    attention_result = await component.integrate_with_relevance(context)
    print(f"   ✓ Attention allocated: {attention_result['attention_allocated']} atoms")
    
    print("\n5. Phase 5: Evolutionary Optimization")
    # Evolve reasoning patterns
    evolution_result = await component.evolve_reasoning([
        'efficiency',
        'connectivity',
        'diversity'
    ])
    
    print(f"   ✓ Evolutions applied: {len(evolution_result['evolutions'])}")
    
    print("\n6. Phase 6: Knowledge Query")
    # Query evolved knowledge
    query_result = await component.query_knowledge({
        'type': 'ConceptNode',
        'name': 'protein'
    })
    
    print(f"   ✓ Query matches: {query_result['matches']}")
    
    print("\n7. Workflow Summary:")
    info = component.get_info()
    stats = info['statistics']
    print(f"   • Reasoning cycles: {stats['reasoning_cycles']}")
    print(f"   • Learned patterns: {stats['learned_patterns']}")
    print(f"   • Total atoms: {stats['atomspace']['total_atoms']}")
    
    await component.cleanup()
    await soc.shutdown()
    print("\n✅ End-to-end AGI workflow test PASSED")


if __name__ == "__main__":
    """Run all oc-skintwin tests."""
    print("\n" + "=" * 80)
    print("🧠 OC-SKINTWIN (OPENCOG AGI) COMPONENT TEST SUITE")
    print("=" * 80)
    
    async def run_all_tests():
        """Run all tests in sequence."""
        tests = [
            test_oc_skintwin_initialization,
            test_atomspace_operations,
            test_cognitive_reasoning,
            test_relevance_integration,
            test_evolutionary_reasoning,
            test_soc_integration,
            test_end_to_end_agi_workflow
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                await test()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"\n❌ Test {test.__name__} FAILED: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print(f"📊 TEST RESULTS: {passed} passed, {failed} failed out of {len(tests)} total")
        print("=" * 80)
        
        return failed == 0
    
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

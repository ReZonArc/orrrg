"""
Enhanced integration tests for oc-skintwin component.
Tests persistence, PLN inference rules, ECAN attention allocation,
and deeper reasoning scenarios.
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path

pytestmark = pytest.mark.asyncio


class TestAtomSpacePersistence:
    """Tests for AtomSpace JSON persistence."""

    async def test_save_and_load_dict(self):
        """Test serializing/deserializing AtomSpace to/from dict."""
        from components.oc_skintwin.opencog_adapter import SimpleAtomSpace, Atom, AtomType

        space = SimpleAtomSpace()
        space.add_atom(Atom(AtomType.CONCEPT, "science", truth_value=(0.9, 0.8)))
        space.add_atom(Atom(AtomType.CONCEPT, "biology", truth_value=(0.85, 0.7)))
        space.create_link(AtomType.INHERITANCE, "ConceptNode:science", "ConceptNode:biology")

        # Save
        data = space.save_to_dict()
        assert data['version'] == 1
        assert len(data['atoms']) == 3  # 2 concepts + 1 link

        # Load into new space
        space2 = SimpleAtomSpace()
        count = space2.load_from_dict(data)
        assert count == 3
        assert space2.get_atom("ConceptNode:science") is not None
        assert space2.get_atom("ConceptNode:science").truth_value == (0.9, 0.8)

    async def test_save_and_load_json_file(self, tmp_path):
        """Test saving/loading AtomSpace to JSON file."""
        from components.oc_skintwin.opencog_adapter import SimpleAtomSpace, Atom, AtomType

        space = SimpleAtomSpace()
        space.add_atom(Atom(AtomType.CONCEPT, "chemistry"))
        space.add_atom(Atom(AtomType.PREDICATE, "reacts_with"))

        filepath = str(tmp_path / "test_atomspace.json")
        space.save_to_json(filepath)

        # Verify file exists and has content
        assert Path(filepath).exists()
        with open(filepath) as f:
            data = json.load(f)
        assert len(data['atoms']) == 2

        # Load back
        space2 = SimpleAtomSpace()
        count = space2.load_from_json(filepath)
        assert count == 2

    async def test_adapter_save_load_operations(self, tmp_path):
        """Test save/load through the adapter process method."""
        from components.oc_skintwin.opencog_adapter import OpenCogAdapter

        adapter = OpenCogAdapter()
        await adapter.initialize({})

        # Add some knowledge
        adapter._add_concept("test_persistence")
        adapter._add_concept("save_load")

        filepath = str(tmp_path / "adapter_state.json")
        result = adapter._save_state({'path': filepath})
        assert result['status'] == 'success'
        assert result['atoms_saved'] > 0

        # Load into fresh adapter
        adapter2 = OpenCogAdapter()
        await adapter2.initialize({})
        result2 = adapter2._load_state({'path': filepath})
        assert result2['status'] == 'success'
        assert result2['atoms_loaded'] > 0


class TestPLNInference:
    """Tests for PLN inference rules."""

    async def test_deduction_strength(self):
        """Test PLN deduction strength formula."""
        from components.oc_skintwin.opencog_adapter import PLNInferenceEngine

        pln = PLNInferenceEngine()

        # If A→B (0.9) and B→C (0.8), deduction should give reasonable A→C
        strength = pln.deduction_strength(sAB=0.9, sBC=0.8, sB=0.7, sC=0.5)
        assert 0 <= strength <= 1
        # With strong premises, result should be high
        assert strength > 0.5

    async def test_deduction_confidence(self):
        """Test PLN deduction confidence formula."""
        from components.oc_skintwin.opencog_adapter import PLNInferenceEngine

        pln = PLNInferenceEngine()
        conf = pln.deduction_confidence(cAB=0.9, cBC=0.8, count_B=10)
        assert 0 <= conf <= 1
        assert conf == pytest.approx(0.72, abs=0.01)

    async def test_induction_weaker_than_deduction(self):
        """Test that induction produces weaker confidence than deduction."""
        from components.oc_skintwin.opencog_adapter import PLNInferenceEngine

        pln = PLNInferenceEngine()
        ded_conf = pln.deduction_confidence(0.9, 0.8)
        ind_conf = pln.induction_confidence(0.9, 0.8)
        assert ind_conf < ded_conf

    async def test_abduction_weakest(self):
        """Test that abduction is the weakest inference rule."""
        from components.oc_skintwin.opencog_adapter import PLNInferenceEngine

        pln = PLNInferenceEngine()
        ded_conf = pln.deduction_confidence(0.9, 0.8)
        ind_conf = pln.induction_confidence(0.9, 0.8)
        abd_conf = pln.abduction_confidence(0.9, 0.8)
        assert abd_conf < ind_conf < ded_conf

    async def test_modus_ponens(self):
        """Test modus ponens rule."""
        from components.oc_skintwin.opencog_adapter import PLNInferenceEngine

        pln = PLNInferenceEngine()
        strength = pln.modus_ponens_strength(sA=0.9, sAB=0.8)
        assert strength == pytest.approx(0.72, abs=0.01)

    async def test_apply_deduction_in_atomspace(self):
        """Test applying deduction rule within AtomSpace."""
        from components.oc_skintwin.opencog_adapter import (
            SimpleAtomSpace, Atom, AtomType, PLNInferenceEngine
        )

        space = SimpleAtomSpace()
        space.add_atom(Atom(AtomType.CONCEPT, "animal", truth_value=(0.9, 0.85)))
        space.add_atom(Atom(AtomType.CONCEPT, "mortal", truth_value=(0.8, 0.9)))
        space.add_atom(Atom(AtomType.CONCEPT, "decays", truth_value=(0.7, 0.8)))

        pln = PLNInferenceEngine()
        result = pln.apply_deduction(
            space,
            "ConceptNode:animal",
            "ConceptNode:mortal",
            "ConceptNode:decays"
        )
        assert result is not None
        strength, confidence = result
        assert 0 <= strength <= 1
        assert 0 <= confidence <= 1

    async def test_pln_deduce_via_adapter(self):
        """Test PLN deduction through the adapter process method."""
        from components.oc_skintwin.opencog_adapter import OpenCogAdapter

        adapter = OpenCogAdapter()
        await adapter.initialize({})

        # The adapter initializes with some concepts
        result = await adapter._pln_deduce({
            'source_id': 'ConceptNode:knowledge',
            'middle_id': 'ConceptNode:understanding',
            'target_id': 'ConceptNode:reasoning',
        })
        assert result['status'] == 'success'
        assert 'truth_value' in result
        assert result['rule'] == 'deduction'


class TestECANAttention:
    """Tests for ECAN attention allocation."""

    async def test_stimulate_and_sti(self):
        """Test stimulating atoms and STI tracking."""
        from components.oc_skintwin.opencog_adapter import ECANAttentionBank

        ecan = ECANAttentionBank()
        ecan.stimulate("atom_1", 0.6)
        assert ecan.get_sti("atom_1") == 0.6

        ecan.stimulate("atom_1", 0.3)
        assert ecan.get_sti("atom_1") == 0.9

    async def test_focus_set(self):
        """Test focus set management."""
        from components.oc_skintwin.opencog_adapter import ECANAttentionBank

        ecan = ECANAttentionBank(focus_boundary=0.5)
        ecan.set_sti("a", 0.8)
        ecan.set_sti("b", 0.3)
        ecan.set_sti("c", 0.6)

        focus = ecan.get_focus_set()
        assert "a" in focus
        assert "c" in focus
        assert "b" not in focus

    async def test_importance_decay(self):
        """Test importance decay reduces STI."""
        from components.oc_skintwin.opencog_adapter import ECANAttentionBank

        ecan = ECANAttentionBank(decay_rate=0.1)
        ecan.set_sti("x", 1.0)

        ecan.decay_importance()
        assert ecan.get_sti("x") == pytest.approx(0.9, abs=0.01)

        ecan.decay_importance()
        assert ecan.get_sti("x") == pytest.approx(0.81, abs=0.01)

    async def test_importance_spreading(self):
        """Test STI spreading to neighbors."""
        from components.oc_skintwin.opencog_adapter import (
            SimpleAtomSpace, Atom, AtomType, ECANAttentionBank
        )

        space = SimpleAtomSpace()
        space.add_atom(Atom(AtomType.CONCEPT, "center"))
        space.add_atom(Atom(AtomType.CONCEPT, "neighbor"))
        space.create_link(AtomType.SIMILARITY, "ConceptNode:center", "ConceptNode:neighbor")

        ecan = ECANAttentionBank(focus_boundary=0.5)
        ecan.set_sti("ConceptNode:center", 0.9)

        affected = ecan.spread_importance(space)
        assert affected > 0

    async def test_hebbian_link_strengthening(self):
        """Test Hebbian link strengthening between co-active atoms."""
        from components.oc_skintwin.opencog_adapter import ECANAttentionBank

        ecan = ECANAttentionBank()
        weight = ecan.strengthen_hebbian("a", "b", 0.2)
        assert weight == 0.2

        weight = ecan.strengthen_hebbian("a", "b", 0.3)
        assert weight == 0.5

    async def test_full_ecan_cycle(self):
        """Test running a complete ECAN cycle."""
        from components.oc_skintwin.opencog_adapter import (
            SimpleAtomSpace, Atom, AtomType, ECANAttentionBank
        )

        space = SimpleAtomSpace()
        space.add_atom(Atom(AtomType.CONCEPT, "alpha"))
        space.add_atom(Atom(AtomType.CONCEPT, "beta"))

        ecan = ECANAttentionBank()
        ecan.set_sti("ConceptNode:alpha", 0.8)
        ecan.set_sti("ConceptNode:beta", 0.6)

        result = ecan.run_cycle(space)
        assert result['cycle'] == 1
        assert result['focus_set_size'] >= 0
        assert 'spread_affected' in result

    async def test_ecan_via_adapter(self):
        """Test ECAN cycle through adapter process method."""
        from components.oc_skintwin.opencog_adapter import OpenCogAdapter

        adapter = OpenCogAdapter()
        await adapter.initialize({})

        result = adapter._ecan_cycle({
            'stimulate': {'ConceptNode:knowledge': 0.9}
        })
        assert result['status'] == 'success'
        assert result['ecan']['cycle'] == 1

    async def test_ecan_statistics(self):
        """Test ECAN statistics reporting."""
        from components.oc_skintwin.opencog_adapter import ECANAttentionBank

        ecan = ECANAttentionBank(focus_boundary=0.4, decay_rate=0.05)
        ecan.set_sti("x", 0.8)
        ecan.strengthen_hebbian("x", "y")

        stats = ecan.get_statistics()
        assert stats['tracked_atoms'] == 1
        assert stats['hebbian_links'] == 1
        assert stats['focus_boundary'] == 0.4


class TestDeepReasoningScenarios:
    """Tests for complex multi-step reasoning."""

    async def test_knowledge_chain_reasoning(self):
        """Test reasoning through a chain of knowledge."""
        from components.oc_skintwin.opencog_adapter import OpenCogAdapter

        adapter = OpenCogAdapter()
        await adapter.initialize({})

        # Build a chain: molecule → protein → function
        adapter._add_concept("molecule")
        adapter._add_concept("protein")
        adapter._add_concept("function")
        adapter._add_link(AtomType.INHERITANCE, "ConceptNode:molecule", "ConceptNode:protein")
        adapter._add_link(AtomType.INHERITANCE, "ConceptNode:protein", "ConceptNode:function")

        # Reason about the chain
        result = await adapter._cognitive_reasoning({
            'query': 'molecule',
            'context': {'domain': 'biology'}
        })
        assert result['status'] == 'success'
        assert result['matches'] > 0

    async def test_learning_and_query_cycle(self):
        """Test learning patterns then querying them."""
        from components.oc_skintwin.opencog_adapter import OpenCogAdapter

        adapter = OpenCogAdapter()
        await adapter.initialize({})

        # Learn patterns
        learn_result = await adapter._learn_patterns({
            'patterns': ['pattern_alpha', 'pattern_beta', 'pattern_gamma']
        })
        assert learn_result['patterns_learned'] == 3

        # Query for them
        query_result = await adapter._query_knowledge({
            'pattern': {'type': 'ConceptNode', 'name': 'pattern_'}
        })
        assert query_result['status'] == 'success'
        assert query_result['matches'] >= 3

    async def test_relevance_guided_reasoning(self):
        """Test reasoning guided by relevance realization."""
        from components.oc_skintwin.opencog_adapter import OpenCogAdapter

        adapter = OpenCogAdapter()
        await adapter.initialize({})

        # Add domain-specific knowledge
        adapter._add_concept("chemical_reaction")
        adapter._add_concept("enzyme")
        adapter._add_concept("catalyst")

        # Apply relevance
        result = await adapter.integrate_with_relevance_realization({
            'task_type': 'chemical',
            'requirements': ['reaction', 'catalyst']
        })
        assert result['status'] == 'success'
        assert result['attention_allocated'] > 0

        # Top atoms should be chemically relevant
        top = result['top_attended_atoms']
        assert len(top) > 0


# Import AtomType at module level for the test class
from components.oc_skintwin.opencog_adapter import AtomType

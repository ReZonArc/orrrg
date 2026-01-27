"""
OpenCog AtomSpace Component Adapter for ORRRG
==============================================

This adapter integrates OpenCog's AtomSpace cognitive architecture with ORRRG's
Self-Organizing Core, Evolution Engine, and Relevance Realization Ennead.

The adapter provides:
- AtomSpace interface for knowledge representation
- Cognitive reasoning pipelines
- Pattern matching and inference
- Integration with relevance realization
- Evolutionary learning capabilities
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AtomType(Enum):
    """Common AtomSpace atom types."""
    CONCEPT = "ConceptNode"
    PREDICATE = "PredicateNode"
    LINK = "Link"
    INHERITANCE = "InheritanceLink"
    SIMILARITY = "SimilarityLink"
    EVALUATION = "EvaluationLink"
    IMPLICATION = "ImplicationLink"
    AND = "AndLink"
    OR = "OrLink"
    NOT = "NotLink"


@dataclass
class Atom:
    """Represents an atom in the AtomSpace."""
    atom_type: AtomType
    name: str
    truth_value: Tuple[float, float] = (1.0, 1.0)  # (strength, confidence)
    attention_value: float = 0.0
    incoming: Set[str] = field(default_factory=set)
    outgoing: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert atom to dictionary."""
        return {
            'type': self.atom_type.value,
            'name': self.name,
            'truth_value': self.truth_value,
            'attention_value': self.attention_value,
            'incoming': list(self.incoming),
            'outgoing': list(self.outgoing)
        }


class SimpleAtomSpace:
    """
    Simplified AtomSpace implementation for ORRRG.
    
    This is a lightweight version that provides core AtomSpace functionality
    without requiring the full OpenCog installation.
    """
    
    def __init__(self):
        self.atoms: Dict[str, Atom] = {}
        self.type_index: Dict[AtomType, Set[str]] = {}
        self.name_index: Dict[str, Set[str]] = {}
        
    def add_atom(self, atom: Atom) -> str:
        """Add an atom to the AtomSpace."""
        atom_id = f"{atom.atom_type.value}:{atom.name}"
        
        if atom_id not in self.atoms:
            self.atoms[atom_id] = atom
            
            # Update indices
            if atom.atom_type not in self.type_index:
                self.type_index[atom.atom_type] = set()
            self.type_index[atom.atom_type].add(atom_id)
            
            if atom.name not in self.name_index:
                self.name_index[atom.name] = set()
            self.name_index[atom.name].add(atom_id)
            
        return atom_id
    
    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Get an atom by ID."""
        return self.atoms.get(atom_id)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a specific type."""
        atom_ids = self.type_index.get(atom_type, set())
        return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    def get_atoms_by_name(self, name: str) -> List[Atom]:
        """Get all atoms with a specific name."""
        atom_ids = self.name_index.get(name, set())
        return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    def create_link(self, link_type: AtomType, source_id: str, target_id: str) -> str:
        """Create a link between two atoms."""
        link_name = f"{source_id}->{target_id}"
        link = Atom(link_type, link_name)
        link.outgoing.add(target_id)
        
        link_id = self.add_atom(link)
        
        # Update incoming/outgoing sets
        if source_id in self.atoms:
            self.atoms[source_id].outgoing.add(link_id)
        if target_id in self.atoms:
            self.atoms[target_id].incoming.add(link_id)
        
        return link_id
    
    def pattern_match(self, pattern: Dict[str, Any]) -> List[Dict[str, Atom]]:
        """Simple pattern matching in the AtomSpace."""
        results = []
        
        # Simple implementation: match by type and name patterns
        atom_type = pattern.get('type')
        name_pattern = pattern.get('name')
        
        candidates = []
        if atom_type:
            candidates = self.get_atoms_by_type(AtomType(atom_type))
        else:
            candidates = list(self.atoms.values())
        
        for atom in candidates:
            if name_pattern:
                if name_pattern in atom.name:
                    results.append({'atom': atom})
            else:
                results.append({'atom': atom})
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get AtomSpace statistics."""
        return {
            'total_atoms': len(self.atoms),
            'atoms_by_type': {
                atom_type.value: len(atom_ids)
                for atom_type, atom_ids in self.type_index.items()
            }
        }


class OpenCogAdapter:
    """
    Adapter for integrating OpenCog AtomSpace with ORRRG.
    
    Provides AGI reasoning capabilities including:
    - Knowledge representation in AtomSpace
    - Pattern matching and inference
    - Relevance-guided attention allocation
    - Evolutionary learning
    """
    
    def __init__(self):
        self.atomspace = SimpleAtomSpace()
        self.initialized = False
        self.reasoning_cycles = 0
        self.learned_patterns: List[Dict[str, Any]] = []
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the OpenCog adapter."""
        logger.info("Initializing OpenCog AtomSpace adapter")
        
        self.config = config
        
        # Initialize core cognitive structures
        await self._initialize_core_structures()
        
        self.initialized = True
        logger.info("OpenCog adapter initialized successfully")
        return True
    
    async def _initialize_core_structures(self):
        """Initialize core cognitive structures in AtomSpace."""
        # Add basic ontology
        self._add_concept("knowledge")
        self._add_concept("understanding")
        self._add_concept("reasoning")
        
        # Add reasoning predicates
        self._add_predicate("is_related_to")
        self._add_predicate("implies")
        self._add_predicate("causes")
        
        logger.info("Core cognitive structures initialized")
    
    def _add_concept(self, name: str) -> str:
        """Add a concept node to AtomSpace."""
        atom = Atom(AtomType.CONCEPT, name)
        return self.atomspace.add_atom(atom)
    
    def _add_predicate(self, name: str) -> str:
        """Add a predicate node to AtomSpace."""
        atom = Atom(AtomType.PREDICATE, name)
        return self.atomspace.add_atom(atom)
    
    def _add_link(self, link_type: AtomType, source: str, target: str) -> str:
        """Add a link between concepts."""
        return self.atomspace.create_link(link_type, source, target)
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through cognitive reasoning."""
        if not self.initialized:
            raise RuntimeError("OpenCog adapter not initialized")
        
        operation = data.get('operation', 'reason')
        
        if operation == 'reason':
            return await self._cognitive_reasoning(data)
        elif operation == 'learn':
            return await self._learn_patterns(data)
        elif operation == 'query':
            return await self._query_knowledge(data)
        elif operation == 'infer':
            return await self._inference(data)
        else:
            logger.warning(f"Unknown operation: {operation}")
            return {'status': 'error', 'message': f'Unknown operation: {operation}'}
    
    async def _cognitive_reasoning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cognitive reasoning on input data."""
        self.reasoning_cycles += 1
        
        query = data.get('query', '')
        context = data.get('context', {})
        
        logger.info(f"Cognitive reasoning cycle {self.reasoning_cycles}")
        
        # Add query to AtomSpace
        query_concept = self._add_concept(f"query_{self.reasoning_cycles}_{query}")
        
        # Pattern match against existing knowledge
        pattern = {'type': AtomType.CONCEPT.value, 'name': query}
        matches = self.atomspace.pattern_match(pattern)
        
        # Perform inference
        inferred_knowledge = await self._inference({
            'premises': matches,
            'context': context
        })
        
        return {
            'status': 'success',
            'reasoning_cycle': self.reasoning_cycles,
            'query': query,
            'matches': len(matches),
            'inferences': inferred_knowledge,
            'atomspace_stats': self.atomspace.get_statistics()
        }
    
    async def _learn_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn patterns from data."""
        patterns = data.get('patterns', [])
        
        learned = []
        for pattern in patterns:
            # Add pattern to AtomSpace
            concept_id = self._add_concept(f"pattern_{len(self.learned_patterns)}")
            
            # Store pattern
            self.learned_patterns.append({
                'concept_id': concept_id,
                'pattern': pattern,
                'learned_at': self.reasoning_cycles
            })
            
            learned.append(concept_id)
        
        logger.info(f"Learned {len(learned)} new patterns")
        
        return {
            'status': 'success',
            'patterns_learned': len(learned),
            'total_learned_patterns': len(self.learned_patterns)
        }
    
    async def _query_knowledge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge base."""
        pattern = data.get('pattern', {})
        
        matches = self.atomspace.pattern_match(pattern)
        
        return {
            'status': 'success',
            'matches': len(matches),
            'results': [
                {
                    'atom': match['atom'].to_dict()
                }
                for match in matches[:10]  # Limit to 10 results
            ]
        }
    
    async def _inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform logical inference."""
        premises = data.get('premises', [])
        
        # Simple inference: if we have related concepts, infer connections
        inferences = []
        
        for i, premise in enumerate(premises):
            atom = premise.get('atom')
            if atom and hasattr(atom, 'name'):
                # Infer related concepts
                related = self.atomspace.get_atoms_by_type(AtomType.CONCEPT)
                
                for related_atom in related[:3]:  # Limit inferences
                    if related_atom.name != atom.name:
                        inference = {
                            'type': 'similarity',
                            'source': atom.name,
                            'target': related_atom.name,
                            'confidence': 0.7  # Simplified confidence
                        }
                        inferences.append(inference)
        
        return {
            'status': 'success',
            'inferences': inferences[:5]  # Limit to 5
        }
    
    async def integrate_with_relevance_realization(
        self,
        relevance_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate cognitive reasoning with relevance realization.
        
        Uses Ennead framework to guide attention allocation in AtomSpace.
        """
        # Extract relevance dimensions
        task_type = relevance_context.get('task_type', '')
        requirements = relevance_context.get('requirements', [])
        
        # Allocate attention based on relevance
        attention_allocation = {}
        
        for atom_type, atom_ids in self.atomspace.type_index.items():
            for atom_id in atom_ids:
                atom = self.atomspace.get_atom(atom_id)
                if atom:
                    # Calculate relevance-based attention
                    attention = self._calculate_relevance_attention(
                        atom, task_type, requirements
                    )
                    atom.attention_value = attention
                    attention_allocation[atom_id] = attention
        
        # Get highly attended atoms
        top_atoms = sorted(
            attention_allocation.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'status': 'success',
            'attention_allocated': len(attention_allocation),
            'top_attended_atoms': [
                {
                    'atom_id': aid,
                    'attention': att,
                    'atom': self.atomspace.get_atom(aid).to_dict()
                }
                for aid, att in top_atoms
            ]
        }
    
    def _calculate_relevance_attention(
        self,
        atom: Atom,
        task_type: str,
        requirements: List[str]
    ) -> float:
        """Calculate attention value based on relevance."""
        attention = 0.0
        
        # Base attention from truth value
        attention += atom.truth_value[0] * atom.truth_value[1]
        
        # Boost attention if atom name matches requirements
        for req in requirements:
            if req.lower() in atom.name.lower():
                attention += 0.5
        
        # Boost attention if atom type is relevant to task
        if task_type and task_type.lower() in atom.name.lower():
            attention += 0.3
        
        # Consider graph connectivity (more connections = more relevant)
        connectivity = len(atom.incoming) + len(atom.outgoing)
        attention += min(connectivity * 0.1, 0.5)
        
        return min(attention, 1.0)  # Cap at 1.0
    
    async def evolve_reasoning_patterns(
        self,
        evolution_objectives: List[str]
    ) -> Dict[str, Any]:
        """Evolve reasoning patterns based on objectives."""
        evolved_patterns = []
        
        for objective in evolution_objectives:
            if objective == 'efficiency':
                # Prune low-attention atoms
                pruned = self._prune_low_attention_atoms()
                evolved_patterns.append({
                    'objective': 'efficiency',
                    'action': 'pruning',
                    'atoms_pruned': pruned
                })
            
            elif objective == 'connectivity':
                # Strengthen highly connected patterns
                strengthened = self._strengthen_connected_atoms()
                evolved_patterns.append({
                    'objective': 'connectivity',
                    'action': 'strengthening',
                    'atoms_strengthened': strengthened
                })
            
            elif objective == 'diversity':
                # Discover new patterns
                new_patterns = await self._discover_novel_patterns()
                evolved_patterns.append({
                    'objective': 'diversity',
                    'action': 'discovery',
                    'patterns_discovered': len(new_patterns)
                })
        
        return {
            'status': 'success',
            'evolutions': evolved_patterns,
            'atomspace_stats': self.atomspace.get_statistics()
        }
    
    def _prune_low_attention_atoms(self) -> int:
        """Prune atoms with low attention values."""
        threshold = 0.1
        to_prune = []
        
        for atom_id, atom in self.atomspace.atoms.items():
            if atom.attention_value < threshold:
                to_prune.append(atom_id)
        
        # Prune (simplified - would need more careful deletion)
        for atom_id in to_prune[:10]:  # Limit pruning
            if atom_id in self.atomspace.atoms:
                del self.atomspace.atoms[atom_id]
        
        return len(to_prune[:10])
    
    def _strengthen_connected_atoms(self) -> int:
        """Strengthen truth values of highly connected atoms."""
        strengthened = 0
        
        for atom_id, atom in self.atomspace.atoms.items():
            connectivity = len(atom.incoming) + len(atom.outgoing)
            if connectivity > 2:
                # Increase truth value
                strength, confidence = atom.truth_value
                atom.truth_value = (min(strength * 1.1, 1.0), min(confidence * 1.05, 1.0))
                strengthened += 1
        
        return strengthened
    
    async def _discover_novel_patterns(self) -> List[Dict[str, Any]]:
        """Discover novel reasoning patterns."""
        # Simple pattern discovery: find highly connected subgraphs
        patterns = []
        
        concepts = self.atomspace.get_atoms_by_type(AtomType.CONCEPT)
        
        for concept in concepts:
            if len(concept.outgoing) > 2:
                pattern = {
                    'center': concept.name,
                    'connections': len(concept.outgoing),
                    'type': 'hub_pattern'
                }
                patterns.append(pattern)
        
        return patterns[:5]  # Return top 5
    
    async def cleanup(self) -> None:
        """Cleanup adapter resources."""
        logger.info("Cleaning up OpenCog adapter")
        self.atomspace = SimpleAtomSpace()
        self.initialized = False
    
    def get_capabilities(self) -> List[str]:
        """Return adapter capabilities."""
        return [
            "cognitive_reasoning",
            "knowledge_representation",
            "pattern_matching",
            "logical_inference",
            "attention_allocation",
            "evolutionary_learning",
            "agi_reasoning"
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            'initialized': self.initialized,
            'reasoning_cycles': self.reasoning_cycles,
            'learned_patterns': len(self.learned_patterns),
            'atomspace': self.atomspace.get_statistics()
        }


# Export main adapter class
__all__ = ['OpenCogAdapter', 'AtomType', 'Atom', 'SimpleAtomSpace']

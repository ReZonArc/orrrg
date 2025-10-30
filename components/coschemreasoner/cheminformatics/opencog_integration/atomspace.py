"""
AtomSpace-Inspired Hypergraph Representation for Cosmeceutical Formulation

This module provides a simplified AtomSpace-like representation for cosmeceutical
ingredients and their relationships, enabling hypergraph-based reasoning and
constraint satisfaction.

Features:
- Hypergraph representation of ingredients and relationships
- OpenCog-style atom types and links
- Knowledge base population and querying
- Constraint propagation and satisfaction
"""

from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict
import networkx as nx
import numpy as np

from ..types.cosmetic_atoms import CosmeticAtom, AtomProperties


class AtomType(Enum):
    """OpenCog-inspired atom types for cosmeceutical formulation"""
    # Node types
    INGREDIENT_NODE = "IngredientNode"
    FORMULATION_NODE = "FormulationNode"
    PROPERTY_NODE = "PropertyNode"
    CONSTRAINT_NODE = "ConstraintNode"
    MULTISCALE_NODE = "MultiscaleNode"
    
    # Link types
    CONTAINS_LINK = "ContainsLink"
    COMPATIBILITY_LINK = "CompatibilityLink"
    INCOMPATIBILITY_LINK = "IncompatibilityLink"
    SYNERGY_LINK = "SynergyLink"
    ANTAGONISM_LINK = "AntagonismLink"
    CONCENTRATION_LINK = "ConcentrationLink"
    PH_RANGE_LINK = "PHRangeLink"
    MULTISCALE_LINK = "MultiscaleLink"


@dataclass
class Atom:
    """OpenCog-inspired atom for cosmeceutical representation"""
    atom_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    atom_type: AtomType = AtomType.INGREDIENT_NODE
    name: str = ""
    truth_value: float = 1.0  # Simple truth value (0-1)
    attention_value: float = 0.0  # ECAN-inspired attention
    properties: Dict[str, Any] = field(default_factory=dict)
    outgoing: List['Atom'] = field(default_factory=list)  # For links
    
    def __hash__(self):
        return hash(self.atom_id)
    
    def __eq__(self, other):
        return isinstance(other, Atom) and self.atom_id == other.atom_id


@dataclass
class MultiscaleLevel:
    """Represents different scales in skin model"""
    level_name: str
    scale_range: Tuple[float, float]  # in micrometers
    relevant_properties: List[str]
    constraints: List[str]


class CosmeceuticalAtomSpace:
    """
    OpenCog AtomSpace-inspired knowledge representation for cosmeceutical formulation.
    
    This class provides hypergraph-based representation and reasoning capabilities
    for complex cosmeceutical formulation problems, supporting multiscale constraints
    and ingredient interactions.
    """
    
    def __init__(self):
        self.atoms: Dict[str, Atom] = {}
        self.atoms_by_type: Dict[AtomType, Set[str]] = defaultdict(set)
        self.atoms_by_name: Dict[str, str] = {}  # name -> atom_id
        self.incoming: Dict[str, Set[str]] = defaultdict(set)  # atom_id -> incoming links
        self.hypergraph = nx.MultiDiGraph()
        
        # Multiscale skin model levels
        self.multiscale_levels = {
            "molecular": MultiscaleLevel(
                "molecular", (0.001, 0.01), 
                ["molecular_weight", "lipophilicity", "hydrogen_bonding"],
                ["size_exclusion", "chemical_compatibility"]
            ),
            "cellular": MultiscaleLevel(
                "cellular", (0.01, 100), 
                ["permeability", "cytotoxicity", "cellular_uptake"],
                ["membrane_compatibility", "cellular_integrity"]
            ),
            "tissue": MultiscaleLevel(
                "tissue", (100, 1000), 
                ["penetration_depth", "diffusion_rate", "barrier_function"],
                ["tissue_compatibility", "inflammation_response"]
            ),
            "organ": MultiscaleLevel(
                "organ", (1000, 10000), 
                ["systemic_absorption", "metabolism", "clearance"],
                ["organ_safety", "systemic_effects"]
            )
        }
        
        # Initialize knowledge base
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with basic cosmeceutical knowledge"""
        # Create multiscale level nodes
        for level_name, level in self.multiscale_levels.items():
            multiscale_node = self.create_atom(
                AtomType.MULTISCALE_NODE,
                level_name,
                properties={
                    "scale_range": level.scale_range,
                    "relevant_properties": level.relevant_properties,
                    "constraints": level.constraints
                }
            )
    
    def create_atom(self, atom_type: AtomType, name: str = "", 
                   truth_value: float = 1.0, attention_value: float = 0.0,
                   properties: Optional[Dict[str, Any]] = None,
                   outgoing: Optional[List[Atom]] = None) -> Atom:
        """Create a new atom in the AtomSpace"""
        atom = Atom(
            atom_type=atom_type,
            name=name,
            truth_value=truth_value,
            attention_value=attention_value,
            properties=properties or {},
            outgoing=outgoing or []
        )
        
        self.atoms[atom.atom_id] = atom
        self.atoms_by_type[atom_type].add(atom.atom_id)
        
        if name:
            self.atoms_by_name[name] = atom.atom_id
        
        # Update hypergraph
        self.hypergraph.add_node(atom.atom_id, **atom.properties)
        
        # Handle outgoing links
        for target_atom in atom.outgoing:
            self.incoming[target_atom.atom_id].add(atom.atom_id)
            self.hypergraph.add_edge(atom.atom_id, target_atom.atom_id)
        
        return atom
    
    def create_ingredient_atom(self, cosmetic_atom: CosmeticAtom, 
                             attention_value: float = 0.0) -> Atom:
        """Create an ingredient atom from a cosmetic atom"""
        properties = {
            "category": cosmetic_atom.__class__.__name__,
            "name": cosmetic_atom.name
        }
        
        if hasattr(cosmetic_atom, 'properties') and cosmetic_atom.properties:
            properties.update({
                "ph_range": cosmetic_atom.properties.ph_range,
                "max_concentration": cosmetic_atom.properties.max_concentration,
                "cost_per_kg": cosmetic_atom.properties.cost_per_kg,
                "allergen_status": cosmetic_atom.properties.allergen_status
            })
        
        return self.create_atom(
            AtomType.INGREDIENT_NODE,
            cosmetic_atom.name,
            attention_value=attention_value,
            properties=properties
        )
    
    def create_compatibility_link(self, ingredient1: Atom, ingredient2: Atom,
                                truth_value: float = 1.0) -> Atom:
        """Create a compatibility link between two ingredients"""
        return self.create_atom(
            AtomType.COMPATIBILITY_LINK,
            f"compatible_{ingredient1.name}_{ingredient2.name}",
            truth_value=truth_value,
            outgoing=[ingredient1, ingredient2]
        )
    
    def create_synergy_link(self, ingredient1: Atom, ingredient2: Atom,
                          synergy_strength: float = 1.0) -> Atom:
        """Create a synergy link between two ingredients"""
        return self.create_atom(
            AtomType.SYNERGY_LINK,
            f"synergy_{ingredient1.name}_{ingredient2.name}",
            truth_value=synergy_strength,
            outgoing=[ingredient1, ingredient2],
            properties={"synergy_strength": synergy_strength}
        )
    
    def create_multiscale_constraint(self, ingredient: Atom, scale_level: str,
                                   constraint_type: str, constraint_value: Any) -> Atom:
        """Create a multiscale constraint for an ingredient"""
        constraint_name = f"{scale_level}_{constraint_type}_{ingredient.name}"
        
        constraint_atom = self.create_atom(
            AtomType.CONSTRAINT_NODE,
            constraint_name,
            properties={
                "scale_level": scale_level,
                "constraint_type": constraint_type,
                "constraint_value": constraint_value
            }
        )
        
        # Link to multiscale level and ingredient
        scale_atom = self.get_atom_by_name(scale_level)
        if scale_atom:
            self.create_atom(
                AtomType.MULTISCALE_LINK,
                f"multiscale_{constraint_name}",
                outgoing=[constraint_atom, ingredient, scale_atom]
            )
        
        return constraint_atom
    
    def get_atom_by_name(self, name: str) -> Optional[Atom]:
        """Get atom by name"""
        atom_id = self.atoms_by_name.get(name)
        return self.atoms.get(atom_id) if atom_id else None
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a specific type"""
        atom_ids = self.atoms_by_type.get(atom_type, set())
        return [self.atoms[atom_id] for atom_id in atom_ids]
    
    def get_ingredient_compatibility(self, ingredient1_name: str, 
                                   ingredient2_name: str) -> Optional[float]:
        """Get compatibility truth value between two ingredients"""
        compatibility_links = self.get_atoms_by_type(AtomType.COMPATIBILITY_LINK)
        
        for link in compatibility_links:
            if len(link.outgoing) == 2:
                names = {atom.name for atom in link.outgoing}
                if names == {ingredient1_name, ingredient2_name}:
                    return link.truth_value
        
        return None
    
    def get_ingredient_synergies(self, ingredient_name: str) -> List[Tuple[str, float]]:
        """Get all synergistic relationships for an ingredient"""
        synergy_links = self.get_atoms_by_type(AtomType.SYNERGY_LINK)
        synergies = []
        
        for link in synergy_links:
            if len(link.outgoing) == 2:
                if link.outgoing[0].name == ingredient_name:
                    synergies.append((link.outgoing[1].name, link.truth_value))
                elif link.outgoing[1].name == ingredient_name:
                    synergies.append((link.outgoing[0].name, link.truth_value))
        
        return synergies
    
    def get_multiscale_constraints(self, ingredient_name: str, 
                                 scale_level: Optional[str] = None) -> List[Atom]:
        """Get multiscale constraints for an ingredient"""
        constraints = []
        multiscale_links = self.get_atoms_by_type(AtomType.MULTISCALE_LINK)
        
        for link in multiscale_links:
            if len(link.outgoing) >= 2:
                constraint_atom = link.outgoing[0]
                ingredient_atom = link.outgoing[1]
                
                if (ingredient_atom.name == ingredient_name and
                    constraint_atom.atom_type == AtomType.CONSTRAINT_NODE):
                    
                    if scale_level is None or constraint_atom.properties.get("scale_level") == scale_level:
                        constraints.append(constraint_atom)
        
        return constraints
    
    def evaluate_formulation_constraints(self, formulation_atoms: List[Atom]) -> Dict[str, float]:
        """Evaluate constraints for a formulation across multiple scales"""
        constraint_scores = {}
        
        for scale_level in self.multiscale_levels:
            scale_score = 0.0
            constraint_count = 0
            
            for ingredient_atom in formulation_atoms:
                constraints = self.get_multiscale_constraints(ingredient_atom.name, scale_level)
                
                for constraint in constraints:
                    # Simple constraint evaluation - can be extended
                    constraint_score = self._evaluate_constraint(constraint, ingredient_atom)
                    scale_score += constraint_score
                    constraint_count += 1
            
            if constraint_count > 0:
                constraint_scores[scale_level] = scale_score / constraint_count
            else:
                constraint_scores[scale_level] = 1.0
        
        return constraint_scores
    
    def _evaluate_constraint(self, constraint: Atom, ingredient: Atom) -> float:
        """Evaluate a single constraint - placeholder for more complex logic"""
        # This is a simplified evaluation - can be extended with more sophisticated
        # constraint satisfaction algorithms
        constraint_type = constraint.properties.get("constraint_type", "")
        constraint_value = constraint.properties.get("constraint_value", None)
        
        if constraint_type == "max_concentration":
            current_concentration = ingredient.properties.get("concentration", 0.0)
            if current_concentration <= constraint_value:
                return 1.0
            else:
                return max(0.0, 1.0 - (current_concentration - constraint_value) / constraint_value)
        
        # Default neutral score
        return 0.5
    
    def propagate_attention(self, initial_atoms: List[Atom], decay_factor: float = 0.9):
        """Propagate attention values through the hypergraph (ECAN-inspired)"""
        # Simple attention propagation algorithm
        for atom in initial_atoms:
            self._propagate_attention_recursive(atom, atom.attention_value, decay_factor, set())
    
    def _propagate_attention_recursive(self, atom: Atom, attention: float, 
                                     decay_factor: float, visited: Set[str]):
        """Recursively propagate attention through connected atoms"""
        if atom.atom_id in visited or attention < 0.01:
            return
        
        visited.add(atom.atom_id)
        atom.attention_value = max(atom.attention_value, attention)
        
        # Propagate to connected atoms
        for connected_id in self.incoming[atom.atom_id]:
            connected_atom = self.atoms[connected_id]
            self._propagate_attention_recursive(
                connected_atom, attention * decay_factor, decay_factor, visited
            )
        
        for outgoing_atom in atom.outgoing:
            self._propagate_attention_recursive(
                outgoing_atom, attention * decay_factor, decay_factor, visited
            )
    
    def get_high_attention_atoms(self, threshold: float = 0.1) -> List[Atom]:
        """Get atoms with attention values above threshold"""
        return [atom for atom in self.atoms.values() 
                if atom.attention_value >= threshold]
    
    def export_hypergraph(self) -> nx.MultiDiGraph:
        """Export the internal hypergraph representation"""
        return self.hypergraph.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the AtomSpace"""
        return {
            "total_atoms": len(self.atoms),
            "atoms_by_type": {atom_type.value: len(atom_ids) 
                            for atom_type, atom_ids in self.atoms_by_type.items()},
            "multiscale_levels": list(self.multiscale_levels.keys()),
            "hypergraph_nodes": self.hypergraph.number_of_nodes(),
            "hypergraph_edges": self.hypergraph.number_of_edges()
        }
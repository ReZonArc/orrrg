"""
Adaptive Attention Allocation for Cosmeceutical Formulation

This module implements ECAN-inspired (Economic Attention Networks) adaptive attention
allocation mechanisms for focusing computational resources on promising ingredient
combinations and formulation subspaces.

Features:
- ECAN-inspired attention dynamics
- Importance-based resource allocation
- Dynamic attention spreading
- Attention-guided search space pruning
- Economic models for computational resource management
"""

from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import heapq
from collections import defaultdict, deque
import math

from .atomspace import CosmeceuticalAtomSpace, Atom, AtomType


class AttentionType(Enum):
    """Types of attention values (ECAN-inspired)"""
    SHORT_TERM = "short_term"  # STI - Short Term Importance
    LONG_TERM = "long_term"    # LTI - Long Term Importance
    VERY_LONG_TERM = "very_long_term"  # VLTI - Very Long Term Importance


@dataclass
class AttentionValue:
    """ECAN-inspired attention value with multiple temporal scales"""
    short_term: float = 0.0     # STI - current relevance
    long_term: float = 0.0      # LTI - historical relevance  
    very_long_term: float = 0.0 # VLTI - structural importance
    confidence: float = 1.0     # confidence in attention values
    
    @property
    def total_attention(self) -> float:
        """Calculate total attention as weighted sum"""
        return (0.5 * self.short_term + 
                0.3 * self.long_term + 
                0.2 * self.very_long_term)
    
    def decay(self, sti_decay: float = 0.95, lti_decay: float = 0.99, vlti_decay: float = 0.999):
        """Apply temporal decay to attention values"""
        self.short_term *= sti_decay
        self.long_term *= lti_decay
        self.very_long_term *= vlti_decay


@dataclass
class AttentionBank:
    """ECAN-inspired attention bank for resource management"""
    total_sti: float = 1000.0
    total_lti: float = 1000.0
    sti_funds: float = 1000.0
    lti_funds: float = 1000.0
    
    def can_allocate_sti(self, amount: float) -> bool:
        """Check if STI can be allocated"""
        return self.sti_funds >= amount
    
    def can_allocate_lti(self, amount: float) -> bool:
        """Check if LTI can be allocated"""
        return self.lti_funds >= amount
    
    def allocate_sti(self, amount: float) -> bool:
        """Allocate STI from bank"""
        if self.can_allocate_sti(amount):
            self.sti_funds -= amount
            return True
        return False
    
    def allocate_lti(self, amount: float) -> bool:
        """Allocate LTI from bank"""
        if self.can_allocate_lti(amount):
            self.lti_funds -= amount
            return True
        return False
    
    def collect_sti(self, amount: float):
        """Return STI to bank"""
        self.sti_funds = min(self.total_sti, self.sti_funds + amount)
    
    def collect_lti(self, amount: float):
        """Return LTI to bank"""
        self.lti_funds = min(self.total_lti, self.lti_funds + amount)


class AdaptiveAttentionAllocator:
    """
    ECAN-inspired adaptive attention allocation system for cosmeceutical formulation.
    
    This class manages attention allocation across ingredient combinations,
    formulation subspaces, and optimization trajectories, enabling efficient
    resource allocation in large search spaces.
    """
    
    def __init__(self, atomspace: CosmeceuticalAtomSpace, 
                 initial_sti_total: float = 1000.0,
                 initial_lti_total: float = 1000.0):
        self.atomspace = atomspace
        self.attention_bank = AttentionBank(initial_sti_total, initial_lti_total)
        self.attention_values: Dict[str, AttentionValue] = {}
        
        # Attention dynamics parameters
        self.sti_decay_rate = 0.95
        self.lti_decay_rate = 0.99
        self.vlti_decay_rate = 0.999
        self.spreading_factor = 0.7
        self.importance_threshold = 0.1
        
        # Attention allocation strategies
        self.allocation_strategies: Dict[str, Callable] = {
            "novelty_based": self._novelty_based_allocation,
            "importance_based": self._importance_based_allocation,
            "synergy_based": self._synergy_based_allocation,
            "constraint_based": self._constraint_based_allocation
        }
        
        # Initialize attention values for all atoms
        self._initialize_attention_values()
    
    def _initialize_attention_values(self):
        """Initialize attention values for all atoms in AtomSpace"""
        for atom_id, atom in self.atomspace.atoms.items():
            self.attention_values[atom_id] = AttentionValue(
                short_term=1.0,  # Base STI
                long_term=0.0,   # Base LTI
                very_long_term=self._calculate_structural_importance(atom)
            )
    
    def _calculate_structural_importance(self, atom: Atom) -> float:
        """Calculate structural importance (VLTI) based on atom's role in hypergraph"""
        # Base importance by atom type
        type_importance = {
            AtomType.INGREDIENT_NODE: 0.8,
            AtomType.FORMULATION_NODE: 0.9,
            AtomType.COMPATIBILITY_LINK: 0.7,
            AtomType.SYNERGY_LINK: 0.8,
            AtomType.CONSTRAINT_NODE: 0.6,
            AtomType.MULTISCALE_NODE: 0.9
        }
        
        base_importance = type_importance.get(atom.atom_type, 0.5)
        
        # Adjust based on connectivity (centrality)
        incoming_count = len(self.atomspace.incoming[atom.atom_id])
        outgoing_count = len(atom.outgoing)
        connectivity_bonus = min(0.3, 0.05 * (incoming_count + outgoing_count))
        
        return min(1.0, base_importance + connectivity_bonus)
    
    def _novelty_based_allocation(self, atoms: List[Atom], budget: float) -> Dict[str, float]:
        """Allocate attention based on novelty of ingredient combinations"""
        allocation = {}
        novelty_scores = {}
        
        for atom in atoms:
            # Calculate novelty score based on how rarely this combination appears
            if atom.atom_type == AtomType.SYNERGY_LINK:
                # For synergy links, novelty is inverse of frequency
                similar_synergies = len([
                    a for a in self.atomspace.get_atoms_by_type(AtomType.SYNERGY_LINK)
                    if len(set(ingredient.name for ingredient in a.outgoing) & 
                          set(ingredient.name for ingredient in atom.outgoing)) > 0
                ])
                novelty_scores[atom.atom_id] = 1.0 / (1.0 + similar_synergies)
            else:
                # For other atoms, use structural uniqueness
                novelty_scores[atom.atom_id] = 1.0 / (1.0 + len(atom.outgoing))
        
        # Normalize and allocate
        total_novelty = sum(novelty_scores.values())
        if total_novelty > 0:
            for atom_id, novelty in novelty_scores.items():
                allocation[atom_id] = (novelty / total_novelty) * budget
        
        return allocation
    
    def _importance_based_allocation(self, atoms: List[Atom], budget: float) -> Dict[str, float]:
        """Allocate attention based on structural importance"""
        allocation = {}
        importance_scores = {}
        
        for atom in atoms:
            current_attention = self.attention_values.get(atom.atom_id, AttentionValue())
            importance_scores[atom.atom_id] = current_attention.very_long_term
        
        # Normalize and allocate
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            for atom_id, importance in importance_scores.items():
                allocation[atom_id] = (importance / total_importance) * budget
        
        return allocation
    
    def _synergy_based_allocation(self, atoms: List[Atom], budget: float) -> Dict[str, float]:
        """Allocate attention based on synergistic potential"""
        allocation = {}
        synergy_scores = {}
        
        for atom in atoms:
            if atom.atom_type == AtomType.INGREDIENT_NODE:
                # Calculate synergy potential based on connected synergy links
                synergies = self.atomspace.get_ingredient_synergies(atom.name)
                synergy_strength = sum(strength for _, strength in synergies)
                synergy_scores[atom.atom_id] = synergy_strength
            elif atom.atom_type == AtomType.SYNERGY_LINK:
                # Direct synergy links get attention based on their truth value
                synergy_scores[atom.atom_id] = atom.truth_value
            else:
                synergy_scores[atom.atom_id] = 0.0
        
        # Normalize and allocate
        total_synergy = sum(synergy_scores.values())
        if total_synergy > 0:
            for atom_id, synergy in synergy_scores.items():
                allocation[atom_id] = (synergy / total_synergy) * budget
        
        return allocation
    
    def _constraint_based_allocation(self, atoms: List[Atom], budget: float) -> Dict[str, float]:
        """Allocate attention based on constraint satisfaction needs"""
        allocation = {}
        constraint_scores = {}
        
        for atom in atoms:
            if atom.atom_type == AtomType.INGREDIENT_NODE:
                # Check how many constraints this ingredient participates in
                constraints = self.atomspace.get_multiscale_constraints(atom.name)
                constraint_count = len(constraints)
                
                # Higher attention for ingredients with more constraints
                constraint_scores[atom.atom_id] = constraint_count
            elif atom.atom_type == AtomType.CONSTRAINT_NODE:
                # Constraints themselves get attention based on violation severity
                constraint_scores[atom.atom_id] = 1.0 - atom.truth_value  # Higher for violated constraints
            else:
                constraint_scores[atom.atom_id] = 0.0
        
        # Normalize and allocate
        total_constraints = sum(constraint_scores.values())
        if total_constraints > 0:
            for atom_id, constraint_score in constraint_scores.items():
                allocation[atom_id] = (constraint_score / total_constraints) * budget
        
        return allocation
    
    def allocate_attention(self, target_atoms: List[Atom], 
                         strategy: str = "importance_based",
                         sti_budget: float = 100.0,
                         lti_budget: float = 50.0) -> Dict[str, Tuple[float, float]]:
        """
        Allocate attention to target atoms using specified strategy.
        
        Returns dict mapping atom_id to (sti_allocated, lti_allocated)
        """
        if strategy not in self.allocation_strategies:
            raise ValueError(f"Unknown allocation strategy: {strategy}")
        
        # Check if we have sufficient funds
        if not (self.attention_bank.can_allocate_sti(sti_budget) and
                self.attention_bank.can_allocate_lti(lti_budget)):
            # Try to collect some attention from low-importance atoms
            self._garbage_collect_attention()
        
        # Get allocation from strategy
        sti_allocation = self.allocation_strategies[strategy](target_atoms, sti_budget)
        lti_allocation = self.allocation_strategies[strategy](target_atoms, lti_budget)
        
        # Apply allocations
        actual_allocations = {}
        
        for atom in target_atoms:
            atom_id = atom.atom_id
            sti_amount = sti_allocation.get(atom_id, 0.0)
            lti_amount = lti_allocation.get(atom_id, 0.0)
            
            # Try to allocate from bank
            sti_allocated = 0.0
            lti_allocated = 0.0
            
            if self.attention_bank.allocate_sti(sti_amount):
                sti_allocated = sti_amount
                if atom_id not in self.attention_values:
                    self.attention_values[atom_id] = AttentionValue()
                self.attention_values[atom_id].short_term += sti_allocated
            
            if self.attention_bank.allocate_lti(lti_amount):
                lti_allocated = lti_amount
                if atom_id not in self.attention_values:
                    self.attention_values[atom_id] = AttentionValue()
                self.attention_values[atom_id].long_term += lti_allocated
            
            actual_allocations[atom_id] = (sti_allocated, lti_allocated)
        
        return actual_allocations
    
    def spread_attention(self, source_atoms: List[Atom], 
                        max_spread_distance: int = 3,
                        spreading_factor: float = 0.7):
        """Spread attention from source atoms to connected atoms"""
        spread_queue = deque([(atom, 0, self.attention_values[atom.atom_id].short_term) 
                             for atom in source_atoms if atom.atom_id in self.attention_values])
        visited = set()
        
        while spread_queue:
            atom, distance, attention_amount = spread_queue.popleft()
            
            if (atom.atom_id in visited or 
                distance >= max_spread_distance or 
                attention_amount < 0.01):
                continue
            
            visited.add(atom.atom_id)
            
            # Spread to connected atoms
            connected_atoms = []
            
            # Add outgoing connections
            connected_atoms.extend(atom.outgoing)
            
            # Add incoming connections
            for incoming_id in self.atomspace.incoming[atom.atom_id]:
                if incoming_id in self.atomspace.atoms:
                    connected_atoms.append(self.atomspace.atoms[incoming_id])
            
            # Spread attention to connected atoms
            spread_amount = attention_amount * spreading_factor
            spread_per_atom = spread_amount / max(1, len(connected_atoms))
            
            for connected_atom in connected_atoms:
                if connected_atom.atom_id not in visited:
                    # Add attention to connected atom
                    if connected_atom.atom_id not in self.attention_values:
                        self.attention_values[connected_atom.atom_id] = AttentionValue()
                    
                    self.attention_values[connected_atom.atom_id].short_term += spread_per_atom
                    
                    # Add to spread queue
                    spread_queue.append((connected_atom, distance + 1, spread_per_atom))
    
    def update_attention_dynamics(self):
        """Update attention dynamics with temporal decay and normalization"""
        # Apply temporal decay
        for attention_value in self.attention_values.values():
            attention_value.decay(self.sti_decay_rate, self.lti_decay_rate, self.vlti_decay_rate)
        
        # Normalize attention values to prevent inflation
        self._normalize_attention_values()
        
        # Update attention bank funds
        self._update_attention_bank()
    
    def _normalize_attention_values(self):
        """Normalize attention values to maintain economic balance"""
        total_sti = sum(av.short_term for av in self.attention_values.values())
        total_lti = sum(av.long_term for av in self.attention_values.values())
        
        # Normalize if totals exceed bank capacity
        if total_sti > self.attention_bank.total_sti:
            normalization_factor = self.attention_bank.total_sti / total_sti
            for av in self.attention_values.values():
                av.short_term *= normalization_factor
        
        if total_lti > self.attention_bank.total_lti:
            normalization_factor = self.attention_bank.total_lti / total_lti
            for av in self.attention_values.values():
                av.long_term *= normalization_factor
    
    def _update_attention_bank(self):
        """Update attention bank funds based on current allocations"""
        total_allocated_sti = sum(av.short_term for av in self.attention_values.values())
        total_allocated_lti = sum(av.long_term for av in self.attention_values.values())
        
        self.attention_bank.sti_funds = self.attention_bank.total_sti - total_allocated_sti
        self.attention_bank.lti_funds = self.attention_bank.total_lti - total_allocated_lti
    
    def _garbage_collect_attention(self, threshold: float = 0.05):
        """Collect attention from atoms with low attention values"""
        collected_sti = 0.0
        collected_lti = 0.0
        
        atoms_to_clean = []
        for atom_id, attention_value in self.attention_values.items():
            if attention_value.total_attention < threshold:
                collected_sti += attention_value.short_term
                collected_lti += attention_value.long_term
                atoms_to_clean.append(atom_id)
        
        # Remove low-attention atoms and return funds to bank
        for atom_id in atoms_to_clean:
            del self.attention_values[atom_id]
        
        self.attention_bank.collect_sti(collected_sti)
        self.attention_bank.collect_lti(collected_lti)
    
    def get_high_attention_atoms(self, attention_type: AttentionType = AttentionType.SHORT_TERM,
                               count: int = 10) -> List[Tuple[Atom, float]]:
        """Get atoms with highest attention values"""
        atom_attention_pairs = []
        
        for atom_id, attention_value in self.attention_values.items():
            if atom_id in self.atomspace.atoms:
                atom = self.atomspace.atoms[atom_id]
                
                if attention_type == AttentionType.SHORT_TERM:
                    attention_score = attention_value.short_term
                elif attention_type == AttentionType.LONG_TERM:
                    attention_score = attention_value.long_term
                elif attention_type == AttentionType.VERY_LONG_TERM:
                    attention_score = attention_value.very_long_term
                else:
                    attention_score = attention_value.total_attention
                
                atom_attention_pairs.append((atom, attention_score))
        
        # Sort by attention score and return top N
        atom_attention_pairs.sort(key=lambda x: x[1], reverse=True)
        return atom_attention_pairs[:count]
    
    def focus_attention_on_subspace(self, subspace_atoms: List[Atom], 
                                  focus_strength: float = 2.0):
        """Focus attention on a specific subspace of atoms"""
        # Increase attention for atoms in subspace
        for atom in subspace_atoms:
            if atom.atom_id in self.attention_values:
                self.attention_values[atom.atom_id].short_term *= focus_strength
                self.attention_values[atom.atom_id].long_term *= (focus_strength * 0.5)
        
        # Spread attention within subspace
        self.spread_attention(subspace_atoms, max_spread_distance=2, spreading_factor=0.5)
    
    def get_promising_ingredient_combinations(self, min_attention: float = 0.1,
                                            max_combinations: int = 20) -> List[List[Atom]]:
        """Get ingredient combinations with high attention values"""
        high_attention_ingredients = []
        
        # Get high-attention ingredient nodes
        for atom_id, attention_value in self.attention_values.items():
            if (atom_id in self.atomspace.atoms and
                attention_value.total_attention >= min_attention):
                atom = self.atomspace.atoms[atom_id]
                if atom.atom_type == AtomType.INGREDIENT_NODE:
                    high_attention_ingredients.append(atom)
        
        # Sort by attention
        high_attention_ingredients.sort(
            key=lambda atom: self.attention_values[atom.atom_id].total_attention,
            reverse=True
        )
        
        # Generate combinations based on synergy links
        combinations = []
        processed_pairs = set()
        
        for ingredient in high_attention_ingredients[:10]:  # Top 10 ingredients
            synergies = self.atomspace.get_ingredient_synergies(ingredient.name)
            
            for synergy_partner, strength in synergies:
                partner_atom = self.atomspace.get_atom_by_name(synergy_partner)
                if partner_atom and partner_atom.atom_id in self.attention_values:
                    pair = tuple(sorted([ingredient.name, synergy_partner]))
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        combinations.append([ingredient, partner_atom])
        
        return combinations[:max_combinations]
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get statistics about attention allocation"""
        if not self.attention_values:
            return {"total_atoms": 0}
        
        attention_values_list = list(self.attention_values.values())
        total_sti = sum(av.short_term for av in attention_values_list)
        total_lti = sum(av.long_term for av in attention_values_list)
        total_vlti = sum(av.very_long_term for av in attention_values_list)
        
        return {
            "total_atoms": len(self.attention_values),
            "total_sti_allocated": total_sti,
            "total_lti_allocated": total_lti,
            "total_vlti": total_vlti,
            "sti_funds_remaining": self.attention_bank.sti_funds,
            "lti_funds_remaining": self.attention_bank.lti_funds,
            "average_attention": sum(av.total_attention for av in attention_values_list) / len(attention_values_list),
            "high_attention_atoms": len([av for av in attention_values_list if av.total_attention > 0.5]),
            "allocation_strategies": list(self.allocation_strategies.keys())
        }
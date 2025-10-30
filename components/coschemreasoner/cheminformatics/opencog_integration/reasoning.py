"""
PLN-Inspired Reasoning Engine for Cosmeceutical Formulation

This module implements Probabilistic Logic Networks (PLN) inspired reasoning
for ingredient interactions, formulation constraints, and multiscale optimization.

Features:
- Probabilistic reasoning over ingredient relationships
- Uncertainty propagation through formulation networks
- Inference rules for cosmeceutical knowledge
- Bayesian updating of ingredient compatibility
- Fuzzy logic for constraint satisfaction
"""

from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import math
from collections import defaultdict
from abc import ABC, abstractmethod

from .atomspace import CosmeceuticalAtomSpace, Atom, AtomType


class TruthValueType(Enum):
    """Types of truth values in PLN-inspired reasoning"""
    SIMPLE = "simple"
    COUNT = "count"
    INDEFINITE = "indefinite"
    FUZZY = "fuzzy"


@dataclass
class TruthValue:
    """PLN-inspired truth value with strength and confidence"""
    strength: float = 0.5      # Probability or degree of truth (0-1)
    confidence: float = 0.0    # Confidence in the truth value (0-1)
    count: float = 0.0         # Count of evidence (for count truth values)
    
    def __post_init__(self):
        # Ensure values are in valid ranges
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.count = max(0.0, self.count)
    
    @property
    def expectation(self) -> float:
        """Expected value of the truth value"""
        return self.strength
    
    @property
    def weight(self) -> float:
        """Weight of evidence (related to confidence)"""
        return self.count / (1.0 + self.count)
    
    def update_with_evidence(self, new_strength: float, evidence_weight: float = 1.0):
        """Update truth value with new evidence using Bayesian updating"""
        old_weight = self.count
        new_count = self.count + evidence_weight
        
        if new_count > 0:
            # Weighted average of old and new strength
            self.strength = ((self.strength * old_weight) + 
                           (new_strength * evidence_weight)) / new_count
            self.count = new_count
            self.confidence = min(1.0, self.count / (10.0 + self.count))  # Sigmoid-like confidence


class InferenceRule(ABC):
    """Abstract base class for PLN-inspired inference rules"""
    
    def __init__(self, name: str, priority: float = 1.0):
        self.name = name
        self.priority = priority
    
    @abstractmethod
    def can_apply(self, premises: List[Atom], atomspace: CosmeceuticalAtomSpace) -> bool:
        """Check if the rule can be applied to the given premises"""
        pass
    
    @abstractmethod
    def apply(self, premises: List[Atom], atomspace: CosmeceuticalAtomSpace) -> List[Atom]:
        """Apply the inference rule and return new atoms"""
        pass
    
    @abstractmethod
    def calculate_conclusion_tv(self, premise_tvs: List[TruthValue]) -> TruthValue:
        """Calculate truth value for the conclusion"""
        pass


class DeductionRule(InferenceRule):
    """PLN deduction rule: If A->B and B->C then A->C"""
    
    def __init__(self):
        super().__init__("deduction", priority=0.8)
    
    def can_apply(self, premises: List[Atom], atomspace: CosmeceuticalAtomSpace) -> bool:
        """Check for deduction pattern: two implication links sharing a common term"""
        if len(premises) != 2:
            return False
        
        # Look for pattern where premise1 and premise2 share a common atom
        if (premises[0].atom_type in [AtomType.COMPATIBILITY_LINK, AtomType.SYNERGY_LINK] and
            premises[1].atom_type in [AtomType.COMPATIBILITY_LINK, AtomType.SYNERGY_LINK]):
            
            # Check if they share a common ingredient
            ingredients1 = set(atom.name for atom in premises[0].outgoing)
            ingredients2 = set(atom.name for atom in premises[1].outgoing)
            return len(ingredients1 & ingredients2) > 0
        
        return False
    
    def apply(self, premises: List[Atom], atomspace: CosmeceuticalAtomSpace) -> List[Atom]:
        """Apply deduction rule"""
        if not self.can_apply(premises, atomspace):
            return []
        
        # Find the common and unique ingredients
        ingredients1 = {atom.name: atom for atom in premises[0].outgoing}
        ingredients2 = {atom.name: atom for atom in premises[1].outgoing}
        
        common_ingredients = set(ingredients1.keys()) & set(ingredients2.keys())
        unique1 = set(ingredients1.keys()) - common_ingredients
        unique2 = set(ingredients2.keys()) - common_ingredients
        
        if len(unique1) == 1 and len(unique2) == 1:
            # Create new compatibility/synergy link between unique ingredients
            unique_ingredient1 = ingredients1[list(unique1)[0]]
            unique_ingredient2 = ingredients2[list(unique2)[0]]
            
            # Calculate conclusion truth value
            premise_tvs = [
                TruthValue(premises[0].truth_value, 0.8),
                TruthValue(premises[1].truth_value, 0.8)
            ]
            conclusion_tv = self.calculate_conclusion_tv(premise_tvs)
            
            # Create new atom based on premise types
            if (premises[0].atom_type == AtomType.COMPATIBILITY_LINK and
                premises[1].atom_type == AtomType.COMPATIBILITY_LINK):
                new_atom = atomspace.create_compatibility_link(
                    unique_ingredient1, unique_ingredient2, conclusion_tv.strength
                )
            elif (premises[0].atom_type == AtomType.SYNERGY_LINK and
                  premises[1].atom_type == AtomType.SYNERGY_LINK):
                new_atom = atomspace.create_synergy_link(
                    unique_ingredient1, unique_ingredient2, conclusion_tv.strength
                )
            else:
                # Mixed types - create compatibility link with lower confidence
                new_atom = atomspace.create_compatibility_link(
                    unique_ingredient1, unique_ingredient2, conclusion_tv.strength * 0.7
                )
            
            return [new_atom]
        
        return []
    
    def calculate_conclusion_tv(self, premise_tvs: List[TruthValue]) -> TruthValue:
        """Calculate truth value using PLN deduction formula"""
        if len(premise_tvs) != 2:
            return TruthValue(0.5, 0.0)
        
        tv1, tv2 = premise_tvs
        
        # PLN deduction strength formula (simplified)
        conclusion_strength = tv1.strength * tv2.strength
        
        # Confidence decreases with inference chain length
        conclusion_confidence = min(tv1.confidence, tv2.confidence) * 0.9
        
        return TruthValue(conclusion_strength, conclusion_confidence)


class InductionRule(InferenceRule):
    """PLN induction rule: generalize from specific instances"""
    
    def __init__(self):
        super().__init__("induction", priority=0.6)
    
    def can_apply(self, premises: List[Atom], atomspace: CosmeceuticalAtomSpace) -> bool:
        """Check for induction pattern: multiple similar relationships"""
        if len(premises) < 2:
            return False
        
        # All premises should be of the same link type
        link_type = premises[0].atom_type
        return all(premise.atom_type == link_type for premise in premises)
    
    def apply(self, premises: List[Atom], atomspace: CosmeceuticalAtomSpace) -> List[Atom]:
        """Apply induction rule to generalize patterns"""
        if not self.can_apply(premises, atomspace):
            return []
        
        # Group premises by ingredient categories
        category_patterns = defaultdict(list)
        
        for premise in premises:
            if len(premise.outgoing) == 2:
                categories = tuple(sorted([
                    ingredient.properties.get("category", "unknown")
                    for ingredient in premise.outgoing
                ]))
                category_patterns[categories].append(premise)
        
        # Create generalized rules for patterns with sufficient evidence
        new_atoms = []
        for categories, instances in category_patterns.items():
            if len(instances) >= 3:  # Require at least 3 instances for induction
                # Calculate average truth value
                avg_strength = np.mean([instance.truth_value for instance in instances])
                confidence = min(0.9, len(instances) / 10.0)  # Confidence based on evidence count
                
                # Create property atom representing the general rule
                rule_name = f"general_rule_{categories[0]}_{categories[1]}"
                rule_atom = atomspace.create_atom(
                    AtomType.PROPERTY_NODE,
                    rule_name,
                    truth_value=avg_strength,
                    properties={
                        "rule_type": "inductive",
                        "categories": categories,
                        "evidence_count": len(instances),
                        "confidence": confidence
                    }
                )
                new_atoms.append(rule_atom)
        
        return new_atoms
    
    def calculate_conclusion_tv(self, premise_tvs: List[TruthValue]) -> TruthValue:
        """Calculate truth value using inductive reasoning"""
        if not premise_tvs:
            return TruthValue(0.5, 0.0)
        
        # Inductive strength is average of premises
        avg_strength = np.mean([tv.strength for tv in premise_tvs])
        
        # Confidence increases with number of consistent instances
        evidence_count = len(premise_tvs)
        confidence = min(0.9, evidence_count / (evidence_count + 10.0))
        
        return TruthValue(avg_strength, confidence)


class AbductionRule(InferenceRule):
    """PLN abduction rule: infer best explanation"""
    
    def __init__(self):
        super().__init__("abduction", priority=0.7)
    
    def can_apply(self, premises: List[Atom], atomspace: CosmeceuticalAtomSpace) -> bool:
        """Check for abduction pattern: observed effect with possible causes"""
        return len(premises) >= 1
    
    def apply(self, premises: List[Atom], atomspace: CosmeceuticalAtomSpace) -> List[Atom]:
        """Apply abduction rule to infer explanations"""
        new_atoms = []
        
        for premise in premises:
            if premise.atom_type == AtomType.SYNERGY_LINK:
                # Abduce possible mechanisms for synergy
                mechanism_atom = atomspace.create_atom(
                    AtomType.PROPERTY_NODE,
                    f"synergy_mechanism_{premise.name}",
                    truth_value=premise.truth_value * 0.6,  # Lower confidence for abduced mechanisms
                    properties={
                        "rule_type": "abductive",
                        "explanation_for": premise.atom_id,
                        "mechanism_type": "biochemical_pathway"
                    }
                )
                new_atoms.append(mechanism_atom)
        
        return new_atoms
    
    def calculate_conclusion_tv(self, premise_tvs: List[TruthValue]) -> TruthValue:
        """Calculate truth value for abductive conclusions"""
        if not premise_tvs:
            return TruthValue(0.5, 0.0)
        
        # Abductive conclusions have lower confidence
        max_strength = max(tv.strength for tv in premise_tvs)
        avg_confidence = np.mean([tv.confidence for tv in premise_tvs]) * 0.6
        
        return TruthValue(max_strength * 0.8, avg_confidence)


class IngredientReasoningEngine:
    """
    PLN-inspired reasoning engine for cosmeceutical ingredient interactions.
    
    This class implements probabilistic reasoning over ingredient relationships,
    handling uncertainty and evidence accumulation in formulation optimization.
    """
    
    def __init__(self, atomspace: CosmeceuticalAtomSpace):
        self.atomspace = atomspace
        self.truth_values: Dict[str, TruthValue] = {}
        self.inference_rules: List[InferenceRule] = []
        self.evidence_log: List[Dict[str, Any]] = []
        
        # Initialize inference rules
        self._initialize_inference_rules()
        
        # Initialize truth values for existing atoms
        self._initialize_truth_values()
    
    def _initialize_inference_rules(self):
        """Initialize the set of inference rules"""
        self.inference_rules = [
            DeductionRule(),
            InductionRule(),
            AbductionRule()
        ]
        
        # Sort by priority
        self.inference_rules.sort(key=lambda rule: rule.priority, reverse=True)
    
    def _initialize_truth_values(self):
        """Initialize truth values for all atoms"""
        for atom_id, atom in self.atomspace.atoms.items():
            # Initialize with basic truth value based on atom properties
            initial_strength = atom.truth_value
            initial_confidence = 0.5  # Medium confidence initially
            
            if atom.atom_type in [AtomType.COMPATIBILITY_LINK, AtomType.SYNERGY_LINK]:
                # Links start with higher confidence if they have supporting properties
                if "evidence_count" in atom.properties:
                    evidence_count = atom.properties["evidence_count"]
                    initial_confidence = min(0.9, evidence_count / (evidence_count + 5.0))
            
            self.truth_values[atom_id] = TruthValue(initial_strength, initial_confidence)
    
    def add_evidence(self, atom_id: str, evidence_strength: float, 
                    evidence_weight: float = 1.0, source: str = "unknown"):
        """Add evidence for an atom's truth value"""
        if atom_id in self.truth_values:
            old_tv = self.truth_values[atom_id]
            self.truth_values[atom_id].update_with_evidence(evidence_strength, evidence_weight)
            
            # Log the evidence
            self.evidence_log.append({
                "atom_id": atom_id,
                "old_strength": old_tv.strength,
                "new_strength": self.truth_values[atom_id].strength,
                "evidence_strength": evidence_strength,
                "evidence_weight": evidence_weight,
                "source": source
            })
    
    def evaluate_ingredient_compatibility(self, ingredient1_name: str, 
                                        ingredient2_name: str) -> TruthValue:
        """Evaluate compatibility between two ingredients using reasoning"""
        # Direct compatibility check
        direct_compatibility = self.atomspace.get_ingredient_compatibility(
            ingredient1_name, ingredient2_name
        )
        
        if direct_compatibility is not None:
            # Return stored compatibility with current truth value
            compatibility_links = [
                atom for atom in self.atomspace.get_atoms_by_type(AtomType.COMPATIBILITY_LINK)
                if len(atom.outgoing) == 2 and
                {atom.name for atom in atom.outgoing} == {ingredient1_name, ingredient2_name}
            ]
            
            if compatibility_links:
                atom_id = compatibility_links[0].atom_id
                return self.truth_values.get(atom_id, TruthValue(direct_compatibility, 0.5))
        
        # Infer compatibility using reasoning rules
        return self._infer_ingredient_compatibility(ingredient1_name, ingredient2_name)
    
    def _infer_ingredient_compatibility(self, ingredient1_name: str, 
                                      ingredient2_name: str) -> TruthValue:
        """Infer compatibility using PLN-inspired reasoning"""
        # Get atoms for both ingredients
        ingredient1 = self.atomspace.get_atom_by_name(ingredient1_name)
        ingredient2 = self.atomspace.get_atom_by_name(ingredient2_name)
        
        if not (ingredient1 and ingredient2):
            return TruthValue(0.5, 0.1)  # Unknown compatibility with low confidence
        
        # Try to find indirect evidence through common connections
        common_partners = set()
        
        # Find ingredients that are compatible with both
        for atom in self.atomspace.get_atoms_by_type(AtomType.COMPATIBILITY_LINK):
            if len(atom.outgoing) == 2:
                names = {a.name for a in atom.outgoing}
                if ingredient1_name in names:
                    common_partners.update(names - {ingredient1_name})
                elif ingredient2_name in names:
                    common_partners.intersection_update(names - {ingredient2_name})
        
        if common_partners:
            # Use transitivity: if both are compatible with common partners,
            # they're likely compatible with each other
            compatibility_scores = []
            
            for partner in common_partners:
                comp1 = self.atomspace.get_ingredient_compatibility(ingredient1_name, partner)
                comp2 = self.atomspace.get_ingredient_compatibility(ingredient2_name, partner)
                
                if comp1 is not None and comp2 is not None:
                    # Transitive compatibility score
                    transitivity_score = math.sqrt(comp1 * comp2)  # Geometric mean
                    compatibility_scores.append(transitivity_score)
            
            if compatibility_scores:
                avg_compatibility = np.mean(compatibility_scores)
                confidence = min(0.7, len(compatibility_scores) / 10.0)
                return TruthValue(avg_compatibility, confidence)
        
        # Check category-based compatibility
        category_compatibility = self._evaluate_category_compatibility(ingredient1, ingredient2)
        if category_compatibility:
            return category_compatibility
        
        # Default to neutral compatibility with low confidence
        return TruthValue(0.5, 0.2)
    
    def _evaluate_category_compatibility(self, ingredient1: Atom, 
                                       ingredient2: Atom) -> Optional[TruthValue]:
        """Evaluate compatibility based on ingredient categories"""
        cat1 = ingredient1.properties.get("category", "").lower()
        cat2 = ingredient2.properties.get("category", "").lower()
        
        if not (cat1 and cat2):
            return None
        
        # Category compatibility rules (simplified)
        compatibility_rules = {
            ("active_ingredient", "humectant"): TruthValue(0.8, 0.6),
            ("active_ingredient", "emulsifier"): TruthValue(0.7, 0.6),
            ("humectant", "emollient"): TruthValue(0.9, 0.7),
            ("emulsifier", "thickener"): TruthValue(0.8, 0.7),
            ("antioxidant", "active_ingredient"): TruthValue(0.8, 0.6),
            ("preservative", "humectant"): TruthValue(0.9, 0.8),
        }
        
        # Check both orders
        category_pair = (cat1, cat2)
        reverse_pair = (cat2, cat1)
        
        if category_pair in compatibility_rules:
            return compatibility_rules[category_pair]
        elif reverse_pair in compatibility_rules:
            return compatibility_rules[reverse_pair]
        
        return None
    
    def run_inference_cycle(self, max_iterations: int = 10) -> List[Atom]:
        """Run inference cycle to derive new knowledge"""
        new_atoms = []
        
        for iteration in range(max_iterations):
            iteration_new_atoms = []
            
            # Apply each inference rule
            for rule in self.inference_rules:
                # Get potential premises for this rule
                all_atoms = list(self.atomspace.atoms.values())
                
                # Try to apply rule to combinations of atoms
                for i in range(len(all_atoms)):
                    for j in range(i + 1, len(all_atoms)):
                        premises = [all_atoms[i], all_atoms[j]]
                        
                        if rule.can_apply(premises, self.atomspace):
                            rule_new_atoms = rule.apply(premises, self.atomspace)
                            iteration_new_atoms.extend(rule_new_atoms)
                            
                            # Update truth values for new atoms
                            for new_atom in rule_new_atoms:
                                if new_atom.atom_id not in self.truth_values:
                                    self.truth_values[new_atom.atom_id] = TruthValue(
                                        new_atom.truth_value, 0.5
                                    )
            
            # Stop if no new atoms were created
            if not iteration_new_atoms:
                break
            
            new_atoms.extend(iteration_new_atoms)
        
        return new_atoms
    
    def evaluate_formulation_consistency(self, ingredient_atoms: List[Atom]) -> TruthValue:
        """Evaluate overall consistency of a formulation"""
        if len(ingredient_atoms) < 2:
            return TruthValue(1.0, 1.0)  # Single ingredient is always consistent
        
        compatibility_scores = []
        incompatibility_count = 0
        
        # Check all pairwise compatibilities
        for i in range(len(ingredient_atoms)):
            for j in range(i + 1, len(ingredient_atoms)):
                ingredient1 = ingredient_atoms[i]
                ingredient2 = ingredient_atoms[j]
                
                compatibility_tv = self.evaluate_ingredient_compatibility(
                    ingredient1.name, ingredient2.name
                )
                compatibility_scores.append(compatibility_tv.strength)
                
                # Check for explicit incompatibilities
                incompatibility = self.atomspace.get_ingredient_compatibility(
                    ingredient1.name, ingredient2.name
                )
                if incompatibility is not None and incompatibility < 0.3:
                    incompatibility_count += 1
        
        if not compatibility_scores:
            return TruthValue(0.5, 0.1)
        
        # Calculate overall consistency
        avg_compatibility = np.mean(compatibility_scores)
        
        # Penalize for incompatibilities
        incompatibility_penalty = incompatibility_count * 0.2
        consistency_strength = max(0.0, avg_compatibility - incompatibility_penalty)
        
        # Confidence based on number of evaluations
        confidence = min(0.9, len(compatibility_scores) / (len(compatibility_scores) + 5.0))
        
        return TruthValue(consistency_strength, confidence)
    
    def update_truth_values_from_feedback(self, feedback_data: Dict[str, Any]):
        """Update truth values based on experimental or user feedback"""
        for ingredient_pair, feedback in feedback_data.items():
            if isinstance(feedback, dict) and "compatibility" in feedback:
                compatibility_score = feedback["compatibility"]
                confidence = feedback.get("confidence", 0.8)
                
                # Find corresponding atoms
                ingredient_names = ingredient_pair.split("_")
                if len(ingredient_names) == 2:
                    compatibility_links = [
                        atom for atom in self.atomspace.get_atoms_by_type(AtomType.COMPATIBILITY_LINK)
                        if len(atom.outgoing) == 2 and
                        {atom.name for atom in atom.outgoing} == set(ingredient_names)
                    ]
                    
                    for link in compatibility_links:
                        self.add_evidence(
                            link.atom_id, 
                            compatibility_score, 
                            confidence,
                            source="experimental_feedback"
                        )
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reasoning process"""
        return {
            "total_truth_values": len(self.truth_values),
            "evidence_entries": len(self.evidence_log),
            "inference_rules": len(self.inference_rules),
            "average_confidence": np.mean([tv.confidence for tv in self.truth_values.values()]) if self.truth_values else 0.0,
            "high_confidence_atoms": len([tv for tv in self.truth_values.values() if tv.confidence > 0.8]),
            "recent_evidence": len([entry for entry in self.evidence_log[-100:]]),  # Last 100 entries
        }
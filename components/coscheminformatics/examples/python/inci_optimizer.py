#!/usr/bin/env python3
"""
inci_optimizer.py

INCI-Driven Search Space Reduction for Cosmeceutical Formulation

This module implements algorithms for reducing the potential ingredient search space
based on INCI (International Nomenclature of Cosmetic Ingredients) regulations and
trade name ingredient listings. It provides functionality for:

1. Parsing INCI ingredient lists from product formulations
2. Estimating absolute concentrations from INCI ordering and volume constraints
3. Filtering formulation space based on regulatory compliance
4. Reducing search space to INCI subset relationships

Key Features:
- INCI list parsing and concentration estimation
- Regulatory compliance checking
- Search space pruning based on subset relationships
- Integration with OpenCog AtomSpace for knowledge representation
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Mock OpenCog imports for this implementation
# In actual use, these would be:
# from opencog.atomspace import AtomSpace
# from opencog.type_constructors import *
# from opencog.cheminformatics import *

class MockAtomSpace:
    """Mock AtomSpace for demonstration purposes"""
    def __init__(self):
        self.atoms = {}
    
    def add_atom(self, atom):
        self.atoms[str(atom)] = atom
        return atom

# Mock atom types
class MockAtom:
    def __init__(self, name, atom_type="CONCEPT_NODE"):
        self.name = name
        self.type = atom_type
    
    def __str__(self):
        return f"({self.type} \"{self.name}\")"

@dataclass
class IngredientInfo:
    """Information about a cosmetic ingredient"""
    name: str
    inci_name: str
    category: str
    typical_concentration_range: Tuple[float, float]  # (min%, max%)
    regulatory_limit: Optional[float] = None
    cas_number: Optional[str] = None
    molecular_weight: Optional[float] = None
    ph_stability_range: Optional[Tuple[float, float]] = None

@dataclass 
class FormulationConstraint:
    """Constraints for formulation optimization"""
    target_ph: Optional[Tuple[float, float]] = None
    max_total_actives: Optional[float] = None
    required_ingredients: List[str] = None
    forbidden_ingredients: List[str] = None
    cost_limit: Optional[float] = None

class ConcentrationEstimationMethod(Enum):
    """Methods for estimating concentrations from INCI ordering"""
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay" 
    CATEGORY_BASED = "category_based"
    REGULATORY_INFORMED = "regulatory_informed"

class INCISearchSpaceReducer:
    """
    Main class for INCI-driven search space reduction in cosmeceutical formulation.
    
    This class provides methods to parse INCI ingredient lists, estimate concentrations,
    filter formulation spaces based on regulatory constraints, and integrate with
    OpenCog AtomSpace for knowledge representation.
    """
    
    def __init__(self, atomspace: MockAtomSpace = None):
        self.atomspace = atomspace or MockAtomSpace()
        self.ingredient_database = {}
        self.regulatory_limits = {}
        self.formulation_database = {}
        self._load_ingredient_data()
        self._load_regulatory_data()
    
    def _load_ingredient_data(self):
        """Load ingredient database with INCI names, categories, and properties"""
        # In a real implementation, this would load from a comprehensive database
        self.ingredient_database = {
            "water": IngredientInfo(
                name="water",
                inci_name="Aqua",
                category="solvent",
                typical_concentration_range=(60.0, 90.0),
                regulatory_limit=None
            ),
            "hyaluronic_acid": IngredientInfo(
                name="hyaluronic_acid", 
                inci_name="Sodium Hyaluronate",
                category="active_ingredient",
                typical_concentration_range=(0.1, 2.0),
                regulatory_limit=2.0,
                ph_stability_range=(4.0, 8.0)
            ),
            "retinol": IngredientInfo(
                name="retinol",
                inci_name="Retinol", 
                category="active_ingredient",
                typical_concentration_range=(0.01, 1.0),
                regulatory_limit=1.0,
                ph_stability_range=(5.5, 7.0)
            ),
            "vitamin_c": IngredientInfo(
                name="vitamin_c",
                inci_name="L-Ascorbic Acid",
                category="active_ingredient", 
                typical_concentration_range=(0.5, 20.0),
                regulatory_limit=20.0,
                ph_stability_range=(2.0, 4.0)
            ),
            "niacinamide": IngredientInfo(
                name="niacinamide",
                inci_name="Niacinamide",
                category="active_ingredient",
                typical_concentration_range=(1.0, 10.0), 
                regulatory_limit=10.0,
                ph_stability_range=(5.0, 7.0)
            ),
            "glycerin": IngredientInfo(
                name="glycerin",
                inci_name="Glycerin",
                category="humectant",
                typical_concentration_range=(1.0, 15.0),
                regulatory_limit=None
            ),
            "phenoxyethanol": IngredientInfo(
                name="phenoxyethanol",
                inci_name="Phenoxyethanol",
                category="preservative",
                typical_concentration_range=(0.1, 1.0),
                regulatory_limit=1.0
            ),
            "cetyl_alcohol": IngredientInfo(
                name="cetyl_alcohol",
                inci_name="Cetyl Alcohol", 
                category="emulsifier",
                typical_concentration_range=(1.0, 5.0),
                regulatory_limit=None
            )
        }
    
    def _load_regulatory_data(self):
        """Load regulatory limits and compliance data"""
        # Sample regulatory data - in practice would come from comprehensive database
        self.regulatory_limits = {
            "retinol": {"max_concentration": 1.0, "regions": ["EU", "US"]},
            "vitamin_c": {"max_concentration": 20.0, "regions": ["EU", "US"]}, 
            "niacinamide": {"max_concentration": 10.0, "regions": ["EU", "US"]},
            "phenoxyethanol": {"max_concentration": 1.0, "regions": ["EU", "US"]}
        }
    
    def parse_inci_list(self, inci_string: str) -> List[Dict[str, any]]:
        """
        Parse INCI ingredient list and extract ingredient information.
        
        Args:
            inci_string: Comma-separated INCI ingredient list
            
        Returns:
            List of ingredient dictionaries with names and estimated positions
        """
        # Clean and split INCI string
        ingredients = [ing.strip() for ing in inci_string.split(',')]
        
        parsed_ingredients = []
        for i, inci_name in enumerate(ingredients):
            # Find ingredient by INCI name
            ingredient_key = None
            for key, info in self.ingredient_database.items():
                if info.inci_name.lower() == inci_name.lower():
                    ingredient_key = key
                    break
            
            if ingredient_key:
                parsed_ingredients.append({
                    'inci_name': inci_name,
                    'ingredient_key': ingredient_key,
                    'position': i + 1,
                    'info': self.ingredient_database[ingredient_key]
                })
            else:
                # Unknown ingredient - add with minimal info
                parsed_ingredients.append({
                    'inci_name': inci_name,
                    'ingredient_key': inci_name.lower().replace(' ', '_'),
                    'position': i + 1,
                    'info': None
                })
        
        return parsed_ingredients
    
    def estimate_concentrations(
        self, 
        parsed_ingredients: List[Dict],
        total_volume: float = 100.0,
        method: ConcentrationEstimationMethod = ConcentrationEstimationMethod.CATEGORY_BASED
    ) -> Dict[str, float]:
        """
        Estimate ingredient concentrations from INCI ordering and constraints.
        
        Args:
            parsed_ingredients: List of parsed ingredient dictionaries
            total_volume: Total formulation volume (default 100% for percentage calculations)
            method: Concentration estimation method
            
        Returns:
            Dictionary mapping ingredient keys to estimated concentrations
        """
        concentrations = {}
        
        if method == ConcentrationEstimationMethod.CATEGORY_BASED:
            # Estimate based on typical category concentrations and position
            remaining_volume = total_volume
            
            for ingredient in parsed_ingredients:
                if ingredient['info']:
                    # Use category-based estimation
                    min_conc, max_conc = ingredient['info'].typical_concentration_range
                    
                    # Adjust based on position (earlier = higher concentration)
                    position_factor = 1.0 / (ingredient['position'] ** 0.5)
                    estimated_conc = min(max_conc * position_factor, remaining_volume)
                    
                    # Ensure minimum viable concentration
                    estimated_conc = max(estimated_conc, min_conc)
                    
                    concentrations[ingredient['ingredient_key']] = estimated_conc
                    remaining_volume -= estimated_conc
                else:
                    # Unknown ingredient - estimate conservatively
                    estimated_conc = min(1.0, remaining_volume * 0.1)
                    concentrations[ingredient['ingredient_key']] = estimated_conc
                    remaining_volume -= estimated_conc
        
        elif method == ConcentrationEstimationMethod.LINEAR_DECAY:
            # Simple linear decay based on position
            total_positions = len(parsed_ingredients)
            base_concentration = total_volume / (total_positions * 2)  # Conservative estimate
            
            for ingredient in parsed_ingredients:
                decay_factor = (total_positions - ingredient['position'] + 1) / total_positions
                concentrations[ingredient['ingredient_key']] = base_concentration * decay_factor
        
        return concentrations
    
    def filter_formulation_space(
        self, 
        target_inci: str, 
        constraints: FormulationConstraint,
        candidate_formulations: List[Dict] = None
    ) -> List[Dict]:
        """
        Filter formulation search space based on INCI constraints and regulatory compliance.
        
        Args:
            target_inci: Target INCI ingredient list
            constraints: Formulation constraints
            candidate_formulations: List of candidate formulations to filter
            
        Returns:
            Filtered list of viable formulations
        """
        target_ingredients = self.parse_inci_list(target_inci)
        target_keys = {ing['ingredient_key'] for ing in target_ingredients}
        
        filtered_formulations = []
        
        # If no candidates provided, generate from ingredient database
        if candidate_formulations is None:
            candidate_formulations = self._generate_candidate_formulations(target_keys)
        
        for formulation in candidate_formulations:
            # Check if formulation INCI is subset of target
            formulation_keys = set(formulation.get('ingredients', {}).keys())
            
            if formulation_keys.issubset(target_keys):
                # Check regulatory compliance
                if self.check_regulatory_compliance(formulation):
                    # Check additional constraints
                    if self._meets_constraints(formulation, constraints):
                        filtered_formulations.append(formulation)
        
        return filtered_formulations
    
    def check_regulatory_compliance(self, formulation: Dict) -> bool:
        """
        Check if formulation meets regulatory requirements.
        
        Args:
            formulation: Formulation dictionary with ingredients and concentrations
            
        Returns:
            Boolean indicating regulatory compliance
        """
        ingredients = formulation.get('ingredients', {})
        
        for ingredient_key, concentration in ingredients.items():
            # Check concentration limits
            if ingredient_key in self.regulatory_limits:
                limit_data = self.regulatory_limits[ingredient_key]
                max_allowed = limit_data['max_concentration']
                
                if concentration > max_allowed:
                    return False
            
            # Check ingredient-specific regulations
            if ingredient_key in self.ingredient_database:
                ingredient_info = self.ingredient_database[ingredient_key]
                if ingredient_info.regulatory_limit and concentration > ingredient_info.regulatory_limit:
                    return False
        
        return True
    
    def _generate_candidate_formulations(self, target_keys: Set[str]) -> List[Dict]:
        """Generate candidate formulations based on target ingredient keys"""
        # Simple candidate generation - in practice would be more sophisticated
        candidates = []
        
        # Generate formulations with subsets of target ingredients
        import itertools
        for r in range(1, min(len(target_keys) + 1, 6)):  # Max 5 ingredients for simplicity
            for ingredient_combo in itertools.combinations(target_keys, r):
                # Estimate concentrations for this combination
                mock_inci = ', '.join([self.ingredient_database.get(key, IngredientInfo(key, key, "unknown", (1.0, 5.0))).inci_name for key in ingredient_combo if key in self.ingredient_database])
                parsed = self.parse_inci_list(mock_inci)
                concentrations = self.estimate_concentrations(parsed)
                
                candidates.append({
                    'ingredients': concentrations,
                    'formulation_type': 'skincare_serum',
                    'estimated_efficacy': 0.5  # Placeholder
                })
        
        return candidates
    
    def _meets_constraints(self, formulation: Dict, constraints: FormulationConstraint) -> bool:
        """Check if formulation meets additional constraints"""
        ingredients = formulation.get('ingredients', {})
        
        # Check required ingredients
        if constraints.required_ingredients:
            for required in constraints.required_ingredients:
                if required not in ingredients:
                    return False
        
        # Check forbidden ingredients  
        if constraints.forbidden_ingredients:
            for forbidden in constraints.forbidden_ingredients:
                if forbidden in ingredients:
                    return False
        
        # Check total actives concentration
        if constraints.max_total_actives:
            total_actives = 0
            for ingredient_key, concentration in ingredients.items():
                if ingredient_key in self.ingredient_database:
                    ingredient_info = self.ingredient_database[ingredient_key]
                    if ingredient_info.category == 'active_ingredient':
                        total_actives += concentration
            
            if total_actives > constraints.max_total_actives:
                return False
        
        return True
    
    def create_atomspace_representation(self, formulation: Dict) -> List[MockAtom]:
        """
        Create AtomSpace representation of formulation for OpenCog reasoning.
        
        Args:
            formulation: Formulation dictionary
            
        Returns:
            List of atoms representing the formulation
        """
        atoms = []
        
        # Create formulation node
        formulation_atom = MockAtom(f"formulation_{formulation.get('id', 'unknown')}", "SKINCARE_FORMULATION")
        atoms.append(formulation_atom)
        self.atomspace.add_atom(formulation_atom)
        
        # Create ingredient atoms and links
        for ingredient_key, concentration in formulation.get('ingredients', {}).items():
            if ingredient_key in self.ingredient_database:
                ingredient_info = self.ingredient_database[ingredient_key]
                
                # Create ingredient atom
                ingredient_atom = MockAtom(ingredient_key, "ACTIVE_INGREDIENT" if ingredient_info.category == "active_ingredient" else "COSMETIC_INGREDIENT")
                atoms.append(ingredient_atom)
                self.atomspace.add_atom(ingredient_atom)
                
                # Create concentration property
                conc_atom = MockAtom(f"{ingredient_key}_concentration_{concentration}", "CONCENTRATION_PROPERTY")
                atoms.append(conc_atom)
                self.atomspace.add_atom(conc_atom)
        
        return atoms

# Example usage and testing functions
def example_inci_analysis():
    """Example demonstrating INCI analysis and search space reduction"""
    print("=== INCI-Driven Search Space Reduction Example ===")
    
    reducer = INCISearchSpaceReducer()
    
    # Example target product INCI list
    target_inci = "Aqua, Sodium Hyaluronate, Niacinamide, Glycerin, Phenoxyethanol"
    
    print(f"\nTarget INCI list: {target_inci}")
    
    # Parse INCI list
    parsed = reducer.parse_inci_list(target_inci)
    print(f"\nParsed ingredients:")
    for ing in parsed:
        print(f"  {ing['position']}. {ing['inci_name']} ({ing['ingredient_key']})")
    
    # Estimate concentrations
    concentrations = reducer.estimate_concentrations(parsed)
    print(f"\nEstimated concentrations:")
    for key, conc in concentrations.items():
        print(f"  {key}: {conc:.2f}%")
    
    # Define constraints
    constraints = FormulationConstraint(
        target_ph=(5.0, 7.0),
        max_total_actives=15.0,
        required_ingredients=["water"]
    )
    
    # Filter formulation space
    filtered_formulations = reducer.filter_formulation_space(target_inci, constraints)
    print(f"\nFound {len(filtered_formulations)} viable formulations after filtering")
    
    # Show first few results
    for i, formulation in enumerate(filtered_formulations[:3]):
        print(f"\nFormulation {i+1}:")
        for ingredient, conc in formulation['ingredients'].items():
            print(f"  {ingredient}: {conc:.2f}%")
    
    print("\n=== INCI Analysis Complete ===")

if __name__ == "__main__":
    example_inci_analysis()
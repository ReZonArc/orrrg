"""
INCI-Driven Search Space Reduction for Cosmeceutical Formulation

This module implements algorithms to reduce the potential ingredient search space
based on INCI (International Nomenclature of Cosmetic Ingredients) data and
regulatory constraints.

Features:
- INCI-based search space pruning
- Concentration estimation from INCI ordering
- Regulatory compliance filtering
- Trade name to INCI mapping
- Search space optimization algorithms
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict
import numpy as np

from .atomspace import CosmeceuticalAtomSpace, Atom, AtomType


class RegulationRegion(Enum):
    """Regulatory regions for cosmetic ingredients"""
    FDA = "FDA"
    EU = "EU" 
    ASEAN = "ASEAN"
    CHINA = "CHINA"
    JAPAN = "JAPAN"
    CANADA = "CANADA"


@dataclass
class INCIEntry:
    """Represents an INCI ingredient entry"""
    inci_name: str
    trade_names: List[str] = field(default_factory=list)
    cas_number: Optional[str] = None
    function: List[str] = field(default_factory=list)
    concentration_range: Optional[Tuple[float, float]] = None
    regulatory_status: Dict[RegulationRegion, str] = field(default_factory=dict)
    restrictions: Dict[RegulationRegion, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ProductFormulation:
    """Represents a cosmetic product with INCI list"""
    product_name: str
    inci_list: List[str]  # Ordered by concentration (highest to lowest)
    product_type: str
    region: RegulationRegion
    total_active_content: Optional[float] = None  # If known


class INCISearchOptimizer:
    """
    INCI-driven search space reduction and optimization for cosmeceutical formulation.
    
    This class implements algorithms to reduce the search space of potential 
    ingredients based on INCI listings, regulatory constraints, and concentration
    estimation from ingredient ordering.
    """
    
    def __init__(self, atomspace: CosmeceuticalAtomSpace):
        self.atomspace = atomspace
        self.inci_database: Dict[str, INCIEntry] = {}
        self.trade_name_mapping: Dict[str, str] = {}  # trade_name -> inci_name
        self.regulatory_limits: Dict[RegulationRegion, Dict[str, float]] = {}
        self.concentration_estimation_rules: Dict[str, Tuple[float, float]] = {}
        
        # Initialize databases
        self._initialize_inci_database()
        self._initialize_regulatory_limits()
        self._initialize_concentration_rules()
    
    def _initialize_inci_database(self):
        """Initialize the INCI ingredient database"""
        # Common cosmetic ingredients with INCI names
        ingredients_data = [
            {
                "inci_name": "HYALURONIC ACID",
                "trade_names": ["Hyaluronic Acid", "Sodium Hyaluronate", "HA"],
                "cas_number": "9067-32-7",
                "function": ["humectant", "skin_conditioning"],
                "concentration_range": (0.01, 2.0)
            },
            {
                "inci_name": "NIACINAMIDE", 
                "trade_names": ["Niacinamide", "Nicotinamide", "Vitamin B3"],
                "cas_number": "98-92-0",
                "function": ["skin_conditioning", "anti_aging"],
                "concentration_range": (1.0, 10.0)
            },
            {
                "inci_name": "ASCORBIC ACID",
                "trade_names": ["Vitamin C", "L-Ascorbic Acid", "Ascorbic Acid"],
                "cas_number": "50-81-7", 
                "function": ["antioxidant", "skin_brightening"],
                "concentration_range": (5.0, 20.0)
            },
            {
                "inci_name": "RETINOL",
                "trade_names": ["Retinol", "Vitamin A"],
                "cas_number": "68-26-8",
                "function": ["anti_aging", "skin_conditioning"],
                "concentration_range": (0.01, 1.0)
            },
            {
                "inci_name": "GLYCERIN",
                "trade_names": ["Glycerin", "Glycerol", "Propanetriol"],
                "cas_number": "56-81-5",
                "function": ["humectant", "solvent"],
                "concentration_range": (1.0, 50.0)
            },
            {
                "inci_name": "PHENOXYETHANOL",
                "trade_names": ["Phenoxyethanol", "2-Phenoxyethanol"],
                "cas_number": "122-99-6",
                "function": ["preservative"],
                "concentration_range": (0.1, 1.0)
            },
            {
                "inci_name": "CETYL ALCOHOL",
                "trade_names": ["Cetyl Alcohol", "1-Hexadecanol"],
                "cas_number": "36653-82-4",
                "function": ["emulsifier", "thickener"],
                "concentration_range": (0.5, 10.0)
            },
            {
                "inci_name": "TOCOPHEROL",
                "trade_names": ["Vitamin E", "Alpha-Tocopherol", "Tocopherol"],
                "cas_number": "59-02-9",
                "function": ["antioxidant", "skin_conditioning"],
                "concentration_range": (0.1, 1.0)
            }
        ]
        
        for data in ingredients_data:
            entry = INCIEntry(
                inci_name=data["inci_name"],
                trade_names=data["trade_names"],
                cas_number=data["cas_number"],
                function=data["function"],
                concentration_range=data["concentration_range"]
            )
            
            self.inci_database[data["inci_name"]] = entry
            
            # Build trade name mapping
            for trade_name in data["trade_names"]:
                self.trade_name_mapping[trade_name.lower()] = data["inci_name"]
    
    def _initialize_regulatory_limits(self):
        """Initialize regulatory concentration limits by region"""
        # EU regulatory limits (simplified)
        self.regulatory_limits[RegulationRegion.EU] = {
            "PHENOXYETHANOL": 1.0,
            "RETINOL": 0.3,
            "ASCORBIC ACID": 15.0,
            "NIACINAMIDE": 10.0,
            "HYALURONIC ACID": 2.0
        }
        
        # FDA limits (simplified - FDA doesn't set specific limits for cosmetics)
        self.regulatory_limits[RegulationRegion.FDA] = {
            "PHENOXYETHANOL": 1.0,
            "RETINOL": 1.0,
            "ASCORBIC ACID": 20.0,
            "NIACINAMIDE": 10.0,
            "HYALURONIC ACID": 2.0
        }
        
        # Other regions - simplified placeholder data
        for region in [RegulationRegion.ASEAN, RegulationRegion.CHINA, 
                      RegulationRegion.JAPAN, RegulationRegion.CANADA]:
            self.regulatory_limits[region] = self.regulatory_limits[RegulationRegion.EU].copy()
    
    def _initialize_concentration_rules(self):
        """Initialize concentration estimation rules based on INCI position"""
        # Typical concentration ranges based on INCI list position
        self.concentration_estimation_rules = {
            "water": (60.0, 90.0),     # Usually first and highest
            "glycerin": (5.0, 15.0),   # Common humectant
            "preservative": (0.1, 1.0), # Usually low concentration
            "active": (0.5, 10.0),     # Variable based on specific active
            "emulsifier": (1.0, 8.0),  # Moderate concentration
            "thickener": (0.1, 3.0),   # Usually low concentration
            "fragrance": (0.1, 2.0)    # Usually very low concentration
        }
    
    def normalize_ingredient_name(self, name: str) -> str:
        """Normalize ingredient name to INCI format"""
        # Remove common descriptors and normalize
        name = re.sub(r'\b(extract|oil|powder|solution)\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[^\w\s]', '', name)  # Remove special characters
        name = ' '.join(name.split()).upper()  # Normalize whitespace and uppercase
        
        # Check trade name mapping
        if name.lower() in self.trade_name_mapping:
            return self.trade_name_mapping[name.lower()]
        
        return name
    
    def get_inci_entry(self, ingredient_name: str) -> Optional[INCIEntry]:
        """Get INCI entry for an ingredient"""
        normalized_name = self.normalize_ingredient_name(ingredient_name)
        return self.inci_database.get(normalized_name)
    
    def estimate_concentrations_from_inci(self, inci_list: List[str], 
                                        total_volume: float = 100.0) -> Dict[str, float]:
        """
        Estimate absolute concentrations from INCI list ordering.
        
        Uses regulatory rules and typical formulation patterns to estimate
        concentrations based on INCI list position and ingredient types.
        """
        estimated_concentrations = {}
        remaining_volume = total_volume
        
        # Handle water first (if present)
        if inci_list and inci_list[0].upper() in ["WATER", "AQUA"]:
            water_concentration = min(remaining_volume * 0.7, 85.0)  # Typical water content
            estimated_concentrations["AQUA"] = water_concentration
            remaining_volume -= water_concentration
            inci_list = inci_list[1:]  # Remove water from further processing
        
        # Process remaining ingredients
        n_ingredients = len(inci_list)
        if n_ingredients == 0:
            return estimated_concentrations
        
        # Use exponential decay for concentration estimation
        # First ingredient gets highest share, subsequent ones get exponentially less
        decay_factor = 0.7
        base_concentrations = []
        
        for i in range(n_ingredients):
            base_conc = remaining_volume * (decay_factor ** i)
            base_concentrations.append(base_conc)
        
        # Normalize to fit remaining volume
        total_base = sum(base_concentrations)
        if total_base > 0:
            normalization_factor = remaining_volume / total_base
            base_concentrations = [conc * normalization_factor for conc in base_concentrations]
        
        # Apply ingredient-specific constraints
        for i, inci_name in enumerate(inci_list):
            estimated_conc = base_concentrations[i]
            normalized_name = self.normalize_ingredient_name(inci_name)
            
            # Apply INCI database constraints
            inci_entry = self.inci_database.get(normalized_name)
            if inci_entry and inci_entry.concentration_range:
                min_conc, max_conc = inci_entry.concentration_range
                estimated_conc = max(min_conc, min(estimated_conc, max_conc))
            
            # Apply function-based constraints
            if inci_entry:
                for function in inci_entry.function:
                    if function in self.concentration_estimation_rules:
                        func_min, func_max = self.concentration_estimation_rules[function]
                        estimated_conc = max(func_min, min(estimated_conc, func_max))
            
            estimated_concentrations[normalized_name] = estimated_conc
        
        return estimated_concentrations
    
    def filter_by_regulatory_compliance(self, ingredients: List[str], 
                                      region: RegulationRegion) -> List[str]:
        """Filter ingredients based on regulatory compliance"""
        compliant_ingredients = []
        regional_limits = self.regulatory_limits.get(region, {})
        
        for ingredient in ingredients:
            normalized_name = self.normalize_ingredient_name(ingredient)
            inci_entry = self.inci_database.get(normalized_name)
            
            # Check if ingredient is allowed in region
            if inci_entry:
                if region in inci_entry.regulatory_status:
                    status = inci_entry.regulatory_status[region]
                    if status in ["approved", "allowed"]:
                        compliant_ingredients.append(ingredient)
                else:
                    # If no specific status, assume allowed but check limits
                    compliant_ingredients.append(ingredient)
            else:
                # Unknown ingredient - flag for review but don't exclude
                compliant_ingredients.append(ingredient)
        
        return compliant_ingredients
    
    def reduce_search_space_by_inci_subset(self, target_inci_list: List[str], 
                                         candidate_formulations: List[ProductFormulation]) -> List[ProductFormulation]:
        """
        Reduce search space to formulations whose INCI lists are subsets of target.
        
        This enables finding formulations that can be created using only ingredients
        from a target product's INCI list.
        """
        target_ingredients = set(self.normalize_ingredient_name(name) for name in target_inci_list)
        compatible_formulations = []
        
        for formulation in candidate_formulations:
            formulation_ingredients = set(
                self.normalize_ingredient_name(name) for name in formulation.inci_list
            )
            
            # Check if formulation ingredients are subset of target
            if formulation_ingredients.issubset(target_ingredients):
                compatible_formulations.append(formulation)
        
        return compatible_formulations
    
    def generate_optimized_inci_combinations(self, available_ingredients: List[str],
                                           target_functions: List[str],
                                           region: RegulationRegion,
                                           max_ingredients: int = 10) -> List[List[str]]:
        """
        Generate optimized INCI combinations for target functions.
        
        Uses functional requirements and regulatory constraints to generate
        viable ingredient combinations.
        """
        # Filter for regulatory compliance
        compliant_ingredients = self.filter_by_regulatory_compliance(available_ingredients, region)
        
        # Group ingredients by function
        ingredients_by_function = defaultdict(list)
        for ingredient in compliant_ingredients:
            normalized_name = self.normalize_ingredient_name(ingredient)
            inci_entry = self.inci_database.get(normalized_name)
            
            if inci_entry:
                for function in inci_entry.function:
                    ingredients_by_function[function].append(ingredient)
        
        # Generate combinations that cover target functions
        combinations = []
        
        # Start with base formulation (water + preservative)
        base_ingredients = ["AQUA"]
        
        # Add preservative
        preservatives = ingredients_by_function.get("preservative", [])
        if preservatives:
            base_ingredients.extend(preservatives[:1])  # Use first available preservative
        
        # Add ingredients for each target function
        functional_ingredients = []
        for function in target_functions:
            if function in ingredients_by_function:
                # Select best ingredient for this function (could be enhanced with scoring)
                best_ingredient = ingredients_by_function[function][0]
                functional_ingredients.append(best_ingredient)
        
        # Combine base + functional ingredients
        if len(base_ingredients + functional_ingredients) <= max_ingredients:
            combinations.append(base_ingredients + functional_ingredients)
        
        # Generate alternative combinations with different ingredient selections
        for i in range(min(3, len(target_functions))):  # Generate up to 3 alternatives
            alt_ingredients = base_ingredients.copy()
            
            for function in target_functions:
                available_for_function = ingredients_by_function.get(function, [])
                if len(available_for_function) > i + 1:
                    alt_ingredients.append(available_for_function[i + 1])
                elif available_for_function:
                    alt_ingredients.append(available_for_function[0])
            
            if len(alt_ingredients) <= max_ingredients and alt_ingredients not in combinations:
                combinations.append(alt_ingredients)
        
        return combinations[:5]  # Return top 5 combinations
    
    def create_atomspace_from_inci_formulation(self, formulation: ProductFormulation) -> List[Atom]:
        """Create AtomSpace representation from INCI formulation"""
        ingredient_atoms = []
        estimated_concentrations = self.estimate_concentrations_from_inci(formulation.inci_list)
        
        for inci_name in formulation.inci_list:
            normalized_name = self.normalize_ingredient_name(inci_name)
            inci_entry = self.inci_database.get(normalized_name)
            
            # Create ingredient atom
            properties = {
                "inci_name": normalized_name,
                "concentration": estimated_concentrations.get(normalized_name, 1.0),
                "regulatory_region": formulation.region.value
            }
            
            if inci_entry:
                properties.update({
                    "cas_number": inci_entry.cas_number,
                    "functions": inci_entry.function,
                    "concentration_range": inci_entry.concentration_range
                })
            
            atom = self.atomspace.create_atom(
                AtomType.INGREDIENT_NODE,
                normalized_name,
                properties=properties
            )
            ingredient_atoms.append(atom)
        
        # Create formulation atom and links
        formulation_atom = self.atomspace.create_atom(
            AtomType.FORMULATION_NODE,
            formulation.product_name,
            properties={
                "product_type": formulation.product_type,
                "region": formulation.region.value,
                "total_ingredients": len(formulation.inci_list)
            }
        )
        
        # Link ingredients to formulation
        for ingredient_atom in ingredient_atoms:
            self.atomspace.create_atom(
                AtomType.CONTAINS_LINK,
                f"contains_{formulation.product_name}_{ingredient_atom.name}",
                outgoing=[formulation_atom, ingredient_atom]
            )
        
        return ingredient_atoms
    
    def optimize_formulation_for_constraints(self, base_formulation: List[str],
                                           constraints: Dict[str, Any],
                                           region: RegulationRegion) -> Dict[str, Any]:
        """
        Optimize formulation to meet specified constraints.
        
        Returns optimization results including modified INCI list and compliance status.
        """
        optimization_result = {
            "original_inci": base_formulation.copy(),
            "optimized_inci": [],
            "concentration_adjustments": {},
            "compliance_status": {},
            "optimization_score": 0.0
        }
        
        # Estimate initial concentrations
        initial_concentrations = self.estimate_concentrations_from_inci(base_formulation)
        
        # Apply constraints and optimize
        optimized_concentrations = {}
        compliance_issues = []
        
        for ingredient in base_formulation:
            normalized_name = self.normalize_ingredient_name(ingredient)
            initial_conc = initial_concentrations.get(normalized_name, 1.0)
            
            # Apply regulatory limits
            regional_limits = self.regulatory_limits.get(region, {})
            if normalized_name in regional_limits:
                max_allowed = regional_limits[normalized_name]
                if initial_conc > max_allowed:
                    optimized_concentrations[normalized_name] = max_allowed
                    optimization_result["concentration_adjustments"][normalized_name] = {
                        "original": initial_conc,
                        "adjusted": max_allowed,
                        "reason": f"Regulatory limit for {region.value}"
                    }
                else:
                    optimized_concentrations[normalized_name] = initial_conc
            else:
                optimized_concentrations[normalized_name] = initial_conc
            
            # Check INCI database constraints
            inci_entry = self.inci_database.get(normalized_name)
            if inci_entry and inci_entry.concentration_range:
                min_conc, max_conc = inci_entry.concentration_range
                current_conc = optimized_concentrations[normalized_name]
                
                if current_conc < min_conc:
                    optimized_concentrations[normalized_name] = min_conc
                    optimization_result["concentration_adjustments"][normalized_name] = {
                        "original": current_conc,
                        "adjusted": min_conc,
                        "reason": "Below minimum effective concentration"
                    }
                elif current_conc > max_conc:
                    optimized_concentrations[normalized_name] = max_conc
                    optimization_result["concentration_adjustments"][normalized_name] = {
                        "original": current_conc,
                        "adjusted": max_conc,
                        "reason": "Above maximum safe concentration"
                    }
        
        # Rebuild INCI list based on optimized concentrations
        sorted_ingredients = sorted(
            optimized_concentrations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        optimization_result["optimized_inci"] = [name for name, _ in sorted_ingredients]
        optimization_result["compliance_status"] = {
            "regulatory_compliant": len(compliance_issues) == 0,
            "issues": compliance_issues,
            "region": region.value
        }
        
        # Calculate optimization score (simplified)
        total_adjustments = len(optimization_result["concentration_adjustments"])
        optimization_result["optimization_score"] = max(0.0, 1.0 - (total_adjustments * 0.1))
        
        return optimization_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the INCI optimizer"""
        return {
            "inci_database_size": len(self.inci_database),
            "trade_name_mappings": len(self.trade_name_mapping),
            "regulatory_regions": len(self.regulatory_limits),
            "concentration_estimation_rules": len(self.concentration_estimation_rules),
            "supported_functions": list(set(
                func for entry in self.inci_database.values() 
                for func in entry.function
            ))
        }
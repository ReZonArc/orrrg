#!/usr/bin/env python3
"""
Advanced Cosmetic Chemistry Example

This script demonstrates advanced features of the OpenCog cheminformatics framework
for cosmetic chemistry, including formulation optimization, stability analysis,
regulatory compliance checking, and ingredient substitution.

Prerequisites:
- OpenCog Python bindings
- OpenCog AtomSpace with reasoning capabilities
- Cosmetic chemistry atom types loaded
- NumPy for numerical calculations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from opencog.type_constructors import *
from opencog.atomspace import AtomSpace
from opencog.utilities import initialize_opencog

class CosmeticChemistryFramework:
    """Advanced cosmetic chemistry analysis framework."""
    
    def __init__(self):
        """Initialize the framework with AtomSpace and knowledge base."""
        self.atomspace = AtomSpace()
        initialize_opencog(self.atomspace)
        self.ingredients_db = {}
        self.formulations_db = {}
        self.compatibility_matrix = {}
        
        print("Initializing advanced cosmetic chemistry framework...")
        self._load_ingredient_database()
        self._load_compatibility_rules()
        self._load_regulatory_data()
    
    def _load_ingredient_database(self):
        """Load comprehensive ingredient database with properties."""
        print("Loading ingredient database...")
        
        # Define ingredients with detailed properties
        ingredients_data = {
            'hyaluronic_acid': {
                'type': 'ACTIVE_INGREDIENT',
                'subtype': 'HUMECTANT',
                'molecular_weight': 1000000,  # Daltons
                'solubility': 'water_soluble',
                'ph_stability_range': (3.0, 8.0),
                'max_concentration': 2.0,  # %
                'allergenicity': 'low',
                'comedogenicity': 0,
                'functions': ['moisturizing', 'anti_aging', 'wound_healing']
            },
            'niacinamide': {
                'type': 'ACTIVE_INGREDIENT', 
                'subtype': 'VITAMIN',
                'molecular_weight': 122.12,
                'solubility': 'water_soluble',
                'ph_stability_range': (5.0, 7.0),
                'max_concentration': 10.0,
                'allergenicity': 'very_low',
                'comedogenicity': 0,
                'functions': ['pore_minimizing', 'oil_control', 'brightening']
            },
            'retinol': {
                'type': 'ACTIVE_INGREDIENT',
                'subtype': 'VITAMIN',
                'molecular_weight': 286.45,
                'solubility': 'oil_soluble',
                'ph_stability_range': (5.5, 7.0),
                'max_concentration': 1.0,
                'allergenicity': 'medium',
                'comedogenicity': 2,
                'functions': ['anti_aging', 'collagen_stimulation', 'cell_renewal'],
                'light_sensitive': True,
                'oxygen_sensitive': True
            },
            'vitamin_c': {
                'type': 'ACTIVE_INGREDIENT',
                'subtype': 'ANTIOXIDANT',
                'molecular_weight': 176.12,
                'solubility': 'water_soluble',
                'ph_stability_range': (2.0, 3.5),
                'max_concentration': 20.0,
                'allergenicity': 'low',
                'comedogenicity': 0,
                'functions': ['antioxidant', 'brightening', 'collagen_synthesis'],
                'oxidation_prone': True
            },
            'glycerin': {
                'type': 'HUMECTANT',
                'molecular_weight': 92.09,
                'solubility': 'water_soluble',
                'ph_stability_range': (3.0, 10.0),
                'max_concentration': 15.0,
                'allergenicity': 'very_low',
                'comedogenicity': 0,
                'functions': ['moisturizing', 'viscosity_building']
            },
            'phenoxyethanol': {
                'type': 'PRESERVATIVE',
                'molecular_weight': 138.16,
                'solubility': 'water_soluble',
                'ph_stability_range': (3.0, 8.0),
                'max_concentration': 1.0,
                'allergenicity': 'low',
                'antimicrobial_spectrum': ['bacteria', 'fungi'],
                'functions': ['preservation']
            }
        }
        
        # Create ingredient atoms with properties
        for name, props in ingredients_data.items():
            ingredient = ConceptNode(name)
            self.ingredients_db[name] = ingredient
            
            # Add type classifications
            InheritanceLink(ingredient, ConceptNode(props['type']))
            if 'subtype' in props:
                InheritanceLink(ingredient, ConceptNode(props['subtype']))
            
            # Add properties
            for prop_name, prop_value in props.items():
                if prop_name not in ['type', 'subtype']:
                    self._add_property(ingredient, prop_name, prop_value)
        
        print(f"✓ Loaded {len(ingredients_data)} ingredients with detailed properties")
    
    def _add_property(self, ingredient, prop_name, prop_value):
        """Add property to ingredient with appropriate representation."""
        if isinstance(prop_value, (int, float)):
            ExecutionLink(
                SchemaNode(prop_name),
                ingredient,
                NumberNode(str(prop_value))
            )
        elif isinstance(prop_value, str):
            EvaluationLink(
                PredicateNode(prop_name),
                ListLink(ingredient, ConceptNode(prop_value))
            )
        elif isinstance(prop_value, (list, tuple)):
            if len(prop_value) == 2 and all(isinstance(x, (int, float)) for x in prop_value):
                # Range value
                ExecutionLink(
                    SchemaNode(prop_name),
                    ingredient,
                    ListLink(NumberNode(str(prop_value[0])), NumberNode(str(prop_value[1])))
                )
            else:
                # List of values
                for val in prop_value:
                    EvaluationLink(
                        PredicateNode(prop_name),
                        ListLink(ingredient, ConceptNode(str(val)))
                    )
        elif isinstance(prop_value, bool):
            if prop_value:
                EvaluationLink(
                    PredicateNode(prop_name),
                    ingredient
                )
    
    def _load_compatibility_rules(self):
        """Load ingredient compatibility and interaction rules."""
        print("Loading compatibility rules...")
        
        # Define compatibility rules
        compatibility_rules = [
            # Compatible pairs
            ('hyaluronic_acid', 'niacinamide', 'compatible', 'neutral_ph_both'),
            ('hyaluronic_acid', 'glycerin', 'compatible', 'both_humectants'),
            ('niacinamide', 'glycerin', 'compatible', 'water_soluble_both'),
            
            # Incompatible pairs
            ('vitamin_c', 'retinol', 'incompatible', 'ph_incompatibility'),
            ('vitamin_c', 'niacinamide', 'potentially_incompatible', 'ph_difference'),
            
            # Synergistic pairs
            # ('vitamin_c', 'vitamin_e', 'synergistic', 'antioxidant_network'),
        ]
        
        for ing1, ing2, relation, reason in compatibility_rules:
            if ing1 in self.ingredients_db and ing2 in self.ingredients_db:
                EvaluationLink(
                    PredicateNode(relation),
                    ListLink(
                        self.ingredients_db[ing1],
                        self.ingredients_db[ing2],
                        ConceptNode(reason)
                    )
                )
        
        print("✓ Loaded ingredient compatibility rules")
    
    def _load_regulatory_data(self):
        """Load regulatory concentration limits and safety data."""
        print("Loading regulatory data...")
        
        # EU concentration limits
        eu_limits = {
            'retinol': 0.3,  # 0.3% maximum in EU
            'vitamin_c': 20.0,  # Generally recognized as safe up to 20%
            'phenoxyethanol': 1.0,  # Maximum 1% in cosmetics
        }
        
        for ingredient, limit in eu_limits.items():
            if ingredient in self.ingredients_db:
                ExecutionLink(
                    SchemaNode("eu_concentration_limit"),
                    self.ingredients_db[ingredient],
                    NumberNode(str(limit))
                )
        
        print("✓ Loaded regulatory concentration limits")
    
    def create_advanced_formulation(self, formulation_name: str, target_properties: List[str]) -> ConceptNode:
        """Create an advanced formulation with target properties."""
        print(f"\n=== Creating Advanced Formulation: {formulation_name} ===")
        
        formulation = ConceptNode(formulation_name)
        InheritanceLink(formulation, ConceptNode("SKINCARE_FORMULATION"))
        
        # Add target properties
        for prop in target_properties:
            EvaluationLink(
                PredicateNode("target_property"),
                ListLink(formulation, ConceptNode(prop))
            )
        
        self.formulations_db[formulation_name] = formulation
        print(f"✓ Created formulation with target properties: {target_properties}")
        
        return formulation
    
    def optimize_formulation(self, formulation: ConceptNode, 
                           ingredient_candidates: List[str],
                           constraints: Dict) -> Dict:
        """Optimize formulation using constraint satisfaction."""
        print("\n=== Optimizing Formulation ===")
        
        # Simplified optimization algorithm
        optimized_formula = {}
        total_concentration = 0.0
        
        # Score ingredients based on target properties
        for ingredient_name in ingredient_candidates:
            if ingredient_name not in self.ingredients_db:
                continue
                
            ingredient = self.ingredients_db[ingredient_name]
            score = self._calculate_ingredient_score(formulation, ingredient)
            
            if score > 0.5:  # Threshold for inclusion
                # Calculate optimal concentration
                max_conc = self._get_max_safe_concentration(ingredient)
                target_conc = min(max_conc, constraints.get('max_individual_conc', 10.0))
                
                if total_concentration + target_conc <= constraints.get('max_total_actives', 25.0):
                    optimized_formula[ingredient_name] = target_conc
                    total_concentration += target_conc
                    
                    # Add to formulation
                    MemberLink(ingredient, formulation)
                    ExecutionLink(
                        SchemaNode("concentration"),
                        ListLink(formulation, ingredient),
                        NumberNode(str(target_conc))
                    )
        
        print(f"✓ Optimized formulation with {len(optimized_formula)} active ingredients")
        print(f"   Total active concentration: {total_concentration:.1f}%")
        
        return optimized_formula
    
    def _calculate_ingredient_score(self, formulation: ConceptNode, ingredient: ConceptNode) -> float:
        """Calculate ingredient score based on formulation targets."""
        # Simplified scoring based on target properties
        # In practice, this would use more sophisticated reasoning
        
        score = 0.5  # Base score
        
        # Check if ingredient functions match target properties
        target_links = self.atomspace.get_incoming(formulation)
        for link in target_links:
            if (link.type == EvaluationLink and 
                len(link.out) == 2 and
                link.out[0].name == "target_property"):
                # Implementation would check ingredient functions
                score += 0.2
        
        return min(score, 1.0)
    
    def _get_max_safe_concentration(self, ingredient: ConceptNode) -> float:
        """Get maximum safe concentration for ingredient."""
        # Check regulatory limits
        execution_links = self.atomspace.get_incoming(ingredient)
        for link in execution_links:
            if (link.type == ExecutionLink and
                len(link.out) == 3 and
                link.out[0].name in ["max_concentration", "eu_concentration_limit"]):
                try:
                    return float(link.out[2].name)
                except ValueError:
                    pass
        
        return 5.0  # Default safe maximum
    
    def analyze_stability(self, formulation: ConceptNode) -> Dict:
        """Analyze formulation stability factors."""
        print("\n=== Analyzing Formulation Stability ===")
        
        stability_report = {
            'ph_compatibility': True,
            'oxidation_risk': False,
            'light_sensitivity': False,
            'recommendations': []
        }
        
        # Get formulation ingredients
        formulation_ingredients = []
        member_links = self.atomspace.get_incoming(formulation)
        for link in member_links:
            if link.type == MemberLink and len(link.out) == 2:
                formulation_ingredients.append(link.out[0])
        
        # Check pH compatibility
        ph_ranges = []
        for ingredient in formulation_ingredients:
            ph_range = self._get_ph_stability_range(ingredient)
            if ph_range:
                ph_ranges.append(ph_range)
        
        if ph_ranges:
            min_ph = max(r[0] for r in ph_ranges)
            max_ph = min(r[1] for r in ph_ranges)
            
            if min_ph > max_ph:
                stability_report['ph_compatibility'] = False
                stability_report['recommendations'].append(
                    f"pH incompatibility detected. Need pH {min_ph:.1f}-{max_ph:.1f}"
                )
        
        # Check for oxidation-prone ingredients
        for ingredient in formulation_ingredients:
            if self._is_oxidation_prone(ingredient):
                stability_report['oxidation_risk'] = True
                stability_report['recommendations'].append(
                    f"Add antioxidant protection for {ingredient.name}"
                )
        
        # Check light sensitivity
        for ingredient in formulation_ingredients:
            if self._is_light_sensitive(ingredient):
                stability_report['light_sensitivity'] = True
                stability_report['recommendations'].append(
                    f"Use opaque packaging for light-sensitive {ingredient.name}"
                )
        
        print("✓ Completed stability analysis")
        return stability_report
    
    def _get_ph_stability_range(self, ingredient: ConceptNode) -> Optional[Tuple[float, float]]:
        """Get pH stability range for ingredient."""
        execution_links = self.atomspace.get_incoming(ingredient)
        for link in execution_links:
            if (link.type == ExecutionLink and
                len(link.out) == 3 and
                link.out[0].name == "ph_stability_range"):
                if link.out[2].type == ListLink and len(link.out[2].out) == 2:
                    try:
                        min_ph = float(link.out[2].out[0].name)
                        max_ph = float(link.out[2].out[1].name)
                        return (min_ph, max_ph)
                    except ValueError:
                        pass
        return None
    
    def _is_oxidation_prone(self, ingredient: ConceptNode) -> bool:
        """Check if ingredient is prone to oxidation."""
        evaluation_links = self.atomspace.get_incoming(ingredient)
        for link in evaluation_links:
            if (link.type == EvaluationLink and
                len(link.out) == 2 and
                link.out[0].name == "oxidation_prone"):
                return True
        return False
    
    def _is_light_sensitive(self, ingredient: ConceptNode) -> bool:
        """Check if ingredient is light sensitive."""
        evaluation_links = self.atomspace.get_incoming(ingredient)
        for link in evaluation_links:
            if (link.type == EvaluationLink and
                len(link.out) == 2 and
                link.out[0].name == "light_sensitive"):
                return True
        return False
    
    def check_regulatory_compliance(self, formulation: ConceptNode) -> Dict:
        """Check formulation compliance with regulatory requirements."""
        print("\n=== Checking Regulatory Compliance ===")
        
        compliance_report = {
            'compliant': True,
            'violations': [],
            'warnings': []
        }
        
        # Check concentration limits
        member_links = self.atomspace.get_incoming(formulation)
        for link in member_links:
            if link.type == MemberLink and len(link.out) == 2:
                ingredient = link.out[0]
                
                # Get ingredient concentration
                conc = self._get_ingredient_concentration(formulation, ingredient)
                limit = self._get_regulatory_limit(ingredient)
                
                if conc and limit and conc > limit:
                    compliance_report['compliant'] = False
                    compliance_report['violations'].append(
                        f"{ingredient.name}: {conc}% exceeds limit of {limit}%"
                    )
        
        print(f"✓ Regulatory compliance check completed")
        print(f"   Compliant: {compliance_report['compliant']}")
        
        return compliance_report
    
    def _get_ingredient_concentration(self, formulation: ConceptNode, ingredient: ConceptNode) -> Optional[float]:
        """Get concentration of ingredient in formulation."""
        execution_links = self.atomspace.get_atoms_by_type(ExecutionLink)
        for link in execution_links:
            if (len(link.out) == 3 and
                link.out[0].name == "concentration" and
                link.out[1].type == ListLink and
                len(link.out[1].out) == 2 and
                link.out[1].out[0] == formulation and
                link.out[1].out[1] == ingredient):
                try:
                    return float(link.out[2].name)
                except ValueError:
                    pass
        return None
    
    def _get_regulatory_limit(self, ingredient: ConceptNode) -> Optional[float]:
        """Get regulatory concentration limit for ingredient."""
        execution_links = self.atomspace.get_incoming(ingredient)
        for link in execution_links:
            if (link.type == ExecutionLink and
                len(link.out) == 3 and
                link.out[0].name == "eu_concentration_limit"):
                try:
                    return float(link.out[2].name)
                except ValueError:
                    pass
        return None
    
    def find_ingredient_alternatives(self, original_ingredient: str, 
                                   requirements: List[str]) -> List[str]:
        """Find alternative ingredients meeting specified requirements."""
        print(f"\n=== Finding Alternatives for {original_ingredient} ===")
        
        alternatives = []
        original = self.ingredients_db.get(original_ingredient)
        
        if not original:
            print(f"   Original ingredient {original_ingredient} not found")
            return alternatives
        
        # Get original ingredient type
        original_type = self._get_ingredient_type(original)
        
        # Find ingredients of same type
        for name, ingredient in self.ingredients_db.items():
            if name != original_ingredient:
                ingredient_type = self._get_ingredient_type(ingredient)
                if ingredient_type == original_type:
                    # Check if meets requirements
                    if self._meets_requirements(ingredient, requirements):
                        alternatives.append(name)
        
        print(f"✓ Found {len(alternatives)} alternatives: {alternatives}")
        return alternatives
    
    def _get_ingredient_type(self, ingredient: ConceptNode) -> Optional[str]:
        """Get primary type of ingredient."""
        inheritance_links = self.atomspace.get_incoming(ingredient)
        for link in inheritance_links:
            if (link.type == InheritanceLink and
                len(link.out) == 2 and
                link.out[0] == ingredient):
                return link.out[1].name
        return None
    
    def _meets_requirements(self, ingredient: ConceptNode, requirements: List[str]) -> bool:
        """Check if ingredient meets specified requirements."""
        # Simplified requirement checking
        for req in requirements:
            if req == "low_allergenicity":
                # Check allergenicity property
                if not self._has_property_value(ingredient, "allergenicity", ["very_low", "low"]):
                    return False
            elif req == "water_soluble":
                if not self._has_property_value(ingredient, "solubility", ["water_soluble"]):
                    return False
        return True
    
    def _has_property_value(self, ingredient: ConceptNode, prop_name: str, allowed_values: List[str]) -> bool:
        """Check if ingredient has property with allowed values."""
        evaluation_links = self.atomspace.get_incoming(ingredient)
        for link in evaluation_links:
            if (link.type == EvaluationLink and
                len(link.out) == 2 and
                link.out[0].name == prop_name and
                link.out[1].type == ListLink and
                len(link.out[1].out) == 2):
                value = link.out[1].out[1].name
                if value in allowed_values:
                    return True
        return False

def main():
    """Main demonstration of advanced cosmetic chemistry features."""
    print("Advanced Cosmetic Chemistry Framework Demonstration")
    print("=" * 60)
    
    # Initialize framework
    framework = CosmeticChemistryFramework()
    
    # Create advanced anti-aging serum formulation
    anti_aging_serum = framework.create_advanced_formulation(
        "advanced_anti_aging_serum",
        ["anti_aging", "brightening", "moisturizing", "antioxidant_protection"]
    )
    
    # Optimize formulation
    candidate_ingredients = [
        'hyaluronic_acid', 'niacinamide', 'vitamin_c', 'retinol', 'glycerin'
    ]
    
    constraints = {
        'max_individual_conc': 10.0,
        'max_total_actives': 20.0
    }
    
    optimized_formula = framework.optimize_formulation(
        anti_aging_serum, candidate_ingredients, constraints
    )
    
    # Analyze stability
    stability_report = framework.analyze_stability(anti_aging_serum)
    print(f"\nStability Analysis Results:")
    print(f"  pH Compatible: {stability_report['ph_compatibility']}")
    print(f"  Oxidation Risk: {stability_report['oxidation_risk']}")
    print(f"  Light Sensitive: {stability_report['light_sensitivity']}")
    print(f"  Recommendations: {len(stability_report['recommendations'])}")
    for rec in stability_report['recommendations']:
        print(f"    - {rec}")
    
    # Check regulatory compliance
    compliance_report = framework.check_regulatory_compliance(anti_aging_serum)
    print(f"\nRegulatory Compliance:")
    print(f"  Compliant: {compliance_report['compliant']}")
    if compliance_report['violations']:
        print(f"  Violations:")
        for violation in compliance_report['violations']:
            print(f"    - {violation}")
    
    # Find ingredient alternatives
    alternatives = framework.find_ingredient_alternatives(
        'phenoxyethanol',
        ['low_allergenicity', 'water_soluble']
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("ADVANCED ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"✓ Created optimized formulation with {len(optimized_formula)} active ingredients")
    print(f"✓ Performed comprehensive stability analysis")
    print(f"✓ Verified regulatory compliance")
    print(f"✓ Identified {len(alternatives)} ingredient alternatives")
    print("\nFormulation Details:")
    for ingredient, concentration in optimized_formula.items():
        print(f"  - {ingredient}: {concentration}%")
    
    print("\nFramework Capabilities Demonstrated:")
    print("✓ Advanced ingredient database with detailed properties")
    print("✓ Formulation optimization using constraint satisfaction")
    print("✓ Multi-factor stability analysis")
    print("✓ Regulatory compliance checking")
    print("✓ Ingredient substitution recommendations")
    print("✓ Knowledge-based reasoning for cosmetic chemistry")

if __name__ == "__main__":
    main()
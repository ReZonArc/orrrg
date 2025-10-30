#!/usr/bin/env python3
"""
Advanced Cosmetic Chemistry Example

This example demonstrates advanced features of the cosmetic chemistry framework
including formulation optimization, stability prediction, regulatory compliance
checking, and ingredient substitution analysis.

Requirements:
- OpenCog AtomSpace with bioscience extensions
- Python OpenCog bindings
- NumPy (for calculations)

Usage:
    python3 cosmetic_chemistry_example.py

Author: OpenCog Cosmetic Chemistry Framework
License: AGPL-3.0
"""

import sys
from opencog.atomspace import AtomSpace, types
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("⚠ NumPy not available - some calculations will be simplified")
    HAS_NUMPY = False

class CosmeticFormulationAnalyzer:
    """Advanced analyzer for cosmetic formulations."""
    
    def __init__(self, atomspace):
        self.atomspace = atomspace
        self.ingredient_database = self._initialize_ingredient_database()
        self.compatibility_rules = self._initialize_compatibility_rules()
        self.regulatory_limits = self._initialize_regulatory_limits()
    
    def _initialize_ingredient_database(self):
        """Initialize comprehensive ingredient database."""
        print("🔬 Initializing ingredient database...")
        
        ingredients = {
            # Active ingredients
            'hyaluronic_acid': {
                'type': 'ACTIVE_INGREDIENT',
                'functions': ['HUMECTANT'],
                'max_concentration': 5.0,
                'optimal_ph': (4.0, 7.0),
                'stability_factors': ['temperature_sensitive', 'light_stable'],
                'cost_per_kg': 2500.0,
                'allergen_risk': 'low'
            },
            'niacinamide': {
                'type': 'ACTIVE_INGREDIENT', 
                'functions': [],
                'max_concentration': 10.0,
                'optimal_ph': (5.0, 7.0),
                'stability_factors': ['temperature_stable', 'light_stable'],
                'cost_per_kg': 45.0,
                'allergen_risk': 'low'
            },
            'retinol': {
                'type': 'ACTIVE_INGREDIENT',
                'functions': [],
                'max_concentration': 1.0,
                'optimal_ph': (5.5, 6.5),
                'stability_factors': ['temperature_sensitive', 'light_sensitive', 'oxygen_sensitive'],
                'cost_per_kg': 8000.0,
                'allergen_risk': 'medium'
            },
            'vitamin_c': {
                'type': 'ACTIVE_INGREDIENT',
                'functions': ['ANTIOXIDANT'],
                'max_concentration': 20.0,
                'optimal_ph': (3.0, 4.0),
                'stability_factors': ['temperature_sensitive', 'light_sensitive', 'oxygen_sensitive'],
                'cost_per_kg': 120.0,
                'allergen_risk': 'low'
            },
            'salicylic_acid': {
                'type': 'ACTIVE_INGREDIENT',
                'functions': [],
                'max_concentration': 2.0,
                'optimal_ph': (3.0, 4.0),
                'stability_factors': ['temperature_stable', 'light_stable'],
                'cost_per_kg': 25.0,
                'allergen_risk': 'medium'
            },
            
            # Functional ingredients
            'glycerin': {
                'type': 'HUMECTANT',
                'functions': [],
                'max_concentration': 20.0,
                'optimal_ph': (4.0, 8.0),
                'stability_factors': ['temperature_stable', 'light_stable'],
                'cost_per_kg': 3.0,
                'allergen_risk': 'low'
            },
            'cetyl_alcohol': {
                'type': 'EMULSIFIER',
                'functions': ['EMOLLIENT'],
                'max_concentration': 10.0,
                'optimal_ph': (4.0, 8.0),
                'stability_factors': ['temperature_stable', 'light_stable'],
                'cost_per_kg': 8.0,
                'allergen_risk': 'low'
            },
            'phenoxyethanol': {
                'type': 'PRESERVATIVE',
                'functions': [],
                'max_concentration': 1.0,
                'optimal_ph': (4.0, 8.0),
                'stability_factors': ['temperature_stable', 'light_stable'],
                'cost_per_kg': 12.0,
                'allergen_risk': 'low'
            },
            'xanthan_gum': {
                'type': 'THICKENER',
                'functions': [],
                'max_concentration': 2.0,
                'optimal_ph': (4.0, 10.0),
                'stability_factors': ['temperature_stable', 'light_stable'],
                'cost_per_kg': 15.0,
                'allergen_risk': 'low'
            },
            'squalane': {
                'type': 'EMOLLIENT',
                'functions': [],
                'max_concentration': 30.0,
                'optimal_ph': (4.0, 8.0),
                'stability_factors': ['temperature_stable', 'light_stable', 'oxygen_stable'],
                'cost_per_kg': 35.0,
                'allergen_risk': 'low'
            },
            'vitamin_e': {
                'type': 'ANTIOXIDANT',
                'functions': [],
                'max_concentration': 1.0,
                'optimal_ph': (4.0, 8.0),
                'stability_factors': ['temperature_stable', 'light_sensitive'],
                'cost_per_kg': 18.0,
                'allergen_risk': 'low'
            }
        }
        
        # Create atoms for each ingredient
        for name, data in ingredients.items():
            ingredient_node = ConceptNode(name)
            
            # Type classification
            InheritanceLink(ingredient_node, ConceptNode(data['type']))
            
            # Additional functions
            for function in data['functions']:
                InheritanceLink(ingredient_node, ConceptNode(function))
            
            # Properties
            for prop, value in data.items():
                if prop not in ['type', 'functions']:
                    if isinstance(value, (int, float)):
                        EvaluationLink(
                            PredicateNode(prop),
                            ListLink(ingredient_node, NumberNode(str(value)))
                        )
                    elif isinstance(value, str):
                        EvaluationLink(
                            PredicateNode(prop),
                            ListLink(ingredient_node, ConceptNode(value))
                        )
                    elif isinstance(value, tuple):
                        EvaluationLink(
                            PredicateNode(prop),
                            ListLink(ingredient_node, 
                                   NumberNode(str(value[0])),
                                   NumberNode(str(value[1])))
                        )
                    elif isinstance(value, list):
                        for item in value:
                            EvaluationLink(
                                PredicateNode(prop),
                                ListLink(ingredient_node, ConceptNode(str(item)))
                            )
        
        print(f"  ✓ {len(ingredients)} ingredients loaded into database")
        return ingredients
    
    def _initialize_compatibility_rules(self):
        """Initialize ingredient compatibility rules."""
        print("🔗 Initializing compatibility rules...")
        
        # Compatible combinations
        compatible_pairs = [
            ('hyaluronic_acid', 'niacinamide', 'Enhanced hydration and barrier function'),
            ('niacinamide', 'glycerin', 'Complementary moisturizing effects'),
            ('vitamin_c', 'vitamin_e', 'Antioxidant synergy and stability'),
            ('retinol', 'squalane', 'Reduced irritation from emollient'),
            ('salicylic_acid', 'niacinamide', 'Balanced exfoliation and barrier support')
        ]
        
        # Incompatible combinations
        incompatible_pairs = [
            ('vitamin_c', 'retinol', 'pH incompatibility and instability'),
            ('vitamin_c', 'niacinamide', 'Potential interaction reducing efficacy'),
            ('salicylic_acid', 'retinol', 'Increased irritation risk'),
            ('retinol', 'hyaluronic_acid', 'pH sensitivity issues')
        ]
        
        # Synergistic combinations
        synergistic_pairs = [
            ('vitamin_c', 'vitamin_e', 'Enhanced antioxidant activity'),
            ('hyaluronic_acid', 'glycerin', 'Superior moisture retention'),
            ('cetyl_alcohol', 'glycerin', 'Improved emulsion stability')
        ]
        
        # Create compatibility links
        for ing1, ing2, description in compatible_pairs:
            if ing1 in self.ingredient_database and ing2 in self.ingredient_database:
                compatibility = EvaluationLink(
                    PredicateNode("compatible_with"),
                    ListLink(ConceptNode(ing1), ConceptNode(ing2))
                )
                
                EvaluationLink(
                    PredicateNode("interaction_description"),
                    ListLink(compatibility, ConceptNode(description))
                )
        
        # Create incompatibility links
        for ing1, ing2, reason in incompatible_pairs:
            if ing1 in self.ingredient_database and ing2 in self.ingredient_database:
                incompatibility = EvaluationLink(
                    PredicateNode("incompatible_with"),
                    ListLink(ConceptNode(ing1), ConceptNode(ing2))
                )
                
                EvaluationLink(
                    PredicateNode("incompatibility_reason"),
                    ListLink(incompatibility, ConceptNode(reason))
                )
        
        # Create synergy links
        for ing1, ing2, benefit in synergistic_pairs:
            if ing1 in self.ingredient_database and ing2 in self.ingredient_database:
                synergy = EvaluationLink(
                    PredicateNode("synergistic_with"),
                    ListLink(ConceptNode(ing1), ConceptNode(ing2))
                )
                
                EvaluationLink(
                    PredicateNode("synergy_benefit"),
                    ListLink(synergy, ConceptNode(benefit))
                )
        
        print(f"  ✓ Compatibility rules established")
        
        return {
            'compatible': compatible_pairs,
            'incompatible': incompatible_pairs,
            'synergistic': synergistic_pairs
        }
    
    def _initialize_regulatory_limits(self):
        """Initialize regulatory concentration limits."""
        print("⚖️ Initializing regulatory limits...")
        
        limits = {
            'EU': {
                'phenoxyethanol': 1.0,
                'retinol': 0.3,
                'salicylic_acid': 2.0,
                'vitamin_c': 20.0
            },
            'FDA': {
                'phenoxyethanol': 1.0,
                'retinol': 1.0,
                'salicylic_acid': 2.0,
                'vitamin_c': 20.0
            }
        }
        
        # Create regulatory limit atoms
        for region, ingredient_limits in limits.items():
            for ingredient, limit in ingredient_limits.items():
                EvaluationLink(
                    PredicateNode(f"regulatory_limit_{region}"),
                    ListLink(ConceptNode(ingredient), NumberNode(str(limit)))
                )
        
        print(f"  ✓ Regulatory limits for {len(limits)} regions loaded")
        return limits

def demonstrate_formulation_optimization(analyzer):
    """Demonstrate formulation optimization capabilities."""
    print("\n🔬 === Formulation Optimization ===")
    
    # Define target formulation goals
    formulation_goals = {
        'product_type': 'anti_aging_serum',
        'target_ph': 5.5,
        'target_viscosity': 'medium',
        'budget_per_100g': 5.0,
        'shelf_life_months': 24,
        'skin_types': ['all'],
        'key_benefits': ['hydration', 'anti_aging', 'barrier_support']
    }
    
    print("Formulation Goals:")
    for goal, value in formulation_goals.items():
        print(f"  • {goal}: {value}")
    
    # Create candidate formulation
    candidate_formulation = ConceptNode("anti_aging_serum_v1")
    InheritanceLink(candidate_formulation, ConceptNode("SKINCARE_FORMULATION"))
    
    # Define ingredient concentrations
    formulation_components = [
        ('hyaluronic_acid', 2.0),
        ('niacinamide', 5.0), 
        ('vitamin_c', 10.0),
        ('glycerin', 8.0),
        ('squalane', 5.0),
        ('phenoxyethanol', 0.8),
        ('xanthan_gum', 0.3)
    ]
    
    print(f"\nCandidate formulation: {candidate_formulation.name}")
    total_active_concentration = 0
    estimated_cost = 0
    
    for ingredient, concentration in formulation_components:
        # Add to formulation
        EvaluationLink(
            PredicateNode("concentration"),
            ListLink(candidate_formulation, ConceptNode(ingredient), NumberNode(str(concentration)))
        )
        
        # Calculate metrics
        if ingredient in analyzer.ingredient_database:
            ingredient_data = analyzer.ingredient_database[ingredient]
            if ingredient_data['type'] == 'ACTIVE_INGREDIENT':
                total_active_concentration += concentration
            
            # Estimate cost
            cost_per_kg = ingredient_data.get('cost_per_kg', 0)
            ingredient_cost = (concentration / 100) * (cost_per_kg / 1000)  # per 100g
            estimated_cost += ingredient_cost
        
        print(f"  • {ingredient}: {concentration}%")
    
    print(f"\nFormulation Analysis:")
    print(f"  • Total active concentration: {total_active_concentration}%")
    print(f"  • Estimated cost per 100g: ${estimated_cost:.2f}")
    print(f"  • Within budget: {'✓' if estimated_cost <= formulation_goals['budget_per_100g'] else '✗'}")
    
    # Analyze compatibility
    print(f"\nCompatibility Analysis:")
    compatibility_issues = []
    synergies = []
    
    ingredients = [comp[0] for comp in formulation_components]
    for i, ing1 in enumerate(ingredients):
        for ing2 in ingredients[i+1:]:
            # Check for incompatibilities
            for incomp_pair in analyzer.compatibility_rules['incompatible']:
                if (ing1, ing2) == incomp_pair[:2] or (ing2, ing1) == incomp_pair[:2]:
                    compatibility_issues.append((ing1, ing2, incomp_pair[2]))
            
            # Check for synergies
            for syn_pair in analyzer.compatibility_rules['synergistic']:
                if (ing1, ing2) == syn_pair[:2] or (ing2, ing1) == syn_pair[:2]:
                    synergies.append((ing1, ing2, syn_pair[2]))
    
    if compatibility_issues:
        print("  ⚠ Compatibility Issues:")
        for ing1, ing2, reason in compatibility_issues:
            print(f"    - {ing1} + {ing2}: {reason}")
    else:
        print("  ✓ No compatibility issues detected")
    
    if synergies:
        print("  ⚡ Synergistic Combinations:")
        for ing1, ing2, benefit in synergies:
            print(f"    - {ing1} + {ing2}: {benefit}")
    
    return candidate_formulation

def demonstrate_stability_prediction(analyzer, formulation):
    """Demonstrate stability prediction capabilities."""
    print("\n🧪 === Stability Prediction ===")
    
    # Analyze stability factors
    stability_risks = []
    stability_score = 100  # Start with perfect score
    
    print("Analyzing formulation stability factors:")
    
    # Get formulation ingredients
    formulation_ingredients = []
    for atom in analyzer.atomspace:
        if (atom.type == types.EvaluationLink and 
            len(atom.out) == 2 and 
            atom.out[0].name == "concentration" and
            atom.out[1].out[0] == formulation):
            ingredient_name = atom.out[1].out[1].name
            formulation_ingredients.append(ingredient_name)
    
    print(f"  Ingredients to analyze: {formulation_ingredients}")
    
    # Analyze each ingredient's stability factors
    for ingredient in formulation_ingredients:
        if ingredient in analyzer.ingredient_database:
            ingredient_data = analyzer.ingredient_database[ingredient]
            stability_factors = ingredient_data.get('stability_factors', [])
            
            print(f"  • {ingredient}:")
            for factor in stability_factors:
                if 'sensitive' in factor:
                    risk_level = 'High' if 'temperature_sensitive' in factor else 'Medium'
                    stability_risks.append((ingredient, factor, risk_level))
                    stability_score -= 10 if risk_level == 'High' else 5
                    print(f"    ⚠ {factor} ({risk_level} risk)")
                else:
                    print(f"    ✓ {factor}")
    
    # pH compatibility analysis
    print(f"\n  pH Compatibility Analysis:")
    ph_ranges = []
    for ingredient in formulation_ingredients:
        if ingredient in analyzer.ingredient_database:
            optimal_ph = analyzer.ingredient_database[ingredient].get('optimal_ph')
            if optimal_ph:
                ph_ranges.append((ingredient, optimal_ph))
    
    if ph_ranges:
        # Find overlapping pH range
        min_ph = max(ph_range[1][0] for ph_range in ph_ranges)
        max_ph = min(ph_range[1][1] for ph_range in ph_ranges)
        
        if min_ph <= max_ph:
            optimal_range = (min_ph, max_ph)
            print(f"    ✓ Compatible pH range: {optimal_range[0]:.1f} - {optimal_range[1]:.1f}")
            
            # Add pH property to formulation
            target_ph = (min_ph + max_ph) / 2
            EvaluationLink(
                PredicateNode("optimal_pH"),
                ListLink(formulation, NumberNode(f"{target_ph:.1f}"))
            )
        else:
            print(f"    ✗ pH incompatibility detected")
            stability_score -= 20
            stability_risks.append(("pH_incompatibility", "Multiple ingredients", "High"))
    
    # Stability recommendations
    print(f"\n  Stability Assessment:")
    print(f"    • Overall stability score: {stability_score}/100")
    
    if stability_score >= 80:
        print(f"    ✓ Excellent stability expected")
    elif stability_score >= 60:
        print(f"    ⚠ Good stability with proper storage")
    else:
        print(f"    ✗ Stability concerns - reformulation recommended")
    
    # Recommend stabilization strategies
    if stability_risks:
        print(f"\n  Stabilization Recommendations:")
        for ingredient, factor, risk_level in stability_risks:
            if 'light_sensitive' in factor:
                print(f"    • Use dark packaging for {ingredient}")
            elif 'temperature_sensitive' in factor:
                print(f"    • Store {ingredient} formulations below 25°C")
            elif 'oxygen_sensitive' in factor:
                print(f"    • Use antioxidants and airless packaging for {ingredient}")
    
    return stability_score

def demonstrate_regulatory_compliance(analyzer, formulation):
    """Demonstrate regulatory compliance checking."""
    print("\n⚖️ === Regulatory Compliance ===")
    
    regions = ['EU', 'FDA']
    compliance_results = {}
    
    for region in regions:
        print(f"\n{region} Compliance Check:")
        compliance_issues = []
        
        # Check concentration limits
        for atom in analyzer.atomspace:
            if (atom.type == types.EvaluationLink and 
                len(atom.out) == 2 and 
                atom.out[0].name == "concentration" and
                atom.out[1].out[0] == formulation):
                
                ingredient_name = atom.out[1].out[1].name
                concentration = float(atom.out[1].out[2].name)
                
                # Check against regulatory limits
                if (region in analyzer.regulatory_limits and 
                    ingredient_name in analyzer.regulatory_limits[region]):
                    
                    limit = analyzer.regulatory_limits[region][ingredient_name]
                    if concentration > limit:
                        compliance_issues.append(
                            f"{ingredient_name}: {concentration}% exceeds {region} limit of {limit}%"
                        )
                    else:
                        print(f"  ✓ {ingredient_name}: {concentration}% (limit: {limit}%)")
        
        # Check allergen declarations
        allergen_ingredients = []
        for atom in analyzer.atomspace:
            if (atom.type == types.EvaluationLink and 
                len(atom.out) == 2 and 
                atom.out[0].name == "concentration" and
                atom.out[1].out[0] == formulation):
                
                ingredient_name = atom.out[1].out[1].name
                if ingredient_name in analyzer.ingredient_database:
                    allergen_risk = analyzer.ingredient_database[ingredient_name].get('allergen_risk', 'low')
                    if allergen_risk in ['medium', 'high']:
                        allergen_ingredients.append((ingredient_name, allergen_risk))
        
        if allergen_ingredients:
            print(f"  ⚠ Allergen Declaration Required:")
            for ingredient, risk in allergen_ingredients:
                print(f"    - {ingredient} ({risk} risk)")
        
        # Overall compliance
        if compliance_issues:
            print(f"  ✗ Compliance Issues:")
            for issue in compliance_issues:
                print(f"    - {issue}")
            compliance_results[region] = False
        else:
            print(f"  ✓ {region} compliant")
            compliance_results[region] = True
    
    return compliance_results

def demonstrate_ingredient_substitution(analyzer):
    """Demonstrate ingredient substitution analysis."""
    print("\n🔄 === Ingredient Substitution Analysis ===")
    
    # Scenario: Replace vitamin C due to stability issues
    print("Scenario: Replace vitamin C due to stability concerns")
    print("Requirements: Antioxidant activity, stable, cost-effective")
    
    # Find potential substitutes
    substitute_candidates = []
    for ingredient, data in analyzer.ingredient_database.items():
        if (data['type'] == 'ANTIOXIDANT' or 'ANTIOXIDANT' in data.get('functions', [])):
            stability_factors = data.get('stability_factors', [])
            is_stable = not any('sensitive' in factor for factor in stability_factors)
            cost = data.get('cost_per_kg', float('inf'))
            
            if is_stable and cost < 100:  # Arbitrary cost threshold
                substitute_candidates.append((ingredient, data, cost))
    
    # Sort by cost
    substitute_candidates.sort(key=lambda x: x[2])
    
    print(f"\nPotential substitutes for vitamin_c:")
    for ingredient, data, cost in substitute_candidates[:3]:  # Top 3
        print(f"  • {ingredient}:")
        print(f"    - Cost: ${cost:.2f}/kg")
        print(f"    - Max concentration: {data['max_concentration']}%")
        print(f"    - Allergen risk: {data['allergen_risk']}")
        print(f"    - Stability: {', '.join(data['stability_factors'])}")
    
    # Recommend best substitute
    if substitute_candidates:
        best_substitute = substitute_candidates[0]
        print(f"\n  Recommended substitute: {best_substitute[0]}")
        print(f"  Rationale: Most cost-effective stable antioxidant")
        
        # Create substitution link
        substitution = EvaluationLink(
            PredicateNode("can_substitute"),
            ListLink(ConceptNode(best_substitute[0]), ConceptNode("vitamin_c"))
        )
        
        EvaluationLink(
            PredicateNode("substitution_reason"),
            ListLink(substitution, ConceptNode("improved_stability_lower_cost"))
        )

def print_advanced_summary(atomspace):
    """Print advanced analysis summary."""
    print("\n📊 === Advanced Analysis Summary ===")
    
    # Count different types of relationships
    relationship_counts = {
        'formulations': 0,
        'ingredients': 0,
        'compatibility_links': 0,
        'properties': 0
    }
    
    for atom in atomspace:
        if atom.type == types.ConceptNode:
            if 'formulation' in atom.name.lower():
                relationship_counts['formulations'] += 1
            elif atom.name in ['hyaluronic_acid', 'niacinamide', 'vitamin_c', 'glycerin', 
                             'cetyl_alcohol', 'phenoxyethanol', 'xanthan_gum', 'squalane', 'vitamin_e']:
                relationship_counts['ingredients'] += 1
        elif atom.type == types.EvaluationLink:
            if atom.out[0].name in ['compatible_with', 'incompatible_with', 'synergistic_with']:
                relationship_counts['compatibility_links'] += 1
            elif atom.out[0].name in ['concentration', 'pH', 'viscosity', 'stability_months']:
                relationship_counts['properties'] += 1
    
    print("Knowledge representation metrics:")
    for category, count in relationship_counts.items():
        print(f"  • {category}: {count}")
    
    print(f"\nTotal knowledge atoms: {len(atomspace)}")

def main():
    """Main function for advanced cosmetic chemistry demonstration."""
    print("🧪 Advanced Cosmetic Chemistry Analysis")
    print("=====================================")
    
    # Initialize atomspace and analyzer
    atomspace = AtomSpace()
    initialize_opencog(atomspace)
    
    try:
        import opencog.bioscience
        print("✓ Bioscience module loaded successfully")
    except ImportError:
        print("⚠ Warning: Using standard atom types for demonstration")
    
    # Initialize advanced analyzer
    analyzer = CosmeticFormulationAnalyzer(atomspace)
    
    # Run advanced demonstrations
    formulation = demonstrate_formulation_optimization(analyzer)
    stability_score = demonstrate_stability_prediction(analyzer, formulation)
    compliance_results = demonstrate_regulatory_compliance(analyzer, formulation)
    demonstrate_ingredient_substitution(analyzer)
    
    # Print comprehensive summary
    print_advanced_summary(atomspace)
    
    print("\n✅ Advanced analysis completed!")
    print(f"   • Formulation stability score: {stability_score}/100")
    print(f"   • Regulatory compliance: {sum(compliance_results.values())}/{len(compliance_results)} regions")
    print("\nRecommendations:")
    print("  • Review stability analysis for optimization opportunities")
    print("  • Ensure regulatory compliance before market launch")
    print("  • Consider ingredient substitutions for cost optimization")
    print("  • Explore synergistic combinations for enhanced efficacy")

if __name__ == "__main__":
    main()
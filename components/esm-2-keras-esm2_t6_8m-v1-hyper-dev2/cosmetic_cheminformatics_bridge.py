#!/usr/bin/env python3
"""
Cosmetic Cheminformatics Bridge

This module creates a bridge between the existing Hypergredient Framework
and the new OpenCog Cheminformatics Framework for Cosmetic Chemistry,
enabling seamless integration and enhanced functionality.

This bridge allows users to:
- Convert between hypergredient and atom type representations
- Use cheminformatics analysis with existing hypergredient data
- Leverage both frameworks' strengths simultaneously
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import existing frameworks
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'python'))

from hypergredient_framework import (
    HypergredientClass, Hypergredient, HypergredientDatabase, 
    HypergredientOptimizer, FormulationRequest, FormulationResult
)

# Import cosmetic chemistry examples with proper path handling
try:
    from examples.python.cosmetic_intro_example import (
        AtomType, CosmeticIngredient, CosmeticFormulation, CosmeticChemistryFramework
    )
    from examples.python.cosmetic_chemistry_example import (
        AdvancedCosmeticChemistryFramework, StabilityAssessment, RegulatoryCompliance
    )
except ImportError:
    # Fallback for direct execution
    import cosmetic_intro_example as intro
    import cosmetic_chemistry_example as advanced
    
    AtomType = intro.AtomType
    CosmeticIngredient = intro.CosmeticIngredient
    CosmeticFormulation = intro.CosmeticFormulation
    CosmeticChemistryFramework = intro.CosmeticChemistryFramework
    AdvancedCosmeticChemistryFramework = advanced.AdvancedCosmeticChemistryFramework
    StabilityAssessment = advanced.StabilityAssessment
    RegulatoryCompliance = advanced.RegulatoryCompliance


class HypergredientToAtomMapping:
    """Mapping between Hypergredient classes and Atom types"""
    
    # Direct mappings where functionality aligns
    DIRECT_MAPPINGS = {
        HypergredientClass.CT: AtomType.ACTIVE_INGREDIENT,  # Cellular Turnover -> Active
        HypergredientClass.CS: AtomType.ACTIVE_INGREDIENT,  # Collagen Synthesis -> Active
        HypergredientClass.AO: AtomType.ANTIOXIDANT,        # Antioxidant Systems -> Antioxidant
        HypergredientClass.ML: AtomType.ACTIVE_INGREDIENT,  # Melanin Modulators -> Active
        HypergredientClass.HY: AtomType.HUMECTANT,          # Hydration Systems -> Humectant
        HypergredientClass.AI: AtomType.ACTIVE_INGREDIENT,  # Anti-Inflammatory -> Active
        HypergredientClass.SE: AtomType.ACTIVE_INGREDIENT,  # Sebum Regulators -> Active
        HypergredientClass.PD: AtomType.ACTIVE_INGREDIENT,  # Penetration/Delivery -> Active
    }
    
    # Complex mappings that may require additional logic
    COMPLEX_MAPPINGS = {
        HypergredientClass.BR: [AtomType.EMOLLIENT, AtomType.ACTIVE_INGREDIENT],  # Barrier Repair
        HypergredientClass.MB: [AtomType.ACTIVE_INGREDIENT, AtomType.PRESERVATIVE],  # Microbiome Balance
    }


@dataclass
class BridgedIngredient:
    """Ingredient that combines both hypergredient and cheminformatics data"""
    hypergredient: Hypergredient
    cosmetic_ingredient: CosmeticIngredient
    atom_type_primary: AtomType
    atom_type_secondary: Optional[AtomType] = None
    enhanced_properties: Dict[str, Any] = field(default_factory=dict)


class CosmeticCheminformaticsBridge:
    """Bridge between Hypergredient Framework and Cosmetic Cheminformatics"""
    
    def __init__(self):
        # Initialize both frameworks
        self.hypergredient_db = HypergredientDatabase()
        self.hypergredient_optimizer = HypergredientOptimizer(self.hypergredient_db)
        self.cheminformatics_framework = AdvancedCosmeticChemistryFramework()
        
        # Bridge-specific data structures
        self.bridged_ingredients = {}
        self.mapping = HypergredientToAtomMapping()
        
        # Initialize bridge
        self._create_bridged_ingredients()
        self._enhance_compatibility_rules()
    
    def _create_bridged_ingredients(self):
        """Create bridged ingredients combining both representations"""
        
        for hypergredient_id, hypergredient in self.hypergredient_db.hypergredients.items():
            # Determine primary atom type
            primary_atom_type = self._map_hypergredient_to_atom_type(hypergredient)
            
            # Create corresponding cosmetic ingredient
            cosmetic_ingredient = self._create_cosmetic_ingredient_from_hypergredient(
                hypergredient, primary_atom_type
            )
            
            # Add to cheminformatics framework
            self.cheminformatics_framework.add_ingredient(cosmetic_ingredient)
            
            # Create bridged ingredient
            bridged = BridgedIngredient(
                hypergredient=hypergredient,
                cosmetic_ingredient=cosmetic_ingredient,
                atom_type_primary=primary_atom_type,
                enhanced_properties=self._extract_enhanced_properties(hypergredient)
            )
            
            self.bridged_ingredients[hypergredient_id] = bridged
    
    def _map_hypergredient_to_atom_type(self, hypergredient: Hypergredient) -> AtomType:
        """Map hypergredient class to primary atom type"""
        hypergredient_class = hypergredient.hypergredient_class
        
        # Check direct mappings first
        if hypergredient_class in self.mapping.DIRECT_MAPPINGS:
            return self.mapping.DIRECT_MAPPINGS[hypergredient_class]
        
        # Handle complex mappings
        elif hypergredient_class in self.mapping.COMPLEX_MAPPINGS:
            return self.mapping.COMPLEX_MAPPINGS[hypergredient_class][0]  # Use primary
        
        # Default fallback
        else:
            return AtomType.ACTIVE_INGREDIENT
    
    def _create_cosmetic_ingredient_from_hypergredient(
        self, hypergredient: Hypergredient, atom_type: AtomType
    ) -> CosmeticIngredient:
        """Convert hypergredient to cosmetic ingredient"""
        
        # Extract typical usage percentage based on hypergredient class
        typical_usage = self._estimate_usage_percentage(hypergredient)
        
        # Create cosmetic ingredient
        return CosmeticIngredient(
            name=hypergredient.name,
            inci_name=hypergredient.inci_name,
            atom_type=atom_type,
            concentration=typical_usage,
            ph_min=hypergredient.ph_min,
            ph_max=hypergredient.ph_max,
            max_concentration=self._estimate_max_concentration(hypergredient),
            properties={
                "hypergredient_class": hypergredient.hypergredient_class.value,
                "efficacy_score": hypergredient.efficacy_score,
                "safety_score": hypergredient.safety_score,
                "bioavailability": hypergredient.bioavailability,
                "stability_index": hypergredient.stability_index,
                "cost_per_gram": hypergredient.cost_per_gram,
                "primary_function": hypergredient.primary_function,
                "secondary_functions": hypergredient.secondary_functions,
                "clinical_evidence": hypergredient.clinical_evidence_level,
                "incompatibilities": hypergredient.incompatibilities,
                "synergies": hypergredient.synergies
            }
        )
    
    def _estimate_usage_percentage(self, hypergredient: Hypergredient) -> float:
        """Estimate typical usage percentage for hypergredient"""
        usage_map = {
            HypergredientClass.CT: 0.5,   # Cellular turnover actives (low %)
            HypergredientClass.CS: 3.0,   # Collagen synthesis (peptides)
            HypergredientClass.AO: 1.0,   # Antioxidants
            HypergredientClass.BR: 2.0,   # Barrier repair
            HypergredientClass.ML: 2.0,   # Melanin modulators
            HypergredientClass.HY: 1.5,   # Hydration systems
            HypergredientClass.AI: 5.0,   # Anti-inflammatory (niacinamide)
            HypergredientClass.MB: 1.0,   # Microbiome balancers
            HypergredientClass.SE: 2.0,   # Sebum regulators
            HypergredientClass.PD: 1.0,   # Penetration enhancers
        }
        return usage_map.get(hypergredient.hypergredient_class, 1.0)
    
    def _estimate_max_concentration(self, hypergredient: Hypergredient) -> float:
        """Estimate maximum safe concentration"""
        max_concentration_map = {
            HypergredientClass.CT: 2.0,    # Cellular turnover
            HypergredientClass.CS: 10.0,   # Collagen synthesis  
            HypergredientClass.AO: 5.0,    # Antioxidants
            HypergredientClass.BR: 10.0,   # Barrier repair
            HypergredientClass.ML: 5.0,    # Melanin modulators
            HypergredientClass.HY: 10.0,   # Hydration systems
            HypergredientClass.AI: 15.0,   # Anti-inflammatory
            HypergredientClass.MB: 5.0,    # Microbiome balancers
            HypergredientClass.SE: 10.0,   # Sebum regulators
            HypergredientClass.PD: 5.0,    # Penetration enhancers
        }
        return max_concentration_map.get(hypergredient.hypergredient_class, 5.0)
    
    def _extract_enhanced_properties(self, hypergredient: Hypergredient) -> Dict[str, Any]:
        """Extract enhanced properties for bridge analysis"""
        return {
            "composite_score": hypergredient.calculate_composite_score({
                'efficacy': 0.3, 'safety': 0.3, 'stability': 0.2, 
                'bioavailability': 0.1, 'cost_efficiency': 0.1
            }),
            "risk_score": 10.0 - hypergredient.safety_score,
            "value_score": hypergredient.efficacy_score / (hypergredient.cost_per_gram + 1.0),
            "stability_category": self._categorize_stability(hypergredient.stability_index)
        }
    
    def _categorize_stability(self, stability_index: float) -> str:
        """Categorize stability based on index"""
        if stability_index >= 0.8:
            return "excellent"
        elif stability_index >= 0.6:
            return "good"
        elif stability_index >= 0.4:
            return "moderate"
        else:
            return "poor"
    
    def _enhance_compatibility_rules(self):
        """Enhance compatibility rules using hypergredient synergy data"""
        
        for ingredient_id, bridged in self.bridged_ingredients.items():
            hypergredient = bridged.hypergredient
            
            # Add synergies from hypergredient data
            for synergy_ingredient in hypergredient.synergies:
                # Find corresponding bridged ingredient
                synergy_bridged = self._find_bridged_ingredient_by_name(synergy_ingredient)
                if synergy_bridged:
                    # Add to cheminformatics synergy rules
                    key = (hypergredient.name, synergy_bridged.hypergredient.name)
                    self.cheminformatics_framework.synergy_rules[key] = \
                        f"Hypergredient synergy: Enhanced {hypergredient.primary_function}"
            
            # Add incompatibilities from hypergredient data
            for incompatible_ingredient in hypergredient.incompatibilities:
                incompatible_bridged = self._find_bridged_ingredient_by_name(incompatible_ingredient)
                if incompatible_bridged:
                    key = (hypergredient.name, incompatible_bridged.hypergredient.name)
                    self.cheminformatics_framework.incompatibility_rules[key] = \
                        f"Hypergredient incompatibility: Chemical or pH conflict"
    
    def _find_bridged_ingredient_by_name(self, name: str) -> Optional[BridgedIngredient]:
        """Find bridged ingredient by name or partial match"""
        name_lower = name.lower()
        for bridged in self.bridged_ingredients.values():
            if (name_lower in bridged.hypergredient.name.lower() or 
                name_lower in bridged.hypergredient.inci_name.lower()):
                return bridged
        return None
    
    def create_enhanced_formulation_from_hypergredient_result(
        self, result: FormulationResult
    ) -> CosmeticFormulation:
        """Convert hypergredient formulation result to enhanced cosmetic formulation"""
        
        formulation = CosmeticFormulation(
            name=f"Enhanced_{result.selected_hypergredients}",
            formulation_type=AtomType.SKINCARE_FORMULATION
        )
        
        # Add ingredients from hypergredient result
        for class_name, data in result.selected_hypergredients.items():
            ingredient_name = data['ingredient'].name
            bridged = self._find_bridged_ingredient_by_name(ingredient_name)
            
            if bridged:
                # Create enhanced cosmetic ingredient with optimization data
                enhanced_ingredient = CosmeticIngredient(
                    name=bridged.cosmetic_ingredient.name,
                    inci_name=bridged.cosmetic_ingredient.inci_name,
                    atom_type=bridged.cosmetic_ingredient.atom_type,
                    concentration=data['percentage'],
                    ph_min=bridged.cosmetic_ingredient.ph_min,
                    ph_max=bridged.cosmetic_ingredient.ph_max,
                    max_concentration=bridged.cosmetic_ingredient.max_concentration,
                    properties={
                        **bridged.cosmetic_ingredient.properties,
                        "optimization_reasoning": data['reasoning'],
                        "optimization_cost": data['cost'],
                        "hypergredient_class": class_name
                    }
                )
                formulation.add_ingredient(enhanced_ingredient)
        
        # Add formulation-level properties from hypergredient result
        formulation.properties = {
            "total_cost": result.total_cost,
            "predicted_efficacy": result.predicted_efficacy,
            "safety_score": result.safety_score,
            "stability_months": result.stability_months,
            "synergy_score": result.synergy_score,
            "optimization_reasoning": result.reasoning
        }
        
        return formulation
    
    def enhanced_formulation_analysis(
        self, formulation_request: FormulationRequest
    ) -> Dict[str, Any]:
        """Perform enhanced formulation analysis using both frameworks"""
        
        # Get hypergredient optimization result
        hypergredient_result = self.hypergredient_optimizer.optimize_formulation(formulation_request)
        
        # Convert to enhanced cosmetic formulation
        cosmetic_formulation = self.create_enhanced_formulation_from_hypergredient_result(
            hypergredient_result
        )
        
        # Perform cheminformatics analysis
        stability_assessment = self.cheminformatics_framework.assess_stability(cosmetic_formulation)
        regulatory_compliance = self.cheminformatics_framework.check_regulatory_compliance(
            cosmetic_formulation, "EU"
        )
        validation = self.cheminformatics_framework.validate_formulation(cosmetic_formulation)
        
        # Combine results
        return {
            "hypergredient_optimization": {
                "selected_ingredients": hypergredient_result.selected_hypergredients,
                "total_cost": hypergredient_result.total_cost,
                "predicted_efficacy": hypergredient_result.predicted_efficacy,
                "safety_score": hypergredient_result.safety_score,
                "stability_months": hypergredient_result.stability_months,
                "synergy_score": hypergredient_result.synergy_score
            },
            "cheminformatics_analysis": {
                "stability_assessment": {
                    "overall_score": stability_assessment.overall_score,
                    "ph_stability": stability_assessment.ph_stability,
                    "oxidation_resistance": stability_assessment.oxidation_resistance,
                    "light_stability": stability_assessment.light_stability,
                    "shelf_life_months": stability_assessment.shelf_life_months,
                    "warnings": stability_assessment.warnings,
                    "storage_conditions": stability_assessment.storage_conditions
                },
                "regulatory_compliance": {
                    "compliant": regulatory_compliance.compliant,
                    "violations": regulatory_compliance.violations,
                    "warnings": regulatory_compliance.warnings
                },
                "formulation_validation": {
                    "valid": validation["valid"],
                    "warnings": validation["warnings"],
                    "errors": validation["errors"]
                }
            },
            "enhanced_formulation": cosmetic_formulation,
            "bridged_ingredients": {
                ing.name: {
                    "hypergredient_class": ing.properties.get("hypergredient_class"),
                    "atom_type": ing.atom_type.value,
                    "enhanced_properties": self.bridged_ingredients.get(
                        ing.name, BridgedIngredient(None, ing, ing.atom_type)
                    ).enhanced_properties
                }
                for ing in cosmetic_formulation.ingredients
            }
        }
    
    def generate_comprehensive_report(
        self, formulation_request: FormulationRequest
    ) -> str:
        """Generate comprehensive report combining both frameworks"""
        
        analysis = self.enhanced_formulation_analysis(formulation_request)
        
        report = "🔬 COMPREHENSIVE COSMETIC FORMULATION ANALYSIS\n"
        report += "=" * 60 + "\n\n"
        
        # Request summary
        report += f"Target Concerns: {', '.join(formulation_request.target_concerns)}\n"
        report += f"Secondary Concerns: {', '.join(formulation_request.secondary_concerns)}\n"
        report += f"Skin Type: {formulation_request.skin_type}\n"
        report += f"Budget: R{formulation_request.budget:.2f}\n\n"
        
        # Hypergredient optimization results
        hyper_results = analysis["hypergredient_optimization"]
        report += "🧬 HYPERGREDIENT OPTIMIZATION RESULTS\n"
        report += "-" * 40 + "\n"
        report += f"Total Cost: R{hyper_results['total_cost']:.2f}\n"
        report += f"Predicted Efficacy: {hyper_results['predicted_efficacy']:.1%}\n"
        report += f"Safety Score: {hyper_results['safety_score']:.1f}/10\n"
        report += f"Stability: {hyper_results['stability_months']} months\n"
        report += f"Synergy Score: {hyper_results['synergy_score']:.2f}\n\n"
        
        report += "Selected Ingredients:\n"
        for class_name, data in hyper_results["selected_ingredients"].items():
            report += f"  • {data['ingredient'].name} ({data['percentage']:.1f}%)\n"
            report += f"    Class: {class_name}, Cost: R{data['cost']:.2f}\n"
        report += "\n"
        
        # Cheminformatics analysis results
        chem_results = analysis["cheminformatics_analysis"]
        report += "⚗️ CHEMINFORMATICS ANALYSIS RESULTS\n"
        report += "-" * 40 + "\n"
        
        # Stability assessment
        stability = chem_results["stability_assessment"]
        report += f"Stability Score: {stability['overall_score']:.2f}/1.0\n"
        report += f"pH Stability: {stability['ph_stability']:.2f}/1.0\n"
        report += f"Oxidation Resistance: {stability['oxidation_resistance']:.2f}/1.0\n"
        report += f"Light Stability: {stability['light_stability']:.2f}/1.0\n"
        report += f"Predicted Shelf Life: {stability['shelf_life_months']} months\n"
        
        if stability["warnings"]:
            report += "\nStability Warnings:\n"
            for warning in stability["warnings"]:
                report += f"  ⚠️ {warning}\n"
        
        if stability["storage_conditions"]:
            report += "\nStorage Requirements:\n"
            for condition in stability["storage_conditions"]:
                report += f"  • {condition.replace('_', ' ').title()}\n"
        report += "\n"
        
        # Regulatory compliance
        regulatory = chem_results["regulatory_compliance"]
        report += f"EU Regulatory Compliance: {'✅ Compliant' if regulatory['compliant'] else '❌ Non-compliant'}\n"
        
        if regulatory["violations"]:
            report += "Violations:\n"
            for violation in regulatory["violations"]:
                report += f"  ❌ {violation}\n"
        
        if regulatory["warnings"]:
            report += "Regulatory Warnings:\n"
            for warning in regulatory["warnings"]:
                report += f"  ⚠️ {warning}\n"
        report += "\n"
        
        # Enhanced insights
        report += "🎯 ENHANCED INSIGHTS\n"
        report += "-" * 20 + "\n"
        
        bridged_ingredients = analysis["bridged_ingredients"]
        for name, data in bridged_ingredients.items():
            enhanced_props = data["enhanced_properties"]
            report += f"• {name}:\n"
            report += f"  Composite Score: {enhanced_props.get('composite_score', 0):.2f}\n"
            report += f"  Value Score: {enhanced_props.get('value_score', 0):.2f}\n"
            report += f"  Stability Category: {enhanced_props.get('stability_category', 'unknown')}\n"
        
        report += "\n✅ Analysis Complete!\n"
        return report


def main():
    """Demonstrate the cosmetic cheminformatics bridge"""
    print("🌉 Cosmetic Cheminformatics Bridge Demo")
    print("=" * 50)
    
    # Initialize bridge
    bridge = CosmeticCheminformaticsBridge()
    print(f"\n🔗 Bridge initialized with {len(bridge.bridged_ingredients)} bridged ingredients")
    
    # Example analysis request
    analysis_request = FormulationRequest(
        target_concerns=['wrinkles', 'firmness'],
        secondary_concerns=['dryness', 'dullness'],
        skin_type='normal_to_dry',
        budget=800.0,
        preferences=['gentle', 'stable']
    )
    
    print("\n📋 Analysis Request:")
    print(f"   Target Concerns: {analysis_request.target_concerns}")
    print(f"   Budget: R{analysis_request.budget}")
    print(f"   Preferences: {analysis_request.preferences}")
    
    # Perform enhanced analysis
    print("\n🔬 Performing Enhanced Analysis...")
    report = bridge.generate_comprehensive_report(analysis_request)
    print(report)
    
    # Demonstrate bridged ingredient capabilities
    print("🔍 Bridged Ingredient Examples:")
    print("-" * 32)
    
    for ingredient_id, bridged in list(bridge.bridged_ingredients.items())[:3]:
        print(f"\n• {bridged.hypergredient.name}:")
        print(f"  Hypergredient Class: {bridged.hypergredient.hypergredient_class.value}")
        print(f"  Atom Type: {bridged.atom_type_primary.value}")
        print(f"  Efficacy Score: {bridged.hypergredient.efficacy_score:.1f}/10")
        print(f"  Safety Score: {bridged.hypergredient.safety_score:.1f}/10")
        print(f"  Composite Score: {bridged.enhanced_properties.get('composite_score', 0):.2f}")
        print(f"  Stability Category: {bridged.enhanced_properties.get('stability_category')}")
    
    print("\n✅ Bridge demonstration completed!")
    print("\nThe bridge successfully combines:")
    print("- Hypergredient optimization algorithms")
    print("- Cheminformatics stability analysis")
    print("- Regulatory compliance checking")
    print("- Enhanced ingredient intelligence")


if __name__ == "__main__":
    main()
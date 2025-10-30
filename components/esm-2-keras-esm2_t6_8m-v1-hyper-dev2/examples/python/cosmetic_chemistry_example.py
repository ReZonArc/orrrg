#!/usr/bin/env python3
"""
Cosmetic Chemistry Framework - Advanced Formulation Analysis

This example demonstrates advanced formulation analysis and optimization using
the cosmetic chemistry atom types, including multi-objective optimization,
stability prediction, and regulatory compliance checking.

Based on the OpenCog Cheminformatics Framework atom type system.
"""

import json
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
# Handle imports properly for both direct execution and module imports
try:
    from cosmetic_intro_example import AtomType, CosmeticIngredient, CosmeticFormulation, CosmeticChemistryFramework
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import cosmetic_intro_example as intro
    AtomType = intro.AtomType
    CosmeticIngredient = intro.CosmeticIngredient
    CosmeticFormulation = intro.CosmeticFormulation
    CosmeticChemistryFramework = intro.CosmeticChemistryFramework


@dataclass
class StabilityAssessment:
    """Represents stability assessment for a formulation"""
    overall_score: float
    ph_stability: float
    temperature_stability: float
    oxidation_resistance: float
    light_stability: float
    shelf_life_months: int
    storage_conditions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RegulatoryCompliance:
    """Represents regulatory compliance assessment"""
    compliant: bool
    region: str
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    required_declarations: List[str] = field(default_factory=list)


@dataclass
class FormulationOptimization:
    """Represents formulation optimization result"""
    original_formulation: CosmeticFormulation
    optimized_formulation: CosmeticFormulation
    improvements: Dict[str, float]
    cost_analysis: Dict[str, float]
    efficacy_prediction: float
    safety_score: float


class AdvancedCosmeticChemistryFramework(CosmeticChemistryFramework):
    """Advanced framework with optimization and analysis capabilities"""
    
    def __init__(self):
        super().__init__()
        self.regulatory_database = {}
        self.stability_rules = {}
        self.optimization_weights = {}
        self._initialize_advanced_features()
    
    def _initialize_advanced_features(self):
        """Initialize advanced analysis features"""
        
        # Add more advanced ingredients to database
        self._add_advanced_ingredients()
        
        # Initialize regulatory database
        self._initialize_regulatory_database()
        
        # Initialize stability rules
        self._initialize_stability_rules()
        
        # Initialize optimization weights
        self.optimization_weights = {
            'efficacy': 0.3,
            'safety': 0.25,
            'stability': 0.2,
            'cost': 0.15,
            'regulatory_compliance': 0.1
        }
    
    def _add_advanced_ingredients(self):
        """Add advanced ingredients for comprehensive analysis"""
        
        # Advanced actives
        self.add_ingredient(CosmeticIngredient(
            name="Bakuchiol",
            inci_name="Bakuchiol",
            atom_type=AtomType.ACTIVE_INGREDIENT,
            concentration=1.0,
            ph_min=4.0,
            ph_max=8.0,
            max_concentration=2.0,
            properties={
                "function": "retinol_alternative",
                "stability": "excellent",
                "sensitivity": "low",
                "pregnancy_safe": True,
                "cost_per_gram": 280.0
            }
        ))
        
        self.add_ingredient(CosmeticIngredient(
            name="Alpha Arbutin",
            inci_name="Alpha Arbutin",
            atom_type=AtomType.ACTIVE_INGREDIENT,
            concentration=2.0,
            ph_min=4.0,
            ph_max=6.0,
            max_concentration=2.0,
            properties={
                "function": "brightening",
                "stability": "good",
                "tyrosinase_inhibition": True,
                "cost_per_gram": 120.0
            }
        ))
        
        # Advanced preservatives
        self.add_ingredient(CosmeticIngredient(
            name="Leucidal Liquid",
            inci_name="Leuconostoc/Radish Root Ferment Filtrate",
            atom_type=AtomType.PRESERVATIVE,
            concentration=2.0,
            ph_min=4.0,
            ph_max=6.0,
            max_concentration=5.0,
            properties={
                "function": "natural_preservation",
                "spectrum": "limited",
                "natural": True,
                "cost_per_gram": 45.0
            }
        ))
        
        # Advanced antioxidants
        self.add_ingredient(CosmeticIngredient(
            name="Astaxanthin",
            inci_name="Astaxanthin",
            atom_type=AtomType.ANTIOXIDANT,
            concentration=0.1,
            ph_min=4.0,
            ph_max=7.0,
            max_concentration=0.2,
            properties={
                "function": "powerful_antioxidant",
                "stability": "light_sensitive",
                "potency": "very_high",
                "cost_per_gram": 800.0
            }
        ))
        
        # UV Filters
        self.add_ingredient(CosmeticIngredient(
            name="Zinc Oxide",
            inci_name="Zinc Oxide",
            atom_type=AtomType.UV_FILTER,
            concentration=15.0,
            ph_min=6.0,
            ph_max=8.0,
            max_concentration=25.0,
            properties={
                "function": "mineral_uv_protection",
                "spectrum": "broad_spectrum",
                "safety": "excellent",
                "cost_per_gram": 8.0
            }
        ))
    
    def _initialize_regulatory_database(self):
        """Initialize regulatory compliance database"""
        
        self.regulatory_database = {
            "EU": {
                "banned_ingredients": ["hydroquinone", "mercury_compounds"],
                "restricted_ingredients": {
                    "retinol": 0.3,  # 0.3% max in EU
                    "salicylic_acid": 2.0,
                    "kojic_acid": 1.0
                },
                "allergen_threshold": 0.001,  # 0.001% for leave-on products
                "required_declarations": [
                    "linalool", "limonene", "citronellol", "geraniol",
                    "benzyl_alcohol", "benzyl_salicylate", "citral"
                ]
            },
            "FDA": {
                "banned_ingredients": ["chloroform", "vinyl_chloride"],
                "restricted_ingredients": {
                    "hydroquinone": 2.0,  # OTC drug classification
                    "mercury": 0.0001
                },
                "drug_ingredients": [
                    "zinc_oxide", "titanium_dioxide", "salicylic_acid"
                ]
            },
            "HEALTH_CANADA": {
                "banned_ingredients": ["lead_acetate", "mercury_compounds"],
                "restricted_ingredients": {
                    "retinol": 1.0,
                    "hydroquinone": 2.0
                }
            }
        }
    
    def _initialize_stability_rules(self):
        """Initialize stability assessment rules"""
        
        self.stability_rules = {
            "ph_sensitive": {
                "vitamin_c": {"min": 3.5, "max": 4.0},
                "retinol": {"min": 5.5, "max": 6.5},
                "niacinamide": {"min": 5.0, "max": 7.0}
            },
            "oxidation_sensitive": [
                "vitamin_c", "retinol", "vitamin_e", "unsaturated_oils"
            ],
            "light_sensitive": [
                "vitamin_c", "retinol", "aha", "bha", "kojic_acid"
            ],
            "temperature_sensitive": [
                "vitamin_c", "retinol", "peptides", "enzymes"
            ]
        }
    
    def assess_stability(self, formulation: CosmeticFormulation) -> StabilityAssessment:
        """Comprehensive stability assessment"""
        
        stability = StabilityAssessment(
            overall_score=0.0,
            ph_stability=0.0,
            temperature_stability=0.0,
            oxidation_resistance=0.0,
            light_stability=0.0,
            shelf_life_months=36
        )
        
        ingredient_names = [ing.name.lower().replace(" ", "_") 
                           for ing in formulation.ingredients]
        
        # pH stability assessment
        ph_conflicts = 0
        ph_requirements = []
        for ingredient in formulation.ingredients:
            if ingredient.ph_min and ingredient.ph_max:
                ph_requirements.append((ingredient.ph_min, ingredient.ph_max))
        
        if ph_requirements:
            overall_min = max(req[0] for req in ph_requirements)
            overall_max = min(req[1] for req in ph_requirements)
            
            if overall_min > overall_max:
                ph_conflicts += 1
                stability.warnings.append("pH incompatibility detected")
                stability.ph_stability = 0.3
            else:
                stability.ph_stability = 0.9
        else:
            stability.ph_stability = 0.7
        
        # Oxidation sensitivity
        oxidation_sensitive_count = sum(1 for name in ingredient_names 
                                      if name in self.stability_rules["oxidation_sensitive"])
        if oxidation_sensitive_count > 0:
            stability.oxidation_resistance = max(0.2, 1.0 - (oxidation_sensitive_count * 0.2))
            stability.storage_conditions.append("inert_atmosphere")
            stability.warnings.append("Contains oxidation-sensitive ingredients")
        else:
            stability.oxidation_resistance = 0.9
        
        # Light sensitivity
        light_sensitive_count = sum(1 for name in ingredient_names 
                                  if name in self.stability_rules["light_sensitive"])
        if light_sensitive_count > 0:
            stability.light_stability = max(0.3, 1.0 - (light_sensitive_count * 0.15))
            stability.storage_conditions.append("opaque_packaging")
            stability.warnings.append("Requires protection from light")
        else:
            stability.light_stability = 0.9
        
        # Temperature sensitivity
        temp_sensitive_count = sum(1 for name in ingredient_names 
                                 if name in self.stability_rules["temperature_sensitive"])
        if temp_sensitive_count > 0:
            stability.temperature_stability = max(0.4, 1.0 - (temp_sensitive_count * 0.1))
            stability.storage_conditions.append("cool_storage")
            stability.shelf_life_months = min(stability.shelf_life_months, 24)
        else:
            stability.temperature_stability = 0.9
        
        # Calculate overall stability score
        stability.overall_score = (
            stability.ph_stability * 0.3 +
            stability.oxidation_resistance * 0.25 +
            stability.light_stability * 0.25 +
            stability.temperature_stability * 0.2
        )
        
        # Adjust shelf life based on overall score
        if stability.overall_score < 0.5:
            stability.shelf_life_months = min(stability.shelf_life_months, 12)
        elif stability.overall_score < 0.7:
            stability.shelf_life_months = min(stability.shelf_life_months, 24)
        
        return stability
    
    def check_regulatory_compliance(self, formulation: CosmeticFormulation, 
                                  region: str = "EU") -> RegulatoryCompliance:
        """Check regulatory compliance for specific region"""
        
        compliance = RegulatoryCompliance(
            compliant=True,
            region=region
        )
        
        if region not in self.regulatory_database:
            compliance.compliant = False
            compliance.violations.append(f"Region '{region}' not supported")
            return compliance
        
        region_rules = self.regulatory_database[region]
        
        # Check banned ingredients
        for ingredient in formulation.ingredients:
            ingredient_key = ingredient.name.lower().replace(" ", "_")
            if ingredient_key in region_rules.get("banned_ingredients", []):
                compliance.compliant = False
                compliance.violations.append(
                    f"Banned ingredient: {ingredient.name}"
                )
        
        # Check concentration limits
        for ingredient in formulation.ingredients:
            ingredient_key = ingredient.name.lower().replace(" ", "_")
            if ingredient_key in region_rules.get("restricted_ingredients", {}):
                max_allowed = region_rules["restricted_ingredients"][ingredient_key]
                if ingredient.concentration and ingredient.concentration > max_allowed:
                    compliance.compliant = False
                    compliance.violations.append(
                        f"{ingredient.name} exceeds maximum allowed concentration "
                        f"({ingredient.concentration}% > {max_allowed}%)"
                    )
        
        # Check allergen declarations
        if "required_declarations" in region_rules:
            for ingredient in formulation.ingredients:
                ingredient_key = ingredient.name.lower().replace(" ", "_")
                if ingredient_key in region_rules["required_declarations"]:
                    compliance.required_declarations.append(ingredient.name)
        
        # Check drug ingredients (FDA specific)
        if region == "FDA" and "drug_ingredients" in region_rules:
            for ingredient in formulation.ingredients:
                ingredient_key = ingredient.name.lower().replace(" ", "_")
                if ingredient_key in region_rules["drug_ingredients"]:
                    compliance.warnings.append(
                        f"{ingredient.name} requires drug classification"
                    )
        
        return compliance
    
    def optimize_formulation(self, formulation: CosmeticFormulation, 
                           optimization_goals: List[str]) -> FormulationOptimization:
        """Multi-objective formulation optimization"""
        
        # Create copy for optimization
        optimized = CosmeticFormulation(
            name=f"Optimized_{formulation.name}",
            formulation_type=formulation.formulation_type,
            ingredients=formulation.ingredients.copy()
        )
        
        improvements = {}
        cost_analysis = {}
        
        # Calculate baseline metrics
        baseline_stability = self.assess_stability(formulation)
        baseline_compliance = self.check_regulatory_compliance(formulation)
        baseline_validation = self.validate_formulation(formulation)
        
        # Optimization strategies
        if "stability" in optimization_goals:
            improvements["stability"] = self._optimize_for_stability(optimized)
        
        if "cost" in optimization_goals:
            improvements["cost_reduction"] = self._optimize_for_cost(optimized)
        
        if "efficacy" in optimization_goals:
            improvements["efficacy"] = self._optimize_for_efficacy(optimized)
        
        if "safety" in optimization_goals:
            improvements["safety"] = self._optimize_for_safety(optimized)
        
        # Calculate optimized metrics
        optimized_stability = self.assess_stability(optimized)
        
        # Cost analysis
        original_cost = sum(ing.properties.get("cost_per_gram", 10.0) * (ing.concentration or 1.0) / 100 
                           for ing in formulation.ingredients)
        optimized_cost = sum(ing.properties.get("cost_per_gram", 10.0) * (ing.concentration or 1.0) / 100 
                            for ing in optimized.ingredients)
        
        cost_analysis = {
            "original_cost_per_100g": original_cost,
            "optimized_cost_per_100g": optimized_cost,
            "cost_savings_percent": ((original_cost - optimized_cost) / original_cost * 100) if original_cost > 0 else 0
        }
        
        # Efficacy prediction (simplified model)
        efficacy_prediction = self._predict_efficacy(optimized)
        
        # Safety score calculation
        safety_score = self._calculate_safety_score(optimized)
        
        return FormulationOptimization(
            original_formulation=formulation,
            optimized_formulation=optimized,
            improvements=improvements,
            cost_analysis=cost_analysis,
            efficacy_prediction=efficacy_prediction,
            safety_score=safety_score
        )
    
    def _optimize_for_stability(self, formulation: CosmeticFormulation) -> float:
        """Optimize formulation for stability"""
        stability_improvement = 0.0
        
        # Add antioxidants if missing and oxidation-sensitive ingredients present
        ingredient_names = [ing.name.lower().replace(" ", "_") for ing in formulation.ingredients]
        has_antioxidant = any(ing.atom_type == AtomType.ANTIOXIDANT for ing in formulation.ingredients)
        has_oxidation_sensitive = any(name in self.stability_rules["oxidation_sensitive"] 
                                    for name in ingredient_names)
        
        if has_oxidation_sensitive and not has_antioxidant:
            vitamin_e = self.get_ingredient("Vitamin E")
            if vitamin_e:
                formulation.add_ingredient(vitamin_e)
                stability_improvement += 0.2
        
        return stability_improvement
    
    def _optimize_for_cost(self, formulation: CosmeticFormulation) -> float:
        """Optimize formulation for cost"""
        cost_reduction = 0.0
        
        # Identify expensive ingredients and suggest alternatives
        expensive_threshold = 100.0  # Cost per gram threshold
        
        for i, ingredient in enumerate(formulation.ingredients):
            ingredient_cost = ingredient.properties.get("cost_per_gram", 0.0)
            if ingredient_cost > expensive_threshold:
                # Find cheaper alternative with similar function
                alternative = self._find_cheaper_alternative(ingredient)
                if alternative:
                    original_cost = ingredient_cost * (ingredient.concentration or 1.0) / 100
                    new_cost = alternative.properties.get("cost_per_gram", 0.0) * (alternative.concentration or 1.0) / 100
                    cost_reduction += (original_cost - new_cost) / original_cost
                    formulation.ingredients[i] = alternative
        
        return cost_reduction
    
    def _optimize_for_efficacy(self, formulation: CosmeticFormulation) -> float:
        """Optimize formulation for efficacy"""
        efficacy_improvement = 0.0
        
        # Add synergistic combinations
        ingredient_names = [ing.name for ing in formulation.ingredients]
        
        # Check for beneficial synergies that can be added
        if "Vitamin C" in ingredient_names and "Vitamin E" not in ingredient_names:
            vitamin_e = self.get_ingredient("Vitamin E")
            if vitamin_e:
                formulation.add_ingredient(vitamin_e)
                efficacy_improvement += 0.15
        
        return efficacy_improvement
    
    def _optimize_for_safety(self, formulation: CosmeticFormulation) -> float:
        """Optimize formulation for safety"""
        safety_improvement = 0.0
        
        # Replace harsh ingredients with gentler alternatives
        for i, ingredient in enumerate(formulation.ingredients):
            if ingredient.name == "Retinol":
                # Suggest bakuchiol as gentler alternative
                bakuchiol = self.get_ingredient("Bakuchiol")
                if bakuchiol:
                    formulation.ingredients[i] = bakuchiol
                    safety_improvement += 0.2
        
        return safety_improvement
    
    def _find_cheaper_alternative(self, ingredient: CosmeticIngredient) -> Optional[CosmeticIngredient]:
        """Find cheaper alternative with similar function"""
        # Simplified alternative finding
        alternatives = {
            "Astaxanthin": "Vitamin E",  # Cheaper antioxidant
            "Alpha Arbutin": "Kojic Acid"  # Cheaper brightening agent
        }
        
        alternative_name = alternatives.get(ingredient.name)
        if alternative_name:
            return self.get_ingredient(alternative_name)
        
        return None
    
    def _predict_efficacy(self, formulation: CosmeticFormulation) -> float:
        """Predict formulation efficacy (simplified model)"""
        efficacy_score = 0.0
        
        # Base efficacy from active ingredients
        for ingredient in formulation.ingredients:
            if ingredient.atom_type == AtomType.ACTIVE_INGREDIENT:
                efficacy_score += (ingredient.concentration or 1.0) * 0.1
        
        # Synergy bonuses
        ingredient_names = [ing.name for ing in formulation.ingredients]
        if "Vitamin C" in ingredient_names and "Vitamin E" in ingredient_names:
            efficacy_score *= 1.2  # 20% synergy bonus
        
        if "Hyaluronic Acid" in ingredient_names and "Glycerin" in ingredient_names:
            efficacy_score *= 1.1  # 10% hydration synergy
        
        return min(efficacy_score, 1.0)  # Cap at 100%
    
    def _calculate_safety_score(self, formulation: CosmeticFormulation) -> float:
        """Calculate safety score for formulation"""
        safety_score = 10.0  # Start with perfect score
        
        # Deduct points for potentially irritating ingredients
        irritating_ingredients = ["retinol", "vitamin_c", "aha", "bha"]
        
        for ingredient in formulation.ingredients:
            ingredient_key = ingredient.name.lower().replace(" ", "_")
            if ingredient_key in irritating_ingredients:
                concentration = ingredient.concentration or 0
                safety_score -= concentration * 0.1  # Deduct based on concentration
        
        # Bonus for gentle ingredients
        gentle_ingredients = ["hyaluronic_acid", "niacinamide", "bakuchiol"]
        for ingredient in formulation.ingredients:
            ingredient_key = ingredient.name.lower().replace(" ", "_")
            if ingredient_key in gentle_ingredients:
                safety_score += 0.2
        
        return max(0.0, min(10.0, safety_score))
    
    def generate_stability_report(self, formulation: CosmeticFormulation) -> str:
        """Generate comprehensive stability report"""
        stability = self.assess_stability(formulation)
        
        report = f"🔬 Stability Assessment Report: {formulation.name}\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Overall Stability Score: {stability.overall_score:.2f}/1.0\n"
        report += f"Predicted Shelf Life: {stability.shelf_life_months} months\n\n"
        
        report += "Detailed Scores:\n"
        report += f"  pH Stability: {stability.ph_stability:.2f}/1.0\n"
        report += f"  Temperature Stability: {stability.temperature_stability:.2f}/1.0\n"
        report += f"  Oxidation Resistance: {stability.oxidation_resistance:.2f}/1.0\n"
        report += f"  Light Stability: {stability.light_stability:.2f}/1.0\n\n"
        
        if stability.storage_conditions:
            report += "Required Storage Conditions:\n"
            for condition in stability.storage_conditions:
                report += f"  • {condition.replace('_', ' ').title()}\n"
            report += "\n"
        
        if stability.warnings:
            report += "Stability Warnings:\n"
            for warning in stability.warnings:
                report += f"  ⚠️ {warning}\n"
            report += "\n"
        
        # Recommendations
        report += "Recommendations:\n"
        if stability.overall_score < 0.6:
            report += "  • Consider reformulation to improve stability\n"
            report += "  • Add stabilizing ingredients (antioxidants, pH buffers)\n"
        if "opaque_packaging" in stability.storage_conditions:
            report += "  • Use UV-protective packaging\n"
        if "cool_storage" in stability.storage_conditions:
            report += "  • Recommend refrigerated storage\n"
        
        return report


def main():
    """Demonstrate advanced cosmetic chemistry analysis"""
    print("🧬 Advanced Cosmetic Chemistry Framework")
    print("=" * 50)
    
    # Initialize advanced framework
    framework = AdvancedCosmeticChemistryFramework()
    
    print(f"\n📚 Initialized database with {len(framework.ingredients_database)} ingredients")
    
    # Example 1: Create an advanced anti-aging serum
    print("\n1️⃣ Advanced Anti-Aging Serum Creation")
    print("-" * 40)
    
    serum = framework.create_formulation(
        "Advanced Anti-Aging Serum",
        AtomType.SKINCARE_FORMULATION
    )
    
    # Add premium ingredients
    serum.add_ingredient(framework.get_ingredient("Retinol"))
    serum.add_ingredient(framework.get_ingredient("Vitamin C"))
    serum.add_ingredient(framework.get_ingredient("Hyaluronic Acid"))
    serum.add_ingredient(framework.get_ingredient("Astaxanthin"))
    serum.add_ingredient(framework.get_ingredient("Phenoxyethanol"))
    
    print(f"Created: {serum}")
    print("Ingredients:")
    for ingredient in serum.ingredients:
        cost = ingredient.properties.get("cost_per_gram", 0)
        print(f"  • {ingredient.name} ({ingredient.concentration}%) - R{cost:.2f}/g")
    
    # Example 2: Comprehensive stability assessment
    print("\n2️⃣ Comprehensive Stability Assessment")
    print("-" * 38)
    
    stability = framework.assess_stability(serum)
    print(f"Overall Stability Score: {stability.overall_score:.2f}/1.0")
    print(f"Predicted Shelf Life: {stability.shelf_life_months} months")
    print(f"pH Stability: {stability.ph_stability:.2f}")
    print(f"Oxidation Resistance: {stability.oxidation_resistance:.2f}")
    print(f"Light Stability: {stability.light_stability:.2f}")
    
    if stability.warnings:
        print("\nStability Warnings:")
        for warning in stability.warnings:
            print(f"  ⚠️ {warning}")
    
    if stability.storage_conditions:
        print("\nStorage Requirements:")
        for condition in stability.storage_conditions:
            print(f"  • {condition.replace('_', ' ').title()}")
    
    # Example 3: Regulatory compliance checking
    print("\n3️⃣ Regulatory Compliance Analysis")
    print("-" * 35)
    
    # Check EU compliance
    eu_compliance = framework.check_regulatory_compliance(serum, "EU")
    print(f"EU Compliance: {'✅ Compliant' if eu_compliance.compliant else '❌ Non-compliant'}")
    
    if eu_compliance.violations:
        print("Violations:")
        for violation in eu_compliance.violations:
            print(f"  ❌ {violation}")
    
    if eu_compliance.warnings:
        print("Warnings:")
        for warning in eu_compliance.warnings:
            print(f"  ⚠️ {warning}")
    
    if eu_compliance.required_declarations:
        print("Required Label Declarations:")
        for declaration in eu_compliance.required_declarations:
            print(f"  • {declaration}")
    
    # Check FDA compliance
    fda_compliance = framework.check_regulatory_compliance(serum, "FDA")
    print(f"\nFDA Compliance: {'✅ Compliant' if fda_compliance.compliant else '❌ Non-compliant'}")
    
    if fda_compliance.warnings:
        print("FDA Warnings:")
        for warning in fda_compliance.warnings:
            print(f"  ⚠️ {warning}")
    
    # Example 4: Formulation optimization
    print("\n4️⃣ Multi-Objective Formulation Optimization")
    print("-" * 44)
    
    optimization_goals = ["stability", "cost", "safety"]
    optimization = framework.optimize_formulation(serum, optimization_goals)
    
    print(f"Original Formulation: {optimization.original_formulation.name}")
    print(f"Optimized Formulation: {optimization.optimized_formulation.name}")
    
    print(f"\nPredicted Efficacy: {optimization.efficacy_prediction:.1%}")
    print(f"Safety Score: {optimization.safety_score:.1f}/10")
    
    print("\nCost Analysis:")
    cost_analysis = optimization.cost_analysis
    print(f"  Original Cost: R{cost_analysis['original_cost_per_100g']:.2f}/100g")
    print(f"  Optimized Cost: R{cost_analysis['optimized_cost_per_100g']:.2f}/100g")
    print(f"  Cost Savings: {cost_analysis['cost_savings_percent']:.1f}%")
    
    if optimization.improvements:
        print("\nOptimization Improvements:")
        for goal, improvement in optimization.improvements.items():
            print(f"  • {goal.title()}: {improvement:.1%} improvement")
    
    print("\nOptimized Ingredients:")
    for ingredient in optimization.optimized_formulation.ingredients:
        print(f"  • {ingredient.name} ({ingredient.concentration}%)")
    
    # Example 5: Generate comprehensive report
    print("\n5️⃣ Comprehensive Stability Report")
    print("-" * 35)
    
    stability_report = framework.generate_stability_report(serum)
    print(stability_report)
    
    # Example 6: Create natural/gentle formulation
    print("\n6️⃣ Natural & Gentle Formulation")
    print("-" * 32)
    
    gentle_moisturizer = framework.create_formulation(
        "Gentle Natural Moisturizer",
        AtomType.SKINCARE_FORMULATION
    )
    
    # Use gentler alternatives
    gentle_moisturizer.add_ingredient(framework.get_ingredient("Bakuchiol"))  # Instead of retinol
    gentle_moisturizer.add_ingredient(framework.get_ingredient("Hyaluronic Acid"))
    gentle_moisturizer.add_ingredient(framework.get_ingredient("Glycerin"))
    gentle_moisturizer.add_ingredient(framework.get_ingredient("Leucidal Liquid"))  # Natural preservative
    
    print(f"Created: {gentle_moisturizer}")
    
    # Assess the gentle formulation
    gentle_stability = framework.assess_stability(gentle_moisturizer)
    gentle_safety = framework._calculate_safety_score(gentle_moisturizer)
    
    print(f"Stability Score: {gentle_stability.overall_score:.2f}/1.0")
    print(f"Safety Score: {gentle_safety:.1f}/10")
    print(f"Shelf Life: {gentle_stability.shelf_life_months} months")
    
    if len(gentle_stability.warnings) == 0:
        print("✅ No stability concerns detected")
    
    print("\n✅ Advanced analysis completed successfully!")
    print("\nKey Insights:")
    print("- Stability assessment identifies potential formulation issues")
    print("- Regulatory compliance prevents market access problems")
    print("- Multi-objective optimization balances multiple goals")
    print("- Natural alternatives can maintain efficacy with improved safety")


if __name__ == "__main__":
    main()
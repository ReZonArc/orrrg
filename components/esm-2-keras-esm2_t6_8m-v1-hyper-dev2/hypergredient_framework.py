#!/usr/bin/env python3
"""
Hypergredient Framework Architecture

Revolutionary formulation design system for cosmetic ingredients optimization.
Implements advanced algorithms for ingredient selection, compatibility analysis,
and multi-objective formulation optimization.

Based on the Hypergredient Framework specification:
Hypergredient(*) := {ingredient_i | function(*) ∈ F_i, 
                     constraints ∈ C_i, 
                     performance ∈ P_i}
"""

import json
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class HypergredientClass(Enum):
    """Core Hypergredient Classes"""
    CT = "H.CT"  # Cellular Turnover Agents
    CS = "H.CS"  # Collagen Synthesis Promoters
    AO = "H.AO"  # Antioxidant Systems
    BR = "H.BR"  # Barrier Repair Complex
    ML = "H.ML"  # Melanin Modulators
    HY = "H.HY"  # Hydration Systems
    AI = "H.AI"  # Anti-Inflammatory Agents
    MB = "H.MB"  # Microbiome Balancers
    SE = "H.SE"  # Sebum Regulators
    PD = "H.PD"  # Penetration/Delivery Enhancers


@dataclass
class Hypergredient:
    """Core Hypergredient data structure"""
    id: str
    name: str
    inci_name: str
    hypergredient_class: HypergredientClass
    primary_function: str
    secondary_functions: List[str] = field(default_factory=list)
    
    # Performance metrics
    efficacy_score: float = 0.0  # 0-10 scale
    bioavailability: float = 0.0  # 0-1 scale
    safety_score: float = 0.0  # 0-10 scale
    stability_index: float = 0.0  # 0-1 scale
    
    # Physical properties
    ph_min: float = 4.0
    ph_max: float = 9.0
    cost_per_gram: float = 0.0  # ZAR
    
    # Interactions
    incompatibilities: List[str] = field(default_factory=list)
    synergies: List[str] = field(default_factory=list)
    
    # Clinical evidence
    clinical_evidence_level: str = "moderate"  # weak, moderate, strong
    
    def calculate_composite_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        metrics = {
            'efficacy': self.efficacy_score / 10.0,
            'bioavailability': self.bioavailability,
            'safety': self.safety_score / 10.0,
            'stability': self.stability_index,
            'cost_efficiency': 1.0 / (self.cost_per_gram + 1.0)  # Inverse cost
        }
        
        return sum(metrics.get(metric, 0) * weight 
                  for metric, weight in weights.items())


@dataclass
class FormulationConstraints:
    """Formulation constraints and requirements"""
    ph_range: Tuple[float, float] = (4.5, 7.0)
    total_actives_range: Tuple[float, float] = (5.0, 25.0)  # percentage
    max_budget: float = 1000.0  # ZAR
    max_irritation_score: float = 5.0
    required_stability_months: int = 24
    excluded_ingredients: List[str] = field(default_factory=list)
    preferred_ingredients: List[str] = field(default_factory=list)


@dataclass
class FormulationRequest:
    """User formulation request"""
    target_concerns: List[str]
    secondary_concerns: List[str] = field(default_factory=list)
    skin_type: str = "normal"
    budget: float = 1000.0
    preferences: List[str] = field(default_factory=list)
    constraints: FormulationConstraints = field(default_factory=FormulationConstraints)


@dataclass
class FormulationResult:
    """Optimized formulation result"""
    selected_hypergredients: Dict[str, Dict[str, Any]]
    total_cost: float
    predicted_efficacy: float
    safety_score: float
    stability_months: int
    synergy_score: float
    reasoning: Dict[str, str]


@dataclass
class PersonaProfile:
    """User persona profile for personalized training"""
    persona_id: str
    name: str
    description: str
    skin_type: str
    primary_concerns: List[str]
    sensitivity_level: float  # 0.0 = not sensitive, 1.0 = highly sensitive
    budget_preference: str  # "budget", "mid-range", "premium"
    ingredient_preferences: List[str] = field(default_factory=list)
    ingredient_aversions: List[str] = field(default_factory=list)
    efficacy_tolerance: float = 0.5  # How long they're willing to wait for results
    safety_priority: float = 0.8  # How much they prioritize safety vs efficacy
    natural_preference: float = 0.3  # Preference for natural ingredients
    
    def get_persona_weights(self) -> Dict[str, float]:
        """Get ML model weights based on persona characteristics"""
        weights = {
            'efficacy': 1.0 - self.safety_priority,
            'safety': self.safety_priority,
            'cost_efficiency': 1.0 if self.budget_preference == "budget" else 0.5,
            'bioavailability': self.efficacy_tolerance,
            'stability': 0.8  # Generally important
        }
        return weights


@dataclass 
class PersonaTrainingData:
    """Training data for a specific persona"""
    persona_id: str
    formulation_requests: List[FormulationRequest] = field(default_factory=list)
    formulation_results: List[FormulationResult] = field(default_factory=list)
    feedback_scores: List[Dict[str, float]] = field(default_factory=list)
    timestamp_created: float = 0.0
    training_iterations: int = 0


class PersonaTrainingSystem:
    """Persona-based training system for hypergredient models"""
    
    def __init__(self):
        self.personas: Dict[str, PersonaProfile] = {}
        self.training_data: Dict[str, PersonaTrainingData] = {}
        self.active_persona: Optional[str] = None
        self._initialize_default_personas()
    
    def _initialize_default_personas(self):
        """Initialize common skin care personas"""
        
        # Sensitive skin persona
        sensitive_persona = PersonaProfile(
            persona_id="sensitive_skin",
            name="Sensitive Skin Specialist",
            description="Prioritizes gentle, hypoallergenic formulations",
            skin_type="sensitive",
            primary_concerns=["sensitivity", "redness", "barrier_repair"],
            sensitivity_level=0.9,
            budget_preference="mid-range",
            ingredient_preferences=["ceramides", "niacinamide", "hyaluronic_acid"],
            ingredient_aversions=["alcohol", "fragrances", "essential_oils"],
            efficacy_tolerance=0.8,  # Willing to wait longer for gentler results
            safety_priority=0.95,    # Safety over efficacy
            natural_preference=0.6
        )
        
        # Anti-aging persona
        anti_aging_persona = PersonaProfile(
            persona_id="anti_aging",
            name="Anti-Aging Enthusiast", 
            description="Seeks powerful anti-aging ingredients with proven efficacy",
            skin_type="normal",
            primary_concerns=["wrinkles", "firmness", "hyperpigmentation"],
            sensitivity_level=0.3,
            budget_preference="premium",
            ingredient_preferences=["retinoids", "peptides", "vitamin_c"],
            ingredient_aversions=["parabens"],
            efficacy_tolerance=0.3,  # Wants fast results
            safety_priority=0.6,     # Efficacy over safety (within reason)
            natural_preference=0.2
        )
        
        # Acne-prone persona
        acne_persona = PersonaProfile(
            persona_id="acne_prone",
            name="Acne-Prone Specialist",
            description="Focuses on oil control and acne treatment",
            skin_type="oily",
            primary_concerns=["acne", "oiliness", "pore_size"],
            sensitivity_level=0.4,
            budget_preference="budget",
            ingredient_preferences=["salicylic_acid", "niacinamide", "zinc"],
            ingredient_aversions=["comedogenic_oils", "heavy_moisturizers"],
            efficacy_tolerance=0.4,
            safety_priority=0.7,
            natural_preference=0.3
        )
        
        # Natural beauty persona
        natural_persona = PersonaProfile(
            persona_id="natural_beauty",
            name="Natural Beauty Advocate",
            description="Prefers natural and organic ingredients",
            skin_type="normal",
            primary_concerns=["dryness", "general_health"],
            sensitivity_level=0.5,
            budget_preference="mid-range", 
            ingredient_preferences=["plant_extracts", "oils", "botanicals"],
            ingredient_aversions=["sulfates", "parabens", "synthetic_fragrances"],
            efficacy_tolerance=0.7,
            safety_priority=0.8,
            natural_preference=0.9
        )
        
        # Add to system
        for persona in [sensitive_persona, anti_aging_persona, acne_persona, natural_persona]:
            self.add_persona(persona)
    
    def add_persona(self, persona: PersonaProfile):
        """Add a new persona to the system"""
        self.personas[persona.persona_id] = persona
        self.training_data[persona.persona_id] = PersonaTrainingData(
            persona_id=persona.persona_id
        )
    
    def set_active_persona(self, persona_id: str):
        """Set the active persona for predictions"""
        if persona_id not in self.personas:
            raise ValueError(f"Persona {persona_id} not found")
        self.active_persona = persona_id
    
    def train_persona(self, persona_id: str, requests: List[FormulationRequest], 
                     results: List[FormulationResult], feedback: List[Dict[str, float]]):
        """Train a specific persona with formulation data"""
        if persona_id not in self.personas:
            raise ValueError(f"Persona {persona_id} not found")
        
        training_data = self.training_data[persona_id]
        training_data.formulation_requests.extend(requests)
        training_data.formulation_results.extend(results)
        training_data.feedback_scores.extend(feedback)
        training_data.training_iterations += 1
        
        print(f"🎭 Trained persona '{persona_id}' with {len(requests)} formulations")
        print(f"   Total training samples: {len(training_data.formulation_requests)}")
    
    def get_persona_adjusted_features(self, requirements: FormulationRequest, 
                                    base_features: Dict[str, float]) -> Dict[str, float]:
        """Adjust features based on active persona"""
        if not self.active_persona:
            return base_features
        
        persona = self.personas[self.active_persona]
        adjusted_features = base_features.copy()
        
        # Adjust based on persona characteristics
        adjusted_features['persona_sensitivity'] = persona.sensitivity_level
        adjusted_features['persona_safety_priority'] = persona.safety_priority
        adjusted_features['persona_natural_preference'] = persona.natural_preference
        adjusted_features['persona_efficacy_tolerance'] = persona.efficacy_tolerance
        
        # Adjust budget normalization based on persona budget preference
        budget_multipliers = {"budget": 0.5, "mid-range": 1.0, "premium": 1.5}
        budget_mult = budget_multipliers.get(persona.budget_preference, 1.0)
        adjusted_features['budget_normalized'] *= budget_mult
        
        return adjusted_features
    
    def get_persona_ingredient_preferences(self, persona_id: Optional[str] = None) -> Dict[str, float]:
        """Get ingredient preference scores for a persona"""
        if persona_id is None:
            persona_id = self.active_persona
        
        if not persona_id or persona_id not in self.personas:
            return {}
        
        persona = self.personas[persona_id]
        preferences = {}
        
        # Positive preferences
        for ingredient in persona.ingredient_preferences:
            preferences[ingredient] = 1.5  # Boost preferred ingredients
        
        # Negative preferences  
        for ingredient in persona.ingredient_aversions:
            preferences[ingredient] = 0.1  # Penalize avoided ingredients
        
        return preferences
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of persona training status"""
        summary = {
            "total_personas": len(self.personas),
            "active_persona": self.active_persona,
            "personas": {}
        }
        
        for persona_id, persona in self.personas.items():
            training_data = self.training_data[persona_id]
            summary["personas"][persona_id] = {
                "name": persona.name,
                "description": persona.description,
                "training_samples": len(training_data.formulation_requests),
                "training_iterations": training_data.training_iterations,
                "skin_type": persona.skin_type,
                "primary_concerns": persona.primary_concerns,
                "sensitivity_level": persona.sensitivity_level
            }
        
        return summary


class HypergredientDatabase:
    """Dynamic Hypergredient Database"""
    
    def __init__(self):
        self.hypergredients: Dict[str, Hypergredient] = {}
        self.interaction_matrix: Dict[Tuple[str, str], float] = {}
        self._initialize_database()
        self._initialize_interactions()
    
    def _initialize_database(self):
        """Initialize database with core hypergredients"""
        
        # H.CT - Cellular Turnover Agents
        self.add_hypergredient(Hypergredient(
            id="tretinoin",
            name="Tretinoin",
            inci_name="Tretinoin",
            hypergredient_class=HypergredientClass.CT,
            primary_function="Accelerated cellular turnover",
            secondary_functions=["Collagen stimulation", "Hyperpigmentation reduction"],
            efficacy_score=10.0,
            bioavailability=0.85,
            safety_score=6.0,
            stability_index=0.3,  # UV-sensitive
            ph_min=5.5,
            ph_max=6.5,
            cost_per_gram=15.00,
            incompatibilities=["benzoyl_peroxide", "strong_acids"],
            clinical_evidence_level="strong"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="bakuchiol",
            name="Bakuchiol",
            inci_name="Bakuchiol",
            hypergredient_class=HypergredientClass.CT,
            primary_function="Gentle retinol alternative",
            secondary_functions=["Antioxidant", "Anti-inflammatory"],
            efficacy_score=7.0,
            bioavailability=0.70,
            safety_score=9.0,
            stability_index=0.9,
            ph_min=4.0,
            ph_max=9.0,
            cost_per_gram=240.00,
            synergies=["vitamin_c", "niacinamide"],
            clinical_evidence_level="moderate"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="retinol",
            name="Retinol",
            inci_name="Retinol",
            hypergredient_class=HypergredientClass.CT,
            primary_function="Cellular turnover stimulation",
            secondary_functions=["Wrinkle reduction", "Texture improvement"],
            efficacy_score=8.0,
            bioavailability=0.60,
            safety_score=7.0,
            stability_index=0.4,  # Oxygen-sensitive
            ph_min=5.5,
            ph_max=6.5,
            cost_per_gram=180.00,
            incompatibilities=["aha", "bha", "vitamin_c"],
            clinical_evidence_level="strong"
        ))
        
        # H.CS - Collagen Synthesis Promoters
        self.add_hypergredient(Hypergredient(
            id="matrixyl_3000",
            name="Matrixyl 3000",
            inci_name="Palmitoyl Tripeptide-1, Palmitoyl Tetrapeptide-7",
            hypergredient_class=HypergredientClass.CS,
            primary_function="Signal peptide collagen stimulation",
            secondary_functions=["Wrinkle reduction", "Skin firmness"],
            efficacy_score=9.0,
            bioavailability=0.75,
            safety_score=9.0,
            stability_index=0.8,
            ph_min=5.0,
            ph_max=7.0,
            cost_per_gram=120.00,
            synergies=["vitamin_c", "niacinamide"],
            clinical_evidence_level="strong"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="vitamin_c_sap",
            name="Vitamin C (Sodium Ascorbyl Phosphate)",
            inci_name="Sodium Ascorbyl Phosphate",
            hypergredient_class=HypergredientClass.CS,
            primary_function="Stable vitamin C for collagen synthesis",
            secondary_functions=["Antioxidant", "Brightening"],
            efficacy_score=6.0,
            bioavailability=0.70,
            safety_score=9.0,
            stability_index=0.8,
            ph_min=6.0,
            ph_max=8.0,
            cost_per_gram=70.00,
            synergies=["niacinamide", "peptides"],
            clinical_evidence_level="moderate"
        ))
        
        # Re-add the original vitamin C LAA
        self.add_hypergredient(Hypergredient(
            id="vitamin_c_laa",
            name="Vitamin C (L-Ascorbic Acid)",
            inci_name="L-Ascorbic Acid",
            hypergredient_class=HypergredientClass.CS,
            primary_function="Collagen synthesis cofactor",
            secondary_functions=["Antioxidant", "Brightening"],
            efficacy_score=8.0,
            bioavailability=0.85,
            safety_score=7.0,
            stability_index=0.2,  # Very unstable
            ph_min=3.0,
            ph_max=4.0,
            cost_per_gram=85.00,
            incompatibilities=["copper_peptides", "retinol"],
            synergies=["vitamin_e", "ferulic_acid"],
            clinical_evidence_level="strong"
        ))
        
        # Add more antioxidants
        self.add_hypergredient(Hypergredient(
            id="resveratrol",
            name="Resveratrol",
            inci_name="Resveratrol",
            hypergredient_class=HypergredientClass.AO,
            primary_function="Polyphenol antioxidant",
            secondary_functions=["Anti-inflammatory", "Longevity"],
            efficacy_score=7.0,
            bioavailability=0.60,
            safety_score=8.0,
            stability_index=0.6,
            ph_min=4.0,
            ph_max=7.0,
            cost_per_gram=190.00,
            synergies=["vitamin_e", "ferulic_acid"],
            clinical_evidence_level="moderate"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="ferulic_acid",
            name="Ferulic Acid",
            inci_name="Ferulic Acid",
            hypergredient_class=HypergredientClass.AO,
            primary_function="Antioxidant stabilizer",
            secondary_functions=["UV protection", "Vitamin C stabilizer"],
            efficacy_score=6.0,
            bioavailability=0.75,
            safety_score=9.0,
            stability_index=0.7,
            ph_min=4.0,
            ph_max=6.0,
            cost_per_gram=125.00,
            synergies=["vitamin_c_laa", "vitamin_e"],
            clinical_evidence_level="strong"
        ))
        
        # H.AO - Antioxidant Systems
        self.add_hypergredient(Hypergredient(
            id="astaxanthin",
            name="Astaxanthin",
            inci_name="Astaxanthin",
            hypergredient_class=HypergredientClass.AO,
            primary_function="Powerful antioxidant protection",
            secondary_functions=["UV protection", "Anti-inflammatory"],
            efficacy_score=9.0,
            bioavailability=0.65,
            safety_score=9.0,
            stability_index=0.6,  # Light-sensitive
            ph_min=4.0,
            ph_max=8.0,
            cost_per_gram=360.00,
            synergies=["vitamin_e", "resveratrol"],
            clinical_evidence_level="strong"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="vitamin_e",
            name="Vitamin E",
            inci_name="Tocopherol",
            hypergredient_class=HypergredientClass.AO,
            primary_function="Lipid antioxidant protection",
            secondary_functions=["Stabilizer", "Moisturizer"],
            efficacy_score=6.0,
            bioavailability=0.90,
            safety_score=9.0,
            stability_index=0.9,
            ph_min=4.0,
            ph_max=9.0,
            cost_per_gram=50.00,
            synergies=["vitamin_c", "ferulic_acid"],
            clinical_evidence_level="strong"
        ))
        
        # Add more brightening agents  
        self.add_hypergredient(Hypergredient(
            id="tranexamic_acid",
            name="Tranexamic Acid",
            inci_name="Tranexamic Acid",
            hypergredient_class=HypergredientClass.ML,
            primary_function="Melasma and hyperpigmentation treatment",
            secondary_functions=["Anti-inflammatory", "Vascular protection"],
            efficacy_score=8.0,
            bioavailability=0.85,
            safety_score=9.0,
            stability_index=0.9,
            ph_min=4.0,
            ph_max=8.0,
            cost_per_gram=220.00,
            synergies=["vitamin_c", "niacinamide"],
            clinical_evidence_level="strong"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="kojic_acid",
            name="Kojic Acid",
            inci_name="Kojic Acid",
            hypergredient_class=HypergredientClass.ML,
            primary_function="Tyrosinase inhibitor",
            secondary_functions=["Antioxidant"],
            efficacy_score=7.0,
            bioavailability=0.75,
            safety_score=7.0,
            stability_index=0.5,  # Can be unstable
            ph_min=4.0,
            ph_max=6.0,
            cost_per_gram=95.00,
            synergies=["alpha_arbutin", "vitamin_c"],
            clinical_evidence_level="moderate"
        ))
        
        # Add barrier repair ingredients
        self.add_hypergredient(Hypergredient(
            id="ceramide_np",
            name="Ceramide NP",
            inci_name="Ceramide NP",
            hypergredient_class=HypergredientClass.BR,
            primary_function="Barrier lipid restoration",
            secondary_functions=["Moisturizing", "Anti-aging"],
            efficacy_score=8.0,
            bioavailability=0.70,
            safety_score=10.0,
            stability_index=0.8,
            ph_min=4.0,
            ph_max=8.0,
            cost_per_gram=280.00,
            synergies=["cholesterol", "fatty_acids"],
            clinical_evidence_level="strong"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="cholesterol",
            name="Cholesterol",
            inci_name="Cholesterol",
            hypergredient_class=HypergredientClass.BR,
            primary_function="Barrier lipid component",
            secondary_functions=["Membrane fluidity"],
            efficacy_score=6.0,
            bioavailability=0.60,
            safety_score=10.0,
            stability_index=0.9,
            ph_min=4.0,
            ph_max=9.0,
            cost_per_gram=85.00,
            synergies=["ceramide_np", "fatty_acids"],
            clinical_evidence_level="strong"
        ))
        
        # H.HY - Hydration Systems
        self.add_hypergredient(Hypergredient(
            id="hyaluronic_acid",
            name="Hyaluronic Acid",
            inci_name="Sodium Hyaluronate",
            hypergredient_class=HypergredientClass.HY,
            primary_function="Multi-depth hydration",
            secondary_functions=["Plumping", "Barrier support"],
            efficacy_score=8.0,
            bioavailability=0.85,
            safety_score=10.0,
            stability_index=0.9,
            ph_min=4.0,
            ph_max=9.0,
            cost_per_gram=150.00,
            synergies=["ceramides", "peptides"],
            clinical_evidence_level="strong"
        ))
        
        # H.AI - Anti-Inflammatory Agents
        self.add_hypergredient(Hypergredient(
            id="niacinamide",
            name="Niacinamide",
            inci_name="Niacinamide",
            hypergredient_class=HypergredientClass.AI,
            primary_function="Anti-inflammatory",
            secondary_functions=["Sebum regulation", "Barrier repair", "Brightening"],
            efficacy_score=8.0,
            bioavailability=0.90,
            safety_score=9.0,
            stability_index=0.95,
            ph_min=5.0,
            ph_max=7.0,
            cost_per_gram=45.00,
            synergies=["zinc", "peptides", "hyaluronic_acid"],
            clinical_evidence_level="strong"
        ))
    
    def add_hypergredient(self, hypergredient: Hypergredient):
        """Add hypergredient to database"""
        self.hypergredients[hypergredient.id] = hypergredient
    
    def _initialize_interactions(self):
        """Initialize interaction matrix"""
        interactions = {
            ("H.CT", "H.CS"): 1.5,  # Positive synergy
            ("H.CT", "H.AO"): 0.8,  # Mild antagonism (oxidation)
            ("H.CS", "H.AO"): 2.0,  # Strong synergy
            ("H.BR", "H.HY"): 2.5,  # Excellent synergy
            ("H.ML", "H.AO"): 1.8,  # Good synergy
            ("H.AI", "H.MB"): 2.2,  # Strong synergy
            ("H.SE", "H.CT"): 0.6,  # Potential irritation
            ("H.CS", "H.HY"): 1.6,  # Good synergy
            ("H.AI", "H.HY"): 1.4,  # Moderate synergy
        }
        
        # Create bidirectional interactions
        for (class1, class2), score in interactions.items():
            self.interaction_matrix[(class1, class2)] = score
            self.interaction_matrix[(class2, class1)] = score
    
    def get_by_class(self, hypergredient_class: HypergredientClass) -> List[Hypergredient]:
        """Get all hypergredients by class"""
        return [h for h in self.hypergredients.values() 
                if h.hypergredient_class == hypergredient_class]
    
    def search(self, query: str) -> List[Hypergredient]:
        """Search hypergredients by name or function"""
        query_lower = query.lower()
        results = []
        
        for hypergredient in self.hypergredients.values():
            if (query_lower in hypergredient.name.lower() or
                query_lower in hypergredient.primary_function.lower() or
                any(query_lower in func.lower() for func in hypergredient.secondary_functions)):
                results.append(hypergredient)
        
        return results


class HypergredientOptimizer:
    """Multi-Objective Formulation Optimizer"""
    
    def __init__(self, database: HypergredientDatabase):
        self.database = database
        self.concern_to_class_mapping = {
            'wrinkles': [HypergredientClass.CT, HypergredientClass.CS],
            'aging': [HypergredientClass.CT, HypergredientClass.CS, HypergredientClass.AO],
            'acne': [HypergredientClass.CT, HypergredientClass.AI, HypergredientClass.SE],
            'hyperpigmentation': [HypergredientClass.ML, HypergredientClass.AO],
            'dryness': [HypergredientClass.HY, HypergredientClass.BR],
            'sensitivity': [HypergredientClass.AI, HypergredientClass.BR],
            'dullness': [HypergredientClass.ML, HypergredientClass.AO, HypergredientClass.CT],
            'firmness': [HypergredientClass.CS, HypergredientClass.AO],
            'texture': [HypergredientClass.CT, HypergredientClass.HY],
            'redness': [HypergredientClass.AI, HypergredientClass.BR]
        }
    
    def optimize_formulation(self, request: FormulationRequest) -> FormulationResult:
        """Generate optimal formulation using hypergredients"""
        
        # Define objective weights (check for adaptive weights first)
        if hasattr(self, '_adaptive_weights') and self._adaptive_weights:
            objective_weights = self._adaptive_weights
        else:
            objective_weights = {
                'efficacy': 0.35,
                'safety': 0.25,
                'stability': 0.20,
                'cost_efficiency': 0.15,
                'bioavailability': 0.05
            }
        
        selected_hypergredients = {}
        total_cost = 0.0
        reasoning = {}
        
        # Process each concern
        for concern in request.target_concerns + request.secondary_concerns:
            if concern not in self.concern_to_class_mapping:
                continue
            
            classes = self.concern_to_class_mapping[concern]
            weight = 1.0 if concern in request.target_concerns else 0.5
            
            for hypergredient_class in classes:
                if hypergredient_class.value in selected_hypergredients:
                    continue  # Already selected for this class
                
                candidates = self.database.get_by_class(hypergredient_class)
                if not candidates:
                    continue
                
                # Score each candidate
                best_candidate = None
                best_score = -1.0
                
                for candidate in candidates:
                    if candidate.id in request.constraints.excluded_ingredients:
                        continue
                    
                    # Check budget constraint
                    estimated_usage = self._estimate_usage_percentage(candidate)
                    estimated_cost = candidate.cost_per_gram * estimated_usage / 100.0 * 50  # 50g formulation
                    
                    if total_cost + estimated_cost > request.budget:
                        continue
                    
                    # Calculate compatibility with already selected ingredients
                    compatibility_score = self._calculate_compatibility(
                        candidate, list(selected_hypergredients.keys())
                    )
                    
                    if compatibility_score < 0.5:  # Too incompatible
                        continue
                    
                    # Calculate composite score
                    score = candidate.calculate_composite_score(objective_weights)
                    score *= weight * compatibility_score
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
                
                if best_candidate:
                    usage_percentage = self._estimate_usage_percentage(best_candidate)
                    ingredient_cost = best_candidate.cost_per_gram * usage_percentage / 100.0 * 50
                    
                    selected_hypergredients[hypergredient_class.value] = {
                        'ingredient': best_candidate,
                        'percentage': usage_percentage,
                        'cost': ingredient_cost,
                        'score': best_score,
                        'reasoning': self._generate_reasoning(best_candidate, concern)
                    }
                    
                    total_cost += ingredient_cost
                    reasoning[hypergredient_class.value] = self._generate_reasoning(best_candidate, concern)
        
        # Calculate overall metrics
        efficacy_score = self._calculate_predicted_efficacy(selected_hypergredients)
        safety_score = self._calculate_overall_safety(selected_hypergredients)
        synergy_score = self._calculate_synergy_score(selected_hypergredients)
        stability_months = self._estimate_stability(selected_hypergredients)
        
        return FormulationResult(
            selected_hypergredients=selected_hypergredients,
            total_cost=total_cost,
            predicted_efficacy=efficacy_score,
            safety_score=safety_score,
            stability_months=stability_months,
            synergy_score=synergy_score,
            reasoning=reasoning
        )
    
    def _estimate_usage_percentage(self, hypergredient: Hypergredient) -> float:
        """Estimate typical usage percentage for hypergredient"""
        usage_map = {
            HypergredientClass.CT: 1.0,  # Low percentage actives
            HypergredientClass.CS: 3.0,  # Peptides typically 2-5%
            HypergredientClass.AO: 0.5,  # Strong antioxidants
            HypergredientClass.BR: 2.0,  # Barrier ingredients
            HypergredientClass.ML: 2.0,  # Brightening agents
            HypergredientClass.HY: 1.0,  # Hyaluronic acid
            HypergredientClass.AI: 5.0,  # Niacinamide can go higher
            HypergredientClass.MB: 1.0,  # Prebiotics/probiotics
            HypergredientClass.SE: 2.0,  # Sebum regulators
            HypergredientClass.PD: 1.0,  # Penetration enhancers
        }
        
        base_percentage = usage_map.get(hypergredient.hypergredient_class, 1.0)
        
        # Adjust based on potency and safety
        if hypergredient.efficacy_score > 8.0 and hypergredient.safety_score < 7.0:
            base_percentage *= 0.5  # Reduce for high potency, lower safety
        elif hypergredient.safety_score > 9.0:
            base_percentage *= 1.2  # Can use more of very safe ingredients
        
        return min(base_percentage, 10.0)  # Cap at 10%
    
    def _calculate_compatibility(self, candidate: Hypergredient, selected_ids: List[str]) -> float:
        """Calculate compatibility score with selected ingredients"""
        if not selected_ids:
            return 1.0
        
        compatibility_scores = []
        
        for selected_id in selected_ids:
            selected_class = None
            for hypergredient in self.database.hypergredients.values():
                if hypergredient.hypergredient_class.value == selected_id:
                    selected_class = hypergredient.hypergredient_class.value
                    break
            
            if selected_class:
                interaction_key = (candidate.hypergredient_class.value, selected_class)
                interaction_score = self.database.interaction_matrix.get(interaction_key, 1.0)
                compatibility_scores.append(min(interaction_score, 2.0) / 2.0)  # Normalize to 0-1
        
        return sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 1.0
    
    def _calculate_predicted_efficacy(self, selected: Dict[str, Dict[str, Any]]) -> float:
        """Calculate predicted formulation efficacy"""
        if not selected:
            return 0.0
        
        efficacy_scores = []
        for data in selected.values():
            ingredient = data['ingredient']
            percentage = data['percentage']
            
            # Weight by usage percentage and bioavailability
            weighted_efficacy = (ingredient.efficacy_score / 10.0) * (percentage / 10.0) * ingredient.bioavailability
            efficacy_scores.append(weighted_efficacy)
        
        # Apply synergy bonus
        base_efficacy = sum(efficacy_scores) / len(efficacy_scores)
        synergy_bonus = self._calculate_synergy_score(selected) * 0.2
        
        return min(base_efficacy + synergy_bonus, 1.0)
    
    def _calculate_overall_safety(self, selected: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall formulation safety score"""
        if not selected:
            return 10.0
        
        safety_scores = [data['ingredient'].safety_score for data in selected.values()]
        return sum(safety_scores) / len(safety_scores)
    
    def _calculate_synergy_score(self, selected: Dict[str, Dict[str, Any]]) -> float:
        """Calculate synergy score between selected ingredients"""
        if len(selected) < 2:
            return 0.0
        
        synergy_scores = []
        classes = list(selected.keys())
        
        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                interaction_key = (class1, class2)
                interaction_score = self.database.interaction_matrix.get(interaction_key, 1.0)
                if interaction_score > 1.0:  # Positive synergy
                    synergy_scores.append((interaction_score - 1.0) / 1.5)  # Normalize
        
        return sum(synergy_scores) / len(synergy_scores) if synergy_scores else 0.0
    
    def _estimate_stability(self, selected: Dict[str, Dict[str, Any]]) -> int:
        """Estimate formulation stability in months"""
        if not selected:
            return 24
        
        stability_indices = [data['ingredient'].stability_index for data in selected.values()]
        min_stability = min(stability_indices)
        
        # Convert stability index to months (0.0 = 6 months, 1.0 = 24 months)
        return int(6 + (min_stability * 18))
    
    def _generate_reasoning(self, hypergredient: Hypergredient, concern: str) -> str:
        """Generate reasoning for ingredient selection"""
        reasons = []
        
        if hypergredient.efficacy_score >= 8.0:
            reasons.append("High efficacy")
        if hypergredient.safety_score >= 9.0:
            reasons.append("Excellent safety profile")
        if hypergredient.stability_index >= 0.8:
            reasons.append("Good stability")
        if hypergredient.cost_per_gram <= 100.0:
            reasons.append("Cost-effective")
        if hypergredient.clinical_evidence_level == "strong":
            reasons.append("Strong clinical evidence")
        
        base_reason = f"Selected for {concern} targeting"
        if reasons:
            return f"{base_reason}: {', '.join(reasons)}"
        else:
            return base_reason


class HypergredientAnalyzer:
    """Analysis and reporting tools for hypergredients"""
    
    def __init__(self, database: HypergredientDatabase):
        self.database = database
    
    def generate_compatibility_report(self, ingredient_ids: List[str]) -> Dict[str, Any]:
        """Generate compatibility analysis report"""
        ingredients = [self.database.hypergredients[id] for id in ingredient_ids 
                      if id in self.database.hypergredients]
        
        if len(ingredients) < 2:
            return {"error": "Need at least 2 ingredients for compatibility analysis"}
        
        compatibility_matrix = {}
        warnings = []
        recommendations = []
        
        for i, ing1 in enumerate(ingredients):
            for ing2 in ingredients[i+1:]:
                key = f"{ing1.name} + {ing2.name}"
                
                # Check direct incompatibilities
                if (ing2.id in ing1.incompatibilities or 
                    ing1.id in ing2.incompatibilities):
                    compatibility_matrix[key] = "Incompatible"
                    warnings.append(f"⚠️ {ing1.name} and {ing2.name} are incompatible")
                    continue
                
                # Check pH compatibility
                ph_overlap = min(ing1.ph_max, ing2.ph_max) - max(ing1.ph_min, ing2.ph_min)
                if ph_overlap <= 0:
                    compatibility_matrix[key] = "pH Incompatible"
                    warnings.append(f"⚠️ {ing1.name} and {ing2.name} have incompatible pH ranges")
                    continue
                
                # Check for synergies
                class_interaction = self.database.interaction_matrix.get(
                    (ing1.hypergredient_class.value, ing2.hypergredient_class.value), 1.0
                )
                
                if class_interaction > 1.5:
                    compatibility_matrix[key] = "Excellent Synergy"
                    recommendations.append(f"✅ {ing1.name} and {ing2.name} work synergistically")
                elif class_interaction > 1.0:
                    compatibility_matrix[key] = "Good Synergy"
                elif class_interaction >= 0.8:
                    compatibility_matrix[key] = "Compatible"
                else:
                    compatibility_matrix[key] = "Potentially Problematic"
                    warnings.append(f"⚠️ {ing1.name} and {ing2.name} may interfere with each other")
        
        return {
            "compatibility_matrix": compatibility_matrix,
            "warnings": warnings,
            "recommendations": recommendations,
            "overall_compatibility": "Good" if not warnings else "Needs Attention"
        }
    
    def generate_ingredient_profile(self, ingredient_id: str) -> Dict[str, Any]:
        """Generate detailed ingredient profile"""
        if ingredient_id not in self.database.hypergredients:
            return {"error": f"Ingredient '{ingredient_id}' not found"}
        
        ingredient = self.database.hypergredients[ingredient_id]
        
        # Calculate derived metrics
        cost_efficiency = ingredient.efficacy_score / max(ingredient.cost_per_gram, 1.0)
        risk_benefit_ratio = ingredient.efficacy_score / max(10.0 - ingredient.safety_score, 1.0)
        
        return {
            "basic_info": {
                "name": ingredient.name,
                "inci_name": ingredient.inci_name,
                "class": ingredient.hypergredient_class.value,
                "primary_function": ingredient.primary_function,
                "secondary_functions": ingredient.secondary_functions
            },
            "performance_metrics": {
                "efficacy_score": ingredient.efficacy_score,
                "bioavailability": ingredient.bioavailability,
                "safety_score": ingredient.safety_score,
                "stability_index": ingredient.stability_index
            },
            "formulation_properties": {
                "ph_range": f"{ingredient.ph_min}-{ingredient.ph_max}",
                "cost_per_gram": ingredient.cost_per_gram,
                "typical_usage": f"{self._get_typical_usage(ingredient)}%"
            },
            "interactions": {
                "incompatibilities": ingredient.incompatibilities,
                "synergies": ingredient.synergies
            },
            "derived_metrics": {
                "cost_efficiency": round(cost_efficiency, 2),
                "risk_benefit_ratio": round(risk_benefit_ratio, 2),
                "clinical_evidence": ingredient.clinical_evidence_level
            }
        }
    
    def _get_typical_usage(self, ingredient: Hypergredient) -> float:
        """Get typical usage percentage for ingredient"""
        optimizer = HypergredientOptimizer(self.database)
        return optimizer._estimate_usage_percentage(ingredient)


class FormulationEvolution:
    """Evolutionary Formulation Improvement System"""
    
    def __init__(self, base_formula: FormulationResult):
        self.generation = 0
        self.formula = base_formula
        self.performance_history = []
        self.market_feedback = []
    
    def add_market_feedback(self, feedback: Dict[str, Any]):
        """Add market feedback for evolutionary improvement"""
        self.market_feedback.append({
            'generation': self.generation,
            'feedback': feedback,
            'timestamp': self.generation  # Simplified timestamp
        })
    
    def evolve(self, database: HypergredientDatabase, 
               target_improvements: Dict[str, float]) -> FormulationResult:
        """
        Evolve formulation based on feedback and target improvements
        
        Args:
            database: Hypergredient database
            target_improvements: Dict of metrics to improve with target values
        """
        # Analyze performance gaps
        gaps = self._analyze_performance_gaps(target_improvements)
        
        # Search for better hypergredients
        optimizer = HypergredientOptimizer(database)
        
        # Create enhanced request based on gaps
        enhanced_request = self._create_enhanced_request(gaps)
        
        # Generate next generation formula
        next_gen_formula = optimizer.optimize_formulation(enhanced_request)
        
        # Track performance history
        self.performance_history.append({
            'generation': self.generation,
            'efficacy': self.formula.predicted_efficacy,
            'safety': self.formula.safety_score,
            'synergy': self.formula.synergy_score,
            'cost': self.formula.total_cost
        })
        
        # Update formula and increment generation
        self.formula = next_gen_formula
        self.generation += 1
        
        return next_gen_formula
    
    def _analyze_performance_gaps(self, targets: Dict[str, float]) -> Dict[str, Dict]:
        """Analyze gaps between current performance and targets"""
        current_metrics = {
            'efficacy': self.formula.predicted_efficacy,
            'safety': self.formula.safety_score / 10.0,  # Normalize to 0-1
            'synergy': self.formula.synergy_score,
            'cost_efficiency': 1.0 / (self.formula.total_cost / 1000.0)  # Inverse normalized cost
        }
        
        gaps = {}
        for metric, target in targets.items():
            if metric in current_metrics:
                gap = target - current_metrics[metric]
                if gap > 0:  # Only consider improvements needed
                    gaps[metric] = {
                        'current': current_metrics[metric],
                        'target': target,
                        'gap': gap,
                        'priority': gap / max(current_metrics[metric], 0.1)  # Relative gap
                    }
        
        return gaps
    
    def _create_enhanced_request(self, gaps: Dict[str, Dict]) -> FormulationRequest:
        """Create enhanced request based on performance gaps"""
        # Start with original concerns but add new ones based on gaps
        concerns = ['wrinkles', 'firmness']
        secondary_concerns = ['dryness']
        
        if 'efficacy' in gaps and gaps['efficacy']['gap'] > 0.1:
            concerns.extend(['aging', 'texture'])
        
        if 'safety' in gaps and gaps['safety']['gap'] > 0.1:
            secondary_concerns.append('sensitivity')
        
        # Adjust budget based on cost efficiency needs
        budget = 1000.0
        if 'cost_efficiency' in gaps:
            budget = max(800.0, budget - (gaps['cost_efficiency']['gap'] * 200))
        
        return FormulationRequest(
            target_concerns=concerns,
            secondary_concerns=secondary_concerns,
            skin_type='normal',
            budget=budget,
            preferences=['gentle', 'effective', 'evolved']
        )
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """Generate comprehensive evolution report"""
        return {
            'current_generation': self.generation,
            'performance_history': self.performance_history,
            'market_feedback': self.market_feedback,
            'current_formula': {
                'total_cost': self.formula.total_cost,
                'predicted_efficacy': self.formula.predicted_efficacy,
                'safety_score': self.formula.safety_score,
                'synergy_score': self.formula.synergy_score,
                'stability_months': self.formula.stability_months,
                'selected_ingredients': {
                    class_name: {
                        'name': data['ingredient'].name,
                        'percentage': data['percentage'],
                        'reasoning': data['reasoning']
                    }
                    for class_name, data in self.formula.selected_hypergredients.items()
                }
            },
            'evolution_metrics': self._calculate_evolution_metrics()
        }
    
    def _calculate_evolution_metrics(self) -> Dict[str, float]:
        """Calculate evolution performance metrics"""
        if len(self.performance_history) < 2:
            return {'evolution_not_available': True}
        
        first = self.performance_history[0]
        latest = self.performance_history[-1]
        
        return {
            'efficacy_improvement': latest['efficacy'] - first['efficacy'],
            'safety_improvement': latest['safety'] - first['safety'],
            'synergy_improvement': latest['synergy'] - first['synergy'],
            'cost_change': latest['cost'] - first['cost'],
            'generations_evolved': len(self.performance_history)
        }


class HypergredientAI:
    """Machine Learning Integration for Hypergredient Prediction with Persona Support"""
    
    def __init__(self, persona_system: Optional[PersonaTrainingSystem] = None):
        self.model_version = "v1.0"
        self.confidence_threshold = 0.7
        self.feedback_data = []
        self.persona_system = persona_system or PersonaTrainingSystem()
    
    def predict_optimal_combination(self, requirements: FormulationRequest) -> Dict[str, Any]:
        """Predict best hypergredient combinations using simulated ML with persona awareness"""
        
        # Simulate feature extraction
        base_features = self._extract_features(requirements)
        
        # Apply persona adjustments if active persona exists
        features = self.persona_system.get_persona_adjusted_features(requirements, base_features)
        
        # Simulate ML predictions (in real implementation, this would use trained models)
        predictions = self._simulate_ml_predictions(features, requirements)
        
        # Rank by confidence
        ranked_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        result = {
            'model_version': self.model_version,
            'active_persona': self.persona_system.active_persona,
            'predictions': ranked_predictions[:5],  # Top 5 predictions
            'confidence_scores': {pred['ingredient_class']: pred['confidence'] for pred in ranked_predictions[:5]},
            'feature_importance': self._get_feature_importance(features),
            'persona_adjustments': self._get_persona_adjustments() if self.persona_system.active_persona else None
        }
        
        return result
    
    def _extract_features(self, requirements: FormulationRequest) -> Dict[str, float]:
        """Extract features from formulation requirements"""
        
        # Concern encoding
        concern_weights = {
            'wrinkles': 1.0, 'aging': 0.9, 'firmness': 0.8, 'dryness': 0.7,
            'acne': 0.6, 'sensitivity': 0.5, 'hyperpigmentation': 0.8
        }
        
        concern_score = sum(concern_weights.get(concern, 0.3) 
                          for concern in requirements.target_concerns + requirements.secondary_concerns)
        
        # Skin type encoding
        skin_type_scores = {
            'oily': 0.2, 'dry': 0.8, 'sensitive': 0.9, 
            'normal': 0.5, 'combination': 0.6
        }
        
        features = {
            'concern_complexity': concern_score,
            'budget_normalized': min(requirements.budget / 1500.0, 1.0),
            'skin_sensitivity': skin_type_scores.get(requirements.skin_type, 0.5),
            'preference_gentleness': 1.0 if 'gentle' in requirements.preferences else 0.3,
            'preference_effectiveness': 1.0 if 'effective' in requirements.preferences else 0.7
        }
        
        return features
    
    def _simulate_ml_predictions(self, features: Dict[str, float], requirements: FormulationRequest) -> List[Dict[str, Any]]:
        """Simulate ML model predictions with persona awareness"""
        import random
        
        # Simulate predictions for different hypergredient classes
        base_predictions = [
            {'ingredient_class': 'H.CT', 'base_confidence': 0.8},
            {'ingredient_class': 'H.CS', 'base_confidence': 0.9},
            {'ingredient_class': 'H.AO', 'base_confidence': 0.7},
            {'ingredient_class': 'H.ML', 'base_confidence': 0.6},
            {'ingredient_class': 'H.HY', 'base_confidence': 0.85},
            {'ingredient_class': 'H.AI', 'base_confidence': 0.75},
            {'ingredient_class': 'H.BR', 'base_confidence': 0.65}
        ]
        
        predictions = []
        for pred in base_predictions:
            # Adjust confidence based on features
            confidence_adjustment = (
                features['concern_complexity'] * 0.1 +
                features['budget_normalized'] * 0.1 +
                features['skin_sensitivity'] * 0.05 +
                features['preference_gentleness'] * 0.05
            )
            
            # Add persona-specific adjustments
            persona_adjustment = 0.0
            if self.persona_system.active_persona:
                persona = self.persona_system.personas[self.persona_system.active_persona]
                
                # Adjust based on persona safety priority
                if pred['ingredient_class'] in ['H.CT'] and persona.safety_priority > 0.8:
                    persona_adjustment -= 0.2  # Reduce confidence in strong actives for safety-conscious personas
                
                # Adjust based on persona concerns
                concern_boosts = {
                    'H.AI': 0.3 if 'sensitivity' in persona.primary_concerns else 0.0,
                    'H.BR': 0.3 if 'barrier_repair' in persona.primary_concerns else 0.0,
                    'H.CT': 0.3 if 'wrinkles' in persona.primary_concerns else 0.0,
                    'H.HY': 0.3 if 'dryness' in persona.primary_concerns else 0.0
                }
                persona_adjustment += concern_boosts.get(pred['ingredient_class'], 0.0)
            
            adjusted_confidence = min(pred['base_confidence'] + confidence_adjustment + persona_adjustment, 1.0)
            adjusted_confidence = max(adjusted_confidence, 0.1)  # Minimum confidence
            
            predictions.append({
                'ingredient_class': pred['ingredient_class'],
                'confidence': adjusted_confidence,
                'reasoning': self._generate_ml_reasoning(pred['ingredient_class'], features)
            })
        
        return predictions
    
    def _generate_ml_reasoning(self, ingredient_class: str, features: Dict[str, float]) -> str:
        """Generate reasoning for ML predictions"""
        reasons = []
        
        if features['concern_complexity'] > 0.8:
            reasons.append("High concern complexity detected")
        if features['skin_sensitivity'] > 0.7:
            reasons.append("Sensitive skin considerations")
        if features['preference_gentleness'] > 0.8:
            reasons.append("Gentleness prioritized")
        
        class_specific = {
            'H.CT': "Strong anti-aging efficacy predicted",
            'H.CS': "Collagen synthesis highly beneficial",
            'H.AO': "Antioxidant protection recommended",
            'H.ML': "Brightening effects suitable",
            'H.HY': "Hydration enhancement predicted",
            'H.AI': "Anti-inflammatory benefits expected",
            'H.BR': "Barrier repair highly recommended"
        }
        
        base_reason = class_specific.get(ingredient_class, "Standard recommendation")
        if reasons:
            return f"{base_reason}; {'; '.join(reasons)}"
        return base_reason
    
    def _get_feature_importance(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get feature importance scores"""
        # Simulate feature importance (in real ML, this would come from model)
        importance = {
            'concern_complexity': 0.35,
            'skin_sensitivity': 0.25,
            'budget_normalized': 0.20,
            'preference_gentleness': 0.15,
            'preference_effectiveness': 0.05
        }
        return importance
    
    def _get_persona_adjustments(self) -> Dict[str, Any]:
        """Get current persona adjustments information"""
        if not self.persona_system.active_persona:
            return None
        
        persona = self.persona_system.personas[self.persona_system.active_persona]
        return {
            'persona_name': persona.name,
            'persona_id': persona.persona_id,
            'sensitivity_level': persona.sensitivity_level,
            'safety_priority': persona.safety_priority,
            'primary_concerns': persona.primary_concerns,
            'ingredient_preferences': persona.ingredient_preferences,
            'ingredient_aversions': persona.ingredient_aversions
        }
    
    def train_with_persona(self, persona_id: str, training_requests: List[FormulationRequest],
                          training_results: List[FormulationResult], 
                          feedback_scores: List[Dict[str, float]]):
        """Train the model with persona-specific data"""
        # Set the persona for training
        original_persona = self.persona_system.active_persona
        self.persona_system.set_active_persona(persona_id)
        
        try:
            # Train the persona system
            self.persona_system.train_persona(persona_id, training_requests, training_results, feedback_scores)
            
            # Simulate persona-specific model adaptation
            self._simulate_persona_adaptation(persona_id, len(training_requests))
            
        finally:
            # Restore original persona
            if original_persona:
                self.persona_system.set_active_persona(original_persona)
            else:
                self.persona_system.active_persona = None
    
    def _simulate_persona_adaptation(self, persona_id: str, training_samples: int):
        """Simulate persona-specific model adaptation"""
        persona = self.persona_system.personas[persona_id]
        print(f"🎭 Adapted model for persona '{persona.name}' with {training_samples} samples")
        print(f"   Persona characteristics: safety_priority={persona.safety_priority:.2f}, "
              f"sensitivity={persona.sensitivity_level:.2f}")
    
    def update_from_results(self, formulation_id: str, results: Dict[str, Any], persona_id: Optional[str] = None):
        """Update model from real-world results with optional persona context"""
        feedback_entry = {
            'formulation_id': formulation_id,
            'results': results,
            'timestamp': len(self.feedback_data),  # Simplified timestamp
            'model_version': self.model_version,
            'persona_id': persona_id or self.persona_system.active_persona
        }
        
        self.feedback_data.append(feedback_entry)
        
        # Simulate model retraining trigger
        if len(self.feedback_data) >= 100:  # Retrain after 100 data points
            self._simulate_model_retraining()
    
    def _simulate_model_retraining(self):
        """Simulate model retraining process with persona awareness"""
        # Count persona-specific feedback
        persona_feedback = defaultdict(int)
        for entry in self.feedback_data[-100:]:  # Last 100 entries
            if entry.get('persona_id'):
                persona_feedback[entry['persona_id']] += 1
        
        # In real implementation, this would retrain the ML model
        self.model_version = f"v{float(self.model_version[1:]) + 0.1:.1f}"
        print(f"🤖 Model retrained to version {self.model_version}")
        
        if persona_feedback:
            print(f"   Persona-specific feedback incorporated:")
            for persona_id, count in persona_feedback.items():
                persona_name = self.persona_system.personas.get(persona_id, {}).get('name', persona_id)
                print(f"     - {persona_name}: {count} samples")


class MetaOptimizationStrategy:
    """Meta-optimization strategy to generate optimal formulations for every possible condition and treatment"""
    
    def __init__(self, database: HypergredientDatabase):
        self.database = database
        self.optimizer = HypergredientOptimizer(database)
        
        # Comprehensive condition and treatment matrix
        self.skin_conditions = [
            'wrinkles', 'aging', 'acne', 'hyperpigmentation', 'dryness', 
            'sensitivity', 'dullness', 'firmness', 'texture', 'redness',
            'oiliness', 'blackheads', 'enlarged_pores', 'dehydration',
            'loss_of_elasticity', 'sun_damage', 'melasma', 'rosacea',
            'eczema', 'barrier_damage', 'inflammation', 'scarring'
        ]
        
        self.skin_types = ['oily', 'dry', 'sensitive', 'normal', 'combination', 'mature']
        
        self.severity_levels = ['mild', 'moderate', 'severe']
        
        self.treatment_goals = [
            'prevention', 'maintenance', 'treatment', 'intensive_treatment',
            'post_treatment_care', 'long_term_management'
        ]
        
        # Meta-optimization parameters
        self.optimization_cache = {}
        self.performance_matrix = {}
        self.adaptive_weights = {}
        
        # Initialize base objective weights for different scenarios
        self._initialize_adaptive_weights()
        
    def _initialize_adaptive_weights(self):
        """Initialize adaptive objective weights for different scenarios"""
        self.adaptive_weights = {
            # Weights by skin type
            'skin_type': {
                'sensitive': {'efficacy': 0.25, 'safety': 0.40, 'stability': 0.20, 'cost_efficiency': 0.10, 'bioavailability': 0.05},
                'oily': {'efficacy': 0.35, 'safety': 0.20, 'stability': 0.25, 'cost_efficiency': 0.15, 'bioavailability': 0.05},
                'dry': {'efficacy': 0.30, 'safety': 0.25, 'stability': 0.20, 'cost_efficiency': 0.15, 'bioavailability': 0.10},
                'normal': {'efficacy': 0.35, 'safety': 0.25, 'stability': 0.20, 'cost_efficiency': 0.15, 'bioavailability': 0.05},
                'combination': {'efficacy': 0.30, 'safety': 0.25, 'stability': 0.25, 'cost_efficiency': 0.15, 'bioavailability': 0.05},
                'mature': {'efficacy': 0.40, 'safety': 0.30, 'stability': 0.15, 'cost_efficiency': 0.10, 'bioavailability': 0.05}
            },
            # Weights by severity level
            'severity': {
                'mild': {'efficacy': 0.25, 'safety': 0.35, 'stability': 0.20, 'cost_efficiency': 0.15, 'bioavailability': 0.05},
                'moderate': {'efficacy': 0.35, 'safety': 0.25, 'stability': 0.20, 'cost_efficiency': 0.15, 'bioavailability': 0.05},
                'severe': {'efficacy': 0.45, 'safety': 0.20, 'stability': 0.20, 'cost_efficiency': 0.10, 'bioavailability': 0.05}
            },
            # Weights by treatment goal
            'treatment_goal': {
                'prevention': {'efficacy': 0.20, 'safety': 0.40, 'stability': 0.25, 'cost_efficiency': 0.10, 'bioavailability': 0.05},
                'maintenance': {'efficacy': 0.30, 'safety': 0.30, 'stability': 0.25, 'cost_efficiency': 0.10, 'bioavailability': 0.05},
                'treatment': {'efficacy': 0.40, 'safety': 0.25, 'stability': 0.20, 'cost_efficiency': 0.10, 'bioavailability': 0.05},
                'intensive_treatment': {'efficacy': 0.50, 'safety': 0.20, 'stability': 0.15, 'cost_efficiency': 0.10, 'bioavailability': 0.05},
                'post_treatment_care': {'efficacy': 0.15, 'safety': 0.45, 'stability': 0.25, 'cost_efficiency': 0.10, 'bioavailability': 0.05},
                'long_term_management': {'efficacy': 0.25, 'safety': 0.35, 'stability': 0.25, 'cost_efficiency': 0.10, 'bioavailability': 0.05}
            }
        }
    
    def generate_comprehensive_formulation_matrix(self, max_combinations: int = 1000) -> Dict[str, Any]:
        """Generate optimal formulations for all possible condition and treatment combinations"""
        print("🧬 Starting Meta-Optimization Strategy")
        print("=" * 50)
        
        formulation_matrix = {}
        combinations_processed = 0
        total_combinations = min(max_combinations, len(self.skin_conditions) * len(self.skin_types) * len(self.severity_levels))
        
        print(f"Processing up to {total_combinations} condition/treatment combinations...")
        
        for condition in self.skin_conditions:
            if combinations_processed >= max_combinations:
                break
                
            for skin_type in self.skin_types:
                if combinations_processed >= max_combinations:
                    break
                    
                for severity in self.severity_levels:
                    if combinations_processed >= max_combinations:
                        break
                    
                    for treatment_goal in self.treatment_goals:
                        if combinations_processed >= max_combinations:
                            break
                        
                        # Create unique combination key
                        combo_key = f"{condition}_{skin_type}_{severity}_{treatment_goal}"
                        
                        # Skip if already processed
                        if combo_key in self.optimization_cache:
                            continue
                        
                        # Generate optimal formulation for this combination
                        optimal_formulation = self._optimize_for_combination(
                            condition, skin_type, severity, treatment_goal
                        )
                        
                        if optimal_formulation:
                            formulation_matrix[combo_key] = {
                                'condition': condition,
                                'skin_type': skin_type,
                                'severity': severity,
                                'treatment_goal': treatment_goal,
                                'formulation': optimal_formulation,
                                'optimization_score': self._calculate_combination_score(optimal_formulation, condition, skin_type, severity, treatment_goal),
                                'meta_insights': self._generate_meta_insights(optimal_formulation, condition, skin_type, severity, treatment_goal)
                            }
                            
                            # Cache for future use
                            self.optimization_cache[combo_key] = formulation_matrix[combo_key]
                            
                            combinations_processed += 1
                            
                            if combinations_processed % 50 == 0:
                                print(f"  Processed {combinations_processed}/{total_combinations} combinations...")
        
        print(f"✓ Generated {combinations_processed} optimal formulations")
        
        # Generate comprehensive analysis
        analysis = self._analyze_formulation_matrix(formulation_matrix)
        
        return {
            'formulation_matrix': formulation_matrix,
            'meta_analysis': analysis,
            'optimization_statistics': {
                'total_combinations_processed': combinations_processed,
                'cache_size': len(self.optimization_cache),
                'performance_matrix_size': len(self.performance_matrix)
            }
        }
    
    def _optimize_for_combination(self, condition: str, skin_type: str, severity: str, treatment_goal: str) -> Optional[FormulationResult]:
        """Optimize formulation for specific condition/treatment combination"""
        try:
            # Create dynamic formulation request
            request = self._create_dynamic_request(condition, skin_type, severity, treatment_goal)
            
            # Get adaptive weights for this combination
            objective_weights = self._get_adaptive_weights(skin_type, severity, treatment_goal)
            
            # Store original weights and temporarily modify optimizer
            original_weights = None
            if hasattr(self.optimizer, '_get_objective_weights'):
                original_weights = self.optimizer._get_objective_weights()
            
            # Temporarily modify the optimizer to use adaptive weights
            self.optimizer._adaptive_weights = objective_weights
            
            # Generate formulation
            result = self.optimizer.optimize_formulation(request)
            
            # Restore original weights
            if original_weights and hasattr(self.optimizer, '_set_objective_weights'):
                self.optimizer._set_objective_weights(original_weights)
            
            return result
            
        except Exception as e:
            print(f"  Warning: Failed to optimize for {condition}_{skin_type}_{severity}_{treatment_goal}: {e}")
            return None
    
    def _create_dynamic_request(self, condition: str, skin_type: str, severity: str, treatment_goal: str) -> FormulationRequest:
        """Create dynamic formulation request based on combination parameters"""
        # Base concerns
        target_concerns = [condition]
        secondary_concerns = []
        
        # Add related concerns based on condition
        condition_relationships = {
            'wrinkles': ['aging', 'firmness'],
            'aging': ['wrinkles', 'dullness', 'loss_of_elasticity'],
            'acne': ['oiliness', 'inflammation', 'enlarged_pores'],
            'hyperpigmentation': ['dullness', 'sun_damage'],
            'dryness': ['dehydration', 'barrier_damage'],
            'sensitivity': ['redness', 'inflammation', 'barrier_damage'],
            'dullness': ['texture', 'dehydration'],
            'oiliness': ['acne', 'enlarged_pores'],
            'rosacea': ['sensitivity', 'redness', 'inflammation']
        }
        
        if condition in condition_relationships:
            secondary_concerns.extend(condition_relationships[condition])
        
        # Adjust budget based on severity and treatment goal
        base_budget = 1000.0
        severity_multipliers = {'mild': 0.8, 'moderate': 1.0, 'severe': 1.3}
        goal_multipliers = {
            'prevention': 0.7, 'maintenance': 0.8, 'treatment': 1.0,
            'intensive_treatment': 1.5, 'post_treatment_care': 0.9, 'long_term_management': 1.1
        }
        
        budget = base_budget * severity_multipliers.get(severity, 1.0) * goal_multipliers.get(treatment_goal, 1.0)
        
        # Set preferences based on skin type and treatment goal
        preferences = []
        if skin_type == 'sensitive':
            preferences.extend(['gentle', 'hypoallergenic'])
        if treatment_goal in ['prevention', 'maintenance']:
            preferences.append('gentle')
        if treatment_goal in ['treatment', 'intensive_treatment']:
            preferences.append('effective')
        if severity == 'severe':
            preferences.append('potent')
        
        # Create constraints
        constraints = FormulationConstraints(
            max_budget=budget,
            max_irritation_score=2.0 if skin_type == 'sensitive' else 5.0,
            required_stability_months=12 if treatment_goal == 'intensive_treatment' else 24
        )
        
        return FormulationRequest(
            target_concerns=target_concerns,
            secondary_concerns=secondary_concerns[:3],  # Limit to avoid over-complexity
            skin_type=skin_type,
            budget=budget,
            preferences=preferences,
            constraints=constraints
        )
    
    def _get_adaptive_weights(self, skin_type: str, severity: str, treatment_goal: str) -> Dict[str, float]:
        """Get adaptive objective weights based on combination parameters"""
        # Start with base weights
        weights = {'efficacy': 0.35, 'safety': 0.25, 'stability': 0.20, 'cost_efficiency': 0.15, 'bioavailability': 0.05}
        
        # Apply skin type adjustments
        if skin_type in self.adaptive_weights['skin_type']:
            skin_weights = self.adaptive_weights['skin_type'][skin_type]
            for key in weights.keys():
                weights[key] = (weights[key] + skin_weights.get(key, weights[key])) / 2
        
        # Apply severity adjustments
        if severity in self.adaptive_weights['severity']:
            severity_weights = self.adaptive_weights['severity'][severity]
            for key in weights.keys():
                weights[key] = (weights[key] + severity_weights.get(key, weights[key])) / 2
        
        # Apply treatment goal adjustments
        if treatment_goal in self.adaptive_weights['treatment_goal']:
            goal_weights = self.adaptive_weights['treatment_goal'][treatment_goal]
            for key in weights.keys():
                weights[key] = (weights[key] + goal_weights.get(key, weights[key])) / 2
        
        # Normalize to ensure sum = 1.0
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def _calculate_combination_score(self, formulation: FormulationResult, condition: str, skin_type: str, severity: str, treatment_goal: str) -> float:
        """Calculate optimization score for this specific combination"""
        base_score = (formulation.predicted_efficacy * 0.4 + 
                     (formulation.safety_score / 10.0) * 0.3 + 
                     formulation.synergy_score * 0.2 + 
                     (1.0 / (formulation.total_cost / 1000.0)) * 0.1)
        
        # Apply combination-specific bonuses
        bonus = 0.0
        
        # Severity bonus for appropriate efficacy
        if severity == 'severe' and formulation.predicted_efficacy > 0.7:
            bonus += 0.1
        elif severity == 'mild' and formulation.safety_score > 9.0:
            bonus += 0.1
        
        # Skin type bonus
        if skin_type == 'sensitive' and formulation.safety_score > 8.5:
            bonus += 0.1
        elif skin_type == 'oily' and formulation.predicted_efficacy > 0.6:
            bonus += 0.05
        
        return min(base_score + bonus, 1.0)
    
    def _generate_meta_insights(self, formulation: FormulationResult, condition: str, skin_type: str, severity: str, treatment_goal: str) -> Dict[str, Any]:
        """Generate meta-optimization insights for this combination"""
        insights = {
            'optimization_rationale': f"Optimized for {condition} in {skin_type} skin with {severity} severity for {treatment_goal}",
            'key_trade_offs': [],
            'alternative_approaches': [],
            'contraindications': [],
            'synergy_highlights': []
        }
        
        # Analyze trade-offs
        if formulation.predicted_efficacy > 0.8 and formulation.safety_score < 8.0:
            insights['key_trade_offs'].append("High efficacy prioritized over maximum safety")
        elif formulation.safety_score > 9.0 and formulation.predicted_efficacy < 0.5:
            insights['key_trade_offs'].append("Maximum safety prioritized over peak efficacy")
        
        # Alternative approaches
        if severity == 'severe' and formulation.total_cost < 800:
            insights['alternative_approaches'].append("Consider higher-potency actives for severe cases")
        elif skin_type == 'sensitive' and len(formulation.selected_hypergredients) > 4:
            insights['alternative_approaches'].append("Simplified formula may be better for sensitive skin")
        
        # Contraindications based on combination
        if condition == 'acne' and 'H.HY' in formulation.selected_hypergredients:
            insights['contraindications'].append("Monitor for potential pore-clogging with heavy hydrators")
        elif skin_type == 'sensitive' and formulation.predicted_efficacy > 0.7:
            insights['contraindications'].append("High efficacy may cause irritation in sensitive skin")
        
        # Synergy highlights
        if formulation.synergy_score > 0.6:
            insights['synergy_highlights'].append(f"Excellent ingredient synergy (score: {formulation.synergy_score:.2f})")
        
        return insights
    
    def _analyze_formulation_matrix(self, matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the complete formulation matrix for patterns and insights"""
        analysis = {
            'pattern_analysis': {},
            'ingredient_usage_patterns': {},
            'efficacy_patterns': {},
            'cost_patterns': {},
            'optimization_recommendations': []
        }
        
        if not matrix:
            return analysis
        
        # Analyze patterns by condition
        condition_stats = defaultdict(list)
        skin_type_stats = defaultdict(list)
        severity_stats = defaultdict(list)
        
        for combo_key, data in matrix.items():
            condition = data['condition']
            skin_type = data['skin_type']
            severity = data['severity']
            formulation = data['formulation']
            
            condition_stats[condition].append(formulation.predicted_efficacy)
            skin_type_stats[skin_type].append(formulation.safety_score)
            severity_stats[severity].append(formulation.total_cost)
        
        # Calculate average efficacy by condition
        analysis['efficacy_patterns'] = {
            condition: {
                'avg_efficacy': sum(efficacies) / len(efficacies),
                'count': len(efficacies)
            }
            for condition, efficacies in condition_stats.items()
        }
        
        # Calculate average safety by skin type
        analysis['pattern_analysis']['safety_by_skin_type'] = {
            skin_type: sum(scores) / len(scores)
            for skin_type, scores in skin_type_stats.items()
        }
        
        # Calculate average cost by severity
        analysis['cost_patterns'] = {
            severity: sum(costs) / len(costs)
            for severity, costs in severity_stats.items()
        }
        
        # Generate optimization recommendations
        if analysis['efficacy_patterns']:
            best_condition = max(analysis['efficacy_patterns'].items(), key=lambda x: x[1]['avg_efficacy'])
            analysis['optimization_recommendations'].append(
                f"Best efficacy achieved for {best_condition[0]} (avg: {best_condition[1]['avg_efficacy']:.2%})"
            )
        
        return analysis
    
    def get_optimal_formulation_for_profile(self, condition: str, skin_type: str, severity: str = 'moderate', treatment_goal: str = 'treatment') -> Optional[Dict[str, Any]]:
        """Get optimal formulation for specific user profile"""
        combo_key = f"{condition}_{skin_type}_{severity}_{treatment_goal}"
        
        if combo_key in self.optimization_cache:
            return self.optimization_cache[combo_key]
        
        # Generate if not in cache
        formulation = self._optimize_for_combination(condition, skin_type, severity, treatment_goal)
        if formulation:
            result = {
                'condition': condition,
                'skin_type': skin_type,
                'severity': severity,
                'treatment_goal': treatment_goal,
                'formulation': formulation,
                'optimization_score': self._calculate_combination_score(formulation, condition, skin_type, severity, treatment_goal),
                'meta_insights': self._generate_meta_insights(formulation, condition, skin_type, severity, treatment_goal)
            }
            self.optimization_cache[combo_key] = result
            return result
        
        return None
    
    def generate_meta_optimization_report(self) -> str:
        """Generate comprehensive meta-optimization report"""
        report = []
        report.append("🧬 Meta-Optimization Strategy Report")
        report.append("=" * 50)
        report.append("")
        
        report.append(f"Total Cached Formulations: {len(self.optimization_cache)}")
        report.append(f"Conditions Covered: {len(self.skin_conditions)}")
        report.append(f"Skin Types Covered: {len(self.skin_types)}")
        report.append(f"Severity Levels: {len(self.severity_levels)}")
        report.append(f"Treatment Goals: {len(self.treatment_goals)}")
        report.append("")
        
        if self.optimization_cache:
            # Analyze cached formulations
            total_formulations = len(self.optimization_cache)
            avg_efficacy = sum(data['formulation'].predicted_efficacy for data in self.optimization_cache.values()) / total_formulations
            avg_safety = sum(data['formulation'].safety_score for data in self.optimization_cache.values()) / total_formulations
            avg_cost = sum(data['formulation'].total_cost for data in self.optimization_cache.values()) / total_formulations
            
            report.append("Performance Averages:")
            report.append(f"  Average Efficacy: {avg_efficacy:.2%}")
            report.append(f"  Average Safety: {avg_safety:.1f}/10")
            report.append(f"  Average Cost: R{avg_cost:.2f}")
            report.append("")
            
            # Find best formulations
            best_efficacy = max(self.optimization_cache.values(), key=lambda x: x['formulation'].predicted_efficacy)
            best_safety = max(self.optimization_cache.values(), key=lambda x: x['formulation'].safety_score)
            
            report.append("Top Performers:")
            report.append(f"  Best Efficacy: {best_efficacy['condition']} for {best_efficacy['skin_type']} skin ({best_efficacy['formulation'].predicted_efficacy:.2%})")
            report.append(f"  Best Safety: {best_safety['condition']} for {best_safety['skin_type']} skin ({best_safety['formulation'].safety_score:.1f}/10)")
        
        report.append("")
        report.append("Meta-optimization capabilities:")
        report.append("✓ Systematic exploration of all condition/treatment combinations")
        report.append("✓ Adaptive objective weights based on user profile")
        report.append("✓ Dynamic formulation request generation")
        report.append("✓ Comprehensive caching and performance tracking")
        report.append("✓ Pattern analysis and optimization insights")
        
        return "\n".join(report)


class HypergredientVisualizer:
    """Visualization Dashboard for Hypergredient Framework"""
    
    def __init__(self, database: HypergredientDatabase):
        self.database = database
    
    def generate_formulation_report(self, formulation: FormulationResult, 
                                  request: FormulationRequest) -> Dict[str, Any]:
        """Create comprehensive visual report for formulation"""
        
        report = {
            "title": "🧬 Hypergredient Formulation Analysis Report",
            "timestamp": "Generated with Hypergredient Framework v1.0",
            "formulation_overview": self._create_formulation_overview(formulation, request),
            "performance_radar": self._create_performance_radar(formulation),
            "ingredient_breakdown": self._create_ingredient_breakdown(formulation),
            "cost_analysis": self._create_cost_analysis(formulation),
            "synergy_network": self._create_synergy_network(formulation),
            "risk_assessment": self._create_risk_assessment(formulation),
            "recommendations": self._generate_recommendations(formulation)
        }
        
        return report
    
    def _create_formulation_overview(self, formulation: FormulationResult, 
                                   request: FormulationRequest) -> Dict[str, Any]:
        """Create formulation overview section"""
        return {
            "formulation_id": f"HF-{hash(str(formulation.selected_hypergredients)) % 10000:04d}",
            "target_concerns": request.target_concerns,
            "skin_type": request.skin_type,
            "budget_allocated": f"R{formulation.total_cost:.2f} / R{request.budget:.2f}",
            "budget_utilization": f"{(formulation.total_cost / request.budget) * 100:.1f}%",
            "total_ingredients": len(formulation.selected_hypergredients),
            "predicted_outcomes": {
                "efficacy": f"{formulation.predicted_efficacy:.1%}",
                "safety_score": f"{formulation.safety_score:.1f}/10",
                "synergy_bonus": f"{formulation.synergy_score:.2f}",
                "stability": f"{formulation.stability_months} months"
            }
        }
    
    def _create_performance_radar(self, formulation: FormulationResult) -> Dict[str, Any]:
        """Create performance radar chart data"""
        metrics = {
            "Efficacy": formulation.predicted_efficacy * 100,  # Convert to percentage
            "Safety": (formulation.safety_score / 10) * 100,
            "Synergy": formulation.synergy_score * 100,
            "Stability": min((formulation.stability_months / 24) * 100, 100),  # 24 months = 100%
            "Cost Efficiency": max(100 - (formulation.total_cost / 1000 * 100), 0)  # Inverse cost
        }
        
        return {
            "chart_type": "radar",
            "data": metrics,
            "description": "Multi-dimensional performance analysis",
            "interpretation": {
                "strengths": [k for k, v in metrics.items() if v > 70],
                "areas_for_improvement": [k for k, v in metrics.items() if v < 50]
            }
        }
    
    def _create_ingredient_breakdown(self, formulation: FormulationResult) -> Dict[str, Any]:
        """Create detailed ingredient breakdown"""
        ingredients = []
        
        for class_name, data in formulation.selected_hypergredients.items():
            ingredient = data['ingredient']
            ingredients.append({
                "class": class_name,
                "name": ingredient.name,
                "inci_name": ingredient.inci_name,
                "percentage": data['percentage'],
                "cost": data['cost'],
                "cost_per_percent": data['cost'] / data['percentage'] if data['percentage'] > 0 else 0,
                "efficacy_score": ingredient.efficacy_score,
                "safety_score": ingredient.safety_score,
                "primary_function": ingredient.primary_function,
                "secondary_functions": ingredient.secondary_functions,
                "reasoning": data['reasoning']
            })
        
        # Sort by cost contribution
        ingredients.sort(key=lambda x: x['cost'], reverse=True)
        
        return {
            "ingredients": ingredients,
            "summary": {
                "total_actives_percentage": sum(ing['percentage'] for ing in ingredients),
                "most_expensive": max(ingredients, key=lambda x: x['cost'])['name'],
                "highest_efficacy": max(ingredients, key=lambda x: x['efficacy_score'])['name'],
                "safest_ingredient": max(ingredients, key=lambda x: x['safety_score'])['name']
            }
        }
    
    def _create_cost_analysis(self, formulation: FormulationResult) -> Dict[str, Any]:
        """Create cost breakdown analysis"""
        cost_breakdown = []
        total_cost = formulation.total_cost
        
        for class_name, data in formulation.selected_hypergredients.items():
            cost_breakdown.append({
                "category": class_name,
                "ingredient": data['ingredient'].name,
                "cost": data['cost'],
                "percentage_of_budget": (data['cost'] / total_cost) * 100,
                "cost_per_gram": data['ingredient'].cost_per_gram,
                "usage_amount": data['percentage']
            })
        
        cost_breakdown.sort(key=lambda x: x['cost'], reverse=True)
        
        return {
            "breakdown": cost_breakdown,
            "cost_efficiency_metrics": {
                "cost_per_efficacy_point": total_cost / (formulation.predicted_efficacy * 100) if formulation.predicted_efficacy > 0 else float('inf'),
                "cost_per_safety_point": total_cost / formulation.safety_score,
                "premium_ingredients": [item for item in cost_breakdown if item['cost_per_gram'] > 200],
                "budget_friendly": [item for item in cost_breakdown if item['cost_per_gram'] < 100]
            }
        }
    
    def _create_synergy_network(self, formulation: FormulationResult) -> Dict[str, Any]:
        """Create synergy network analysis"""
        interactions = []
        classes = list(formulation.selected_hypergredients.keys())
        
        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                interaction_score = self.database.interaction_matrix.get((class1, class2), 1.0)
                
                if interaction_score != 1.0:  # Only include non-neutral interactions
                    interactions.append({
                        "source": class1,
                        "target": class2,
                        "strength": interaction_score,
                        "type": "synergy" if interaction_score > 1.0 else "antagonism",
                        "description": self._describe_interaction(class1, class2, interaction_score)
                    })
        
        return {
            "interactions": interactions,
            "network_strength": formulation.synergy_score,
            "positive_interactions": len([i for i in interactions if i['strength'] > 1.0]),
            "negative_interactions": len([i for i in interactions if i['strength'] < 1.0]),
            "network_description": self._describe_network_quality(formulation.synergy_score)
        }
    
    def _create_risk_assessment(self, formulation: FormulationResult) -> Dict[str, Any]:
        """Create comprehensive risk assessment"""
        risks = []
        warnings = []
        
        # Analyze individual ingredient risks
        for class_name, data in formulation.selected_hypergredients.items():
            ingredient = data['ingredient']
            
            # Safety score analysis
            if ingredient.safety_score < 7.0:
                risks.append({
                    "level": "moderate",
                    "ingredient": ingredient.name,
                    "concern": "Lower safety score",
                    "recommendation": "Consider patch testing and gradual introduction"
                })
            
            # Stability analysis
            if ingredient.stability_index < 0.5:
                warnings.append({
                    "ingredient": ingredient.name,
                    "issue": "Stability concerns",
                    "recommendation": "Store in cool, dark conditions. Use within stability period."
                })
            
            # pH compatibility
            ph_range_size = ingredient.ph_max - ingredient.ph_min
            if ph_range_size < 2.0:
                warnings.append({
                    "ingredient": ingredient.name,
                    "issue": f"Narrow pH range ({ingredient.ph_min}-{ingredient.ph_max})",
                    "recommendation": "Careful pH balancing required in formulation"
                })
        
        # Overall formulation risks
        overall_risk_level = "low"
        if formulation.safety_score < 7.0:
            overall_risk_level = "high" 
        elif formulation.safety_score < 8.5:
            overall_risk_level = "moderate"
        
        return {
            "overall_risk_level": overall_risk_level,
            "safety_score": formulation.safety_score,
            "individual_risks": risks,
            "formulation_warnings": warnings,
            "recommendations": self._generate_safety_recommendations(formulation, risks, warnings)
        }
    
    def _generate_recommendations(self, formulation: FormulationResult) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if formulation.predicted_efficacy < 0.15:
            recommendations.append("⚡ Consider adding more potent actives to improve efficacy")
        
        if formulation.synergy_score < 0.3:
            recommendations.append("🔄 Review ingredient combinations to enhance synergistic effects")
        
        if formulation.stability_months < 18:
            recommendations.append("🛡️ Add stabilizing ingredients or improve packaging to extend shelf life")
        
        if formulation.total_cost > 800:
            recommendations.append("💰 Consider cost-effective alternatives to reduce formulation cost")
        
        if formulation.safety_score < 8.5:
            recommendations.append("⚠️ Conduct additional safety testing before market release")
        
        if len(formulation.selected_hypergredients) < 4:
            recommendations.append("🌟 Consider additional complementary ingredients for comprehensive benefits")
        
        return recommendations
    
    def _describe_interaction(self, class1: str, class2: str, score: float) -> str:
        """Describe interaction between two ingredient classes"""
        if score > 1.5:
            return f"{class1} and {class2} work synergistically to enhance overall performance"
        elif score > 1.0:
            return f"{class1} and {class2} have complementary benefits"
        elif score < 0.8:
            return f"{class1} and {class2} may interfere with each other's effectiveness"
        else:
            return f"{class1} and {class2} have neutral interaction"
    
    def _describe_network_quality(self, synergy_score: float) -> str:
        """Describe overall network quality"""
        if synergy_score > 0.6:
            return "Excellent synergistic network with strong ingredient interactions"
        elif synergy_score > 0.4:
            return "Good ingredient network with moderate synergistic effects"
        elif synergy_score > 0.2:
            return "Fair ingredient network with limited synergistic benefits"
        else:
            return "Weak ingredient network requiring optimization for better synergy"
    
    def _generate_safety_recommendations(self, formulation: FormulationResult, 
                                       risks: List[Dict], warnings: List[Dict]) -> List[str]:
        """Generate safety-specific recommendations"""
        recommendations = []
        
        if risks:
            recommendations.append("🧪 Conduct patch testing before full application")
            recommendations.append("📋 Provide clear usage instructions and warnings")
        
        if warnings:
            recommendations.append("📦 Implement proper storage and packaging requirements")
            recommendations.append("⏰ Establish clear expiration and stability guidelines")
        
        if formulation.safety_score < 8.0:
            recommendations.append("🔬 Consider reformulation with safer alternatives")
            recommendations.append("👥 Consult with dermatological experts for validation")
        
        return recommendations


def main():
    """Demo of hypergredient framework capabilities"""
    print("🧬 Hypergredient Framework Architecture Demo")
    print("=" * 50)
    
    # Initialize system with persona support
    persona_system = PersonaTrainingSystem()
    database = HypergredientDatabase()
    optimizer = HypergredientOptimizer(database)
    analyzer = HypergredientAnalyzer(database)
    visualizer = HypergredientVisualizer(database)
    ai_system = HypergredientAI(persona_system)
    
    print(f"\nInitialized database with {len(database.hypergredients)} hypergredients")
    print(f"Hypergredient classes: {[cls.value for cls in HypergredientClass]}")
    
    # Demo 0: Persona Training System
    print("\n🎭 Persona-Based Training System Demo")
    print("-" * 40)
    
    # Show available personas
    persona_summary = persona_system.get_training_summary()
    print(f"✓ Initialized {persona_summary['total_personas']} persona profiles:")
    for persona_id, info in persona_summary['personas'].items():
        print(f"  • {info['name']}: {info['description']}")
        print(f"    Skin type: {info['skin_type']}, Concerns: {info['primary_concerns']}")
    
    # Demonstrate persona-aware predictions
    print(f"\n📊 Persona-Aware Predictions Demo:")
    test_request = FormulationRequest(
        target_concerns=['wrinkles', 'sensitivity'],
        skin_type='sensitive',
        budget=600.0,
        preferences=['gentle']
    )
    
    # Test different personas
    personas_to_test = ['sensitive_skin', 'anti_aging']
    results_by_persona = {}
    
    for persona_id in personas_to_test:
        ai_system.persona_system.set_active_persona(persona_id)
        prediction = ai_system.predict_optimal_combination(test_request)
        results_by_persona[persona_id] = prediction
        
        persona_name = persona_system.personas[persona_id].name
        print(f"\n  {persona_name} persona recommendations:")
        top_3 = prediction['predictions'][:3]
        for i, pred in enumerate(top_3, 1):
            print(f"    {i}. {pred['ingredient_class']}: {pred['confidence']:.3f} confidence")
            print(f"       Reasoning: {pred['reasoning']}")
    
    # Show persona training simulation
    print(f"\n🎓 Persona Training Simulation:")
    training_requests = [
        FormulationRequest(['sensitivity', 'redness'], skin_type='sensitive', budget=400),
        FormulationRequest(['barrier_repair'], skin_type='sensitive', budget=500),
        FormulationRequest(['dryness', 'sensitivity'], skin_type='sensitive', budget=600)
    ]
    
    # Simulate training results
    training_results = []
    training_feedback = []
    for req in training_requests:
        # Generate a mock result
        mock_result = FormulationResult(
            selected_hypergredients={'H.AI': {'ingredient': database.hypergredients['niacinamide'], 'percentage': 5.0, 'cost': 25.0, 'reasoning': 'Anti-inflammatory for sensitive skin'}},
            total_cost=300.0,
            predicted_efficacy=0.7,
            safety_score=9.5,
            stability_months=24,
            synergy_score=0.8,
            reasoning={'H.AI': 'Excellent for sensitive skin'}
        )
        training_results.append(mock_result)
        training_feedback.append({'efficacy': 8.5, 'safety': 9.8, 'user_satisfaction': 9.0})
    
    # Train the sensitive skin persona
    ai_system.train_with_persona('sensitive_skin', training_requests, training_results, training_feedback)
    
    # Show updated training summary
    updated_summary = persona_system.get_training_summary()
    sensitive_info = updated_summary['personas']['sensitive_skin']
    print(f"  Sensitive Skin persona now has {sensitive_info['training_samples']} training samples")
    
    # Demo 1: Generate anti-aging formulation (with persona awareness)
    print("\n1. Anti-Aging Formulation Optimization")
    print("-" * 40)
    
    anti_aging_request = FormulationRequest(
        target_concerns=['wrinkles', 'firmness'],
        secondary_concerns=['dryness', 'dullness'],
        skin_type='normal_to_dry',
        budget=800.0,
        preferences=['gentle', 'stable']
    )
    
    result = optimizer.optimize_formulation(anti_aging_request)
    
    print(f"✓ Generated formulation with {len(result.selected_hypergredients)} hypergredients")
    print(f"  Total cost: R{result.total_cost:.2f}")
    print(f"  Predicted efficacy: {result.predicted_efficacy:.2%}")
    print(f"  Safety score: {result.safety_score:.1f}/10")
    print(f"  Synergy score: {result.synergy_score:.2f}")
    print(f"  Stability: {result.stability_months} months")
    
    print("\nSelected Hypergredients:")
    for class_name, data in result.selected_hypergredients.items():
        ingredient = data['ingredient']
        print(f"  • {class_name}: {ingredient.name} ({data['percentage']:.1f}%)")
        print(f"    Reasoning: {data['reasoning']}")
    
    # Demo 2: Compatibility analysis
    print("\n2. Compatibility Analysis")
    print("-" * 40)
    
    test_ingredients = ['retinol', 'vitamin_c_laa', 'niacinamide']
    compatibility_report = analyzer.generate_compatibility_report(test_ingredients)
    
    print("Compatibility Matrix:")
    for pair, status in compatibility_report['compatibility_matrix'].items():
        print(f"  {pair}: {status}")
    
    if compatibility_report['warnings']:
        print("\nWarnings:")
        for warning in compatibility_report['warnings']:
            print(f"  {warning}")
    
    if compatibility_report['recommendations']:
        print("\nRecommendations:")
        for rec in compatibility_report['recommendations']:
            print(f"  {rec}")
    
    # Demo 3: Ingredient profile
    print("\n3. Ingredient Profile Analysis")
    print("-" * 40)
    
    profile = analyzer.generate_ingredient_profile('bakuchiol')
    print(f"Ingredient: {profile['basic_info']['name']}")
    print(f"Class: {profile['basic_info']['class']}")
    print(f"Function: {profile['basic_info']['primary_function']}")
    print(f"Efficacy: {profile['performance_metrics']['efficacy_score']}/10")
    print(f"Safety: {profile['performance_metrics']['safety_score']}/10")
    print(f"Cost efficiency: {profile['derived_metrics']['cost_efficiency']}")
    
    # Demo 4: AI-driven predictions
    print("\n4. AI-Driven Ingredient Predictions")
    print("-" * 40)
    
    ai_system = HypergredientAI()
    ai_predictions = ai_system.predict_optimal_combination(anti_aging_request)
    
    print(f"Model version: {ai_predictions['model_version']}")
    print("Top AI Predictions:")
    for pred in ai_predictions['predictions'][:3]:
        print(f"  • {pred['ingredient_class']}: {pred['confidence']:.1%} confidence")
        print(f"    Reasoning: {pred['reasoning']}")
    
    # Demo 5: Evolutionary formulation improvement
    print("\n5. Evolutionary Formulation Improvement")
    print("-" * 40)
    
    evolution_system = FormulationEvolution(result)
    
    # Simulate market feedback
    evolution_system.add_market_feedback({
        'efficacy_rating': 7.5,
        'safety_rating': 9.0,
        'user_satisfaction': 8.2,
        'improvement_requests': ['more moisturizing', 'faster results']
    })
    
    # Evolve the formulation
    target_improvements = {
        'efficacy': 0.25,  # Target 25% efficacy
        'safety': 0.95     # Target 95% safety
    }
    
    evolved_formula = evolution_system.evolve(database, target_improvements)
    
    print(f"✓ Evolution complete - Generation {evolution_system.generation}")
    print(f"  Evolved efficacy: {evolved_formula.predicted_efficacy:.2%}")
    print(f"  Evolved safety: {evolved_formula.safety_score:.1f}/10")
    print(f"  Evolved cost: R{evolved_formula.total_cost:.2f}")
    
    evolution_report = evolution_system.get_evolution_report()
    
    # Demo 6: Visualization dashboard
    print("\n6. Visualization Dashboard Report")
    print("-" * 40)
    
    visual_report = visualizer.generate_formulation_report(result, anti_aging_request)
    
    print(f"✓ Generated comprehensive visualization report")
    print(f"  Formulation ID: {visual_report['formulation_overview']['formulation_id']}")
    print(f"  Performance strengths: {', '.join(visual_report['performance_radar']['interpretation']['strengths'])}")
    print(f"  Most expensive ingredient: {visual_report['ingredient_breakdown']['summary']['most_expensive']}")
    print(f"  Network quality: {visual_report['synergy_network']['network_description']}")
    print(f"  Risk level: {visual_report['risk_assessment']['overall_risk_level']}")
    print(f"  Recommendations: {len(visual_report['recommendations'])} actionable items")
    
    # Demo 7: Meta-Optimization Strategy
    print("\n7. Meta-Optimization Strategy for All Conditions/Treatments")
    print("-" * 60)
    
    # Initialize meta-optimization system
    meta_optimizer = MetaOptimizationStrategy(database)
    
    # Generate comprehensive formulation matrix (limited for demo)
    print("Generating optimal formulations for condition/treatment combinations...")
    matrix_result = meta_optimizer.generate_comprehensive_formulation_matrix(max_combinations=50)
    
    print(f"✓ Generated {len(matrix_result['formulation_matrix'])} optimized formulations")
    print(f"  Cache size: {matrix_result['optimization_statistics']['cache_size']}")
    
    # Show some examples
    print("\nSample Optimized Formulations:")
    sample_count = 0
    for combo_key, data in matrix_result['formulation_matrix'].items():
        if sample_count >= 3:
            break
        print(f"  • {data['condition']} for {data['skin_type']} skin ({data['severity']} severity)")
        print(f"    Goal: {data['treatment_goal']}")
        print(f"    Efficacy: {data['formulation'].predicted_efficacy:.2%}, Safety: {data['formulation'].safety_score:.1f}/10")
        print(f"    Cost: R{data['formulation'].total_cost:.2f}")
        print(f"    Optimization Score: {data['optimization_score']:.3f}")
        if data['meta_insights']['key_trade_offs']:
            print(f"    Trade-offs: {', '.join(data['meta_insights']['key_trade_offs'])}")
        print()
        sample_count += 1
    
    # Test specific profile optimization
    print("Testing specific profile optimization:")
    specific_profile = meta_optimizer.get_optimal_formulation_for_profile(
        condition='acne', 
        skin_type='oily', 
        severity='moderate', 
        treatment_goal='treatment'
    )
    
    if specific_profile:
        print(f"  Profile: Moderate acne in oily skin for treatment")
        print(f"  Efficacy: {specific_profile['formulation'].predicted_efficacy:.2%}")
        print(f"  Safety: {specific_profile['formulation'].safety_score:.1f}/10")
        print(f"  Cost: R{specific_profile['formulation'].total_cost:.2f}")
    
    # Generate and display meta-optimization report
    meta_report = meta_optimizer.generate_meta_optimization_report()
    print(f"\n{meta_report}")
    
    # Analyze patterns if we have enough data
    if matrix_result['meta_analysis']['efficacy_patterns']:
        print("\nPattern Analysis:")
        best_conditions = sorted(
            matrix_result['meta_analysis']['efficacy_patterns'].items(), 
            key=lambda x: x[1]['avg_efficacy'], 
            reverse=True
        )[:3]
        
        print("  Top performing conditions:")
        for condition, stats in best_conditions:
            print(f"    • {condition}: {stats['avg_efficacy']:.2%} avg efficacy ({stats['count']} formulations)")
    
    # Save demo results
    demo_results = {
        "formulation_result": {
            "request": {
                "target_concerns": anti_aging_request.target_concerns,
                "budget": anti_aging_request.budget,
                "skin_type": anti_aging_request.skin_type
            },
            "result": {
                "total_cost": result.total_cost,
                "predicted_efficacy": result.predicted_efficacy,
                "safety_score": result.safety_score,
                "synergy_score": result.synergy_score,
                "stability_months": result.stability_months,
                "selected_ingredients": {
                    class_name: {
                        "name": data['ingredient'].name,
                        "percentage": data['percentage'],
                        "cost": data['cost'],
                        "reasoning": data['reasoning']
                    }
                    for class_name, data in result.selected_hypergredients.items()
                }
            }
        },
        "compatibility_analysis": compatibility_report,
        "ingredient_profile": profile,
        "ai_predictions": ai_predictions,
        "evolution_report": evolution_report,
        "visualization_report": visual_report,
        "meta_optimization": {
            "total_combinations": len(matrix_result['formulation_matrix']),
            "sample_formulations": {
                k: {
                    "condition": v['condition'],
                    "skin_type": v['skin_type'],
                    "severity": v['severity'],
                    "treatment_goal": v['treatment_goal'],
                    "efficacy": v['formulation'].predicted_efficacy,
                    "safety": v['formulation'].safety_score,
                    "cost": v['formulation'].total_cost,
                    "optimization_score": v['optimization_score']
                }
                for k, v in list(matrix_result['formulation_matrix'].items())[:5]
            },
            "meta_analysis": matrix_result['meta_analysis'],
            "optimization_statistics": matrix_result['optimization_statistics']
        }
    }
    
    with open("hypergredient_demo_results.json", "w") as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n✓ Demo results saved to hypergredient_demo_results.json")
    print("\nHypergredient Framework successfully demonstrates:")
    print("• Multi-objective formulation optimization")
    print("• Real-time compatibility analysis") 
    print("• Ingredient profiling and scoring")
    print("• Synergy calculation and recommendations")
    print("• AI-driven ingredient predictions")
    print("• Evolutionary formulation improvement")
    print("• Comprehensive visualization dashboard")
    print("• Meta-optimization strategy for all condition/treatment combinations")


if __name__ == "__main__":
    main()
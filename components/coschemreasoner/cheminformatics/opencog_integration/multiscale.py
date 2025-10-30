"""
Multiscale Constraint Optimization Framework for Cosmeceutical Formulation

This module implements the multiscale skin model integration and constraint
optimization framework, handling simultaneous local and global optimization
across molecular, cellular, tissue, and organ scales.

Features:
- Multiscale skin model representation
- Cross-scale constraint propagation
- Hierarchical optimization strategies
- Delivery mechanism optimization
- Clinical effectiveness modeling
"""

from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import math
from collections import defaultdict
from abc import ABC, abstractmethod

from .atomspace import CosmeceuticalAtomSpace, Atom, AtomType, MultiscaleLevel
from .reasoning import IngredientReasoningEngine, TruthValue
from .optimization import MultiscaleOptimizer, FormulationGenome, OptimizationObjective


class SkinLayer(Enum):
    """Skin layers for multiscale modeling"""
    STRATUM_CORNEUM = "stratum_corneum"      # 10-20 μm
    VIABLE_EPIDERMIS = "viable_epidermis"    # 50-100 μm
    DERMIS = "dermis"                        # 1000-4000 μm
    HYPODERMIS = "hypodermis"                # Variable thickness


class DeliveryMechanism(Enum):
    """Delivery mechanisms for ingredient transport"""
    PASSIVE_DIFFUSION = "passive_diffusion"
    FACILITATED_TRANSPORT = "facilitated_transport"
    LIPOSOMAL_DELIVERY = "liposomal_delivery"
    NANOPARTICLE_DELIVERY = "nanoparticle_delivery"
    IONTOPHORESIS = "iontophoresis"
    MICRONEEDLE = "microneedle"


@dataclass
class SkinBarrierProperties:
    """Properties of skin barriers at different scales"""
    layer: SkinLayer
    thickness: float  # micrometers
    permeability_coefficient: float  # cm/s
    partition_coefficient: float
    diffusion_coefficient: float  # cm²/s
    lipid_content: float  # percentage
    water_content: float  # percentage
    ph_value: float
    temperature: float = 32.0  # skin temperature in Celsius


@dataclass
class IngredientPenetrationProfile:
    """Penetration profile of an ingredient through skin layers"""
    ingredient_name: str
    molecular_weight: float
    log_p: float  # lipophilicity
    concentration_profile: Dict[SkinLayer, float] = field(default_factory=dict)
    flux_profile: Dict[SkinLayer, float] = field(default_factory=dict)
    penetration_depth: float = 0.0  # maximum penetration depth in μm
    delivery_mechanism: DeliveryMechanism = DeliveryMechanism.PASSIVE_DIFFUSION
    bioavailability: float = 0.0


@dataclass
class TherapeuticVector:
    """Represents a therapeutic vector (clinical target)"""
    vector_name: str
    target_layer: SkinLayer
    required_concentration: float  # minimum effective concentration
    optimal_concentration: float  # optimal concentration for maximum effect
    safety_limit: float  # maximum safe concentration
    mechanism_of_action: str
    clinical_endpoint: str


class SkinModelIntegrator:
    """
    Multiscale skin model integration system for cosmeceutical optimization.
    
    This class manages the multiscale representation of skin structure and
    implements constraint optimization across molecular, cellular, tissue,
    and organ scales.
    """
    
    def __init__(self, atomspace: CosmeceuticalAtomSpace,
                 reasoning_engine: IngredientReasoningEngine,
                 optimizer: MultiscaleOptimizer):
        self.atomspace = atomspace
        self.reasoning_engine = reasoning_engine
        self.optimizer = optimizer
        
        # Skin model components
        self.skin_layers: Dict[SkinLayer, SkinBarrierProperties] = {}
        self.therapeutic_vectors: Dict[str, TherapeuticVector] = {}
        self.penetration_models: Dict[str, Callable] = {}
        
        # Multiscale constraints
        self.molecular_constraints: Dict[str, Any] = {}
        self.cellular_constraints: Dict[str, Any] = {}
        self.tissue_constraints: Dict[str, Any] = {}
        self.organ_constraints: Dict[str, Any] = {}
        
        # Initialize skin model
        self._initialize_skin_model()
        self._initialize_therapeutic_vectors()
        self._initialize_penetration_models()
    
    def _initialize_skin_model(self):
        """Initialize default skin barrier properties"""
        self.skin_layers = {
            SkinLayer.STRATUM_CORNEUM: SkinBarrierProperties(
                layer=SkinLayer.STRATUM_CORNEUM,
                thickness=15.0,
                permeability_coefficient=1e-6,
                partition_coefficient=0.1,
                diffusion_coefficient=1e-10,
                lipid_content=20.0,
                water_content=15.0,
                ph_value=5.5
            ),
            SkinLayer.VIABLE_EPIDERMIS: SkinBarrierProperties(
                layer=SkinLayer.VIABLE_EPIDERMIS,
                thickness=75.0,
                permeability_coefficient=1e-5,
                partition_coefficient=0.5,
                diffusion_coefficient=1e-9,
                lipid_content=5.0,
                water_content=70.0,
                ph_value=6.5
            ),
            SkinLayer.DERMIS: SkinBarrierProperties(
                layer=SkinLayer.DERMIS,
                thickness=2000.0,
                permeability_coefficient=1e-4,
                partition_coefficient=0.8,
                diffusion_coefficient=1e-8,
                lipid_content=2.0,
                water_content=75.0,
                ph_value=7.0
            ),
            SkinLayer.HYPODERMIS: SkinBarrierProperties(
                layer=SkinLayer.HYPODERMIS,
                thickness=5000.0,
                permeability_coefficient=1e-3,
                partition_coefficient=1.0,
                diffusion_coefficient=1e-7,
                lipid_content=80.0,
                water_content=10.0,
                ph_value=7.2
            )
        }
    
    def _initialize_therapeutic_vectors(self):
        """Initialize common therapeutic vectors"""
        self.therapeutic_vectors = {
            "moisturizing": TherapeuticVector(
                vector_name="moisturizing",
                target_layer=SkinLayer.STRATUM_CORNEUM,
                required_concentration=1.0,
                optimal_concentration=5.0,
                safety_limit=50.0,
                mechanism_of_action="water_retention",
                clinical_endpoint="skin_hydration"
            ),
            "anti_aging": TherapeuticVector(
                vector_name="anti_aging",
                target_layer=SkinLayer.VIABLE_EPIDERMIS,
                required_concentration=0.1,
                optimal_concentration=1.0,
                safety_limit=2.0,
                mechanism_of_action="collagen_synthesis",
                clinical_endpoint="wrinkle_reduction"
            ),
            "brightening": TherapeuticVector(
                vector_name="brightening",
                target_layer=SkinLayer.VIABLE_EPIDERMIS,
                required_concentration=0.5,
                optimal_concentration=2.0,
                safety_limit=5.0,
                mechanism_of_action="melanogenesis_inhibition",
                clinical_endpoint="pigmentation_reduction"
            ),
            "anti_inflammatory": TherapeuticVector(
                vector_name="anti_inflammatory",
                target_layer=SkinLayer.DERMIS,
                required_concentration=0.1,
                optimal_concentration=0.5,
                safety_limit=2.0,
                mechanism_of_action="cytokine_modulation",
                clinical_endpoint="inflammation_reduction"
            )
        }
    
    def _initialize_penetration_models(self):
        """Initialize penetration models for different mechanisms"""
        self.penetration_models = {
            "fick_diffusion": self._fick_diffusion_model,
            "permeation_coefficient": self._permeation_coefficient_model,
            "lipophilic_pathway": self._lipophilic_pathway_model,
            "hydrophilic_pathway": self._hydrophilic_pathway_model
        }
    
    def _fick_diffusion_model(self, ingredient_name: str, formulation_conc: float,
                            target_layer: SkinLayer, time_hours: float = 24.0) -> float:
        """Calculate penetration using Fick's diffusion model"""
        # Get ingredient properties
        atom = self.atomspace.get_atom_by_name(ingredient_name)
        if not atom:
            return 0.0
        
        molecular_weight = atom.properties.get("molecular_weight", 500.0)
        log_p = atom.properties.get("log_p", 0.0)
        
        # Estimate diffusion coefficient based on molecular weight
        # D ∝ MW^(-0.5) (simplified relationship)
        base_diffusion = 1e-9  # cm²/s
        mw_factor = (500.0 / molecular_weight) ** 0.5
        diffusion_coeff = base_diffusion * mw_factor
        
        # Get skin layer properties
        layer_properties = self.skin_layers.get(target_layer)
        if not layer_properties:
            return 0.0
        
        # Steady-state flux calculation (simplified)
        thickness_cm = layer_properties.thickness / 10000.0  # μm to cm
        partition_coeff = layer_properties.partition_coefficient
        
        # Adjust partition coefficient based on lipophilicity
        if log_p > 2.0:  # lipophilic
            partition_coeff *= (1.0 + log_p / 5.0)
        elif log_p < 0.0:  # hydrophilic
            partition_coeff *= (1.0 + abs(log_p) / 10.0)
        
        # Calculate steady-state concentration
        flux = (diffusion_coeff * partition_coeff * formulation_conc) / thickness_cm
        
        # Time-dependent penetration (simplified)
        time_factor = 1.0 - math.exp(-time_hours / 24.0)  # approaches steady-state
        
        return flux * time_factor
    
    def _permeation_coefficient_model(self, ingredient_name: str, formulation_conc: float,
                                    target_layer: SkinLayer, time_hours: float = 24.0) -> float:
        """Calculate penetration using permeation coefficient model"""
        layer_properties = self.skin_layers.get(target_layer)
        if not layer_properties:
            return 0.0
        
        # Get ingredient properties
        atom = self.atomspace.get_atom_by_name(ingredient_name)
        molecular_weight = atom.properties.get("molecular_weight", 500.0) if atom else 500.0
        
        # Adjust permeability coefficient based on molecular weight
        base_permeability = layer_properties.permeability_coefficient
        mw_factor = (300.0 / molecular_weight) ** 0.6  # MW effect on permeability
        adjusted_permeability = base_permeability * mw_factor
        
        # Calculate cumulative penetration
        penetrated_amount = adjusted_permeability * formulation_conc * time_hours * 3600.0
        
        return penetrated_amount
    
    def _lipophilic_pathway_model(self, ingredient_name: str, formulation_conc: float,
                                target_layer: SkinLayer, time_hours: float = 24.0) -> float:
        """Model penetration through lipophilic pathway"""
        atom = self.atomspace.get_atom_by_name(ingredient_name)
        if not atom:
            return 0.0
        
        log_p = atom.properties.get("log_p", 0.0)
        
        # Lipophilic pathway is favored for log P > 1
        if log_p > 1.0:
            lipophilic_enhancement = min(5.0, log_p)
            base_penetration = self._fick_diffusion_model(ingredient_name, formulation_conc, target_layer, time_hours)
            return base_penetration * lipophilic_enhancement
        else:
            return self._fick_diffusion_model(ingredient_name, formulation_conc, target_layer, time_hours) * 0.1
    
    def _hydrophilic_pathway_model(self, ingredient_name: str, formulation_conc: float,
                                 target_layer: SkinLayer, time_hours: float = 24.0) -> float:
        """Model penetration through hydrophilic pathway"""
        atom = self.atomspace.get_atom_by_name(ingredient_name)
        if not atom:
            return 0.0
        
        log_p = atom.properties.get("log_p", 0.0)
        molecular_weight = atom.properties.get("molecular_weight", 500.0)
        
        # Hydrophilic pathway is favored for small, hydrophilic molecules
        if log_p < 0.0 and molecular_weight < 200.0:
            hydrophilic_enhancement = min(3.0, abs(log_p) + (200.0 / molecular_weight))
            base_penetration = self._fick_diffusion_model(ingredient_name, formulation_conc, target_layer, time_hours)
            return base_penetration * hydrophilic_enhancement
        else:
            return self._fick_diffusion_model(ingredient_name, formulation_conc, target_layer, time_hours) * 0.5
    
    def calculate_ingredient_penetration_profile(self, ingredient_name: str,
                                               formulation_concentration: float,
                                               delivery_mechanism: DeliveryMechanism = DeliveryMechanism.PASSIVE_DIFFUSION,
                                               time_hours: float = 24.0) -> IngredientPenetrationProfile:
        """Calculate complete penetration profile for an ingredient"""
        atom = self.atomspace.get_atom_by_name(ingredient_name)
        if not atom:
            return IngredientPenetrationProfile(ingredient_name, 0.0, 0.0)
        
        molecular_weight = atom.properties.get("molecular_weight", 500.0)
        log_p = atom.properties.get("log_p", 0.0)
        
        profile = IngredientPenetrationProfile(
            ingredient_name=ingredient_name,
            molecular_weight=molecular_weight,
            log_p=log_p,
            delivery_mechanism=delivery_mechanism
        )
        
        # Calculate concentration in each skin layer
        cumulative_penetration = formulation_concentration
        
        for layer in [SkinLayer.STRATUM_CORNEUM, SkinLayer.VIABLE_EPIDERMIS, 
                     SkinLayer.DERMIS, SkinLayer.HYPODERMIS]:
            
            # Choose penetration model based on delivery mechanism
            if delivery_mechanism == DeliveryMechanism.PASSIVE_DIFFUSION:
                if log_p > 1.0:
                    penetrated_conc = self._lipophilic_pathway_model(
                        ingredient_name, cumulative_penetration, layer, time_hours
                    )
                else:
                    penetrated_conc = self._hydrophilic_pathway_model(
                        ingredient_name, cumulative_penetration, layer, time_hours
                    )
            else:
                # Enhanced delivery mechanisms
                base_penetration = self._fick_diffusion_model(
                    ingredient_name, cumulative_penetration, layer, time_hours
                )
                
                enhancement_factor = self._get_delivery_enhancement_factor(
                    delivery_mechanism, layer, molecular_weight
                )
                penetrated_conc = base_penetration * enhancement_factor
            
            profile.concentration_profile[layer] = penetrated_conc
            cumulative_penetration = max(0.0, cumulative_penetration - penetrated_conc * 0.1)
            
            # Update penetration depth
            layer_properties = self.skin_layers.get(layer)
            if layer_properties and penetrated_conc > 0.01:  # minimum detectable concentration
                profile.penetration_depth += layer_properties.thickness
        
        # Calculate overall bioavailability
        profile.bioavailability = sum(profile.concentration_profile.values()) / formulation_concentration
        
        return profile
    
    def _get_delivery_enhancement_factor(self, mechanism: DeliveryMechanism,
                                       layer: SkinLayer, molecular_weight: float) -> float:
        """Get enhancement factor for different delivery mechanisms"""
        enhancement_factors = {
            DeliveryMechanism.PASSIVE_DIFFUSION: 1.0,
            DeliveryMechanism.FACILITATED_TRANSPORT: 2.0,
            DeliveryMechanism.LIPOSOMAL_DELIVERY: 3.0,
            DeliveryMechanism.NANOPARTICLE_DELIVERY: 4.0,
            DeliveryMechanism.IONTOPHORESIS: 10.0,
            DeliveryMechanism.MICRONEEDLE: 50.0
        }
        
        base_factor = enhancement_factors.get(mechanism, 1.0)
        
        # Adjust based on skin layer
        if layer == SkinLayer.STRATUM_CORNEUM:
            # Enhancement is most effective for barrier layer
            return base_factor
        elif layer == SkinLayer.VIABLE_EPIDERMIS:
            return base_factor * 0.8
        elif layer == SkinLayer.DERMIS:
            return base_factor * 0.6
        else:  # HYPODERMIS
            return base_factor * 0.4
    
    def optimize_formulation_for_therapeutic_vectors(self, 
                                                   target_vectors: List[str],
                                                   available_ingredients: List[str],
                                                   constraints: Optional[Dict[str, Any]] = None) -> List[FormulationGenome]:
        """
        Optimize formulation to achieve therapeutic vectors at optimal concentrations.
        
        This is the main multiscale optimization function that considers all scales
        simultaneously to maximize clinical effectiveness.
        """
        # Validate therapeutic vectors
        valid_vectors = [v for v in target_vectors if v in self.therapeutic_vectors]
        if not valid_vectors:
            raise ValueError("No valid therapeutic vectors specified")
        
        # Create multiscale optimization objectives
        objectives = [
            OptimizationObjective.CLINICAL_EFFECTIVENESS,
            OptimizationObjective.SAFETY_MAXIMIZATION
        ]
        
        # Add custom fitness evaluator for therapeutic vectors
        self.optimizer.fitness_evaluators[OptimizationObjective.CLINICAL_EFFECTIVENESS] = \
            TherapeuticVectorFitness(self, valid_vectors)
        
        # Set up constraints for multiscale optimization
        multiscale_constraints = constraints or {}
        
        # Add therapeutic vector constraints
        multiscale_constraints.update({
            "therapeutic_vectors": valid_vectors,
            "target_skin_layers": [self.therapeutic_vectors[v].target_layer for v in valid_vectors],
            "required_penetration": True
        })
        
        # Filter available ingredients based on therapeutic potential
        therapeutic_ingredients = self._filter_ingredients_for_vectors(
            available_ingredients, valid_vectors
        )
        
        # Run optimization
        optimal_solutions = self.optimizer.optimize(
            objectives=objectives,
            target_ingredients=therapeutic_ingredients,
            constraints=multiscale_constraints
        )
        
        # Post-process solutions to include penetration profiles
        enhanced_solutions = []
        for solution in optimal_solutions:
            enhanced_solution = self._enhance_solution_with_penetration_data(solution)
            enhanced_solutions.append(enhanced_solution)
        
        return enhanced_solutions
    
    def _filter_ingredients_for_vectors(self, available_ingredients: List[str],
                                      therapeutic_vectors: List[str]) -> List[str]:
        """Filter ingredients based on therapeutic vector requirements"""
        relevant_ingredients = []
        
        for ingredient_name in available_ingredients:
            atom = self.atomspace.get_atom_by_name(ingredient_name)
            if not atom:
                continue
            
            # Check if ingredient has functions matching therapeutic vectors
            ingredient_functions = atom.properties.get("functions", [])
            
            for vector in therapeutic_vectors:
                vector_obj = self.therapeutic_vectors[vector]
                
                # Simple mapping of therapeutic vectors to ingredient functions
                vector_function_mapping = {
                    "moisturizing": ["humectant", "emollient"],
                    "anti_aging": ["anti_aging", "antioxidant"],
                    "brightening": ["skin_brightening", "antioxidant"],
                    "anti_inflammatory": ["anti_inflammatory", "soothing"]
                }
                
                required_functions = vector_function_mapping.get(vector, [])
                if any(func in ingredient_functions for func in required_functions):
                    relevant_ingredients.append(ingredient_name)
                    break
        
        return relevant_ingredients
    
    def _enhance_solution_with_penetration_data(self, solution: FormulationGenome) -> FormulationGenome:
        """Enhance solution with penetration profile data"""
        enhanced_solution = solution.clone()
        
        penetration_data = {}
        for ingredient_name, concentration in solution.ingredients.items():
            profile = self.calculate_ingredient_penetration_profile(
                ingredient_name, concentration
            )
            penetration_data[ingredient_name] = {
                "penetration_depth": profile.penetration_depth,
                "bioavailability": profile.bioavailability,
                "layer_concentrations": profile.concentration_profile
            }
        
        enhanced_solution.properties["penetration_profiles"] = penetration_data
        
        return enhanced_solution
    
    def evaluate_therapeutic_vector_achievement(self, formulation: FormulationGenome,
                                              therapeutic_vectors: List[str]) -> Dict[str, float]:
        """Evaluate how well a formulation achieves therapeutic vectors"""
        achievement_scores = {}
        
        for vector_name in therapeutic_vectors:
            if vector_name not in self.therapeutic_vectors:
                continue
            
            vector = self.therapeutic_vectors[vector_name]
            target_layer = vector.target_layer
            required_conc = vector.required_concentration
            optimal_conc = vector.optimal_concentration
            safety_limit = vector.safety_limit
            
            # Calculate total effective concentration in target layer
            total_effective_conc = 0.0
            
            for ingredient_name, formulation_conc in formulation.ingredients.items():
                # Get penetration profile for this ingredient
                profile = self.calculate_ingredient_penetration_profile(ingredient_name, formulation_conc)
                
                # Get concentration in target layer
                layer_conc = profile.concentration_profile.get(target_layer, 0.0)
                
                # Check if ingredient contributes to this therapeutic vector
                atom = self.atomspace.get_atom_by_name(ingredient_name)
                if atom:
                    functions = atom.properties.get("functions", [])
                    vector_functions = self._get_vector_functions(vector_name)
                    
                    if any(func in functions for func in vector_functions):
                        total_effective_conc += layer_conc
            
            # Calculate achievement score
            if total_effective_conc < required_conc:
                # Below minimum effective concentration
                achievement_scores[vector_name] = total_effective_conc / required_conc * 0.5
            elif total_effective_conc <= optimal_conc:
                # Between minimum and optimal
                achievement_scores[vector_name] = 0.5 + (total_effective_conc - required_conc) / (optimal_conc - required_conc) * 0.4
            elif total_effective_conc <= safety_limit:
                # Between optimal and safety limit
                excess_factor = (total_effective_conc - optimal_conc) / (safety_limit - optimal_conc)
                achievement_scores[vector_name] = 0.9 - excess_factor * 0.3
            else:
                # Above safety limit
                achievement_scores[vector_name] = 0.1
        
        return achievement_scores
    
    def _get_vector_functions(self, vector_name: str) -> List[str]:
        """Get ingredient functions associated with a therapeutic vector"""
        vector_function_mapping = {
            "moisturizing": ["humectant", "emollient", "occlusive"],
            "anti_aging": ["anti_aging", "antioxidant", "collagen_booster"],
            "brightening": ["skin_brightening", "antioxidant", "tyrosinase_inhibitor"],
            "anti_inflammatory": ["anti_inflammatory", "soothing", "calming"]
        }
        
        return vector_function_mapping.get(vector_name, [])
    
    def get_multiscale_statistics(self) -> Dict[str, Any]:
        """Get statistics about the multiscale system"""
        return {
            "skin_layers": len(self.skin_layers),
            "therapeutic_vectors": len(self.therapeutic_vectors),
            "penetration_models": len(self.penetration_models),
            "constraint_categories": {
                "molecular": len(self.molecular_constraints),
                "cellular": len(self.cellular_constraints),
                "tissue": len(self.tissue_constraints),
                "organ": len(self.organ_constraints)
            },
            "available_delivery_mechanisms": len(DeliveryMechanism),
            "skin_model_depth": sum(layer.thickness for layer in self.skin_layers.values())
        }


class TherapeuticVectorFitness:
    """Fitness evaluator for therapeutic vector achievement"""
    
    def __init__(self, skin_integrator: SkinModelIntegrator, target_vectors: List[str]):
        self.skin_integrator = skin_integrator
        self.target_vectors = target_vectors
    
    def evaluate(self, genome: FormulationGenome, atomspace: CosmeceuticalAtomSpace) -> float:
        """Evaluate fitness based on therapeutic vector achievement"""
        achievement_scores = self.skin_integrator.evaluate_therapeutic_vector_achievement(
            genome, self.target_vectors
        )
        
        if not achievement_scores:
            return 0.0
        
        # Calculate weighted average achievement
        total_score = sum(achievement_scores.values())
        max_possible_score = len(achievement_scores)
        
        return (total_score / max_possible_score) * 100.0 if max_possible_score > 0 else 0.0
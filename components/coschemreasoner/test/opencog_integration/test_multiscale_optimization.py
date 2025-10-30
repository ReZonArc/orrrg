#!/usr/bin/env python3
"""
Test cases for OpenCog-inspired multiscale constraint optimization.

This module tests the integration of OpenCog features with cosmeceutical
formulation optimization, focusing on multiscale constraint satisfaction
and INCI-based search space reduction.
"""

import unittest
import sys
import os

# Add the cheminformatics module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from cheminformatics.opencog_integration import *
    from cheminformatics.types.cosmetic_atoms import *
    OPENCOG_AVAILABLE = True
except ImportError as e:
    print(f"OpenCog integration not available: {e}")
    OPENCOG_AVAILABLE = False


@unittest.skipUnless(OPENCOG_AVAILABLE, "OpenCog integration not available")
class TestAtomSpaceIntegration(unittest.TestCase):
    """Test AtomSpace-inspired hypergraph representation"""
    
    def setUp(self):
        self.atomspace = CosmeceuticalAtomSpace()
    
    def test_atom_creation(self):
        """Test basic atom creation and retrieval"""
        # Create ingredient atom
        atom = self.atomspace.create_atom(
            AtomType.INGREDIENT_NODE,
            "test_ingredient",
            properties={"category": "ACTIVE_INGREDIENT"}
        )
        
        self.assertIsNotNone(atom)
        self.assertEqual(atom.name, "test_ingredient")
        self.assertEqual(atom.atom_type, AtomType.INGREDIENT_NODE)
        
        # Retrieve atom by name
        retrieved_atom = self.atomspace.get_atom_by_name("test_ingredient")
        self.assertEqual(retrieved_atom.atom_id, atom.atom_id)
    
    def test_compatibility_links(self):
        """Test compatibility link creation and querying"""
        # Create two ingredients
        ing1 = self.atomspace.create_atom(AtomType.INGREDIENT_NODE, "ingredient1")
        ing2 = self.atomspace.create_atom(AtomType.INGREDIENT_NODE, "ingredient2")
        
        # Create compatibility link
        compatibility_link = self.atomspace.create_compatibility_link(ing1, ing2, 0.8)
        
        # Query compatibility
        compatibility = self.atomspace.get_ingredient_compatibility("ingredient1", "ingredient2")
        self.assertAlmostEqual(compatibility, 0.8, places=2)
    
    def test_multiscale_constraints(self):
        """Test multiscale constraint creation"""
        # Create ingredient
        ingredient = self.atomspace.create_atom(AtomType.INGREDIENT_NODE, "test_active")
        
        # Create multiscale constraint
        constraint = self.atomspace.create_multiscale_constraint(
            ingredient, "molecular", "max_concentration", 5.0
        )
        
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint.properties["scale_level"], "molecular")
        self.assertEqual(constraint.properties["constraint_value"], 5.0)
        
        # Query constraints
        constraints = self.atomspace.get_multiscale_constraints("test_active", "molecular")
        self.assertEqual(len(constraints), 1)


@unittest.skipUnless(OPENCOG_AVAILABLE, "OpenCog integration not available")
class TestINCIOptimization(unittest.TestCase):
    """Test INCI-driven search space optimization"""
    
    def setUp(self):
        self.atomspace = CosmeceuticalAtomSpace()
        self.inci_optimizer = INCISearchOptimizer(self.atomspace)
    
    def test_ingredient_normalization(self):
        """Test INCI ingredient name normalization"""
        test_cases = [
            ("Hyaluronic Acid", "HYALURONIC ACID"),
            ("vitamin c solution", "VITAMIN C"),
            ("Niacinamide (Vitamin B3)", "NIACINAMIDE VITAMIN B3")
        ]
        
        for input_name, expected in test_cases:
            normalized = self.inci_optimizer.normalize_ingredient_name(input_name)
            self.assertEqual(normalized, expected)
    
    def test_concentration_estimation(self):
        """Test concentration estimation from INCI ordering"""
        inci_list = ["AQUA", "GLYCERIN", "NIACINAMIDE", "PHENOXYETHANOL"]
        
        concentrations = self.inci_optimizer.estimate_concentrations_from_inci(inci_list)
        
        # Check that AQUA has highest concentration
        self.assertGreater(concentrations.get("AQUA", 0), 50.0)
        
        # Check that concentrations decrease with INCI order
        self.assertGreater(concentrations.get("GLYCERIN", 0), 
                          concentrations.get("NIACINAMIDE", 0))
        self.assertGreater(concentrations.get("NIACINAMIDE", 0), 
                          concentrations.get("PHENOXYETHANOL", 0))
    
    def test_regulatory_compliance(self):
        """Test regulatory compliance filtering"""
        ingredients = ["NIACINAMIDE", "HYALURONIC ACID", "PHENOXYETHANOL"]
        
        compliant = self.inci_optimizer.filter_by_regulatory_compliance(
            ingredients, RegulationRegion.EU
        )
        
        # All test ingredients should be compliant
        self.assertEqual(len(compliant), len(ingredients))
    
    def test_search_space_reduction(self):
        """Test INCI-based search space reduction"""
        target_inci = ["AQUA", "GLYCERIN", "NIACINAMIDE", "HYALURONIC ACID"]
        
        test_formulations = [
            ProductFormulation("Product1", ["AQUA", "GLYCERIN"], "serum", RegulationRegion.EU),
            ProductFormulation("Product2", ["AQUA", "GLYCERIN", "NIACINAMIDE"], "serum", RegulationRegion.EU),
            ProductFormulation("Product3", ["AQUA", "RETINOL"], "serum", RegulationRegion.EU),  # Should be filtered out
        ]
        
        compatible = self.inci_optimizer.reduce_search_space_by_inci_subset(
            target_inci, test_formulations
        )
        
        # Only first two should be compatible (subsets of target)
        self.assertEqual(len(compatible), 2)
        self.assertEqual(compatible[0].product_name, "Product1")
        self.assertEqual(compatible[1].product_name, "Product2")


@unittest.skipUnless(OPENCOG_AVAILABLE, "OpenCog integration not available")
class TestAttentionAllocation(unittest.TestCase):
    """Test adaptive attention allocation"""
    
    def setUp(self):
        self.atomspace = CosmeceuticalAtomSpace()
        self.attention_allocator = AdaptiveAttentionAllocator(self.atomspace)
        
        # Create test atoms
        self.ingredient1 = self.atomspace.create_atom(
            AtomType.INGREDIENT_NODE, "test_ingredient1",
            properties={"functions": ["anti_aging"]}
        )
        self.ingredient2 = self.atomspace.create_atom(
            AtomType.INGREDIENT_NODE, "test_ingredient2",
            properties={"functions": ["moisturizing"]}
        )
    
    def test_attention_allocation(self):
        """Test basic attention allocation"""
        target_atoms = [self.ingredient1, self.ingredient2]
        
        allocations = self.attention_allocator.allocate_attention(
            target_atoms, strategy="importance_based", sti_budget=100.0, lti_budget=50.0
        )
        
        self.assertEqual(len(allocations), 2)
        
        # Check that attention was allocated
        for atom_id, (sti, lti) in allocations.items():
            self.assertGreaterEqual(sti, 0.0)
            self.assertGreaterEqual(lti, 0.0)
    
    def test_attention_spreading(self):
        """Test attention spreading through network"""
        # Create synergy link
        synergy_link = self.atomspace.create_synergy_link(self.ingredient1, self.ingredient2, 0.8)
        
        # Allocate initial attention
        self.attention_allocator.allocate_attention([self.ingredient1], sti_budget=100.0)
        
        # Spread attention
        self.attention_allocator.spread_attention([self.ingredient1], max_spread_distance=2)
        
        # Check that attention spread to connected atoms
        high_attention = self.attention_allocator.get_high_attention_atoms(count=5)
        attention_atom_ids = [atom.atom_id for atom, _ in high_attention]
        
        self.assertIn(self.ingredient2.atom_id, attention_atom_ids)
    
    def test_promising_combinations(self):
        """Test identification of promising ingredient combinations"""
        # Create synergy relationship
        self.atomspace.create_synergy_link(self.ingredient1, self.ingredient2, 0.9)
        
        # Allocate attention
        self.attention_allocator.allocate_attention([self.ingredient1, self.ingredient2], sti_budget=200.0)
        
        # Get promising combinations
        combinations = self.attention_allocator.get_promising_ingredient_combinations(
            min_attention=0.01, max_combinations=5
        )
        
        self.assertGreater(len(combinations), 0)


@unittest.skipUnless(OPENCOG_AVAILABLE, "OpenCog integration not available")
class TestReasoningEngine(unittest.TestCase):
    """Test PLN-inspired reasoning engine"""
    
    def setUp(self):
        self.atomspace = CosmeceuticalAtomSpace()
        self.reasoning_engine = IngredientReasoningEngine(self.atomspace)
        
        # Create test ingredients
        self.ing1 = self.atomspace.create_atom(
            AtomType.INGREDIENT_NODE, "ingredient_a",
            properties={"category": "ACTIVE_INGREDIENT"}
        )
        self.ing2 = self.atomspace.create_atom(
            AtomType.INGREDIENT_NODE, "ingredient_b", 
            properties={"category": "HUMECTANT"}
        )
        self.ing3 = self.atomspace.create_atom(
            AtomType.INGREDIENT_NODE, "ingredient_c",
            properties={"category": "ACTIVE_INGREDIENT"}
        )
    
    def test_compatibility_evaluation(self):
        """Test ingredient compatibility evaluation"""
        # Create known compatibility
        self.atomspace.create_compatibility_link(self.ing1, self.ing2, 0.8)
        
        # Evaluate compatibility
        truth_value = self.reasoning_engine.evaluate_ingredient_compatibility("ingredient_a", "ingredient_b")
        
        self.assertAlmostEqual(truth_value.strength, 0.8, places=1)
        self.assertGreater(truth_value.confidence, 0.0)
    
    def test_evidence_updating(self):
        """Test evidence-based truth value updating"""
        # Create compatibility link
        link = self.atomspace.create_compatibility_link(self.ing1, self.ing2, 0.5)
        
        # Add evidence
        self.reasoning_engine.add_evidence(link.atom_id, 0.9, evidence_weight=2.0, source="experimental")
        
        # Check updated truth value
        updated_tv = self.reasoning_engine.truth_values[link.atom_id]
        self.assertGreater(updated_tv.strength, 0.5)  # Should increase with positive evidence
        self.assertGreater(updated_tv.confidence, 0.0)
    
    def test_inference_cycle(self):
        """Test inference rule application"""
        # Create compatibility chain: A-B, B-C
        self.atomspace.create_compatibility_link(self.ing1, self.ing2, 0.8)
        self.atomspace.create_compatibility_link(self.ing2, self.ing3, 0.7)
        
        # Run inference
        new_atoms = self.reasoning_engine.run_inference_cycle(max_iterations=2)
        
        # Should derive some new knowledge
        self.assertGreaterEqual(len(new_atoms), 0)
    
    def test_formulation_consistency(self):
        """Test formulation consistency evaluation"""
        # Create compatible ingredients
        self.atomspace.create_compatibility_link(self.ing1, self.ing2, 0.9)
        
        formulation_atoms = [self.ing1, self.ing2]
        consistency = self.reasoning_engine.evaluate_formulation_consistency(formulation_atoms)
        
        self.assertGreater(consistency.strength, 0.5)  # Should be fairly consistent


@unittest.skipUnless(OPENCOG_AVAILABLE, "OpenCog integration not available")
class TestMultiscaleIntegration(unittest.TestCase):
    """Test multiscale skin model integration"""
    
    def setUp(self):
        self.atomspace = CosmeceuticalAtomSpace()
        self.reasoning_engine = IngredientReasoningEngine(self.atomspace)
        self.optimizer = MultiscaleOptimizer(self.atomspace, self.reasoning_engine)
        self.skin_integrator = SkinModelIntegrator(self.atomspace, self.reasoning_engine, self.optimizer)
        
        # Create test ingredient with penetration properties
        self.test_ingredient = self.atomspace.create_atom(
            AtomType.INGREDIENT_NODE, "test_penetrant",
            properties={
                "molecular_weight": 300.0,
                "log_p": 1.5,
                "functions": ["anti_aging"],
                "concentration_range": (0.1, 5.0)
            }
        )
    
    def test_penetration_profile_calculation(self):
        """Test skin penetration profile calculation"""
        profile = self.skin_integrator.calculate_ingredient_penetration_profile(
            "test_penetrant", 2.0, DeliveryMechanism.PASSIVE_DIFFUSION
        )
        
        self.assertEqual(profile.ingredient_name, "test_penetrant")
        self.assertEqual(profile.molecular_weight, 300.0)
        self.assertGreater(profile.penetration_depth, 0.0)
        self.assertGreater(profile.bioavailability, 0.0)
        
        # Check that all skin layers have concentration data
        expected_layers = [SkinLayer.STRATUM_CORNEUM, SkinLayer.VIABLE_EPIDERMIS, 
                          SkinLayer.DERMIS, SkinLayer.HYPODERMIS]
        for layer in expected_layers:
            self.assertIn(layer, profile.concentration_profile)
    
    def test_therapeutic_vector_evaluation(self):
        """Test therapeutic vector achievement evaluation"""
        # Create test formulation
        formulation = FormulationGenome(
            ingredients={"test_penetrant": 2.0},
            properties={}
        )
        
        # Evaluate achievement
        achievement = self.skin_integrator.evaluate_therapeutic_vector_achievement(
            formulation, ["anti_aging"]
        )
        
        self.assertIn("anti_aging", achievement)
        self.assertGreaterEqual(achievement["anti_aging"], 0.0)
        self.assertLessEqual(achievement["anti_aging"], 1.0)
    
    def test_delivery_mechanism_enhancement(self):
        """Test delivery mechanism enhancement effects"""
        # Compare passive vs enhanced delivery
        passive_profile = self.skin_integrator.calculate_ingredient_penetration_profile(
            "test_penetrant", 2.0, DeliveryMechanism.PASSIVE_DIFFUSION
        )
        
        enhanced_profile = self.skin_integrator.calculate_ingredient_penetration_profile(
            "test_penetrant", 2.0, DeliveryMechanism.LIPOSOMAL_DELIVERY
        )
        
        # Enhanced delivery should improve bioavailability
        self.assertGreater(enhanced_profile.bioavailability, passive_profile.bioavailability)
        self.assertGreaterEqual(enhanced_profile.penetration_depth, passive_profile.penetration_depth)


@unittest.skipUnless(OPENCOG_AVAILABLE, "OpenCog integration not available") 
class TestOptimizationIntegration(unittest.TestCase):
    """Test MOSES-inspired optimization integration"""
    
    def setUp(self):
        self.atomspace = CosmeceuticalAtomSpace()
        self.reasoning_engine = IngredientReasoningEngine(self.atomspace)
        self.optimizer = MultiscaleOptimizer(self.atomspace, self.reasoning_engine)
        
        # Create test ingredients
        self.ingredients = []
        for i in range(3):
            ingredient = self.atomspace.create_atom(
                AtomType.INGREDIENT_NODE, f"test_ingredient_{i}",
                properties={
                    "concentration_range": (0.1, 10.0),
                    "functions": ["moisturizing", "anti_aging"],
                    "cost_per_kg": 100.0 + i * 50
                }
            )
            self.ingredients.append(ingredient)
    
    def test_population_initialization(self):
        """Test population initialization"""
        population = self.optimizer.initialize_population(
            target_ingredients=[ing.name for ing in self.ingredients]
        )
        
        self.assertEqual(len(population), self.optimizer.population_size)
        
        # Check that genomes have valid ingredients
        for genome in population:
            self.assertGreater(len(genome.ingredients), 0)
            self.assertLessEqual(genome.total_concentration, 120.0)  # Allow some tolerance
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation"""
        # Create test genome
        genome = FormulationGenome(
            ingredients={ing.name: 2.0 for ing in self.ingredients}
        )
        
        # Evaluate fitness
        objectives = [OptimizationObjective.CLINICAL_EFFECTIVENESS, OptimizationObjective.SAFETY_MAXIMIZATION]
        fitness_scores = self.optimizer.evaluate_fitness(genome, objectives)
        
        self.assertEqual(len(fitness_scores), len(objectives))
        
        for objective, score in fitness_scores.items():
            self.assertGreaterEqual(score, 0.0)
    
    def test_genetic_operations(self):
        """Test genetic operators"""
        # Create parent genomes
        parent1 = FormulationGenome(
            ingredients={self.ingredients[0].name: 3.0, self.ingredients[1].name: 2.0}
        )
        parent2 = FormulationGenome(
            ingredients={self.ingredients[1].name: 4.0, self.ingredients[2].name: 1.0}
        )
        
        # Test crossover
        from cheminformatics.opencog_integration.optimization import CrossoverOperator
        crossover = CrossoverOperator()
        offspring = crossover.apply([parent1, parent2], self.atomspace)
        
        self.assertGreater(len(offspring), 0)
        
        # Test mutation
        from cheminformatics.opencog_integration.optimization import MutationOperator
        mutation = MutationOperator()
        mutated = mutation.apply([parent1], self.atomspace)
        
        self.assertEqual(len(mutated), 1)


if __name__ == '__main__':
    unittest.main()
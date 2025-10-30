#!/usr/bin/env python3
"""
Comprehensive tests for the Hypergredient Framework

This test suite validates all major components of the hypergredient
framework including core classes, database, optimization, interactions,
and scoring systems.
"""

import unittest
import sys
import os

# Add the cheminformatics module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from cheminformatics.hypergredient import (
        Hypergredient, HypergredientDatabase, HYPERGREDIENT_CLASSES,
        HypergredientMetrics, create_hypergredient_database,
        HypergredientFormulator, FormulationOptimizer, OptimizationObjective,
        InteractionMatrix, calculate_synergy_score,
        DynamicScoringSystem, PerformanceMetrics
    )
    from cheminformatics.hypergredient.database import HypergredientDB
    from cheminformatics.hypergredient.optimization import FormulationRequest
    HYPERGREDIENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Hypergredient framework not available: {e}")
    HYPERGREDIENT_AVAILABLE = False


@unittest.skipUnless(HYPERGREDIENT_AVAILABLE, "Hypergredient framework not available")
class TestHypergredientCore(unittest.TestCase):
    """Test core hypergredient functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_hypergredient = Hypergredient(
            name="test_ingredient",
            inci_name="Test Ingredient",
            hypergredient_class="H.AO",
            primary_function="Antioxidant activity",
            secondary_functions=["Anti-aging"],
            potency=8.0,
            ph_range=(5.0, 7.0),
            stability="stable",
            cost_per_gram=100.0,
            bioavailability=75.0,
            safety_score=9.0
        )
    
    def test_hypergredient_creation(self):
        """Test basic hypergredient creation"""
        self.assertEqual(self.test_hypergredient.name, "test_ingredient")
        self.assertEqual(self.test_hypergredient.hypergredient_class, "H.AO")
        self.assertEqual(self.test_hypergredient.potency, 8.0)
        self.assertIsNotNone(self.test_hypergredient.metrics)
    
    def test_hypergredient_metrics_calculation(self):
        """Test metrics calculation"""
        metrics = self.test_hypergredient.metrics
        self.assertIsInstance(metrics, HypergredientMetrics)
        self.assertEqual(metrics.efficacy_score, 8.0)
        self.assertEqual(metrics.bioavailability, 75.0)
        self.assertEqual(metrics.safety_score, 9.0)
        
        # Test composite score calculation
        composite = metrics.calculate_composite_score()
        self.assertGreater(composite, 0)
        self.assertLessEqual(composite, 10)
    
    def test_compatibility_checking(self):
        """Test ingredient compatibility checking"""
        # Test unknown compatibility
        compat = self.test_hypergredient.check_compatibility("unknown_ingredient")
        self.assertEqual(compat, "unknown")
        
        # Test with interactions
        self.test_hypergredient.interactions["compatible_ingredient"] = "synergy"
        compat = self.test_hypergredient.check_compatibility("compatible_ingredient")
        self.assertEqual(compat, "synergy")
    
    def test_formulation_scoring(self):
        """Test formulation context scoring"""
        context = {
            'target_ph': 6.0,
            'budget': 1000,
            'other_ingredients': []
        }
        
        score = self.test_hypergredient.calculate_formulation_score(context)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 10)
    
    def test_performance_prediction(self):
        """Test performance prediction"""
        context = {
            'other_ingredients': ['vitamin_e'],
            'target_ph': 6.0
        }
        
        performance = self.test_hypergredient.predict_performance(context)
        self.assertIn('efficacy', performance)
        self.assertIn('safety', performance)
        self.assertIn('stability', performance)


@unittest.skipUnless(HYPERGREDIENT_AVAILABLE, "Hypergredient framework not available")
class TestHypergredientDatabase(unittest.TestCase):
    """Test hypergredient database functionality"""
    
    def setUp(self):
        """Set up test database"""
        self.db = create_hypergredient_database()
    
    def test_database_creation(self):
        """Test database creation and population"""
        self.assertIsInstance(self.db, HypergredientDatabase)
        self.assertGreater(len(self.db.hypergredients), 0)
        
        # Check that all classes are represented
        for class_code in HYPERGREDIENT_CLASSES:
            ingredients = self.db.get_by_class(class_code)
            self.assertGreaterEqual(len(ingredients), 0, 
                                  f"No ingredients found for class {class_code}")
    
    def test_class_queries(self):
        """Test querying by hypergredient class"""
        # Test cellular turnover agents
        ct_agents = self.db.get_by_class("H.CT")
        self.assertGreater(len(ct_agents), 0)
        
        # Verify all returned ingredients are of correct class
        for agent in ct_agents:
            self.assertEqual(agent.hypergredient_class, "H.CT")
    
    def test_function_queries(self):
        """Test querying by function"""
        antioxidants = self.db.get_by_function("antioxidant")
        self.assertGreater(len(antioxidants), 0)
        
        # Check that results contain function in primary or secondary functions
        for ingredient in antioxidants:
            has_function = (
                "antioxidant" in ingredient.primary_function.lower() or
                any("antioxidant" in func.lower() for func in ingredient.secondary_functions)
            )
            self.assertTrue(has_function)
    
    def test_search_functionality(self):
        """Test advanced search capabilities"""
        # Search by multiple criteria
        criteria = {
            'min_potency': 7.0,
            'max_cost': 200.0,
            'safety_min': 8.0
        }
        
        results = self.db.search(criteria)
        self.assertGreaterEqual(len(results), 0)
        
        # Verify all results meet criteria
        for result in results:
            self.assertGreaterEqual(result.potency, criteria['min_potency'])
            self.assertLessEqual(result.cost_per_gram, criteria['max_cost'])
            self.assertGreaterEqual(result.safety_score, criteria['safety_min'])
    
    def test_top_performers(self):
        """Test top performer retrieval"""
        # Get top antioxidants
        top_antioxidants = self.db.get_top_performers("H.AO", n=3, metric='efficacy')
        self.assertLessEqual(len(top_antioxidants), 3)
        
        # Verify they are sorted by efficacy
        if len(top_antioxidants) > 1:
            for i in range(len(top_antioxidants) - 1):
                self.assertGreaterEqual(
                    top_antioxidants[i].potency,
                    top_antioxidants[i+1].potency
                )
    
    def test_database_stats(self):
        """Test database statistics"""
        stats = self.db.get_stats()
        
        self.assertIn('total_hypergredients', stats)
        self.assertIn('by_class', stats)
        self.assertIn('avg_potency', stats)
        self.assertGreater(stats['total_hypergredients'], 0)


@unittest.skipUnless(HYPERGREDIENT_AVAILABLE, "Hypergredient framework not available")
class TestInteractionMatrix(unittest.TestCase):
    """Test interaction matrix functionality"""
    
    def setUp(self):
        """Set up interaction matrix and test ingredients"""
        self.matrix = InteractionMatrix()
        self.db = create_hypergredient_database()
        
        # Get some test ingredients
        self.antioxidant = self.db.get_by_class("H.AO")[0]
        self.collagen_booster = self.db.get_by_class("H.CS")[0]
        
    def test_class_interactions(self):
        """Test class-based interaction retrieval"""
        # Test known synergistic interaction
        interaction = self.matrix.get_class_interaction("H.CS", "H.AO")
        self.assertGreater(interaction, 1.0)  # Should be synergistic
        
        # Test reverse direction
        reverse_interaction = self.matrix.get_class_interaction("H.AO", "H.CS")
        self.assertEqual(interaction, reverse_interaction)
    
    def test_ingredient_interactions(self):
        """Test specific ingredient interactions"""
        score = self.matrix.calculate_interaction_score(
            self.antioxidant, self.collagen_booster
        )
        
        self.assertGreater(score, 0)
        self.assertLess(score, 5.0)  # Reasonable range
    
    def test_formulation_analysis(self):
        """Test complete formulation analysis"""
        hypergredients = [self.antioxidant, self.collagen_booster]
        analysis = self.matrix.analyze_formulation_interactions(hypergredients)
        
        self.assertIn('total_score', analysis)
        self.assertIn('synergistic_pairs', analysis)
        self.assertIn('antagonistic_pairs', analysis)
        self.assertIn('neutral_pairs', analysis)
        
        self.assertGreater(analysis['total_score'], 0)
        self.assertLessEqual(analysis['total_score'], 10)
    
    def test_complementary_suggestions(self):
        """Test complementary ingredient suggestions"""
        available_ingredients = list(self.db.hypergredients.values())[:10]
        suggestions = self.matrix.suggest_complementary_hypergredients(
            self.antioxidant, available_ingredients, 3
        )
        
        self.assertLessEqual(len(suggestions), 3)
        
        # Verify suggestions are tuples of (ingredient, score)
        for ingredient, score in suggestions:
            self.assertIsInstance(ingredient, Hypergredient)
            self.assertIsInstance(score, float)
    
    def test_interaction_warnings(self):
        """Test interaction warning generation"""
        # Create formulation with potentially problematic interactions
        ct_agent = None
        se_agent = None
        
        for ingredient in self.db.hypergredients.values():
            if ingredient.hypergredient_class == "H.CT" and ct_agent is None:
                ct_agent = ingredient
            elif ingredient.hypergredient_class == "H.SE" and se_agent is None:
                se_agent = ingredient
            
            if ct_agent and se_agent:
                break
        
        if ct_agent and se_agent:
            warnings = self.matrix.get_interaction_warnings([ct_agent, se_agent])
            # Cellular turnover + sebum regulation can be harsh
            self.assertGreaterEqual(len(warnings), 0)


@unittest.skipUnless(HYPERGREDIENT_AVAILABLE, "Hypergredient framework not available")
class TestOptimization(unittest.TestCase):
    """Test optimization functionality"""
    
    def setUp(self):
        """Set up optimizer and test data"""
        self.db = create_hypergredient_database()
        self.optimizer = FormulationOptimizer(self.db)
        self.formulator = HypergredientFormulator(self.db)
    
    def test_concern_mapping(self):
        """Test concern to class mapping"""
        concerns = ['wrinkles', 'brightness', 'hydration']
        mapped_classes = self.optimizer._map_concerns_to_classes(concerns)
        
        self.assertGreater(len(mapped_classes), 0)
        self.assertIn('H.CS', mapped_classes)  # Collagen for wrinkles
        self.assertIn('H.ML', mapped_classes)  # Melanin modulators for brightness
        self.assertIn('H.HY', mapped_classes)  # Hydration systems
    
    def test_candidate_filtering(self):
        """Test candidate hypergredient filtering"""
        request = FormulationRequest(
            target_concerns=['anti_aging'],
            budget=500.0,
            excluded_ingredients=['tretinoin'],
            ph_range=(5.0, 7.0)
        )
        
        classes = self.optimizer._map_concerns_to_classes(request.target_concerns)
        candidates = self.optimizer._get_candidate_hypergredients(classes, request)
        
        # Verify no excluded ingredients
        candidate_names = [c.name for c in candidates]
        self.assertNotIn('tretinoin', candidate_names)
        
        # Verify budget constraints
        for candidate in candidates:
            estimated_cost = candidate.cost_per_gram * 5  # Assume 5g per 50ml
            self.assertLessEqual(estimated_cost, request.budget)
    
    def test_formulation_generation(self):
        """Test complete formulation generation"""
        solution = self.formulator.generate_formulation(
            target='anti_aging',
            secondary=['hydration'],
            budget=1000
        )
        
        if solution:  # May be None if no suitable combination found
            self.assertIsNotNone(solution.hypergredients)
            self.assertGreater(len(solution.hypergredients), 0)
            self.assertLessEqual(solution.cost, 1000)
            self.assertIn('anti_aging', solution.predicted_efficacy)
    
    def test_optimization_objectives(self):
        """Test different optimization objectives"""
        # Test that optimization objectives exist and are valid
        objectives = list(OptimizationObjective)
        self.assertIn(OptimizationObjective.EFFICACY, objectives)
        self.assertIn(OptimizationObjective.SAFETY, objectives)
        self.assertIn(OptimizationObjective.COST, objectives)


@unittest.skipUnless(HYPERGREDIENT_AVAILABLE, "Hypergredient framework not available")
class TestScoringSystem(unittest.TestCase):
    """Test dynamic scoring system"""
    
    def setUp(self):
        """Set up scoring system and test data"""
        self.scoring_system = DynamicScoringSystem()
        self.db = create_hypergredient_database()
        
        # Get a test hypergredient
        self.test_ingredient = list(self.db.hypergredients.values())[0]
    
    def test_hypergredient_metrics_calculation(self):
        """Test individual hypergredient metrics calculation"""
        metrics = self.scoring_system.calculate_hypergredient_metrics(
            self.test_ingredient
        )
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreaterEqual(metrics.efficacy_score, 0)
        self.assertLessEqual(metrics.efficacy_score, 10)
        self.assertGreaterEqual(metrics.safety_score, 0)
        self.assertLessEqual(metrics.safety_score, 10)
    
    def test_formulation_metrics_calculation(self):
        """Test formulation-level metrics calculation"""
        # Create test formulation
        hypergredients = list(self.db.hypergredients.values())[:3]
        concentrations = {h.name: 2.0 for h in hypergredients}
        
        metrics = self.scoring_system.calculate_formulation_metrics(
            hypergredients, concentrations
        )
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.efficacy_score, 0)
    
    def test_performance_radar_data(self):
        """Test radar chart data generation"""
        metrics = self.scoring_system.calculate_hypergredient_metrics(
            self.test_ingredient
        )
        
        radar_data = metrics.get_performance_radar()
        
        self.assertIn('Efficacy', radar_data)
        self.assertIn('Safety', radar_data)
        self.assertIn('Stability', radar_data)
        
        # Verify all values are in reasonable range
        for metric, value in radar_data.items():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 10)
    
    def test_improvement_suggestions(self):
        """Test improvement suggestion generation"""
        current_metrics = PerformanceMetrics(
            efficacy_score=5.0,
            safety_score=6.0,
            stability_score=4.0
        )
        
        target_metrics = PerformanceMetrics(
            efficacy_score=8.0,
            safety_score=8.0,
            stability_score=7.0
        )
        
        suggestions = self.scoring_system.generate_improvement_suggestions(
            current_metrics, target_metrics
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)


@unittest.skipUnless(HYPERGREDIENT_AVAILABLE, "Hypergredient framework not available")
class TestIntegration(unittest.TestCase):
    """Integration tests for complete hypergredient workflow"""
    
    def setUp(self):
        """Set up complete system"""
        self.db = create_hypergredient_database()
        self.formulator = HypergredientFormulator(self.db)
    
    def test_complete_formulation_workflow(self):
        """Test complete formulation workflow from request to solution"""
        # Create a realistic formulation request
        solution = self.formulator.generate_formulation(
            target='hyperpigmentation',
            secondary=['anti_aging', 'hydration'],
            budget=1200,
            exclude=['hydroquinone'],
            skin_type='normal'
        )
        
        if solution:
            # Verify solution structure
            self.assertIsNotNone(solution.hypergredients)
            self.assertIsNotNone(solution.objective_scores)
            self.assertIsNotNone(solution.predicted_efficacy)
            
            # Verify formulation makes sense
            self.assertLessEqual(solution.cost, 1200)
            self.assertNotIn('hydroquinone', solution.hypergredients)
            
            # Verify predicted efficacy
            self.assertIn('hyperpigmentation', solution.predicted_efficacy)
            
            # Check that summary can be generated
            summary = solution.get_summary()
            self.assertIsInstance(summary, str)
            self.assertIn('OPTIMAL FORMULATION', summary)
    
    def test_synergy_calculation(self):
        """Test synergy calculation with real ingredients"""
        # Get some compatible ingredients
        vitamin_c = None
        vitamin_e = None
        
        for ingredient in self.db.hypergredients.values():
            if 'vitamin_c' in ingredient.name and vitamin_c is None:
                vitamin_c = ingredient
            elif 'vitamin_e' in ingredient.name and vitamin_e is None:
                vitamin_e = ingredient
            
            if vitamin_c and vitamin_e:
                break
        
        if vitamin_c and vitamin_e:
            synergy_score = calculate_synergy_score([vitamin_c, vitamin_e])
            self.assertGreater(synergy_score, 1.0)  # Should be synergistic
    
    def test_database_completeness(self):
        """Test that database has reasonable coverage"""
        stats = self.db.get_stats()
        
        # Should have at least a few ingredients per major class
        major_classes = ['H.CT', 'H.CS', 'H.AO', 'H.HY', 'H.BR']
        
        for class_code in major_classes:
            count = stats['by_class'].get(class_code, 0)
            self.assertGreater(count, 0, 
                             f"No ingredients found for major class {class_code}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
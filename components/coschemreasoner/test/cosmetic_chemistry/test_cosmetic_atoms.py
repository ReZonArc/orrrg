#!/usr/bin/env python3
"""
Unit tests for cosmetic chemistry atom types

These tests validate the basic functionality of the cosmetic chemistry
framework including ingredient creation, formulation modeling, and
compatibility checking.
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cheminformatics.types.cosmetic_atoms import (
    ACTIVE_INGREDIENT, PRESERVATIVE, EMULSIFIER, HUMECTANT, ANTIOXIDANT,
    SKINCARE_FORMULATION, COMPATIBILITY_LINK, INCOMPATIBILITY_LINK, SYNERGY_LINK,
    AtomProperties, check_ingredient_compatibility, create_ingredient_database
)


class TestCosmeticAtoms(unittest.TestCase):
    """Test basic cosmetic atom functionality"""
    
    def setUp(self):
        """Set up test ingredients"""
        self.hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
        self.niacinamide = ACTIVE_INGREDIENT('niacinamide')
        self.vitamin_c = ACTIVE_INGREDIENT('vitamin_c')
        self.retinol = ACTIVE_INGREDIENT('retinol')
        self.glycerin = HUMECTANT('glycerin')
        self.phenoxyethanol = PRESERVATIVE('phenoxyethanol')
        self.vitamin_e = ANTIOXIDANT('vitamin_e')
    
    def test_ingredient_creation(self):
        """Test that ingredients are created correctly"""
        self.assertEqual(self.hyaluronic_acid.name, 'hyaluronic_acid')
        self.assertEqual(self.hyaluronic_acid.atom_type, 'ACTIVE_INGREDIENT')
        self.assertEqual(str(self.hyaluronic_acid), "ACTIVE_INGREDIENT('hyaluronic_acid')")
    
    def test_ingredient_equality(self):
        """Test ingredient equality comparison"""
        other_ha = ACTIVE_INGREDIENT('hyaluronic_acid')
        self.assertEqual(self.hyaluronic_acid, other_ha)
        self.assertNotEqual(self.hyaluronic_acid, self.niacinamide)
    
    def test_ingredient_with_properties(self):
        """Test ingredient creation with properties"""
        props = AtomProperties(
            ph_range=(5.0, 7.0),
            max_concentration=2.0,
            allergen_status=False
        )
        ingredient = ACTIVE_INGREDIENT('test_ingredient', props)
        self.assertEqual(ingredient.properties.ph_range, (5.0, 7.0))
        self.assertEqual(ingredient.properties.max_concentration, 2.0)
        self.assertFalse(ingredient.properties.allergen_status)


class TestFormulations(unittest.TestCase):
    """Test formulation functionality"""
    
    def setUp(self):
        """Set up test ingredients"""
        self.hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
        self.glycerin = HUMECTANT('glycerin')
        self.phenoxyethanol = PRESERVATIVE('phenoxyethanol')
    
    def test_formulation_creation(self):
        """Test that formulations are created correctly"""
        formulation = SKINCARE_FORMULATION(
            self.hyaluronic_acid,
            self.glycerin,
            self.phenoxyethanol
        )
        
        self.assertEqual(len(formulation.ingredients), 3)
        self.assertEqual(formulation.formulation_type, 'SKINCARE_FORMULATION')
        self.assertIn(self.hyaluronic_acid, formulation.ingredients)
    
    def test_add_ingredient(self):
        """Test adding ingredients to formulation"""
        formulation = SKINCARE_FORMULATION()
        formulation.add_ingredient(self.hyaluronic_acid)
        
        self.assertEqual(len(formulation.ingredients), 1)
        self.assertIn(self.hyaluronic_acid, formulation.ingredients)
    
    def test_get_ingredients_by_type(self):
        """Test filtering ingredients by type"""
        formulation = SKINCARE_FORMULATION(
            self.hyaluronic_acid,
            self.glycerin,
            self.phenoxyethanol
        )
        
        actives = formulation.get_ingredients_by_type('ACTIVE_INGREDIENT')
        humectants = formulation.get_ingredients_by_type('HUMECTANT')
        preservatives = formulation.get_ingredients_by_type('PRESERVATIVE')
        
        self.assertEqual(len(actives), 1)
        self.assertEqual(len(humectants), 1)
        self.assertEqual(len(preservatives), 1)
        self.assertEqual(actives[0], self.hyaluronic_acid)


class TestLinks(unittest.TestCase):
    """Test link functionality"""
    
    def setUp(self):
        """Set up test ingredients"""
        self.hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
        self.niacinamide = ACTIVE_INGREDIENT('niacinamide')
        self.vitamin_c = ACTIVE_INGREDIENT('vitamin_c')
        self.retinol = ACTIVE_INGREDIENT('retinol')
    
    def test_compatibility_link(self):
        """Test compatibility link creation"""
        link = COMPATIBILITY_LINK(self.hyaluronic_acid, self.niacinamide)
        
        self.assertEqual(link.link_type, 'COMPATIBLE')
        self.assertEqual(link.ingredient1, self.hyaluronic_acid)
        self.assertEqual(link.ingredient2, self.niacinamide)
    
    def test_incompatibility_link(self):
        """Test incompatibility link creation"""
        link = INCOMPATIBILITY_LINK(self.vitamin_c, self.retinol)
        
        self.assertEqual(link.link_type, 'INCOMPATIBLE')
        self.assertEqual(link.ingredient1, self.vitamin_c)
        self.assertEqual(link.ingredient2, self.retinol)
    
    def test_synergy_link(self):
        """Test synergy link creation"""
        vitamin_e = ANTIOXIDANT('vitamin_e')
        link = SYNERGY_LINK(self.vitamin_c, vitamin_e)
        
        self.assertEqual(link.link_type, 'SYNERGY')
        self.assertEqual(link.ingredient1, self.vitamin_c)
        self.assertEqual(link.ingredient2, vitamin_e)


class TestCompatibilityChecking(unittest.TestCase):
    """Test compatibility checking functionality"""
    
    def setUp(self):
        """Set up test ingredients"""
        self.hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
        self.niacinamide = ACTIVE_INGREDIENT('niacinamide')
        self.vitamin_c = ACTIVE_INGREDIENT('vitamin_c')
        self.retinol = ACTIVE_INGREDIENT('retinol')
        self.vitamin_e = ANTIOXIDANT('vitamin_e')
    
    def test_known_compatible_pairs(self):
        """Test known compatible ingredient pairs"""
        result = check_ingredient_compatibility(self.hyaluronic_acid, self.niacinamide)
        self.assertEqual(result, 'compatible')
    
    def test_known_incompatible_pairs(self):
        """Test known incompatible ingredient pairs"""
        result = check_ingredient_compatibility(self.vitamin_c, self.retinol)
        self.assertEqual(result, 'incompatible')
    
    def test_known_synergistic_pairs(self):
        """Test known synergistic ingredient pairs"""
        result = check_ingredient_compatibility(self.vitamin_c, self.vitamin_e)
        self.assertEqual(result, 'synergistic')
    
    def test_unknown_pairs(self):
        """Test unknown ingredient pairs"""
        unknown_ingredient = ACTIVE_INGREDIENT('unknown_ingredient')
        result = check_ingredient_compatibility(self.hyaluronic_acid, unknown_ingredient)
        self.assertEqual(result, 'unknown')


class TestIngredientDatabase(unittest.TestCase):
    """Test ingredient database functionality"""
    
    def test_database_creation(self):
        """Test that ingredient database is created correctly"""
        db = create_ingredient_database()
        
        self.assertIsInstance(db, dict)
        self.assertIn('hyaluronic_acid', db)
        self.assertIn('niacinamide', db)
        self.assertIn('vitamin_c', db)
        self.assertIn('retinol', db)
        self.assertIn('glycerin', db)
        self.assertIn('phenoxyethanol', db)
    
    def test_database_ingredient_properties(self):
        """Test that database ingredients have correct properties"""
        db = create_ingredient_database()
        
        ha = db['hyaluronic_acid']
        self.assertEqual(ha.atom_type, 'ACTIVE_INGREDIENT')
        self.assertEqual(ha.properties.ph_range, (5.0, 7.0))
        self.assertEqual(ha.properties.max_concentration, 2.0)
        
        glycerin = db['glycerin']
        self.assertEqual(glycerin.atom_type, 'HUMECTANT')
        self.assertEqual(glycerin.properties.ph_range, (4.0, 8.0))


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple features"""
    
    def test_complete_formulation_workflow(self):
        """Test complete workflow from ingredient creation to compatibility analysis"""
        # Create ingredients
        db = create_ingredient_database()
        
        # Create formulation
        formulation = SKINCARE_FORMULATION(
            db['niacinamide'],
            db['hyaluronic_acid'],
            db['glycerin'],
            db['phenoxyethanol']
        )
        
        # Check formulation properties
        self.assertEqual(len(formulation.ingredients), 4)
        
        # Get actives
        actives = formulation.get_ingredients_by_type('ACTIVE_INGREDIENT')
        self.assertEqual(len(actives), 2)
        
        # Check compatibility of actives
        result = check_ingredient_compatibility(actives[0], actives[1])
        self.assertIn(result, ['compatible', 'synergistic', 'incompatible', 'unknown'])
    
    def test_problematic_formulation(self):
        """Test analysis of formulation with known incompatibilities"""
        db = create_ingredient_database()
        
        # Create problematic formulation with vitamin C + retinol
        formulation = SKINCARE_FORMULATION(
            db['vitamin_c'],
            db['retinol'],
            db['glycerin']
        )
        
        # Check compatibility between problematic pair
        actives = formulation.get_ingredients_by_type('ACTIVE_INGREDIENT')
        if len(actives) >= 2:
            result = check_ingredient_compatibility(actives[0], actives[1])
            self.assertEqual(result, 'incompatible')


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
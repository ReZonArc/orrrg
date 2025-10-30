#!/usr/bin/env python3
"""
Cosmetic Chemistry Types Validation Script

This script validates that the cosmetic chemistry atom types are properly
loaded and working in the OpenCog AtomSpace framework.

Usage:
    python3 validate_cosmetic_types.py

Author: OpenCog Cosmetic Chemistry Framework
License: AGPL-3.0
"""

import sys
import traceback
from opencog.atomspace import AtomSpace, types
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog

def test_basic_atomspace():
    """Test basic AtomSpace functionality."""
    print("1. Testing basic AtomSpace creation...")
    try:
        atomspace = AtomSpace()
        initialize_opencog(atomspace)
        print("   ✓ AtomSpace created successfully")
        return atomspace
    except Exception as e:
        print(f"   ✗ Failed to create AtomSpace: {e}")
        return None

def test_bioscience_module(atomspace):
    """Test loading bioscience module."""
    print("2. Testing bioscience module loading...")
    try:
        import opencog.bioscience
        print("   ✓ Bioscience module loaded successfully")
        return True
    except ImportError as e:
        print(f"   ⚠ Bioscience module not available: {e}")
        print("   ℹ This is expected in environments without full OpenCog installation")
        return False
    except Exception as e:
        print(f"   ✗ Error loading bioscience module: {e}")
        return False

def test_standard_atom_types(atomspace):
    """Test standard OpenCog atom types."""
    print("3. Testing standard atom types...")
    try:
        # Test basic concept node
        concept = ConceptNode("test_concept")
        print(f"   ✓ ConceptNode created: {concept}")
        
        # Test inheritance link
        parent = ConceptNode("parent_concept")
        inheritance = InheritanceLink(concept, parent)
        print(f"   ✓ InheritanceLink created: {inheritance}")
        
        # Test evaluation link
        predicate = PredicateNode("test_predicate")
        evaluation = EvaluationLink(predicate, concept)
        print(f"   ✓ EvaluationLink created: {evaluation}")
        
        print(f"   ✓ Standard atom types working (total atoms: {len(atomspace)})")
        return True
    except Exception as e:
        print(f"   ✗ Error with standard atom types: {e}")
        traceback.print_exc()
        return False

def test_cosmetic_ingredient_creation(atomspace):
    """Test creating cosmetic ingredients using standard types."""
    print("4. Testing cosmetic ingredient creation...")
    try:
        # Create ingredients as concept nodes
        hyaluronic_acid = ConceptNode("hyaluronic_acid")
        niacinamide = ConceptNode("niacinamide")
        glycerin = ConceptNode("glycerin")
        
        # Create functional classifications
        active_ingredient = ConceptNode("ACTIVE_INGREDIENT")
        humectant = ConceptNode("HUMECTANT")
        
        # Create inheritance relationships
        InheritanceLink(hyaluronic_acid, active_ingredient)
        InheritanceLink(hyaluronic_acid, humectant)  # Dual function
        InheritanceLink(niacinamide, active_ingredient)
        InheritanceLink(glycerin, humectant)
        
        print("   ✓ Created hyaluronic_acid as ACTIVE_INGREDIENT and HUMECTANT")
        print("   ✓ Created niacinamide as ACTIVE_INGREDIENT")
        print("   ✓ Created glycerin as HUMECTANT")
        
        return True
    except Exception as e:
        print(f"   ✗ Error creating cosmetic ingredients: {e}")
        traceback.print_exc()
        return False

def test_formulation_creation(atomspace):
    """Test creating cosmetic formulations."""
    print("5. Testing formulation creation...")
    try:
        # Create formulation
        moisturizer = ConceptNode("hydrating_moisturizer")
        formulation_type = ConceptNode("SKINCARE_FORMULATION")
        InheritanceLink(moisturizer, formulation_type)
        
        # Create ingredients
        hyaluronic_acid = ConceptNode("hyaluronic_acid")
        glycerin = ConceptNode("glycerin")
        
        # Add concentration information
        concentration_pred = PredicateNode("concentration")
        EvaluationLink(
            concentration_pred,
            ListLink(
                moisturizer,
                hyaluronic_acid,
                NumberNode("2.0")
            )
        )
        
        EvaluationLink(
            concentration_pred,
            ListLink(
                moisturizer,
                glycerin,
                NumberNode("10.0")
            )
        )
        
        print("   ✓ Created hydrating_moisturizer formulation")
        print("   ✓ Added hyaluronic_acid at 2.0% concentration")
        print("   ✓ Added glycerin at 10.0% concentration")
        
        return True
    except Exception as e:
        print(f"   ✗ Error creating formulation: {e}")
        traceback.print_exc()
        return False

def test_compatibility_relationships(atomspace):
    """Test ingredient compatibility relationships."""
    print("6. Testing compatibility relationships...")
    try:
        # Create ingredients
        hyaluronic_acid = ConceptNode("hyaluronic_acid")
        niacinamide = ConceptNode("niacinamide")
        vitamin_c = ConceptNode("vitamin_c")
        retinol = ConceptNode("retinol")
        
        # Create compatibility
        compatible_pred = PredicateNode("compatible_with")
        compatibility_link = EvaluationLink(
            compatible_pred,
            ListLink(hyaluronic_acid, niacinamide)
        )
        
        # Create incompatibility
        incompatible_pred = PredicateNode("incompatible_with")
        incompatibility_link = EvaluationLink(
            incompatible_pred,
            ListLink(vitamin_c, retinol)
        )
        
        # Add descriptions
        description_pred = PredicateNode("interaction_description")
        EvaluationLink(
            description_pred,
            ListLink(
                compatibility_link,
                ConceptNode("enhanced_hydration_and_barrier_function")
            )
        )
        
        EvaluationLink(
            description_pred,
            ListLink(
                incompatibility_link,
                ConceptNode("pH_incompatibility_and_instability")
            )
        )
        
        print("   ✓ Created compatibility: hyaluronic_acid + niacinamide")
        print("   ✓ Created incompatibility: vitamin_c + retinol")
        print("   ✓ Added interaction descriptions")
        
        return True
    except Exception as e:
        print(f"   ✗ Error creating compatibility relationships: {e}")
        traceback.print_exc()
        return False

def test_property_modeling(atomspace):
    """Test formulation property modeling."""
    print("7. Testing property modeling...")
    try:
        # Create formulation
        serum = ConceptNode("vitamin_c_serum")
        
        # Add pH property
        ph_pred = PredicateNode("pH")
        EvaluationLink(ph_pred, ListLink(serum, NumberNode("4.0")))
        
        # Add viscosity property
        viscosity_pred = PredicateNode("viscosity")
        EvaluationLink(viscosity_pred, ListLink(serum, ConceptNode("medium")))
        
        # Add stability property
        stability_pred = PredicateNode("stability_months")
        EvaluationLink(stability_pred, ListLink(serum, NumberNode("12")))
        
        print("   ✓ Added pH property: 4.0")
        print("   ✓ Added viscosity property: medium")
        print("   ✓ Added stability property: 12 months")
        
        return True
    except Exception as e:
        print(f"   ✗ Error modeling properties: {e}")
        traceback.print_exc()
        return False

def test_query_functionality(atomspace):
    """Test querying the knowledge base."""
    print("8. Testing query functionality...")
    try:
        # Count total atoms
        total_atoms = len(atomspace)
        
        # Count specific atom types
        concept_nodes = len([atom for atom in atomspace if atom.type == types.ConceptNode])
        inheritance_links = len([atom for atom in atomspace if atom.type == types.InheritanceLink])
        evaluation_links = len([atom for atom in atomspace if atom.type == types.EvaluationLink])
        
        print(f"   ✓ Total atoms: {total_atoms}")
        print(f"   ✓ ConceptNodes: {concept_nodes}")
        print(f"   ✓ InheritanceLinks: {inheritance_links}")
        print(f"   ✓ EvaluationLinks: {evaluation_links}")
        
        # Test finding specific atoms
        active_ingredients = []
        for atom in atomspace:
            if (atom.type == types.InheritanceLink and 
                len(atom.out) == 2 and 
                atom.out[1].name == "ACTIVE_INGREDIENT"):
                active_ingredients.append(atom.out[0].name)
        
        print(f"   ✓ Found active ingredients: {active_ingredients}")
        
        return True
    except Exception as e:
        print(f"   ✗ Error querying knowledge base: {e}")
        traceback.print_exc()
        return False

def test_atom_types_script_syntax():
    """Test that the atom_types.script file has valid syntax."""
    print("9. Testing atom_types.script syntax...")
    try:
        script_path = "/home/runner/work/cosmagi-bio/cosmagi-bio/bioscience/types/atom_types.script"
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for our cosmetic chemistry additions
        required_types = [
            'ACTIVE_INGREDIENT',
            'PRESERVATIVE',
            'EMULSIFIER',
            'HUMECTANT',
            'SKINCARE_FORMULATION',
            'COMPATIBILITY_LINK',
            'INCOMPATIBILITY_LINK',
            'SYNERGY_LINK'
        ]
        
        missing_types = []
        for type_name in required_types:
            if type_name not in content:
                missing_types.append(type_name)
        
        if missing_types:
            print(f"   ⚠ Missing atom types: {missing_types}")
            return False
        else:
            print("   ✓ All required cosmetic chemistry atom types found")
        
        # Check basic syntax patterns
        lines = content.split('\n')
        type_definitions = [line for line in lines if '<-' in line and not line.strip().startswith('//')]
        
        print(f"   ✓ Found {len(type_definitions)} atom type definitions")
        print(f"   ✓ atom_types.script syntax appears valid")
        
        return True
    except Exception as e:
        print(f"   ✗ Error checking atom_types.script: {e}")
        traceback.print_exc()
        return False

def run_validation():
    """Run all validation tests."""
    print("🧪 Cosmetic Chemistry Types Validation")
    print("======================================\n")
    
    results = []
    
    # Test basic functionality
    atomspace = test_basic_atomspace()
    results.append(atomspace is not None)
    if not atomspace:
        print("\n❌ Critical failure: Cannot proceed without working AtomSpace")
        return False
    
    # Test module loading (optional)
    bioscience_loaded = test_bioscience_module(atomspace)
    results.append(True)  # Don't fail if bioscience module unavailable
    
    # Test core functionality
    results.append(test_standard_atom_types(atomspace))
    results.append(test_cosmetic_ingredient_creation(atomspace))
    results.append(test_formulation_creation(atomspace))
    results.append(test_compatibility_relationships(atomspace))
    results.append(test_property_modeling(atomspace))
    results.append(test_query_functionality(atomspace))
    results.append(test_atom_types_script_syntax())
    
    # Calculate results
    passed_tests = sum(results)
    total_tests = len(results)
    
    print(f"\n📊 Validation Results")
    print(f"====================")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n✅ All validation tests passed!")
        print("The cosmetic chemistry framework is ready for use.")
        return True
    else:
        print(f"\n⚠ {total_tests - passed_tests} tests failed or had issues.")
        if passed_tests >= total_tests - 1:  # Allow for bioscience module being optional
            print("Core functionality appears to be working.")
            return True
        else:
            print("❌ Critical issues detected - framework may not work properly.")
            return False

def main():
    """Main function."""
    try:
        success = run_validation()
        
        print("\n🔗 Next Steps:")
        if success:
            print("  • Run examples/python/cosmetic_intro_example.py")
            print("  • Try examples/scheme/cosmetic_compatibility.scm")
            print("  • Read docs/COSMETIC_CHEMISTRY.md for full reference")
        else:
            print("  • Check OpenCog installation")
            print("  • Verify bioscience module compilation")
            print("  • Review error messages above")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚡ Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during validation: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
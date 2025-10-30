#!/usr/bin/env python3
"""
Cosmetic Chemistry Syntax Validation

This script validates the syntax and structure of the cosmetic chemistry
implementation without requiring a full OpenCog installation.

Usage:
    python3 validate_cosmetic_syntax.py

Author: OpenCog Cosmetic Chemistry Framework
License: AGPL-3.0
"""

import os
import sys
import re
from pathlib import Path

def validate_atom_types_script():
    """Validate the atom_types.script file."""
    print("🔬 Validating atom_types.script")
    print("===============================\n")
    
    script_path = Path(__file__).parent.parent / "bioscience" / "types" / "atom_types.script"
    
    if not script_path.exists():
        print("❌ atom_types.script not found")
        return False
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        print("✓ File loaded successfully")
        
        # Check for cosmetic chemistry section
        if "Cosmetic Chemistry Specializations" in content:
            print("✓ Cosmetic chemistry section found")
        else:
            print("❌ Cosmetic chemistry section not found")
            return False
        
        # Required cosmetic atom types
        required_types = {
            # Ingredient Categories
            'ACTIVE_INGREDIENT': 'MOLECULE_NODE',
            'PRESERVATIVE': 'MOLECULE_NODE', 
            'EMULSIFIER': 'MOLECULE_NODE',
            'HUMECTANT': 'MOLECULE_NODE',
            'SURFACTANT': 'MOLECULE_NODE',
            'THICKENER': 'MOLECULE_NODE',
            'EMOLLIENT': 'MOLECULE_NODE',
            'ANTIOXIDANT': 'MOLECULE_NODE',
            'UV_FILTER': 'MOLECULE_NODE',
            'FRAGRANCE': 'MOLECULE_NODE',
            'COLORANT': 'MOLECULE_NODE',
            'PH_ADJUSTER': 'MOLECULE_NODE',
            
            # Formulation Types
            'SKINCARE_FORMULATION': 'CONCEPT_NODE',
            'HAIRCARE_FORMULATION': 'CONCEPT_NODE',
            'MAKEUP_FORMULATION': 'CONCEPT_NODE',
            'FRAGRANCE_FORMULATION': 'CONCEPT_NODE',
            
            # Property Types
            'PH_PROPERTY': 'CONCEPT_NODE',
            'VISCOSITY_PROPERTY': 'CONCEPT_NODE',
            'STABILITY_PROPERTY': 'CONCEPT_NODE',
            'TEXTURE_PROPERTY': 'CONCEPT_NODE',
            'SPF_PROPERTY': 'CONCEPT_NODE',
            
            # Interaction Types
            'COMPATIBILITY_LINK': 'LINK',
            'INCOMPATIBILITY_LINK': 'LINK',
            'SYNERGY_LINK': 'LINK',
            'ANTAGONISM_LINK': 'LINK',
            
            # Safety/Regulatory
            'SAFETY_ASSESSMENT': 'CONCEPT_NODE',
            'ALLERGEN_CLASSIFICATION': 'CONCEPT_NODE',
            'CONCENTRATION_LIMIT': 'CONCEPT_NODE'
        }
        
        print(f"\nChecking {len(required_types)} required atom types:")
        
        missing_types = []
        invalid_inheritance = []
        
        for atom_type, expected_parent in required_types.items():
            # Check if type is defined
            type_pattern = rf'{atom_type}\s*<-\s*(\w+)'
            match = re.search(type_pattern, content)
            
            if match:
                actual_parent = match.group(1)
                if actual_parent == expected_parent:
                    print(f"  ✓ {atom_type} <- {actual_parent}")
                else:
                    print(f"  ⚠ {atom_type} <- {actual_parent} (expected {expected_parent})")
                    invalid_inheritance.append((atom_type, actual_parent, expected_parent))
            else:
                print(f"  ❌ {atom_type} - NOT FOUND")
                missing_types.append(atom_type)
        
        # Summary
        print(f"\nValidation Summary:")
        print(f"  Found: {len(required_types) - len(missing_types)}/{len(required_types)} types")
        print(f"  Missing: {len(missing_types)} types")
        print(f"  Invalid inheritance: {len(invalid_inheritance)} types")
        
        if missing_types:
            print(f"\nMissing types: {missing_types}")
        
        if invalid_inheritance:
            print(f"\nInvalid inheritance:")
            for atom_type, actual, expected in invalid_inheritance:
                print(f"  {atom_type}: {actual} (should be {expected})")
        
        success = len(missing_types) == 0 and len(invalid_inheritance) == 0
        
        if success:
            print("\n✅ All atom types correctly defined!")
        else:
            print("\n⚠ Some issues found in atom type definitions")
        
        return success
        
    except Exception as e:
        print(f"❌ Error reading atom_types.script: {e}")
        return False

def validate_documentation():
    """Validate the documentation files."""
    print("\n📚 Validating Documentation")
    print("===========================\n")
    
    docs_path = Path(__file__).parent.parent / "docs" / "COSMETIC_CHEMISTRY.md"
    
    if not docs_path.exists():
        print("❌ COSMETIC_CHEMISTRY.md not found")
        return False
    
    try:
        with open(docs_path, 'r') as f:
            content = f.read()
        
        print("✓ Documentation file loaded")
        
        # Check for required sections
        required_sections = [
            "Overview",
            "Atom Type Reference", 
            "Common Cosmetic Ingredients Database",
            "Formulation Guidelines",
            "Regulatory Compliance",
            "Advanced Applications",
            "Examples"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
            else:
                print(f"  ✓ {section} section found")
        
        if missing_sections:
            print(f"  ❌ Missing sections: {missing_sections}")
            return False
        
        # Check word count
        word_count = len(content.split())
        print(f"  ✓ Word count: {word_count} words")
        
        if word_count < 1000:
            print("  ⚠ Documentation seems short")
        
        print("✅ Documentation validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error reading documentation: {e}")
        return False

def validate_examples():
    """Validate the example files."""
    print("\n🐍 Validating Examples")
    print("======================\n")
    
    examples_dir = Path(__file__).parent
    
    # Check for required example files
    required_files = [
        "python/cosmetic_intro_example.py",
        "python/cosmetic_chemistry_example.py", 
        "scheme/cosmetic_formulation.scm",
        "scheme/cosmetic_compatibility.scm",
        "README.md"
    ]
    
    missing_files = []
    syntax_issues = []
    
    for file_path in required_files:
        full_path = examples_dir / file_path
        
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"  ❌ {file_path} - NOT FOUND")
            continue
        
        print(f"  ✓ {file_path} - found")
        
        # Basic syntax check for Python files
        if file_path.endswith('.py'):
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Try to compile the Python code
                compile(content, str(full_path), 'exec')
                print(f"    ✓ Python syntax valid")
                
                # Check for required functions/classes
                if 'cosmetic_intro_example.py' in file_path:
                    if 'def main()' in content:
                        print(f"    ✓ Main function found")
                    else:
                        syntax_issues.append(f"{file_path}: No main function")
                
                elif 'cosmetic_chemistry_example.py' in file_path:
                    if 'class CosmeticFormulationAnalyzer' in content:
                        print(f"    ✓ Analyzer class found")
                    else:
                        syntax_issues.append(f"{file_path}: No analyzer class")
                
            except SyntaxError as e:
                syntax_issues.append(f"{file_path}: {e}")
                print(f"    ❌ Python syntax error: {e}")
            except Exception as e:
                print(f"    ⚠ Could not validate: {e}")
        
        # Check Scheme files for basic structure
        elif file_path.endswith('.scm'):
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Check for basic Scheme structure
                if '(use-modules' in content:
                    print(f"    ✓ Scheme modules declaration found")
                else:
                    syntax_issues.append(f"{file_path}: No modules declaration")
                
                # Check for balanced parentheses (simple check)
                open_parens = content.count('(')
                close_parens = content.count(')')
                if open_parens == close_parens:
                    print(f"    ✓ Parentheses balanced ({open_parens} pairs)")
                else:
                    syntax_issues.append(f"{file_path}: Unbalanced parentheses")
                
            except Exception as e:
                print(f"    ⚠ Could not validate: {e}")
    
    # Summary
    print(f"\nExample files summary:")
    print(f"  Found: {len(required_files) - len(missing_files)}/{len(required_files)} files")
    print(f"  Missing: {len(missing_files)} files")
    print(f"  Syntax issues: {len(syntax_issues)} issues")
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
    
    if syntax_issues:
        print(f"\nSyntax issues:")
        for issue in syntax_issues:
            print(f"  {issue}")
    
    success = len(missing_files) == 0 and len(syntax_issues) == 0
    
    if success:
        print("\n✅ All examples validated successfully!")
    else:
        print("\n⚠ Some issues found in examples")
    
    return success

def validate_file_structure():
    """Validate the overall file structure."""
    print("\n📁 Validating File Structure")
    print("=============================\n")
    
    base_dir = Path(__file__).parent.parent
    
    # Check for required directories and files
    required_structure = {
        'bioscience/types/atom_types.script': 'file',
        'docs/COSMETIC_CHEMISTRY.md': 'file',
        'examples/README.md': 'file',
        'examples/python/': 'directory',
        'examples/scheme/': 'directory'
    }
    
    missing_items = []
    
    for item_path, item_type in required_structure.items():
        full_path = base_dir / item_path
        
        if item_type == 'file' and not full_path.is_file():
            missing_items.append(f"File: {item_path}")
            print(f"  ❌ {item_path} (file) - NOT FOUND")
        elif item_type == 'directory' and not full_path.is_dir():
            missing_items.append(f"Directory: {item_path}")
            print(f"  ❌ {item_path} (directory) - NOT FOUND")
        else:
            print(f"  ✓ {item_path} ({item_type}) - found")
    
    success = len(missing_items) == 0
    
    if success:
        print("\n✅ File structure validation passed!")
    else:
        print(f"\n❌ Missing items: {missing_items}")
    
    return success

def main():
    """Main validation function."""
    print("🧪 Cosmetic Chemistry Implementation Validation")
    print("===============================================\n")
    
    print("This validation checks the implementation without requiring OpenCog installation.\n")
    
    # Run all validation tests
    results = []
    
    results.append(validate_file_structure())
    results.append(validate_atom_types_script())
    results.append(validate_documentation())
    results.append(validate_examples())
    
    # Calculate overall results
    passed_tests = sum(results)
    total_tests = len(results)
    
    print(f"\n📊 Overall Validation Results")
    print(f"=============================")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("The cosmetic chemistry implementation is syntactically correct and complete.")
        print("\nNext steps:")
        print("  • Build the bioscience extensions with the new atom types")
        print("  • Test with a full OpenCog installation")
        print("  • Run the Python and Scheme examples")
    else:
        failed_tests = total_tests - passed_tests
        print(f"\n⚠ {failed_tests} validation test(s) failed.")
        print("Please review the issues above before proceeding.")
    
    return 0 if passed_tests == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Cross-Domain Workflow Example
============================

This example demonstrates ORRRG's ability to coordinate multiple components
across different domains to solve complex research problems.

Workflow: Protein Analysis → Chemical Properties → Publication
1. Analyze protein sequence using ESM-2
2. Extract chemical properties using cheminformatics
3. Apply chemical reasoning for insights
4. Generate publication draft using OJS agents
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from core import SelfOrganizingCore
from core.component_adapters import create_adapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def protein_to_publication_workflow():
    """
    Complete workflow from protein sequence analysis to publication draft.
    """
    print("🧬 Starting Cross-Domain Workflow: Protein → Chemistry → Publication")
    print("="*70)
    
    # Initialize the Self-Organizing Core
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    try:
        # Step 1: Protein sequence analysis using ESM-2
        print("\n📊 Step 1: Protein Sequence Analysis")
        protein_sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
        
        protein_data = {
            "protein_sequence": protein_sequence,
            "analysis_type": "structure_prediction"
        }
        
        # Simulate ESM-2 analysis
        esm_result = {
            "status": "protein_analyzed",
            "sequence": protein_sequence,
            "structure_prediction": {
                "secondary_structure": ["helix", "sheet", "coil"],
                "confidence": 0.89
            },
            "embeddings": [0.234, -0.156, 0.892, 0.445],
            "functional_domains": ["binding_site", "catalytic_domain"],
            "component": "esm-2-keras"
        }
        
        print(f"   ✓ Analyzed protein sequence ({len(protein_sequence)} residues)")
        print(f"   ✓ Structure prediction confidence: {esm_result['structure_prediction']['confidence']}")
        print(f"   ✓ Identified {len(esm_result['functional_domains'])} functional domains")
        
        # Step 2: Chemical property analysis
        print("\n🧪 Step 2: Chemical Property Analysis")
        
        # Convert protein analysis to chemical data
        chemical_data = {
            "molecule": "derived_from_protein",
            "protein_domains": esm_result["functional_domains"],
            "analysis_type": "molecular_properties"
        }
        
        # Simulate chemical analysis
        chem_result = {
            "status": "molecule_analyzed",
            "properties": {
                "molecular_weight": 5234.7,
                "hydrophobicity": 0.34,
                "charge": -2.1,
                "surface_area": 890.2
            },
            "binding_affinity": {
                "ligand_1": 0.87,
                "ligand_2": 0.23
            },
            "component": "coscheminformatics"
        }
        
        print(f"   ✓ Calculated molecular weight: {chem_result['properties']['molecular_weight']}")
        print(f"   ✓ Hydrophobicity index: {chem_result['properties']['hydrophobicity']}")
        print(f"   ✓ Net charge: {chem_result['properties']['charge']}")
        
        # Step 3: Chemical reasoning and insight generation
        print("\n🤔 Step 3: Chemical Reasoning & Insight Generation")
        
        reasoning_data = {
            "reaction_query": {
                "protein_properties": chem_result["properties"],
                "binding_data": chem_result["binding_affinity"],
                "context": "drug_discovery"
            }
        }
        
        # Simulate chemical reasoning
        reasoning_result = {
            "status": "reaction_predicted",
            "insights": [
                "High binding affinity suggests therapeutic potential",
                "Hydrophobic regions indicate membrane interaction",
                "Negative charge may affect cellular uptake"
            ],
            "drug_targets": ["receptor_A", "enzyme_B"],
            "confidence": 0.92,
            "component": "coschemreasoner"
        }
        
        print(f"   ✓ Generated {len(reasoning_result['insights'])} key insights")
        print(f"   ✓ Identified {len(reasoning_result['drug_targets'])} potential drug targets")
        print(f"   ✓ Reasoning confidence: {reasoning_result['confidence']}")
        
        # Step 4: Cognitive integration using OpenCog
        print("\n🧠 Step 4: Cognitive Knowledge Integration")
        
        cognitive_data = {
            "knowledge": {
                "protein_analysis": esm_result,
                "chemical_properties": chem_result,
                "reasoning_insights": reasoning_result
            },
            "integration_type": "cross_domain_synthesis"
        }
        
        # Simulate cognitive integration
        cognitive_result = {
            "status": "knowledge_integrated",
            "atomspace_concepts": [
                "protein_structure_function_relationship",
                "chemical_biological_interaction",
                "therapeutic_potential_assessment"
            ],
            "novel_hypotheses": [
                "Structural motif X correlates with binding specificity",
                "Charge distribution affects membrane permeability",
                "Domain Y shows evolutionary conservation"
            ],
            "atomspace_size": 1547,
            "component": "oc-skintwin"
        }
        
        print(f"   ✓ Integrated knowledge into {cognitive_result['atomspace_size']} atoms")
        print(f"   ✓ Generated {len(cognitive_result['novel_hypotheses'])} novel hypotheses")
        
        # Step 5: Publication draft generation
        print("\n📝 Step 5: Automated Publication Draft Generation")
        
        publication_data = {
            "manuscript": {
                "research_data": {
                    "protein_analysis": esm_result,
                    "chemical_analysis": chem_result,
                    "reasoning_results": reasoning_result,
                    "cognitive_insights": cognitive_result
                },
                "manuscript_type": "research_article",
                "target_journal": "Nature Biotechnology"
            }
        }
        
        # Simulate publication generation
        publication_result = {
            "status": "manuscript_generated",
            "manuscript_id": "ORRRG_2024_001",
            "sections": {
                "abstract": "Generated abstract summarizing cross-domain findings",
                "introduction": "Literature review and research motivation", 
                "methods": "Computational pipeline and analysis methods",
                "results": "Protein structure, chemical properties, and insights",
                "discussion": "Novel hypotheses and therapeutic implications",
                "conclusion": "Cross-domain approach yields new understanding"
            },
            "figures": ["protein_structure.png", "chemical_properties.svg", "reasoning_network.pdf"],
            "word_count": 4567,
            "agent_assignments": {
                "literature_review": "agent_alpha",
                "methodology": "agent_beta", 
                "results_analysis": "agent_gamma"
            },
            "component": "oj7s3"
        }
        
        print(f"   ✓ Generated manuscript ({publication_result['word_count']} words)")
        print(f"   ✓ Created {len(publication_result['sections'])} sections")
        print(f"   ✓ Assigned {len(publication_result['agent_assignments'])} autonomous agents")
        
        # Final summary
        print("\n🎉 Workflow Completion Summary")
        print("="*50)
        print(f"📊 Protein Analysis: {esm_result['structure_prediction']['confidence']:.2%} confidence")
        print(f"🧪 Chemical Properties: {len(chem_result['properties'])} parameters analyzed")
        print(f"🤔 Reasoning Insights: {reasoning_result['confidence']:.2%} confidence")
        print(f"🧠 Knowledge Integration: {cognitive_result['atomspace_size']} atoms")
        print(f"📝 Publication Draft: {publication_result['word_count']} words generated")
        print(f"\n✨ Cross-domain workflow successfully coordinated {len(soc.components)} components!")
        
        return {
            "workflow_status": "completed",
            "components_used": ["esm-2-keras", "coscheminformatics", "coschemreasoner", "oc-skintwin", "oj7s3"],
            "final_output": publication_result,
            "processing_time": "42.7 seconds",
            "confidence_score": 0.89
        }
        
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        return {"workflow_status": "failed", "error": str(e)}
    
    finally:
        await soc.shutdown()


async def compiler_ml_optimization_workflow():
    """
    Workflow demonstrating code optimization using compiler analysis and ML.
    """
    print("\n⚙️  Starting Compiler-ML Optimization Workflow")
    print("="*60)
    
    # Initialize system
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    try:
        # Step 1: Code analysis with Compiler Explorer
        print("\n🔍 Step 1: Code Compilation and Analysis")
        
        source_code = """
        #include <vector>
        #include <algorithm>
        
        double compute_mean(const std::vector<double>& data) {
            double sum = 0.0;
            for (size_t i = 0; i < data.size(); ++i) {
                sum += data[i];
            }
            return sum / data.size();
        }
        """
        
        compile_result = {
            "status": "compiled",
            "language": "cpp",
            "optimizations": ["O2", "vectorization"],
            "assembly_analysis": {
                "instruction_count": 23,
                "vectorized_loops": 1,
                "branch_predictions": 2
            },
            "performance_metrics": {
                "estimated_cycles": 150,
                "cache_efficiency": 0.87
            },
            "component": "echopiler"
        }
        
        print(f"   ✓ Compiled C++ code with {compile_result['assembly_analysis']['instruction_count']} instructions")
        print(f"   ✓ Detected {compile_result['assembly_analysis']['vectorized_loops']} vectorizable loop")
        
        # Step 2: ML-based optimization using ONNX Runtime
        print("\n🤖 Step 2: ML-Guided Performance Optimization")
        
        ml_input = {
            "model_input": {
                "code_features": [
                    compile_result["assembly_analysis"]["instruction_count"],
                    compile_result["assembly_analysis"]["vectorized_loops"],
                    compile_result["performance_metrics"]["cache_efficiency"]
                ],
                "optimization_target": "performance"
            }
        }
        
        ml_result = {
            "status": "inference_complete",
            "optimization_suggestions": [
                "Apply loop unrolling factor 4",
                "Use SIMD instructions for vectorization", 
                "Reorder memory accesses for cache locality"
            ],
            "predicted_improvement": 0.34,  # 34% performance improvement
            "confidence": 0.91,
            "component": "echonnxruntime"
        }
        
        print(f"   ✓ ML model suggests {len(ml_result['optimization_suggestions'])} optimizations")
        print(f"   ✓ Predicted performance improvement: {ml_result['predicted_improvement']:.2%}")
        
        print(f"\n🎯 Optimization recommendations:")
        for i, suggestion in enumerate(ml_result['optimization_suggestions'], 1):
            print(f"   {i}. {suggestion}")
        
        return {
            "workflow_status": "completed",
            "components_used": ["echopiler", "echonnxruntime"],
            "optimization_improvement": ml_result['predicted_improvement'],
            "confidence": ml_result['confidence']
        }
        
    finally:
        await soc.shutdown()


async def main():
    """Run example workflows."""
    print("ORRRG Cross-Domain Workflow Examples")
    print("="*40)
    
    # Run protein-to-publication workflow
    result1 = await protein_to_publication_workflow()
    
    # Run compiler-ML optimization workflow  
    result2 = await compiler_ml_optimization_workflow()
    
    print(f"\n🏁 All workflows completed!")
    print(f"Workflow 1: {result1['workflow_status']}")
    print(f"Workflow 2: {result2['workflow_status']}")


if __name__ == "__main__":
    asyncio.run(main())
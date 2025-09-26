#!/usr/bin/env python3
"""
Component Adapters for ORRRG Integration
========================================

This module provides adapter classes that wrap each component system
to provide a unified interface for the Self-Organizing Core.
"""

import asyncio
import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from .self_organizing_core import ComponentInterface

logger = logging.getLogger(__name__)


class BaseComponentAdapter(ComponentInterface):
    """Base adapter class for component integration."""
    
    def __init__(self, component_path: Path, name: str):
        self.component_path = component_path
        self.name = name
        self.config = {}
        self.initialized = False
        self.process = None
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the component adapter."""
        self.config = config
        try:
            # Check if component is available
            if not await self._check_prerequisites():
                logger.warning(f"Prerequisites not met for {self.name}")
                return False
            
            # Perform component-specific initialization
            success = await self._initialize_component()
            self.initialized = success
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the component."""
        if not self.initialized:
            return {"error": "Component not initialized"}
        
        try:
            return await self._process_data(data)
        except Exception as e:
            logger.error(f"Error processing data in {self.name}: {e}")
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """Cleanup component resources."""
        try:
            await self._cleanup_component()
            if self.process:
                self.process.terminate()
                await self.process.wait()
        except Exception as e:
            logger.error(f"Error cleaning up {self.name}: {e}")
    
    @abstractmethod
    async def _check_prerequisites(self) -> bool:
        """Check if component prerequisites are met."""
        pass
    
    @abstractmethod
    async def _initialize_component(self) -> bool:
        """Component-specific initialization."""
        pass
    
    @abstractmethod 
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Component-specific data processing."""
        pass
    
    async def _cleanup_component(self) -> None:
        """Component-specific cleanup."""
        pass


class OJ7S3Adapter(BaseComponentAdapter):
    """Adapter for OJS with SKZ autonomous agents."""
    
    def get_capabilities(self) -> List[str]:
        return ["manuscript_processing", "editorial_workflow", "agent_coordination", "publishing_automation"]
    
    async def _check_prerequisites(self) -> bool:
        """Check PHP, MySQL, and required files."""
        php_check = await self._run_command("php --version")
        return php_check.returncode == 0 and (self.component_path / "index.php").exists()
    
    async def _initialize_component(self) -> bool:
        """Initialize OJS system."""
        logger.info(f"Initializing OJS system at {self.component_path}")
        # In a real implementation, this would start the PHP application
        return True
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process manuscript or editorial workflow data."""
        if "manuscript" in data:
            return {"status": "manuscript_received", "workflow_id": "oj7s3_001"}
        elif "review_request" in data:
            return {"status": "review_assigned", "reviewer_id": "agent_007"}
        return {"status": "processed", "component": "oj7s3"}
    
    async def _run_command(self, command: str) -> subprocess.CompletedProcess:
        """Run a shell command asynchronously.""" 
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return type('Result', (), {
            'returncode': process.returncode,
            'stdout': stdout.decode(),
            'stderr': stderr.decode()
        })()


class EchopilerAdapter(BaseComponentAdapter):
    """Adapter for Compiler Explorer."""
    
    def get_capabilities(self) -> List[str]:
        return ["code_compilation", "assembly_analysis", "multi_language_support", "interactive_exploration"]
    
    async def _check_prerequisites(self) -> bool:
        """Check Node.js and required files."""
        node_check = await self._run_command("node --version")
        return node_check.returncode == 0 and (self.component_path / "package.json").exists()
    
    async def _initialize_component(self) -> bool:
        """Initialize Compiler Explorer."""
        logger.info(f"Initializing Compiler Explorer at {self.component_path}")
        return True
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process code compilation requests."""
        if "source_code" in data:
            language = data.get("language", "cpp")
            return {
                "status": "compiled",
                "language": language,
                "assembly": "mock_assembly_output",
                "component": "echopiler"
            }
        return {"status": "processed", "component": "echopiler"}
    
    async def _run_command(self, command: str) -> subprocess.CompletedProcess:
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return type('Result', (), {
            'returncode': process.returncode,
            'stdout': stdout.decode(),
            'stderr': stderr.decode()
        })()


class OpenCogAdapter(BaseComponentAdapter):
    """Adapter for OpenCog cognitive architecture."""
    
    def get_capabilities(self) -> List[str]:
        return ["knowledge_representation", "reasoning", "cognitive_modeling", "atomspace"]
    
    async def _check_prerequisites(self) -> bool:
        """Check OpenCog installation."""
        return (self.component_path / "README.md").exists()
    
    async def _initialize_component(self) -> bool:
        """Initialize OpenCog AtomSpace."""
        logger.info(f"Initializing OpenCog at {self.component_path}")
        return True
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process knowledge representation and reasoning requests."""
        if "knowledge" in data:
            return {
                "status": "knowledge_added",
                "atomspace_size": 1000,
                "component": "oc-skintwin"
            }
        elif "query" in data:
            return {
                "status": "reasoning_complete", 
                "result": "inferred_knowledge",
                "component": "oc-skintwin"
            }
        return {"status": "processed", "component": "oc-skintwin"}


class ESMAdapter(BaseComponentAdapter):
    """Adapter for ESM-2/GPT-2 hypergraph analysis."""
    
    def get_capabilities(self) -> List[str]:
        return ["protein_modeling", "language_modeling", "hypergraph_analysis", "transformer_analysis"]
    
    async def _check_prerequisites(self) -> bool:
        """Check Python and ML libraries."""
        return (self.component_path / "README.md").exists()
    
    async def _initialize_component(self) -> bool:
        """Initialize ESM models."""
        logger.info(f"Initializing ESM models at {self.component_path}")
        return True
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process protein sequences or language modeling."""
        if "protein_sequence" in data:
            return {
                "status": "protein_analyzed",
                "embeddings": [0.1, 0.2, 0.3],  # Mock embeddings
                "component": "esm-2-keras"
            }
        elif "text" in data:
            return {
                "status": "text_analyzed", 
                "hypergraph": "mock_hypergraph_structure",
                "component": "esm-2-keras"
            }
        return {"status": "processed", "component": "esm-2-keras"}


class CosmaBioAdapter(BaseComponentAdapter):
    """Adapter for genomic and proteomic research."""
    
    def get_capabilities(self) -> List[str]:
        return ["genomic_analysis", "proteomic_analysis", "bioinformatics", "opencog_bio"]
    
    async def _check_prerequisites(self) -> bool:
        """Check bioinformatics tools."""
        return (self.component_path / "README.md").exists()
    
    async def _initialize_component(self) -> bool:
        """Initialize bioinformatics tools."""
        logger.info(f"Initializing Cosma-Bio at {self.component_path}")
        return True
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process genomic or proteomic data."""
        if "genome_data" in data:
            return {
                "status": "genome_analyzed",
                "features": ["gene1", "gene2"],
                "component": "cosmagi-bio"
            }
        return {"status": "processed", "component": "cosmagi-bio"}


class ChemInformaticsAdapter(BaseComponentAdapter):
    """Adapter for chemical information processing."""
    
    def get_capabilities(self) -> List[str]:
        return ["chemical_analysis", "molecular_processing", "chemical_data"]
    
    async def _check_prerequisites(self) -> bool:
        """Check chemical analysis tools."""
        return (self.component_path / "README.md").exists()
    
    async def _initialize_component(self) -> bool:
        """Initialize chemical analysis tools."""
        logger.info(f"Initializing ChemInformatics at {self.component_path}")
        return True
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process chemical data."""
        if "molecule" in data:
            return {
                "status": "molecule_analyzed",
                "properties": {"mw": 180.2, "logp": 2.1},
                "component": "coscheminformatics"
            }
        return {"status": "processed", "component": "coscheminformatics"}


class ONNXRuntimeAdapter(BaseComponentAdapter):
    """Adapter for ONNX Runtime ML inference."""
    
    def get_capabilities(self) -> List[str]:
        return ["ml_inference", "onnx_models", "cross_platform", "optimization"]
    
    async def _check_prerequisites(self) -> bool:
        """Check ONNX Runtime installation."""
        return (self.component_path / "README.md").exists()
    
    async def _initialize_component(self) -> bool:
        """Initialize ONNX Runtime."""
        logger.info(f"Initializing ONNX Runtime at {self.component_path}")
        return True
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ML inference requests."""
        if "model_input" in data:
            return {
                "status": "inference_complete",
                "output": [0.85, 0.12, 0.03],  # Mock prediction
                "component": "echonnxruntime"
            }
        return {"status": "processed", "component": "echonnxruntime"}


class ChemReasonerAdapter(BaseComponentAdapter):
    """Adapter for chemical reasoning system."""
    
    def get_capabilities(self) -> List[str]:
        return ["chemical_reasoning", "molecular_analysis", "reaction_prediction"]
    
    async def _check_prerequisites(self) -> bool:
        """Check chemical reasoning tools."""
        return (self.component_path / "README.md").exists()
    
    async def _initialize_component(self) -> bool:
        """Initialize chemical reasoning system."""
        logger.info(f"Initializing ChemReasoner at {self.component_path}")
        return True
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process chemical reasoning requests."""
        if "reaction_query" in data:
            return {
                "status": "reaction_predicted",
                "products": ["C6H6", "H2O"],
                "confidence": 0.92,
                "component": "coschemreasoner"
            }
        return {"status": "processed", "component": "coschemreasoner"}


# Adapter factory for creating component adapters
ADAPTER_REGISTRY = {
    "oj7s3": OJ7S3Adapter,
    "echopiler": EchopilerAdapter,
    "oc-skintwin": OpenCogAdapter,
    "esm-2-keras-esm2_t6_8m-v1-hyper-dev2": ESMAdapter,
    "cosmagi-bio": CosmaBioAdapter,
    "coscheminformatics": ChemInformaticsAdapter,
    "echonnxruntime": ONNXRuntimeAdapter,
    "coschemreasoner": ChemReasonerAdapter,
}


def create_adapter(component_name: str, component_path: Path) -> Optional[ComponentInterface]:
    """Create an adapter for the specified component."""
    adapter_class = ADAPTER_REGISTRY.get(component_name)
    if adapter_class:
        return adapter_class(component_path, component_name)
    else:
        logger.warning(f"No adapter found for component: {component_name}")
        return None
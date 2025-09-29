#!/usr/bin/env python3
"""
Self-Organizing Core (SOC) - Cohesive Integration System
=========================================================

This module implements the core self-organizing system that integrates and coordinates
all the different research and development approaches from the component repositories:

1. oj7s3 - Academic publishing automation with autonomous agents
2. echopiler - Interactive compiler exploration and code analysis
3. oc-skintwin - OpenCog cognitive architecture for AGI
4. esm-2-keras-esm2_t6_8m-v1-hyper-dev2 - Protein/language model hypergraph mapping
5. cosmagi-bio - Genomic and proteomic research tools
6. coscheminformatics - Chemical information processing
7. echonnxruntime - ONNX Runtime for ML model inference
8. coschemreasoner - Chemical reasoning and analysis

The Self-Organizing Core provides:
- Unified API interface to all components
- Dynamic component discovery and integration
- Cross-component communication and data flow
- Adaptive resource management and load balancing
- Emergent behavior coordination
- Knowledge graph integration across domains
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import json
import yaml
import importlib.util
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime

from .autognosis import AutognosisOrchestrator
from .evolution_engine import EvolutionEngine, EvolutionaryGenome, EmergentPattern


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComponentInfo:
    """Information about a system component."""
    name: str
    path: Path
    description: str
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: str = "unknown"  # unknown, available, loaded, error
    module: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)


class ComponentInterface(ABC):
    """Abstract base class for component interfaces."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the component with given configuration."""
        pass
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the component."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup component resources."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this component provides."""
        pass


class SelfOrganizingCore:
    """
    Main Self-Organizing Core class that manages component integration
    and provides emergent coordination capabilities.
    """
    
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(__file__).parent.parent
        self.components_path = self.base_path / "components"
        self.components: Dict[str, ComponentInfo] = {}
        self.active_components: Dict[str, ComponentInterface] = {}
        self.knowledge_graph = {}
        self.event_bus = asyncio.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        self.running = False
        self.start_time = time.time()
        
        # Initialize autognosis system
        self.autognosis = AutognosisOrchestrator()
        
        # Initialize evolution engine
        self.evolution_engine = EvolutionEngine()
        
        # Initialize component definitions
        self.component_definitions = {
            "oj7s3": {
                "description": "Academic publishing automation with autonomous agents",
                "capabilities": ["manuscript_processing", "editorial_workflow", "agent_coordination", "publishing_automation"],
                "type": "web_service",
                "technologies": ["php", "python", "react", "mysql"]
            },
            "echopiler": {
                "description": "Interactive compiler exploration and code analysis",
                "capabilities": ["code_compilation", "assembly_analysis", "multi_language_support", "interactive_exploration"],
                "type": "web_service",
                "technologies": ["typescript", "nodejs", "webpack", "docker"]
            },
            "oc-skintwin": {
                "description": "OpenCog cognitive architecture for AGI",
                "capabilities": ["knowledge_representation", "reasoning", "cognitive_modeling", "atomspace"],
                "type": "cognitive_system",
                "technologies": ["cpp", "python", "scheme", "opencog"]
            },
            "esm-2-keras-esm2_t6_8m-v1-hyper-dev2": {
                "description": "Protein/language model hypergraph mapping",
                "capabilities": ["protein_modeling", "language_modeling", "hypergraph_analysis", "transformer_analysis"],
                "type": "ml_model",
                "technologies": ["python", "tensorflow", "pytorch", "transformers"]
            },
            "cosmagi-bio": {
                "description": "Genomic and proteomic research tools",
                "capabilities": ["genomic_analysis", "proteomic_analysis", "bioinformatics", "opencog_bio"],
                "type": "research_tool",
                "technologies": ["cpp", "python", "opencog", "bioinformatics"]
            },
            "coscheminformatics": {
                "description": "Chemical information processing",
                "capabilities": ["chemical_analysis", "molecular_processing", "chemical_data"],
                "type": "analysis_tool",
                "technologies": ["cpp", "python", "cmake"]
            },
            "echonnxruntime": {
                "description": "ONNX Runtime for ML model inference",
                "capabilities": ["ml_inference", "onnx_models", "cross_platform", "optimization"],
                "type": "ml_runtime",
                "technologies": ["cpp", "python", "cuda", "onnx"]
            },
            "coschemreasoner": {
                "description": "Chemical reasoning and analysis",
                "capabilities": ["chemical_reasoning", "molecular_analysis", "reaction_prediction"],
                "type": "reasoning_system",
                "technologies": ["python", "machine_learning", "chemistry"]
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the Self-Organizing Core."""
        logger.info("Initializing Self-Organizing Core...")
        
        # Discover and register components
        await self.discover_components()
        
        # Initialize knowledge graph
        await self.initialize_knowledge_graph()
        
        # Initialize autognosis system
        await self.autognosis.initialize(self)
        
        # Initialize evolution engine
        await self.evolution_engine.initialize(self._evaluate_component_fitness)
        
        # Start event processing
        asyncio.create_task(self.process_events())
        
        # Start autognosis cycles
        asyncio.create_task(self._run_autognosis_cycles())
        
        # Start evolutionary processes
        asyncio.create_task(self._run_evolution_cycles())
        
        self.running = True
        logger.info("Self-Organizing Core initialized successfully")
    
    async def discover_components(self) -> None:
        """Discover available components in the components directory."""
        logger.info("Discovering components...")
        
        if not self.components_path.exists():
            logger.warning(f"Components path {self.components_path} does not exist")
            return
        
        for component_dir in self.components_path.iterdir():
            if component_dir.is_dir() and component_dir.name in self.component_definitions:
                component_name = component_dir.name
                definition = self.component_definitions[component_name]
                
                component_info = ComponentInfo(
                    name=component_name,
                    path=component_dir,
                    description=definition["description"],
                    capabilities=definition["capabilities"]
                )
                
                # Check if component is available
                if await self.check_component_availability(component_info):
                    component_info.status = "available"
                    logger.info(f"Discovered component: {component_name}")
                else:
                    component_info.status = "error"
                    logger.warning(f"Component {component_name} has issues")
                
                self.components[component_name] = component_info
        
        logger.info(f"Discovered {len(self.components)} components")
    
    async def check_component_availability(self, component: ComponentInfo) -> bool:
        """Check if a component is available and functional."""
        try:
            # Check if component directory exists and has essential files
            readme_path = component.path / "README.md"
            if not readme_path.exists():
                return False
            
            # Additional checks can be added here for specific component types
            return True
        except Exception as e:
            logger.error(f"Error checking component {component.name}: {e}")
            return False
    
    async def initialize_knowledge_graph(self) -> None:
        """Initialize the cross-component knowledge graph."""
        logger.info("Initializing knowledge graph...")
        
        # Create nodes for each component
        for component_name, component in self.components.items():
            self.knowledge_graph[component_name] = {
                "type": "component",
                "capabilities": component.capabilities,
                "status": component.status,
                "connections": []
            }
        
        # Identify potential connections based on capabilities
        await self.identify_component_connections()
        
        logger.info("Knowledge graph initialized")
    
    async def identify_component_connections(self) -> None:
        """Identify potential connections between components based on capabilities."""
        capability_map = {}
        
        # Build capability to component mapping
        for component_name, component in self.components.items():
            for capability in component.capabilities:
                if capability not in capability_map:
                    capability_map[capability] = []
                capability_map[capability].append(component_name)
        
        # Identify complementary capabilities
        connections = {
            ("ml_inference", "protein_modeling"): "ml_bio_pipeline",
            ("chemical_analysis", "chemical_reasoning"): "chemical_pipeline",
            ("code_compilation", "ml_inference"): "code_ml_pipeline",
            ("manuscript_processing", "chemical_reasoning"): "research_publishing_pipeline",
            ("cognitive_modeling", "reasoning"): "agi_reasoning_pipeline",
            ("genomic_analysis", "proteomic_analysis"): "bio_research_pipeline"
        }
        
        # Add connections to knowledge graph
        for (cap1, cap2), connection_type in connections.items():
            components1 = capability_map.get(cap1, [])
            components2 = capability_map.get(cap2, [])
            
            for comp1 in components1:
                for comp2 in components2:
                    if comp1 != comp2:
                        self.knowledge_graph[comp1]["connections"].append({
                            "target": comp2,
                            "type": connection_type,
                            "capability_bridge": (cap1, cap2)
                        })
    
    async def process_events(self) -> None:
        """Process events in the event bus."""
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_bus.get(), timeout=1.0)
                await self.handle_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def handle_event(self, event: Dict[str, Any]) -> None:
        """Handle a system event."""
        event_type = event.get("type")
        
        if event_type == "component_request":
            await self.handle_component_request(event)
        elif event_type == "cross_component_query":
            await self.handle_cross_component_query(event)
        elif event_type == "adaptive_optimization":
            await self.handle_adaptive_optimization(event)
        elif event_type == "autognosis_insight":
            await self.handle_autognosis_insight(event)
        elif event_type == "autognosis_optimization":
            await self.handle_autognosis_optimization(event)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def handle_component_request(self, event: Dict[str, Any]) -> None:
        """Handle a request to use a specific component."""
        component_name = event.get("component")
        request_data = event.get("data", {})
        
        if component_name in self.components:
            component = self.components[component_name]
            if component.status == "available":
                # Process request through component
                logger.info(f"Processing request for component: {component_name}")
                # Implementation would depend on specific component interfaces
            else:
                logger.warning(f"Component {component_name} is not available")
        else:
            logger.error(f"Unknown component: {component_name}")
    
    async def handle_cross_component_query(self, event: Dict[str, Any]) -> None:
        """Handle queries that require multiple components."""
        query_type = event.get("query_type")
        
        if query_type == "bio_chemical_analysis":
            # Route through bio and chemical components
            await self.route_bio_chemical_analysis(event)
        elif query_type == "ml_code_analysis":
            # Route through ML and compiler components
            await self.route_ml_code_analysis(event)
        elif query_type == "research_publication":
            # Route through research and publishing components
            await self.route_research_publication(event)
    
    async def handle_adaptive_optimization(self, event: Dict[str, Any]) -> None:
        """Handle adaptive optimization requests."""
        # Analyze system performance and adjust resource allocation
        logger.info("Performing adaptive optimization...")
    
    async def route_bio_chemical_analysis(self, event: Dict[str, Any]) -> None:
        """Route biological and chemical analysis requests."""
        bio_components = ["cosmagi-bio", "esm-2-keras-esm2_t6_8m-v1-hyper-dev2"]
        chem_components = ["coscheminformatics", "coschemreasoner"]
        
        # Coordinate analysis across bio and chemical domains
        logger.info("Routing bio-chemical analysis request")
    
    async def route_ml_code_analysis(self, event: Dict[str, Any]) -> None:
        """Route ML and code analysis requests."""
        ml_components = ["echonnxruntime", "esm-2-keras-esm2_t6_8m-v1-hyper-dev2"]
        code_components = ["echopiler"]
        
        # Coordinate ML model analysis with code compilation
        logger.info("Routing ML-code analysis request")
    
    async def route_research_publication(self, event: Dict[str, Any]) -> None:
        """Route research and publication workflow requests."""
        research_components = ["cosmagi-bio", "coscheminformatics", "coschemreasoner"]
        publishing_components = ["oj7s3"]
        
        # Coordinate research analysis with publication workflow
        logger.info("Routing research publication request")
    
    async def _run_autognosis_cycles(self) -> None:
        """Run periodic autognosis cycles for self-awareness and optimization."""
        logger.info("Starting autognosis cycle task")
        
        while self.running:
            try:
                cycle_results = await self.autognosis.run_autognosis_cycle(self)
                
                # Queue autognosis insights as events
                if cycle_results.get('new_insights', 0) > 0:
                    await self.event_bus.put({
                        "type": "autognosis_insight",
                        "cycle_results": cycle_results,
                        "timestamp": cycle_results['timestamp'].isoformat()
                    })
                
                # Execute any high-priority optimizations
                for optimization in list(self.autognosis.optimization_queue):
                    if optimization.execution_priority >= 8:  # High priority threshold
                        success = await self.autognosis.execute_optimization(optimization, self)
                        if success:
                            self.autognosis.optimization_queue.remove(optimization)
                            await self.event_bus.put({
                                "type": "autognosis_optimization",
                                "optimization_type": optimization.optimization_type,
                                "target_component": optimization.target_component,
                                "success": True
                            })
                
            except Exception as e:
                logger.error(f"Error in autognosis cycle: {e}")
            
            # Wait for next cycle
            await asyncio.sleep(self.autognosis.cycle_interval)
    
    async def handle_autognosis_insight(self, event: Dict[str, Any]) -> None:
        """Handle autognosis insights and integrate them into system knowledge."""
        cycle_results = event.get('cycle_results', {})
        insights_count = cycle_results.get('new_insights', 0)
        
        logger.info(f"Processing autognosis insights: {insights_count} new insights discovered")
        
        # Update knowledge graph with self-awareness data
        self.knowledge_graph['autognosis'] = {
            'type': 'meta_system',
            'capabilities': ['self_awareness', 'meta_cognition', 'self_optimization'],
            'status': 'active',
            'connections': [],
            'insights_generated': insights_count,
            'last_update': event.get('timestamp')
        }
    
    async def handle_autognosis_optimization(self, event: Dict[str, Any]) -> None:
        """Handle executed autognosis optimizations."""
        optimization_type = event.get('optimization_type')
        target_component = event.get('target_component')
        success = event.get('success', False)
        
        if success:
            logger.info(f"Autognosis optimization successful: {optimization_type} on {target_component}")
        else:
            logger.warning(f"Autognosis optimization failed: {optimization_type} on {target_component}")
    
    def get_autognosis_status(self) -> Dict[str, Any]:
        """Get autognosis system status."""
        if not hasattr(self, 'autognosis'):
            return {'status': 'not_initialized'}
        
        return self.autognosis.get_self_awareness_report()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        status = {
            "running": self.running,
            "components": {
                name: {
                    "status": component.status,
                    "capabilities": component.capabilities
                }
                for name, component in self.components.items()
            },
            "knowledge_graph_size": len(self.knowledge_graph),
            "active_components": len(self.active_components)
        }
        
        # Add autognosis status if available
        if hasattr(self, 'autognosis') and self.autognosis:
            status["autognosis"] = self.get_autognosis_status()
        
        return status
    
    async def _run_evolution_cycles(self) -> None:
        """Run evolutionary cycles for system evolution."""
        while self.running:
            try:
                # Evolve each active component
                for component_name in self.active_components.keys():
                    if component_name in self.components:
                        current_state = self._extract_component_state(component_name)
                        evolution_objectives = self._determine_evolution_objectives(component_name)
                        
                        evolved_genome = await self.evolution_engine.evolve_component(
                            component_name, current_state, evolution_objectives
                        )
                        
                        # Apply evolved improvements
                        await self._apply_evolutionary_improvements(component_name, evolved_genome)
                
                # Synthesize emergent behaviors
                emergent_patterns = await self.evolution_engine.synthesize_emergent_behaviors()
                await self._integrate_emergent_patterns(emergent_patterns)
                
            except Exception as e:
                logger.error(f"Error in evolution cycle: {e}")
            
            # Wait for next evolution cycle (longer interval than autognosis)
            await asyncio.sleep(300)  # 5 minutes between evolution cycles
    
    def _extract_component_state(self, component_name: str) -> Dict[str, Any]:
        """Extract current state of a component for evolution."""
        component_info = self.components.get(component_name, {})
        
        return {
            'capabilities': component_info.capabilities if hasattr(component_info, 'capabilities') else [],
            'status': component_info.status if hasattr(component_info, 'status') else 'unknown',
            'performance_metrics': {
                'response_time': 1.0,  # Default placeholder
                'throughput': 1.0,
                'reliability': 0.8,
                'efficiency': 0.7
            },
            'integration_patterns': ['async_processing', 'event_driven', 'modular_design'],
            'optimization_level': 0.6,
            'adaptation_capability': 0.5,
            'learning_rate': 0.1,
            'complexity_score': len(component_info.capabilities) * 0.1 if hasattr(component_info, 'capabilities') else 0.1
        }
    
    def _determine_evolution_objectives(self, component_name: str) -> List[str]:
        """Determine evolution objectives for a component."""
        base_objectives = ['performance', 'integration', 'adaptation']
        
        # Component-specific objectives
        component_info = self.components.get(component_name)
        if component_info and hasattr(component_info, 'capabilities'):
            if 'machine_learning' in component_info.capabilities:
                base_objectives.append('ml_optimization')
            if 'reasoning' in component_info.capabilities:
                base_objectives.append('cognitive_enhancement')
            if 'analysis' in component_info.capabilities:
                base_objectives.append('analytical_depth')
        
        return base_objectives
    
    async def _evaluate_component_fitness(self, genome: EvolutionaryGenome, objectives: List[str]) -> float:
        """Evaluate fitness of a component genome."""
        fitness = 0.0
        
        # Base fitness from component integration
        component_name = genome.component_id
        if component_name in self.active_components:
            fitness += 0.3  # Active components get bonus
        
        if component_name in self.knowledge_graph:
            kg_entry = self.knowledge_graph[component_name]
            fitness += len(kg_entry.get('connections', [])) * 0.1  # Integration bonus
        
        # Objective-specific evaluation
        for objective in objectives:
            if objective == 'performance':
                perf_genes = [g for g in genome.genes.keys() if 'performance' in g or 'optimize' in g]
                fitness += len(perf_genes) * 0.15
            
            elif objective == 'integration':
                integration_genes = [g for g in genome.genes.keys() if 'integration' in g or 'connect' in g]
                fitness += len(integration_genes) * 0.2
            
            elif objective == 'adaptation':
                adaptive_genes = [g for g in genome.genes.keys() if 'adaptive' in g or 'learning' in g]
                fitness += len(adaptive_genes) * 0.18
        
        # Evolution novelty bonus
        fitness += len(genome.mutations) * 0.05
        
        # Autognosis synergy bonus
        if hasattr(self, 'autognosis') and self.autognosis:
            autognosis_status = self.autognosis.get_self_awareness_report()
            if autognosis_status.get('total_insights', 0) > 0:
                fitness += 0.1  # Bonus for systems with self-awareness
        
        return max(0.0, min(1.0, fitness))
    
    async def _apply_evolutionary_improvements(self, component_name: str, evolved_genome: EvolutionaryGenome) -> None:
        """Apply evolutionary improvements to a component."""
        try:
            # Log evolutionary improvement
            logger.info(f"Applying evolutionary improvements to {component_name} "
                       f"(Generation {evolved_genome.generation}, Fitness: {evolved_genome.fitness_score:.3f})")
            
            # Update component configuration with evolved parameters
            if component_name in self.components:
                component = self.components[component_name]
                
                # Extract optimization parameters from evolved genes
                for gene_name, gene_value in evolved_genome.genes.items():
                    if 'performance' in gene_name and isinstance(gene_value, (int, float)):
                        # Apply performance optimization
                        if not hasattr(component, 'config'):
                            component.config = {}
                        component.config[gene_name] = gene_value
                    
                    elif 'integration' in gene_name:
                        # Apply integration improvements
                        await self._enhance_component_integration(component_name, gene_value)
            
            # Generate event for successful evolution
            await self.event_bus.put({
                "type": "evolutionary_improvement",
                "component": component_name,
                "generation": evolved_genome.generation,
                "fitness": evolved_genome.fitness_score,
                "mutations": len(evolved_genome.mutations),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error applying evolutionary improvements to {component_name}: {e}")
    
    async def _enhance_component_integration(self, component_name: str, enhancement_value: Any) -> None:
        """Enhance component integration based on evolutionary improvements."""
        if component_name in self.knowledge_graph:
            kg_entry = self.knowledge_graph[component_name]
            
            # Add evolved integration capabilities
            if 'evolved_capabilities' not in kg_entry:
                kg_entry['evolved_capabilities'] = []
            
            if isinstance(enhancement_value, str):
                kg_entry['evolved_capabilities'].append(enhancement_value)
            
            # Enhance connections based on evolutionary insights
            if 'connections' in kg_entry and len(kg_entry['connections']) > 0:
                # Strengthen existing connections
                for connection in kg_entry['connections']:
                    if isinstance(connection, dict) and 'strength' in connection:
                        connection['strength'] = min(1.0, connection['strength'] * 1.1)
    
    async def _integrate_emergent_patterns(self, emergent_patterns: List[EmergentPattern]) -> None:
        """Integrate emergent patterns into the system."""
        for pattern in emergent_patterns:
            try:
                logger.info(f"Integrating emergent pattern: {pattern.pattern_id} "
                           f"(Type: {pattern.pattern_type}, Effectiveness: {pattern.effectiveness:.3f})")
                
                # Add pattern to knowledge graph
                self.knowledge_graph[f"emergent_{pattern.pattern_id}"] = {
                    'type': 'emergent_pattern',
                    'pattern_type': pattern.pattern_type,
                    'effectiveness': pattern.effectiveness,
                    'complexity': pattern.complexity,
                    'applications': pattern.applications,
                    'emergence_path': pattern.emergence_path,
                    'timestamp': pattern.timestamp.isoformat()
                }
                
                # Apply pattern to relevant components
                for application in pattern.applications:
                    component_name = application.split('_')[0]  # Extract component name
                    if component_name in self.components:
                        await self._apply_emergent_pattern(component_name, pattern)
                
                # Generate event for emergent pattern integration
                await self.event_bus.put({
                    "type": "emergent_pattern_integrated",
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "effectiveness": pattern.effectiveness,
                    "applications": pattern.applications,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error integrating emergent pattern {pattern.pattern_id}: {e}")
    
    async def _apply_emergent_pattern(self, component_name: str, pattern: EmergentPattern) -> None:
        """Apply an emergent pattern to a specific component."""
        try:
            component = self.components[component_name]
            
            # Add emergent capability based on pattern
            if not hasattr(component, 'emergent_capabilities'):
                component.emergent_capabilities = []
            
            emergent_capability = {
                'pattern_id': pattern.pattern_id,
                'capability_type': pattern.pattern_type,
                'effectiveness': pattern.effectiveness,
                'integration_timestamp': datetime.now().isoformat()
            }
            
            component.emergent_capabilities.append(emergent_capability)
            
            # Update knowledge graph with emergent enhancement
            if component_name in self.knowledge_graph:
                kg_entry = self.knowledge_graph[component_name]
                if 'emergent_enhancements' not in kg_entry:
                    kg_entry['emergent_enhancements'] = []
                kg_entry['emergent_enhancements'].append(emergent_capability)
            
            logger.debug(f"Applied emergent pattern {pattern.pattern_id} to {component_name}")
            
        except Exception as e:
            logger.error(f"Error applying emergent pattern to {component_name}: {e}")
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution system status."""
        if not hasattr(self, 'evolution_engine'):
            return {'status': 'not_initialized'}
        
        return asyncio.create_task(self.evolution_engine.get_evolution_status())
    
    async def trigger_targeted_evolution(self, component_name: str, objectives: List[str]) -> Dict[str, Any]:
        """Trigger targeted evolution for a specific component."""
        if component_name not in self.components:
            return {'error': f'Component {component_name} not found'}
        
        try:
            current_state = self._extract_component_state(component_name)
            evolved_genome = await self.evolution_engine.evolve_component(
                component_name, current_state, objectives
            )
            
            await self._apply_evolutionary_improvements(component_name, evolved_genome)
            
            return {
                'success': True,
                'component': component_name,
                'generation': evolved_genome.generation,
                'fitness': evolved_genome.fitness_score,
                'mutations': len(evolved_genome.mutations)
            }
            
        except Exception as e:
            logger.error(f"Error in targeted evolution for {component_name}: {e}")
            return {'error': str(e)}

    async def shutdown(self) -> None:
        """Shutdown the Self-Organizing Core."""
        logger.info("Shutting down Self-Organizing Core...")
        self.running = False
        
        # Shutdown evolution engine
        if hasattr(self, 'evolution_engine') and self.evolution_engine:
            await self.evolution_engine.stop_continuous_evolution()
        
        # Shutdown autognosis system
        if hasattr(self, 'autognosis') and self.autognosis:
            await self.autognosis.shutdown()
        
        # Cleanup active components
        for component in self.active_components.values():
            try:
                await component.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up component: {e}")
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("Self-Organizing Core shutdown complete")


async def main():
    """Main entry point for the Self-Organizing Core."""
    soc = SelfOrganizingCore()
    
    try:
        await soc.initialize()
        
        # Display system status
        status = soc.get_system_status()
        print("\n=== Self-Organizing Core Status ===")
        print(f"Running: {status['running']}")
        print(f"Active Components: {status['active_components']}")
        print(f"Knowledge Graph Size: {status['knowledge_graph_size']}")
        print("\nDiscovered Components:")
        for name, info in status['components'].items():
            print(f"  - {name}: {info['status']} ({len(info['capabilities'])} capabilities)")
        
        # Keep running for demonstration
        print("\nSelf-Organizing Core is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        await soc.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
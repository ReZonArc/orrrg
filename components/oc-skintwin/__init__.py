"""
oc-skintwin - ORRRG Component

OpenCog cognitive architecture for artificial general intelligence

Full integration with ORRRG's Relevance Realization Ennead and Evolution Engine.
"""

from typing import Dict, List, Any, Optional
import logging
from .opencog_adapter import OpenCogAdapter, AtomType, Atom, SimpleAtomSpace, PLNInferenceEngine, ECANAttentionBank

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


class Oc_SkintwinComponent:
    """
    ORRRG component for OpenCog cognitive architecture.
    
    Provides AGI reasoning capabilities including:
    - Knowledge representation in AtomSpace
    - Cognitive reasoning and inference
    - Pattern learning and discovery
    - Relevance-guided attention allocation
    - Evolutionary pattern optimization
    
    Integrated with:
    - Relevance Realization Ennead (9-dimensional framework)
    - Evolution Engine (genetic programming)
    - Autognosis (self-awareness)
    """
    
    def __init__(self):
        self.config = {}
        self.initialized = False
        self.adapter: Optional[OpenCogAdapter] = None
        logger.info(f"{self.__class__.__name__} created")
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the component with configuration."""
        self.config = config
        logger.info(f"Initializing {self.__class__.__name__}")
        
        # Initialize OpenCog adapter
        self.adapter = OpenCogAdapter()
        await self.adapter.initialize(config)
        
        self.initialized = True
        logger.info(f"{self.__class__.__name__} initialized successfully")
        return True
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through cognitive reasoning."""
        if not self.initialized or not self.adapter:
            raise RuntimeError("Component not initialized")
        
        logger.info(f"Processing data in {self.__class__.__name__}")
        
        # Route to appropriate cognitive function
        operation = data.get('operation', 'reason')
        
        if operation == 'reason':
            result = await self.adapter.process(data)
        elif operation == 'learn':
            result = await self.adapter.process(data)
        elif operation == 'query':
            result = await self.adapter.process(data)
        elif operation == 'infer':
            result = await self.adapter.process(data)
        elif operation == 'integrate_relevance':
            result = await self.adapter.integrate_with_relevance_realization(
                data.get('relevance_context', {})
            )
        elif operation == 'evolve':
            result = await self.adapter.evolve_reasoning_patterns(
                data.get('evolution_objectives', [])
            )
        else:
            result = await self.adapter.process(data)
        
        result['component'] = 'oc-skintwin'
        return result
    
    async def cognitive_reasoning(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform cognitive reasoning on a query.
        
        Args:
            query: The reasoning query
            context: Additional context for reasoning
            
        Returns:
            Reasoning results including matches and inferences
        """
        return await self.process({
            'operation': 'reason',
            'query': query,
            'context': context or {}
        })
    
    async def learn_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Learn new reasoning patterns.
        
        Args:
            patterns: List of patterns to learn
            
        Returns:
            Learning results
        """
        return await self.process({
            'operation': 'learn',
            'patterns': patterns
        })
    
    async def query_knowledge(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the knowledge base.
        
        Args:
            pattern: Pattern to match against
            
        Returns:
            Query results
        """
        return await self.process({
            'operation': 'query',
            'pattern': pattern
        })
    
    async def integrate_with_relevance(self, relevance_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate cognitive reasoning with relevance realization.
        
        Uses the Ennead framework to guide attention allocation.
        
        Args:
            relevance_context: Context from relevance realization optimizer
            
        Returns:
            Integration results with attention allocation
        """
        return await self.process({
            'operation': 'integrate_relevance',
            'relevance_context': relevance_context
        })
    
    async def evolve_reasoning(self, objectives: List[str]) -> Dict[str, Any]:
        """
        Evolve reasoning patterns based on objectives.
        
        Args:
            objectives: Evolution objectives (e.g., 'efficiency', 'connectivity', 'diversity')
            
        Returns:
            Evolution results
        """
        return await self.process({
            'operation': 'evolve',
            'evolution_objectives': objectives
        })
    
    async def cleanup(self) -> None:
        """Cleanup component resources."""
        logger.info(f"Cleaning up {self.__class__.__name__}")
        
        if self.adapter:
            await self.adapter.cleanup()
        
        self.initialized = False
    
    def get_capabilities(self) -> List[str]:
        """Return list of component capabilities."""
        if self.adapter:
            return self.adapter.get_capabilities()
        
        return [
            "cognitive_reasoning",
            "knowledge_representation",
            "pattern_matching",
            "logical_inference",
            "agi_reasoning"
        ]
    
    def get_info(self) -> Dict[str, Any]:
        """Return component information."""
        info = {
            "name": "oc-skintwin",
            "description": "OpenCog cognitive architecture for AGI reasoning",
            "version": __version__,
            "upstream": "https://github.com/opencog/atomspace",
            "upstream_project": "OpenCog AtomSpace",
            "integration_type": "Cognitive AGI Adapter",
            "dependencies": ['opencog (optional)', 'networkx'],
            "initialized": self.initialized,
            "features": [
                "AtomSpace knowledge representation",
                "Cognitive reasoning and inference",
                "Pattern learning and discovery",
                "Relevance-guided attention",
                "Evolutionary optimization",
                "AGI reasoning capabilities"
            ]
        }
        
        if self.adapter and self.initialized:
            info['statistics'] = self.adapter.get_statistics()
        
        return info


# Export main component class and adapter
__all__ = ['Oc_SkintwinComponent', 'OpenCogAdapter', 'AtomType', 'Atom', 'SimpleAtomSpace']


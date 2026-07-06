"""
echonnxruntime - ORRRG Component

ONNX Runtime for optimized machine learning model inference.

Provides model loading, session management, optimized inference execution,
and integration with ORRRG's Relevance Realization Ennead and Evolution Engine.
"""

from typing import Dict, List, Any, Optional
import logging

from .onnx_adapter import (
    OnnxInferenceAdapter,
    ModelMetadata,
    InferenceResult,
    InferenceStats,
    ExecutionProvider,
    OptimizationLevel,
)

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


class EchonnxruntimeComponent:
    """
    ORRRG component for ONNX Runtime ML inference.
    
    Provides full inference capabilities including:
    - Model loading and session management
    - Optimized inference with provider selection (CPU/CUDA/TensorRT)
    - Graph optimization at multiple levels
    - Inference statistics and performance tracking
    - Integration with Relevance Realization for attention-guided inference
    - Evolution Engine compatibility for model selection optimization
    """
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.initialized = False
        self.adapter: Optional[OnnxInferenceAdapter] = None
        logger.info(f"{self.__class__.__name__} created")
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the component with configuration."""
        self.config = config
        logger.info(f"Initializing {self.__class__.__name__}")
        
        self.adapter = OnnxInferenceAdapter()
        await self.adapter.initialize(config)
        
        self.initialized = True
        logger.info(f"{self.__class__.__name__} initialized (ort_available={self.adapter.is_available()})")
        return True
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through the component.
        
        Supported actions:
        - load_model: Load an ONNX model (requires 'model_path')
        - infer: Run inference (requires 'model_name' and 'inputs')
        - optimize: Optimize a model (requires 'model_path' and 'output_path')
        - unload: Unload a model (requires 'model_name')
        - status: Get adapter status
        - stats: Get inference statistics
        """
        if not self.initialized or not self.adapter:
            raise RuntimeError("Component not initialized")
        
        action = data.get('action', 'infer')
        logger.info(f"Processing action '{action}' in {self.__class__.__name__}")
        
        if action == 'load_model':
            metadata = await self.adapter.load_model(
                model_path=data['model_path'],
                model_name=data.get('model_name'),
                providers=data.get('providers'),
            )
            return {'status': 'success', 'metadata': metadata.to_dict()}
        
        elif action == 'infer':
            result = await self.adapter.run_inference(
                model_name=data['model_name'],
                inputs=data['inputs'],
                output_names=data.get('output_names'),
            )
            return {'status': 'success', 'result': result.to_dict()}
        
        elif action == 'optimize':
            opt_result = await self.adapter.optimize_model(
                model_path=data['model_path'],
                output_path=data['output_path'],
                optimization_level=OptimizationLevel[data.get('level', 'ALL')],
            )
            return {'status': 'success', 'optimization': opt_result}
        
        elif action == 'unload':
            success = await self.adapter.unload_model(data['model_name'])
            return {'status': 'success' if success else 'not_found'}
        
        elif action == 'status':
            return {'status': 'success', 'adapter_status': self.adapter.get_status()}
        
        elif action == 'stats':
            return {'status': 'success', 'stats': self.adapter.get_inference_stats(data.get('model_name'))}
        
        else:
            return {'status': 'error', 'message': f"Unknown action: {action}"}
    
    async def cleanup(self) -> None:
        """Cleanup component resources."""
        logger.info(f"Cleaning up {self.__class__.__name__}")
        if self.adapter:
            await self.adapter.cleanup()
        self.initialized = False
    
    def get_capabilities(self) -> List[str]:
        """Return list of component capabilities."""
        return [
            "model_loading",
            "onnx_inference",
            "model_optimization",
            "multi_provider_execution",
            "inference_statistics",
            "batch_inference",
            "session_management",
        ]
    
    def get_info(self) -> Dict[str, Any]:
        """Return component information."""
        info = {
            "name": "echonnxruntime",
            "description": "ONNX Runtime for optimized machine learning model inference",
            "version": __version__,
            "upstream": "https://github.com/ReZonArc/echonnxruntime",
            "upstream_project": "ONNX Runtime",
            "integration_type": "ML inference engine",
            "dependencies": ['onnxruntime', 'numpy'],
            "initialized": self.initialized,
            "capabilities": self.get_capabilities(),
        }
        if self.adapter:
            info['adapter_status'] = self.adapter.get_status()
        return info

    # --- Relevance Realization Integration ---

    def get_relevance_dimensions(self) -> Dict[str, float]:
        """
        Return relevance scores for the Ennead framework.
        
        Inference is primarily procedural (knowing-how) and nomological
        (understanding causal mechanisms in ML models).
        """
        return {
            'propositional': 0.6,   # Model architectures as knowledge
            'procedural': 0.95,     # Core capability: executing inference
            'perspectival': 0.4,    # Model outputs frame interpretation
            'participatory': 0.3,   # Limited identity transformation
            'nomological': 0.85,    # Understanding how models work
            'normative': 0.3,       # Limited value judgment
            'narrative': 0.5,       # Model versioning/evolution history
            'morality': 0.2,        # Fairness in inference
            'meaning': 0.4,        # Model outputs contribute to meaning
            'mastery': 0.8,         # Optimization as skilled engagement
        }

    def get_evolution_genome(self) -> Dict[str, Any]:
        """
        Return evolutionary parameters for the Evolution Engine.
        
        These can be mutated/crossed to evolve inference strategies.
        """
        return {
            'optimization_level': self.config.get('optimization_level', 'ALL'),
            'providers': self.config.get('providers', ['CPUExecutionProvider']),
            'batch_size': self.config.get('batch_size', 1),
            'enable_profiling': self.config.get('enable_profiling', False),
            'thread_count': self.config.get('thread_count', 0),
        }


# Export main component class and adapter
__all__ = [
    'EchonnxruntimeComponent',
    'OnnxInferenceAdapter',
    'ModelMetadata',
    'InferenceResult',
    'InferenceStats',
    'ExecutionProvider',
    'OptimizationLevel',
]

"""
Unified Inference Pipeline
===========================

Routes ML model inference from any ORRRG component through the echonnxruntime
component for optimized execution.

Components like esm-2-keras, coschemreasoner, etc. can register models and
submit inference requests through this centralized pipeline, benefiting from:
- Shared GPU/accelerator resources
- Model caching and session reuse
- Unified statistics and monitoring
- Provider auto-selection (CPU, CUDA, TensorRT)
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class InferencePipelineRequest:
    """A request from a component for inference."""
    request_id: str
    source_component: str
    model_name: str
    inputs: Dict[str, Any]
    output_names: Optional[List[str]] = None
    priority: int = 5  # 1=highest, 10=lowest
    submitted_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferencePipelineResult:
    """Result of a pipeline inference request."""
    request_id: str
    source_component: str
    model_name: str
    outputs: Dict[str, Any]
    latency_ms: float
    provider: str
    completed_at: float = field(default_factory=time.time)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'source_component': self.source_component,
            'model_name': self.model_name,
            'outputs': {k: v.tolist() if hasattr(v, 'tolist') else v
                        for k, v in self.outputs.items()},
            'latency_ms': round(self.latency_ms, 3),
            'provider': self.provider,
            'success': self.success,
            'error': self.error,
        }


class UnifiedInferencePipeline:
    """
    Central inference pipeline that routes requests through echonnxruntime.
    
    Components register models once, then submit inference requests.
    The pipeline handles model lifecycle, queuing, and resource sharing.
    """

    def __init__(self):
        self._onnx_component = None
        self._registered_models: Dict[str, Dict[str, Any]] = {}
        self._request_count: int = 0
        self._total_latency_ms: float = 0.0
        self._errors: int = 0
        self._initialized = False

    async def initialize(self, onnx_component) -> bool:
        """
        Initialize the pipeline with the echonnxruntime component.
        
        Args:
            onnx_component: Initialized EchonnxruntimeComponent instance
        """
        self._onnx_component = onnx_component
        self._initialized = True
        logger.info("UnifiedInferencePipeline initialized")
        return True

    async def register_model(
        self,
        model_name: str,
        model_path: str,
        owner_component: str,
        providers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a model for inference through the pipeline.
        
        Args:
            model_name: Unique identifier for the model
            model_path: Path to the .onnx model file
            owner_component: Component that owns this model
            providers: Preferred execution providers
            metadata: Additional model metadata
        
        Returns:
            Model metadata from the loaded model
        """
        if not self._initialized or not self._onnx_component:
            raise RuntimeError("Pipeline not initialized")

        # Load model through echonnxruntime
        result = await self._onnx_component.process({
            'action': 'load_model',
            'model_path': model_path,
            'model_name': model_name,
            'providers': providers,
        })

        self._registered_models[model_name] = {
            'owner': owner_component,
            'path': model_path,
            'providers': providers,
            'metadata': metadata or {},
            'registered_at': datetime.utcnow().isoformat(),
        }

        logger.info(f"Model '{model_name}' registered by '{owner_component}'")
        return result

    async def infer(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        source_component: str,
        output_names: Optional[List[str]] = None,
    ) -> InferencePipelineResult:
        """
        Submit an inference request.
        
        Args:
            model_name: Name of a registered model
            inputs: Input tensors as dict of name -> array/list
            source_component: Component making the request
            output_names: Specific outputs to fetch (None = all)
        
        Returns:
            InferencePipelineResult with outputs
        """
        if not self._initialized or not self._onnx_component:
            raise RuntimeError("Pipeline not initialized")

        if model_name not in self._registered_models:
            return InferencePipelineResult(
                request_id=f"req_{self._request_count}",
                source_component=source_component,
                model_name=model_name,
                outputs={},
                latency_ms=0,
                provider="none",
                success=False,
                error=f"Model '{model_name}' not registered",
            )

        self._request_count += 1
        request_id = f"req_{self._request_count}"

        start_time = time.time()
        try:
            result = await self._onnx_component.process({
                'action': 'infer',
                'model_name': model_name,
                'inputs': inputs,
                'output_names': output_names,
            })

            latency_ms = (time.time() - start_time) * 1000
            self._total_latency_ms += latency_ms

            inference_result = result.get('result', {})
            return InferencePipelineResult(
                request_id=request_id,
                source_component=source_component,
                model_name=model_name,
                outputs=inference_result.get('outputs', {}),
                latency_ms=latency_ms,
                provider=inference_result.get('provider_used', 'unknown'),
            )

        except Exception as e:
            self._errors += 1
            latency_ms = (time.time() - start_time) * 1000
            return InferencePipelineResult(
                request_id=request_id,
                source_component=source_component,
                model_name=model_name,
                outputs={},
                latency_ms=latency_ms,
                provider="error",
                success=False,
                error=str(e),
            )

    async def unregister_model(self, model_name: str) -> bool:
        """Unregister and unload a model."""
        if model_name not in self._registered_models:
            return False

        if self._onnx_component:
            await self._onnx_component.process({
                'action': 'unload',
                'model_name': model_name,
            })

        del self._registered_models[model_name]
        logger.info(f"Model '{model_name}' unregistered")
        return True

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models."""
        return dict(self._registered_models)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        avg_latency = (self._total_latency_ms / self._request_count
                       if self._request_count > 0 else 0)
        return {
            'initialized': self._initialized,
            'registered_models': len(self._registered_models),
            'total_requests': self._request_count,
            'total_errors': self._errors,
            'avg_latency_ms': round(avg_latency, 3),
            'models': list(self._registered_models.keys()),
        }

    async def cleanup(self) -> None:
        """Unregister all models and clean up."""
        for name in list(self._registered_models.keys()):
            await self.unregister_model(name)
        self._initialized = False
        logger.info("UnifiedInferencePipeline cleaned up")

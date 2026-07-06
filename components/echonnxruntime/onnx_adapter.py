"""
ONNX Inference Adapter for ORRRG
=================================

Provides model loading, session management, inference execution, and optimization
for ONNX models within the ORRRG framework.

Integrates with:
- Self-Organizing Core for component lifecycle
- Relevance Realization Ennead for attention-guided inference
- Evolution Engine for model selection optimization
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import numpy as np

logger = logging.getLogger(__name__)


class ExecutionProvider(Enum):
    """Available ONNX Runtime execution providers."""
    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"
    DIRECTML = "DmlExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"


class OptimizationLevel(Enum):
    """Graph optimization levels for ONNX Runtime."""
    DISABLED = 0
    BASIC = 1
    EXTENDED = 2
    ALL = 99


@dataclass
class ModelMetadata:
    """Metadata about a loaded ONNX model."""
    name: str
    path: Optional[str] = None
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    input_shapes: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    output_shapes: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    input_types: Dict[str, str] = field(default_factory=dict)
    opset_version: int = 0
    producer: str = ""
    domain: str = ""
    description: str = ""
    load_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'path': self.path,
            'input_names': self.input_names,
            'output_names': self.output_names,
            'input_shapes': self.input_shapes,
            'output_shapes': self.output_shapes,
            'input_types': self.input_types,
            'opset_version': self.opset_version,
            'producer': self.producer,
            'domain': self.domain,
            'description': self.description,
            'load_time_ms': self.load_time_ms,
        }


@dataclass
class InferenceResult:
    """Result from an inference run."""
    outputs: Dict[str, Any]
    latency_ms: float
    model_name: str
    provider_used: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'outputs': {k: v.tolist() if hasattr(v, 'tolist') else v
                        for k, v in self.outputs.items()},
            'latency_ms': self.latency_ms,
            'model_name': self.model_name,
            'provider_used': self.provider_used,
            'timestamp': self.timestamp,
        }


@dataclass
class InferenceStats:
    """Accumulated inference statistics for a model."""
    total_runs: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_runs if self.total_runs > 0 else 0.0

    def record(self, latency_ms: float) -> None:
        self.total_runs += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_runs': self.total_runs,
            'avg_latency_ms': round(self.avg_latency_ms, 3),
            'min_latency_ms': round(self.min_latency_ms, 3) if self.total_runs > 0 else None,
            'max_latency_ms': round(self.max_latency_ms, 3),
            'errors': self.errors,
        }


class OnnxInferenceAdapter:
    """
    ONNX Runtime inference adapter for ORRRG.
    
    Manages model lifecycle, inference sessions, and optimization
    within the ORRRG component framework.
    """

    def __init__(self):
        self.initialized = False
        self.config: Dict[str, Any] = {}
        self._sessions: Dict[str, Any] = {}  # model_name -> InferenceSession
        self._metadata: Dict[str, ModelMetadata] = {}
        self._stats: Dict[str, InferenceStats] = {}
        self._default_providers: List[str] = [ExecutionProvider.CPU.value]
        self._optimization_level: OptimizationLevel = OptimizationLevel.ALL
        self._ort_available = False
        self._ort = None

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the ONNX Runtime adapter."""
        self.config = config
        
        # Try to import onnxruntime
        try:
            import onnxruntime as ort
            self._ort = ort
            self._ort_available = True
            logger.info(f"ONNX Runtime version: {ort.__version__}")
            logger.info(f"Available providers: {ort.get_available_providers()}")
            
            # Configure providers from config
            providers = config.get('providers', ['CPUExecutionProvider'])
            available = ort.get_available_providers()
            self._default_providers = [p for p in providers if p in available]
            if not self._default_providers:
                self._default_providers = [ExecutionProvider.CPU.value]
            
        except ImportError:
            logger.warning(
                "onnxruntime not installed. Running in simulation mode. "
                "Install with: pip install onnxruntime"
            )
            self._ort_available = False

        # Set optimization level
        opt_level = config.get('optimization_level', 'ALL')
        self._optimization_level = OptimizationLevel[opt_level] if opt_level in OptimizationLevel.__members__ else OptimizationLevel.ALL

        self.initialized = True
        logger.info(f"OnnxInferenceAdapter initialized (ort_available={self._ort_available})")
        return True

    async def load_model(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        providers: Optional[List[str]] = None,
        optimization_level: Optional[OptimizationLevel] = None,
    ) -> ModelMetadata:
        """
        Load an ONNX model and create an inference session.
        
        Args:
            model_path: Path to .onnx model file
            model_name: Optional name identifier (defaults to filename)
            providers: Override execution providers for this model
            optimization_level: Override optimization level for this model
        
        Returns:
            ModelMetadata with model information
        """
        if not self.initialized:
            raise RuntimeError("Adapter not initialized")

        path = Path(model_path)
        name = model_name or path.stem

        if name in self._sessions:
            logger.info(f"Model '{name}' already loaded, returning cached metadata")
            return self._metadata[name]

        start_time = time.time()
        metadata = ModelMetadata(name=name, path=str(path))

        if self._ort_available:
            ort = self._ort
            sess_options = ort.SessionOptions()
            opt = optimization_level or self._optimization_level
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel(opt.value)
            sess_options.enable_profiling = self.config.get('enable_profiling', False)

            use_providers = providers or self._default_providers

            try:
                session = ort.InferenceSession(
                    str(path),
                    sess_options=sess_options,
                    providers=use_providers,
                )
                self._sessions[name] = session

                # Extract metadata
                for inp in session.get_inputs():
                    metadata.input_names.append(inp.name)
                    metadata.input_shapes[inp.name] = inp.shape
                    metadata.input_types[inp.name] = inp.type

                for out in session.get_outputs():
                    metadata.output_names.append(out.name)
                    metadata.output_shapes[out.name] = out.shape

                model_meta = session.get_modelmeta()
                metadata.producer = model_meta.producer_name
                metadata.domain = model_meta.domain
                metadata.description = model_meta.description

            except Exception as e:
                logger.error(f"Failed to load model '{name}' from {path}: {e}")
                raise

        else:
            # Simulation mode — create a placeholder session
            self._sessions[name] = {'simulated': True, 'path': str(path)}
            metadata.input_names = ['input']
            metadata.output_names = ['output']
            metadata.input_shapes = {'input': [1, -1]}
            metadata.output_shapes = {'output': [1, -1]}
            metadata.input_types = {'input': 'tensor(float)'}
            logger.info(f"Model '{name}' loaded in simulation mode")

        metadata.load_time_ms = (time.time() - start_time) * 1000
        self._metadata[name] = metadata
        self._stats[name] = InferenceStats()
        
        logger.info(f"Model '{name}' loaded in {metadata.load_time_ms:.1f}ms")
        return metadata

    async def run_inference(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        output_names: Optional[List[str]] = None,
    ) -> InferenceResult:
        """
        Run inference on a loaded model.
        
        Args:
            model_name: Name of the loaded model
            inputs: Dictionary mapping input names to numpy arrays or lists
            output_names: Optional list of output names to fetch (None = all)
        
        Returns:
            InferenceResult with outputs and timing
        """
        if model_name not in self._sessions:
            raise ValueError(f"Model '{model_name}' not loaded. Call load_model first.")

        stats = self._stats[model_name]
        start_time = time.time()

        try:
            if self._ort_available:
                session = self._sessions[model_name]
                # Convert lists to numpy arrays
                np_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, np.ndarray):
                        np_inputs[k] = v
                    else:
                        np_inputs[k] = np.array(v, dtype=np.float32)

                fetch_names = output_names or [o.name for o in session.get_outputs()]
                raw_outputs = session.run(fetch_names, np_inputs)
                
                outputs = {name: out for name, out in zip(fetch_names, raw_outputs)}
                provider_used = session.get_providers()[0] if session.get_providers() else "unknown"

            else:
                # Simulation mode
                outputs = {}
                metadata = self._metadata[model_name]
                for out_name in (output_names or metadata.output_names):
                    # Generate placeholder output
                    shape = metadata.output_shapes.get(out_name, [1, 1])
                    resolved_shape = [max(1, s) if s and s > 0 else 1 for s in shape]
                    outputs[out_name] = np.random.randn(*resolved_shape).astype(np.float32)
                provider_used = "SimulationProvider"

            latency_ms = (time.time() - start_time) * 1000
            stats.record(latency_ms)

            return InferenceResult(
                outputs=outputs,
                latency_ms=latency_ms,
                model_name=model_name,
                provider_used=provider_used,
            )

        except Exception as e:
            stats.errors += 1
            logger.error(f"Inference failed for model '{model_name}': {e}")
            raise

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model and free resources."""
        if model_name not in self._sessions:
            return False
        
        del self._sessions[model_name]
        logger.info(f"Model '{model_name}' unloaded")
        return True

    async def optimize_model(
        self,
        model_path: str,
        output_path: str,
        optimization_level: OptimizationLevel = OptimizationLevel.ALL,
    ) -> Dict[str, Any]:
        """
        Optimize an ONNX model and save the optimized version.
        
        Args:
            model_path: Path to input .onnx model
            output_path: Path to save optimized model
            optimization_level: Optimization level to apply
        
        Returns:
            Dictionary with optimization results
        """
        if not self._ort_available:
            logger.warning("ONNX Runtime not available; optimization skipped")
            return {'status': 'skipped', 'reason': 'onnxruntime not installed'}

        ort = self._ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel(optimization_level.value)
        sess_options.optimized_model_filepath = output_path

        start_time = time.time()
        # Creating session with optimized_model_filepath saves the optimized graph
        ort.InferenceSession(str(model_path), sess_options=sess_options, providers=self._default_providers)
        elapsed_ms = (time.time() - start_time) * 1000

        return {
            'status': 'success',
            'input_path': model_path,
            'output_path': output_path,
            'optimization_level': optimization_level.name,
            'elapsed_ms': round(elapsed_ms, 2),
        }

    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a loaded model."""
        return self._metadata.get(model_name)

    def get_inference_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get inference statistics for one or all models."""
        if model_name:
            stats = self._stats.get(model_name)
            return {model_name: stats.to_dict()} if stats else {}
        return {name: stats.to_dict() for name, stats in self._stats.items()}

    def list_loaded_models(self) -> List[str]:
        """Return names of all currently loaded models."""
        return list(self._sessions.keys())

    def is_available(self) -> bool:
        """Check if ONNX Runtime is actually available."""
        return self._ort_available

    async def cleanup(self) -> None:
        """Release all resources."""
        model_names = list(self._sessions.keys())
        for name in model_names:
            await self.unload_model(name)
        self._metadata.clear()
        self._stats.clear()
        self.initialized = False
        logger.info("OnnxInferenceAdapter cleaned up")

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status summary."""
        return {
            'initialized': self.initialized,
            'ort_available': self._ort_available,
            'ort_version': self._ort.__version__ if self._ort_available else None,
            'default_providers': self._default_providers,
            'optimization_level': self._optimization_level.name,
            'loaded_models': len(self._sessions),
            'models': self.list_loaded_models(),
            'stats': self.get_inference_stats(),
        }

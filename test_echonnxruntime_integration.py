"""
Integration tests for the echonnxruntime component.
Tests model loading, inference (simulation mode), optimization, and cross-component pipeline.
"""

import asyncio
import pytest
import numpy as np

# Mark all tests as async
pytestmark = pytest.mark.asyncio


class TestOnnxAdapter:
    """Tests for OnnxInferenceAdapter."""

    async def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        from components.echonnxruntime.onnx_adapter import OnnxInferenceAdapter
        adapter = OnnxInferenceAdapter()
        assert not adapter.initialized

        result = await adapter.initialize({})
        assert result is True
        assert adapter.initialized

    async def test_adapter_status(self):
        """Test adapter status reporting."""
        from components.echonnxruntime.onnx_adapter import OnnxInferenceAdapter
        adapter = OnnxInferenceAdapter()
        await adapter.initialize({})

        status = adapter.get_status()
        assert status['initialized'] is True
        assert 'ort_available' in status
        assert status['loaded_models'] == 0
        assert status['optimization_level'] == 'ALL'

    async def test_simulation_mode_load_model(self, tmp_path):
        """Test model loading in simulation mode."""
        from components.echonnxruntime.onnx_adapter import OnnxInferenceAdapter

        adapter = OnnxInferenceAdapter()
        await adapter.initialize({})

        # Create a dummy model file
        model_file = tmp_path / "test_model.onnx"
        model_file.write_bytes(b"dummy onnx model")

        metadata = await adapter.load_model(str(model_file), model_name="test")
        assert metadata.name == "test"
        assert metadata.load_time_ms > 0
        assert "test" in adapter.list_loaded_models()

    async def test_simulation_mode_inference(self, tmp_path):
        """Test inference in simulation mode."""
        from components.echonnxruntime.onnx_adapter import OnnxInferenceAdapter

        adapter = OnnxInferenceAdapter()
        await adapter.initialize({})

        model_file = tmp_path / "test_model.onnx"
        model_file.write_bytes(b"dummy")

        await adapter.load_model(str(model_file), model_name="test")
        result = await adapter.run_inference("test", {"input": [[1.0, 2.0, 3.0]]})

        assert result.model_name == "test"
        assert result.latency_ms > 0
        assert "output" in result.outputs

    async def test_inference_stats_tracking(self, tmp_path):
        """Test that inference statistics are accumulated."""
        from components.echonnxruntime.onnx_adapter import OnnxInferenceAdapter

        adapter = OnnxInferenceAdapter()
        await adapter.initialize({})

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")
        await adapter.load_model(str(model_file), model_name="stats_test")

        for _ in range(5):
            await adapter.run_inference("stats_test", {"input": [[1.0]]})

        stats = adapter.get_inference_stats("stats_test")
        assert stats["stats_test"]["total_runs"] == 5
        assert stats["stats_test"]["avg_latency_ms"] > 0

    async def test_model_unload(self, tmp_path):
        """Test model unloading."""
        from components.echonnxruntime.onnx_adapter import OnnxInferenceAdapter

        adapter = OnnxInferenceAdapter()
        await adapter.initialize({})

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")
        await adapter.load_model(str(model_file), model_name="unload_test")

        assert "unload_test" in adapter.list_loaded_models()
        await adapter.unload_model("unload_test")
        assert "unload_test" not in adapter.list_loaded_models()

    async def test_cleanup(self, tmp_path):
        """Test full cleanup."""
        from components.echonnxruntime.onnx_adapter import OnnxInferenceAdapter

        adapter = OnnxInferenceAdapter()
        await adapter.initialize({})

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")
        await adapter.load_model(str(model_file), model_name="cleanup_test")

        await adapter.cleanup()
        assert not adapter.initialized
        assert len(adapter.list_loaded_models()) == 0


class TestEchonnxruntimeComponent:
    """Tests for the full component wrapper."""

    async def test_component_lifecycle(self):
        """Test component initialize and cleanup."""
        from components.echonnxruntime import EchonnxruntimeComponent

        comp = EchonnxruntimeComponent()
        assert not comp.initialized

        await comp.initialize({})
        assert comp.initialized
        assert comp.adapter is not None

        await comp.cleanup()
        assert not comp.initialized

    async def test_component_process_status(self):
        """Test component process with status action."""
        from components.echonnxruntime import EchonnxruntimeComponent

        comp = EchonnxruntimeComponent()
        await comp.initialize({})

        result = await comp.process({'action': 'status'})
        assert result['status'] == 'success'
        assert 'adapter_status' in result

    async def test_component_capabilities(self):
        """Test component reports capabilities."""
        from components.echonnxruntime import EchonnxruntimeComponent

        comp = EchonnxruntimeComponent()
        caps = comp.get_capabilities()
        assert "onnx_inference" in caps
        assert "model_loading" in caps

    async def test_component_info(self):
        """Test component info reporting."""
        from components.echonnxruntime import EchonnxruntimeComponent

        comp = EchonnxruntimeComponent()
        await comp.initialize({})

        info = comp.get_info()
        assert info['name'] == 'echonnxruntime'
        assert info['version'] == '1.0.0'
        assert info['initialized'] is True

    async def test_relevance_dimensions(self):
        """Test Relevance Realization integration."""
        from components.echonnxruntime import EchonnxruntimeComponent

        comp = EchonnxruntimeComponent()
        dims = comp.get_relevance_dimensions()
        assert dims['procedural'] == 0.95
        assert dims['nomological'] == 0.85
        assert all(0 <= v <= 1 for v in dims.values())

    async def test_evolution_genome(self):
        """Test Evolution Engine genome reporting."""
        from components.echonnxruntime import EchonnxruntimeComponent

        comp = EchonnxruntimeComponent()
        await comp.initialize({'optimization_level': 'BASIC', 'batch_size': 4})

        genome = comp.get_evolution_genome()
        assert genome['optimization_level'] == 'BASIC'
        assert genome['batch_size'] == 4


class TestUnifiedInferencePipeline:
    """Tests for the unified inference pipeline."""

    async def test_pipeline_initialization(self):
        """Test pipeline initializes with component."""
        from components.echonnxruntime import EchonnxruntimeComponent
        from core.inference_pipeline import UnifiedInferencePipeline

        comp = EchonnxruntimeComponent()
        await comp.initialize({})

        pipeline = UnifiedInferencePipeline()
        result = await pipeline.initialize(comp)
        assert result is True
        assert pipeline._initialized

    async def test_pipeline_register_and_infer(self, tmp_path):
        """Test registering a model and running inference."""
        from components.echonnxruntime import EchonnxruntimeComponent
        from core.inference_pipeline import UnifiedInferencePipeline

        comp = EchonnxruntimeComponent()
        await comp.initialize({})

        pipeline = UnifiedInferencePipeline()
        await pipeline.initialize(comp)

        model_file = tmp_path / "pipeline_model.onnx"
        model_file.write_bytes(b"dummy")

        await pipeline.register_model(
            model_name="pipe_model",
            model_path=str(model_file),
            owner_component="coschemreasoner",
        )

        result = await pipeline.infer(
            model_name="pipe_model",
            inputs={"input": [[1.0, 2.0]]},
            source_component="coschemreasoner",
        )
        assert result.success
        assert result.model_name == "pipe_model"
        assert result.source_component == "coschemreasoner"

    async def test_pipeline_statistics(self, tmp_path):
        """Test pipeline statistics tracking."""
        from components.echonnxruntime import EchonnxruntimeComponent
        from core.inference_pipeline import UnifiedInferencePipeline

        comp = EchonnxruntimeComponent()
        await comp.initialize({})

        pipeline = UnifiedInferencePipeline()
        await pipeline.initialize(comp)

        model_file = tmp_path / "stats_model.onnx"
        model_file.write_bytes(b"dummy")

        await pipeline.register_model("stat_model", str(model_file), "test")
        await pipeline.infer("stat_model", {"input": [[1.0]]}, "test")
        await pipeline.infer("stat_model", {"input": [[2.0]]}, "test")

        stats = pipeline.get_statistics()
        assert stats['total_requests'] == 2
        assert stats['total_errors'] == 0
        assert stats['registered_models'] == 1

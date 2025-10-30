# echonnxruntime

ONNX Runtime for optimized machine learning model inference

## Status

**Active Implementation** - Integration wrapper for [ONNX Runtime](https://github.com/microsoft/onnxruntime)

## Overview

This component provides a lightweight ML inference wrapper that integrates ONNX Runtime 
into the ORRRG self-organizing core framework.

## Integration Type

**ML inference wrapper** - This is not a full clone of the upstream repository. Instead, it provides:
- Clean Python API for ORRRG integration
- Adapter pattern for the Self-Organizing Core
- Standard ComponentInterface implementation
- Efficient resource management

## Dependencies

This component requires:
- onnxruntime
- numpy

To install dependencies:
```bash
pip install onnxruntime numpy
```

## Usage

```python
from components.echonnxruntime import EchonnxruntimeComponent

# Initialize component
component = EchonnxruntimeComponent()
await component.initialize({})

# Use component functionality
result = await component.process({
    "type": "analysis",
    "data": {...}
})
```

## Integration with ORRRG

This component integrates with the ORRRG Self-Organizing Core through the standard ComponentInterface:

```python
class ComponentInterface(ABC):
    async def initialize(self, config: Dict[str, Any]) -> bool
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]
    async def cleanup(self) -> None
    def get_capabilities(self) -> List[str]
```

## Files

- `inference.py` - Implementation module
- `optimization.py` - Implementation module
- `session.py` - Implementation module

## Upstream Project

This component is based on [ONNX Runtime](https://github.com/microsoft/onnxruntime).

For full documentation of the upstream project, see the [upstream repository](https://github.com/microsoft/onnxruntime).

## License

This integration wrapper is part of ORRRG and licensed under AGPL v3.0.
The upstream project may have its own license - please refer to the upstream repository.

# oc-skintwin

OpenCog cognitive architecture for artificial general intelligence

## Status

**Active Implementation** - Integration wrapper for [OpenCog AtomSpace](https://github.com/opencog/atomspace)

## Overview

This component provides a lightweight Library wrapper that integrates OpenCog AtomSpace 
into the ORRRG self-organizing core framework.

## Integration Type

**Library wrapper** - This is not a full clone of the upstream repository. Instead, it provides:
- Clean Python API for ORRRG integration
- Adapter pattern for the Self-Organizing Core
- Standard ComponentInterface implementation
- Efficient resource management

## Dependencies

This component requires:
- opencog
- networkx

To install dependencies:
```bash
pip install opencog networkx
```

## Usage

```python
from components.oc-skintwin import Oc_SkintwinComponent

# Initialize component
component = Oc_SkintwinComponent()
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

- `atomspace_adapter.py` - Implementation module
- `reasoning.py` - Implementation module
- `knowledge.py` - Implementation module

## Upstream Project

This component is based on [OpenCog AtomSpace](https://github.com/opencog/atomspace).

For full documentation of the upstream project, see the [upstream repository](https://github.com/opencog/atomspace).

## License

This integration wrapper is part of ORRRG and licensed under AGPL v3.0.
The upstream project may have its own license - please refer to the upstream repository.

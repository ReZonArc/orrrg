# echopiler

Interactive compiler exploration and multi-language code analysis platform

## Status

**Active Implementation** - Integration wrapper for [Compiler Explorer](https://github.com/compiler-explorer/compiler-explorer)

## Overview

This component provides a lightweight API client wrapper that integrates Compiler Explorer 
into the ORRRG self-organizing core framework.

## Integration Type

**API client wrapper** - This is not a full clone of the upstream repository. Instead, it provides:
- Clean Python API for ORRRG integration
- Adapter pattern for the Self-Organizing Core
- Standard ComponentInterface implementation
- Efficient resource management

## Dependencies

This component requires:
- requests
- aiohttp

To install dependencies:
```bash
pip install requests aiohttp
```

## Usage

```python
from components.echopiler import EchopilerComponent

# Initialize component
component = EchopilerComponent()
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

- `client.py` - Implementation module
- `analyzer.py` - Implementation module
- `languages.py` - Implementation module

## Upstream Project

This component is based on [Compiler Explorer](https://github.com/compiler-explorer/compiler-explorer).

For full documentation of the upstream project, see the [upstream repository](https://github.com/compiler-explorer/compiler-explorer).

## License

This integration wrapper is part of ORRRG and licensed under AGPL v3.0.
The upstream project may have its own license - please refer to the upstream repository.

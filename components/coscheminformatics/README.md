# coscheminformatics

Chemical information processing and molecular analysis

## Status

**Active Implementation** - Integration wrapper for [RDKit](https://github.com/rdkit/rdkit)

## Overview

This component provides a lightweight Chemical informatics wrapper that integrates RDKit 
into the ORRRG self-organizing core framework.

## Integration Type

**Chemical informatics wrapper** - This is not a full clone of the upstream repository. Instead, it provides:
- Clean Python API for ORRRG integration
- Adapter pattern for the Self-Organizing Core
- Standard ComponentInterface implementation
- Efficient resource management

## Dependencies

This component requires:
- rdkit
- numpy
- pandas

To install dependencies:
```bash
pip install rdkit numpy pandas
```

## Usage

```python
from components.coscheminformatics import CoscheminformaticsComponent

# Initialize component
component = CoscheminformaticsComponent()
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

- `molecular.py` - Implementation module
- `properties.py` - Implementation module
- `fingerprints.py` - Implementation module

## Upstream Project

This component is based on [RDKit](https://github.com/rdkit/rdkit).

For full documentation of the upstream project, see the [upstream repository](https://github.com/rdkit/rdkit).

## License

This integration wrapper is part of ORRRG and licensed under AGPL v3.0.
The upstream project may have its own license - please refer to the upstream repository.

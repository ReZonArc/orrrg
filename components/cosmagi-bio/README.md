# cosmagi-bio

Genomic and proteomic research using OpenCog bioinformatics tools

## Status

**Active Implementation** - Integration wrapper for [OpenCog + BioPython](https://github.com/opencog/opencog)

## Overview

This component provides a lightweight Bioinformatics wrapper that integrates OpenCog + BioPython 
into the ORRRG self-organizing core framework.

## Integration Type

**Bioinformatics wrapper** - This is not a full clone of the upstream repository. Instead, it provides:
- Clean Python API for ORRRG integration
- Adapter pattern for the Self-Organizing Core
- Standard ComponentInterface implementation
- Efficient resource management

## Dependencies

This component requires:
- biopython
- opencog
- numpy

To install dependencies:
```bash
pip install biopython opencog numpy
```

## Usage

```python
from components.cosmagi-bio import Cosmagi_BioComponent

# Initialize component
component = Cosmagi_BioComponent()
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

- `genomics.py` - Implementation module
- `proteins.py` - Implementation module
- `bio_knowledge.py` - Implementation module

## Upstream Project

This component is based on [OpenCog + BioPython](https://github.com/opencog/opencog).

For full documentation of the upstream project, see the [upstream repository](https://github.com/opencog/opencog).

## License

This integration wrapper is part of ORRRG and licensed under AGPL v3.0.
The upstream project may have its own license - please refer to the upstream repository.

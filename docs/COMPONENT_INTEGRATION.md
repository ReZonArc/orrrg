# ORRRG Component Integration Guide

## Overview

ORRRG uses a **lightweight integration wrapper** approach for external components rather than cloning entire upstream repositories into the monorepo. This keeps the repository size manageable while providing clean integration points.

## Integration Philosophy

### Why Not Full Clones?

Cloning entire upstream repositories like Compiler Explorer (100K+ files), RDKit, or ONNX Runtime would:
- Bloat the monorepo to hundreds of megabytes or gigabytes
- Create maintenance challenges for tracking upstream changes
- Mix ORRRG-specific code with upstream project code
- Complicate dependency management

### Wrapper Approach

Instead, each component provides:
- **Clean Python API** for ORRRG integration
- **Adapter pattern** for the Self-Organizing Core
- **Standard ComponentInterface** implementation
- **Efficient resource management**
- **Proper attribution** to upstream projects

## Component Structure

Each component directory contains:

```
components/{component-name}/
├── README.md                # Full documentation with upstream links
├── __init__.py             # Component class implementing ComponentInterface
└── component_info.json     # Metadata about the component
```

### component_info.json

Provides structured metadata:

```json
{
  "name": "component-name",
  "description": "Brief description",
  "version": "0.1.0",
  "upstream_url": "https://github.com/upstream/repo",
  "upstream_project": "Upstream Project Name",
  "integration_type": "API client wrapper | Library wrapper | etc",
  "dependencies": ["package1", "package2"],
  "files": ["module1.py", "module2.py"],
  "status": "active"
}
```

## Component Interface

All components implement the standard interface:

```python
class ComponentInterface(ABC):
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the component with configuration."""
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the component."""
        
    async def cleanup(self) -> None:
        """Cleanup component resources."""
        
    def get_capabilities(self) -> List[str]:
        """Return list of component capabilities."""
        
    def get_info(self) -> Dict[str, Any]:
        """Return component information."""
```

## Current Components

### 1. echopiler
- **Description**: Interactive compiler exploration and multi-language code analysis
- **Upstream**: [Compiler Explorer](https://github.com/compiler-explorer/compiler-explorer)
- **Integration Type**: API client wrapper
- **Dependencies**: `requests`, `aiohttp`

Provides API access to godbolt.org for compilation and analysis without embedding the full Compiler Explorer codebase.

### 2. oc-skintwin
- **Description**: OpenCog cognitive architecture for AGI
- **Upstream**: [OpenCog AtomSpace](https://github.com/opencog/atomspace)
- **Integration Type**: Library wrapper
- **Dependencies**: `opencog`, `networkx`

Wraps OpenCog's AtomSpace for knowledge representation and reasoning within ORRRG.

### 3. cosmagi-bio
- **Description**: Genomic and proteomic research using OpenCog bioinformatics
- **Upstream**: [OpenCog](https://github.com/opencog/opencog) + BioPython
- **Integration Type**: Bioinformatics wrapper
- **Dependencies**: `biopython`, `opencog`, `numpy`

Combines BioPython's biological analysis with OpenCog's knowledge representation.

### 4. coscheminformatics
- **Description**: Chemical information processing and molecular analysis
- **Upstream**: [RDKit](https://github.com/rdkit/rdkit)
- **Integration Type**: Chemical informatics wrapper
- **Dependencies**: `rdkit`, `numpy`, `pandas`

Provides molecular structure handling and chemical property calculations using RDKit.

### 5. echonnxruntime
- **Description**: ONNX Runtime for optimized ML model inference
- **Upstream**: [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- **Integration Type**: ML inference wrapper
- **Dependencies**: `onnxruntime`, `numpy`

Wraps ONNX Runtime for efficient cross-platform ML model deployment.

### 6. coschemreasoner
- **Description**: Chemical reasoning system with reaction prediction
- **Upstream**: RDKit + ML Models
- **Integration Type**: Chemical reasoning engine
- **Dependencies**: `rdkit`, `scikit-learn`, `numpy`

Implements reaction prediction and retrosynthetic analysis using RDKit and ML.

### 7. oj7s3
- **Description**: Enhanced Open Journal Systems with autonomous agents
- **Status**: Fully cloned (existing implementation from ReZonArc/oj7s3)
- **Integration Type**: Full clone (pre-existing in monorepo)

### 8. esm-2-keras-esm2_t6_8m-v1-hyper-dev2
- **Description**: Protein/language model hypergraph mapping
- **Status**: Fully cloned (existing implementation from ReZonArc/esm-2-keras-esm2_t6_8m-v1-hyper)
- **Integration Type**: Full clone (pre-existing in monorepo)

## Installing Dependencies

Each component's dependencies can be installed via pip:

```bash
# Install dependencies for a specific component
cd components/echopiler
pip install -r <(echo "requests\naiohttp")

# Or install all component dependencies at once
pip install requests aiohttp opencog networkx biopython numpy pandas rdkit onnxruntime scikit-learn
```

## Adding New Components

To add a new component to ORRRG:

1. **Create component directory** in `components/`
2. **Write component README** with upstream attribution
3. **Implement ComponentInterface** in `__init__.py`
4. **Create component_info.json** with metadata
5. **Add dependencies** to project requirements if needed
6. **Test component** can be discovered and initialized
7. **Update configuration** in `config/orrrg_config.yaml` if needed

### Example Component Template

```python
"""
{component-name} - ORRRG Component

Brief description.
Integration wrapper for Upstream Project.
"""

from typing import Dict, List, Any
import logging

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

class MyComponent:
    """ORRRG component for specific functionality."""
    
    def __init__(self):
        self.config = {}
        self.initialized = False
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the component."""
        self.config = config
        # TODO: Initialize upstream library/service
        self.initialized = True
        return True
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the component."""
        if not self.initialized:
            raise RuntimeError("Component not initialized")
        # TODO: Implement processing logic
        return {"status": "success", "data": data}
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.initialized = False
    
    def get_capabilities(self) -> List[str]:
        """Return capabilities."""
        return ["capability1", "capability2"]
    
    def get_info(self) -> Dict[str, Any]:
        """Return component info."""
        return {
            "name": "component-name",
            "version": __version__,
            "initialized": self.initialized
        }

__all__ = ['MyComponent']
```

## Discovery and Registration

The Self-Organizing Core automatically discovers components at startup:

```python
from core import SelfOrganizingCore

async def main():
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    # Components are auto-discovered from components/ directory
    status = soc.get_system_status()
    print(f"Active components: {status['active_components']}")
```

Components are discovered by:
1. Scanning `components/` directory
2. Looking for directories with `__init__.py`
3. Attempting to import each component
4. Registering successfully imported components

## Benefits

✅ **Maintainable**: Small, focused integration code
✅ **Upgradeable**: Dependencies managed via pip
✅ **Clear Attribution**: Proper credit to upstream projects
✅ **Testable**: Integration logic can be tested independently
✅ **Flexible**: Easy to swap implementations or add new components
✅ **Documented**: Each component has comprehensive README

## License Compliance

Each component wrapper:
- Is licensed under AGPL v3.0 (ORRRG's license)
- Credits the upstream project in README
- Links to upstream repository and license
- Does not redistribute upstream code (uses as dependency)

This approach respects upstream project licenses while providing clean integration.

## References

- [ORRRG Architecture](ARCHITECTURE.md)
- [Self-Organizing Core](../core/self_organizing_core.py)
- [Component Adapters](../core/component_adapters.py)
- [Configuration Guide](../config/orrrg_config.yaml)

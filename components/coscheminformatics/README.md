Cheminformatics
===============

This is an early attempt to use OpenCog for cheminformatics, with specialized support for cosmetic chemistry applications.

## Features

### Core Cheminformatics
- Chemical element atom types (H, C, N, O, etc.)
- Chemical bond types (single, double, triple, aromatic)
- Molecule and reaction representations
- Basic chemical structure modeling

### Cosmetic Chemistry Specializations
- **Cosmetic Ingredient Types**: Active ingredients, preservatives, emulsifiers, humectants, surfactants, thickeners, emollients, antioxidants, UV filters, fragrances, colorants
- **Formulation Types**: Skincare, haircare, makeup, and fragrance formulations
- **Property Modeling**: pH, viscosity, stability, texture, color, scent, SPF properties
- **Interaction Analysis**: Compatibility, incompatibility, synergy, and antagonism relationships
- **Safety & Regulatory**: Safety assessments, allergen classifications, concentration limits
- **NEW: OpenCog Multiscale Optimization**: Advanced cognitive architecture integration featuring:
  - INCI-driven search space reduction with regulatory compliance
  - ECAN-inspired attention allocation for computational resource management
  - Multiscale constraint optimization from molecular to organ level
  - Probabilistic reasoning for uncertainty handling
  - Evolutionary optimization with multi-objective fitness functions

Building and Installing
=======================
To build the code, you will need to build and install the
[OpenCog AtomSpace](https://github.com/opencog/atomspace) first.
All of the pre-requistes listed there are sufficient to also build
this project. Building is as "usual":
```
    cd to project root dir
    mkdir build
    cd build
    cmake ..
    make -j
    sudo make install
    make -j test
```

## Documentation

- [Cosmetic Chemistry Guide](docs/COSMETIC_CHEMISTRY.md) - Comprehensive guide to cosmetic chemistry specializations
- **NEW: [OpenCog Multiscale Optimization](docs/OPENCOG_MULTISCALE_OPTIMIZATION.md)** - Advanced cognitive architecture integration for cosmeceutical formulation optimization

Examples
========
Examples can be found in the [examples](examples) directory.

### Cosmetic Chemistry Examples
- **Python**: 
  - [cosmetic_intro_example.py](examples/python/cosmetic_intro_example.py) - Basic cosmetic ingredient and formulation creation
  - [cosmetic_chemistry_example.py](examples/python/cosmetic_chemistry_example.py) - Advanced cosmetic formulation analysis
  - **NEW: OpenCog Multiscale Optimization Suite:**
    - [inci_optimizer.py](examples/python/inci_optimizer.py) - INCI-driven search space reduction algorithms
    - [attention_allocation.py](examples/python/attention_allocation.py) - ECAN-inspired attention allocation system
    - [multiscale_optimizer.py](examples/python/multiscale_optimizer.py) - Complete multiscale constraint optimization engine
    - [demo_opencog_multiscale.py](examples/python/demo_opencog_multiscale.py) - Comprehensive system demonstration
    - [test_multiscale_optimization.py](examples/python/test_multiscale_optimization.py) - Validation test suite
- **Scheme**: 
  - [cosmetic_formulation.scm](examples/scheme/cosmetic_formulation.scm) - Complex formulation modeling and compatibility analysis

### Basic Cheminformatics Examples  
- **Python**: [intro_example.py](examples/python/intro_example.py) - Basic molecule creation
- **Scheme**: [reaction.scm](examples/scheme/reaction.scm) - Chemical reaction modeling

If you run python virtualenv, and are experiencing issues with undefined
symbols, then try adding `/usr/local/lib/python3.11/dist-packages/`
to your `PYTHON_PATH` and adding `/usr/local/lib/opencog/` to your
`LD_LIBRARY_PATH`.

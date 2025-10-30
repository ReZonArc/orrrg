# ORRRG - Omnipotent Research and Reasoning Reactive Grid

> **🧬 Revolutionary Self-Evolving Integration System** - A cohesive self-organizing core that integrates multiple research and development approaches from diverse domains into a unified reactive framework with advanced evolutionary capabilities.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Async/Await](https://img.shields.io/badge/async-await-green.svg)](https://docs.python.org/3/library/asyncio.html)
[![Evolution Engine](https://img.shields.io/badge/evolution-enabled-brightgreen.svg)](#evolution-engine)

## 🎯 Overview

ORRRG is an **advanced self-evolving system** that seamlessly integrates and coordinates eight specialized research and development approaches with breakthrough evolutionary capabilities:

### 🧬 NEW: Evolution Engine v1.1

**Revolutionary Self-Evolutionary Capabilities:**
- **Genetic Programming**: Automatic evolution of component behaviors and architectures
- **Quantum-Inspired Algorithms**: Enhanced exploration using quantum superposition and entanglement
- **Emergent Behavior Synthesis**: Discovery and integration of novel behaviors from component interactions
- **Adaptive Learning**: Continuous self-improvement through experience-based optimization
- **Self-Modifying Code**: Safe autonomous modification of system components
- **Cross-Domain Knowledge Fusion**: Intelligent integration of insights across research domains

### Integrated Components

1. **[oj7s3](components/oj7s3)** - Enhanced Open Journal Systems with SKZ autonomous agents for academic publishing automation
2. **[echopiler](components/echopiler)** - Interactive compiler exploration and multi-language code analysis platform  
3. **[oc-skintwin](components/oc-skintwin)** - OpenCog cognitive architecture for artificial general intelligence
4. **[esm-2-keras-esm2_t6_8m-v1-hyper-dev2](components/esm-2-keras-esm2_t6_8m-v1-hyper-dev2)** - Protein/language model hypergraph mapping with transformer analysis
5. **[cosmagi-bio](components/cosmagi-bio)** - Genomic and proteomic research using OpenCog bioinformatics tools
6. **[coscheminformatics](components/coscheminformatics)** - Chemical information processing and molecular analysis
7. **[echonnxruntime](components/echonnxruntime)** - ONNX Runtime for optimized machine learning model inference
8. **[coschemreasoner](components/coschemreasoner)** - Chemical reasoning system with reaction prediction capabilities

## 🚀 Key Features

### 🧬 Evolution Engine Capabilities
- **Genetic Programming Layer**: Evolves system components through adaptive mutations and crossover
- **Quantum-Inspired Evolution**: Uses superposition and entanglement for enhanced exploration
- **Emergent Pattern Synthesis**: Automatically discovers and integrates novel interaction patterns
- **Adaptive Fitness Evaluation**: Learns optimal evaluation criteria for continuous improvement
- **Self-Aware Evolution**: Integration with autognosis for guided evolutionary processes
- **Real-time Performance Optimization**: Continuous monitoring and adaptive enhancement

### Self-Organization Capabilities
- **Dynamic Component Discovery**: Automatically discovers and integrates available components
- **Adaptive Resource Management**: Intelligent allocation and optimization of computational resources  
- **Cross-Domain Knowledge Graph**: Unified knowledge representation across all integrated domains
- **Emergent Behavior Coordination**: Components self-organize to solve complex multi-domain problems
- **Real-time Performance Optimization**: Continuous monitoring and adaptive improvement

### 🧠 Autognosis - Hierarchical Self-Image Building
- **Self-Aware AI System**: ORRRG can understand and optimize its own cognitive processes
- **Multi-Level Self-Modeling**: Builds hierarchical models of its own functioning at 5+ cognitive levels
- **Meta-Cognitive Insights**: Generates higher-order understanding about its own reasoning
- **Autonomous Self-Optimization**: Discovers and implements improvements through introspection
- **Recursive Self-Understanding**: Models its own modeling processes for deep self-awareness

*See [docs/AUTOGNOSIS.md](docs/AUTOGNOSIS.md) for detailed information about the self-awareness capabilities.*

### Integration Patterns
- **Bio-Chemical Pipeline**: Genomics → Chemical Analysis → Molecular Reasoning
- **ML Inference Pipeline**: Model Training → ONNX Optimization → Inference
- **Research Publication Pipeline**: Multi-domain Analysis → Automated Publishing
- **Cognitive Reasoning Pipeline**: Domain Knowledge → AtomSpace → AGI Reasoning

### Unified API Interface
- RESTful API for external integration
- Async/await Python interface for high-performance computing
- WebSocket support for real-time communication
- CLI interface for interactive exploration

## 📦 Installation

### Quick Start

```bash
# Clone the repository (components already integrated)
# Note: This repository uses Git LFS for large binary files
git clone https://github.com/ReZonArc/orrrg.git
cd orrrg

# Run the automated installation
./install.sh

# Activate the environment
source venv/bin/activate

# Start ORRRG interactively
python3 orrrg_main.py --mode interactive
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Initialize the system
python3 orrrg_main.py --help
```

## 🎮 Usage

### Interactive Mode
```bash
python3 orrrg_main.py --mode interactive

# Available commands:
orrrg> status        # Show system status  
orrrg> components    # List all components
orrrg> analyze bio   # Run biological analysis
orrrg> connect oj7s3 cosmagi-bio  # Create component connection
orrrg> optimize      # Run system optimization
orrrg> autognosis    # Show self-awareness status
orrrg> autognosis report    # Detailed self-analysis  
orrrg> autognosis insights  # Meta-cognitive insights

# 🧬 NEW: Evolution Commands
orrrg> evolve        # Show evolution engine status
orrrg> evolve oj7s3 performance adaptation  # Evolve specific component
orrrg> emergence     # Show emergent patterns discovered
orrrg> help          # Show all commands
orrrg> quit          # Exit system
```

### Daemon Mode
```bash
# Run as background service
python3 orrrg_main.py --mode daemon --verbose

# With specific components
python3 orrrg_main.py --mode daemon --components oj7s3,echopiler,oc-skintwin
```

### Batch Processing
```bash
# Run batch analysis with configuration
python3 orrrg_main.py --mode batch --config config/orrrg_config.yaml
```

### Python API
```python
import asyncio
from core import SelfOrganizingCore

async def main():
    # Initialize the self-organizing core with evolution engine
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    # Get system status
    status = soc.get_system_status()
    print(f"Active components: {status['active_components']}")
    
    # 🧬 NEW: Trigger component evolution
    evolution_result = await soc.trigger_targeted_evolution(
        'oj7s3', ['performance', 'integration', 'adaptation']
    )
    print(f"Evolution fitness: {evolution_result['fitness']}")
    
    # 🌱 NEW: Synthesize emergent behaviors
    emergent_patterns = await soc.evolution_engine.synthesize_emergent_behaviors()
    print(f"Discovered {len(emergent_patterns)} emergent patterns")
    
    # Queue cross-component analysis
    await soc.event_bus.put({
        "type": "cross_component_query",
        "query_type": "bio_chemical_analysis",
        "data": {"sequence": "MVLSPADKTNVKAAW..."}
    })
    
    # Cleanup
    await soc.shutdown()

asyncio.run(main())
```

### 🧬 Example: Evolution Engine Usage
```python
# Access evolution capabilities
evolution_status = await soc.evolution_engine.get_evolution_status()
print(f"Total genomes: {evolution_status['total_genomes']}")
print(f"Evolution running: {evolution_status['evolution_running']}")

# Evolve specific component
evolved_genome = await soc.evolution_engine.evolve_component(
    'cosmagi-bio', 
    current_state={'genomic_analysis': 0.8, 'learning_rate': 0.1},
    evolution_objectives=['performance', 'cognitive_enhancement']
)
print(f"New fitness: {evolved_genome.fitness_score}")
print(f"Generation: {evolved_genome.generation}")

# Synthesize emergent behaviors
emergent_patterns = await soc.evolution_engine.synthesize_emergent_behaviors()
for pattern in emergent_patterns:
    print(f"Pattern: {pattern.pattern_type}, Effectiveness: {pattern.effectiveness}")
```

### Example: Autognosis Self-Awareness
```python
# Access the self-aware capabilities
status = soc.get_autognosis_status()
print(f"Self-awareness level: {status['self_image_levels']}")
print(f"Generated insights: {status['total_insights']}")

# Run self-analysis cycle
cycle_results = await soc.autognosis.run_autognosis_cycle(soc) 
print(f"New meta-cognitive insights: {cycle_results['new_insights']}")

# Access hierarchical self-images
for level, self_image in soc.autognosis.current_self_images.items():
    print(f"Level {level}: {self_image.confidence:.2f} confidence")
```

## 🏗️ Architecture

### Self-Organizing Core (SOC)
The heart of ORRRG that provides:
- **Component Registry**: Dynamic discovery and management
- **Event Bus**: Async message passing between components
- **Knowledge Graph**: Cross-domain knowledge integration
- **Resource Manager**: Adaptive resource allocation
- **Performance Monitor**: Real-time optimization

### Component Integration
Each component exposes a standardized interface:
```python
class ComponentInterface(ABC):
    async def initialize(self, config: Dict[str, Any]) -> bool
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]
    async def cleanup(self) -> None
    def get_capabilities(self) -> List[str]
```

### Data Flow Pipelines
Components are connected through configurable data flow pipelines that enable:
- Multi-stage processing workflows
- Data transformation between domains
- Error handling and recovery
- Performance monitoring and optimization

## 🔧 Configuration

The system is configured through `config/orrrg_config.yaml`:

```yaml
system:
  name: "ORRRG - Omnipotent Research and Reasoning Reactive Grid"
  max_concurrent_tasks: 10
  heartbeat_interval: 30

components:
  oj7s3:
    enabled: true
    priority: 8
    resource_allocation:
      cpu_limit: 2
      memory_limit: "4GB"

integration_patterns:
  bio_chemical_pipeline:
    components: [cosmagi-bio, coscheminformatics, coschemreasoner]
    data_flow: [...]

self_organization:
  adaptation_enabled: true
  learning_rate: 0.01
  optimization_interval: 300
```

## 📊 Monitoring and Observability

### Performance Metrics
- Component utilization and response times
- Memory and CPU usage across components
- Data flow throughput and error rates
- Knowledge graph growth and query performance

### Health Monitoring
- Component health checks and status reporting
- Automatic failure detection and recovery
- Resource constraint monitoring
- Performance trend analysis

### Adaptive Optimization
- Dynamic resource reallocation based on load
- Component priority adjustment
- Pipeline optimization based on usage patterns
- Predictive scaling based on historical data

## 🌟 Advanced Capabilities

### Cross-Domain Analysis
```python
# Example: Protein sequence → Chemical analysis → Publication
result = await soc.process_pipeline([
    {"component": "esm-2-keras-esm2_t6_8m-v1-hyper-dev2", "data": {"sequence": "..."}},
    {"component": "coscheminformatics", "transform": "protein_to_chemical"},
    {"component": "coschemreasoner", "transform": "chemical_to_insights"},
    {"component": "oj7s3", "transform": "insights_to_manuscript"}
])
```

### Cognitive Integration
```python
# Route domain-specific knowledge through OpenCog
await soc.route_to_atomspace([
    "chemical_knowledge_from_reasoning",
    "biological_insights_from_genomics", 
    "ml_patterns_from_transformers"
])
```

### Emergent Behavior
The system demonstrates emergent capabilities through component interaction:
- **Automated Research Hypothesis Generation**: Bio + Chemical + Cognitive reasoning
- **Code-Guided Scientific Computing**: Compiler + ML inference optimization  
- **Intelligent Publication Workflows**: Multi-domain analysis → Autonomous writing

## 🔬 Research Applications

### Computational Biology
- Protein structure prediction with chemical validation
- Genomic sequence analysis with reasoning-guided interpretation
- Automated literature review and hypothesis generation

### Chemical Informatics  
- Molecular property prediction with cognitive reasoning
- Reaction pathway optimization using compiler-like analysis
- Automated chemical knowledge extraction from publications

### AI/ML Research
- Hypergraph analysis of transformer architectures
- Cognitive model validation against ML predictions
- Automated model optimization and deployment

## 🤝 Contributing

We welcome contributions to extend ORRRG's capabilities:

1. **Component Integration**: Add new research tools and frameworks
2. **Pipeline Development**: Create new cross-domain analysis workflows  
3. **Self-Organization**: Improve adaptive algorithms and optimization
4. **Domain Expertise**: Contribute domain-specific knowledge and ontologies

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation

Comprehensive technical documentation is available in the `docs/` directory:

### Core Documentation
- **[Technical Architecture](docs/ARCHITECTURE.md)** - Comprehensive system architecture with Mermaid diagrams covering:
  - High-level architecture and component interactions
  - Data flow pipelines and integration patterns
  - Evolution engine architecture
  - Autognosis system hierarchy
  - Deployment and security architecture
  
- **[Formal Specification (Z++)](docs/FORMAL_SPECIFICATION_ZPP.md)** - Mathematical specification using Z++ notation:
  - Complete type system and schemas
  - State transitions and operations
  - Invariants and safety properties
  - Behavioral and temporal specifications
  - Refinement proofs

### Specialized Documentation
- **[Autognosis System](docs/AUTOGNOSIS.md)** - Hierarchical self-awareness capabilities
- **[Holistic Metamodel](docs/HOLISTIC_METAMODEL.md)** - Eric Schwarz's organizational theory implementation

## 🙏 Acknowledgments

ORRRG integrates and builds upon the excellent work of multiple open-source projects:
- [Open Journal Systems](https://pkp.sfu.ca/ojs/) for academic publishing
- [Compiler Explorer](https://godbolt.org/) for interactive code analysis
- [OpenCog](https://opencog.org/) for cognitive architecture
- [Hugging Face Transformers](https://huggingface.co/transformers/) for ML models
- [ONNX Runtime](https://onnxruntime.ai/) for ML inference optimization

## 🚀 Future Roadmap

- **Extended Domain Integration**: Physics, Materials Science, Economics
- **Advanced Self-Organization**: Reinforcement learning for optimization
- **Distributed Computing**: Multi-node cluster deployment
- **Real-time Collaboration**: Multi-user research environments
- **API Ecosystem**: Plugin architecture for third-party extensions

---

**Ready to organize and reason across all domains of knowledge!** 🧠⚡🔬

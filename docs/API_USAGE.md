# ORRRG API Usage Guide

This guide provides comprehensive examples and best practices for using the ORRRG (Omnipotent Research and Reasoning Reactive Grid) API.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core API](#core-api)
3. [Relevance Realization Ennead](#relevance-realization-ennead)
4. [Evolution Engine](#evolution-engine)
5. [Autognosis Self-Awareness](#autognosis-self-awareness)
6. [Component Integration](#component-integration)
7. [Best Practices](#best-practices)
8. [Common Patterns](#common-patterns)

## Quick Start

### Basic Initialization

```python
import asyncio
from core import SelfOrganizingCore

async def main():
    # Initialize the self-organizing core
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    # Get system status
    status = soc.get_system_status()
    print(f"System running: {status['running']}")
    print(f"Active components: {status['active_components']}")
    
    # Cleanup
    await soc.shutdown()

asyncio.run(main())
```

### Running from Command Line

```bash
# Interactive mode
python3 orrrg_main.py --mode interactive

# Daemon mode
python3 orrrg_main.py --mode daemon --verbose

# Batch processing
python3 orrrg_main.py --mode batch --config config/orrrg_config.yaml
```

## Core API

### SelfOrganizingCore

The main entry point for all ORRRG functionality.

#### Methods

**`async initialize() -> None`**
Initializes the entire system including all subsystems.

```python
soc = SelfOrganizingCore()
await soc.initialize()
```

**`get_system_status() -> Dict[str, Any]`**
Returns current system status and metrics.

```python
status = soc.get_system_status()
# Returns:
# {
#     'running': bool,
#     'active_components': int,
#     'components': Dict[str, ComponentInfo],
#     'knowledge_graph_size': int
# }
```

**`async shutdown() -> None`**
Gracefully shuts down all subsystems.

```python
await soc.shutdown()
```

**`async trigger_targeted_evolution(component_name: str, objectives: List[str]) -> Dict`**
Triggers evolution for a specific component.

```python
result = await soc.trigger_targeted_evolution(
    'cosmagi-bio',
    ['performance', 'accuracy', 'cognitive_enhancement']
)
print(f"New fitness: {result['fitness']}")
```

## Relevance Realization Ennead

The nine-dimensional framework for optimal relevance realization.

### Core Concepts

**Triad I - Ways of Knowing:**
- Propositional (knowing-that): Facts and theories
- Procedural (knowing-how): Skills and operations
- Perspectival (knowing-as): Salience and framing
- Participatory (knowing-by-being): Identity transformation

**Triad II - Orders of Understanding:**
- Nomological: How things work (causal mechanisms)
- Normative: What matters (values and significance)
- Narrative: How things develop (temporal trajectories)

**Triad III - Practices of Wisdom:**
- Morality: Virtue cultivation
- Meaning: Existential coherence
- Mastery: Excellence and flow

### API Methods

**`get_ennead_status() -> Dict[str, Any]`**
Returns complete nine-dimensional status.

```python
status = soc.relevance_realization.get_ennead_status()

print(f"Integration Score: {status['ennead_integration_score']:.3f}")
print(f"Triad Coherence: {status['triad_coherence']}")

# Access individual dimensions
print(f"Propositional Knowledge: {status['propositional_knowledge_count']}")
print(f"Nomological Mechanisms: {status['nomological_mechanisms']}")
print(f"Morality Considerations: {status['morality_ethical_considerations']}")
```

**`async optimize_relevance_realization(context: Dict[str, Any]) -> Dict[str, Any]`**
Optimizes relevance for a given context across all nine dimensions.

```python
context = {
    'task_type': 'scientific_research',
    'domain': 'biology',
    'requirements': ['genomic_analysis', 'chemical_analysis', 'reasoning'],
    'objectives': ['discovery', 'validation']
}

result = await soc.relevance_realization.optimize_relevance_realization(context)

# Access results
print(f"Relevance Score: {result['relevance_score']:.3f}")
print(f"Salient Components: {result['salient_components']}")
print(f"Integrated Knowledge: {result['integrated_knowledge']}")
print(f"Integrated Understanding: {result['integrated_understanding']}")
```

**`async realize_perspective_shift(from_component: str, to_component: str) -> Dict`**
Realizes a perspective shift between components, enabling gnostic transformations.

```python
shift = await soc.relevance_realization.realize_perspective_shift(
    'cosmagi-bio',
    'oc-skintwin'
)

if shift['is_gnostic_transformation']:
    print("Identity-level transformation achieved!")
    print(f"Participatory knowing increased by {shift['participatory_increase']:.2%}")
```

**`async generate_ennead_insight() -> str`**
Generates meta-level insights about the system's relevance realization state.

```python
insight = await soc.relevance_realization.generate_ennead_insight()
print(f"System Insight: {insight}")
```

### Example: Multi-Domain Research Workflow

```python
async def research_workflow():
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    # 1. Define research problem
    problem = {
        'task_type': 'protein_structure_analysis',
        'domain': 'computational_biology',
        'requirements': [
            'genomic_sequence_analysis',
            'chemical_property_prediction',
            'ml_inference',
            'cognitive_reasoning'
        ]
    }
    
    # 2. Optimize relevance realization
    relevance = await soc.relevance_realization.optimize_relevance_realization(problem)
    
    print(f"Identified {len(relevance['salient_components'])} relevant components")
    
    # 3. Execute through salient components
    for comp_name in relevance['salient_components']:
        print(f"Processing via: {comp_name}")
        # Component-specific processing here
    
    # 4. Generate insights
    insight = await soc.relevance_realization.generate_ennead_insight()
    print(f"Research Insight: {insight}")
    
    await soc.shutdown()

asyncio.run(research_workflow())
```

## Evolution Engine

Self-evolutionary capabilities with genetic programming and quantum-inspired algorithms.

### API Methods

**`async get_evolution_status() -> Dict[str, Any]`**
Returns evolution engine status.

```python
status = await soc.evolution_engine.get_evolution_status()

print(f"Total Genomes: {status['total_genomes']}")
print(f"Evolution Running: {status['evolution_running']}")
print(f"Current Generation: {status['current_generation']}")
```

**`async evolve_component(component_name: str, current_state: Dict, evolution_objectives: List[str]) -> EvolutionaryGenome`**
Evolves a specific component.

```python
evolved = await soc.evolution_engine.evolve_component(
    'echopiler',
    current_state={'compilation_speed': 0.8, 'optimization_level': 2},
    evolution_objectives=['performance', 'accuracy', 'adaptation']
)

print(f"Fitness: {evolved.fitness_score:.3f}")
print(f"Generation: {evolved.generation}")
print(f"Mutations: {len(evolved.mutations)}")
```

**`async synthesize_emergent_behaviors() -> List[EmergentPattern]`**
Discovers and synthesizes emergent behavioral patterns.

```python
patterns = await soc.evolution_engine.synthesize_emergent_behaviors()

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}")
    print(f"Components: {pattern.participating_components}")
    print(f"Effectiveness: {pattern.effectiveness:.3f}")
```

### Example: Adaptive Component Evolution

```python
async def adaptive_evolution():
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    # Evolve all components over multiple generations
    for generation in range(10):
        print(f"\n=== Generation {generation} ===")
        
        for comp_name in soc.components.keys():
            result = await soc.trigger_targeted_evolution(
                comp_name,
                ['performance', 'integration', 'cognitive_enhancement']
            )
            
            print(f"{comp_name}: fitness={result['fitness']:.3f}")
        
        # Synthesize emergent behaviors
        patterns = await soc.evolution_engine.synthesize_emergent_behaviors()
        print(f"Discovered {len(patterns)} emergent patterns")
    
    await soc.shutdown()

asyncio.run(adaptive_evolution())
```

## Autognosis Self-Awareness

Hierarchical self-image building and meta-cognitive capabilities.

### API Methods

**`get_autognosis_status() -> Dict[str, Any]`**
Returns autognosis system status.

```python
status = soc.get_autognosis_status()

print(f"System Status: {status['system_status']}")
print(f"Self-Image Levels: {status['self_image_levels']}")
print(f"Total Insights: {status['total_insights']}")
print(f"Pending Optimizations: {status['pending_optimizations']}")
```

**`async run_autognosis_cycle(soc: SelfOrganizingCore) -> Dict`**
Runs a complete self-analysis cycle.

```python
cycle_result = await soc.autognosis.run_autognosis_cycle(soc)

print(f"New Insights Generated: {cycle_result['new_insights']}")
print(f"Self-Image Levels Updated: {cycle_result['updated_levels']}")

# Access hierarchical self-images
for level, self_image in soc.autognosis.current_self_images.items():
    print(f"Level {level}: Confidence={self_image.confidence:.2f}")
```

### Example: Self-Optimization Loop

```python
async def self_optimization_loop():
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    # Run continuous self-optimization
    for cycle in range(5):
        print(f"\n=== Self-Optimization Cycle {cycle} ===")
        
        # 1. Run autognosis cycle
        result = await soc.autognosis.run_autognosis_cycle(soc)
        
        # 2. Review generated insights
        status = soc.get_autognosis_status()
        for insight in status.get('recent_insights', [])[-3:]:
            print(f"Insight: {insight['type']} - {insight['description']}")
        
        # 3. Apply top optimizations
        for opt in status.get('top_optimizations', [])[:2]:
            print(f"Applying: {opt['type']} on {opt['target']}")
            # Apply optimization here
        
        # 4. Evolve based on insights
        if status.get('top_optimizations'):
            opt = status['top_optimizations'][0]
            await soc.trigger_targeted_evolution(
                opt['target'],
                [opt['type'], 'self_improvement']
            )
    
    await soc.shutdown()

asyncio.run(self_optimization_loop())
```

## Component Integration

### Discovering Components

Components are automatically discovered from the `components/` directory.

```python
# List all discovered components
status = soc.get_system_status()
for name, info in status['components'].items():
    print(f"{name}: {info.status}")
    print(f"  Capabilities: {info.capabilities}")
    print(f"  Path: {info.path}")
```

### Cross-Component Communication

```python
# Queue cross-component event
event = {
    'type': 'cross_component_query',
    'query_type': 'bio_chemical_analysis',
    'source_component': 'cosmagi-bio',
    'target_component': 'coscheminformatics',
    'data': {
        'sequence': 'MVLSPADKTNVKAAW...',
        'analysis_type': 'protein_structure'
    }
}

await soc.event_bus.put(event)
```

### Creating Component Adapters

```python
from core import ComponentInterface

class MyComponentAdapter(ComponentInterface):
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize your component."""
        self.config = config
        # Setup component
        return True
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through your component."""
        # Implement processing logic
        result = {'status': 'success', 'output': '...'}
        return result
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Cleanup logic
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities."""
        return ['analysis', 'processing', 'reasoning']
```

## Best Practices

### 1. Always Use Context Managers or Explicit Cleanup

```python
async def good_practice():
    soc = SelfOrganizingCore()
    try:
        await soc.initialize()
        # Your code here
    finally:
        await soc.shutdown()
```

### 2. Leverage Relevance Realization for Task Routing

```python
# Let the system determine most relevant components
context = {'task_type': 'your_task', 'requirements': ['req1', 'req2']}
result = await soc.relevance_realization.optimize_relevance_realization(context)

# Use identified salient components
for comp in result['salient_components']:
    # Process through component
    pass
```

### 3. Combine Evolution with Relevance Realization

```python
# Optimize relevance first
relevance = await soc.relevance_realization.optimize_relevance_realization(context)

# Evolve salient components for better performance
for comp_name in relevance['salient_components'][:3]:
    await soc.trigger_targeted_evolution(
        comp_name,
        ['performance', 'accuracy']
    )
```

### 4. Use Autognosis for Continuous Improvement

```python
# Periodic self-reflection
async def monitor_and_improve():
    while running:
        cycle_result = await soc.autognosis.run_autognosis_cycle(soc)
        
        # Apply insights
        status = soc.get_autognosis_status()
        for opt in status.get('top_optimizations', []):
            # Apply optimization
            pass
        
        await asyncio.sleep(300)  # Every 5 minutes
```

### 5. Handle Errors Gracefully

```python
try:
    result = await soc.relevance_realization.optimize_relevance_realization(context)
except Exception as e:
    logger.error(f"Relevance optimization failed: {e}")
    # Fallback strategy
    result = await fallback_processing(context)
```

## Common Patterns

### Pattern 1: Multi-Stage Pipeline

```python
async def multi_stage_pipeline(data):
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    # Stage 1: Genomic Analysis
    genomic_result = await process_component(soc, 'cosmagi-bio', data)
    
    # Stage 2: Chemical Analysis
    chemical_result = await process_component(
        soc, 'coscheminformatics', genomic_result
    )
    
    # Stage 3: Reasoning
    final_result = await process_component(
        soc, 'oc-skintwin', chemical_result
    )
    
    await soc.shutdown()
    return final_result
```

### Pattern 2: Adaptive Workflow

```python
async def adaptive_workflow(initial_data):
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    data = initial_data
    max_iterations = 10
    
    for iteration in range(max_iterations):
        # Determine next step via relevance realization
        context = {'current_data': data, 'iteration': iteration}
        relevance = await soc.relevance_realization.optimize_relevance_realization(context)
        
        # Process through most relevant component
        best_component = relevance['salient_components'][0]
        data = await process_component(soc, best_component, data)
        
        # Check if goal achieved
        if goal_achieved(data):
            break
        
        # Evolve components based on performance
        await soc.trigger_targeted_evolution(
            best_component,
            ['performance', 'accuracy']
        )
    
    await soc.shutdown()
    return data
```

### Pattern 3: Self-Optimizing System

```python
async def self_optimizing_system():
    soc = SelfOrganizingCore()
    await soc.initialize()
    
    async def optimization_loop():
        while True:
            # Self-reflection
            cycle = await soc.autognosis.run_autognosis_cycle(soc)
            
            # Apply insights
            status = soc.get_autognosis_status()
            for opt in status.get('top_optimizations', []):
                await apply_optimization(soc, opt)
            
            # Synthesize emergent patterns
            patterns = await soc.evolution_engine.synthesize_emergent_behaviors()
            for pattern in patterns:
                if pattern.effectiveness > 0.7:
                    await integrate_pattern(soc, pattern)
            
            await asyncio.sleep(300)
    
    # Run optimization loop in background
    optimization_task = asyncio.create_task(optimization_loop())
    
    # Run main application
    await run_main_application(soc)
    
    # Cleanup
    optimization_task.cancel()
    await soc.shutdown()
```

## Interactive Mode Commands

When running in interactive mode:

```bash
python3 orrrg_main.py --mode interactive
```

Available commands:

- `status` - Show system status
- `components` - List all components
- `analyze TYPE` - Perform analysis (bio, chemical, ml, research)
- `connect A B` - Create connection between components
- `optimize` - Run system optimization
- `autognosis` - Show self-awareness status
- `autognosis report` - Detailed self-analysis
- `autognosis insights` - Meta-cognitive insights
- `evolve` - Show evolution status
- `evolve COMPONENT [OBJECTIVES]` - Evolve specific component
- `emergence` - Show emergent patterns
- `relevance` - Show Ennead integration status
- `relevance insight` - Generate relevance insight
- `relevance shift A B` - Realize perspective shift
- `help` - Show all commands
- `quit` - Exit system

## Configuration

Configuration is loaded from `config/orrrg_config.yaml`:

```yaml
system:
  name: "ORRRG"
  max_concurrent_tasks: 10
  heartbeat_interval: 30

components:
  cosmagi-bio:
    enabled: true
    priority: 8
    resource_allocation:
      cpu_limit: 2
      memory_limit: "4GB"

integration_patterns:
  bio_chemical_pipeline:
    components: [cosmagi-bio, coscheminformatics, coschemreasoner]
    data_flow: [...]
```

## Troubleshooting

### Issue: Components not discovered

```python
# Check component path
print(f"Components path: {soc.components_path}")
print(f"Components found: {len(soc.components)}")

# Manual component discovery
await soc.discover_components()
```

### Issue: Evolution not working

```python
# Check evolution engine status
status = await soc.evolution_engine.get_evolution_status()
print(f"Evolution running: {status['evolution_running']}")

# Verify component exists
if component_name in soc.components:
    result = await soc.trigger_targeted_evolution(component_name, ['performance'])
```

### Issue: Relevance optimization slow

```python
# Simplify context
context = {
    'task_type': 'simple_analysis',  # More specific
    'requirements': ['analysis']  # Fewer requirements
}

result = await soc.relevance_realization.optimize_relevance_realization(context)
```

## Further Reading

- [Technical Architecture](docs/ARCHITECTURE.md) - System architecture details
- [Formal Specification](docs/FORMAL_SPECIFICATION_ZPP.md) - Z++ specifications
- [Autognosis Documentation](docs/AUTOGNOSIS.md) - Self-awareness system
- [Relevance Realization](docs/RELEVANCE_REALIZATION.md) - Ennead framework

## Support

For issues and questions:
- GitHub Issues: https://github.com/ReZonArc/orrrg/issues
- Documentation: https://github.com/ReZonArc/orrrg/tree/main/docs

---

**Last Updated**: January 2026  
**ORRRG Version**: 1.2.0

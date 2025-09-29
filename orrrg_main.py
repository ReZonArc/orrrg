#!/usr/bin/env python3
"""
ORRRG - Omnipotent Research and Reasoning Reactive Grid
=======================================================

Main entry point for the integrated self-organizing system that combines
all research approaches into a cohesive reactive framework.

Usage:
    python orrrg_main.py [OPTIONS]

Options:
    --config PATH       Configuration file path
    --mode MODE         Execution mode (interactive, daemon, batch)
    --components LIST   Comma-separated list of components to enable
    --verbose           Enable verbose logging
    --help              Show this help message
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

from core import SelfOrganizingCore


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('orrrg.log')
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ORRRG - Omnipotent Research and Reasoning Reactive Grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python orrrg_main.py --mode interactive
    python orrrg_main.py --mode daemon --components oj7s3,echopiler
    python orrrg_main.py --verbose --config config/orrrg.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--mode',
        choices=['interactive', 'daemon', 'batch'],
        default='interactive',
        help='Execution mode (default: interactive)'
    )
    
    parser.add_argument(
        '--components',
        type=str,
        help='Comma-separated list of components to enable'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


async def interactive_mode(soc: SelfOrganizingCore) -> None:
    """Run in interactive mode with user commands."""
    print("\n" + "="*60)
    print("ORRRG - Omnipotent Research and Reasoning Reactive Grid")
    print("="*60)
    print("Type 'help' for available commands, 'quit' to exit")
    
    while True:
        try:
            command = input("\norrrg> ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                break
            elif command == 'help':
                await show_help()
            elif command == 'status':
                await show_status(soc)
            elif command == 'components':
                await show_components(soc)
            elif command.startswith('analyze '):
                await handle_analysis_command(soc, command)
            elif command.startswith('connect '):
                await handle_connection_command(soc, command)
            elif command == 'optimize':
                await handle_optimization_command(soc)
            elif command == 'autognosis':
                await handle_autognosis_command(soc)
            elif command.startswith('autognosis '):
                await handle_autognosis_command(soc, command)
            elif command == 'evolve':
                await handle_evolution_command(soc)
            elif command.startswith('evolve '):
                await handle_evolution_command(soc, command)
            elif command == 'emergence':
                await handle_emergence_command(soc)
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit gracefully.")
        except EOFError:
            break


async def show_help() -> None:
    """Show available commands."""
    help_text = """
Available Commands:
  status        - Show system status
  components    - List all discovered components
  analyze TYPE  - Perform analysis (bio, chemical, ml, research)
  connect A B   - Create connection between components A and B
  optimize      - Run system optimization
  autognosis    - Show autognosis (self-awareness) status
  autognosis STATUS/REPORT/INSIGHTS - Get detailed autognosis information
  evolve        - Show evolution engine status
  evolve COMPONENT [OBJECTIVES] - Trigger targeted evolution for component
  emergence     - Show emergent patterns discovered by evolution
  help          - Show this help message
  quit          - Exit the system
"""
    print(help_text)


async def show_status(soc: SelfOrganizingCore) -> None:
    """Show system status."""
    status = soc.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Running: {status['running']}")
    print(f"  Active Components: {status['active_components']}")
    print(f"  Knowledge Graph Size: {status['knowledge_graph_size']}")
    print(f"  Total Components: {len(status['components'])}")


async def show_components(soc: SelfOrganizingCore) -> None:
    """Show discovered components."""
    status = soc.get_system_status()
    print(f"\nDiscovered Components ({len(status['components'])}):")
    
    for name, info in status['components'].items():
        capabilities = ', '.join(info['capabilities'][:3])  # Show first 3 capabilities
        if len(info['capabilities']) > 3:
            capabilities += f" (+{len(info['capabilities']) - 3} more)"
        
        print(f"  {name:25} [{info['status']:>9}] - {capabilities}")


async def handle_analysis_command(soc: SelfOrganizingCore, command: str) -> None:
    """Handle analysis commands."""
    parts = command.split()
    if len(parts) < 2:
        print("Usage: analyze <type> where type is: bio, chemical, ml, research")
        return
    
    analysis_type = parts[1]
    
    # Queue analysis event
    event = {
        "type": "cross_component_query",
        "query_type": f"{analysis_type}_analysis",
        "timestamp": asyncio.get_event_loop().time()
    }
    
    await soc.event_bus.put(event)
    print(f"Queued {analysis_type} analysis request")


async def handle_connection_command(soc: SelfOrganizingCore, command: str) -> None:
    """Handle component connection commands."""
    parts = command.split()
    if len(parts) < 3:
        print("Usage: connect <component1> <component2>")
        return
    
    comp1, comp2 = parts[1], parts[2]
    
    if comp1 in soc.components and comp2 in soc.components:
        print(f"Creating connection between {comp1} and {comp2}")
        # Implementation would create actual connections
    else:
        print(f"One or both components not found: {comp1}, {comp2}")


async def handle_optimization_command(soc: SelfOrganizingCore) -> None:
    """Handle optimization command."""
    event = {
        "type": "adaptive_optimization",
        "timestamp": asyncio.get_event_loop().time()
    }
    
    await soc.event_bus.put(event)
    print("Queued system optimization request")


async def handle_autognosis_command(soc: SelfOrganizingCore, command: str = "autognosis") -> None:
    """Handle autognosis commands."""
    parts = command.split()
    
    if len(parts) == 1:
        # Basic autognosis status
        await show_autognosis_status(soc)
    elif len(parts) == 2:
        subcommand = parts[1].lower()
        if subcommand == "status":
            await show_autognosis_status(soc)
        elif subcommand == "report":
            await show_autognosis_report(soc)
        elif subcommand == "insights":
            await show_autognosis_insights(soc)
        else:
            print(f"Unknown autognosis subcommand: {subcommand}")
            print("Available: status, report, insights")
    else:
        print("Usage: autognosis [status|report|insights]")


async def show_autognosis_status(soc: SelfOrganizingCore) -> None:
    """Show basic autognosis status."""
    if not hasattr(soc, 'autognosis') or not soc.autognosis:
        print("Autognosis system not initialized")
        return
    
    status = soc.get_autognosis_status()
    
    print(f"\n🧠 Autognosis - Hierarchical Self-Image Building System")
    print("=" * 55)
    print(f"Status: {status.get('system_status', 'unknown')}")
    print(f"Self-Image Levels: {status.get('self_image_levels', 0)}")
    print(f"Total Insights Generated: {status.get('total_insights', 0)}")
    print(f"Pending Optimizations: {status.get('pending_optimizations', 0)}")
    
    # Show recent insights
    recent_insights = status.get('recent_insights', [])
    if recent_insights:
        print(f"\nRecent Insights ({len(recent_insights)}):")
        for insight in recent_insights[-3:]:  # Show last 3
            print(f"  • {insight['type']}: {insight['description'][:60]}...")
    
    # Show top optimizations
    top_opts = status.get('top_optimizations', [])
    if top_opts:
        print(f"\nTop Optimization Opportunities:")
        for opt in top_opts[:2]:  # Show top 2
            print(f"  • {opt['type']} on {opt['target']} (priority: {opt['priority']})")


async def show_autognosis_report(soc: SelfOrganizingCore) -> None:
    """Show detailed autognosis report."""
    if not hasattr(soc, 'autognosis') or not soc.autognosis:
        print("Autognosis system not initialized")
        return
    
    report = soc.get_autognosis_status()
    
    print(f"\n🧠 Detailed Autognosis Report")
    print("=" * 50)
    print(f"Timestamp: {report.get('timestamp')}")
    print(f"System Status: {report.get('system_status')}")
    
    # Self-images
    self_images = report.get('self_images', {})
    print(f"\nHierarchical Self-Images ({len(self_images)} levels):")
    for level, image_info in self_images.items():
        print(f"  Level {level}: Confidence {image_info['confidence']:.2f}, "
              f"{image_info['reflections_count']} reflections "
              f"[{image_info['image_hash']}]")
    
    # All insights
    insights = report.get('recent_insights', [])
    print(f"\nRecent Meta-Cognitive Insights ({len(insights)}):")
    for insight in insights:
        print(f"  • [{insight['type']}] {insight['description']}")
        print(f"    Confidence: {insight['confidence']:.2f}, "
              f"Time: {insight['timestamp']}")
    
    # All optimizations
    optimizations = report.get('top_optimizations', [])
    print(f"\nOptimization Opportunities ({len(optimizations)}):")
    for opt in optimizations:
        print(f"  • {opt['type']} targeting {opt['target']}")
        print(f"    Expected improvement: {opt['expected_improvement']:.1%}, "
              f"Priority: {opt['priority']}")


async def show_autognosis_insights(soc: SelfOrganizingCore) -> None:
    """Show autognosis insights and self-awareness metrics."""
    if not hasattr(soc, 'autognosis') or not soc.autognosis:
        print("Autognosis system not initialized")
        return
    
    print(f"\n🧠 Autognosis Self-Awareness Analysis")
    print("=" * 50)
    
    # Get current self-images
    current_images = soc.autognosis.current_self_images
    
    for level, self_image in current_images.items():
        print(f"\nLevel {level} Self-Image (Confidence: {self_image.confidence:.2f}):")
        
        # Show behavioral patterns
        if self_image.behavioral_patterns:
            print("  Behavioral Patterns:")
            for pattern, values in self_image.behavioral_patterns.items():
                avg_value = sum(values) / len(values) if values else 0
                print(f"    • {pattern}: {avg_value:.3f} (trend over {len(values)} observations)")
        
        # Show performance metrics
        if self_image.performance_metrics:
            print("  Performance Metrics:")
            for metric, value in self_image.performance_metrics.items():
                status = "Good" if value > 0.7 else "Moderate" if value > 0.4 else "Needs Attention"
                print(f"    • {metric}: {value:.3f} ({status})")
        
        # Show meta-reflections
        if self_image.meta_reflections:
            print("  Meta-Reflections:")
            for reflection in self_image.meta_reflections:
                print(f"    • {reflection}")
    
    # Show self-awareness indicators
    if current_images:
        highest_level = max(current_images.keys())
        if highest_level >= 2:
            awareness_indicators = current_images[highest_level].cognitive_processes.get('self_awareness_indicators', {})
            if awareness_indicators:
                print(f"\nSelf-Awareness Indicators:")
                for indicator, score in awareness_indicators.items():
                    bar_length = int(score * 20)  # 20-character bar
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"  {indicator:20} {bar} {score:.2f}")
    
    print(f"\nAutognosis enables ORRRG to understand and optimize its own cognitive processes")
    print(f"through hierarchical self-image building and meta-cognitive reflection.")


async def handle_evolution_command(soc: SelfOrganizingCore, command: str = "evolve") -> None:
    """Handle evolution-related commands."""
    if not hasattr(soc, 'evolution_engine') or not soc.evolution_engine:
        print("Evolution engine not initialized")
        return
    
    command_parts = command.split()
    
    if len(command_parts) == 1:  # Just 'evolve'
        await show_evolution_status(soc)
    elif len(command_parts) >= 2:  # 'evolve COMPONENT [OBJECTIVES]'
        component_name = command_parts[1]
        objectives = command_parts[2:] if len(command_parts) > 2 else ['performance', 'adaptation', 'integration']
        await trigger_component_evolution(soc, component_name, objectives)


async def show_evolution_status(soc: SelfOrganizingCore) -> None:
    """Show evolution engine status."""
    print(f"\n🧬 Evolution Engine Status")
    print("=" * 50)
    
    try:
        status = await soc.evolution_engine.get_evolution_status()
        
        print(f"Evolution Running: {status.get('evolution_running', False)}")
        print(f"Total Components: {status.get('total_components', 0)}")
        print(f"Total Genomes: {status.get('total_genomes', 0)}")
        print(f"Evolution History Length: {status.get('evolution_history_length', 0)}")
        
        # Show per-component evolution status
        components = status.get('components', {})
        if components:
            print(f"\nComponent Evolution Status:")
            for component_id, comp_status in components.items():
                print(f"  • {component_id}:")
                print(f"    Population Size: {comp_status.get('population_size', 0)}")
                print(f"    Best Fitness: {comp_status.get('best_fitness', 0.0):.3f}")
                print(f"    Current Generation: {comp_status.get('current_generation', 0)}")
                print(f"    Total Mutations: {comp_status.get('total_mutations', 0)}")
        
        print(f"\n🚀 Evolution Engine Uses:")
        print("  • Genetic Programming for component behavior evolution")
        print("  • Quantum-inspired algorithms for enhanced exploration")
        print("  • Emergent behavior synthesis from component interactions")
        print("  • Adaptive mutation rates based on success history")
        print("  • Cross-component genetic crossover for innovation")
        
    except Exception as e:
        print(f"Error getting evolution status: {e}")


async def trigger_component_evolution(soc: SelfOrganizingCore, component_name: str, objectives: List[str]) -> None:
    """Trigger evolution for a specific component."""
    print(f"\n🧬 Evolving Component: {component_name}")
    print(f"Objectives: {', '.join(objectives)}")
    print("=" * 50)
    
    try:
        result = await soc.trigger_targeted_evolution(component_name, objectives)
        
        if 'error' in result:
            print(f"❌ Evolution failed: {result['error']}")
        else:
            print(f"✅ Evolution completed successfully!")
            print(f"  Component: {result['component']}")
            print(f"  Generation: {result['generation']}")
            print(f"  Fitness Score: {result['fitness']:.3f}")
            print(f"  Mutations Applied: {result['mutations']}")
            
            if result['fitness'] > 0.7:
                print(f"  🌟 High fitness achieved - significant improvement!")
            elif result['fitness'] > 0.5:
                print(f"  ✨ Good fitness - moderate improvement")
            else:
                print(f"  🔧 Lower fitness - more evolution cycles may be needed")
    
    except Exception as e:
        print(f"❌ Error during evolution: {e}")


async def handle_emergence_command(soc: SelfOrganizingCore) -> None:
    """Handle emergence pattern analysis."""
    if not hasattr(soc, 'evolution_engine') or not soc.evolution_engine:
        print("Evolution engine not initialized")
        return
    
    print(f"\n🌱 Emergent Behavior Patterns")
    print("=" * 50)
    
    try:
        # Trigger emergent pattern synthesis
        emergent_patterns = await soc.evolution_engine.synthesize_emergent_behaviors()
        
        if not emergent_patterns:
            print("No emergent patterns detected yet.")
            print("Emergent behaviors develop as components evolve and interact.")
            return
        
        print(f"Discovered {len(emergent_patterns)} emergent patterns:")
        
        for i, pattern in enumerate(emergent_patterns, 1):
            print(f"\n{i}. Pattern: {pattern.pattern_id}")
            print(f"   Type: {pattern.pattern_type}")
            print(f"   Effectiveness: {pattern.effectiveness:.3f}")
            print(f"   Complexity: {pattern.complexity:.3f}")
            print(f"   Applications: {', '.join(pattern.applications)}")
            print(f"   Emergence Path: {' -> '.join(pattern.emergence_path)}")
            
            # Show effectiveness rating
            if pattern.effectiveness > 0.8:
                print(f"   🌟 Highly effective emergent behavior")
            elif pattern.effectiveness > 0.6:
                print(f"   ✨ Moderately effective pattern")
            else:
                print(f"   🌱 Developing emergent pattern")
        
        print(f"\n💡 Emergent patterns represent novel behaviors that arise from")
        print(f"component interactions and evolutionary processes. They can be")
        print(f"automatically integrated to enhance system capabilities.")
        
    except Exception as e:
        print(f"Error analyzing emergence: {e}")


async def daemon_mode(soc: SelfOrganizingCore) -> None:
    """Run in daemon mode."""
    logger = logging.getLogger(__name__)
    logger.info("Running in daemon mode...")
    
    try:
        while True:
            await asyncio.sleep(10)  # Heartbeat every 10 seconds
            logger.debug("Daemon heartbeat")
    except KeyboardInterrupt:
        logger.info("Daemon shutdown requested")


async def batch_mode(soc: SelfOrganizingCore, config_path: Optional[Path]) -> None:
    """Run in batch mode."""
    logger = logging.getLogger(__name__)
    logger.info("Running in batch mode...")
    
    # Load and execute batch configuration
    if config_path and config_path.exists():
        logger.info(f"Loading batch configuration from {config_path}")
        # Implementation would load and execute batch tasks
    else:
        logger.warning("No valid configuration file provided for batch mode")


async def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ORRRG - Omnipotent Research and Reasoning Reactive Grid")
    
    # Initialize Self-Organizing Core
    soc = SelfOrganizingCore()
    
    try:
        await soc.initialize()
        
        # Filter components if specified
        if args.components:
            enabled_components = [c.strip() for c in args.components.split(',')]
            logger.info(f"Enabling specific components: {enabled_components}")
            # Implementation would filter components
        
        # Run in specified mode
        if args.mode == 'interactive':
            await interactive_mode(soc)
        elif args.mode == 'daemon':
            await daemon_mode(soc)
        elif args.mode == 'batch':
            await batch_mode(soc, args.config)
            
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)
    finally:
        await soc.shutdown()
        logger.info("ORRRG shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
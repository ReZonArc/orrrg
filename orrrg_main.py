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
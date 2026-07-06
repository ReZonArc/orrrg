"""
Plugin Loader - Hot-Loadable Component Architecture
====================================================

Enables runtime discovery, loading, and unloading of ORRRG components
without system restart.

Features:
- Filesystem-based plugin discovery
- Dynamic module loading/unloading
- Plugin lifecycle management (init, start, stop, cleanup)
- Dependency resolution between plugins
- Plugin metadata and capability registration
"""

import asyncio
import importlib
import importlib.util
import logging
import sys
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Metadata about a loaded plugin."""
    name: str
    path: Path
    version: str = "0.0.0"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    loaded_at: Optional[datetime] = None
    status: str = "discovered"  # discovered, loaded, active, failed, unloaded
    error: Optional[str] = None
    module: Any = None
    instance: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'path': str(self.path),
            'version': self.version,
            'description': self.description,
            'dependencies': self.dependencies,
            'capabilities': self.capabilities,
            'loaded_at': self.loaded_at.isoformat() if self.loaded_at else None,
            'status': self.status,
            'error': self.error,
        }


class PluginLoader:
    """
    Runtime plugin loader for ORRRG components.
    
    Discovers and manages dynamically loaded components from the filesystem.
    """

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugin_dirs: List[Path] = []
        self._load_order: List[str] = []

        if plugin_dirs:
            self._plugin_dirs = [Path(d) for d in plugin_dirs]
        else:
            # Default: look in components/ relative to project root
            self._plugin_dirs = [Path("components")]

    def add_plugin_directory(self, path: str) -> None:
        """Add a directory to search for plugins."""
        p = Path(path)
        if p not in self._plugin_dirs:
            self._plugin_dirs.append(p)
            logger.info(f"Added plugin directory: {p}")

    def discover_plugins(self) -> List[PluginInfo]:
        """
        Scan plugin directories for available plugins.
        
        A valid plugin directory must contain:
        - __init__.py (Python package)
        - component_info.json (metadata, optional)
        """
        discovered = []

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                logger.debug(f"Plugin directory not found: {plugin_dir}")
                continue

            for entry in plugin_dir.iterdir():
                if not entry.is_dir():
                    continue
                if not (entry / "__init__.py").exists():
                    continue
                if entry.name.startswith('.') or entry.name == '__pycache__':
                    continue

                # Already tracked
                if entry.name in self._plugins:
                    continue

                info = PluginInfo(name=entry.name, path=entry)

                # Try loading metadata
                meta_path = entry / "component_info.json"
                if meta_path.exists():
                    try:
                        import json
                        with open(meta_path) as f:
                            meta = json.load(f)
                        info.version = meta.get('version', '0.0.0')
                        info.description = meta.get('description', '')
                        info.dependencies = meta.get('dependencies', [])
                    except Exception as e:
                        logger.warning(f"Failed to read metadata for {entry.name}: {e}")

                self._plugins[entry.name] = info
                discovered.append(info)

        logger.info(f"Discovered {len(discovered)} new plugins")
        return discovered

    async def load_plugin(self, name: str) -> bool:
        """
        Load a discovered plugin by name.
        
        Returns True on success, False on failure.
        """
        if name not in self._plugins:
            logger.error(f"Plugin '{name}' not discovered")
            return False

        info = self._plugins[name]
        if info.status == 'active':
            logger.info(f"Plugin '{name}' already active")
            return True

        try:
            # Check dependencies
            for dep in info.dependencies:
                if dep in self._plugins and self._plugins[dep].status != 'active':
                    logger.warning(f"Dependency '{dep}' not active for plugin '{name}'")

            # Dynamic import
            module_path = info.path / "__init__.py"
            spec = importlib.util.spec_from_file_location(
                f"plugins.{name}", str(module_path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create module spec for {name}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"plugins.{name}"] = module
            spec.loader.exec_module(module)

            info.module = module
            info.status = 'loaded'
            info.loaded_at = datetime.now()

            # Find and instantiate the component class
            component_class = self._find_component_class(module, name)
            if component_class:
                info.instance = component_class()
                if hasattr(info.instance, 'get_capabilities'):
                    info.capabilities = info.instance.get_capabilities()

            info.status = 'active'
            self._load_order.append(name)
            logger.info(f"Plugin '{name}' loaded successfully")
            return True

        except Exception as e:
            info.status = 'failed'
            info.error = str(e)
            logger.error(f"Failed to load plugin '{name}': {e}")
            return False

    def _find_component_class(self, module: Any, name: str) -> Optional[type]:
        """Find the main component class in a plugin module."""
        # Look for a class ending in 'Component' or matching the module name
        import inspect
        for attr_name, attr in inspect.getmembers(module, inspect.isclass):
            if attr_name.endswith('Component') and attr.__module__ == module.__name__:
                return attr
        return None

    async def unload_plugin(self, name: str) -> bool:
        """Unload a plugin, calling its cleanup method."""
        if name not in self._plugins:
            return False

        info = self._plugins[name]
        if info.status not in ('active', 'loaded'):
            return True

        try:
            # Call cleanup if available
            if info.instance and hasattr(info.instance, 'cleanup'):
                await info.instance.cleanup()

            # Remove from sys.modules
            module_key = f"plugins.{name}"
            if module_key in sys.modules:
                del sys.modules[module_key]

            info.instance = None
            info.module = None
            info.status = 'unloaded'
            if name in self._load_order:
                self._load_order.remove(name)

            logger.info(f"Plugin '{name}' unloaded")
            return True

        except Exception as e:
            info.error = str(e)
            logger.error(f"Error unloading plugin '{name}': {e}")
            return False

    async def reload_plugin(self, name: str) -> bool:
        """Unload and reload a plugin."""
        await self.unload_plugin(name)
        return await self.load_plugin(name)

    async def initialize_plugin(self, name: str, config: Dict[str, Any]) -> bool:
        """Initialize a loaded plugin with configuration."""
        if name not in self._plugins:
            return False

        info = self._plugins[name]
        if not info.instance:
            logger.error(f"Plugin '{name}' has no instance to initialize")
            return False

        if hasattr(info.instance, 'initialize'):
            return await info.instance.initialize(config)
        return True

    def get_plugin(self, name: str) -> Optional[PluginInfo]:
        """Get plugin info by name."""
        return self._plugins.get(name)

    def get_plugin_instance(self, name: str) -> Optional[Any]:
        """Get the active plugin instance."""
        info = self._plugins.get(name)
        return info.instance if info else None

    def list_plugins(self, status: Optional[str] = None) -> List[PluginInfo]:
        """List all plugins, optionally filtered by status."""
        if status:
            return [p for p in self._plugins.values() if p.status == status]
        return list(self._plugins.values())

    def get_load_order(self) -> List[str]:
        """Get the order in which plugins were loaded."""
        return list(self._load_order)

    async def load_all(self) -> Dict[str, bool]:
        """Discover and load all available plugins."""
        self.discover_plugins()
        results = {}
        for name in list(self._plugins.keys()):
            if self._plugins[name].status not in ('active', 'loaded'):
                results[name] = await self.load_plugin(name)
        return results

    async def cleanup_all(self) -> None:
        """Unload all plugins in reverse load order."""
        for name in reversed(self._load_order[:]):
            await self.unload_plugin(name)
        logger.info("All plugins unloaded")

    def get_status(self) -> Dict[str, Any]:
        """Get overall plugin system status."""
        status_counts = {}
        for info in self._plugins.values():
            status_counts[info.status] = status_counts.get(info.status, 0) + 1

        return {
            'total_plugins': len(self._plugins),
            'plugin_directories': [str(d) for d in self._plugin_dirs],
            'status_counts': status_counts,
            'load_order': self._load_order,
            'plugins': {name: info.to_dict() for name, info in self._plugins.items()},
        }

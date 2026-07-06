"""
Knowledge Persistence Layer
============================

Provides save/load capabilities for ORRRG's full system state:
- AtomSpace knowledge graph
- Evolution Engine genomes and patterns
- Relevance Realization frames and Ennead state
- Component states and configurations
- Inference statistics and model registries

Supports JSON file-based persistence with optional compression.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import gzip

logger = logging.getLogger(__name__)


@dataclass
class PersistenceSnapshot:
    """A point-in-time snapshot of system state."""
    timestamp: str
    version: str = "1.0.0"
    atomspace: Optional[Dict[str, Any]] = None
    evolution_state: Optional[Dict[str, Any]] = None
    relevance_state: Optional[Dict[str, Any]] = None
    component_states: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'timestamp': self.timestamp,
            'atomspace': self.atomspace,
            'evolution_state': self.evolution_state,
            'relevance_state': self.relevance_state,
            'component_states': self.component_states,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistenceSnapshot':
        return cls(
            version=data.get('version', '1.0.0'),
            timestamp=data.get('timestamp', ''),
            atomspace=data.get('atomspace'),
            evolution_state=data.get('evolution_state'),
            relevance_state=data.get('relevance_state'),
            component_states=data.get('component_states', {}),
            metadata=data.get('metadata', {}),
        )


class PersistenceManager:
    """
    Manages saving and loading of ORRRG system state.
    
    Supports:
    - Full system snapshots
    - Incremental component state saves
    - Compressed storage
    - Snapshot history with rotation
    """

    def __init__(self, storage_dir: str = "data/snapshots", max_snapshots: int = 10):
        self.storage_dir = Path(storage_dir)
        self.max_snapshots = max_snapshots
        self._last_save_time: Optional[float] = None

    def ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def save_snapshot(
        self,
        soc=None,
        label: str = "auto",
        compress: bool = True,
    ) -> str:
        """
        Save a full system snapshot.
        
        Args:
            soc: SelfOrganizingCore instance
            label: Human-readable label for the snapshot
            compress: Whether to gzip the output
        
        Returns:
            Path to the saved snapshot file
        """
        self.ensure_storage_dir()

        snapshot = PersistenceSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            metadata={'label': label},
        )

        if soc:
            snapshot = await self._collect_state(soc, snapshot)

        # Serialize
        data = snapshot.to_dict()
        
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{label}_{timestamp_str}"

        if compress:
            filepath = self.storage_dir / f"{filename}.json.gz"
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=None, default=str)
        else:
            filepath = self.storage_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        self._last_save_time = time.time()
        self._rotate_snapshots()

        logger.info(f"Snapshot saved: {filepath}")
        return str(filepath)

    async def load_snapshot(self, path: Optional[str] = None) -> PersistenceSnapshot:
        """
        Load a snapshot from file.
        
        Args:
            path: Specific file path, or None to load the latest
        
        Returns:
            PersistenceSnapshot instance
        """
        if path is None:
            path = self._find_latest_snapshot()
            if path is None:
                raise FileNotFoundError("No snapshots found")

        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")

        if filepath.suffix == '.gz' or str(filepath).endswith('.json.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)

        snapshot = PersistenceSnapshot.from_dict(data)
        logger.info(f"Snapshot loaded: {filepath} (from {snapshot.timestamp})")
        return snapshot

    async def restore_snapshot(self, soc, snapshot: PersistenceSnapshot) -> Dict[str, Any]:
        """
        Restore system state from a snapshot.
        
        Args:
            soc: SelfOrganizingCore instance to restore into
            snapshot: The snapshot to restore
        
        Returns:
            Dictionary with restore results per component
        """
        results = {}

        # Restore AtomSpace
        if snapshot.atomspace and hasattr(soc, 'components'):
            oc_comp = soc.components.get('oc-skintwin')
            if oc_comp and hasattr(oc_comp, 'adapter') and oc_comp.adapter:
                oc_comp.adapter.atomspace.load_from_dict(snapshot.atomspace)
                results['atomspace'] = 'restored'

        # Restore component states
        for comp_name, state in snapshot.component_states.items():
            if hasattr(soc, 'components') and comp_name in soc.components:
                comp = soc.components[comp_name]
                if hasattr(comp, 'restore_state'):
                    await comp.restore_state(state)
                    results[comp_name] = 'restored'

        logger.info(f"Snapshot restored: {results}")
        return results

    async def _collect_state(self, soc, snapshot: PersistenceSnapshot) -> PersistenceSnapshot:
        """Collect state from SOC into snapshot."""
        # Collect AtomSpace state
        if hasattr(soc, 'components'):
            oc_comp = soc.components.get('oc-skintwin')
            if oc_comp and hasattr(oc_comp, 'adapter') and oc_comp.adapter:
                snapshot.atomspace = oc_comp.adapter.atomspace.save_to_dict()

            # Collect component states
            for name, comp in soc.components.items():
                if hasattr(comp, 'get_info'):
                    snapshot.component_states[name] = comp.get_info()

        # Collect evolution state
        if hasattr(soc, 'evolution_engine') and soc.evolution_engine:
            if hasattr(soc.evolution_engine, 'get_statistics'):
                snapshot.evolution_state = soc.evolution_engine.get_statistics()

        # Collect relevance state
        if hasattr(soc, 'relevance_integrator') and soc.relevance_integrator:
            if hasattr(soc.relevance_integrator, 'get_state'):
                snapshot.relevance_state = soc.relevance_integrator.get_state()

        return snapshot

    def _find_latest_snapshot(self) -> Optional[str]:
        """Find the most recent snapshot file."""
        if not self.storage_dir.exists():
            return None

        snapshots = list(self.storage_dir.glob("snapshot_*"))
        if not snapshots:
            return None

        # Sort by modification time
        snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(snapshots[0])

    def _rotate_snapshots(self) -> None:
        """Remove old snapshots beyond max_snapshots."""
        if not self.storage_dir.exists():
            return

        snapshots = sorted(
            self.storage_dir.glob("snapshot_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old in snapshots[self.max_snapshots:]:
            old.unlink()
            logger.debug(f"Rotated old snapshot: {old}")

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots with metadata."""
        if not self.storage_dir.exists():
            return []

        snapshots = sorted(
            self.storage_dir.glob("snapshot_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        result = []
        for path in snapshots:
            stat = path.stat()
            result.append({
                'path': str(path),
                'filename': path.name,
                'size_bytes': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get persistence system status."""
        return {
            'storage_dir': str(self.storage_dir),
            'max_snapshots': self.max_snapshots,
            'last_save_time': self._last_save_time,
            'snapshot_count': len(list(self.storage_dir.glob("snapshot_*"))) if self.storage_dir.exists() else 0,
        }

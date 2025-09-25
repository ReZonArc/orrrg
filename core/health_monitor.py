#!/usr/bin/env python3
"""
Health Monitor for ORRRG System
==============================

Provides health checking and monitoring capabilities for all system components.
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SystemHealth:
    """System health metrics."""
    timestamp: datetime
    overall_status: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_components: int
    total_components: int
    uptime_seconds: float
    performance_score: float


@dataclass 
class ComponentHealth:
    """Individual component health."""
    name: str
    status: str
    last_check: datetime
    response_time_ms: float
    error_count: int
    cpu_usage: float
    memory_usage_mb: float


class HealthMonitor:
    """System health monitoring service."""
    
    def __init__(self):
        self.start_time = time.time()
        self.component_health: Dict[str, ComponentHealth] = {}
        self.system_metrics: List[SystemHealth] = []
        self.max_history = 100  # Keep last 100 health checks
        
    def get_system_health(self, soc=None) -> SystemHealth:
        """Get current system health status."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate component status
            active_components = 0
            total_components = 0
            
            if soc:
                total_components = len(soc.components)
                active_components = len([c for c in soc.components.values() 
                                       if c.status == "available"])
            
            # Calculate performance score (0-100)
            performance_score = self._calculate_performance_score(
                cpu_percent, memory.percent, active_components, total_components
            )
            
            # Determine overall status
            overall_status = self._determine_overall_status(
                cpu_percent, memory.percent, performance_score
            )
            
            uptime = time.time() - self.start_time
            
            health = SystemHealth(
                timestamp=datetime.now(),
                overall_status=overall_status,
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                active_components=active_components,
                total_components=total_components,
                uptime_seconds=uptime,
                performance_score=performance_score
            )
            
            # Store in history
            self.system_metrics.append(health)
            if len(self.system_metrics) > self.max_history:
                self.system_metrics.pop(0)
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return SystemHealth(
                timestamp=datetime.now(),
                overall_status="error",
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                active_components=0,
                total_components=0,
                uptime_seconds=0.0,
                performance_score=0.0
            )
    
    def _calculate_performance_score(self, cpu: float, memory: float, 
                                   active: int, total: int) -> float:
        """Calculate overall performance score (0-100)."""
        try:
            # Component availability score (40% weight)
            component_score = (active / total * 100) if total > 0 else 0
            
            # Resource utilization score (60% weight)
            # Lower utilization is better, but 0% is also bad (no activity)
            cpu_score = max(0, 100 - cpu) if cpu > 5 else cpu * 10
            memory_score = max(0, 100 - memory) if memory > 10 else memory * 5
            
            resource_score = (cpu_score + memory_score) / 2
            
            # Weighted final score
            performance_score = (component_score * 0.4 + resource_score * 0.6)
            
            return min(100, max(0, performance_score))
            
        except Exception:
            return 50.0  # Default middle score on error
    
    def _determine_overall_status(self, cpu: float, memory: float, 
                                performance: float) -> str:
        """Determine overall system status."""
        if performance >= 80:
            return "healthy"
        elif performance >= 60:
            return "warning" 
        elif performance >= 30:
            return "degraded"
        else:
            return "critical"
    
    async def check_component_health(self, component_name: str, 
                                   component_adapter=None) -> ComponentHealth:
        """Check health of a specific component."""
        start_time = time.time()
        
        try:
            # Simulate component health check
            if component_adapter:
                # In a real implementation, this would ping the component
                await asyncio.sleep(0.01)  # Simulate network delay
                status = "healthy" if component_adapter.initialized else "unavailable"
                error_count = 0
            else:
                status = "unknown"
                error_count = 1
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            health = ComponentHealth(
                name=component_name,
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_count=error_count,
                cpu_usage=0.0,  # Would be measured in real implementation
                memory_usage_mb=0.0  # Would be measured in real implementation
            )
            
            self.component_health[component_name] = health
            return health
            
        except Exception as e:
            logger.error(f"Error checking component {component_name}: {e}")
            error_health = ComponentHealth(
                name=component_name,
                status="error",
                last_check=datetime.now(),
                response_time_ms=-1,
                error_count=1,
                cpu_usage=0.0,
                memory_usage_mb=0.0
            )
            self.component_health[component_name] = error_health
            return error_health
    
    async def run_health_checks(self, soc=None) -> Dict[str, Any]:
        """Run comprehensive health checks."""
        logger.info("Running comprehensive health checks...")
        
        # Get system health
        system_health = self.get_system_health(soc)
        
        # Check component health
        component_checks = {}
        if soc:
            for component_name in soc.components.keys():
                component_health = await self.check_component_health(component_name)
                component_checks[component_name] = asdict(component_health)
        
        # Compile results
        health_report = {
            "system": asdict(system_health),
            "components": component_checks,
            "summary": {
                "overall_status": system_health.overall_status,
                "performance_score": system_health.performance_score,
                "healthy_components": len([c for c in component_checks.values() 
                                         if c["status"] == "healthy"]),
                "total_components": len(component_checks),
                "uptime_hours": system_health.uptime_seconds / 3600
            }
        }
        
        logger.info(f"Health check complete: {health_report['summary']['overall_status']}")
        return health_report
    
    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent health history."""
        recent_metrics = self.system_metrics[-limit:] if self.system_metrics else []
        return [asdict(metric) for metric in recent_metrics]
    
    def get_health_endpoint_data(self, soc=None) -> Dict[str, Any]:
        """Get data for health check endpoint."""
        current_health = self.get_system_health(soc)
        
        return {
            "status": current_health.overall_status,
            "timestamp": current_health.timestamp.isoformat(),
            "uptime": current_health.uptime_seconds,
            "performance_score": current_health.performance_score,
            "resources": {
                "cpu_usage": current_health.cpu_usage,
                "memory_usage": current_health.memory_usage,
                "disk_usage": current_health.disk_usage
            },
            "components": {
                "active": current_health.active_components,
                "total": current_health.total_components
            }
        }
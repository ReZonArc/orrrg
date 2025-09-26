#!/usr/bin/env python3
"""
Autognosis - Hierarchical Self-Image Building System
===================================================

This module implements autognosis capabilities for ORRRG, enabling the system to:
1. Build hierarchical models of its own cognitive processes
2. Perform recursive self-reflection and meta-cognition  
3. Adaptively optimize based on self-understanding
4. Maintain dynamic self-awareness and self-image construction

The autognosis system operates through several interconnected layers:
- Self-Monitoring Layer: Continuous observation of system states and behaviors
- Self-Modeling Layer: Construction of internal models of own processes
- Meta-Cognitive Layer: Higher-order reasoning about own reasoning
- Self-Optimization Layer: Adaptive improvements based on self-insights

This enables ORRRG to become truly self-aware and self-improving.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SelfImage:
    """Represents a hierarchical self-image at a specific level."""
    level: int
    timestamp: datetime
    component_states: Dict[str, Any]
    behavioral_patterns: Dict[str, List[float]]
    performance_metrics: Dict[str, float]
    cognitive_processes: Dict[str, Any]
    meta_reflections: List[str]
    confidence: float
    image_hash: str = field(init=False)
    
    def __post_init__(self):
        """Generate hash for this self-image."""
        content = json.dumps({
            'level': self.level,
            'component_states': self.component_states,
            'behavioral_patterns': self.behavioral_patterns,
            'performance_metrics': self.performance_metrics,
            'cognitive_processes': self.cognitive_processes
        }, sort_keys=True)
        self.image_hash = hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class MetaCognitiveInsight:
    """Represents a meta-cognitive insight about the system's own processes."""
    insight_type: str
    description: str
    confidence: float
    evidence: Dict[str, Any]
    implications: List[str]
    suggested_actions: List[str]
    timestamp: datetime


@dataclass
class SelfOptimization:
    """Represents a self-optimization action discovered through autognosis."""
    optimization_type: str
    target_component: str
    current_state: Dict[str, Any]
    proposed_changes: Dict[str, Any]
    expected_improvement: float
    risk_assessment: float
    execution_priority: int


class SelfMonitor:
    """Monitors system states and behaviors for autognosis."""
    
    def __init__(self):
        self.observation_history = deque(maxlen=1000)
        self.behavioral_patterns = defaultdict(deque)
        self.performance_trends = defaultdict(list)
        self.anomaly_threshold = 2.0  # Standard deviations
        
    async def observe_system(self, soc) -> Dict[str, Any]:
        """Observe current system state."""
        observation = {
            'timestamp': datetime.now(),
            'component_count': len(soc.components),
            'active_components': sum(1 for c in soc.components.values() if c.status == 'active'),
            'knowledge_graph_size': len(soc.knowledge_graph),
            'event_queue_size': soc.event_bus.qsize() if hasattr(soc.event_bus, 'qsize') else 0,
            'system_running': soc.running,
            'uptime': time.time() - getattr(soc, 'start_time', time.time())
        }
        
        # Add component-specific observations
        component_states = {}
        for name, component in soc.components.items():
            component_states[name] = {
                'status': component.status,
                'capabilities_count': len(component.capabilities),
                'last_activity': getattr(component, 'last_activity', None)
            }
        
        observation['components'] = component_states
        self.observation_history.append(observation)
        
        return observation
    
    def detect_patterns(self) -> Dict[str, List[float]]:
        """Detect behavioral patterns from observation history."""
        patterns = {}
        
        if len(self.observation_history) < 2:
            return patterns
            
        # Pattern: Component activation rates
        activation_rates = []
        for obs in list(self.observation_history)[-10:]:  # Last 10 observations
            if obs['component_count'] > 0:
                rate = obs['active_components'] / obs['component_count']
                activation_rates.append(rate)
        
        if activation_rates:
            patterns['component_activation_rate'] = activation_rates
            
        # Pattern: System responsiveness (inverse of queue size)
        responsiveness = []
        for obs in list(self.observation_history)[-10:]:
            queue_size = obs.get('event_queue_size', 0)
            resp = 1.0 / (1.0 + queue_size)  # Sigmoid-like response
            responsiveness.append(resp)
        
        if responsiveness:
            patterns['system_responsiveness'] = responsiveness
            
        return patterns
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in system behavior."""
        anomalies = []
        
        if len(self.observation_history) < 10:
            return anomalies
            
        # Check for sudden drops in component activity
        recent_activity = [obs['active_components'] for obs in list(self.observation_history)[-5:]]
        historical_activity = [obs['active_components'] for obs in list(self.observation_history)[-20:-5]]
        
        if len(historical_activity) > 0:
            historical_mean = sum(historical_activity) / len(historical_activity)
            recent_mean = sum(recent_activity) / len(recent_activity)
            
            if historical_mean > 0 and (recent_mean / historical_mean) < 0.5:
                anomalies.append({
                    'type': 'component_activity_drop',
                    'description': f'Component activity dropped from {historical_mean:.1f} to {recent_mean:.1f}',
                    'severity': 'high',
                    'timestamp': datetime.now()
                })
        
        return anomalies


class HierarchicalSelfModeler:
    """Builds hierarchical models of the system's own processes."""
    
    def __init__(self):
        self.self_images = {}  # level -> list of self images
        self.max_levels = 5
        self.model_update_interval = 60  # seconds
        
    async def build_self_image(self, level: int, monitor: SelfMonitor, soc) -> SelfImage:
        """Build a self-image at the specified hierarchical level."""
        if level == 0:
            # Level 0: Direct system state observation
            observation = await monitor.observe_system(soc)
            patterns = monitor.detect_patterns()
            
            return SelfImage(
                level=0,
                timestamp=datetime.now(),
                component_states=observation['components'],
                behavioral_patterns=patterns,
                performance_metrics={
                    'component_utilization': observation['active_components'] / max(observation['component_count'], 1),
                    'system_responsiveness': 1.0 / (1.0 + observation.get('event_queue_size', 0)),
                    'uptime': observation['uptime']
                },
                cognitive_processes={
                    'knowledge_graph_growth': len(soc.knowledge_graph),
                    'component_discovery_active': len([c for c in soc.components.values() if c.status == 'discovering'])
                },
                meta_reflections=[],
                confidence=0.9
            )
        
        elif level == 1:
            # Level 1: Pattern analysis and behavior modeling
            level0_image = await self.build_self_image(0, monitor, soc)
            
            # Analyze patterns from level 0
            pattern_insights = self._analyze_behavioral_patterns(level0_image.behavioral_patterns)
            performance_trends = self._analyze_performance_trends(level0_image.performance_metrics)
            
            return SelfImage(
                level=1,
                timestamp=datetime.now(),
                component_states=level0_image.component_states,
                behavioral_patterns=level0_image.behavioral_patterns,
                performance_metrics=level0_image.performance_metrics,
                cognitive_processes={
                    'pattern_insights': pattern_insights,
                    'performance_trends': performance_trends,
                    'adaptation_signals': self._detect_adaptation_signals(level0_image)
                },
                meta_reflections=[
                    f"System shows {len(pattern_insights)} distinct behavioral patterns",
                    f"Performance trend analysis reveals {len(performance_trends)} key trends"
                ],
                confidence=0.8
            )
        
        else:
            # Higher levels: Recursive meta-modeling
            lower_image = await self.build_self_image(level - 1, monitor, soc)
            
            # Meta-analysis of lower-level self-image
            meta_insights = self._perform_meta_analysis(lower_image)
            
            return SelfImage(
                level=level,
                timestamp=datetime.now(),
                component_states=lower_image.component_states,
                behavioral_patterns=lower_image.behavioral_patterns,
                performance_metrics=lower_image.performance_metrics,
                cognitive_processes={
                    'meta_insights': meta_insights,
                    'recursive_depth': level,
                    'self_awareness_indicators': self._assess_self_awareness(lower_image)
                },
                meta_reflections=[
                    f"Level {level} meta-analysis of self-understanding",
                    f"Recursive self-modeling depth: {level}",
                    f"Meta-cognitive complexity: {len(meta_insights)}"
                ],
                confidence=max(0.1, 0.9 - (level * 0.1))  # Confidence decreases with meta-level
            )
    
    def _analyze_behavioral_patterns(self, patterns: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Analyze behavioral patterns for insights."""
        insights = []
        
        for pattern_name, values in patterns.items():
            if len(values) > 1:
                trend = 'increasing' if values[-1] > values[0] else 'decreasing'
                volatility = self._calculate_volatility(values)
                
                insights.append({
                    'pattern': pattern_name,
                    'trend': trend,
                    'volatility': volatility,
                    'current_value': values[-1],
                    'stability': 'stable' if volatility < 0.1 else 'volatile'
                })
        
        return insights
    
    def _analyze_performance_trends(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Analyze performance metrics for trends."""
        trends = []
        
        for metric_name, value in metrics.items():
            assessment = 'good' if value > 0.7 else 'moderate' if value > 0.4 else 'poor'
            
            trends.append({
                'metric': metric_name,
                'value': value,
                'assessment': assessment,
                'optimization_potential': 1.0 - value
            })
        
        return trends
    
    def _detect_adaptation_signals(self, self_image: SelfImage) -> List[str]:
        """Detect signals that indicate need for adaptation."""
        signals = []
        
        # Check performance metrics
        for metric, value in self_image.performance_metrics.items():
            if value < 0.5:
                signals.append(f"Low {metric}: {value:.2f}")
        
        # Check behavioral patterns
        for pattern_name, values in self_image.behavioral_patterns.items():
            if len(values) > 2:
                recent_avg = sum(values[-3:]) / 3
                if recent_avg < 0.3:
                    signals.append(f"Declining {pattern_name}: {recent_avg:.2f}")
        
        return signals
    
    def _perform_meta_analysis(self, lower_image: SelfImage) -> List[Dict[str, Any]]:
        """Perform meta-analysis of lower-level self-image."""
        meta_insights = []
        
        # Analyze the quality of self-understanding at lower level
        cognitive_complexity = len(lower_image.cognitive_processes)
        reflection_depth = len(lower_image.meta_reflections)
        
        meta_insights.append({
            'type': 'self_understanding_quality',
            'cognitive_complexity': cognitive_complexity,
            'reflection_depth': reflection_depth,
            'understanding_score': (cognitive_complexity + reflection_depth) / 10.0
        })
        
        # Analyze confidence in self-image
        meta_insights.append({
            'type': 'confidence_analysis',
            'confidence_level': lower_image.confidence,
            'confidence_assessment': 'high' if lower_image.confidence > 0.8 else 'moderate' if lower_image.confidence > 0.5 else 'low'
        })
        
        return meta_insights
    
    def _assess_self_awareness(self, self_image: SelfImage) -> Dict[str, float]:
        """Assess indicators of self-awareness."""
        return {
            'pattern_recognition': len(self_image.behavioral_patterns) / 5.0,
            'performance_awareness': sum(self_image.performance_metrics.values()) / len(self_image.performance_metrics),
            'meta_reflection_depth': len(self_image.meta_reflections) / 3.0,
            'cognitive_complexity': len(self_image.cognitive_processes) / 5.0
        }
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility of a series of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


class MetaCognitiveProcessor:
    """Processes meta-cognitive insights and generates higher-order understanding."""
    
    def __init__(self):
        self.insight_history = deque(maxlen=100)
        self.meta_patterns = defaultdict(list)
        
    async def process_self_image(self, self_image: SelfImage) -> List[MetaCognitiveInsight]:
        """Process a self-image to generate meta-cognitive insights."""
        insights = []
        
        # Generate insights about system state
        if self_image.level == 0:
            insights.extend(self._generate_state_insights(self_image))
        elif self_image.level == 1:
            insights.extend(self._generate_pattern_insights(self_image))
        else:
            insights.extend(self._generate_meta_insights(self_image))
        
        # Store insights for pattern detection
        for insight in insights:
            self.insight_history.append(insight)
        
        return insights
    
    def _generate_state_insights(self, self_image: SelfImage) -> List[MetaCognitiveInsight]:
        """Generate insights about basic system state."""
        insights = []
        
        # Component utilization insight
        utilization = self_image.performance_metrics.get('component_utilization', 0)
        if utilization < 0.5:
            insights.append(MetaCognitiveInsight(
                insight_type='resource_underutilization',
                description=f'System is underutilizing components ({utilization:.1%})',
                confidence=0.8,
                evidence={'utilization_rate': utilization, 'active_components': len([c for c in self_image.component_states.values() if c['status'] == 'active'])},
                implications=['Potential for increased throughput', 'May indicate lack of workload'],
                suggested_actions=['Investigate component activation patterns', 'Consider load balancing optimization'],
                timestamp=datetime.now()
            ))
        
        return insights
    
    def _generate_pattern_insights(self, self_image: SelfImage) -> List[MetaCognitiveInsight]:
        """Generate insights about behavioral patterns."""
        insights = []
        
        # Pattern stability insight
        patterns = self_image.cognitive_processes.get('pattern_insights', [])
        volatile_patterns = [p for p in patterns if p.get('stability') == 'volatile']
        
        if len(volatile_patterns) > len(patterns) / 2:
            insights.append(MetaCognitiveInsight(
                insight_type='behavioral_instability',
                description=f'System shows high behavioral volatility ({len(volatile_patterns)}/{len(patterns)} patterns volatile)',
                confidence=0.7,
                evidence={'volatile_patterns': volatile_patterns, 'total_patterns': len(patterns)},
                implications=['System may be in adaptive phase', 'Could indicate external stress'],
                suggested_actions=['Monitor adaptation progress', 'Consider stabilization measures'],
                timestamp=datetime.now()
            ))
        
        return insights
    
    def _generate_meta_insights(self, self_image: SelfImage) -> List[MetaCognitiveInsight]:
        """Generate meta-level insights about self-understanding."""
        insights = []
        
        # Self-awareness assessment
        awareness_indicators = self_image.cognitive_processes.get('self_awareness_indicators', {})
        avg_awareness = sum(awareness_indicators.values()) / len(awareness_indicators) if awareness_indicators else 0
        
        if avg_awareness > 0.7:
            insights.append(MetaCognitiveInsight(
                insight_type='high_self_awareness',
                description=f'System demonstrates high self-awareness (score: {avg_awareness:.2f})',
                confidence=0.9,
                evidence={'awareness_score': avg_awareness, 'indicators': awareness_indicators},
                implications=['Strong introspective capabilities', 'Good foundation for self-optimization'],
                suggested_actions=['Leverage self-awareness for proactive optimization', 'Develop more sophisticated self-models'],
                timestamp=datetime.now()
            ))
        
        return insights


class AutognosisOrchestrator:
    """Main orchestrator for the autognosis system."""
    
    def __init__(self):
        self.monitor = SelfMonitor()
        self.modeler = HierarchicalSelfModeler()
        self.processor = MetaCognitiveProcessor()
        self.optimization_engine = None  # Will be set by SOC
        
        self.current_self_images = {}  # level -> current self image
        self.insight_history = deque(maxlen=1000)
        self.optimization_queue = deque(maxlen=100)
        
        self.running = False
        self.cycle_interval = 30  # seconds
        
    async def initialize(self, soc) -> None:
        """Initialize the autognosis system."""
        logger.info("Initializing Autognosis - Hierarchical Self-Image Building System")
        
        # Perform initial self-assessment
        await self._perform_initial_assessment(soc)
        
        self.running = True
        logger.info("Autognosis system initialized successfully")
    
    async def _perform_initial_assessment(self, soc) -> None:
        """Perform initial self-assessment to bootstrap autognosis."""
        logger.info("Performing initial self-assessment...")
        
        # Build initial self-images at all levels
        for level in range(3):  # Start with 3 levels
            self_image = await self.modeler.build_self_image(level, self.monitor, soc)
            self.current_self_images[level] = self_image
            
            # Process for insights
            insights = await self.processor.process_self_image(self_image)
            self.insight_history.extend(insights)
            
            logger.info(f"Built level {level} self-image with {len(insights)} insights")
    
    async def run_autognosis_cycle(self, soc) -> Dict[str, Any]:
        """Run a complete autognosis cycle."""
        cycle_start = time.time()
        cycle_results = {
            'timestamp': datetime.now(),
            'self_images_updated': 0,
            'new_insights': 0,
            'optimizations_discovered': 0,
            'meta_reflections': []
        }
        
        try:
            # Update self-images hierarchically
            for level in range(len(self.current_self_images) + 1):
                if level < 5:  # Maximum 5 levels
                    self_image = await self.modeler.build_self_image(level, self.monitor, soc)
                    self.current_self_images[level] = self_image
                    cycle_results['self_images_updated'] += 1
                    
                    # Process for new insights
                    insights = await self.processor.process_self_image(self_image)
                    self.insight_history.extend(insights)
                    cycle_results['new_insights'] += len(insights)
                    
                    # Generate meta-reflections
                    if level > 0:
                        reflection = f"Level {level}: {len(self_image.meta_reflections)} reflections, confidence {self_image.confidence:.2f}"
                        cycle_results['meta_reflections'].append(reflection)
            
            # Discover optimization opportunities
            optimizations = await self._discover_optimizations()
            self.optimization_queue.extend(optimizations)
            cycle_results['optimizations_discovered'] = len(optimizations)
            
            cycle_time = time.time() - cycle_start
            logger.info(f"Autognosis cycle completed in {cycle_time:.2f}s: {cycle_results['new_insights']} insights, {cycle_results['optimizations_discovered']} optimizations")
            
        except Exception as e:
            logger.error(f"Error in autognosis cycle: {e}")
            cycle_results['error'] = str(e)
        
        return cycle_results
    
    async def _discover_optimizations(self) -> List[SelfOptimization]:
        """Discover optimization opportunities from autognosis insights."""
        optimizations = []
        
        # Analyze recent insights for optimization opportunities
        recent_insights = list(self.insight_history)[-10:]
        
        for insight in recent_insights:
            if insight.insight_type == 'resource_underutilization':
                optimization = SelfOptimization(
                    optimization_type='resource_allocation',
                    target_component='system_wide',
                    current_state={'utilization': insight.evidence.get('utilization_rate', 0)},
                    proposed_changes={'increase_component_activation': True, 'load_balancing': True},
                    expected_improvement=0.3,
                    risk_assessment=0.1,
                    execution_priority=5
                )
                optimizations.append(optimization)
            
            elif insight.insight_type == 'behavioral_instability':
                optimization = SelfOptimization(
                    optimization_type='stability_enhancement',
                    target_component='behavior_control',
                    current_state={'volatility': 'high'},
                    proposed_changes={'adaptive_damping': True, 'pattern_smoothing': True},
                    expected_improvement=0.4,
                    risk_assessment=0.2,
                    execution_priority=7
                )
                optimizations.append(optimization)
        
        return optimizations
    
    def get_self_awareness_report(self) -> Dict[str, Any]:
        """Generate a comprehensive self-awareness report."""
        report = {
            'timestamp': datetime.now(),
            'system_status': 'running' if self.running else 'stopped',
            'self_image_levels': len(self.current_self_images),
            'total_insights': len(self.insight_history),
            'pending_optimizations': len(self.optimization_queue)
        }
        
        # Add current self-images summary
        report['self_images'] = {}
        for level, image in self.current_self_images.items():
            report['self_images'][level] = {
                'timestamp': image.timestamp.isoformat(),
                'confidence': image.confidence,
                'reflections_count': len(image.meta_reflections),
                'image_hash': image.image_hash
            }
        
        # Add recent insights summary
        recent_insights = list(self.insight_history)[-5:]
        report['recent_insights'] = [
            {
                'type': insight.insight_type,
                'description': insight.description,
                'confidence': insight.confidence,
                'timestamp': insight.timestamp.isoformat()
            }
            for insight in recent_insights
        ]
        
        # Add optimization opportunities
        top_optimizations = sorted(list(self.optimization_queue), 
                                 key=lambda o: o.execution_priority, reverse=True)[:3]
        report['top_optimizations'] = [
            {
                'type': opt.optimization_type,
                'target': opt.target_component,
                'expected_improvement': opt.expected_improvement,
                'priority': opt.execution_priority
            }
            for opt in top_optimizations
        ]
        
        return report
    
    async def execute_optimization(self, optimization: SelfOptimization, soc) -> bool:
        """Execute a self-optimization action."""
        logger.info(f"Executing autognosis optimization: {optimization.optimization_type} on {optimization.target_component}")
        
        try:
            # Implementation would depend on optimization type
            if optimization.optimization_type == 'resource_allocation':
                # Example: Adjust component priorities or resource limits
                logger.info("Adjusting resource allocation based on autognosis insights")
                return True
            
            elif optimization.optimization_type == 'stability_enhancement':
                # Example: Implement adaptive damping or smoothing
                logger.info("Implementing stability enhancements based on behavioral analysis")
                return True
            
            # Add more optimization types as needed
            
        except Exception as e:
            logger.error(f"Failed to execute optimization {optimization.optimization_type}: {e}")
            return False
        
        return False
    
    async def shutdown(self) -> None:
        """Shutdown the autognosis system."""
        self.running = False
        logger.info("Autognosis system shutdown complete")
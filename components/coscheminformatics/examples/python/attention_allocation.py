#!/usr/bin/env python3
"""
attention_allocation.py

Adaptive Attention Allocation for Cosmeceutical Formulation Optimization

This module implements an ECAN-inspired attention allocation system for managing
computational resources in cosmeceutical formulation optimization. It provides
mechanisms for:

1. Allocating attention to promising ingredient combination subspaces
2. Dynamic priority adjustment based on search progress and constraints
3. Resource management across multiple optimization tasks
4. Adaptive search strategies with attention-based focusing

Key Features:
- Attention value calculation for ingredient combinations
- Dynamic importance and urgency assessment
- Attention spreading across related formulation spaces
- Integration with formulation optimization pipelines
"""

import math
import time
from typing import Dict, List, Tuple, Set, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
import random

class AttentionValueType(Enum):
    """Types of attention values in the system"""
    IMPORTANCE = "importance"  # Long-term significance
    URGENCY = "urgency"       # Short-term priority
    CONFIDENCE = "confidence"  # Reliability of assessment
    NOVELTY = "novelty"       # Unexplored potential

@dataclass
class AttentionValue:
    """Attention value with associated metadata"""
    value: float
    value_type: AttentionValueType
    last_updated: float = field(default_factory=time.time)
    update_count: int = 0
    decay_rate: float = 0.95
    
    def update(self, new_value: float, learning_rate: float = 0.1):
        """Update attention value with learning rate"""
        self.value = (1 - learning_rate) * self.value + learning_rate * new_value
        self.last_updated = time.time()
        self.update_count += 1
    
    def decay(self, time_delta: float):
        """Apply time-based decay to attention value"""
        decay_factor = self.decay_rate ** (time_delta / 3600)  # Hourly decay
        self.value *= decay_factor

@dataclass
class FormulationNode:
    """Node representing a formulation or ingredient combination in attention network"""
    id: str
    ingredients: Dict[str, float]  # ingredient -> concentration
    formulation_type: str
    attention_values: Dict[AttentionValueType, AttentionValue] = field(default_factory=dict)
    neighbors: Set[str] = field(default_factory=set)
    search_history: List[Dict] = field(default_factory=list)
    efficacy_estimates: List[float] = field(default_factory=list)
    cost_estimates: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default attention values"""
        for value_type in AttentionValueType:
            if value_type not in self.attention_values:
                initial_value = 0.5 if value_type != AttentionValueType.NOVELTY else 1.0
                self.attention_values[value_type] = AttentionValue(initial_value, value_type)
    
    def get_total_attention(self) -> float:
        """Calculate total attention as weighted combination of attention values"""
        weights = {
            AttentionValueType.IMPORTANCE: 0.4,
            AttentionValueType.URGENCY: 0.3, 
            AttentionValueType.CONFIDENCE: 0.2,
            AttentionValueType.NOVELTY: 0.1
        }
        
        total = 0.0
        for value_type, weight in weights.items():
            if value_type in self.attention_values:
                total += weight * self.attention_values[value_type].value
        
        return total
    
    def update_from_search_result(self, result: Dict):
        """Update attention values based on search/optimization results"""
        self.search_history.append(result)
        
        # Update efficacy-based importance
        if 'efficacy' in result:
            self.efficacy_estimates.append(result['efficacy'])
            avg_efficacy = sum(self.efficacy_estimates[-10:]) / len(self.efficacy_estimates[-10:])
            self.attention_values[AttentionValueType.IMPORTANCE].update(avg_efficacy)
        
        # Update cost-based urgency
        if 'cost' in result:
            self.cost_estimates.append(result['cost'])
            # Lower cost = higher urgency (more promising)
            normalized_cost = 1.0 - min(result['cost'] / 100.0, 1.0)
            self.attention_values[AttentionValueType.URGENCY].update(normalized_cost)
        
        # Update confidence based on result consistency
        if len(self.search_history) > 1:
            recent_results = [r.get('efficacy', 0.5) for r in self.search_history[-5:]]
            variance = sum((x - sum(recent_results)/len(recent_results))**2 for x in recent_results) / len(recent_results)
            confidence = 1.0 / (1.0 + variance)  # Lower variance = higher confidence
            self.attention_values[AttentionValueType.CONFIDENCE].update(confidence)
        
        # Decay novelty as node is explored more
        exploration_factor = 1.0 / (1.0 + len(self.search_history) * 0.1)
        self.attention_values[AttentionValueType.NOVELTY].update(exploration_factor)

class AttentionAllocationManager:
    """
    Main class for managing attention allocation in cosmeceutical formulation optimization.
    
    Implements ECAN-inspired mechanisms for:
    - Allocating computational resources to promising formulation subspaces
    - Dynamic priority adjustment based on search progress
    - Attention spreading across related ingredient combinations
    """
    
    def __init__(
        self,
        max_active_nodes: int = 100,
        attention_spreading_factor: float = 0.1,
        resource_budget: float = 1000.0
    ):
        self.nodes: Dict[str, FormulationNode] = {}
        self.max_active_nodes = max_active_nodes
        self.attention_spreading_factor = attention_spreading_factor
        self.resource_budget = resource_budget
        self.current_resource_usage = 0.0
        
        # Active formulation queues by priority
        self.high_priority_queue = []  # heap of (-attention_value, node_id)
        self.medium_priority_queue = []
        self.low_priority_queue = []
        
        # Attention spreading network
        self.similarity_threshold = 0.7
        self.spreading_history = deque(maxlen=1000)
        
        # Resource allocation tracking
        self.resource_allocation_history = []
        
    def add_formulation_node(self, formulation: Dict) -> str:
        """Add a new formulation node to the attention network"""
        node_id = self._generate_node_id(formulation)
        
        node = FormulationNode(
            id=node_id,
            ingredients=formulation.get('ingredients', {}),
            formulation_type=formulation.get('type', 'unknown')
        )
        
        self.nodes[node_id] = node
        self._update_similarity_connections(node_id)
        self._add_to_priority_queues(node_id)
        
        return node_id
    
    def _generate_node_id(self, formulation: Dict) -> str:
        """Generate unique ID for formulation node"""
        ingredients = formulation.get('ingredients', {})
        ingredient_signature = '_'.join(sorted(ingredients.keys()))
        return f"form_{hash(ingredient_signature) % 10000}"
    
    def _update_similarity_connections(self, node_id: str):
        """Update similarity-based connections between nodes"""
        target_node = self.nodes[node_id]
        
        for other_id, other_node in self.nodes.items():
            if other_id != node_id:
                similarity = self._calculate_formulation_similarity(target_node, other_node)
                
                if similarity > self.similarity_threshold:
                    target_node.neighbors.add(other_id)
                    other_node.neighbors.add(node_id)
    
    def _calculate_formulation_similarity(
        self, 
        node1: FormulationNode, 
        node2: FormulationNode
    ) -> float:
        """Calculate similarity between two formulation nodes"""
        ingredients1 = set(node1.ingredients.keys())
        ingredients2 = set(node2.ingredients.keys())
        
        # Jaccard similarity for ingredient overlap
        intersection = len(ingredients1.intersection(ingredients2))
        union = len(ingredients1.union(ingredients2))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # Consider concentration similarity for shared ingredients
        concentration_similarity = 0.0
        shared_ingredients = ingredients1.intersection(ingredients2)
        
        if shared_ingredients:
            conc_diffs = []
            for ingredient in shared_ingredients:
                conc1 = node1.ingredients[ingredient]
                conc2 = node2.ingredients[ingredient]
                diff = abs(conc1 - conc2) / max(conc1, conc2, 1.0)
                conc_diffs.append(1.0 - diff)
            
            concentration_similarity = sum(conc_diffs) / len(conc_diffs)
        
        # Weighted combination
        return 0.6 * jaccard_similarity + 0.4 * concentration_similarity
    
    def _add_to_priority_queues(self, node_id: str):
        """Add node to appropriate priority queue based on attention values"""
        node = self.nodes[node_id]
        total_attention = node.get_total_attention()
        
        if total_attention > 0.7:
            heapq.heappush(self.high_priority_queue, (-total_attention, node_id))
        elif total_attention > 0.4:
            heapq.heappush(self.medium_priority_queue, (-total_attention, node_id))
        else:
            heapq.heappush(self.low_priority_queue, (-total_attention, node_id))
    
    def allocate_computational_resources(
        self, 
        optimization_tasks: List[Callable],
        time_budget: float = 300.0  # 5 minutes default
    ) -> List[Tuple[str, Dict]]:
        """
        Allocate computational resources to optimization tasks based on attention values.
        
        Args:
            optimization_tasks: List of optimization functions to execute
            time_budget: Total time budget in seconds
            
        Returns:
            List of (node_id, optimization_result) tuples
        """
        results = []
        start_time = time.time()
        
        # Process high priority queue first
        available_time = time_budget
        queue_weights = [0.6, 0.3, 0.1]  # High, medium, low priority weights
        queues = [self.high_priority_queue, self.medium_priority_queue, self.low_priority_queue]
        
        for queue, weight in zip(queues, queue_weights):
            queue_time = available_time * weight
            queue_results = self._process_priority_queue(
                queue, optimization_tasks, queue_time
            )
            results.extend(queue_results)
            
            elapsed = time.time() - start_time
            available_time = max(0, time_budget - elapsed)
            
            if available_time <= 0:
                break
        
        # Update attention values based on results
        self._update_attention_from_results(results)
        
        # Perform attention spreading
        self._spread_attention()
        
        # Record resource allocation
        self.resource_allocation_history.append({
            'timestamp': time.time(),
            'nodes_processed': len(results),
            'time_used': time.time() - start_time,
            'high_priority_processed': len([r for r in results if self._is_high_priority(r[0])]),
        })
        
        return results
    
    def _process_priority_queue(
        self, 
        queue: List,
        optimization_tasks: List[Callable],
        time_budget: float
    ) -> List[Tuple[str, Dict]]:
        """Process nodes from a priority queue within time budget"""
        results = []
        start_time = time.time()
        
        while queue and (time.time() - start_time) < time_budget:
            try:
                neg_attention, node_id = heapq.heappop(queue)
                node = self.nodes.get(node_id)
                
                if node is None:
                    continue
                
                # Select appropriate optimization task
                task = random.choice(optimization_tasks)  # In practice, would be more sophisticated
                
                # Execute optimization with attention-based parameters
                task_params = {
                    'formulation': {
                        'ingredients': node.ingredients,
                        'type': node.formulation_type
                    },
                    'attention_weight': -neg_attention,
                    'time_limit': min(30.0, time_budget * 0.1)  # Max 30 seconds per task
                }
                
                result = task(task_params)
                results.append((node_id, result))
                
                # Update node with results
                node.update_from_search_result(result)
                
            except Exception as e:
                print(f"Error processing node {node_id}: {e}")
                continue
        
        return results
    
    def _is_high_priority(self, node_id: str) -> bool:
        """Check if node is high priority based on current attention values"""
        node = self.nodes.get(node_id)
        return node is not None and node.get_total_attention() > 0.7
    
    def _update_attention_from_results(self, results: List[Tuple[str, Dict]]):
        """Update attention values based on optimization results"""
        for node_id, result in results:
            node = self.nodes.get(node_id)
            if node:
                node.update_from_search_result(result)
    
    def _spread_attention(self):
        """Spread attention across connected nodes in the network"""
        for node_id, node in self.nodes.items():
            if not node.neighbors:
                continue
            
            # Calculate attention to spread
            total_attention = node.get_total_attention()
            spread_amount = total_attention * self.attention_spreading_factor
            spread_per_neighbor = spread_amount / len(node.neighbors)
            
            # Spread to neighbors
            for neighbor_id in node.neighbors:
                neighbor = self.nodes.get(neighbor_id)
                if neighbor:
                    # Boost importance of neighbors
                    current_importance = neighbor.attention_values[AttentionValueType.IMPORTANCE].value
                    new_importance = min(1.0, current_importance + spread_per_neighbor * 0.5)
                    neighbor.attention_values[AttentionValueType.IMPORTANCE].update(new_importance)
            
            # Record spreading event
            self.spreading_history.append({
                'source_node': node_id,
                'target_nodes': list(node.neighbors),
                'spread_amount': spread_amount,
                'timestamp': time.time()
            })
    
    def get_attention_statistics(self) -> Dict:
        """Get statistics about current attention allocation"""
        if not self.nodes:
            return {}
        
        attention_values = [node.get_total_attention() for node in self.nodes.values()]
        
        return {
            'total_nodes': len(self.nodes),
            'mean_attention': sum(attention_values) / len(attention_values),
            'max_attention': max(attention_values),
            'min_attention': min(attention_values),
            'high_priority_nodes': len([a for a in attention_values if a > 0.7]),
            'medium_priority_nodes': len([a for a in attention_values if 0.4 < a <= 0.7]),
            'low_priority_nodes': len([a for a in attention_values if a <= 0.4]),
            'spreading_events': len(self.spreading_history),
            'resource_allocations': len(self.resource_allocation_history)
        }
    
    def adjust_attention_for_constraint(self, constraint_type: str, affected_nodes: List[str]):
        """Dynamically adjust attention based on new constraints or objectives"""
        urgency_boost = 0.2
        
        for node_id in affected_nodes:
            node = self.nodes.get(node_id)
            if node:
                # Boost urgency for constraint-affected nodes
                current_urgency = node.attention_values[AttentionValueType.URGENCY].value
                new_urgency = min(1.0, current_urgency + urgency_boost)
                node.attention_values[AttentionValueType.URGENCY].update(new_urgency)
                
                # Re-add to priority queues
                self._add_to_priority_queues(node_id)

# Example optimization task functions
def mock_formulation_optimization(params: Dict) -> Dict:
    """Mock optimization function for testing"""
    formulation = params['formulation']
    attention_weight = params.get('attention_weight', 0.5)
    
    # Simulate optimization with random results
    efficacy = random.uniform(0.3, 0.9) * attention_weight
    cost = random.uniform(10, 100)
    stability = random.uniform(0.4, 0.95)
    
    return {
        'efficacy': efficacy,
        'cost': cost,
        'stability': stability,
        'optimization_time': random.uniform(1, 10),
        'converged': random.choice([True, False])
    }

def mock_safety_optimization(params: Dict) -> Dict:
    """Mock safety optimization function"""
    formulation = params['formulation']
    
    safety_score = random.uniform(0.6, 1.0)
    regulatory_compliance = random.choice([True, False])
    
    return {
        'safety_score': safety_score,
        'regulatory_compliance': regulatory_compliance,
        'efficacy': random.uniform(0.4, 0.8),
        'cost': random.uniform(15, 80)
    }

# Example usage
def example_attention_allocation():
    """Example demonstrating attention allocation system"""
    print("=== Attention Allocation System Example ===")
    
    manager = AttentionAllocationManager(max_active_nodes=50)
    
    # Add some example formulations
    example_formulations = [
        {
            'ingredients': {'hyaluronic_acid': 1.0, 'niacinamide': 5.0, 'glycerin': 3.0},
            'type': 'hydrating_serum'
        },
        {
            'ingredients': {'retinol': 0.5, 'hyaluronic_acid': 0.8, 'glycerin': 2.0},
            'type': 'anti_aging_serum'
        },
        {
            'ingredients': {'vitamin_c': 10.0, 'vitamin_e': 0.5, 'glycerin': 4.0},
            'type': 'antioxidant_serum'
        },
        {
            'ingredients': {'niacinamide': 3.0, 'zinc_oxide': 15.0, 'glycerin': 2.0},
            'type': 'acne_treatment'
        }
    ]
    
    # Add formulations to attention network
    node_ids = []
    for formulation in example_formulations:
        node_id = manager.add_formulation_node(formulation)
        node_ids.append(node_id)
        print(f"Added formulation: {node_id}")
    
    print(f"\nInitial attention statistics:")
    stats = manager.get_attention_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Define optimization tasks
    optimization_tasks = [mock_formulation_optimization, mock_safety_optimization]
    
    # Allocate resources and run optimization
    print(f"\nRunning attention-based optimization...")
    results = manager.allocate_computational_resources(
        optimization_tasks, 
        time_budget=60.0  # 1 minute
    )
    
    print(f"\nOptimization results:")
    for node_id, result in results:
        node = manager.nodes[node_id]
        print(f"  {node_id}: efficacy={result.get('efficacy', 0):.3f}, attention={node.get_total_attention():.3f}")
    
    print(f"\nFinal attention statistics:")
    final_stats = manager.get_attention_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate constraint-based attention adjustment
    print(f"\nAdjusting attention for regulatory constraint...")
    manager.adjust_attention_for_constraint("regulatory_compliance", node_ids[:2])
    
    updated_stats = manager.get_attention_statistics()
    print(f"High priority nodes after constraint: {updated_stats['high_priority_nodes']}")
    
    print("\n=== Attention Allocation Example Complete ===")

if __name__ == "__main__":
    example_attention_allocation()
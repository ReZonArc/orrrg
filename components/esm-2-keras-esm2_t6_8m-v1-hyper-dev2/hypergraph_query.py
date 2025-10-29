#!/usr/bin/env python3
"""
Hypergraph Query Utilities
Provides tools to query and analyze the ESM-2 hypergraph structure.
"""

import json
import argparse
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from esm2_hypergraph import ESM2Hypergraph, create_esm2_hypergraph


class HypergraphQueryEngine:
    """Query engine for hypergraph analysis"""
    
    def __init__(self, hypergraph: ESM2Hypergraph):
        self.hypergraph = hypergraph
    
    def find_nodes_by_type(self, node_type: str) -> List[str]:
        """Find all nodes of a specific type"""
        return [
            node_id for node_id, node in self.hypergraph.nodes.items()
            if node.type == node_type
        ]
    
    def find_nodes_by_layer(self, layer_idx: int) -> List[str]:
        """Find all nodes in a specific layer"""
        return [
            node_id for node_id, node in self.hypergraph.nodes.items()
            if node.layer_idx == layer_idx
        ]
    
    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get all nodes that this node depends on (inputs)"""
        dependencies = []
        for edge in self.hypergraph.edges.values():
            if node_id in edge.target_nodes:
                dependencies.extend(edge.source_nodes)
        return list(set(dependencies))
    
    def get_node_dependents(self, node_id: str) -> List[str]:
        """Get all nodes that depend on this node (outputs)"""
        dependents = []
        for edge in self.hypergraph.edges.values():
            if node_id in edge.source_nodes:
                dependents.extend(edge.target_nodes)
        return list(set(dependents))
    
    def get_computational_path(self, start_node: str, end_node: str) -> Optional[List[str]]:
        """Find computational path between two nodes"""
        from collections import deque
        
        if start_node == end_node:
            return [start_node]
        
        queue = deque([(start_node, [start_node])])
        visited = {start_node}
        
        while queue:
            current, path = queue.popleft()
            
            # Get next nodes
            next_nodes = self.get_node_dependents(current)
            
            for next_node in next_nodes:
                if next_node == end_node:
                    return path + [next_node]
                
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))
        
        return None
    
    def analyze_parameter_flow(self) -> Dict[str, Any]:
        """Analyze parameter flow through the network"""
        param_count = {}
        
        for node_id, node in self.hypergraph.nodes.items():
            if node.type == "linear":
                input_size = node.input_shape[-1] if node.input_shape else 0
                output_size = node.output_shape[-1] if node.output_shape else 0
                
                # Calculate parameters (weights + bias if used)
                params = input_size * output_size
                if node.parameters.get("use_bias", False):
                    params += output_size
                
                param_count[node_id] = params
            elif node.type == "embedding":
                vocab_size = node.parameters.get("vocab_size", 0)
                embed_dim = node.parameters.get("embed_dim", 0)
                param_count[node_id] = vocab_size * embed_dim
            else:
                param_count[node_id] = 0
        
        total_params = sum(param_count.values())
        
        return {
            "total_parameters": total_params,
            "parameters_by_node": param_count,
            "parameters_by_layer": self._group_params_by_layer(param_count)
        }
    
    def _group_params_by_layer(self, param_count: Dict[str, int]) -> Dict[int, int]:
        """Group parameters by layer"""
        layer_params = defaultdict(int)
        
        for node_id, params in param_count.items():
            node = self.hypergraph.nodes[node_id]
            layer_params[node.layer_idx] += params
        
        return dict(layer_params)
    
    def get_attention_structure(self) -> Dict[str, Any]:
        """Analyze the attention structure"""
        attention_nodes = self.find_nodes_by_type("attention")
        
        structure = {
            "num_attention_layers": len(attention_nodes),
            "attention_details": {}
        }
        
        for node_id in attention_nodes:
            node = self.hypergraph.nodes[node_id]
            layer_idx = node.layer_idx
            
            structure["attention_details"][f"layer_{layer_idx}"] = {
                "node_id": node_id,
                "num_heads": node.parameters.get("num_heads", 0),
                "head_dim": node.parameters.get("head_dim", 0),
                "scale": node.parameters.get("scale", 0)
            }
        
        return structure
    
    def find_bottlenecks(self) -> List[Dict[str, Any]]:
        """Find potential bottlenecks in the architecture"""
        bottlenecks = []
        
        # Find nodes with high fan-in or fan-out
        for node_id, node in self.hypergraph.nodes.items():
            fan_in = len(self.get_node_dependencies(node_id))
            fan_out = len(self.get_node_dependents(node_id))
            
            if fan_in > 3 or fan_out > 3:
                bottlenecks.append({
                    "node_id": node_id,
                    "node_name": node.name,
                    "fan_in": fan_in,
                    "fan_out": fan_out,
                    "type": node.type
                })
        
        return sorted(bottlenecks, key=lambda x: x["fan_in"] + x["fan_out"], reverse=True)
    
    def export_subgraph(self, layer_start: int, layer_end: int) -> Dict[str, Any]:
        """Export a subgraph for specific layers"""
        subgraph_nodes = {}
        subgraph_edges = {}
        
        # Get nodes in the specified layer range
        for node_id, node in self.hypergraph.nodes.items():
            if layer_start <= node.layer_idx <= layer_end:
                subgraph_nodes[node_id] = node
        
        # Get edges that connect nodes within the subgraph
        for edge_id, edge in self.hypergraph.edges.items():
            if (all(n in subgraph_nodes for n in edge.source_nodes) and
                all(n in subgraph_nodes for n in edge.target_nodes)):
                subgraph_edges[edge_id] = edge
        
        return {
            "nodes": {nid: node.__dict__ for nid, node in subgraph_nodes.items()},
            "edges": {eid: edge.__dict__ for eid, edge in subgraph_edges.items()},
            "layer_range": [layer_start, layer_end]
        }


def main():
    """Main CLI interface for hypergraph queries"""
    parser = argparse.ArgumentParser(description="Query ESM-2 Hypergraph")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--query", choices=[
        "stats", "attention", "params", "bottlenecks", "path", "subgraph"
    ], required=True, help="Query type")
    parser.add_argument("--start", help="Start node for path query")
    parser.add_argument("--end", help="End node for path query")
    parser.add_argument("--layer-start", type=int, help="Start layer for subgraph")
    parser.add_argument("--layer-end", type=int, help="End layer for subgraph")
    parser.add_argument("--type", help="Node type to filter")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        "name": "esm_backbone",
        "trainable": True,
        "vocabulary_size": 33,
        "num_layers": 6,
        "num_heads": 20,
        "hidden_dim": 320,
        "intermediate_dim": 1280,
        "dropout": 0,
        "max_wavelength": 10000,
        "use_bias": True,
        "activation": "gelu",
        "layer_norm_eps": 0.00001,
        "use_pre_layer_norm": False,
        "position_embedding_type": "rotary",
        "max_sequence_length": 1026,
        "pad_token_id": 1
    }
    
    # Create hypergraph and query engine
    hypergraph = create_esm2_hypergraph(config)
    query_engine = HypergraphQueryEngine(hypergraph)
    
    # Execute query
    if args.query == "stats":
        stats = hypergraph.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.query == "attention":
        attention = query_engine.get_attention_structure()
        print(json.dumps(attention, indent=2))
    
    elif args.query == "params":
        params = query_engine.analyze_parameter_flow()
        print(json.dumps(params, indent=2))
    
    elif args.query == "bottlenecks":
        bottlenecks = query_engine.find_bottlenecks()
        print(json.dumps(bottlenecks, indent=2))
    
    elif args.query == "path":
        if not args.start or not args.end:
            print("Error: --start and --end required for path query")
            return
        
        path = query_engine.get_computational_path(args.start, args.end)
        if path:
            print(f"Path from {args.start} to {args.end}:")
            print(" -> ".join(path))
        else:
            print(f"No path found from {args.start} to {args.end}")
    
    elif args.query == "subgraph":
        if args.layer_start is None or args.layer_end is None:
            print("Error: --layer-start and --layer-end required for subgraph query")
            return
        
        subgraph = query_engine.export_subgraph(args.layer_start, args.layer_end)
        print(json.dumps(subgraph, indent=2))


if __name__ == "__main__":
    main()
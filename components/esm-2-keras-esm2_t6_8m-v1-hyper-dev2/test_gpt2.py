#!/usr/bin/env python3
"""
Test suite for GPT-2 hypergraph implementation
"""

import unittest
from gpt2_hypergraph import create_gpt2_hypergraph, GPT2Hypergraph
from gpt2_metagraph import create_gpt2_metagraph, GPT2MetaGraph


class TestGPT2Hypergraph(unittest.TestCase):
    """Test cases for GPT-2 hypergraph implementation"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            "name": "gpt2_test",
            "trainable": True,
            "vocabulary_size": 50257,
            "num_layers": 2,  # Small for testing
            "num_heads": 4,   # Small for testing
            "hidden_dim": 128,
            "intermediate_dim": 512,
            "dropout": 0.1,
            "max_wavelength": 10000,
            "use_bias": True,
            "activation": "gelu",
            "layer_norm_eps": 1e-5,
            "use_pre_layer_norm": True,
            "position_embedding_type": "learned",
            "max_sequence_length": 256,
            "pad_token_id": 50256
        }
    
    def test_hypergraph_creation(self):
        """Test basic hypergraph creation"""
        hypergraph = create_gpt2_hypergraph(self.config)
        
        # Test basic properties
        self.assertIsInstance(hypergraph, GPT2Hypergraph)
        self.assertEqual(hypergraph.vocab_size, 50257)
        self.assertEqual(hypergraph.num_layers, 2)
        self.assertEqual(hypergraph.num_heads, 4)
        self.assertEqual(hypergraph.hidden_dim, 128)
        
        # Test nodes exist
        self.assertGreater(len(hypergraph.nodes), 0)
        self.assertGreater(len(hypergraph.edges), 0)
        
        # Test specific GPT-2 nodes exist
        self.assertIn("token_embedding", hypergraph.nodes)
        self.assertIn("position_embedding", hypergraph.nodes)
        self.assertIn("final_layer_norm", hypergraph.nodes)
        self.assertIn("lm_head", hypergraph.nodes)
    
    def test_transformer_layers(self):
        """Test transformer layer structure"""
        hypergraph = create_gpt2_hypergraph(self.config)
        
        # Test layer nodes exist
        for layer_idx in range(self.config["num_layers"]):
            layer_prefix = f"layer_{layer_idx}"
            
            # Test attention components
            self.assertIn(f"{layer_prefix}_pre_attn_norm", hypergraph.nodes)
            self.assertIn(f"{layer_prefix}_query_proj", hypergraph.nodes)
            self.assertIn(f"{layer_prefix}_key_proj", hypergraph.nodes)
            self.assertIn(f"{layer_prefix}_value_proj", hypergraph.nodes)
            self.assertIn(f"{layer_prefix}_multihead_attn", hypergraph.nodes)
            self.assertIn(f"{layer_prefix}_attn_output_proj", hypergraph.nodes)
            
            # Test FFN components
            self.assertIn(f"{layer_prefix}_pre_ffn_norm", hypergraph.nodes)
            self.assertIn(f"{layer_prefix}_ffn_intermediate", hypergraph.nodes)
            self.assertIn(f"{layer_prefix}_ffn_activation", hypergraph.nodes)
            self.assertIn(f"{layer_prefix}_ffn_output", hypergraph.nodes)
    
    def test_causal_attention(self):
        """Test that causal attention is used"""
        hypergraph = create_gpt2_hypergraph(self.config)
        
        # Check that attention nodes have causal_attention type
        for layer_idx in range(self.config["num_layers"]):
            attn_node = hypergraph.nodes[f"layer_{layer_idx}_multihead_attn"]
            self.assertEqual(attn_node.type, "causal_attention")
            self.assertTrue(attn_node.parameters.get("causal_mask", False))
    
    def test_pre_layer_norm(self):
        """Test that pre-layer normalization is used (GPT-2 style)"""
        hypergraph = create_gpt2_hypergraph(self.config)
        
        # Check for pre-norm nodes
        for layer_idx in range(self.config["num_layers"]):
            self.assertIn(f"layer_{layer_idx}_pre_attn_norm", hypergraph.nodes)
            self.assertIn(f"layer_{layer_idx}_pre_ffn_norm", hypergraph.nodes)
    
    def test_position_embeddings(self):
        """Test learned position embeddings (not rotary like ESM-2)"""
        hypergraph = create_gpt2_hypergraph(self.config)
        
        pos_emb_node = hypergraph.nodes["position_embedding"]
        self.assertEqual(pos_emb_node.type, "embedding")
        self.assertIn("max_position", pos_emb_node.parameters)
    
    def test_statistics(self):
        """Test hypergraph statistics"""
        hypergraph = create_gpt2_hypergraph(self.config)
        stats = hypergraph.get_statistics()
        
        # Test statistics structure
        self.assertIn("total_nodes", stats)
        self.assertIn("total_edges", stats)
        self.assertIn("node_types", stats)
        self.assertIn("edge_types", stats)
        
        # Test node types include GPT-2 specific types
        self.assertIn("causal_attention", stats["node_types"])
        self.assertIn("embedding", stats["node_types"])
        self.assertEqual(stats["node_types"]["embedding"], 2)  # token + position
    
    def test_json_serialization(self):
        """Test JSON serialization works"""
        hypergraph = create_gpt2_hypergraph(self.config)
        
        # Test saving to JSON (should not raise exception)
        try:
            hypergraph.save_to_json("/tmp/test_gpt2_hypergraph.json")
        except Exception as e:
            self.fail(f"JSON serialization failed: {e}")


class TestGPT2MetaGraph(unittest.TestCase):
    """Test cases for GPT-2 metagraph implementation"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            "name": "gpt2_test",
            "trainable": True,
            "vocabulary_size": 50257,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 128,
            "intermediate_dim": 512,
            "dropout": 0.1,
            "max_wavelength": 10000,
            "use_bias": True,
            "activation": "gelu",
            "layer_norm_eps": 1e-5,
            "use_pre_layer_norm": True,
            "position_embedding_type": "learned",
            "max_sequence_length": 256,
            "pad_token_id": 50256
        }
    
    def test_metagraph_creation(self):
        """Test metagraph creation with tensor types"""
        metagraph = create_gpt2_metagraph(self.config)
        
        # Test inheritance from hypergraph
        self.assertIsInstance(metagraph, GPT2MetaGraph)
        self.assertIsInstance(metagraph, GPT2Hypergraph)
        
        # Test tensor type system
        self.assertIsNotNone(metagraph.shape_registry)
        self.assertIsInstance(metagraph.tensor_bundles, dict)
        self.assertIsInstance(metagraph.typed_edges, dict)
    
    def test_tensor_types(self):
        """Test tensor shape types are created"""
        metagraph = create_gpt2_metagraph(self.config)
        
        # Check that nodes have tensor types
        for node in metagraph.nodes.values():
            self.assertIsNotNone(node.input_shape_type)
            self.assertIsNotNone(node.output_shape_type)
    
    def test_tensor_bundles(self):
        """Test tensor bundles are created"""
        metagraph = create_gpt2_metagraph(self.config)
        
        # Should have tensor bundles
        self.assertGreater(len(metagraph.tensor_bundles), 0)
        
        # Check bundle properties
        for bundle in metagraph.tensor_bundles.values():
            self.assertIsInstance(bundle.fiber_nodes, set)
            self.assertGreater(len(bundle.fiber_nodes), 0)
            self.assertIn(bundle.computational_mode, ["spatial_concurrent", "temporal_asymmetric"])
            self.assertIn(bundle.operator_type, ["product_grammar", "prime_index_grammar"])
    
    def test_typed_edges(self):
        """Test typed edges are created"""
        metagraph = create_gpt2_metagraph(self.config)
        
        # Should have typed edges
        self.assertGreater(len(metagraph.typed_edges), 0)
        
        # Check typed edge properties
        for typed_edge in metagraph.typed_edges.values():
            self.assertIsInstance(typed_edge.input_types, list)
            self.assertIsInstance(typed_edge.output_types, list)
            self.assertIn(typed_edge.transformation_type, ["identity", "fusion", "split", "transformation"])
            self.assertGreaterEqual(typed_edge.compatibility_score, 0.0)
            self.assertLessEqual(typed_edge.compatibility_score, 1.0)
    
    def test_metagraph_statistics(self):
        """Test metagraph statistics"""
        metagraph = create_gpt2_metagraph(self.config)
        stats = metagraph.get_metagraph_statistics()
        
        # Test extended statistics structure
        self.assertIn("tensor_types", stats)
        self.assertIn("optimization_config", stats)
        self.assertIn("tensor_bundles", stats)
        self.assertIn("type_compatibility", stats)
        
        # Test specific metrics
        self.assertGreater(stats["tensor_types"]["total_shape_types"], 0)
        self.assertGreater(stats["tensor_bundles"]["total_bundles"], 0)
        self.assertGreaterEqual(stats["type_compatibility"]["average_compatibility_score"], 0.0)
    
    def test_json_serialization(self):
        """Test metagraph JSON serialization"""
        metagraph = create_gpt2_metagraph(self.config)
        
        # Test saving to JSON (should not raise exception)
        try:
            metagraph.save_metagraph_to_json("/tmp/test_gpt2_metagraph.json")
        except Exception as e:
            self.fail(f"Metagraph JSON serialization failed: {e}")


class TestGPT2vsESM2Differences(unittest.TestCase):
    """Test architectural differences between GPT-2 and ESM-2"""
    
    def setUp(self):
        """Set up configurations for both models"""
        self.gpt2_config = {
            "name": "gpt2_test",
            "vocabulary_size": 50257,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 128,
            "intermediate_dim": 512,
            "dropout": 0.1,
            "use_bias": True,
            "activation": "gelu",
            "layer_norm_eps": 1e-5,
            "use_pre_layer_norm": True,
            "position_embedding_type": "learned",
            "max_sequence_length": 256,
            "pad_token_id": 50256,
            "trainable": True,
            "max_wavelength": 10000
        }
        
        self.esm2_config = {
            "name": "esm2_test",
            "vocabulary_size": 33,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 128,
            "intermediate_dim": 512,
            "dropout": 0.0,
            "use_bias": True,
            "activation": "gelu",
            "layer_norm_eps": 1e-5,
            "use_pre_layer_norm": False,
            "position_embedding_type": "rotary",
            "max_sequence_length": 256,
            "pad_token_id": 1,
            "trainable": True,
            "max_wavelength": 10000
        }
    
    def test_attention_differences(self):
        """Test attention mechanism differences"""
        from esm2_hypergraph import create_esm2_hypergraph
        
        gpt2_hypergraph = create_gpt2_hypergraph(self.gpt2_config)
        esm2_hypergraph = create_esm2_hypergraph(self.esm2_config)
        
        # GPT-2 should have causal attention
        gpt2_attn = gpt2_hypergraph.nodes["layer_0_multihead_attn"]
        self.assertEqual(gpt2_attn.type, "causal_attention")
        self.assertTrue(gpt2_attn.parameters.get("causal_mask", False))
        
        # ESM-2 should have bidirectional attention
        esm2_attn = esm2_hypergraph.nodes["layer_0_multihead_attn"]
        self.assertEqual(esm2_attn.type, "attention")
        self.assertNotIn("causal_mask", esm2_attn.parameters)
    
    def test_position_embedding_differences(self):
        """Test position embedding differences"""
        from esm2_hypergraph import create_esm2_hypergraph
        
        gpt2_hypergraph = create_gpt2_hypergraph(self.gpt2_config)
        esm2_hypergraph = create_esm2_hypergraph(self.esm2_config)
        
        # GPT-2 should have learned position embeddings
        gpt2_pos = gpt2_hypergraph.nodes["position_embedding"]
        self.assertEqual(gpt2_pos.type, "embedding")
        self.assertIn("max_position", gpt2_pos.parameters)
        
        # ESM-2 should have rotary position encoding
        esm2_pos = esm2_hypergraph.nodes["positional_encoding"]
        self.assertEqual(esm2_pos.type, "positional")
        self.assertIn("max_wavelength", esm2_pos.parameters)
    
    def test_layer_norm_placement(self):
        """Test layer normalization placement differences"""
        from esm2_hypergraph import create_esm2_hypergraph
        
        gpt2_hypergraph = create_gpt2_hypergraph(self.gpt2_config)
        esm2_hypergraph = create_esm2_hypergraph(self.esm2_config)
        
        # GPT-2 should use pre-layer norm
        self.assertIn("layer_0_pre_attn_norm", gpt2_hypergraph.nodes)
        self.assertIn("layer_0_pre_ffn_norm", gpt2_hypergraph.nodes)
        
        # ESM-2 should use post-layer norm
        self.assertIn("layer_0_post_attn_norm", esm2_hypergraph.nodes)
        self.assertIn("layer_0_post_ffn_norm", esm2_hypergraph.nodes)


if __name__ == "__main__":
    unittest.main()
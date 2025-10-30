# GPT-2 Transformer Implementation using Hypergraph Model

This document describes the implementation of GPT-2 transformer architecture using the hypergraph representation approach, following the same methodology used for the ESM-2 model.

## Overview

The GPT-2 implementation extends the hypergraph-based model representation to support causal language modeling architectures. This provides a structured way to analyze and understand the computational flow in GPT-2 transformers.

## Key Files

- `gpt2_hypergraph.py` - Core GPT-2 hypergraph implementation
- `gpt2_metagraph.py` - Enhanced metagraph with tensor shape types
- `test_gpt2.py` - Comprehensive test suite (16 tests)
- `examples/gpt2_example.py` - Usage examples and comparisons

## Architecture Differences: GPT-2 vs ESM-2

| Component | GPT-2 | ESM-2 |
|-----------|--------|--------|
| **Attention** | Causal/Masked | Bidirectional |
| **Position Encoding** | Learned Embeddings | Rotary (RoPE) |
| **Layer Normalization** | Pre-norm (before attention/FFN) | Post-norm (after attention/FFN) |
| **Vocabulary** | 50,257 tokens (text) | 33 amino acids (protein) |
| **Use Case** | Text generation | Protein understanding |
| **Attention Mask** | Lower triangular (causal) | Full attention matrix |

## Implementation Details

### 1. Hypergraph Structure

The GPT-2 hypergraph consists of:

```
Total Nodes: 149 (for 12-layer GPT-2 Small)
Total Hyperedges: 112
Node Types:
- embedding: 2 (token + position)
- dropout: 25 
- layernorm: 25 (pre-norm style)
- linear: 72 (QKV projections + FFN)
- causal_attention: 12 (masked attention)
- activation: 12 (GELU)
- output: 1 (language modeling head)
```

### 2. Key Components

#### Input Processing
```python
# Token embedding
"token_embedding" -> (None, seq_len) -> (None, seq_len, hidden_dim)

# Position embedding (learnable, not rotary)
"position_embedding" -> (None, seq_len) -> (None, seq_len, hidden_dim)

# Embedding dropout
"embedding_dropout" -> (None, seq_len, hidden_dim) -> (None, seq_len, hidden_dim)
```

#### Transformer Layer (GPT-2 Style)
```python
# Pre-attention layer norm (GPT-2 characteristic)
f"layer_{i}_pre_attn_norm" -> (None, seq_len, hidden_dim) -> (None, seq_len, hidden_dim)

# Causal self-attention
f"layer_{i}_multihead_attn" -> type: "causal_attention", causal_mask: True

# Pre-FFN layer norm (GPT-2 characteristic)  
f"layer_{i}_pre_ffn_norm" -> (None, seq_len, hidden_dim) -> (None, seq_len, hidden_dim)

# Feed-forward network
f"layer_{i}_ffn_intermediate" -> (None, seq_len, hidden_dim) -> (None, seq_len, intermediate_dim)
f"layer_{i}_ffn_activation" -> activation: "gelu"
f"layer_{i}_ffn_output" -> (None, seq_len, intermediate_dim) -> (None, seq_len, hidden_dim)
```

#### Output Processing
```python
# Final layer normalization
"final_layer_norm" -> (None, seq_len, hidden_dim) -> (None, seq_len, hidden_dim)

# Language modeling head
"lm_head" -> (None, seq_len, hidden_dim) -> (None, seq_len, vocab_size)
```

### 3. Hyperedge Types

The implementation uses several types of hyperedges to represent data flow:

- **data_flow**: Basic tensor flow between nodes
- **residual**: Skip connections around attention and FFN blocks
- **attention_prep**: Preparation for attention computation (QKV projections)
- **attention**: Multi-head attention computation
- **feed_forward**: FFN computation flow

### 4. MetaGraph with Tensor Types

The GPT-2 metagraph extends the basic hypergraph with:

```python
Tensor Shape Type System:
- Total Shape Types: 4
- Unique Mathematical Structures: 4  
- Nodes with Types: 298

Optimization Configuration:
- Operator Types: product_grammar (4 types)
- Computational Modes: spatial_concurrent (4 types)
- Topological Classes: euclidean_space (1), projective_space (3)

Type Compatibility Analysis:
- Average Compatibility Score: 0.929
- Transformation Types: Identity (51), Fusion (49), Split (12)
```

## Usage Examples

### Basic GPT-2 Hypergraph
```python
from gpt2_hypergraph import create_gpt2_hypergraph

config = {
    "name": "gpt2_small",
    "vocabulary_size": 50257,
    "num_layers": 12,
    "num_heads": 12,
    "hidden_dim": 768,
    "intermediate_dim": 3072,
    "dropout": 0.1,
    "use_bias": True,
    "activation": "gelu",
    "layer_norm_eps": 1e-5,
    "use_pre_layer_norm": True,
    "position_embedding_type": "learned",
    "max_sequence_length": 1024,
    "pad_token_id": 50256,
    "max_wavelength": 10000,
    "trainable": True
}

hypergraph = create_gpt2_hypergraph(config)
print(hypergraph.visualize_summary())
hypergraph.save_to_json("gpt2.json")
```

### Enhanced MetaGraph
```python
from gpt2_metagraph import create_gpt2_metagraph

metagraph = create_gpt2_metagraph(config)
print(metagraph.visualize_metagraph_summary())
metagraph.save_metagraph_to_json("gpt2_metagraph.json")
```

### Accessing Components
```python
# Get specific nodes
token_emb = hypergraph.nodes["token_embedding"]
first_attn = hypergraph.nodes["layer_0_multihead_attn"]
lm_head = hypergraph.nodes["lm_head"]

# Get statistics
stats = hypergraph.get_statistics()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Causal attention layers: {stats['node_types']['causal_attention']}")

# Check causal attention configuration
assert first_attn.type == "causal_attention"
assert first_attn.parameters["causal_mask"] == True
```

## Model Variants

The implementation supports different GPT-2 model sizes:

### GPT-2 Small (117M parameters)
```python
config = {
    "num_layers": 12,
    "num_heads": 12,
    "hidden_dim": 768,
    "intermediate_dim": 3072,
    # ... other config
}
```

### GPT-2 Medium (345M parameters)
```python
config = {
    "num_layers": 24,
    "num_heads": 16, 
    "hidden_dim": 1024,
    "intermediate_dim": 4096,
    # ... other config
}
```

### GPT-2 Large (762M parameters)
```python
config = {
    "num_layers": 36,
    "num_heads": 20,
    "hidden_dim": 1280,
    "intermediate_dim": 5120,
    # ... other config
}
```

### GPT-2 XL (1.5B parameters)
```python
config = {
    "num_layers": 48,
    "num_heads": 25,
    "hidden_dim": 1600,
    "intermediate_dim": 6400,
    # ... other config
}
```

## Testing

Run the comprehensive test suite:

```bash
python test_gpt2.py
```

The test suite includes:
- Basic hypergraph creation (5 tests)
- Transformer layer structure validation (3 tests)
- GPT-2 specific features (causal attention, pre-norm) (3 tests)
- MetaGraph with tensor types (5 tests)

## Comparison with ESM-2

Run the architectural comparison:

```bash
python examples/gpt2_example.py
```

This will show side-by-side analysis of GPT-2 vs ESM-2 architectures, highlighting:

- Different attention mechanisms (causal vs bidirectional)
- Position encoding strategies (learned vs rotary)
- Layer normalization placement (pre-norm vs post-norm)
- Domain-specific optimizations (text vs protein)

## Integration with Main System

The GPT-2 implementation is integrated into the main system:

```bash
python main.py  # Creates both ESM-2 and GPT-2 hypergraphs
```

This generates:
- `gpt2_hypergraph.json` - Full GPT-2 hypergraph data
- `gpt2_metagraph.json` - Enhanced metagraph with tensor types
- Comparative analysis between ESM-2 and GPT-2 architectures

## Future Extensions

Potential extensions to the GPT-2 hypergraph implementation:

1. **Attention Pattern Analysis**: Visualize causal attention patterns
2. **Layer-wise Analysis**: Analyze information flow through transformer layers
3. **Scaling Analysis**: Study how hypergraph complexity scales with model size
4. **Optimization Patterns**: Identify computational bottlenecks through hypergraph analysis
5. **Multi-Modal Extensions**: Extend to GPT-4 style multi-modal architectures

## Benefits of Hypergraph Representation

The hypergraph approach provides several advantages for understanding GPT-2:

1. **Explicit Data Flow**: Clear visualization of tensor flow through the model
2. **Component Isolation**: Easy identification of specific architectural components
3. **Comparative Analysis**: Structured comparison with other transformer variants
4. **Optimization Insights**: Understanding of computational complexity patterns
5. **Educational Value**: Clear representation for learning transformer architectures

This implementation demonstrates how the hypergraph methodology can be extended to different transformer architectures while maintaining consistency and enabling meaningful comparisons.
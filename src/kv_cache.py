"""
KV-Cache Implementation for Transformer Models

This module provides utilities for implementing Key-Value caching in transformer
attention layers. KV-caching is a critical optimization for autoregressive text
generation that avoids redundant computation of previous tokens' keys and values.

Performance Impact:
-------------------
Without KV-cache: O(n²) computation per token (recompute attention for all positions)
With KV-cache: O(n) computation per token (only compute for new token)

For a 15-token generation:
- Non-cached: ~1.5 tok/s (quadratic growth in compute)
- Cached: ~24 tok/s (linear growth in compute)
- Speedup: ~16x

Cache Structure:
----------------
The cache is a nested dictionary:
{
    0: {  # Layer 0
        'key': [batch, num_kv_heads, max_seq_len, head_dim],
        'value': [batch, num_kv_heads, max_seq_len, head_dim]
    },
    1: {...},  # Layer 1
    ...
}

Each layer stores its own key and value tensors, pre-allocated to max_seq_len.
During generation, we fill positions incrementally and track cache_length.

Grouped-Query Attention (GQA):
-------------------------------
For models with GQA (e.g., Mistral), num_kv_heads < num_heads:
- GPT-2: num_kv_heads = num_heads = 12 (standard multi-head attention)
- Mistral: num_kv_heads = 8, num_heads = 32 (grouped-query attention)

The cache stores only num_kv_heads. Use repeat_kv() from cached_generation.py
to expand KV heads to match query heads during attention computation.

JAX Functional Programming:
---------------------------
JAX requires functional programming - arrays are immutable. Updates create new
arrays using jax.lax.dynamic_update_slice rather than in-place modification.

Usage Example:
--------------
    from src.kv_cache import initialize_cache, update_cache, get_cached_kv

    # Initialize cache (GPT-2)
    cache_gpt2 = initialize_cache(
        num_layers=12,
        batch_size=1,
        num_kv_heads=12,  # Same as num_heads for GPT-2
        max_seq_len=1024,
        head_dim=64
    )

    # Initialize cache (Mistral)
    cache_mistral = initialize_cache(
        num_layers=32,
        batch_size=1,
        num_kv_heads=8,   # Fewer than num_heads (32) for GQA
        max_seq_len=4096,
        head_dim=128
    )

    # During generation loop
    for pos in range(seq_len):
        # Compute new K, V for current token
        new_k, new_v = compute_kv(...)

        # Update cache
        cache = update_cache(cache, layer_idx, new_k, new_v, pos)

        # Retrieve all cached keys/values for attention
        cached_k, cached_v = get_cached_kv(cache, layer_idx, pos + 1)
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple

def initialize_cache(num_layers:int,
                     batch_size:int,
                     num_kv_heads:int,      # Changed from num_heads
                     max_seq_len:int,
                     head_dim:int,
                     dtype:jnp.dtype = jnp.float32) -> Dict[int, Dict[str, jnp.ndarray]]:
    """
    Initialize empty KV-cache for all transformer layers.
    
    For Grouped-Query Attention (GQA), uses num_kv_heads instead of num_heads.
    GPT-2: num_kv_heads = num_heads = 12 (standard MHA)
    Mistral: num_kv_heads = 8, num_heads = 32 (GQA)
    
    Args:
        num_layers: Number of transformer layers
        batch_size: Batch size (typically 1 for generation)
        num_kv_heads: Number of KV heads (may be less than query heads for GQA)
        max_seq_len: Maximum sequence length to cache
        head_dim: Dimension of each attention head
        dtype: Data type for cache tensors
    
    Returns:
        Cache dictionary with KV heads (not query heads!)
    """
    
    # Will return a nested dictionary structure
    # {
    #   0: {'key': jnp.zeros([batch, num_kv_heads, max_seq_len, head_dim]),
    #       'value': jnp.zeros([batch, num_kv_heads, max_seq_len, head_dim])},
    #   1: {...}
    #   ...
    #   Till Layer n
    # }
    cache = {}

    # Create cache for each layer
    for layer_idx in range(num_layers):
        cache[layer_idx] = {
            'key': jnp.zeros((batch_size, num_kv_heads, max_seq_len, head_dim), dtype=dtype),
            'value':jnp.zeros((batch_size, num_kv_heads, max_seq_len, head_dim), dtype=dtype)
        }

    return cache

def update_cache(cache, layer_idx, new_keys, new_values, cache_position):
    """
    Update cache with new key/value tensors at specified position.

    Uses JAX's functional programming paradigm - creates a new cache rather
    than modifying in place. Uses jax.lax.dynamic_update_slice for efficient
    updates.

    Args:
        cache: Current cache dictionary
        layer_idx: Which transformer layer to update
        new_keys: New key tensor [batch, num_kv_heads, 1, head_dim]
        new_values: New value tensor [batch, num_kv_heads, 1, head_dim]
        cache_position: Position in sequence to insert (0-indexed)

    Returns:
        New cache dictionary with updated keys/values for specified layer

    Example:
        # After computing attention for token at position 5
        cache = update_cache(cache, layer_idx=0, new_k, new_v, cache_position=5)
    """
    # Extract current arrays
    # Then insert new keys/values at cache position
    # (JAX Functional Programming) -> Create new cache dict, not modifying in place
    # Returns new cache
    
    current_keys = cache[layer_idx]['key']
    current_values = cache[layer_idx]['value']

    updated_keys = jax.lax.dynamic_update_slice(
        current_keys,
        new_keys,
        (0, 0, cache_position, 0)   # Start at the position cache_position
        )
    
    updated_values = jax.lax.dynamic_update_slice(
        current_values,
        new_values,
        (0, 0, cache_position, 0)
    )    
    
    # Create new cache dict
    new_cache = cache.copy()
    new_cache[layer_idx] = {
        'key': updated_keys,
        'value': updated_values
    }

    return new_cache

def get_cached_kv(cache: Dict[int, Dict[str, jnp.ndarray]],
                  layer_idx: int,
                  cache_length: int)-> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Retrieve cached keys and values up to current sequence length.

    The cache is pre-allocated to max_seq_len, but only cache_length positions
    are valid. This function returns the slice containing valid cached data.

    Args:
        cache: Current cache dictionary
        layer_idx: Which transformer layer to retrieve from
        cache_length: Number of valid positions filled so far

    Returns:
        Tuple of (keys, values) where each has shape:
        [batch, num_kv_heads, cache_length, head_dim]

    Example:
        # After generating 10 tokens, retrieve all cached K,V for attention
        cached_k, cached_v = get_cached_kv(cache, layer_idx=0, cache_length=10)
        # For GPT-2: cached_k.shape = [1, 12, 10, 64]
        # For Mistral: cached_k.shape = [1, 8, 10, 128]
    """
    # The cache will always be of length max_seq_len
    # But we might've only filled cache_length positions so far
    # Extract full cache, but return only sliced version uptill filled positions
    
    full_keys = cache[layer_idx]['key']
    full_values = cache[layer_idx]['value']

    # Shape: [batch, num_heads, cache_length, head_dim]
    keys = full_keys[:, :, :cache_length, :]
    values = full_values[:, :, :cache_length, :]

    return keys, values
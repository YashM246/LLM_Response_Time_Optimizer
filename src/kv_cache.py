import jax
import jax.numpy as jnp
from typing import Dict, Tuple

def initialize_cache(num_layers:int,
                     batch_size:int,
                     num_heads:int,
                     max_seq_len:int,
                     head_dim:int,
                     dtype:jnp.dtype = jnp.float32) -> Dict[int, Dict[str, jnp.ndarray]]:
    
    # Will return a nested dictionary structure
    # {
    #   0: {'key': jnp.zeros([batch, num_heads, max_seq_len, head_dim]),
    #       'value': jnp.zeros([batch, num_heads, max_seq_len, head_dim])},
    #   1: {...}
    #   ...
    #   Till Layer n
    # }
    cache = {}

    # Create cache for each layer
    for layer_idx in range(num_layers):
        cache[layer_idx] = {
            'key': jnp.zeros((batch_size, num_heads, max_seq_len, head_dim), dtype=dtype),
            'value':jnp.zeros((batch_size, num_heads, max_seq_len, head_dim), dtype=dtype)
        }

    return cache

def update_cache(cache, layer_idx, new_keys, new_values, cache_position):
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
    # The cache will always be of length max_seq_len
    # But we might've only filled cache_length positions so far
    # Extract full cache, but return only sliced version uptill filled positions
    
    full_keys = cache[layer_idx]['keys']
    full_values = cache[layer_idx]['value']

    # Shape: [batch, num_heads, cache_length, head_dim]
    keys = full_keys[:, :, :cache_length, :]
    values = full_values[:, :, :cache_length, :]

    return keys, values
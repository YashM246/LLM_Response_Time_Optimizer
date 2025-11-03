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
    pass

def update_cache(cache, layer_idx, new_keys, new_values, cache_position):
    # Extract current arrays
    # Then insert new keys/values at cache position
    # (JAX Functional Programming) -> Create new cache dict, not modifying in place
    # Returns new cache
    pass

def get_cached_kv(cache: Dict[int, Dict[str, jnp.ndarray]],
                  layer_idx: int,
                  cache_length: int)-> Tuple[jnp.ndarray, jnp.ndarray]:
    # The cache will always be of length max_seq_len
    # But we might've only filled cache_length positions so far
    # Extract full cache, but return only sliced version uptill filled positions
    pass
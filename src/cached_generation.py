import jax
import jax.numpy as jnp
from typing import Dict, Tuple

def split_heads(x: jnp.ndarray, num_heads: int)-> jnp.ndarray:
    # Split the hidden dimension into multiple attention heads
    #
    # Args:
    #           x: Input Tensor [batch, seq_len, hidden_dim]
    #           num_heads: Number of attention heads
    #
    # Returns:
    #           x_split: Tensor [batch, num_heads, seq_len, head_dim]
    pass

def merge_heads():
    # Merge Attention Heads back int0 hidden dimensions
    #
    # Args:
    #           x: Input Tensor [batch, num_heads, seq_len, head_dim]
    # Returns:
    #           x_merged: Tensor [batch, seq_len, hidden_dim]
    pass

def compute_qkv(hidden_states: jnp.ndarray,
                attn_weights: jnp.ndarray,  # Combined W_qkv weights
                num_heads: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Compute Q, K, V projections and split into heads
    #
    # Args:
    #           hidden_states: Input [batch, seq_len, hidden_dim]
    #           attn_weights: Attention weights [hidden_dim, 3*hidden_dim]
    #                         (GPT-2 combines Q, K, V into single weight matrix)
    #           num_heads: Number of attention heads
    #
    # Returns:
    #           Q: Query [batch, num_heads, seq_len, head_dim]
    #           K: Key [batch, num_heads, seq_len, head_dim]
    #           V: Value [batch, num_heads, seq_len, head_dim]
    pass

def causal_mask(seq_len:int)-> jnp.ndarray:
    # Create causal attention mask (lower triangle)
    #
    # Args:
    #           seq_len: Sequence Length
    # 
    # Returns:
    #           mask: Causal Mask [seq_len, seq_len]
    #                 0.0 for allowed positions, -inf for masked
    pass

def cached_attention(hidden_states: jnp.ndarray,    # [batch, 1, hiddem_dim]
                     attn_weights: jnp.ndarray,     # [hidden_dim, 3*hidden_dim]
                     num_heads: int,
                     cache: dict,
                     layer_idx: int,
                     position: int,
                     use_cache: bool=True) -> Tuple[jnp.ndarray, dict]:
    # Multi-headed Attention with KV-Caching
    #
    # Args:
    #           hidden_states: Input Embeddings [batch, seq_len, hidden_dim]
    #                          During generation, seq_len=1 (single new token)
    #           attn_weights: Combined QKV weight matrix
    #           num_heads: Number of attn heads
    #           cache: KV-Cache from prev positions
    #           layer_idx: Which transformer layer this is
    #           position: Current token position: (0, 1, 2...)
    #           use_cache: Whether to use cache (False for testing)
    #
    # Returns:
    #           output: Attention output [batch, seq_len, hidden_dim]
    #           cache: Updated cache with new K, V
    pass


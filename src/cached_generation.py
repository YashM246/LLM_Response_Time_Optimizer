"""
Cached Text Generation with KV-Cache and JIT Compilation

Supports GPT-2 and Mistral-7B-Instruct-v0.2 with three optimization layers:
  1. KV-Cache            — avoids recomputing past keys/values (O(n) decode vs O(n²))
  2. JIT Compilation     — XLA-compiles core functions via @jax.jit
  3. JIT-step decode     — fuses all 32 layers into one XLA kernel per token (Mistral)

Performance Results:
- GPT-2  (cached vs uncached, same JAX pipeline): 16.32x speedup, 24.45 tok/s
- Mistral (JAX optimized vs PyTorch generate()):   ~2.27x speedup (Python loop baseline)

IMPORTANT — Why the GPT-2 and Mistral speedups differ:
--------------------------------------------------------
The GPT-2 16.32x compared OUR OWN uncached JAX (naive O(n²) recomputation every
step) against our cached JAX (O(n)). Eliminating ~99% of redundant work naturally
gives a large multiplier.

The Mistral comparison is against PyTorch's generate(), which ALREADY uses KV cache
internally. Both sides are O(n). We compete on JIT efficiency and XLA optimization,
not on eliminating quadratic work — fundamentally harder.

JIT-Step Decode (Mistral):
--------------------------
The plain Python decode loop dispatches ~10 JAX ops × 32 layers × 50 tokens ≈ 16,000
individual Python→XLA round trips per generation call. Each dispatch adds ~13 µs of
overhead, totalling ~210 ms/token of pure Python overhead on top of the ~7 ms/token
of real GPU work (A100 memory-bandwidth bound at 2 TB/s × 14 GB weights).

jax.lax.scan was tried first but is slower in practice: XLA's while-loop backend
introduces carry-state materialization overhead between scan steps, and the stacked
KV cache ([32, 1, 8, seq, 128]) creates a 32-step write-chain that prevents XLA from
pipelining layer computations efficiently.

The JIT-step approach (mistral_jit_decode) is the best balance:
  - @jax.jit compiles all 32 layers into ONE XLA kernel per token step.
    XLA sees the full unrolled graph and applies kernel fusion across all layers.
  - Called from a Python loop 50 times (50 dispatches vs 16,000 in plain Python loop).
  - Dict KV cache (no stacking): 32 independent dynamic_update_slice operations,
    one per layer, with no cross-layer array dependency chain.
  - Compact cache: prompt_len + max_new_tokens (not full 32768-position window).

JAX JIT Shape Compilation:
---------------------------
JAX JIT compiles separately for each unique input shape. Warmup must cover all
prompt lengths used in the benchmark with the exact same temperature/top_k to avoid
first-call compilation overhead.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from functools import partial


#####################################
#       Model Configurations
#####################################

GPT2_CONFIG = {
    'model_type': 'gpt2',
    'num_layers': 12,
    'num_heads': 12,
    'num_kv_heads': 12,  # Same as num_heads for standard MHA
    'hidden_dim': 768,
    'head_dim': 64,
    'intermediate_dim': 3072,  # 4 * hidden_dim
    'vocab_size': 50257,
    'max_seq_len': 1024,
    'use_rope': False,
    'use_rms_norm': False,
    'use_swiglu': False,
    'rope_theta': 10000.0,
}

MISTRAL_CONFIG = {
    'model_type': 'mistral',
    'num_layers': 32,
    'num_heads': 32,
    'num_kv_heads': 8,  # GQA: fewer KV heads than Q heads
    'hidden_dim': 4096,
    'head_dim': 128,
    'intermediate_dim': 14336,
    'vocab_size': 32000,
    'max_seq_len': 32768,
    'use_rope': True,
    'use_rms_norm': True,
    'use_swiglu': True,
    'rope_theta': 10000.0,
}

def get_model(model_type:str)  -> dict:
    # Get configuration for specified model type
    if model_type == "gpt2":
        return GPT2_CONFIG
    elif model_type == "mistral":
        return MISTRAL_CONFIG
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: 'gpt2', 'mistral'")
    

########################
# RMS Norm for  Mistral
########################
@jax.jit
def rms_norm(x: jnp.ndarray, weight:jnp.ndarray, eps:float= 1e-6)-> jnp.ndarray:
    #
    # Root Mean Square Normalization
    # Has no bias parameter
    #
    # x * weight / sqrt(mean(x^2) + eps)

    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_normed = x * jax.lax.rsqrt(variance + eps)
    return x_normed*weight


##########################################
#   RoPE (Rotary Position Embeddings)
##########################################

@partial(jax.jit, static_argnums=(0,1,2))
def precompute_rope_frequencies(head_dim:int, max_seq_len:int, theta:float= 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #
    # Precompute rotary position embedding frequencies
    # RoPE encodes position information by rotating query/key vectors

    # First compute inverse freq: 1/(theta^(2i/d)) for i in [0, d/2]
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))

    # Position indices
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)

    # Outer Product [max_seq_len, head_dim//2]
    freqs = jnp.outer(positions, inv_freq)

    # Duplicate for pairing [max_seq_len, head_dim]
    freqs = jnp.concatenate([freqs, freqs], axis=-1)

    return jnp.cos(freqs), jnp.sin(freqs)

@jax.jit
def rotate_half(x:jnp.ndarray)->jnp.ndarray:
    # Rotate half the hidden dimensions of x for RoPE
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return jnp.concatenate([-x2, x1], axis=-1)


@jax.jit
def apply_rotary_pos_emb(q: jnp.ndarray,
                         k: jnp.ndarray,
                         cos: jnp.ndarray,
                         sin: jnp.ndarray,
                         position_ids: jnp.ndarray)-> Tuple[jnp.ndarray, jnp.ndarray]:
    # Apply rotary position embeddings to keys and queries
    #
    # x_rot = x*cos(pos) + rotate_half(x) * sin(pos)
    #
    
    cos_pos = cos[position_ids]
    sin_pos = sin[position_ids]

    # Reshape to broadcast
    cos_pos = cos_pos[:, None, :, :]
    sin_pos = sin_pos[:, None, :, :]

    # Apply rotation
    q_embed = (q*cos_pos) + (rotate_half(q)*sin_pos)
    k_embed = (k*cos_pos) + (rotate_half(k)*sin_pos)

    return q_embed, k_embed

####################
#    SwiGLU Func
####################

@jax.jit
def swiglu(x:jnp.ndarray, gate_proj:jnp.ndarray, up_proj:jnp.ndarray)-> jnp.ndarray:
    #
    # Formula: SwiGLU(x) = Swish(x @ gate_proj) * (x @ up_proj)
    #
    gate = jax.nn.silu(x@gate_proj)
    up = x @ up_proj

    return gate*up


#############################
#  Grouped Query Attention
#############################

@partial(jax.jit, static_argnums=(1,))
def repeat_kv(hidden_states:jnp.ndarray, n_rep:int )->jnp.ndarray:
    #
    # Repeat KV heads to match number of query heads for GQA
    # In GQA, we have fewer KV heads than Q heads
    #
    if n_rep == 1:
        return hidden_states
    
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

    # Expand and repeat
    hidden_states = jnp.expand_dims(hidden_states, axis=2)
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=2)

    # Reshape: [batch, kv_heads*n_reps, seq, head_dim]
    return hidden_states.reshape(batch, num_kv_heads*n_rep, seq_len, head_dim)


################################
#       GPT 2 Components
################################

@partial(jax.jit, static_argnums=(1,))
def split_heads(x: jnp.ndarray, num_heads: int)-> jnp.ndarray:
    # Split the hidden dimension into multiple attention heads
    #
    # Args:
    #           x: Input Tensor [batch, seq_len, hidden_dim]
    #           num_heads: Number of attention heads
    #
    # Returns:
    #           x_split: Tensor [batch, num_heads, seq_len, head_dim]
    
    batch_size, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads

    # Reshape
    x = x.reshape(batch_size, seq_len, num_heads, head_dim)

    # Transpose [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
    x = jnp.transpose(x, (0, 2, 1, 3))

    return x

@jax.jit
def merge_heads(x: jnp.ndarray)-> jnp.ndarray:
    # Merge Attention Heads back int0 hidden dimensions
    #
    # Args:
    #           x: Input Tensor [batch, num_heads, seq_len, head_dim]
    # Returns:
    #           x_merged: Tensor [batch, seq_len, hidden_dim]
    
    batch_size, num_heads, seq_len, head_dim = x.shape

    # Transpose back
    x = jnp.transpose(x, (0, 2, 1, 3))

    # Reshape
    hidden_dim = num_heads*head_dim
    x = x.reshape(batch_size, seq_len, hidden_dim)

    return x

@partial(jax.jit, static_argnums=(3,))
def compute_qkv(hidden_states: jnp.ndarray,
                attn_weights: jnp.ndarray,  # Combined W_qkv weights
                attn_bias: jnp.ndarray,     # Combined bias for Q, K, V
                num_heads: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Compute Q, K, V projections and split into heads
    #
    # Args:
    #           hidden_states: Input [batch, seq_len, hidden_dim]
    #           attn_weights: Attention weights [hidden_dim, 3*hidden_dim]
    #                         (GPT-2 combines Q, K, V into single weight matrix)
    #           attn_bias: Attention bias [3*hidden_dim]
    #           num_heads: Number of attention heads
    #
    # Returns:
    #           Q: Query [batch, num_heads, seq_len, head_dim]
    #           K: Key [batch, num_heads, seq_len, head_dim]
    #           V: Value [batch, num_heads, seq_len, head_dim]

    batch_size, seq_len, hidden_dim = hidden_states.shape

    # Linear Proj: [batch, seq, hidden] @ [hidden, 3*hidden] + [3*hidden]
    #            = [batch, seq, 3*hidden]

    qkv = hidden_states @ attn_weights + attn_bias

    # Split into Q, K, V along last dim: 3 x [batch, seq, hidden]
    Q, K, V = jnp.split(qkv, 3, axis=-1)

    # Split into heads
    Q = split_heads(Q, num_heads)   # [batch, num_heads, seq, hidden_dim]
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)

    return Q, K, V


def compute_qkv_mistral(hidden_states: jnp.ndarray,
                        q_proj: jnp.ndarray,
                        k_proj: jnp.ndarray,
                        v_proj: jnp.ndarray,
                        num_heads: int,
                        num_kv_heads: int,
                        position_ids: jnp.ndarray,
                        cos: jnp.ndarray,
                        sin: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    #
    # Compute Q, K, V for Mistral with RoPE and GQA
    #
    # Args:
    #           hidden_states: Input [batch, seq_len, hidden_dim]
    #           q_proj: Query projection weights [hidden_dim, num_heads*head_dim]
    #           k_proj: Key Projection weights [hidden_dim, num_kv_heads*head_dim]
    #           v_proj: Value Projection weights [hidden_dim, num_kv_heads*head_dim]
    #           num_heads: Number of query heads (32 for Mistral)
    #           num_kv_heads: Number of KV heads (8 for Mistral, for GQA)
    #           position_ids: Position indices [batch, seq_len]
    #           cos: Precomputed RoPE cosines [max_seq_len, head_dim]
    #           sin: Precomputed RoPE sines [max_seq_len, head_dim]
    #
    # Returns:
    #           Q: Query [batch, num_heads, seq_len, head_dim]
    #           K: Key [batch, num_kv_heads, seq_len, head_dim]
    #           V: Value [batch, num_kv_heads, seq_len, head_dim]

    batch_size, seq_len, hidden_dim = hidden_states.shape
    head_dim = hidden_dim // num_heads

    # Separate projections (no bias in Mistral)
    Q = hidden_states @ q_proj
    K = hidden_states @ k_proj
    V = hidden_states @ v_proj

    # Reshape to separate heads
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim)
    K = K.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    V = V.reshape(batch_size, seq_len, num_kv_heads, head_dim)

    # Transpose to [batch, num_heads, seq_len, head_dim]
    Q = jnp.transpose(Q, (0, 2, 1, 3))
    K = jnp.transpose(K, (0, 2, 1, 3))
    V = jnp.transpose(V, (0, 2, 1, 3))

    # Apply RoPE to Q and K
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

    return Q, K, V

def mistral_prefill_layer(hidden_states: jnp.ndarray,
                          layer_params: dict,
                          cache: dict,
                          layer_idx: int,
                          rope_cos: jnp.ndarray,
                          rope_sin: jnp.ndarray,
                          num_heads: int,
                          num_kv_heads: int) -> Tuple[jnp.ndarray, dict]:
    # Process all prompt tokens in one forward pass (batch prefill)
    # Populates the KV cache for all prompt positions simultaneously
    # Much faster than token-by-token prefill for long prompts

    batch_size, seq_len, hidden_dim = hidden_states.shape
    head_dim = hidden_dim // num_heads

    # 1) Pre-attention RMS Norm
    normed = rms_norm(hidden_states, layer_params['input_layernorm']["kernel"])

    # 2) Compute Q, K, V for all prompt tokens at once
    position_ids = jnp.arange(seq_len).reshape(1, seq_len)
    Q, K, V = compute_qkv_mistral(
        normed,
        layer_params['self_attn']['q_proj']['kernel'],
        layer_params['self_attn']['k_proj']['kernel'],
        layer_params['self_attn']['v_proj']['kernel'],
        num_heads,
        num_kv_heads,
        position_ids,
        rope_cos,
        rope_sin
    )

    # 3) Write all prompt K, V into cache in one slice operation
    current_keys = cache[layer_idx]['key']
    current_values = cache[layer_idx]['value']
    updated_keys = jax.lax.dynamic_update_slice(current_keys, K.astype(current_keys.dtype), (0, 0, 0, 0))
    updated_values = jax.lax.dynamic_update_slice(current_values, V.astype(current_values.dtype), (0, 0, 0, 0))
    new_cache = dict(cache)
    new_cache[layer_idx] = {"key": updated_keys, "value": updated_values}
    cache = new_cache

    # 4) Expand KV Heads to match Q heads for GQA
    n_rep = num_heads // num_kv_heads
    K_full = repeat_kv(K, n_rep)
    V_full = repeat_kv(V, n_rep)

    # 5) Attention with causal mask over all prompt tokens
    scores = jnp.matmul(Q, jnp.transpose(K_full, (0, 1, 3, 2))) / jnp.sqrt(head_dim)
    scores = scores + causal_mask(seq_len)
    attn_w = jax.nn.softmax(scores, axis=-1)
    attn_output = merge_heads(jnp.matmul(attn_w, V_full))

    # 6) Output projection + residual
    attn_output = attn_output @ layer_params['self_attn']['o_proj']['kernel']
    hidden_states = hidden_states + attn_output

    # 7) Post-attention RMSNorm + MLP + Residual
    normed = rms_norm(hidden_states, layer_params['post_attention_layernorm']['kernel'])
    output = hidden_states + mlp(normed, layer_params['mlp'], 'mistral')

    return output, cache


@partial(jax.jit, static_argnums=(0,))
def causal_mask(seq_len:int)-> jnp.ndarray:
    # Create causal attention mask (lower triangle)
    #
    # Args:
    #           seq_len: Sequence Length
    # 
    # Returns:
    #           mask: Causal Mask [seq_len, seq_len]
    #                 0.0 for allowed positions, -inf for masked
    
    # Create lower triangular matrix of 1s
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))

    # Convert to attention mask
    mask = jnp.where(mask==0, -1e10, 0.0)

    return mask


def cached_attention(hidden_states: jnp.ndarray,    # [batch, 1, hiddem_dim]
                     attn_weights: jnp.ndarray,     # [hidden_dim, 3*hidden_dim]
                     attn_bias: jnp.ndarray,        # [3*hidden_dim]
                     num_heads: int,
                     cache: dict,
                     layer_idx: int,
                     position: int,
                     config: dict,
                     q_proj: jnp.ndarray= None,
                     k_proj: jnp.ndarray= None,
                     v_proj: jnp.ndarray= None,
                     rope_cos: jnp.ndarray= None,
                     rope_sin: jnp.ndarray= None,
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
    from src.kv_cache import update_cache, get_cached_kv
    
    batch_size, seq_len, hidden_dim = hidden_states.shape
    head_dim = hidden_dim//num_heads

    # Step 1: Compute Q, K, V for New positions only
    model_type = config.get("model_type", "gpt2")

    if model_type == 'mistral':
        # Mistral: Separate Projections + RoPE
        batch_size, seq_len, hidden_dim = hidden_states.shape
        position_ids = jnp.arange(position, position + seq_len).reshape(1, seq_len)

        Q, K_new, V_new = compute_qkv_mistral(hidden_states, q_proj, k_proj, v_proj,
                                              num_heads, config['num_kv_heads'],
                                              position_ids, rope_cos, rope_sin)
        
    else:
        # GPT2: Combined QKV projection
        Q, K_new, V_new = compute_qkv(hidden_states, attn_weights, attn_bias, num_heads)

    if position>0:
        # Step 2: Update cache with new K, V
        cache = update_cache(cache=cache,
                             layer_idx=layer_idx,
                             new_keys=K_new,
                             new_values=V_new,
                             cache_position=position)
        
        # Step 3: Retrieve all cached K, V (from 0 to position)
        K_all, V_all = get_cached_kv(cache, layer_idx, cache_length=position+1)
    
    else:
        # First position (position == 0)
        # Always update cache at position 0
        cache = update_cache(cache, layer_idx, K_new, V_new, 0)
        K_all = K_new
        V_all = V_new

    # Step 4: Expand KV Heads for GQA (if Mistral)
    if model_type == "mistral":
        n_rep = num_heads // config["num_kv_heads"]
        K_all = repeat_kv(K_all, n_rep)
        V_all = repeat_kv(V_all, n_rep)

    # Step 5: Compute attention scores
    # Q: [batch, num_heads, 1, head_dim]
    # K_all: [batch, num_heads, position+1, head_dim]
    # scores: [batch, num_heads, 1, position+1]
    scores = jnp.matmul(Q, jnp.transpose(K_all, (0, 1, 3, 2)))
    scores = scores/jnp.sqrt(head_dim)

    # Step 6: Apply Causal mask (optional, but good practice)
    # Since we only attend to past positions, mask is already satisfied
    # But add it for correctness
    _, _, query_len, _ = Q.shape
    _, _, kv_len, _ = K_all.shape

    if query_len == 1 and kv_len > 1:
        # Cached generation: query is single token, attending to multiple cached positions
        # Use last row of causal mask
        mask = causal_mask(kv_len)
        mask = mask[-1:, :]  # [1, kv_len]
    else:
        # Non-cached or first token: query and kv have same length
        mask = causal_mask(query_len)
        # Slice to match shapes
        mask = mask[:query_len, :kv_len]  # [query_len, kv_len]

    scores = scores + mask

    # Step 7: Softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)

    # Step 8: Attention Output
    # attn_weights: [batch, num_heads, 1, position+1]
    # V_all: [batch, num_heads, position+1, head_dim]
    # output: [batch, num_heads, 1, head_dim]
    output = jnp.matmul(attn_weights, V_all)

    # Step 9: Merge Heads back
    output = merge_heads(output)

    return output, cache

@partial(jax.jit, static_argnums=(3,))
def batch_attention(hidden_states: jnp.ndarray,
                    attn_weights: jnp.ndarray,
                    attn_bias: jnp.ndarray,
                    num_heads: int) -> jnp.ndarray:
    
    # Multi head attention for batch processing without cache
    # Process all tokens in parallel

    batch_size, seq_len, hidden_dim = hidden_states.shape
    head_dim = hidden_dim // num_heads

    # Compute Q, K, V for all positions at once
    Q, K, V = compute_qkv(hidden_states, attn_weights, attn_bias, num_heads)
    # Q, K, V : [batch, num_heads, seq_len, head_dim]

    # Compute attention scores
    scores = jnp.matmul(Q, jnp.transpose(K, (0, 1, 3, 2)))
    scores = scores / jnp.sqrt(head_dim)

    # Apply causal mask
    mask = causal_mask(seq_len)
    scores = scores + mask

    # Softmax and apply to values
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attn_weights, V)

    # Merge Heads
    output = merge_heads(output)

    return output   # [batch, seq_len, hidden_dim]


def get_embeddings(input_ids: jnp.ndarray,
                   params: dict,
                   position:int= None,
                   model_type:str="gpt2")-> jnp.ndarray:
    # Get token embeddings from input token IDs
    #
    # Args:
    #           input_ids: Token IDs [batch, seq_len]
    #           params: Model Parameters (converted JAX params)
    #           position: Starting position (for KV-cache, position offset)
    #           model_type: 'gpt2' or 'mistral'
    #
    # Returns:
    #           embeddings: Token embeddings [batch, seq_len, hidden_dim]
    #
    # 1) Extract embedding weights from params
    # 2) For GPT2: params['params']['transformer']['wte']['embedding']
    # 3) Use jnp.take() or embeddings[input_ids] to lookup
    # 4) Add position embeddings for GPT2
    
    if model_type == "gpt2":
        # Token Embedding
        token_emb = params['params']['transformer']['wte']['embedding'][input_ids]
        # token_emb: [batch, seq_len, hidden_dim]

        if position is not None:
            # Single position mode (Cached)
            pos_emb = params['params']['transformer']['wpe']['embedding'][position]
            pos_emb = pos_emb[None, None, :]  # [1, 1, hidden_dim]
        else:
            # Batch Mode - create posn array [0, 1, 2, ..., seq_len-1]
            seq_len = input_ids.shape[1]
            positions = jnp.arange(seq_len)
            pos_emb = params['params']['transformer']['wpe']['embedding'][positions]
            pos_emb = pos_emb[None, :, :]  # [1, seq_len, hidden_dim]
        
        embeddings = token_emb + pos_emb
        return embeddings
    
    elif model_type == "mistral":
        # Mistral has no learned positional embeddings (RoPE handles positions)
        token_emb = params['params']['model']['embed_tokens']['embedding'][input_ids]
        return token_emb

@jax.jit
def layer_norm(x: jnp.ndarray,
               gamma: jnp.ndarray,  # Scale Parameter
               beta: jnp.ndarray,   # Shift Parameter
               eps:float= 1e-5)-> jnp.ndarray:
    # Layer Normalization
    #
    # Args:
    #           x: Input [batch, seq_len, hidden_dim]
    #           gamma: Scale [hidden_dim]
    #           beta: Bias [hidden_dim]
    #           eps: Small constant for numerical stability
    #
    # Returns:
    #           normalized: [batch, seq_len, hidden_dim]
    
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)

    # Normalize
    x_norm = (x - mean) / jnp.sqrt(variance + eps)

    # Scale and shift
    output = gamma * x_norm + beta

    return output

@partial(jax.jit, static_argnums=(2,))
def mlp(x: jnp.ndarray,
        mlp_params: dict,
        model_type: str= "gpt2")-> jnp.ndarray:
    # MLP (feed-forward network) with GeLU activation
    #
    # Args:
    #           x: Input [batch, seq_len, hidden_dim]
    #           mlp_params: MLP weights (c_fc, c_proj for GPT2)
    #           model_type: "gpt2" or "mistral"
    #
    # Returns:
    #           output: [batch, seq_len, hidden_dim]
    # 
    # Structure for GPT2:
    #       1) Linear: hidden_dim -> 4*hidden_dim (expansion)
    #       2) GELu activation
    #       3) Linear: 4*hidden_dim -> hidden_dim (projection)

    if model_type == "gpt2":
        # Expansion
        c_fc_weight = mlp_params['c_fc']['kernel']  # [3072, 768] - transposed
        c_fc_weight = c_fc_weight.T  # Transpose to [768, 3072] for correct shape
        c_fc_bias = mlp_params['c_fc']['bias']

        hidden = x @ c_fc_weight + c_fc_bias    # [batch, seq, 4*hidden]

        # GeLU
        hidden = jax.nn.gelu(hidden)

        # Projection
        c_proj_weight = mlp_params['c_proj']['kernel']  # [768, 3072] - transposed
        c_proj_weight = c_proj_weight.T  # Transpose to [3072, 768] for correct shape
        c_proj_bias = mlp_params['c_proj']['bias']

        output = hidden @ c_proj_weight + c_proj_bias
    
    elif model_type == "mistral":
        # Mistral: SwiGLU activation
        # Structure:
        #       1) gate_proj: hidden_dim -> intermediate_dim
        #       2) up_proj: hidden_dim -> intermediate_dim
        #       3) SwiGLU: gate * SiLU(up)
        #       4) down_proj: intermediate_dim -> hidden_dim

        gate_proj_weight = mlp_params['gate_proj']['kernel']
        up_proj_weight = mlp_params['up_proj']['kernel']
        down_proj_weight = mlp_params['down_proj']['kernel']

        # SwiGLU activation (no bias in Mistral)
        hidden = swiglu(x, gate_proj_weight, up_proj_weight)

        # Down Projection
        output = hidden @ down_proj_weight
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return output

def transformer_layer(hidden_states: jnp.ndarray,
                     layer_params: dict,
                     cache: dict,
                     layer_idx: int,
                     position: int,
                     num_heads: int,
                     config: dict,
                     rope_cos: jnp.ndarray= None,
                     rope_sin: jnp.ndarray= None,
                     use_cache: bool = True,
                     model_type: str = "gpt2")-> Tuple[jnp.ndarray, dict]:
    # Complete transfomer later with cached attention
    #
    # Args:
    #           hidden_states: Input [batch, seq_len, hidden_dim]
    #           layer_params: Layer parameter
    #           cache: KV-cache
    #           layer_idx: Layer Index
    #           position: Current position
    #           num_heads: Number of attention heads
    #           use_cache: Whether to use cache
    #           model_type: "gpt2" or "mistral"
    # 
    # Returns:
    #           output: Layer output [batch, seq_len, hidden_dim]
    #           cache: Updated cache

    if model_type == "gpt2":
        # 1) Pre Layer Norm for Attention
        ln_1_weight = layer_params['ln_1']['kernel']
        ln_1_bias = layer_params['ln_1']['bias']
        normed = layer_norm(hidden_states, ln_1_weight, ln_1_bias)

        # 2) Attention (choose cached or batch based on use_cache)
        attn_weights = layer_params['attn']['c_attn']['kernel']     # [2304, 768] - transposed
        attn_weights = attn_weights.T  # Transpose to [768, 2304] for correct shape
        attn_bias = layer_params['attn']['c_attn']['bias']          # [2304]

        if use_cache:
            # Token-by-token processing with cache
            attn_output, cache = cached_attention(hidden_states=normed,
                                                  attn_weights=attn_weights,
                                                  attn_bias=attn_bias,
                                                  num_heads=num_heads,
                                                  cache=cache,
                                                  layer_idx=layer_idx,
                                                  position=position,
                                                  config = config,
                                                  use_cache=use_cache)
        else:
            # Batch processing without cache (all tokens at once)
            attn_output = batch_attention(hidden_states=normed,
                                         attn_weights=attn_weights,
                                         attn_bias=attn_bias,
                                         num_heads=num_heads)
            # Cache is not used in batch mode
        
        # 3) Attention output projection
        c_proj_weight = layer_params['attn']['c_proj']['kernel']  # [768, 768] - transposed
        c_proj_weight = c_proj_weight.T  # Transpose to [768, 768] for correct orientation
        c_proj_bias = layer_params['attn']['c_proj']['bias']
        attn_output = attn_output @ c_proj_weight + c_proj_bias

        # 4) Residual connection
        hidden_states = hidden_states + attn_output

        # 5) Pre-LayerNorm for MLP
        ln_2_weight = layer_params['ln_2']['kernel']
        ln_2_bias = layer_params['ln_2']['bias']
        normed = layer_norm(hidden_states, ln_2_weight, ln_2_bias)

        # 6) MLP
        mlp_output = mlp(normed, layer_params['mlp'], model_type)

        # 7) Residual connection
        output = hidden_states + mlp_output

    elif model_type == "mistral":
        # 1) Pre-attention RMSNorm
        input_layernorm_weight = layer_params['input_layernorm']['kernel']
        normed = rms_norm(hidden_states, input_layernorm_weight)

        # 2) Attention with separate Q/K/V projections + RoPE
        q_proj = layer_params['self_attn']['q_proj']['kernel']
        k_proj = layer_params['self_attn']['k_proj']['kernel']
        v_proj = layer_params['self_attn']['v_proj']['kernel']

        if use_cache:
            attn_output, cache = cached_attention(hidden_states=normed,
                                                  attn_weights=None,    # Not used for Mistral
                                                  attn_bias= None,       # Not used for Mistral
                                                  num_heads= num_heads,
                                                  cache = cache,
                                                  layer_idx= layer_idx,
                                                  position= position,
                                                  config= config,
                                                  q_proj= q_proj,
                                                  k_proj= k_proj,
                                                  v_proj= v_proj,
                                                  rope_cos= rope_cos,
                                                  rope_sin= rope_sin,
                                                  use_cache= use_cache
                                                  )
            
        else:
            raise NotImplementedError("Batch Attention for Mistral not yet implemented")
        
        # 3) Attention output projection
        o_proj = layer_params['self_attn']['o_proj']['kernel']
        attn_output = attn_output @ o_proj

        # 4) Residual connection
        hidden_states = hidden_states + attn_output

        # 5) Pre-MLP RMSNorm
        post_attention_layernorm_weight = layer_params['post_attention_layernorm']['kernel']
        normed = rms_norm(hidden_states, post_attention_layernorm_weight)

        # 6) MLP with SwiGLU
        mlp_output = mlp(normed, layer_params['mlp'], model_type)

        # 7) Residual connection
        output = hidden_states + mlp_output

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return output, cache

@partial(jax.jit, static_argnums=(2,))
def lm_head(hidden_states: jnp.ndarray,
            params: dict,
            model_type: str= "gpt2")-> jnp.ndarray:
    # Language Model Head
    # Project hidden state to vocabulary logits
    #
    # Args:
    #           hidden_states: [batch, seq_len, hidden_dim]
    #           params: Model_parameters
    #           model_type: "gpt2" or "mistral"
    #
    # Returns:
    #           logits: [batch, seq_len, vocab_size]

    if model_type=="gpt2":
        # Final layer norm
        ln_f_weight = params['params']['transformer']['ln_f']['kernel']
        ln_f_bias = params['params']['transformer']['ln_f']['bias']
        hidden_states = layer_norm(hidden_states, ln_f_weight, ln_f_bias)
        
        # Project to vocabulary
        # GPT-2 ties weights: lm_head uses same weights as token embedding
        wte = params['params']['transformer']['wte']['embedding']  # [vocab_size, hidden_dim]
        logits = hidden_states @ wte.T  # [batch, seq_len, vocab_size]
        
    elif model_type == "mistral":
        # Final RMS Norm
        norm_weight = params['params']['model']['norm']['kernel']
        hidden_states = rms_norm(hidden_states, norm_weight)

        # Project to Vocabulary
        lm_head_weight = params['params']['lm_head']['kernel']
        logits = hidden_states @ lm_head_weight
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return logits

def sample_token(logits: jnp.ndarray,
                 temperature: float=1.0,
                 top_k:int=50,
                 key:jax.random.PRNGKey = None)-> jnp.ndarray:
    # Sample next token from logits
    #
    # Args:
    #           logits: Model output logits [batch, vocab_size]
    #           temperature: Sampling temperature (higher=more random)
    #           top_k: Only sample from top K tokens (0 = no filtering)
    #           key = JAX random key
    #
    # Returns:
    #           next_token: Sampled token ID [batch]

    if key is None:
        key = jax.random.PRNGKey(0)

    # Greedy Decoding
    if temperature==0.0:
        return jnp.argmax(logits, axis=-1)
    
    # Temperature sampling
    logits = logits / temperature

    # Top-k Filtering
    if top_k>0:
        # Get top-k values and indices
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)

        # Sample from top-k
        probs = jax.nn.softmax(top_k_logits, axis=-1)
        sampled_idx = jax.random.categorical(key, jnp.log(probs), axis=-1)

        # Map back to original vocab
        next_token = top_k_indices[jnp.arange(logits.shape[0]), sampled_idx]
    else:
        # Sample from full distribution
        probs = jax.nn.softmax(logits, axis=-1)
        next_token = jax.random.categorical(key, jnp.log(probs), axis=-1)
    
    return next_token


def mistral_scan_decode(params,
                        cache,
                        prefill_logits,
                        prompt_len,
                        max_new_tokens,
                        num_heads,
                        num_kv_heads,
                        rope_cos,
                        rope_sin,
                        temperature,
                        top_k,
                        rng_key):
    """
    Decode phase for Mistral using jax.lax.scan.

    Replaces the Python for-loop over max_new_tokens with a single traced
    XLA computation. The entire 50-step decode is compiled once and dispatched
    in one call, eliminating ~16,000 Python→JAX round trips and enabling
    cross-step XLA optimization (kernel fusion, better memory reuse).

    Key design decisions:
    - KV cache stored as stacked arrays [num_layers, batch, kv_heads, seq, head_dim]
      instead of Python dict, because lax.scan carry must be JAX arrays.
    - Attention uses position masking (future positions → -1e10) instead of dynamic
      slicing, because XLA requires static array shapes throughout the computation.
    - Cache sized to prompt_len + max_new_tokens (compact), not 32768, so the
      masked attention matrix is [1, 32, 1, ~70] not [1, 32, 1, 32768].

    Args:
        params:          Model parameters PyTree
        cache:           Dict KV cache from prefill {layer_idx: {'key':..., 'value':...}}
        prefill_logits:  Logits from last prefill token [batch, vocab_size]
        prompt_len:      Number of prompt tokens (Python int — determines cache start pos)
        max_new_tokens:  Exact number of tokens to generate (Python int — scan length)
        num_heads:       Query heads (32 for Mistral)
        num_kv_heads:    KV heads (8 for Mistral, GQA)
        rope_cos:        Precomputed RoPE cosines [max_seq_len, head_dim]
        rope_sin:        Precomputed RoPE sines [max_seq_len, head_dim]
        temperature:     Sampling temperature (Python float, baked into XLA at trace time)
        top_k:           Top-k filter (Python int, baked into XLA at trace time)
        rng_key:         JAX PRNGKey for sampling

    Returns:
        generated_tokens: JAX array of token IDs [max_new_tokens]
    """
    num_layers = MISTRAL_CONFIG['num_layers']   # 32, Python int → unrolls loop at trace time
    head_dim   = MISTRAL_CONFIG['head_dim']     # 128
    n_rep      = num_heads // num_kv_heads      # 4  (GQA expansion factor)

    # Convert dict cache to stacked JAX arrays.
    # Dict: {0: {'key': [1,8,seq,128], 'value': ...}, 1: {...}, ...}
    # Stacked: [32, 1, 8, seq, 128]  ← all layers in one contiguous array
    cache_k = jnp.stack([cache[i]['key']   for i in range(num_layers)])
    cache_v = jnp.stack([cache[i]['value'] for i in range(num_layers)])
    max_seq_len = cache_k.shape[3]   # compact size = prompt_len + max_new_tokens

    def step_fn(carry, _):
        cache_k, cache_v, logits, cache_pos, rng_key = carry

        # 1. Sample next token from current logits
        rng_key, subkey = jax.random.split(rng_key)
        next_token = sample_token(logits, temperature, top_k, subkey)  # [batch]
        token_2d   = next_token.reshape(1, 1)                          # [batch, 1]

        # 2. Token embedding (no positional embedding — RoPE handles position)
        hidden = params['params']['model']['embed_tokens']['embedding'][token_2d]
        # hidden: [1, 1, 4096]

        # 3. Run through all 32 Mistral layers.
        #    Python for loop is unrolled by XLA at trace time → becomes part of the
        #    static computation graph. All 32 layers fused into one XLA program.
        for layer_idx in range(num_layers):
            layer_p = params['params']['model']['layers'][str(layer_idx)]

            # Pre-attention RMSNorm
            normed = rms_norm(hidden, layer_p['input_layernorm']['kernel'])

            # Q, K, V with RoPE applied at the current position
            position_ids = cache_pos.reshape(1, 1)   # [1, 1] — single token position
            Q, K_new, V_new = compute_qkv_mistral(
                normed,
                layer_p['self_attn']['q_proj']['kernel'],
                layer_p['self_attn']['k_proj']['kernel'],
                layer_p['self_attn']['v_proj']['kernel'],
                num_heads, num_kv_heads,
                position_ids, rope_cos, rope_sin
            )
            # Q: [1, 32, 1, 128]   K_new/V_new: [1, 8, 1, 128]

            # Write new K, V into stacked cache at cache_pos.
            # dynamic_update_slice handles the dynamic (traced) cache_pos.
            layer_k = cache_k[layer_idx]   # [1, 8, max_seq_len, 128]
            layer_v = cache_v[layer_idx]
            updated_k = jax.lax.dynamic_update_slice(
                layer_k, K_new.astype(layer_k.dtype), (0, 0, cache_pos, 0))
            updated_v = jax.lax.dynamic_update_slice(
                layer_v, V_new.astype(layer_v.dtype), (0, 0, cache_pos, 0))
            # Functional update — returns new stacked array
            cache_k = cache_k.at[layer_idx].set(updated_k)
            cache_v = cache_v.at[layer_idx].set(updated_v)

            # GQA: expand 8 KV heads to 32 Q heads
            K_full = repeat_kv(updated_k, n_rep)   # [1, 32, max_seq_len, 128]
            V_full = repeat_kv(updated_v, n_rep)

            # Attention: Q [1,32,1,128] × K_full.T [1,32,128,max_seq_len]
            scores = jnp.matmul(Q, jnp.transpose(K_full, (0, 1, 3, 2)))
            scores = scores / jnp.sqrt(head_dim)   # [1, 32, 1, max_seq_len]

            # Position mask: future positions (> cache_pos) set to -inf.
            # Replaces dynamic slicing — XLA requires static shapes, so we mask
            # instead of slicing the cache to [:cache_pos+1].
            pos_mask = jnp.where(
                jnp.arange(max_seq_len) > cache_pos, -1e10, 0.0)
            scores = scores + pos_mask[None, None, None, :]

            attn_w   = jax.nn.softmax(scores, axis=-1)
            attn_out = merge_heads(jnp.matmul(attn_w, V_full))   # [1, 1, 4096]
            attn_out = attn_out @ layer_p['self_attn']['o_proj']['kernel']

            # Residual + post-attention norm + SwiGLU MLP + residual
            hidden  = hidden + attn_out
            normed2 = rms_norm(hidden, layer_p['post_attention_layernorm']['kernel'])
            hidden  = hidden + mlp(normed2, layer_p['mlp'], 'mistral')

        # 4. LM head: final RMSNorm + vocab projection
        hidden_out = rms_norm(hidden, params['params']['model']['norm']['kernel'])
        new_logits = (hidden_out @ params['params']['lm_head']['kernel'])[:, 0, :]
        # new_logits: [1, 32000]

        new_carry = (cache_k, cache_v, new_logits, cache_pos + 1, rng_key)
        return new_carry, next_token   # output token: [batch]

    # Initial carry: start from prefill logits at position prompt_len
    initial_carry = (
        cache_k,
        cache_v,
        prefill_logits,
        jnp.array(prompt_len, dtype=jnp.int32),
        rng_key
    )

    # Single XLA dispatch for all max_new_tokens decode steps
    _, generated_tokens = jax.lax.scan(
        step_fn, initial_carry, None, length=max_new_tokens)

    # generated_tokens: [max_new_tokens, batch] → [max_new_tokens]
    return generated_tokens.reshape(-1)


def mistral_jit_decode(params,
                       cache,
                       prefill_logits,
                       prompt_len,
                       max_new_tokens,
                       num_heads,
                       num_kv_heads,
                       rope_cos,
                       rope_sin,
                       temperature,
                       top_k,
                       rng_key):
    """
    Decode phase for Mistral using a JIT-compiled single-step function.

    This is the fastest Mistral decode strategy. It avoids both the Python-loop
    dispatch overhead (16,000 round trips) and lax.scan's XLA while-loop overhead
    (carry-state materialisation between scan steps).

    Strategy:
      - A single @jax.jit function (decode_step) captures params and rope frequencies
        as a closure and compiles all 32 Mistral layers into ONE XLA kernel.
      - Python calls decode_step 50 times (one per token) — only 50 dispatches.
      - The dict KV cache is passed in/out of each JIT call as a pytree. JAX keeps
        all cache arrays on-device; only the sampled token scalar crosses to CPU.
      - No stacked [32, 1, 8, seq, 128] array — each layer updates its own
        [1, 8, seq, 128] slice independently, avoiding the .at[i].set() chain
        that hurt lax.scan.

    Why lax.scan was slower:
      XLA's while-loop backend adds ~100 ms/token of overhead for:
        a) Carry-state (full stacked cache) materialised at every loop boundary.
        b) The 32-step .at[layer_idx].set() chain on the stacked cache preventing
           XLA from pipelining layer computations across the boundary.
      The JIT-step approach removes the loop boundary entirely — each 50-dispatch
      call gives XLA a flat, fully-unrolled 32-layer graph to optimise at once.

    Args:
        params:          Model parameters PyTree
        cache:           Dict KV cache from prefill {layer_idx: {'key':..., 'value':...}}
        prefill_logits:  Logits from last prefill token [batch, vocab_size]
        prompt_len:      Number of prompt tokens (determines cache start pos)
        max_new_tokens:  Number of tokens to generate
        num_heads:       Query heads (32 for Mistral)
        num_kv_heads:    KV heads (8 for Mistral, GQA)
        rope_cos:        Precomputed RoPE cosines [max_seq_len, head_dim]
        rope_sin:        Precomputed RoPE sines [max_seq_len, head_dim]
        temperature:     Sampling temperature (baked into JIT at trace time)
        top_k:           Top-k filter (baked into JIT at trace time)
        rng_key:         JAX PRNGKey

    Returns:
        generated_tokens: Python list of token IDs (length ≤ max_new_tokens)
    """
    num_layers = MISTRAL_CONFIG['num_layers']   # 32
    head_dim   = MISTRAL_CONFIG['head_dim']     # 128
    n_rep      = num_heads // num_kv_heads      # 4

    @jax.jit
    def decode_step(cache, logits, cache_pos, rng_key):
        """
        One full decode step: sample token → embed → 32 Mistral layers → LM head.

        The Python for-loop over 32 layers is UNROLLED by JAX at trace time into a
        single flat XLA computation graph. XLA then applies kernel fusion across all
        32 layers — this is the key advantage over separate per-layer JIT calls.

        Each layer updates only its own cache slice [1, 8, seq, 128] via
        dynamic_update_slice (no stacked-array dependency chain).
        """
        # 1. Sample next token from previous step's logits
        rng_key, subkey = jax.random.split(rng_key)
        next_token = sample_token(logits, temperature, top_k, subkey)   # [batch]
        token_2d   = next_token.reshape(1, 1)                           # [batch, 1]

        # 2. Token embedding (RoPE handles position, no positional embedding table)
        hidden = params['params']['model']['embed_tokens']['embedding'][token_2d]
        # hidden: [1, 1, 4096]

        # 3. All 32 Mistral layers — unrolled to a flat XLA graph at trace time.
        for layer_idx in range(num_layers):
            layer_p = params['params']['model']['layers'][str(layer_idx)]

            # Pre-attention RMSNorm
            normed = rms_norm(hidden, layer_p['input_layernorm']['kernel'])

            # Q, K, V with RoPE at the current position
            position_ids = cache_pos.reshape(1, 1)
            Q, K_new, V_new = compute_qkv_mistral(
                normed,
                layer_p['self_attn']['q_proj']['kernel'],
                layer_p['self_attn']['k_proj']['kernel'],
                layer_p['self_attn']['v_proj']['kernel'],
                num_heads, num_kv_heads,
                position_ids, rope_cos, rope_sin
            )
            # Q: [1, 32, 1, 128]   K_new/V_new: [1, 8, 1, 128]

            # Update per-layer cache slice (independent of other layers).
            curr_k = cache[layer_idx]['key']    # [1, 8, max_seq_len, 128]
            curr_v = cache[layer_idx]['value']
            new_k = jax.lax.dynamic_update_slice(
                curr_k, K_new.astype(curr_k.dtype), (0, 0, cache_pos, 0))
            new_v = jax.lax.dynamic_update_slice(
                curr_v, V_new.astype(curr_v.dtype), (0, 0, cache_pos, 0))
            # Functional pytree update — only this layer's arrays change.
            cache = {**cache, layer_idx: {'key': new_k, 'value': new_v}}

            # GQA: expand 8 KV heads → 32 Q heads
            K_full = repeat_kv(new_k, n_rep)   # [1, 32, max_seq_len, 128]
            V_full = repeat_kv(new_v, n_rep)

            # Masked attention over full cache (position masking instead of slicing,
            # since XLA requires static shapes).
            max_seq_len = curr_k.shape[2]
            scores = jnp.matmul(Q, jnp.transpose(K_full, (0, 1, 3, 2)))
            scores = scores / jnp.sqrt(head_dim)
            pos_mask = jnp.where(
                jnp.arange(max_seq_len) > cache_pos, -1e10, 0.0)
            scores = scores + pos_mask[None, None, None, :]

            attn_w   = jax.nn.softmax(scores, axis=-1)
            attn_out = merge_heads(jnp.matmul(attn_w, V_full))   # [1, 1, 4096]
            attn_out = attn_out @ layer_p['self_attn']['o_proj']['kernel']

            # Residual + post-attention norm + SwiGLU MLP + residual
            hidden  = hidden + attn_out
            normed2 = rms_norm(hidden, layer_p['post_attention_layernorm']['kernel'])
            hidden  = hidden + mlp(normed2, layer_p['mlp'], 'mistral')

        # 4. LM head: final RMSNorm + vocab projection
        hidden_out = rms_norm(hidden, params['params']['model']['norm']['kernel'])
        new_logits = (hidden_out @ params['params']['lm_head']['kernel'])[:, 0, :]
        # new_logits: [1, 32000]

        return cache, new_logits, cache_pos + 1, rng_key, next_token

    # ---- Python loop: 50 dispatches (vs 16,000 in plain Python loop) ----
    cache_pos = jnp.array(prompt_len, dtype=jnp.int32)
    logits    = prefill_logits
    generated_tokens = []

    for _ in range(max_new_tokens):
        cache, logits, cache_pos, rng_key, token = decode_step(
            cache, logits, cache_pos, rng_key)
        tok = int(token[0])   # scalar transfer to CPU (forces per-token sync)
        generated_tokens.append(tok)

    return generated_tokens


def generate_text_with_cache(params: dict,
                             tokenizer,
                             prompt: str,
                             max_new_tokens: int=50,
                             temperature: float=1.0,
                             top_k: int= 50,
                             use_cache: bool= True,
                             model_type: str= "gpt2") -> Tuple[str, dict]:
    # Generate text using cached attention
    #
    # Args:
    #           params: Converted JAX model parameters
    #           tokenizer: HuggingFace tokenizer
    #           prompt: Input prompt string
    #           max_new_tokens: Number of tokens to generate
    #           temperature: Sampling temperature
    #           top_k: Top K Sampling
    #           use_cache: Whether to use KV-Cache
    #           model_type: "gpt2" or "mistral"
    #
    # Returns:
    #           generated_text: Full generated string
    #           stats: Generation statistics
    #
    # 1) Encode prompt to token IDs
    # 2) Initialize KV Cache
    # 3) For each posn:
    #       a) Get Embeddings
    #       b) Run thru transf layers with cache
    #       c) Sample next token
    #       d) Append to sequence
    # 4) Decode and return

    import time
    from src.kv_cache import initialize_cache
    
    # Model config
    if model_type == "gpt2":
        config = get_model("gpt2")  # Uses the GPT2_CONFIG defined earlier
    elif model_type == "mistral":
        config = get_model("mistral")  # Uses the MISTRAL_CONFIG defined earlier
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='np')
    input_ids = jnp.array(input_ids)  # Convert to JAX array
    batch_size, prompt_len = input_ids.shape
    
    print(f"Prompt: '{prompt}'")
    print(f"Prompt length: {prompt_len} tokens")
    
    # Initialize cache (only needed for cached mode)
    if use_cache:
        # Mistral: use compact cache sized to actual generation needs.
        # Full window (32768) would make the scan's masked attention matrix
        # [1, 32, 1, 32768] for every decode step — wasteful for 70-token sequences.
        # Compact cache [1, 32, 1, prompt_len+max_new_tokens] is ~500x smaller.
        # GPT-2: use full model max_seq_len (unchanged).
        cache_max_seq = (prompt_len + max_new_tokens) if model_type == "mistral" else config['max_seq_len']
        cache = initialize_cache(
            num_layers=config['num_layers'],
            batch_size=batch_size,
            num_kv_heads=config['num_kv_heads'],
            max_seq_len=cache_max_seq,
            head_dim=config['hidden_dim'] // config['num_heads'],
            dtype=jnp.float16
        )
    else:
        cache = None

    # Precompute RoPE frequencies for Mistral
    if model_type == "mistral":
        rope_cos, rope_sin = precompute_rope_frequencies(
            head_dim=config['head_dim'],
            max_seq_len=config['max_seq_len'],
            theta=config['rope_theta']
        )
    else:
        rope_cos, rope_sin = None, None

    # Track generated tokens
    generated_ids = input_ids.tolist()[0]  # Start with prompt tokens

    # Generation loop
    start_time = time.time()

    # PHASE 1: Prefill (process prompt)
    print("\nPrefill phase (processing prompt)...")

    if model_type == "mistral" and use_cache:
        # Batch prefill: all prompt tokens in one forward pass
        # Populates the KV cache for all prompt positions simultaneously
        print("Mode: Batch prefill (all tokens at once)")
        hidden_states = get_embeddings(input_ids, params, position=None, model_type=model_type)

        for layer_idx in range(config['num_layers']):
            layer_params = params['params']['model']['layers'][str(layer_idx)]
            hidden_states, cache = mistral_prefill_layer(
                hidden_states=hidden_states,
                layer_params=layer_params,
                cache=cache,
                layer_idx=layer_idx,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                num_heads=config['num_heads'],
                num_kv_heads=config['num_kv_heads']
            )

        logits = lm_head(hidden_states, params, model_type)
        logits = logits[:, -1, :]  # logits for the last prompt token

    elif use_cache:
        # GPT-2: token-by-token prefill with cache
        print("Mode: Token-by-token with cache")
        for pos in range(prompt_len):
            token_id = input_ids[:, pos:pos+1]  # [batch, 1]
            hidden_states = get_embeddings(token_id, params, position=pos, model_type=model_type)

            for layer_idx in range(config['num_layers']):
                if model_type == "gpt2":
                    layer_params = params['params']['transformer']['h'][str(layer_idx)]
                elif model_type == "mistral":
                    layer_params = params['params']['model']['layers'][str(layer_idx)]
                hidden_states, cache = transformer_layer(
                    hidden_states=hidden_states,
                    layer_params=layer_params,
                    cache=cache,
                    layer_idx=layer_idx,
                    position=pos,
                    num_heads=config['num_heads'],
                    config=config,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    use_cache=use_cache,
                    model_type=model_type
                )

            if pos == prompt_len - 1:
                logits = lm_head(hidden_states, params, model_type)
                logits = logits[:, -1, :]

    else:
        # Batch prefill (all tokens at once) - no cache
        print("Mode: Batch processing (all tokens at once)")

        # Get embeddings for ALL prompt tokens at once
        hidden_states = get_embeddings(input_ids, params, position=None, model_type=model_type)
        # hidden_states: [batch, prompt_len, hidden_dim]

        # Forward through all layers
        for layer_idx in range(config['num_layers']):
            if model_type == "gpt2":
                layer_params = params['params']['transformer']['h'][str(layer_idx)]
            elif model_type == "mistral":
                layer_params = params['params']['model']['layers'][str(layer_idx)]
            hidden_states, _ = transformer_layer(
                hidden_states=hidden_states,
                layer_params=layer_params,
                cache=None,
                layer_idx=layer_idx,
                position=0,  # Not used in batch mode
                num_heads=config['num_heads'],
                config=config,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                use_cache=False,
                model_type=model_type
            )

        # Get logits for last position only
        logits = lm_head(hidden_states, params, model_type)
        logits = logits[:, -1, :]  # [batch, vocab_size]
    
    print(f"[OK] Prefill complete ({prompt_len} tokens)")
    
    # PHASE 2: Decode (generate new tokens)
    print(f"\nGenerating {max_new_tokens} new tokens...")

    key = jax.random.PRNGKey(42)

    if model_type == "mistral" and use_cache:
        # JIT-step decode: all 32 Mistral layers compiled into one XLA kernel per token.
        # Python calls this JIT function max_new_tokens times (50 dispatches total),
        # versus ~16,000 dispatches in the plain Python loop or a single XLA while-loop
        # (scan) that has carry-state materialisation overhead at every step boundary.
        print("Mode: JIT-step decode (all 32 layers in one XLA kernel per token)")
        generated_scan = mistral_jit_decode(
            params=params,
            cache=cache,
            prefill_logits=logits,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            num_heads=config['num_heads'],
            num_kv_heads=config['num_kv_heads'],
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            temperature=temperature,
            top_k=top_k,
            rng_key=key
        )
        # generated_scan is a Python list of token IDs
        for tok in generated_scan:
            generated_ids.append(tok)
            if tok == tokenizer.eos_token_id:
                print("[OK] EOS token encountered")
                break
        print(f"  Generated {len(generated_ids) - prompt_len}/{max_new_tokens} tokens...")

    else:
        # GPT-2 cached or non-cached: original Python token-by-token loop
        for step in range(max_new_tokens):
            current_pos = prompt_len + step

            # Sample next token from previous step's logits
            key, subkey = jax.random.split(key)
            next_token = sample_token(logits, temperature, top_k, subkey)
            generated_ids.append(int(next_token[0]))

            if next_token[0] == tokenizer.eos_token_id:
                print(f"[OK] EOS token generated at step {step}")
                break

            if use_cache:
                next_token_id = next_token.reshape(1, 1)
                hidden_states = get_embeddings(next_token_id, params, position=current_pos, model_type=model_type)
                for layer_idx in range(config['num_layers']):
                    if model_type == "gpt2":
                        layer_params = params['params']['transformer']['h'][str(layer_idx)]
                    elif model_type == "mistral":
                        layer_params = params['params']['model']['layers'][str(layer_idx)]
                    hidden_states, cache = transformer_layer(
                        hidden_states=hidden_states,
                        layer_params=layer_params,
                        cache=cache,
                        layer_idx=layer_idx,
                        position=current_pos,
                        num_heads=config['num_heads'],
                        config=config,
                        rope_cos=rope_cos,
                        rope_sin=rope_sin,
                        use_cache=True,
                        model_type=model_type
                    )
                logits = lm_head(hidden_states, params, model_type)
                logits = logits[:, -1, :]
            else:
                # Non-cached: reprocess all tokens from scratch (slow, for comparison)
                all_token_ids = jnp.array([generated_ids])
                hidden_states = get_embeddings(all_token_ids, params, position=None, model_type=model_type)
                for layer_idx in range(config['num_layers']):
                    if model_type == "gpt2":
                        layer_params = params['params']['transformer']['h'][str(layer_idx)]
                    elif model_type == "mistral":
                        layer_params = params['params']['model']['layers'][str(layer_idx)]
                    hidden_states, _ = transformer_layer(
                        hidden_states=hidden_states,
                        layer_params=layer_params,
                        cache=None,
                        layer_idx=layer_idx,
                        position=0,
                        num_heads=config['num_heads'],
                        config=config,
                        rope_cos=rope_cos,
                        rope_sin=rope_sin,
                        use_cache=False,
                        model_type=model_type
                    )
                logits = lm_head(hidden_states, params, model_type)
                logits = logits[:, -1, :]

            if (step + 1) % 10 == 0:
                print(f"  Generated {step + 1}/{max_new_tokens} tokens...")
    
    # Decode
    elapsed = time.time() - start_time
    generated_text = tokenizer.decode(generated_ids)
    
    # Stats
    num_generated = len(generated_ids) - prompt_len
    stats = {
        'prompt_length': prompt_len,
        'generated_tokens': num_generated,
        'total_tokens': len(generated_ids),
        'time_elapsed': elapsed,
        'tokens_per_sec': num_generated / elapsed if elapsed > 0 else 0,
        'use_cache': use_cache
    }
    
    return generated_text, stats
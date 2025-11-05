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
    
    batch_size, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads

    # Reshape
    x = x.reshape(batch_size, seq_len, num_heads, head_dim)

    # Transpose [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
    x = jnp.transpose(x, (0, 2, 1, 3))

    return x

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
    
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # Linear Proj: [batch, seq, hidden] @ [hidden, 3*hidden]
    #            = [batch, seq, 3*hidden]

    qkv = hidden_states @ attn_weights

    # Split into Q, K, V along last dim: 3 x [batch, seq, hidden]
    Q, K, V = jnp.split(qkv, 3, axis=-1)

    # Split into heads
    Q = split_heads(Q, num_heads)   # [batch, num_heads, seq, hidden_dim]
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)

    return Q, K, V

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
    from src.kv_cache import update_cache, get_cached_kv
    
    batch_size, seq_len, hidden_dim = hidden_states.shape
    head_dim = hidden_dim//num_heads

    # Step 1: Compute Q, K, V for New positions only
    Q, K_new, V_new = compute_qkv(hidden_states, attn_weights, num_heads)
    # Q: [batch, num_heads, 1, head_dim]
    # K_new: [batch, num_heads, 1, head_dim]
    # V_new: [batch, num_heads, 1, head_dim]

    if use_cache and position>0:
        # Step 2: Update cache with new K, V
        cache = update_cache(cache=cache,
                             layer_idx=layer_idx,
                             new_keys=K_new,
                             new_values=V_new,
                             cache_position=position)
        
        # Step 3: Retrieve all cached K, V (from 0 to position)
        K_all, V_all = get_cached_kv(cache, layer_idx, cache_length=position+1)
    
    else:
        # Either first position or no cache -> Use new K and V
        if use_cache:
            cache = update_cache(cache, layer_idx, K_new, V_new, 0)

        K_all = K_new
        V_all = V_new

    # Step 4: Compute attention scores
    # Q: [batch, num_heads, 1, head_dim]
    # K_all: [batch, num_heads, position+1, head_dim]
    # scores: [batch, num_heads, 1, position+1]
    scores = jnp.matmul(Q, jnp.transpose(K_all, (0, 1, 3, 2)))
    scores = scores/jnp.sqrt(head_dim)

    # Step 5: Apply Causal mask (optional, but good practice)
    # Since we only attend to past positions, mask is already satisfied
    # But add it for correctness
    current_len = position+1
    mask = causal_mask(current_len)
    # For generation with seq_len=1, we need mask for [1,current_len]
    # Take last row of mask
    mask = mask[-1:, :]
    scores = scores + mask

    # Step 6: Softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)

    # Step 7: Attention Output
    # attn_weights: [batch, num_heads, 1, position+1]
    # V_all: [batch, num_heads, position+1, head_dim]
    # output: [batch, num_heads, 1, head_dim]
    output = jnp.matmul(attn_weights, V_all)

    # Step 8: Merge Heads back
    output = merge_heads(output)

    return output, cache

def get_embeddings(input_ids: jnp.ndarray,
                   params: dict,
                   model_type:str="gpt2")-> jnp.ndarray:
    # Get token embeddings from input token IDs
    #
    # Args:
    #           input_ids: Token IDs [batch, seq_len]
    #           params: Model Parameters (converted JAX params)
    #           model_type: 'gpt2' or 'mistral'
    #
    # Returns:
    #           embeddings: Token embeddings [batch, seq_len, hidden_dim]
    #
    # 1) Extract embedding weights from params
    # 2) For GPT2: params['params']['transformer']['wte']['embedding']
    # 3) Use jnp.take() or embeddings[input_ids] to lookup
    # 4) Add position embeddings for GPT2
    pass

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


def forward_pass_single_layer(hidden_states: jnp.ndarray,
                              layer_params: dict,
                              cache: dict,
                              layer_idx: int,
                              position: int,
                              num_heads: int,
                              use_cache: bool=True)-> Tuple[jnp.ndarray, dict]:
    # Forward Pass through a single transformer layer
    #
    # Args:
    #           hidden_states: Input [batch, 1, hidden_dim]
    #           layer_params: Parameters for this layer
    #           cache: KV-cache
    #           layer_idx: Layer Index
    #           position: Current Position
    #           num_heads: Number of attention heads
    #           use_cache: Whether to use cache
    #
    # Returns:
    #           output: Layer output [batch, 1, hidden_dim]
    #           cache: Updated cache
    pass


def generate_text_with_cache(params: dict,
                             tokenizer,
                             prompt: str,
                             max_new_tokens: int=50,
                             temperature: float=1.0,
                             use_cache: bool= True,
                             model_config: dict= None) -> Tuple[str, dict]:
    # Generate text using cached attention
    #
    # Args:
    #           params: Converted JAX model parameters
    #           tokenizer: HuggingFace tokenizer
    #           prompt: Input prompt string
    #           max_new_tokens: Number of tokens to generate
    #           temperature: Sampling temperature
    #           use_cache: Whether to use KV-Cache
    #           model_config: Model configuration (num_layers, num_heads, etc)
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
    pass

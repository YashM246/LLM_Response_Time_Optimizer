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
        # GPT2: Token Embeddings + Position Embeddings

        wte = params['params']['transformer']['wte']['embedding']
        token_embeds = wte[input_ids]

        # Position Embedding
        wpe = params['params']['transformer']['wpe']['embedding']
        batch_size, seq_len = input_ids.shape

        if position is not None:
            # During generation: Use offset posn
            positions = jnp.arange(position, position+seq_len)
        else:
            # During prefill: Use sequential posn
            positions = jnp.arange(seq_len)
        
        position_embeds = wpe[positions]

        # Add token+posn embeddings
        embeddings = token_embeds + position_embeds
    
    else:
        # Mistral uses RoPE (rotary position embeddings)
        # For now, just token embeddings
        embed_tokens = params['params']['model']['embed_tokens']['embedding']
        embeddings = embed_tokens[input_ids]
    
    return embeddings

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
        c_fc_weight = mlp_params['c_fc']['kernel']
        c_fc_bias = mlp_params['c_fc']['bias']

        hidden = x @ c_fc_weight + c_fc_bias    # [batch, seq, 4*hidden]

        # GeLU
        hidden = jax.nn.gelu(hidden)

        # Projection
        c_proj_weight = mlp_params['c_proj']['kernel']
        c_proj_bias = mlp_params['c_proj']['bias']

        output = hidden @ c_proj_weight + c_proj_bias
    
    else:
        # Mistral uses different naming and SwiGLU
        raise NotImplementedError("Mistral MLP not yet implemented")
    
    return output

def transformer_layer(hidden_states: jnp.ndarray,
                     layer_params: dict,
                     cache: dict,
                     layer_idx: int,
                     position: int,
                     num_heads: int,
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

        # 2) Cached Attention
        attn_weights = layer_params['attn']['c_attn']['kernel']     # [2304, 768] - transposed
        attn_weights = attn_weights.T  # Transpose to [768, 2304] for correct shape
        attn_output, cache = cached_attention(hidden_states= normed,
                                              attn_weights= attn_weights,
                                              num_heads= num_heads,
                                              cache= cache,
                                              layer_idx= layer_idx,
                                              position= position,
                                              use_cache= use_cache)
        
        # 3) Attention output projection
        c_proj_weight = layer_params['attn']['c_proj']['kernel']
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

    else:
        raise NotImplementedError("Mistral transformer layer not yet implemented")
    
    return output, cache

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
        
    else:
        raise NotImplementedError("Mistral LM head not yet implemented")
    
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
        config = {
            'num_layers': 12,
            'num_heads': 12,
            'hidden_dim': 768,
            'max_seq_len': 1024,
            'vocab_size': 50257
        }
    else:
        raise NotImplementedError("Mistral config not yet defined")
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='np')
    input_ids = jnp.array(input_ids)  # Convert to JAX array
    batch_size, prompt_len = input_ids.shape
    
    print(f"Prompt: '{prompt}'")
    print(f"Prompt length: {prompt_len} tokens")
    
    # Initialize cache
    if use_cache:
        cache = initialize_cache(
            num_layers=config['num_layers'],
            batch_size=batch_size,
            num_heads=config['num_heads'],
            max_seq_len=config['max_seq_len'],
            head_dim=config['hidden_dim'] // config['num_heads'],
            dtype=jnp.float16  # Match model parameter dtype
        )
    else:
        cache = None
    
    # Track generated tokens
    generated_ids = input_ids.tolist()[0]  # Start with prompt tokens
    
    # Generation loop
    start_time = time.time()
    
    # PHASE 1: Prefill (process prompt)
    # For simplicity, we'll process prompt token-by-token (not optimal but works)
    # Optimal would be parallel processing, but requires batch attention
    
    print("\nPrefill phase (processing prompt)...")
    for pos in range(prompt_len):
        # Get single token
        token_id = input_ids[:, pos:pos+1]  # [batch, 1]
        
        # Get embeddings
        hidden_states = get_embeddings(token_id, params, position=pos, model_type=model_type)
        
        # Forward through all layers
        for layer_idx in range(config['num_layers']):
            layer_params = params['params']['transformer']['h'][str(layer_idx)]
            hidden_states, cache = transformer_layer(
                hidden_states=hidden_states,
                layer_params=layer_params,
                cache=cache,
                layer_idx=layer_idx,
                position=pos,
                num_heads=config['num_heads'],
                use_cache=use_cache,
                model_type=model_type
            )
        
        # LM head (only need logits on last prefill token)
        if pos == prompt_len - 1:
            logits = lm_head(hidden_states, params, model_type)
            logits = logits[:, -1, :]  # [batch, vocab_size]
    
    print(f"✓ Prefill complete ({prompt_len} tokens)")
    
    # PHASE 2: Generate new tokens
    print(f"\nGenerating {max_new_tokens} new tokens...")
    
    key = jax.random.PRNGKey(42)
    
    for step in range(max_new_tokens):
        current_pos = prompt_len + step
        
        # Sample next token
        key, subkey = jax.random.split(key)
        next_token = sample_token(logits, temperature, top_k, subkey)
        
        # Append to sequence
        generated_ids.append(int(next_token[0]))
        
        # Check for EOS
        if next_token[0] == tokenizer.eos_token_id:
            print(f"✓ EOS token generated at step {step}")
            break
        
        # Prepare next input
        next_token_id = next_token.reshape(1, 1)  # [batch, 1]
        
        # Get embeddings
        hidden_states = get_embeddings(next_token_id, params, position=current_pos, model_type=model_type)
        
        # Forward through all layers
        for layer_idx in range(config['num_layers']):
            layer_params = params['params']['transformer']['h'][str(layer_idx)]
            hidden_states, cache = transformer_layer(
                hidden_states=hidden_states,
                layer_params=layer_params,
                cache=cache,
                layer_idx=layer_idx,
                position=current_pos,
                num_heads=config['num_heads'],
                use_cache=use_cache,
                model_type=model_type
            )
        
        # LM head
        logits = lm_head(hidden_states, params, model_type)
        logits = logits[:, -1, :]  # [batch, vocab_size]
        
        # Progress indicator
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
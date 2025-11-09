"""
This debug script is to compare cached vs non cached generation

"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np
from src.model_conversion import load_pytorch_model, convert_pytorch_to_jax, build_flax_pytree
from src.cached_generation import transformer_layer, lm_head
from src.kv_cache import initialize_cache

config = {
    "num_layers": 12,
    "num_heads": 12,
    "hidden_dim": 768,
    'max_seq_len': 1024,
    'vocab_size': 50257
}

def compare_generation_step_by_step():

    print("=" * 80)
    print("DEBUGGING: Cached vs Non-Cached Generation Comparison")
    print("=" * 80)

    # Step 1: Load model
    print("\n[1/5] Loading model...")
    pytorch_state_dict, tokenizer = load_pytorch_model(use_small_model=True)
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    params_tree = build_flax_pytree(jax_state_dict)
    params = {'params': params_tree}

    print(f"Model config: {config}")

    # Step 2: Prepare input
    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt)
    print(f"\n[2/5] Prompt: '{prompt}'")
    print(f"Input IDs: {input_ids}")
    print(f"Prompt length: {len(input_ids)} tokens")

    # Step 3: Initialize TWO separate caches
    print("\n[3/5] Initializing caches...")
    cache_cached = initialize_cache(
        num_layers=config['num_layers'],
        batch_size=1,
        num_heads=config['num_heads'],
        max_seq_len=config['max_seq_len'],
        head_dim=config['hidden_dim'] // config['num_heads'],
        dtype=jnp.float16
    )

    cache_noncached = initialize_cache(
        num_layers=config['num_layers'],
        batch_size=1,
        num_heads=config['num_heads'],
        max_seq_len=config['max_seq_len'],
        head_dim=config['hidden_dim'] // config['num_heads'],
        dtype=jnp.float16
    )

    # Step 4: Process prompt tokens (prefill phase)
    # Both modes should do this identically
    print("\n[4/5] Processing prompt tokens...")
    
    for pos, token_id in enumerate(input_ids):
        print(f"\nPrompt token {pos}: {token_id} ('{tokenizer.decode([token_id])}')")
        
        # Cached Mode
        token_emb_cached = params["params"]['transformer']['wte']['embedding'][token_id]
        pos_emb_cached = params["params"]['transformer']['wpe']['embedding'][pos]
        hidden_cached = token_emb_cached + pos_emb_cached
        hidden_cached = hidden_cached[None, None, :]  # Add batch and seq_len dimensions: [1, 1, hidden_dim]

        # Run through all transf layers
        for layer_idx in range(config['num_layers']):
            layer_params = params['params']['transformer']['h'][str(layer_idx)]
            hidden_cached, cache_cached = transformer_layer(
                hidden_states=hidden_cached,
                layer_params=layer_params,
                cache=cache_cached,
                layer_idx=layer_idx,
                position=pos,
                num_heads=config['num_heads'],
                use_cache=True,
                model_type="gpt2"
            )

        # NON-CACHED MODE
        # Get embeddings
        token_emb_noncached = params['params']['transformer']['wte']['embedding'][token_id]
        pos_emb_noncached = params['params']['transformer']['wpe']['embedding'][pos]
        hidden_noncached = token_emb_noncached + pos_emb_noncached
        hidden_noncached = hidden_noncached[None, None, :]  # Add batch and seq_len dimensions: [1, 1, hidden_dim]
        
        # Run through all transformer layers
        for layer_idx in range(config['num_layers']):
            layer_params = params['params']['transformer']['h'][str(layer_idx)]
            hidden_noncached, cache_noncached = transformer_layer(
                hidden_states=hidden_noncached,
                layer_params=layer_params,
                cache=cache_noncached,
                layer_idx=layer_idx,
                position=pos,
                num_heads=config['num_heads'],
                use_cache=False,
                model_type="gpt2"
            )
        
        # COMPARE
        emb_diff = jnp.max(jnp.abs(token_emb_cached - token_emb_noncached))
        pos_emb_diff = jnp.max(jnp.abs(pos_emb_cached - pos_emb_noncached))
        output_diff = jnp.max(jnp.abs(hidden_cached - hidden_noncached))
        
        print(f"  Token embedding diff: {emb_diff:.6e}")
        print(f"  Position embedding diff: {pos_emb_diff:.6e}")
        print(f"  Final output diff: {output_diff:.6e}")
        
        if output_diff > 1e-5:
            print(f"  ⚠️  WARNING: Outputs diverged at prompt position {pos}!")
            print(f"     Cached output sample: {hidden_cached[0, 0, :5]}")
            print(f"     Non-cached output sample: {hidden_noncached[0, 0, :5]}")

    # Step 5: Generate new tokens (autoregressive generation)
    print("\n[5/5] Generating new tokens...")
    num_new_tokens = 5

    generated_tokens_cached = []
    generated_tokens_noncached = []

    current_token_cached = input_ids[-1]
    current_token_noncached = input_ids[-1]

    for step in range(num_new_tokens):
        position = len(input_ids) + step
        print(f"\n--- Generation step {step + 1}, position {position} ---")

        # CACHED MODE
        print("\nCACHED MODE:")
        
        # Get embeddings
        token_emb_cached = params['params']['transformer']['wte']['embedding'][current_token_cached]
        pos_emb_cached = params['params']['transformer']['wpe']['embedding'][position]
        hidden_cached = token_emb_cached + pos_emb_cached
        hidden_cached = hidden_cached[None, None, :]  # Add batch and seq_len dimensions: [1, 1, hidden_dim]
        
        print(f"  Input token: {current_token_cached} ('{tokenizer.decode([current_token_cached])}')")
        print(f"  Position: {position}")
        print(f"  Hidden state sample: {hidden_cached[0, 0, :5]}")
        
        # Run through all transformer layers
        for layer_idx in range(config['num_layers']):
            layer_params = params['params']['transformer']['h'][str(layer_idx)]
            hidden_cached, cache_cached = transformer_layer(
                hidden_states=hidden_cached,
                layer_params=layer_params,
                cache=cache_cached,
                layer_idx=layer_idx,
                position=position,
                num_heads=config['num_heads'],
                use_cache=True,
                model_type="gpt2"
            )
        
        # Get logits from LM head
        logits_cached = lm_head(hidden_cached, params)
        
        # Greedy sampling (argmax)
        next_token_cached = jnp.argmax(logits_cached[0]).item()
        generated_tokens_cached.append(next_token_cached)
        
        print(f"  Output token: {next_token_cached} ('{tokenizer.decode([next_token_cached])}')")
        print(f"  Top logit value: {jnp.max(logits_cached[0]):.4f}")

        # NON-CACHED MODE
        print("\nNON-CACHED MODE:")
        
        # Get embeddings
        token_emb_noncached = params['params']['transformer']['wte']['embedding'][current_token_noncached]
        pos_emb_noncached = params['params']['transformer']['wpe']['embedding'][position]
        hidden_noncached = token_emb_noncached + pos_emb_noncached
        hidden_noncached = hidden_noncached[None, None, :]  # Add batch and seq_len dimensions: [1, 1, hidden_dim]
        
        print(f"  Input token: {current_token_noncached} ('{tokenizer.decode([current_token_noncached])}')")
        print(f"  Position: {position}")
        print(f"  Hidden state sample: {hidden_noncached[0, 0, :5]}")
        
        # Run through all transformer layers
        for layer_idx in range(config['num_layers']):
            layer_params = params['params']['transformer']['h'][str(layer_idx)]
            hidden_noncached, cache_noncached = transformer_layer(
                hidden_states=hidden_noncached,
                layer_params=layer_params,
                cache=cache_noncached,
                layer_idx=layer_idx,
                position=position,
                num_heads=config['num_heads'],
                use_cache=False,
                model_type="gpt2"
            )
        
        # Get logits from LM head
        logits_noncached = lm_head(hidden_noncached, params)
        
        # Greedy sampling (argmax)
        next_token_noncached = jnp.argmax(logits_noncached[0]).item()
        generated_tokens_noncached.append(next_token_noncached)
        
        print(f"  Output token: {next_token_noncached} ('{tokenizer.decode([next_token_noncached])}')")
        print(f"  Top logit value: {jnp.max(logits_noncached[0]):.4f}")

        # COMPARE
        print("\nCOMPARISON:")
        
        # Compare embeddings
        token_emb_diff = jnp.max(jnp.abs(token_emb_cached - token_emb_noncached))
        pos_emb_diff = jnp.max(jnp.abs(pos_emb_cached - pos_emb_noncached))
        
        # Compare logits
        logits_diff = jnp.max(jnp.abs(logits_cached - logits_noncached))
        logits_mean_diff = jnp.mean(jnp.abs(logits_cached - logits_noncached))
        
        print(f"  Token embedding diff: {token_emb_diff:.6e}")
        print(f"  Position embedding diff: {pos_emb_diff:.6e}")
        print(f"  Logits max diff: {logits_diff:.6e}")
        print(f"  Logits mean diff: {logits_mean_diff:.6e}")
        print(f"  Tokens match: {next_token_cached == next_token_noncached}")
        
        if next_token_cached != next_token_noncached:
            print(f"\n  ❌ DIVERGENCE DETECTED at step {step + 1}!")
            print(f"     Cached generated so far: '{tokenizer.decode(generated_tokens_cached)}'")
            print(f"     Non-cached generated so far: '{tokenizer.decode(generated_tokens_noncached)}'")
            
            # Show top-5 predictions for both
            top5_cached_indices = jnp.argsort(logits_cached[0])[-5:][::-1]
            top5_noncached_indices = jnp.argsort(logits_noncached[0])[-5:][::-1]
            
            print(f"\n     Cached top-5 predictions:")
            for i, tok_idx in enumerate(top5_cached_indices):
                tok_idx_int = int(tok_idx)
                print(f"       {i+1}. Token {tok_idx_int} ('{tokenizer.decode([tok_idx_int])}') - logit: {logits_cached[0, 0, tok_idx]:.4f}")

            print(f"\n     Non-cached top-5 predictions:")
            for i, tok_idx in enumerate(top5_noncached_indices):
                tok_idx_int = int(tok_idx)
                print(f"       {i+1}. Token {tok_idx_int} ('{tokenizer.decode([tok_idx_int])}') - logit: {logits_noncached[0, 0, tok_idx]:.4f}")
            
            # Break on first divergence to make output easier to read
            break
        else:
            print(f"  ✓ Tokens match!")
        
        # Update current tokens for next iteration
        current_token_cached = next_token_cached
        current_token_noncached = next_token_noncached

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Cached generated: '{tokenizer.decode(generated_tokens_cached)}'")
    print(f"Non-cached generated: '{tokenizer.decode(generated_tokens_noncached)}'")
    print(f"Outputs match: {generated_tokens_cached == generated_tokens_noncached}")


if __name__ == "__main__":
    compare_generation_step_by_step()


import sys
sys.path.append('.')

import jax.numpy as jnp
from src.cached_generation import split_heads, merge_heads, compute_qkv, causal_mask, cached_attention
from src.kv_cache import initialize_cache

def test_split_merge_heads():
    """Test split_heads and merge_heads are inverses."""
    print("\n" + "=" * 70)
    print("Test 1: Split and Merge Heads")
    print("=" * 70)
    
    # Create test input: [batch=2, seq=5, hidden=768]
    batch, seq, hidden = 2, 5, 768
    num_heads = 12
    head_dim = hidden // num_heads  # 64
    
    x = jnp.ones((batch, seq, hidden))
    
    # Split heads
    x_split = split_heads(x, num_heads)
    expected_shape = (batch, num_heads, seq, head_dim)
    
    assert x_split.shape == expected_shape, \
        f"Split shape mismatch: {x_split.shape} vs {expected_shape}"
    print(f"✓ split_heads: {x.shape} -> {x_split.shape}")
    
    # Merge heads back
    x_merged = merge_heads(x_split)
    
    assert x_merged.shape == x.shape, \
        f"Merge shape mismatch: {x_merged.shape} vs {x.shape}"
    print(f"✓ merge_heads: {x_split.shape} -> {x_merged.shape}")
    
    # Check values are preserved
    assert jnp.allclose(x, x_merged), "Values not preserved after split/merge"
    print(f"✓ Values preserved after split/merge")
    
    print("✓ Split and merge heads test passed!")

def test_causal_mask():
    """Test causal mask creates lower triangular structure."""
    print("\n" + "=" * 70)
    print("Test 2: Causal Mask")
    print("=" * 70)
    
    seq_len = 4
    mask = causal_mask(seq_len)
    
    # Check shape
    assert mask.shape == (seq_len, seq_len), f"Mask shape: {mask.shape}"
    print(f"✓ Mask shape: {mask.shape}")
    
    # Check structure
    # Position 0 can only attend to position 0 (mask[0,1:] should be -inf)
    assert mask[0, 0] == 0.0, "Position 0,0 should be allowed"
    assert mask[0, 1] < -1e9, "Position 0,1 should be masked"
    print(f"✓ Position 0 can only attend to position 0")
    
    # Position 2 can attend to positions 0,1,2 (mask[2,:3] = 0, mask[2,3] = -inf)
    assert jnp.all(mask[2, :3] == 0.0), "Positions 0-2 should be allowed for position 2"
    assert mask[2, 3] < -1e9, "Position 3 should be masked for position 2"
    print(f"✓ Position 2 can attend to positions 0-2 only")
    
    # Last position can attend to all
    assert jnp.all(mask[-1, :] == 0.0), "Last position should attend to all"
    print(f"✓ Last position can attend to all positions")
    
    print("✓ Causal mask test passed!")

def test_compute_qkv():
    """Test QKV computation with dummy weights."""
    print("\n" + "=" * 70)
    print("Test 3: Compute QKV")
    print("=" * 70)
    
    batch, seq, hidden = 1, 3, 768
    num_heads = 12
    head_dim = hidden // num_heads
    
    # Create dummy input and weights
    hidden_states = jnp.ones((batch, seq, hidden))
    attn_weights = jnp.ones((hidden, 3 * hidden)) * 0.01  # [768, 2304]
    
    # Compute QKV
    Q, K, V = compute_qkv(hidden_states, attn_weights, num_heads)
    
    # Check shapes
    expected_shape = (batch, num_heads, seq, head_dim)
    assert Q.shape == expected_shape, f"Q shape: {Q.shape} vs {expected_shape}"
    assert K.shape == expected_shape, f"K shape: {K.shape} vs {expected_shape}"
    assert V.shape == expected_shape, f"V shape: {V.shape} vs {expected_shape}"
    
    print(f"✓ Q shape: {Q.shape}")
    print(f"✓ K shape: {K.shape}")
    print(f"✓ V shape: {V.shape}")
    
    print("✓ Compute QKV test passed!")


def test_cached_attention():
    """Test cached attention with KV-cache."""
    print("\n" + "=" * 70)
    print("Test 4: Cached Attention")
    print("=" * 70)
    
    # Config
    num_layers = 12
    batch_size = 1
    num_heads = 12
    max_seq_len = 128
    hidden_dim = 768
    head_dim = hidden_dim // num_heads
    
    # Initialize cache
    cache = initialize_cache(
        num_layers=num_layers,
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim
    )
    print(f"✓ Initialized cache for {num_layers} layers")
    
    # Create dummy weights and input
    attn_weights = jnp.ones((hidden_dim, 3 * hidden_dim)) * 0.01
    
    # Test position 0 (first token)
    hidden_states_0 = jnp.ones((batch_size, 1, hidden_dim))
    output_0, cache = cached_attention(
        hidden_states=hidden_states_0,
        attn_weights=attn_weights,
        num_heads=num_heads,
        cache=cache,
        layer_idx=0,
        position=0,
        use_cache=True
    )
    
    assert output_0.shape == (batch_size, 1, hidden_dim), \
        f"Output shape: {output_0.shape}"
    print(f"✓ Position 0 output shape: {output_0.shape}")
    
    # Test position 1 (second token - should use cached K,V from position 0)
    hidden_states_1 = jnp.ones((batch_size, 1, hidden_dim)) * 2.0  # Different values
    output_1, cache = cached_attention(
        hidden_states=hidden_states_1,
        attn_weights=attn_weights,
        num_heads=num_heads,
        cache=cache,
        layer_idx=0,
        position=1,
        use_cache=True
    )
    
    assert output_1.shape == (batch_size, 1, hidden_dim), \
        f"Output shape: {output_1.shape}"
    print(f"✓ Position 1 output shape: {output_1.shape}")
    
    # Verify cache was updated
    from src.kv_cache import get_cached_kv
    K_cached, V_cached = get_cached_kv(cache, layer_idx=0, cache_length=2)
    
    assert K_cached.shape == (batch_size, num_heads, 2, head_dim), \
        f"Cached K shape: {K_cached.shape}"
    assert V_cached.shape == (batch_size, num_heads, 2, head_dim), \
        f"Cached V shape: {V_cached.shape}"
    print(f"✓ Cache updated with 2 positions")
    print(f"  Cached K shape: {K_cached.shape}")
    print(f"  Cached V shape: {V_cached.shape}")
    
    # Test position 2 (third token)
    hidden_states_2 = jnp.ones((batch_size, 1, hidden_dim)) * 3.0
    output_2, cache = cached_attention(
        hidden_states=hidden_states_2,
        attn_weights=attn_weights,
        num_heads=num_heads,
        cache=cache,
        layer_idx=0,
        position=2,
        use_cache=True
    )
    
    assert output_2.shape == (batch_size, 1, hidden_dim), \
        f"Output shape: {output_2.shape}"
    print(f"✓ Position 2 output shape: {output_2.shape}")
    
    # Verify final cache size
    K_cached, V_cached = get_cached_kv(cache, layer_idx=0, cache_length=3)
    assert K_cached.shape == (batch_size, num_heads, 3, head_dim)
    print(f"✓ Final cache has 3 positions")
    
    print("✓ Cached attention test passed!")

def test_attention_without_cache():
    """Test attention works without cache (for comparison)."""
    print("\n" + "=" * 70)
    print("Test 5: Attention Without Cache")
    print("=" * 70)
    
    # Config
    batch_size = 1
    num_heads = 12
    hidden_dim = 768
    
    # Create dummy input and weights
    hidden_states = jnp.ones((batch_size, 1, hidden_dim))
    attn_weights = jnp.ones((hidden_dim, 3 * hidden_dim)) * 0.01
    
    # Create empty cache (won't be used)
    cache = initialize_cache(1, batch_size, num_heads, 128, hidden_dim // num_heads)
    
    # Run attention without cache
    output, _ = cached_attention(
        hidden_states=hidden_states,
        attn_weights=attn_weights,
        num_heads=num_heads,
        cache=cache,
        layer_idx=0,
        position=0,
        use_cache=False  # Disable cache
    )
    
    assert output.shape == (batch_size, 1, hidden_dim), \
        f"Output shape: {output.shape}"
    print(f"✓ Attention works without cache")
    print(f"  Output shape: {output.shape}")
    
    print("✓ Non-cached attention test passed!")

if __name__ == "__main__":
    try:
        test_split_merge_heads()
        test_causal_mask()
        test_compute_qkv()
        test_cached_attention()
        test_attention_without_cache()
        
        print("\n" + "=" * 70)
        print("✓ All Cached Attention Tests Passed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
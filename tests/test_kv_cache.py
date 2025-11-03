# This file is to validate all 3 kv_cache.py functions

import sys
sys.path.append('.')

import jax.numpy as jnp
from src.kv_cache import initialize_cache, update_cache, get_cached_kv

def test_initialize_cache():
    print("\n" + "=" * 70)
    print("Test 1: Cache Initialization")
    print("=" * 70)

    # GPT-2 small config
    num_layers = 12
    batch_size = 1
    num_heads = 12
    max_seq_len = 128
    head_dim = 64

    cache = initialize_cache(
        num_layers=num_layers,
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim
    )

    # Check 1: Correct number of layers
    assert len(cache) == num_layers, f"Expected {num_layers} layers, got {len(cache)}"
    print(f"✓ Cache has {num_layers} layers")

    # Check 2: Each layer has key and value
    for layer_idx in range(num_layers):
        assert layer_idx in cache, f"Layer {layer_idx} missing"
        assert 'key' in cache[layer_idx], f"Layer {layer_idx} missing 'key'"
        assert 'value' in cache[layer_idx], f"Layer {layer_idx} missing 'value'"
    print(f"✓ All layers have 'key' and 'value'")

    # Check 3: Correct shapes
    expected_shape = (batch_size, num_heads, max_seq_len, head_dim)
    layer_0_cache = cache[0]
    
    assert layer_0_cache['key'].shape == expected_shape, \
        f"Key shape mismatch: {layer_0_cache['key'].shape} vs {expected_shape}"
    assert layer_0_cache['value'].shape == expected_shape, \
        f"Value shape mismatch: {layer_0_cache['value'].shape} vs {expected_shape}"
    
    print(f"✓ Layer 0 cache shapes: {expected_shape}")

    # Check 4: Initialized to zeros
    assert jnp.allclose(layer_0_cache['key'], 0.0), "Keys not initialized to zero"
    assert jnp.allclose(layer_0_cache['value'], 0.0), "Values not initialized to zero"
    print(f"✓ Cache initialized to zeros")

    print(f"✓ Cache initialization test passed!")

def test_update_cache():
    print("\n" + "=" * 70)
    print("Test 2: Cache Update")
    print("=" * 70)

    # Small config for testing
    cache = initialize_cache(
        num_layers=2,
        batch_size=1,
        num_heads=4,
        max_seq_len=8,
        head_dim=16
    )

    # Test 1: Update position 0
    new_keys = jnp.ones((1, 4, 1, 16)) * 1.0  # Fill with 1.0
    new_values = jnp.ones((1, 4, 1, 16)) * 2.0  # Fill with 2.0

    cache = update_cache(
        cache=cache,
        layer_idx=0,
        new_keys=new_keys,
        new_values=new_values,
        cache_position=0
    )

    # Verify position 0 was updated
    assert jnp.allclose(cache[0]['key'][0, 0, 0, :], 1.0), "Keys at position 0 not updated"
    assert jnp.allclose(cache[0]['value'][0, 0, 0, :], 2.0), "Values at position 0 not updated"
    print(f"✓ Position 0 updated correctly")

    # Test 2: Update position 1 (should not corrupt position 0)
    new_keys = jnp.ones((1, 4, 1, 16)) * 3.0
    new_values = jnp.ones((1, 4, 1, 16)) * 4.0

    cache = update_cache(
        cache=cache,
        layer_idx=0,
        new_keys=new_keys,
        new_values=new_values,
        cache_position=1
    )

    # Verify position 0 unchanged, position 1 updated
    assert jnp.allclose(cache[0]['key'][0, 0, 0, :], 1.0), "Position 0 corrupted!"
    assert jnp.allclose(cache[0]['key'][0, 0, 1, :], 3.0), "Position 1 keys not updated"
    assert jnp.allclose(cache[0]['value'][0, 0, 1, :], 4.0), "Position 1 values not updated"
    print(f"✓ Position 1 updated correctly")

    # Test 3: Update multiple positions
    for pos in range(2, 5):
        new_keys = jnp.ones((1, 4, 1, 16)) * (pos + 1)
        new_values = jnp.ones((1, 4, 1, 16)) * (pos + 1) * 10
        cache = update_cache(cache, 0, new_keys, new_values, pos)

    # Verify all positions
    assert jnp.allclose(cache[0]['key'][0, 0, 2, :], 3.0), "Position 2 incorrect"
    assert jnp.allclose(cache[0]['key'][0, 0, 4, :], 5.0), "Position 4 incorrect"
    print(f"✓ Multiple positions updated correctly")

    print(f"✓ Cache update test passed!")

def test_get_cached_kv():
    print("\n" + "=" * 70)
    print("Test 3: Cache Retrieval")
    print("=" * 70)

    # Initialize and populate cache
    cache = initialize_cache(
        num_layers=1,
        batch_size=1,
        num_heads=2,
        max_seq_len=10,
        head_dim=8
    )

    # Add 5 tokens worth of cache
    for pos in range(5):
        new_keys = jnp.ones((1, 2, 1, 8)) * (pos + 1)  # 1.0, 2.0, 3.0, 4.0, 5.0
        new_values = jnp.ones((1, 2, 1, 8)) * (pos + 1) * 10  # 10, 20, 30, 40, 50
        cache = update_cache(cache, 0, new_keys, new_values, pos)

    # Test 1: Retrieve first 3 positions
    keys, values = get_cached_kv(cache, layer_idx=0, cache_length=3)

    # Verify shape
    expected_shape = (1, 2, 3, 8)
    assert keys.shape == expected_shape, f"Keys shape mismatch: {keys.shape} vs {expected_shape}"
    assert values.shape == expected_shape, f"Values shape mismatch: {values.shape} vs {expected_shape}"
    print(f"✓ Retrieved shapes for cache_length=3: {keys.shape}")

    # Verify values
    assert jnp.allclose(keys[0, 0, 0, :], 1.0), "Position 0 keys incorrect"
    assert jnp.allclose(keys[0, 0, 2, :], 3.0), "Position 2 keys incorrect"
    assert jnp.allclose(values[0, 0, 1, :], 20.0), "Position 1 values incorrect"
    print(f"✓ Retrieved values correct for cache_length=3")

    # Test 2: Retrieve all 5 positions
    keys, values = get_cached_kv(cache, layer_idx=0, cache_length=5)
    
    assert keys.shape == (1, 2, 5, 8), f"Shape mismatch for cache_length=5"
    assert jnp.allclose(keys[0, 0, 4, :], 5.0), "Position 4 keys incorrect"
    assert jnp.allclose(values[0, 0, 4, :], 50.0), "Position 4 values incorrect"
    print(f"✓ Retrieved shapes for cache_length=5: {keys.shape}")

    print(f"✓ Cache retrieval test passed!")

if __name__ == "__main__":
    try:
        test_initialize_cache()
        test_update_cache()
        test_get_cached_kv()

        print("\n" + "=" * 70)
        print("✓ All KV-Cache tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
# Comprehensive comparison of optimization levels
#
# This test demonstrates the performance impact of KV-cache + JIT optimization
# on GPT-2 text generation using JAX.
#
# Tests two configurations:
# 1) Non cached (Baseline) - Full recomputation for each token
# 2) Cached + JIT (Full Optimization) - KV-cache + JIT compilation
#
# Expected Results:
# - Non-cached: ~1.5 tok/s
# - Cached + JIT: ~24 tok/s
# - Speedup: ~16x
#
# Note: Cannot test cache vs JIT separately with decorator approach
# (decorators are applied at function definition time)
#
# IMPORTANT: JAX JIT Compilation Behavior
# ----------------------------------------
# JAX compiles separately for each input shape. During autoregressive generation,
# each token produces a different sequence length. The warmup MUST generate at least
# as many tokens as the actual test to ensure all shapes are pre-compiled.
#
# See BENCHMARKING.md for detailed best practices.
#

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from src.model_conversion import load_pytorch_model, convert_pytorch_to_jax, build_flax_pytree
from src.cached_generation import generate_text_with_cache

def test_optimization_comparison():
    """
    Compare non-cached vs cached+JIT performance.

    IMPORTANT: JAX JIT compiles separately for each input shape.
    During autoregressive generation, each token produces a different sequence length.
    The warmup must generate at least as many tokens as the actual test to ensure
    all shapes are pre-compiled for accurate benchmarking.
    """

    print("=" * 80)
    print("OPTIMIZATION COMPARISON TEST")
    print("Testing: Non-Cached vs Cached+JIT")
    print("=" * 80)

    # Load model once
    print("\nLoading GPT-2...")
    pytorch_state_dict, tokenizer = load_pytorch_model(use_small_model=True)
    print("Converting to JAX...")
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    params_tree = build_flax_pytree(jax_state_dict)
    params = {'params': params_tree}
    print("[OK] Model loaded\n")

    prompt = "The quick brown fox"
    max_new_tokens = 15

    # ========================================================================
    # Test 1: NON-CACHED (baseline)
    # ========================================================================
    print("=" * 80)
    print("TEST 1: NON-CACHED (Baseline)")
    print("=" * 80)

    start = time.time()
    text_noncached, stats_noncached = generate_text_with_cache(
        params=params,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_cache=False,  # No cache
        model_type="gpt2"
    )
    time_noncached = time.time() - start

    print(f"\nNon-cached result:")
    print(f"  Speed: {stats_noncached['tokens_per_sec']:.2f} tok/s")
    print(f"  Time:  {time_noncached:.3f}s")

    # ========================================================================
    # Test 2: CACHED + JIT (full optimization)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: CACHED + JIT (Full Optimization)")
    print("=" * 80)

    # Warm-up run for JIT
    # IMPORTANT: Must generate at least as many tokens as the test
    # because JAX JIT compiles separately for each sequence length
    print("Warm-up run (JIT compilation)...")
    _, _ = generate_text_with_cache(
        params=params,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,  # Same as test to compile all shapes
        temperature=0.0,
        use_cache=True,
        model_type="gpt2"
    )
    print("JIT compilation complete\n")

    # Actual test run
    start = time.time()
    text_cached, stats_cached = generate_text_with_cache(
        params=params,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_cache=True,   # With cache + JIT (decorators)
        model_type="gpt2"
    )
    time_cached = time.time() - start

    print(f"\nCached + JIT result:")
    print(f"  Speed: {stats_cached['tokens_per_sec']:.2f} tok/s")
    print(f"  Time:  {time_cached:.3f}s")

    # ========================================================================
    # COMPARISON & ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    speedup_total = stats_cached['tokens_per_sec'] / stats_noncached['tokens_per_sec']

    print(f"\n{'Configuration':<30} {'Speed (tok/s)':<15} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 80)
    print(f"{'1. Non-Cached (Baseline)':<30} {stats_noncached['tokens_per_sec']:<15.2f} {time_noncached:<12.3f} {'1.00x':<10}")
    print(f"{'2. Cached + JIT':<30} {stats_cached['tokens_per_sec']:<15.2f} {time_cached:<12.3f} {f'{speedup_total:.2f}x':<10}")

    print("\n" + "=" * 80)
    print("OPTIMIZATION BREAKDOWN")
    print("=" * 80)

    print(f"\nTotal improvement: {speedup_total:.2f}x speedup")
    print(f"\nNote: JIT decorators are applied throughout, achieving maximum performance.")
    print(f"Cannot measure KV-cache vs JIT separately with decorator approach.")

    # Verify outputs match
    print("\n" + "=" * 80)
    print("OUTPUT VERIFICATION")
    print("=" * 80)
    print(f"Non-cached output: {text_noncached}")
    print(f"Cached + JIT output: {text_cached}")

    if text_noncached == text_cached:
        print("\n[OK] Both methods produce identical output")
    else:
        print("\n[WARNING] Outputs differ!")

    print("\n" + "=" * 80)
    print("[OK] OPTIMIZATION COMPARISON COMPLETE")
    print("=" * 80)

    return {
        'noncached_speed': stats_noncached['tokens_per_sec'],
        'cached_speed': stats_cached['tokens_per_sec'],
        'total_speedup': speedup_total
    }


if __name__ == "__main__":
    try:
        results = test_optimization_comparison()
        print(f"\n[OK] Test completed successfully!")
        print(f"Total speedup: {results['total_speedup']:.2f}x")
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
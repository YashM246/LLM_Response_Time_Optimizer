# JAX Optimization Guide for Transformer Models

This guide documents the optimization techniques used in this project to achieve **16.32x speedup** on GPT-2 text generation using JAX.

## Table of Contents
1. [Overview](#overview)
2. [KV-Cache Optimization](#kv-cache-optimization)
3. [JIT Compilation](#jit-compilation)
4. [Performance Measurement Best Practices](#performance-measurement-best-practices)
5. [Common Pitfalls](#common-pitfalls)
6. [Results](#results)

---

## Overview

### Baseline vs Optimized Performance

| Metric | Non-Cached | Cached + JIT | Improvement |
|--------|------------|--------------|-------------|
| Speed | 1.50 tok/s | 24.45 tok/s | **16.32x** |
| Time (15 tokens) | 10.03s | 0.63s | **16x faster** |

### Optimization Stack

```
┌─────────────────────────────────────┐
│  Application Layer                   │
│  (Text Generation)                   │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  KV-Cache Layer                      │  ← Avoids redundant computation
│  (Caches attention K,V)              │     ~10-15x speedup
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  JIT Compilation Layer               │  ← XLA optimization
│  (@jax.jit decorators)               │     Additional optimization
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Hardware (CPU/GPU/TPU)              │
└─────────────────────────────────────┘
```

---

## KV-Cache Optimization

### Problem: Quadratic Computation

Without caching, each new token requires recomputing attention for **all previous tokens**:

```
Token 1: Compute attention for positions [0]          → 1 computation
Token 2: Compute attention for positions [0, 1]       → 2 computations
Token 3: Compute attention for positions [0, 1, 2]    → 3 computations
...
Token N: Compute attention for positions [0..N-1]     → N computations

Total: 1 + 2 + 3 + ... + N = O(N²)
```

For 15 tokens: **1 + 2 + 3 + ... + 15 = 120 attention computations**

### Solution: Cache Keys and Values

With KV-cache, we store previously computed keys and values:

```
Token 1: Compute K,V for position 0, cache it         → 1 computation
Token 2: Compute K,V for position 1, append to cache  → 1 computation
Token 3: Compute K,V for position 2, append to cache  → 1 computation
...
Token N: Compute K,V for position N-1, append         → 1 computation

Total: N = O(N)
```

For 15 tokens: **15 attention computations** (vs 120 without cache)

### Implementation

```python
from src.kv_cache import initialize_cache, update_cache, get_cached_kv

# 1. Initialize cache (pre-allocate memory)
cache = initialize_cache(
    num_layers=12,
    batch_size=1,
    num_heads=12,
    max_seq_len=1024,
    head_dim=64
)

# 2. During generation loop
for position in range(num_tokens):
    # Compute new K, V only for current token
    new_k, new_v = compute_qkv(hidden_states, ...)

    # Update cache with new K, V
    cache = update_cache(cache, layer_idx, new_k, new_v, position)

    # Retrieve all cached K, V for attention computation
    cached_k, cached_v = get_cached_kv(cache, layer_idx, position + 1)

    # Compute attention using cached values
    attention_output = compute_attention(query, cached_k, cached_v)
```

### Memory Trade-off

- **Memory increase**: Cache stores `num_layers * batch * heads * max_seq * head_dim` floats
  - GPT-2: 12 layers × 1 batch × 12 heads × 1024 max_seq × 64 head_dim = ~9.4M floats (~38MB)
- **Speedup**: ~10-15x faster generation
- **Worth it?** YES - Small memory cost for massive speed gain

---

## JIT Compilation

### What is JIT?

JIT (Just-In-Time) compilation uses XLA (Accelerated Linear Algebra) to:
1. Fuse operations together
2. Optimize memory access patterns
3. Generate hardware-specific machine code

### How to Use @jax.jit

JAX provides the `@jax.jit` decorator to mark functions for compilation:

```python
import jax
import jax.numpy as jnp
from functools import partial

# Simple JIT (no static arguments)
@jax.jit
def simple_function(x, y):
    return x + y

# JIT with static arguments (compile-time constants)
@partial(jax.jit, static_argnums=(1,))
def function_with_static(x, num_heads):
    # num_heads is known at compile time
    # Allows for shape specialization
    head_dim = x.shape[-1] // num_heads
    return x.reshape(..., num_heads, head_dim)
```

### Static Arguments (`static_argnums`)

Some arguments must be **compile-time constants**:
- Shape-determining values (num_heads, seq_len)
- Control flow conditions (model_type)
- Loop bounds

**Why?** XLA needs to know shapes and control flow at compile time to optimize.

```python
# CORRECT: num_heads is static (index 1)
@partial(jax.jit, static_argnums=(1,))
def split_heads(x: jnp.ndarray, num_heads: int):
    batch, seq, hidden = x.shape
    head_dim = hidden // num_heads  # Uses static num_heads
    return x.reshape(batch, seq, num_heads, head_dim)

# WRONG: num_heads is traced (dynamic)
@jax.jit
def split_heads_wrong(x: jnp.ndarray, num_heads: int):
    head_dim = x.shape[-1] // num_heads  # Error: traced value in shape!
    return x.reshape(..., num_heads, head_dim)
```

### JIT-Compiled Functions in This Project

| Function | Static Arguments | Why Static? |
|----------|------------------|-------------|
| `split_heads` | `num_heads` (index 1) | Determines reshape dimensions |
| `compute_qkv` | `num_heads` (index 3) | Used in split_heads call |
| `batch_attention` | `num_heads` (index 3) | Shape-dependent operations |
| `causal_mask` | `seq_len` (index 0) | Determines mask size |
| `mlp` | `model_type` (index 2) | Controls activation function |
| `lm_head` | `model_type` (index 2) | Controls layernorm location |

---

## Performance Measurement Best Practices

### Critical Issue: Shape-Dependent Compilation

**JAX JIT compiles separately for each unique input shape!**

During autoregressive generation:
```
Token 1: Shape [1, 4]  → Compile version A
Token 2: Shape [1, 5]  → Compile version B
Token 3: Shape [1, 6]  → Compile version C
...
Token 15: Shape [1, 18] → Compile version O
```

Each new shape triggers a **new compilation** (~0.5-2 seconds overhead).

### Warmup Requirements

**RULE: Warmup must generate ≥ test token count**

```python
# WRONG: Insufficient warmup
warmup_tokens = 5
test_tokens = 15
# Shapes [1,4] through [1,8] compiled during warmup
# Shapes [1,9] through [1,18] compiled during TEST → slow!

# CORRECT: Adequate warmup
warmup_tokens = 15
test_tokens = 15
# All shapes [1,4] through [1,18] compiled during warmup
# Test runs with all shapes pre-compiled → fast!
```

### Benchmark Template

```python
def benchmark_with_warmup(params, tokenizer, prompt, max_tokens):
    """
    Proper benchmarking with adequate warmup.
    """
    # STEP 1: Warmup (compile all shapes)
    print("Warming up JIT compilation...")
    _, _ = generate_text_with_cache(
        params=params,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,  # ← SAME as test!
        temperature=0.0,
        use_cache=True,
        model_type="gpt2"
    )

    # STEP 2: Actual benchmark (all shapes compiled)
    start = time.time()
    text, stats = generate_text_with_cache(
        params=params,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.0,
        use_cache=True,
        model_type="gpt2"
    )
    elapsed = time.time() - start

    print(f"Speed: {stats['tokens_per_sec']:.2f} tok/s")
    print(f"Time: {elapsed:.3f}s")

    return stats
```

### What We Measured Wrong Initially

**Incorrect Result (2.44x speedup):**
- Warmup: 5 tokens → compiled shapes [1,4] to [1,8]
- Test: 15 tokens → shapes [1,9] to [1,18] compiled during measurement
- Result: 4.23 tok/s (includes compilation overhead)

**Correct Result (16.32x speedup):**
- Warmup: 15 tokens → compiled shapes [1,4] to [1,18]
- Test: 15 tokens → all shapes pre-compiled
- Result: 24.45 tok/s (no compilation overhead)

---

## Common Pitfalls

### 1. ❌ Insufficient Warmup

```python
# BAD: Only warms up 5 tokens
warmup(max_tokens=5)
test(max_tokens=20)  # Compiles during test!
```

**Fix:** Match warmup to test length

```python
# GOOD: Warms up all needed shapes
warmup(max_tokens=20)
test(max_tokens=20)  # All shapes pre-compiled
```

### 2. ❌ Using Traced Values in Control Flow

```python
@jax.jit
def bad_function(x, use_cache):
    if use_cache:  # ← ERROR! use_cache is traced
        return cached_path(x)
    else:
        return non_cached_path(x)
```

**Fix:** Make boolean static or use functional approach

```python
@partial(jax.jit, static_argnums=(1,))
def good_function(x, use_cache: bool):
    if use_cache:  # ← OK! use_cache is static
        return cached_path(x)
    else:
        return non_cached_path(x)
```

### 3. ❌ Dynamic Shapes in JIT Functions

```python
@jax.jit
def bad_reshape(x, new_shape):
    return x.reshape(new_shape)  # ← ERROR! Dynamic shape
```

**Fix:** Use static arguments or fixed shapes

```python
@partial(jax.jit, static_argnums=(1,))
def good_reshape(x, new_shape: tuple):
    return x.reshape(new_shape)  # ← OK! new_shape is static
```

### 4. ❌ In-Place Modifications

```python
@jax.jit
def bad_update(cache, new_value):
    cache[0] = new_value  # ← ERROR! JAX arrays are immutable
    return cache
```

**Fix:** Use functional updates

```python
@jax.jit
def good_update(cache, new_value):
    return cache.at[0].set(new_value)  # ← OK! Functional update
```

---

## Results

### Final Performance (GPT-2)

| Configuration | Speed (tok/s) | Time (15 tokens) | Speedup |
|---------------|---------------|------------------|---------|
| Non-Cached (Baseline) | 1.50 | 10.03s | 1.00x |
| **Cached + JIT** | **24.45** | **0.63s** | **16.32x** |

### Breakdown of Optimizations

- **KV-Cache**: Reduces computation from O(N²) to O(N)
  - Estimated contribution: ~10-13x
- **JIT Compilation**: XLA optimization and operation fusion
  - Additional optimization on top of cache
- **Combined**: 16.32x measured speedup

### Memory Overhead

| Component | Memory |
|-----------|--------|
| Model (INT8) | 163 MB |
| KV-Cache (1024 max_seq) | ~38 MB |
| **Total** | **~201 MB** |

---

## Next Steps

To apply these optimizations to other models:

1. **Implement KV-Cache**
   - Use `src/kv_cache.py` utilities
   - Modify attention to use cached K,V

2. **Add JIT Decorators**
   - Mark compute-heavy functions with `@jax.jit`
   - Identify static arguments (shapes, model config)
   - Use `static_argnums` for compile-time constants

3. **Benchmark Correctly**
   - Warmup ≥ test token count
   - Measure after JIT compilation complete
   - Run multiple trials for variance

4. **Profile and Iterate**
   - Use `jax.profiler` to find bottlenecks
   - Add JIT to hot functions
   - Consider additional optimizations (mixed precision, etc.)

---

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX JIT Compilation Guide](https://jax.readthedocs.io/en/latest/jit-compilation.html)
- [KV-Cache in Transformers](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [This Project's README](README.md)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-13
**Status**: Complete - Phase 4 GPT-2 Optimization

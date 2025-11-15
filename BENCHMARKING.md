# Benchmarking Best Practices for JAX Models

This document outlines best practices for benchmarking JAX models, particularly for autoregressive text generation with JIT compilation.

## Quick Start

```python
from src.model_conversion import load_pytorch_model, convert_pytorch_to_jax, build_flax_pytree
from src.cached_generation import generate_text_with_cache
import time

# Load model
pytorch_state_dict, tokenizer = load_pytorch_model(use_small_model=True)
jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
params_tree = build_flax_pytree(jax_state_dict)
params = {'params': params_tree}

prompt = "The quick brown fox"
max_new_tokens = 15

# STEP 1: Warmup (CRITICAL!)
print("Warming up JIT compilation...")
_, _ = generate_text_with_cache(
    params=params,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=max_new_tokens,  # Must match test!
    temperature=0.0,
    use_cache=True,
    model_type="gpt2"
)

# STEP 2: Measure
start = time.time()
text, stats = generate_text_with_cache(
    params=params,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=max_new_tokens,
    temperature=0.0,
    use_cache=True,
    model_type="gpt2"
)
elapsed = time.time() - start

print(f"Speed: {stats['tokens_per_sec']:.2f} tok/s")
print(f"Time: {elapsed:.3f}s")
```

## Why Warmup is Critical

### JAX JIT Behavior

JAX compiles functions **separately for each unique input shape**:

```
Generation of 15 tokens with prompt length 4:

Token 1 (pos 4):  Input shape [1, 4]  → Compile & execute
Token 2 (pos 5):  Input shape [1, 5]  → Compile & execute
Token 3 (pos 6):  Input shape [1, 6]  → Compile & execute
...
Token 15 (pos 18): Input shape [1, 18] → Compile & execute
```

**Each shape → New compilation (~0.5-2 seconds overhead)**

### Case Study: Our Investigation

#### ❌ Incorrect Benchmark (Insufficient Warmup)

```python
# Warmup: 5 tokens
generate_text_with_cache(..., max_new_tokens=5)
# Compiles shapes [1,4], [1,5], [1,6], [1,7], [1,8]

# Test: 15 tokens (MEASURED TIME)
generate_text_with_cache(..., max_new_tokens=15)
# Shapes [1,4] to [1,8]: Already compiled ✓
# Shapes [1,9] to [1,18]: Compiling during measurement! ✗

# Result: 4.23 tok/s (includes compilation overhead)
# Speedup: 2.44x (WRONG!)
```

#### ✅ Correct Benchmark (Proper Warmup)

```python
# Warmup: 15 tokens (matches test)
generate_text_with_cache(..., max_new_tokens=15)
# Compiles shapes [1,4] to [1,18]

# Test: 15 tokens (MEASURED TIME)
generate_text_with_cache(..., max_new_tokens=15)
# Shapes [1,4] to [1,18]: Already compiled ✓

# Result: 24.45 tok/s (pure inference time)
# Speedup: 16.32x (CORRECT!)
```

**6.8x difference** in measured performance due to compilation overhead!

## Benchmarking Checklist

### ✅ Before Running Benchmarks

- [ ] Load model outside timing loop
- [ ] Prepare all test prompts beforehand
- [ ] Determine maximum token length for tests
- [ ] Run warmup with **same or longer** token count as test

### ✅ During Warmup

- [ ] Use same `use_cache` setting as test
- [ ] Use same `model_type` as test
- [ ] Generate ≥ test token count
- [ ] Wait for completion (don't time this!)

### ✅ During Measurement

- [ ] Start timer **after** warmup complete
- [ ] Run test with pre-compiled shapes
- [ ] Record: time, tokens/sec, memory usage
- [ ] Verify outputs are correct (quality check)

### ✅ After Measurement

- [ ] Run multiple trials (3-5x) for variance
- [ ] Report mean and std deviation
- [ ] Check for outliers (first run may be slower)
- [ ] Document: hardware, JAX version, model size

## Common Mistakes

### 1. Measuring Cold Start Performance

```python
# ❌ WRONG: First run includes compilation
start = time.time()
text, stats = generate_text_with_cache(...)  # Compiling!
elapsed = time.time() - start
```

**Fix:** Always warmup first

```python
# ✓ CORRECT: Warmup before measurement
generate_text_with_cache(...)  # Warmup
start = time.time()
text, stats = generate_text_with_cache(...)  # Measure
elapsed = time.time() - start
```

### 2. Insufficient Warmup Length

```python
# ❌ WRONG: Warmup shorter than test
warmup(max_tokens=5)
test(max_tokens=20)  # Compiles shapes [1,9] to [1,24] during test!
```

**Fix:** Match or exceed test length

```python
# ✓ CORRECT: Warmup matches test length
warmup(max_tokens=20)
test(max_tokens=20)  # All shapes pre-compiled
```

### 3. Changing Configuration Between Warmup and Test

```python
# ❌ WRONG: Different settings
warmup(use_cache=False)  # Compiles non-cached path
test(use_cache=True)     # Compiles cached path during test!
```

**Fix:** Match all configuration

```python
# ✓ CORRECT: Identical settings
warmup(use_cache=True, temperature=0.0, model_type="gpt2")
test(use_cache=True, temperature=0.0, model_type="gpt2")
```

### 4. Including Model Loading in Timing

```python
# ❌ WRONG: Timing includes model loading
start = time.time()
params = load_and_convert_model()  # Slow!
text, stats = generate_text_with_cache(...)
elapsed = time.time() - start
```

**Fix:** Load once, time only inference

```python
# ✓ CORRECT: Load before timing
params = load_and_convert_model()  # Outside timing
start = time.time()
text, stats = generate_text_with_cache(...)
elapsed = time.time() - start
```

## Statistical Best Practices

### Running Multiple Trials

```python
import numpy as np

def benchmark_with_stats(params, tokenizer, prompt, max_tokens, num_trials=5):
    """Run benchmark multiple times and report statistics."""

    # Warmup
    generate_text_with_cache(
        params=params, tokenizer=tokenizer, prompt=prompt,
        max_new_tokens=max_tokens, temperature=0.0,
        use_cache=True, model_type="gpt2"
    )

    # Run trials
    speeds = []
    for trial in range(num_trials):
        start = time.time()
        text, stats = generate_text_with_cache(
            params=params, tokenizer=tokenizer, prompt=prompt,
            max_new_tokens=max_tokens, temperature=0.0,
            use_cache=True, model_type="gpt2"
        )
        elapsed = time.time() - start
        speeds.append(stats['tokens_per_sec'])

    # Report statistics
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    min_speed = np.min(speeds)
    max_speed = np.max(speeds)

    print(f"Speed: {mean_speed:.2f} ± {std_speed:.2f} tok/s")
    print(f"Range: [{min_speed:.2f}, {max_speed:.2f}]")

    return {
        'mean': mean_speed,
        'std': std_speed,
        'min': min_speed,
        'max': max_speed,
        'trials': speeds
    }
```

### Handling Outliers

- **First run slower?** Normal - may include some lazy initialization
- **Last run faster?** Could indicate CPU thermal throttling or caching
- **High variance?** Check for background processes, use more trials

### Reporting Results

**Minimum reporting:**
- Mean speed ± standard deviation
- Hardware used (CPU/GPU model)
- JAX version
- Model size and configuration

**Full reporting:**
```
Configuration: GPT-2 (124M params), INT8 quantized
Hardware: NVIDIA RTX 4070, 8GB VRAM
JAX version: 0.4.23
CUDA version: 12.2

Results (5 trials):
- Mean: 24.45 ± 0.83 tok/s
- Range: [23.12, 25.67] tok/s
- Speedup vs non-cached: 16.32x
```

## Memory Profiling

### Measuring Memory Usage

```python
import jax

# Before generation
jax.clear_backends()  # Clear any cached compilations
initial_memory = jax.devices()[0].memory_stats()['bytes_in_use']

# Run generation
text, stats = generate_text_with_cache(...)

# After generation
final_memory = jax.devices()[0].memory_stats()['bytes_in_use']
memory_used = (final_memory - initial_memory) / 1024**2  # Convert to MB

print(f"Memory used: {memory_used:.2f} MB")
```

### Tracking Peak Memory

```python
# Track peak memory during generation
memory_before = jax.devices()[0].memory_stats()['peak_bytes_in_use']
text, stats = generate_text_with_cache(...)
memory_after = jax.devices()[0].memory_stats()['peak_bytes_in_use']
peak_memory_mb = (memory_after - memory_before) / 1024**2

print(f"Peak memory: {peak_memory_mb:.2f} MB")
```

## Comparing Configurations

### Template for A/B Testing

```python
def compare_configurations(params, tokenizer, prompt, max_tokens):
    """Compare non-cached vs cached performance."""

    configs = [
        ("Non-Cached", False),
        ("Cached + JIT", True)
    ]

    results = {}

    for name, use_cache in configs:
        print(f"\nTesting {name}...")

        # Warmup
        generate_text_with_cache(
            params=params, tokenizer=tokenizer, prompt=prompt,
            max_new_tokens=max_tokens, temperature=0.0,
            use_cache=use_cache, model_type="gpt2"
        )

        # Measure
        start = time.time()
        text, stats = generate_text_with_cache(
            params=params, tokenizer=tokenizer, prompt=prompt,
            max_new_tokens=max_tokens, temperature=0.0,
            use_cache=use_cache, model_type="gpt2"
        )
        elapsed = time.time() - start

        results[name] = {
            'speed': stats['tokens_per_sec'],
            'time': elapsed,
            'text': text
        }

        print(f"  Speed: {stats['tokens_per_sec']:.2f} tok/s")
        print(f"  Time: {elapsed:.3f}s")

    # Compute speedup
    baseline_speed = results["Non-Cached"]['speed']
    optimized_speed = results["Cached + JIT"]['speed']
    speedup = optimized_speed / baseline_speed

    print(f"\nSpeedup: {speedup:.2f}x")

    return results, speedup
```

## Environment Setup

### Reproducible Benchmarks

```python
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
os.environ['JAX_ENABLE_X64'] = 'False'  # Use FP32 for consistency

import jax
jax.config.update('jax_platform_name', 'gpu')  # Force GPU
```

### Disabling JIT (for debugging)

```python
# Disable JIT to measure non-JIT performance
with jax.disable_jit():
    text, stats = generate_text_with_cache(...)
```

## Troubleshooting

### Slow First Run

**Symptom:** First measurement much slower than subsequent runs

**Cause:** JAX lazy initialization or XLA compilation

**Fix:** Add an extra warmup run and discard first measurement

### Inconsistent Results

**Symptom:** High variance across trials (>10% std deviation)

**Possible causes:**
- Background processes consuming resources
- GPU thermal throttling
- OS scheduling interference

**Fixes:**
- Close unnecessary applications
- Monitor GPU temperature
- Increase number of trials
- Use `nice` (Linux) or process priority (Windows)

### Out of Memory

**Symptom:** JAX OOM error during benchmarking

**Fixes:**
- Reduce `max_seq_len` in cache initialization
- Use smaller batch size
- Enable gradient checkpointing (if applicable)
- Use INT8 quantization

## Summary

**Golden Rules for JAX Benchmarking:**

1. ✅ **Warmup ≥ Test Length**: Compile all shapes before measurement
2. ✅ **Match Configuration**: Warmup and test must use same settings
3. ✅ **Load Once**: Model loading outside timing loop
4. ✅ **Multiple Trials**: Run 3-5 times, report mean ± std
5. ✅ **Document Everything**: Hardware, JAX version, configuration

Following these practices ensures **accurate, reproducible benchmarks** that truly measure inference performance rather than compilation overhead.

---

**Related Documents:**
- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - Optimization techniques
- [README.md](README.md) - Project overview and results
- [tests/test_optimization_comparison.py](tests/test_optimization_comparison.py) - Reference benchmark implementation

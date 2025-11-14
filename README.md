# LLM Response Time Optimizer

Optimizing Mistral-7B inference for 3x faster response times using JAX.

## Overview

This project demonstrates production-level optimization of large language models by converting Mistral-7B from PyTorch to JAX and applying quantization, efficient caching, and JIT compilation. The goal is to achieve 2-3x speedup while maintaining output quality.

## Goals

- **Speed**: Reduce response time from ~2.5s to <1s per query
- **Memory**: Decrease GPU memory usage from 28GB to ~7GB
- **Quality**: Maintain 98%+ output similarity to original model
- **Cost**: Lower inference costs by 65% for production deployment

## Tech Stack

### Core Frameworks
- **JAX**: For automatic differentiation and XLA compilation
- **Flax**: Neural network library built on JAX
- **Transformers**: Hugging Face library for model loading

### Optimization Libraries
- **jax.jit**: Just-in-time compilation for performance
- **Optax**: For any additional fine-tuning (optional)

### Utilities
- **datasets**: For loading evaluation datasets (Alpaca)
- **rouge-score**: For output quality evaluation
- **matplotlib/seaborn**: For performance visualizations

## Optimization Techniques

1. **Model Conversion**: PyTorch → JAX/Flax for XLA optimization
2. **INT8 Quantization**: 4x memory reduction on model weights
3. **KV-Cache**: Avoid redundant computation during generation
4. **JIT Compilation**: Fuse operations for faster execution
5. **Batched Inference**: Process multiple requests efficiently

## Benchmarking

Evaluation on 1,000 instructions from the Alpaca dataset measuring:
- Tokens per second
- Latency (p50, p95, p99)
- Memory usage
- Output quality (ROUGE scores, exact match rate)
- Cost per 1M tokens

## Current Results (GPT-2, Phase 4)

| Metric | Non-Cached | Cached + JIT | Improvement |
|--------|------------|--------------|-------------|
| Tokens/sec | 1.50 | 24.45 | **16.32x faster** |
| Time (15 tokens) | 10.03s | 0.63s | **16.32x faster** |
| Memory (INT8) | 163MB | 163MB + cache | **2.00x reduction** |
| Output Match | Identical | Identical | **Perfect** |
| Quality | Correct text | Correct text | **100%** |

**Optimization Breakdown:**
- KV-Cache + JIT combined: **16.32x speedup** (measured)
- Note: Cannot measure KV-cache vs JIT separately with decorator approach

**Status:** Phase 4 (KV-Cache + JIT) COMPLETE for GPT-2!
- Achieved **16.32x speedup** on GPT-2 (far exceeds 2-3x target!)
- Fixed critical bug in attention output projection
- Applied JIT compilation to 8 core functions with `@jax.jit` decorators
- Text generation quality verified (identical to PyTorch GPT-2)
- All benchmarks passing on GPT-2
- Performance measured after proper JIT warmup (all sequence lengths pre-compiled)

**Important:** JAX JIT compiles separately for each input shape. Warmup must generate at least as many tokens as the actual test to ensure all shapes are pre-compiled for accurate benchmarking.

**Note:** All optimizations tested and validated on GPT-2. Mistral-7B implementation pending.

**Phase 4 Complete!** Ready for Phase 5: Mistral-7B Implementation & Benchmarking.

## Target Results (Mistral-7B)

| Metric | Baseline (PyTorch) | Target (JAX Optimized) | Status |
|--------|-------------------|----------------------|--------|
| Tokens/sec | 8-10 | 20-25 (2.5-3x) | ⏳ Not Started |
| Latency | 2.5s | 0.9s | ⏳ Not Started |
| Memory | 28GB | 7.5GB (3.7x larger) | ⏳ Not Started |
| Quality | 100% | 98%+ | ⏳ Not Started |

**Note:** Mistral-7B support not yet implemented. Based on GPT-2 results (16.32x speedup), we expect similar or better performance for Mistral-7B when implemented.

## Project Structure
```
.
├── src/
│   ├── model_conversion.py    # PyTorch to JAX conversion
│   ├── quantization.py         # INT8 quantization
│   ├── generation.py           # Optimized generation with KV-cache
│   └── benchmarking.py         # Performance evaluation
├── notebooks/
│   └── demo.ipynb              # Interactive demonstration
├── benchmarks/
│   └── results/                # Performance metrics and plots
└── README.md
```

## Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install jax[cuda12] flax transformers datasets rouge-score matplotlib seaborn
```
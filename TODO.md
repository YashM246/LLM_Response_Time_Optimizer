# Project TODO List

## Current Status
- **Phase 1**: ✅ Complete (Environment setup, PyTorch baseline)
- **Phase 2**: ✅ Complete (PyTorch → JAX conversion)
- **Phase 3**: ✅ Complete (INT8 quantization - 2.00x memory reduction)
- **Phase 4**: ⏳ Next (KV-Cache + JIT optimization)
- **Phase 5**: ⏳ Pending (Benchmarking)

---

## Future Work (Post-Project)

### 1. Checkpoint Saving/Loading for Quantized Models
**Priority**: Medium
**Estimated Time**: 2-3 hours
**Why Deferred**: Current workflow (re-convert each session) is acceptable for 2-3 week timeline

**Implementation Requirements**:
- [ ] Create `save_quantized_model(quantized_params, scales, save_dir)` function
  - Save quantized PyTree using Orbax or pickle
  - Save scales dictionary as JSON
  - Include metadata (model name, quantization config)

- [ ] Create `load_quantized_model(save_dir)` function
  - Load quantized PyTree
  - Load scales dictionary
  - Validate shapes and dtypes

- [ ] Add validation tests
  - Round-trip test: save → load → verify identical
  - Test with both GPT-2 and Mistral
  - Verify dequantization produces correct values

- [ ] Handle edge cases
  - Missing scale factors
  - Corrupted checkpoints
  - Version compatibility (Orbax updates)

**Files to modify**:
- `src/quantization.py` - Add save/load functions
- `tests/test_quantization.py` - Add save/load tests
- `notebooks/02_mistral_jax_conversion_colab.ipynb` - Add optional save cell

**Decision points**:
- **Pickle vs Orbax**: Pickle is simpler but Orbax handles large models better
- **Storage format**: Single file vs directory structure
- **Compression**: Whether to compress INT8 arrays (minimal gains)

---

### 2. Quantization Analysis Notebook (Local)
**Priority**: Low
**Estimated Time**: 30 minutes
**Why Deferred**: Results already validated, visualizations nice-to-have

**Implementation**:
- [ ] Create `notebooks/03_quantization_analysis.ipynb`
- [ ] Visualize memory reduction (bar chart)
- [ ] Analyze quantization error distribution (histograms)
- [ ] Show parameter type breakdown (pie chart)
- [ ] Document findings in notebook

---

### 3. Advanced Quantization Techniques
**Priority**: Low (research/optimization)
**Estimated Time**: 1-2 weeks
**Why Deferred**: Current 2x reduction meets project goals

**Potential improvements**:
- [ ] Per-channel quantization (vs per-tensor)
- [ ] Mixed precision (INT4 for some layers)
- [ ] Activation quantization (beyond weight-only)
- [ ] Quantization-aware training (QAT)
- [ ] GPTQ or AWQ algorithms

---

### 4. Model Serving Infrastructure
**Priority**: Low (deployment focus)
**If building production API**:
- [ ] FastAPI/Flask server
- [ ] Model checkpoint management
- [ ] Request batching
- [ ] Caching layer
- [ ] Monitoring/logging

---

## Notes

**Current Workflow** (acceptable for project timeline):
```
Each Colab session:
1. Convert PyTorch → JAX (2-3 min)
2. Quantize to INT8 (1 min)
3. Run benchmarks (rest of session)

Total overhead: 3-4 minutes per session
Colab sessions: 4+ hours each
Overhead ratio: <2% - negligible
```

**When to implement checkpoint saving**:
- Publishing final trained/optimized model
- If conversion time becomes bottleneck (>10 min)
- If need to distribute model to others
- Post-project for production deployment

---

## Tracking

**Last Updated**: Phase 3 completion
**Next Milestone**: Phase 4 KV-Cache implementation
**Target Completion**: 2-3 weeks from project start

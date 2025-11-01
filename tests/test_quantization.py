import sys
import jax
import jax.numpy as jnp

sys.path.append('.')

from src.quantization import quantize_weights, dequantize_weights, quantize_model_params

def test_basic_quantization():

    weights = jnp.array([0.52, -0.31, -0.67, 0.23])

    print(f"Original weights: {weights}")

    # Quantize
    quantized, scale = quantize_weights(weights)
    print(f"Quantized Weights: {quantized}")
    print(f"Scale: {scale}")
    print(f"DType: {quantized.dtype}")

    # Dequantize
    reconstructed = dequantize_weights(quantized, scale)
    print(f"Reconstructed: {reconstructed}")

    # Measure Error:
    error = jnp.abs(weights - reconstructed).mean()
    max_error = jnp.abs(weights - reconstructed).max()

    print(f"\nMean error: {error}")
    print(f"Max error: {max_error}")
    
    # Check if error is acceptable
    assert error < 0.01, f"Error too high: {error}"
    assert quantized.dtype == jnp.int8, f"Wrong dtype: {quantized.dtype}"
    
    print("\n✓ Basic quantization test passed!")

def test_model_quantization():
    """Test quantizing a full GPT-2 model"""
    
    print("\n" + "=" * 70)
    print("Testing Full Model Quantization")
    print("=" * 70)
    
    # Import conversion functions
    from src.model_conversion import load_pytorch_model, convert_pytorch_to_jax, build_flax_pytree
    
    print("\nLoading GPT-2 from PyTorch...")
    pytorch_state_dict, tokenizer = load_pytorch_model(use_small_model=True)
    
    print("Converting to JAX...")
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    
    print("Building PyTree...")
    params_tree = build_flax_pytree(jax_state_dict)
    
    # Wrap in params structure (like Flax expects)
    params = {'params': params_tree}
    
    print(f"\n✓ Model loaded")
    print(f"  Parameters structure: {list(params.keys())}")
    
    # Quantize
    print("\nQuantizing parameters...")
    quantized_params, scales = quantize_model_params(params)
    
    print(f"\n✓ Quantization complete!")
    print(f"  Number of quantized parameters: {len(scales)}")
    
    # Measure memory
    def calculate_size(tree):
        """Calculate total size in bytes"""
        total = 0
        def count_leaf(leaf):
            nonlocal total
            if isinstance(leaf, jnp.ndarray):
                total += leaf.nbytes
        jax.tree_util.tree_map(count_leaf, tree)
        return total
    
    fp16_size = calculate_size(params)
    int8_size = calculate_size(quantized_params)
    
    print(f"\nMemory comparison:")
    print(f"  FP16 size: {fp16_size / 1e6:.2f} MB")
    print(f"  INT8 size: {int8_size / 1e6:.2f} MB")
    print(f"  Reduction: {fp16_size / int8_size:.2f}x")
    
    # Verify reduction is significant
    assert fp16_size / int8_size > 1.5, "Memory reduction not significant enough"
    
    print("\n✓ Model quantization test passed!")

if __name__ == "__main__":
    test_basic_quantization()
    test_model_quantization()
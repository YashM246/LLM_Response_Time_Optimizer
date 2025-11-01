import sys
import jax.numpy as jnp

sys.path.append('.')

from src.quantization import quantize_weights, dequantize_weights

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

if __name__ == "__main__":
    test_basic_quantization()
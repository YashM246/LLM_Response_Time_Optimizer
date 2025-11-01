import jax
import jax.numpy as jnp

def quantize_weights(weights:jnp.ndarray)-> tuple[jnp.ndarray, float]:
    # Function to convert FP16/FP32 arrays to INT8
    # Takes a floating point array, finds a max absolute value
    # Calculates a scale factor, and converts to INT8 range [-128,127]
    # Args:
    #           weights: Array of FP weights to quantize
    # Returns:
    #           quantized: INT8 array
    #           scale: Scaling factor for dequantization

    max_val = jnp.abs(weights).max()

    # Calculate scale
    scale = max_val/127.0
    # 127 to maintain symmetry around zero

    if scale == 0:
        scale = 1.0

    # Start quantizing
    #       - Divide by scale
    #       - round to nearest integer, clip to INT8 range

    quantized = jnp.round(weights/scale)
    quantized = jnp.clip(quantized, -128, 127)
    quantized = quantized.astype(jnp.int8)

    return quantized, scale

def dequantize_weights(quantized: jnp.ndarray, scale:float)-> jnp.ndarray:
    # Dequantize weights back to FP32
    #
    # Args:
    #           quantized: INT8 array from quantize_weights()
    #           scale: scaling factor from quantize_weights()
    # Returns:
    #           dequantized: FP32 array (approximate reconstruction)

    dequantized = quantized.astype(jnp.float32)

    dequantized = dequantized*scale

    return dequantized

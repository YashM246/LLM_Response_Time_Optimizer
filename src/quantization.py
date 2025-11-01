"""
INT8 Quantization for JAX/Flax models.

This module provides functions to:
1. Quantize FP16/FP32 weights to INT8 (reduce memory)
2. Dequantize INT8 back to FP32 (for inference)
3. Quantize entire model parameter PyTrees
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple

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

def should_quantize(path:str, array:jnp.ndarray)-> bool:
    # Decides if a parameter should be quantized
    # Only 2D matrices are quantized
    # Only large arrays
    # No norms and biases

    if array.ndim != 2:
        return False
    
    if array.size < 1000:
        return False
    
    if 'norm' in path.lower():
        return False
    
    if 'bias' in path.lower():
        return False
    
    return True

def quantize_model_params(params:Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    # Quantize entire model parameter PyTree
    # 
    # Strategy:
    #           - Quantize 2D weight matrices (linear layers, embeddings)
    #           - Keep 1D parameters in FP32 (biases, layernorm)
    #           - Skip small arrays (<1000 elements)
    #
    # Args:
    #           params: Flax PyTree with FP16/FP32 parameters
    #
    # Returns:
    #           quantized_params: PyTree with INT8 array (where applicable)
    #           scales: Dictionary mapping parameter path to scales

    scales = {}
    # To keep scales for each quantized parameter

    def quantize_leaf(path, leaf):

        # Convert path tuple to string for readability
        path_str = '/'.join(str(k.key) if hasattr(k, 'key') else str(k) for k in path)

        if should_quantize(path_str, leaf):
            quantized, scale = quantize_weights(leaf)
            scales[path_str] = scale
            print(f"  ✓ Quantized {path_str}: {leaf.shape} -> INT8 (scale: {scale:.6f})")
            return quantized

        else:
            print(f"  ○ Kept FP32 {path_str}: {leaf.shape}")
            return leaf
    
    # Traverse PyTree and apply quantization
    quantized_params = jax.tree_util.tree_map_with_path(quantize_leaf, params)

    return quantized_params, scales

"""
Manual PyTorch to JAX/Flax model conversion for Mistral-7B.

This module handles:
1. Loading PyTorch checkpoint
2. Mapping weight tensors to JAX arrays
3. Transposing linear layer weights (PyTorch vs Flax convention)
4. Building Flax PyTree structure
5. Initializing FlaxMistralForCausalLM with converted weights

"""

import torch
import jax
import jax.numpy as jnp
from transformers import (
    AutoModelForCausalLM,
    FlaxMistralForCausalLM,
    AutoTokenizer
)
from typing import Dict, Any
import numpy as np

def load_pytorch_model(model_type:str = "gpt2"):
    # Load PyTorch model and extract state_dict
    # Args:
    #           model_type: Either "gpt2" or "mistral"
    # Returns:
    #           pytorch_state_dict: Dictionary of PyTorch Tensors
    #           tokenizer: Loaded tokenizer
    #           model_type: The model type (for downstream use)

    if model_type=="gpt2":
        print("Loading GPT-2 model...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype= torch.float16,
                                                     device_map= "cpu")
    
    elif model_type=="mistral":
        print("Loading Mistral-7B model...")
        model_name = "mistralai/Mistral-7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='cpu',
            low_cpu_mem_usage=True
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'gpt2' or 'mistral'")
    
    state_dict = model.state_dict()

    print(f"[OK] Loaded {len(state_dict)} parameters from {model_name}")

    # show example keys for debugging
    print("Example weight keys:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"    {key}: {state_dict[key].shape}")

    return state_dict, tokenizer, model_type


def convert_pytorch_to_jax(pytorch_state_dict:Dict[str, torch.Tensor])->Dict[str, jnp.ndarray]:
    # Convert PyTorch state_dict to JAX Arrays
    #
    # 1) Convert torch.Tensor to numpy to jax.Array
    # 2) Transpose linear layer weights PyTorch[out,in] -> JAX[in,out]
    # 3) Rename parameters (weight->kernel, etc)
    #
    # Args:
    #           pytorch_state_dict: PyTorch model parameters
    # Returns:
    #           jax_state_dict: Flat Dictionary of JAX arrays with renamed keys

    jax_state_dict = {}

    for name, param in pytorch_state_dict.items():
        
        # First convert tensor to JAX array
        numpy_param = param.cpu().numpy()
        jax_param = jnp.array(numpy_param)

        # Check if transposition is required
        # Need to transpose 2D weights that are not embeddings
        is_2d_wt = name.endswith('.weight') and param.ndim==2
        is_embed = 'embed' in name or "wte" in name or "wpe" in name

        if is_2d_wt and not is_embed:
            # This will be a linear layer - Need to transpose
            jax_param = jax_param.T         # [out,in] = [in,out]
            print(f"  [OK] Transposed {name}: {param.shape} -> {jax_param.shape}")

        # Rename the parameter
        new_name = name
        if is_embed:
            # Embeddings: .weight -> .embedding
            new_name = name.replace('.weight', '.embedding')
        elif name.endswith('.weight'):
            # Linear layers: .weight -> .kernel
            new_name = name.replace('.weight', '.kernel')
        
        # Otherwise, keep the name
        jax_state_dict[new_name] = jax_param

    print(f"\n[OK] Converted {len(jax_state_dict)} parameters to JAX")
    return jax_state_dict

def build_flax_pytree(jax_state_dict: Dict[str, jnp.ndarray])-> Dict[str, Any]:
    # Convert flat JAX state_dict to nested Flax PyTree structure
    #
    # Example:
    #           Flat: {'model.layers.0.self_attn.q_proj.kernel': array(...)}
    #           Nested: {'model':{'layers':{'0': {'self_attn': {'q_proj': {'kernel': array(...)}}}}
    # Args:
    #           jax_state_dict: Flat dictionary of JAX Arrays
    # Returns:
    #           params: Nested PyTree structure

    pytree = {}

    for flat_key, value in jax_state_dict.items():

        # Split key into parts
        # Eg: 'transformer.h.0.attn.c_attn.kernel' → ['transformer', 'h', '0', 'attn', 'c_attn', 'kernel']
        keys = flat_key.split('.')

        # Navigate through tree, creating nested dicts as needed
        current = pytree
        # Process all keys except last
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}   # Create new nested dict
            current = current[key]  # Move deeper into tree

        # Set final value
        current[keys[-1]] = value
    
    print(f"[OK] Built PyTree with {len(jax_state_dict)} parameters")
    return pytree
    

def convert_model(model_type: str= "gpt2"):
    # Main Conversion Pipeline
    # PyTorch -> JAX/Flax
    #
    # Returns:
    #           params: Model parameters (PyTree wrapped in {'params': ...})
    #           tokenizer: Tokenizer
    #           model_type: The model type

    print("\n" + "=" * 60)
    print(f"PyTorch -> JAX Conversion Pipeline ({model_type.upper()})")
    print("=" * 60)
    
    print("\n[1/4] Loading PyTorch model...")
    pytorch_state_dict, tokenizer = load_pytorch_model(model_type)
    
    print("\n[2/4] Converting to JAX arrays...")
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    
    print("\n[3/4] Building Flax PyTree...")
    params_tree = build_flax_pytree(jax_state_dict)
    
    print("\n[4/4] Wrapping parameters...")
    params = {'params': params_tree} 
    
    print("\n[OK] Conversion complete!")
    print("=" * 60)
    
    return params, tokenizer, model_type
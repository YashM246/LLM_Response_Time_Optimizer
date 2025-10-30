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

def load_pytorch_model(model_name:str="mistralai/Mistral-7B-Instruct-v0.2", use_small_model:bool=False):
    # Load PyTorch Mistral model and extract state_dict
    # Args:
    #           model_name: HuggingFace Model Identifier
    #           use_small_model: If True, use a tiny model for local testing
    # Returns:
    #           pytorch_state_dict: Dictionary of PyTorch Tensors
    #           tokenizer: Loaded tokenizer

    if use_small_model:
        print("⚠️ Using GPT-2 for local testing (structure similar to Mistral)")
        test_model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(test_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            test_model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='cpu',
            low_cpu_mem_usage=True
            )
        
    state_dict = model.state_dict()

    # Inspect structure of state_dict
    # print(f"Loaded {len(state_dict)} parameters")
    # print("Example keys:")
    # for i, key in enumerate(list(state_dict.keys())[:5]):
    #     print(f"  {key}: {state_dict[key].shape}")

    return state_dict, tokenizer

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

    pass

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

    pass

def load_flax_model_with_params(params:Dict[str,Any],
                                model_name:str="mistralai/Mistral-7B-Instruct-v0.2")-> FlaxMistralForCausalLM:
    # Initiate FlaxMistralForCausalLM with converted parameters
    # Args:
    #           params: Flax PyTree parameters
    #           model_name: Model Identifier (For config)
    # Returns:
    #           model: FlaxMistralForCausalLM with loaded weights

    pass

def convert_model(model_name:str="mistralai/Mistral-7B-Instruct-v0.2"):
    # Main Conversion Pipeline
    # PyTorch -> JAX/Flax
    #
    # Returns:
    #           model: FlaxMistralForCausalLM
    #           tokenizer: Tokenizer
    #           params: Model parameters (PyTree)

    print("Step 1: Loading PyTorch Model ...")
    pytorch_state_dict, tokenizer = load_pytorch_model(model_name)

    print("Step 2: Converting to JAX Arrays ...")
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)

    print("Step 3: Building Flax PyTree ...")
    params = build_flax_pytree(jax_state_dict)

    print("Step 4: Initializing Flax Model ...")
    model = load_flax_model_with_params(params, model_name)

    print("--- Conversion Complete ---")
    return model, tokenizer, params
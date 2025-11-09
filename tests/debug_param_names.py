"""
Debug script to inspect converted GPT-2 parameter names.
"""

import sys
sys.path.append('.')

from src.model_conversion import load_pytorch_model, convert_pytorch_to_jax, build_flax_pytree


def print_nested_keys(d, prefix=''):
    """Recursively print dictionary keys."""
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            print(f"{full_key}:")
            print_nested_keys(value, full_key)
        else:
            # It's an array
            print(f"{full_key}: shape={value.shape}, dtype={value.dtype}")


print("Loading GPT-2...")
pytorch_state_dict, tokenizer = load_pytorch_model(use_small_model=True)

print("\nConverting to JAX...")
jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
params_tree = build_flax_pytree(jax_state_dict)
params = {'params': params_tree}

print("\n" + "=" * 70)
print("PARAMETER STRUCTURE")
print("=" * 70)

# Print top-level structure
print("\nTop level keys:")
print(list(params.keys()))

print("\nparams['params'] keys:")
print(list(params['params'].keys()))

print("\ntransformer keys:")
print(list(params['params']['transformer'].keys()))

print("\n" + "=" * 70)
print("LAYER 0 STRUCTURE")
print("=" * 70)
layer_0 = params['params']['transformer']['h']['0']
print("\nLayer 0 keys:")
print(list(layer_0.keys()))

if 'ln_1' in layer_0:
    print("\nln_1 keys:")
    print(list(layer_0['ln_1'].keys()))

if 'attn' in layer_0:
    print("\nattn keys:")
    print(list(layer_0['attn'].keys()))
    if 'c_attn' in layer_0['attn']:
        print("  c_attn keys:")
        print("  ", list(layer_0['attn']['c_attn'].keys()))
    if 'c_proj' in layer_0['attn']:
        print("  c_proj keys:")
        print("  ", list(layer_0['attn']['c_proj'].keys()))

if 'ln_2' in layer_0:
    print("\nln_2 keys:")
    print(list(layer_0['ln_2'].keys()))

if 'mlp' in layer_0:
    print("\nmlp keys:")
    print(list(layer_0['mlp'].keys()))
    if 'c_fc' in layer_0['mlp']:
        print("  c_fc keys:")
        print("  ", list(layer_0['mlp']['c_fc'].keys()))
    if 'c_proj' in layer_0['mlp']:
        print("  c_proj keys:")
        print("  ", list(layer_0['mlp']['c_proj'].keys()))

print("\n" + "=" * 70)
print("SAMPLE PARAMETER SHAPES")
print("=" * 70)

# Print full structure for layer 0
print("\nFull layer 0 structure:")
print_nested_keys(layer_0, "layer_0")

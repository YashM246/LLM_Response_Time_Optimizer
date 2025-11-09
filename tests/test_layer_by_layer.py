"""
Test to compare outputs layer by layer between PyTorch and JAX.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import jax.numpy as jnp
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.model_conversion import load_pytorch_model, convert_pytorch_to_jax, build_flax_pytree


def test_embeddings():
    """Test if embeddings match."""
    print("=" * 80)
    print("TEST 1: EMBEDDINGS")
    print("=" * 80)

    # Load models
    pytorch_model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    pytorch_state_dict, _ = load_pytorch_model(use_small_model=True)
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    params_tree = build_flax_pytree(jax_state_dict)
    jax_params = {'params': params_tree}

    # Test input
    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt)

    # PyTorch embeddings
    with torch.no_grad():
        pt_input = torch.tensor([input_ids])
        pt_token_emb = pytorch_model.transformer.wte(pt_input)
        pt_pos_emb = pytorch_model.transformer.wpe(torch.arange(len(input_ids)).unsqueeze(0))
        pt_embeddings = (pt_token_emb + pt_pos_emb).numpy()

    # JAX embeddings
    from src.cached_generation import get_embeddings
    jax_input = jnp.array([input_ids])
    jax_embeddings = get_embeddings(jax_input, jax_params, position=None, model_type="gpt2")
    jax_embeddings = np.array(jax_embeddings)

    # Compare
    diff = np.abs(pt_embeddings - jax_embeddings)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Embeddings shape: {pt_embeddings.shape}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-5:
        print("[PASS] Embeddings match!")
        return True
    else:
        print("[FAIL] Embeddings don't match!")
        print(f"PyTorch sample: {pt_embeddings[0, 0, :5]}")
        print(f"JAX sample: {jax_embeddings[0, 0, :5]}")
        return False


def test_first_layer():
    """Test if first transformer layer output matches."""
    print("\n" + "=" * 80)
    print("TEST 2: FIRST LAYER OUTPUT")
    print("=" * 80)

    # Load models
    pytorch_model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    pytorch_model.eval()

    pytorch_state_dict, _ = load_pytorch_model(use_small_model=True)
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    params_tree = build_flax_pytree(jax_state_dict)
    jax_params = {'params': params_tree}

    # Test input
    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt)

    # PyTorch: run through first layer
    with torch.no_grad():
        pt_input = torch.tensor([input_ids])
        # Get embeddings
        hidden = pytorch_model.transformer.wte(pt_input)
        positions = torch.arange(len(input_ids)).unsqueeze(0)
        hidden = hidden + pytorch_model.transformer.wpe(positions)
        # First layer
        hidden = pytorch_model.transformer.h[0](hidden)[0]
        pt_output = hidden.numpy()

    # JAX: run through first layer
    from src.cached_generation import get_embeddings, transformer_layer

    jax_input = jnp.array([input_ids])
    hidden = get_embeddings(jax_input, jax_params, position=None, model_type="gpt2")

    # First layer
    layer_params = jax_params['params']['transformer']['h']['0']
    hidden, _ = transformer_layer(
        hidden_states=hidden,
        layer_params=layer_params,
        cache=None,
        layer_idx=0,
        position=0,
        num_heads=12,
        use_cache=False,
        model_type="gpt2"
    )
    jax_output = np.array(hidden)

    # Compare
    diff = np.abs(pt_output - jax_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Output shape: {pt_output.shape}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-3:
        print("[PASS] First layer output matches!")
        return True
    else:
        print("[FAIL] First layer output doesn't match!")
        print(f"PyTorch sample: {pt_output[0, 0, :5]}")
        print(f"JAX sample: {jax_output[0, 0, :5]}")
        return False


def test_final_layer_norm():
    """Test if final layer norm matches."""
    print("\n" + "=" * 80)
    print("TEST 3: FINAL LAYER NORM")
    print("=" * 80)

    # Load models
    pytorch_model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    pytorch_model.eval()

    pytorch_state_dict, _ = load_pytorch_model(use_small_model=True)
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    params_tree = build_flax_pytree(jax_state_dict)
    jax_params = {'params': params_tree}

    # Test input
    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt)

    # PyTorch: run through all layers but before LM head
    with torch.no_grad():
        pt_input = torch.tensor([input_ids])
        hidden = pytorch_model.transformer(pt_input)[0]  # All layers + final LN
        pt_output = hidden.numpy()

    # JAX: run through all layers
    from src.cached_generation import get_embeddings, transformer_layer, layer_norm

    jax_input = jnp.array([input_ids])
    hidden = get_embeddings(jax_input, jax_params, position=None, model_type="gpt2")

    # All layers
    for layer_idx in range(12):
        layer_params = jax_params['params']['transformer']['h'][str(layer_idx)]
        hidden, _ = transformer_layer(
            hidden_states=hidden,
            layer_params=layer_params,
            cache=None,
            layer_idx=layer_idx,
            position=0,
            num_heads=12,
            use_cache=False,
            model_type="gpt2"
        )

    # Final layer norm
    ln_f_weight = jax_params['params']['transformer']['ln_f']['kernel']
    ln_f_bias = jax_params['params']['transformer']['ln_f']['bias']
    hidden = layer_norm(hidden, ln_f_weight, ln_f_bias)

    jax_output = np.array(hidden)

    # Compare
    diff = np.abs(pt_output - jax_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Output shape: {pt_output.shape}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-2:
        print("[PASS] Final layer norm matches!")
        return True
    else:
        print("[FAIL] Final layer norm doesn't match!")
        print(f"PyTorch sample: {pt_output[0, -1, :5]}")
        print(f"JAX sample: {jax_output[0, -1, :5]}")
        return False


def test_lm_head_projection():
    """Test if LM head projection matches."""
    print("\n" + "=" * 80)
    print("TEST 4: LM HEAD PROJECTION")
    print("=" * 80)

    # Load models
    pytorch_model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    pytorch_model.eval()

    pytorch_state_dict, _ = load_pytorch_model(use_small_model=True)
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    params_tree = build_flax_pytree(jax_state_dict)
    jax_params = {'params': params_tree}

    # Test input
    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt)

    # PyTorch: full forward pass
    with torch.no_grad():
        pt_input = torch.tensor([input_ids])
        pt_logits = pytorch_model(pt_input).logits.numpy()

    # JAX: full forward pass
    from src.cached_generation import get_embeddings, transformer_layer, lm_head

    jax_input = jnp.array([input_ids])
    hidden = get_embeddings(jax_input, jax_params, position=None, model_type="gpt2")

    # All layers
    for layer_idx in range(12):
        layer_params = jax_params['params']['transformer']['h'][str(layer_idx)]
        hidden, _ = transformer_layer(
            hidden_states=hidden,
            layer_params=layer_params,
            cache=None,
            layer_idx=layer_idx,
            position=0,
            num_heads=12,
            use_cache=False,
            model_type="gpt2"
        )

    # LM head
    jax_logits = lm_head(hidden, jax_params, model_type="gpt2")
    jax_logits = np.array(jax_logits)

    # Compare
    diff = np.abs(pt_logits - jax_logits)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Logits shape: {pt_logits.shape}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    # Check top-5 predictions
    pt_top5 = np.argsort(pt_logits[0, -1])[-5:][::-1]
    jax_top5 = np.argsort(jax_logits[0, -1])[-5:][::-1]

    print("\nPyTorch top-5:")
    for idx in pt_top5:
        print(f"  {idx}: {pt_logits[0, -1, idx]:.4f}")

    print("\nJAX top-5:")
    for idx in jax_top5:
        print(f"  {idx}: {jax_logits[0, -1, idx]:.4f}")

    if max_diff < 1e-1:
        print("\n[PASS] LM head matches!")
        return True
    else:
        print("\n[FAIL] LM head doesn't match!")
        return False


if __name__ == "__main__":
    # Run all tests
    results = []
    results.append(("Embeddings", test_embeddings()))
    results.append(("First Layer", test_first_layer()))
    results.append(("Final Layer Norm", test_final_layer_norm()))
    results.append(("LM Head", test_lm_head_projection()))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name}: {status}")

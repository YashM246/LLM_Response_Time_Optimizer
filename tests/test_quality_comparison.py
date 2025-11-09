"""
Test to compare generation quality between PyTorch and JAX models.

This will help identify why JAX model generates repetitive text.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import jax
import jax.numpy as jnp
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.model_conversion import load_pytorch_model, convert_pytorch_to_jax, build_flax_pytree
from src.cached_generation import generate_text_with_cache


def test_pytorch_baseline():
    """Test PyTorch GPT-2 baseline generation."""

    print("=" * 80)
    print("PYTORCH BASELINE TEST")
    print("=" * 80)

    # Load PyTorch model
    print("\nLoading PyTorch GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()

    # Test prompts
    prompts = [
        "The quick brown fox",
        "Hello, my name is",
        "Once upon a time"
    ]

    print("\n" + "-" * 80)
    print("PYTORCH GENERATION (Greedy Decoding)")
    print("-" * 80)

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        # Encode
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # Generate with greedy decoding (temperature=0 equivalent)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,  # Greedy (deterministic)
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print(f"New tokens: {tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)}")


def test_jax_generation():
    """Test JAX generation with our implementation."""

    print("\n" + "=" * 80)
    print("JAX GENERATION TEST")
    print("=" * 80)

    # Load JAX model
    print("\nLoading and converting to JAX...")
    pytorch_state_dict, tokenizer = load_pytorch_model(use_small_model=True)
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    params_tree = build_flax_pytree(jax_state_dict)
    params = {'params': params_tree}

    # Test prompts (same as PyTorch)
    prompts = [
        "The quick brown fox",
        "Hello, my name is",
        "Once upon a time"
    ]

    print("\n" + "-" * 80)
    print("JAX GENERATION (Greedy Decoding)")
    print("-" * 80)

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        # Generate with greedy (temperature=0.0)
        generated_text, stats = generate_text_with_cache(
            params=params,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.0,  # Greedy (should be deterministic)
            top_k=50,
            use_cache=True,
            model_type="gpt2"
        )

        print(f"Generated: {generated_text}")
        # Extract just the new tokens
        prompt_tokens = tokenizer.encode(prompt)
        full_tokens = tokenizer.encode(generated_text)
        new_tokens = full_tokens[len(prompt_tokens):]
        print(f"New tokens: {tokenizer.decode(new_tokens)}")


def test_token_by_token_comparison():
    """Compare PyTorch and JAX generation token-by-token."""

    print("\n" + "=" * 80)
    print("TOKEN-BY-TOKEN COMPARISON")
    print("=" * 80)

    prompt = "The quick brown fox"
    max_new_tokens = 10

    # Load models
    print("\nLoading models...")

    # PyTorch
    pytorch_model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    pytorch_model.eval()

    # JAX
    pytorch_state_dict, _ = load_pytorch_model(use_small_model=True)
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    params_tree = build_flax_pytree(jax_state_dict)
    jax_params = {'params': params_tree}

    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {max_new_tokens} tokens...\n")

    # Encode prompt
    input_ids_pt = tokenizer.encode(prompt, return_tensors='pt')

    # PyTorch generation (manual loop for comparison)
    print("-" * 80)
    print("PYTORCH TOKEN-BY-TOKEN:")
    print("-" * 80)

    pytorch_tokens = []
    current_ids = input_ids_pt

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = pytorch_model(current_ids)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Greedy: pick token with highest logit
            next_token = torch.argmax(logits).item()
            pytorch_tokens.append(next_token)

            # Show token and top-5 logits
            top5_logits, top5_indices = torch.topk(logits, 5)
            print(f"Step {step + 1}:")
            print(f"  Selected: {next_token} ('{tokenizer.decode([next_token])}')")
            print(f"  Top-5 logits: {top5_logits.tolist()[:3]}...")

            # Append and continue
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]])], dim=1)

    pytorch_text = tokenizer.decode(pytorch_tokens)
    print(f"\nPyTorch generated: '{pytorch_text}'")

    # JAX generation
    print("\n" + "-" * 80)
    print("JAX TOKEN-BY-TOKEN:")
    print("-" * 80)

    jax_generated, _ = generate_text_with_cache(
        params=jax_params,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=50,
        use_cache=True,
        model_type="gpt2"
    )

    prompt_len = len(tokenizer.encode(prompt))
    full_tokens = tokenizer.encode(jax_generated)
    jax_tokens = full_tokens[prompt_len:]
    jax_text = tokenizer.decode(jax_tokens)

    print(f"JAX generated: '{jax_text}'")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    print(f"PyTorch: '{pytorch_text}'")
    print(f"JAX:     '{jax_text}'")
    print(f"\nTokens match: {pytorch_tokens == jax_tokens}")

    if pytorch_tokens != jax_tokens:
        print("\n⚠️  DIVERGENCE DETECTED!")
        print("\nToken-by-token comparison:")
        for i, (pt_tok, jax_tok) in enumerate(zip(pytorch_tokens, jax_tokens)):
            match = "✓" if pt_tok == jax_tok else "✗"
            print(f"  {i+1}. PyTorch: {pt_tok} ('{tokenizer.decode([pt_tok])}')  |  "
                  f"JAX: {jax_tok} ('{tokenizer.decode([jax_tok])}')  {match}")
            if pt_tok != jax_tok:
                print(f"     >>> First divergence at position {i+1}")
                break
    else:
        print("✅ Outputs are IDENTICAL!")


def test_logits_comparison():
    """Compare logits from PyTorch and JAX on same input."""

    print("\n" + "=" * 80)
    print("LOGITS COMPARISON TEST")
    print("=" * 80)

    prompt = "The quick brown fox"

    print(f"\nPrompt: '{prompt}'")
    print("Comparing logits from both models...\n")

    # Load models
    pytorch_model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    pytorch_model.eval()

    pytorch_state_dict, _ = load_pytorch_model(use_small_model=True)
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    params_tree = build_flax_pytree(jax_state_dict)
    jax_params = {'params': params_tree}

    # Encode
    input_ids = tokenizer.encode(prompt)

    # PyTorch forward pass
    with torch.no_grad():
        pt_input = torch.tensor([input_ids])
        pt_outputs = pytorch_model(pt_input)
        pt_logits = pt_outputs.logits[0, -1, :].numpy()  # Last token logits

    # JAX forward pass
    from src.cached_generation import get_embeddings, transformer_layer, lm_head
    from src.kv_cache import initialize_cache

    config = {
        'num_layers': 12,
        'num_heads': 12,
        'hidden_dim': 768,
        'max_seq_len': 1024,
        'vocab_size': 50257
    }

    # Process through JAX model
    jax_input = jnp.array([input_ids])
    hidden_states = get_embeddings(jax_input, jax_params, position=None, model_type="gpt2")

    cache = None  # Use batch mode (no cache)
    for layer_idx in range(config['num_layers']):
        layer_params = jax_params['params']['transformer']['h'][str(layer_idx)]
        hidden_states, _ = transformer_layer(
            hidden_states=hidden_states,
            layer_params=layer_params,
            cache=cache,
            layer_idx=layer_idx,
            position=0,
            num_heads=config['num_heads'],
            use_cache=False,
            model_type="gpt2"
        )

    jax_logits_output = lm_head(hidden_states, jax_params, model_type="gpt2")
    jax_logits = np.array(jax_logits_output[0, -1, :])  # Last token logits

    # Compare
    logits_diff = np.abs(pt_logits - jax_logits)
    max_diff = np.max(logits_diff)
    mean_diff = np.mean(logits_diff)

    print(f"Logits shape: {pt_logits.shape}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    # Top-5 from each
    pt_top5_indices = np.argsort(pt_logits)[-5:][::-1]
    jax_top5_indices = np.argsort(jax_logits)[-5:][::-1]

    print("\nPyTorch top-5:")
    for i, idx in enumerate(pt_top5_indices):
        print(f"  {i+1}. {idx} ('{tokenizer.decode([idx])}') - {pt_logits[idx]:.4f}")

    print("\nJAX top-5:")
    for i, idx in enumerate(jax_top5_indices):
        print(f"  {i+1}. {idx} ('{tokenizer.decode([idx])}') - {jax_logits[idx]:.4f}")

    if max_diff < 1e-3:
        print("\n✅ Logits are very close (diff < 1e-3)")
    elif max_diff < 1e-1:
        print("\n⚠️  Logits have small differences (diff < 1e-1)")
    else:
        print("\n❌ Logits are significantly different!")


if __name__ == "__main__":
    # Run all tests
    test_pytorch_baseline()
    test_jax_generation()
    test_token_by_token_comparison()
    test_logits_comparison()

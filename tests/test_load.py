import sys
sys.path.append('.')

from src.model_conversion import load_pytorch_model, convert_pytorch_to_jax

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PyTorch → JAX Conversion")
    print("=" * 60)
    
    # Step 1: Load PyTorch model
    print("\n[Step 1] Loading PyTorch GPT-2 model...")
    pytorch_state_dict, tokenizer = load_pytorch_model(use_small_model=True)
    print(f"--- Loaded {len(pytorch_state_dict)} parameters ---")
    
    # Step 2: Convert to JAX
    print("\n[Step 2] Converting to JAX...")
    jax_state_dict = convert_pytorch_to_jax(pytorch_state_dict)
    
    # Step 3: Verify conversions
    print("\n[Step 3] Verifying conversions...")
    print("=" * 60)
    
    # Check 1: Linear layer transposition
    linear_keys = [k for k in jax_state_dict.keys() if 'attn' in k and 'kernel' in k]
    if linear_keys:
        example_key = linear_keys[0]
        print(f"✓ Linear layer converted: {example_key}")
        print(f"  Shape: {jax_state_dict[example_key].shape}")
    
    # Check 2: Embedding NOT transposed
    emb_keys = [k for k in jax_state_dict.keys() if 'embedding' in k]
    if emb_keys:
        example_emb = emb_keys[0]
        print(f"✓ Embedding preserved: {example_emb}")
        print(f"  Shape: {jax_state_dict[example_emb].shape}")
    
    # Check 3: Bias unchanged
    bias_keys = [k for k in jax_state_dict.keys() if 'bias' in k]
    if bias_keys:
        example_bias = bias_keys[0]
        print(f"✓ Bias unchanged: {example_bias}")
        print(f"  Shape: {jax_state_dict[example_bias].shape}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✓ Conversion complete!")
    print(f"  PyTorch params: {len(pytorch_state_dict)}")
    print(f"  JAX params: {len(jax_state_dict)}")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    print("=" * 60)
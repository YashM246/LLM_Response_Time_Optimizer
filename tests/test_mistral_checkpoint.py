import sys
import orbax.checkpoint as ocp
import os

sys.path.append('.')

# Use os.path.join for cross-platform compatibility (Windows uses backslashes)
checkpoint_dir = os.path.join('models', 'Mistral-7B-JAX', 'mistral_jax_checkpoint')
params_path = os.path.join(checkpoint_dir, 'params')

print("=" * 70)
print("Testing Mistral JAX Checkpoint Load")
print("=" * 70)
print(f"Checkpoint path: {params_path}")

try:
    # Verify directory exists
    if not os.path.exists(params_path):
        print(f"\n✗ Error: Directory does not exist: {params_path}")
        print(f"   Current working directory: {os.getcwd()}")
        sys.exit(1)

    print("\nLoading checkpoint...")
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(params_path)
    
    print(f"✓ Checkpoint loaded successfully!")
    print(f"✓ Top-level keys: {list(params.keys())}")
    
    if 'params' in params:
        inner = params['params']
        print(f"✓ Inner keys: {list(inner.keys())}")
        
        if 'model' in inner:
            model_params = inner['model']
            print(f"✓ Model components: {list(model_params.keys())}")
            
            if 'layers' in model_params:
                print(f"✓ Number of layers: {len(model_params['layers'])}")
                
                # Check layer 0
                layer_0 = model_params['layers']['0']
                print(f"✓ Layer 0 components: {list(layer_0.keys())}")
            
            if 'embed_tokens' in model_params and 'embedding' in model_params['embed_tokens']:
                emb = model_params['embed_tokens']['embedding']
                print(f"✓ Embedding shape: {emb.shape}")
                print(f"✓ Embedding dtype: {emb.dtype}")
    
    print("\n" + "=" * 70)
    print("✓ Checkpoint is valid and ready to use!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
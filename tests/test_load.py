import sys
sys.path.append('.')

from src.model_conversion import convert_model

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Full PyTorch → JAX Conversion Pipeline")
    print("=" * 60)
    
    try:
        # Run full conversion
        params, tokenizer, model_type = convert_model(model_type="gpt2")
        
        print("\n" + "=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        print(f"✓ Model type: {model_type}")
        print(f"✓ Tokenizer vocab size: {len(tokenizer)}")
        print(f"✓ Top-level param keys: {list(params.keys())}")
        
        # Inspect parameter structure
        if 'params' in params:
            print(f"✓ Nested param keys: {list(params['params'].keys())}")
        
        print("\n" + "=" * 60)
        print("✓ Full conversion pipeline successful!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
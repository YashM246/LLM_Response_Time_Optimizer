import sys
sys.path.append('.')

from src.model_conversion import convert_model

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Full PyTorch → JAX Conversion Pipeline")
    print("=" * 60)
    
    try:
        # Run full conversion
        model, params, tokenizer = convert_model(use_small_model=True)
        
        print("\n" + "=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        print(f"✓ Model type: {type(model).__name__}")
        print(f"✓ Tokenizer vocab size: {len(tokenizer)}")
        print(f"✓ Top-level param keys: {list(params.keys())}")
        
        # Try to inspect model structure
        print(f"✓ Model config: {model.config.to_dict()}")
        
        print("\n" + "=" * 60)
        print("✓ Full conversion pipeline successful!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
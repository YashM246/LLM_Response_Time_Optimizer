import sys
sys.path.append('.')

from src.model_conversion import convert_model

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Mistral-7B Loading")
    print("=" * 60)

    try:
        # Load Mistral-7B
        print("\n[1/1] Loading Mistral-7B model...")
        params, tokenizer, model_type = convert_model(model_type="mistral")

        print("\n" + "=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        print(f"[OK] Model type: {model_type}")
        print(f"[OK] Tokenizer vocab size: {len(tokenizer)}")
        print(f"[OK] Top-level param keys: {list(params.keys())}")

        # Inspect parameter structure
        if 'params' in params:
            model_keys = list(params['params'].keys())
            print(f"[OK] Nested param keys: {model_keys}")

            # Check for expected Mistral structure
            if 'model' in params['params']:
                print(f"[OK] Found 'model' key (Mistral structure)")
                model_params = params['params']['model']
                if 'layers' in model_params:
                    num_layers = len(model_params['layers'])
                    print(f"[OK] Number of layers: {num_layers}")

                    # Inspect first layer structure
                    if '0' in model_params['layers']:
                        layer_0_keys = list(model_params['layers']['0'].keys())
                        print(f"[OK] Layer 0 keys: {layer_0_keys}")

        print("\n" + "=" * 60)
        print("[OK] Mistral-7B loading test successful!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

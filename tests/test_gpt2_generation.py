import sys
sys.path.append('.')

from src.model_conversion import convert_model
from src.cached_generation import generate_text_with_cache

if __name__ == "__main__":
    print("=" * 60)
    print("Testing GPT-2 Generation (Regression Test)")
    print("=" * 60)

    try:
        # Load GPT-2
        print("\n[1/2] Loading GPT-2 model...")
        params, tokenizer, model_type = convert_model(model_type="gpt2")
        print(f"[OK] Model loaded: {model_type}")

        # Test generation
        print("\n[2/2] Testing text generation...")
        prompt = "The quick brown fox"
        print(f"Prompt: '{prompt}'")

        generated_text, stats = generate_text_with_cache(
            params=params,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            temperature=1.0,
            use_cache=True,
            model_type="gpt2"
        )

        print(f"\n[OK] Generated text: '{generated_text}'")
        print(f"[OK] Generation stats: {stats}")

        print("\n" + "=" * 60)
        print("[OK] GPT-2 generation test successful!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

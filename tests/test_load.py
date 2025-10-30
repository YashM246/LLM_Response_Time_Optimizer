import sys
sys.path.append('.')

from src.model_conversion import load_pytorch_model

if __name__ == "__main__":
    print("--- Testing PyTorch Model Loading ---")
    # Using Small model for local testing
    state_dict, tokenizer = load_pytorch_model(use_small_model=True)

    print(f"\n--- Loaded {len(state_dict)} parameters ---")
    print(f"--- Tokenizer Vocab Size: {len(tokenizer)} ---")

    # Check for specific layers
    expected_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight"
    ]

    for key in expected_keys:
        if key in state_dict:
            print(f"✓ Found {key}")
        else:
            print(f"✗ Missing {key}")
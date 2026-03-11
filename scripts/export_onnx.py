#!/usr/bin/env python3
"""Export the MiniLM model to ONNX format for runtime inference.

Requires: pip install sentence-transformers torch onnx

Usage:
    python scripts/export_onnx.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mental_models_mcp.models import DATA_DIR

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Get the transformer model
    transformer = model[0].auto_model
    tokenizer = model[0].tokenizer

    # Export to ONNX
    onnx_path = DATA_DIR / "model.onnx"
    tokenizer_path = DATA_DIR / "tokenizer"

    print(f"Exporting ONNX model to {onnx_path}...")

    # Create dummy input
    dummy_text = "This is a test sentence for export"
    encoded = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)

    # Export
    torch.onnx.export(
        transformer,
        (encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
    )
    print(f"ONNX model saved to {onnx_path}")

    # Save tokenizer
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_path))

    # Also save as tokenizers JSON for the Tokenizer library
    from tokenizers import Tokenizer as HFTokenizer

    fast_tokenizer = HFTokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    fast_tokenizer.save(str(tokenizer_path / "tokenizer.json"))
    print(f"Tokenizer saved to {tokenizer_path}")

    # Verify
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    print(f"ONNX model verified. Inputs: {[i.name for i in session.get_inputs()]}")
    print(f"Output shape: {session.get_outputs()[0].shape}")
    print("Export complete!")


if __name__ == "__main__":
    main()

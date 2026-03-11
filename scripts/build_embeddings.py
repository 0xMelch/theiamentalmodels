#!/usr/bin/env python3
"""Build pre-computed embeddings for all mental models.

Uses sentence-transformers with all-MiniLM-L6-v2 model.
Requires: pip install sentence-transformers torch

Usage:
    python scripts/build_embeddings.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mental_models_mcp.models import DATA_DIR

MODEL_NAME = "all-MiniLM-L6-v2"


def model_text(model: dict) -> str:
    """Create composite embedding text for a model."""
    parts = [
        model.get("name", ""),
        model.get("discipline", ""),
        model.get("summary", ""),
        model.get("heart", "") or "",
    ]
    return ". ".join(p for p in parts if p)


def main():
    parsed_path = DATA_DIR / "models_parsed.json"
    if not parsed_path.exists():
        print(f"Error: {parsed_path} not found. Run parse_models.py first.")
        sys.exit(1)

    with open(parsed_path, "r", encoding="utf-8") as f:
        models = json.load(f)

    print(f"Loaded {len(models)} models")
    print(f"Loading sentence-transformers model: {MODEL_NAME}...")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(MODEL_NAME)

    # Build composite texts
    texts = [model_text(m) for m in models]
    ids = [m["id"] for m in models]

    print(f"Encoding {len(texts)} texts...")
    embeddings = encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    print(f"Embeddings shape: {embeddings.shape}")

    output_path = DATA_DIR / "embeddings.npz"
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        ids=np.array(ids),
    )
    print(f"Saved to {output_path}")

    # Quick sanity check
    query = "negotiation strategy when both parties have strong alternatives"
    query_emb = encoder.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_emb
    top_5 = np.argsort(scores)[::-1][:5]
    print("\nSanity check — top 5 for 'negotiation strategy':")
    for i in top_5:
        print(f"  {scores[i]:.4f}  {models[i]['name']} ({models[i]['discipline']})")


if __name__ == "__main__":
    main()

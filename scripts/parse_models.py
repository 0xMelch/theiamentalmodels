#!/usr/bin/env python3
"""Parse raw mental_models.json into structured models_parsed.json.

Usage:
    python scripts/parse_models.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mental_models_mcp.models import DATA_DIR, parse_all_models, save_parsed_models


def main():
    raw_path = DATA_DIR / "mental_models.json"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found")
        sys.exit(1)

    print(f"Parsing models from {raw_path}...")
    models = parse_all_models(raw_path)
    print(f"Parsed {len(models)} models")

    # Validate
    disciplines = set(m["discipline"] for m in models)
    print(f"Disciplines: {sorted(disciplines)}")

    with_heart = sum(1 for m in models if m.get("heart"))
    with_limitations = sum(1 for m in models if m.get("limitations"))
    with_math = sum(1 for m in models if m.get("math"))
    with_examples = sum(1 for m in models if m.get("examples"))
    print(f"Heart of the Model: {with_heart}/{len(models)}")
    print(f"Limitations: {with_limitations}/{len(models)}")
    print(f"Math Intuition: {with_math}/{len(models)}")
    print(f"Examples: {with_examples}/{len(models)}")

    output_path = DATA_DIR / "models_parsed.json"
    save_parsed_models(models, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

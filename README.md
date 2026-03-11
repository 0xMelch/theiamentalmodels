# Mental Models MCP Server

An MCP server that exposes 700 mental models as a decision-coaching tool. Describe a situation — deal, negotiation, resource allocation, investment — and the server surfaces relevant mental models, asks sharpening questions, and provides structured analysis.

The models are the engine. You don't need to know which model is being applied.

## Install

```bash
pip install mental-models-mcp
```

### Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mental-models": {
      "command": "python",
      "args": ["-m", "mental_models_mcp"]
    }
  }
}
```

## Tools

### `evaluate_decision`

The primary tool. Takes a decision context and returns structured coaching through three stages:

- **initial_framing** — Surfaces 5-8 relevant models, sharpening questions, and key tensions
- **deep_analysis** — Rigorously applies top 3-5 models with predictions, variables, and blind spots
- **final_check** — Pre-mortem stress test with contrarian models

```
situation: "We're considering acquiring a competitor for $50M. They have strong tech but declining revenue."
decision_type: "deal_evaluation"
stage: "initial_framing"
```

Decision types: `deal_evaluation`, `negotiation`, `resource_allocation`, `investment`, `hiring`, `strategy`, `general`

### `search_models`

Semantic search across all 700 models. Returns name, discipline, summary, and relevance score.

```
query: "cognitive biases in group decision making"
top_k: 5
```

### `get_model`

Fetch a single model by name (exact or fuzzy match). Returns the full parsed structure including heart of the model, limitations, math intuition, and examples.

```
name: "Nash Equilibrium"
```

### `list_disciplines`

Returns all disciplines with model counts and chapter breakdowns. No input needed.

## How It Works

1. **Pre-computed embeddings** — All 700 models are embedded at build time using all-MiniLM-L6-v2
2. **Semantic search** — User situations are embedded at runtime using ONNX Runtime (no API key needed)
3. **Decision-type boosting** — Relevant disciplines get boosted based on decision type
4. **Stage-based analysis** — Three stages mirror how good advisors work: frame → analyze → stress-test

## Disciplines

The 700 models span 10 disciplines:

- Behavioral Economics
- Biology
- Economics
- Elementary Models
- Financial Theory
- Game Theory
- Investing
- Philosophy
- Physics
- Probability

## Development

### Prerequisites

```bash
pip install -e ".[build,dev]"
```

### Build Pipeline

1. Parse raw models:
   ```bash
   python scripts/parse_models.py
   ```

2. Generate embeddings:
   ```bash
   python scripts/build_embeddings.py
   ```

3. Export ONNX model:
   ```bash
   python scripts/export_onnx.py
   ```

4. Run tests:
   ```bash
   pytest tests/
   ```

### Project Structure

```
src/mental_models_mcp/
├── __init__.py
├── __main__.py          # Entry point
├── server.py            # MCP server + tool handlers
├── models.py            # Data loading, parsing, indexing
├── search.py            # Embedding-based semantic search
├── prompts.py           # System prompts and templates
└── data/
    ├── mental_models.json       # Raw models
    ├── models_parsed.json       # Parsed structured format
    └── embeddings.npz           # Pre-computed embeddings
```

## License

MIT

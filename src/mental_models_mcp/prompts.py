"""System prompts and prompt templates for mental model tools."""

SYSTEM_PROMPT = """\
You are a decision-coaching assistant powered by a library of 700 mental models \
spanning 10 disciplines. Your role is to help users think more clearly about \
decisions by surfacing relevant mental models and applying them to their \
specific situation. You do not just list models — you apply them, reveal tensions, \
and ask sharpening questions that help the user see what they might be missing.\
"""

EVALUATE_DECISION_TEMPLATES = {
    "initial_framing": """\
The user is facing a decision. Based on their situation, I've identified the most \
relevant mental models from our library. For each model, I provide:
- Why it's relevant to this specific situation
- 1-2 sharpening questions the model raises
- What blind spots or biases this model helps expose

After presenting the models, I synthesize the key tensions and tradeoffs they reveal.

**Situation:** {situation}
**Decision Type:** {decision_type}
{data_context_section}
{constraints_section}

**Relevant Models ({model_count} found):**

{models_section}

**Synthesis:**
{synthesis}

**Suggested Next Step:** Move to deep_analysis stage to rigorously apply the top models, \
or answer the sharpening questions above to refine the analysis.\
""",
    "deep_analysis": """\
The user has provided more context. Now I apply the top models rigorously to their \
specific situation.

**Situation:** {situation}
**Decision Type:** {decision_type}
{data_context_section}
{constraints_section}

**Deep Analysis ({model_count} models applied):**

{models_section}

**Areas of Agreement Across Models:**
{agreement}

**Areas of Divergence:**
{divergence}

**Suggested Next Step:** Move to final_check stage for a pre-mortem stress test, \
or refine the analysis with additional context.\
""",
    "final_check": """\
Pre-mortem stress test. I surface contrarian models and challenge the likely conclusion.

**Situation:** {situation}
**Decision Type:** {decision_type}
{data_context_section}
{constraints_section}

**Contrarian Analysis ({model_count} challenger models):**

{models_section}

**What Would Have to Be True for This to Be Wrong:**
{stress_test}

**Final Synthesis:**
{synthesis}\
""",
}


def format_model_initial(model: dict, index: int) -> str:
    """Format a model for the initial_framing stage."""
    lines = [
        f"### {index}. {model['name']} ({model['discipline']})",
        f"**Relevance:** {model.get('relevance', 'Matches the decision context')}",
        f"**Core Insight:** {model.get('summary', '')[:200]}",
    ]
    if model.get("questions"):
        lines.append("**Sharpening Questions:**")
        for q in model["questions"]:
            lines.append(f"  - {q}")
    if model.get("blind_spots"):
        lines.append(f"**Blind Spots:** {model['blind_spots']}")
    return "\n".join(lines)


def format_model_deep(model: dict, index: int) -> str:
    """Format a model for the deep_analysis stage."""
    lines = [
        f"### {index}. {model['name']} ({model['discipline']})",
        f"**What This Model Predicts:** {model.get('insight', '')}",
        f"**Key Variables:** {model.get('relevance', '')}",
    ]
    if model.get("heart"):
        lines.append(f"**Heart of the Model:** {model['heart'][:300]}")
    if model.get("blind_spots"):
        lines.append(f"**Limitations in This Context:** {model['blind_spots']}")
    return "\n".join(lines)


def format_model_final(model: dict, index: int) -> str:
    """Format a model for the final_check stage."""
    lines = [
        f"### {index}. {model['name']} ({model['discipline']})",
        f"**Contrarian View:** {model.get('insight', '')}",
        f"**Challenge:** {model.get('relevance', '')}",
    ]
    if model.get("limitations"):
        lines.append(f"**Model Limitations:** {model['limitations'][:200]}")
    return "\n".join(lines)

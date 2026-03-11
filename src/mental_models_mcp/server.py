"""MCP server definition and tool handlers."""

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from .models import ModelIndex
from .search import ModelSearch

logger = logging.getLogger(__name__)

app = Server("mental-models-mcp")

# Global state — initialized on first use
_index: ModelIndex | None = None
_search: ModelSearch | None = None


def _get_index() -> ModelIndex:
    global _index
    if _index is None:
        _index = ModelIndex()
    return _index


def _get_search() -> ModelSearch:
    global _search
    if _search is None:
        index = _get_index()
        _search = ModelSearch(parsed_models=index.all_models)
    return _search


def _build_query(situation: str, decision_type: str = "", data_context: str = "") -> str:
    """Build a composite search query from user inputs."""
    parts = [situation]
    if decision_type and decision_type != "general":
        parts.append(decision_type.replace("_", " "))
    if data_context:
        parts.append(data_context)
    return ". ".join(parts)


def _generate_questions(model: dict, situation: str) -> list[str]:
    """Generate sharpening questions based on a model and situation."""
    questions = []
    name = model.get("name", "")
    heart = model.get("heart", "") or ""
    limitations = model.get("limitations", "") or ""

    if heart:
        questions.append(
            f"Considering {name}: what assumptions are you making that this model would challenge?"
        )
    if limitations:
        questions.append(
            f"Where might {name}'s limitations apply to your specific situation?"
        )
    if not questions:
        questions.append(
            f"How does {name} change your perspective on this decision?"
        )
    return questions[:2]


def _generate_synthesis(models: list[dict], stage: str) -> str:
    """Generate a synthesis of the models applied."""
    if not models:
        return "No strongly relevant models found. Try providing more context."

    disciplines = set(m.get("discipline", "") for m in models)
    names = [m.get("name", "") for m in models]

    if stage == "initial_framing":
        return (
            f"Across {len(disciplines)} disciplines ({', '.join(sorted(disciplines))}), "
            f"the models {', '.join(names[:3])} and others highlight key tensions in your "
            f"situation. The sharpening questions above will help refine the analysis. "
            f"Consider which models resonate most before moving to deep analysis."
        )
    elif stage == "deep_analysis":
        return (
            f"The {len(models)} models applied span {', '.join(sorted(disciplines))}. "
            f"Look for where they agree — those are likely robust insights. "
            f"Where they diverge signals genuine uncertainty worth investigating."
        )
    else:  # final_check
        return (
            f"The contrarian lens through {', '.join(names[:3])} reveals potential blind spots. "
            f"Stress-test your assumptions against these challenges before committing."
        )


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="evaluate_decision",
            description=(
                "Evaluate a decision by surfacing relevant mental models from a library "
                "of 700 models across 10 disciplines. Provides structured coaching through "
                "three stages: initial_framing, deep_analysis, and final_check."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "situation": {
                        "type": "string",
                        "description": "Describe the deal, contract, or decision",
                    },
                    "decision_type": {
                        "type": "string",
                        "enum": [
                            "deal_evaluation",
                            "negotiation",
                            "resource_allocation",
                            "investment",
                            "hiring",
                            "strategy",
                            "general",
                        ],
                        "default": "general",
                        "description": "Type of decision being evaluated",
                    },
                    "data_context": {
                        "type": "string",
                        "description": "Additional data, numbers, terms, or context",
                    },
                    "constraints": {
                        "type": "string",
                        "description": "Timeline, budget, non-negotiables",
                    },
                    "stage": {
                        "type": "string",
                        "enum": ["initial_framing", "deep_analysis", "final_check"],
                        "default": "initial_framing",
                        "description": "Analysis stage",
                    },
                },
                "required": ["situation"],
            },
        ),
        Tool(
            name="search_models",
            description=(
                "Search the library of 700 mental models using semantic search. "
                "Returns matching models with name, discipline, summary, and relevance score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you're looking for",
                    },
                    "discipline": {
                        "type": "string",
                        "description": "Filter by discipline (optional)",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Number of results to return",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_model",
            description="Fetch a single mental model by name. Returns the full parsed structure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact or fuzzy model name",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="list_disciplines",
            description=(
                "List all disciplines with model counts and chapter breakdowns. "
                "Useful for orientation and discovering what's available."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "evaluate_decision":
        return await _evaluate_decision(**arguments)
    elif name == "search_models":
        return await _search_models(**arguments)
    elif name == "get_model":
        return await _get_model(**arguments)
    elif name == "list_disciplines":
        return await _list_disciplines()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _evaluate_decision(
    situation: str,
    decision_type: str = "general",
    data_context: str = "",
    constraints: str = "",
    stage: str = "initial_framing",
) -> list[TextContent]:
    search = _get_search()
    query = _build_query(situation, decision_type, data_context)

    # Retrieve more candidates, then narrow down
    top_k_retrieve = 20
    top_k_return = {"initial_framing": 8, "deep_analysis": 5, "final_check": 5}
    k = top_k_return.get(stage, 8)

    results = search.search(
        query=query,
        top_k=top_k_retrieve,
        decision_type=decision_type,
    )

    # For final_check, prefer models that challenge — pick from lower-ranked results too
    if stage == "final_check" and len(results) > k:
        # Mix top and middle results for contrarian perspective
        results = results[2 : 2 + k]
    else:
        results = results[:k]

    # Enrich each result
    models_applied = []
    for m in results:
        questions = _generate_questions(m, situation)
        models_applied.append({
            "name": m["name"],
            "discipline": m["discipline"],
            "relevance": f"Relevance score: {m.get('relevance_score', 0):.3f}",
            "insight": m.get("summary", ""),
            "questions": questions,
            "blind_spots": (m.get("limitations", "") or "")[:200],
        })

    synthesis = _generate_synthesis(results, stage)

    suggested_next = {
        "initial_framing": "Answer the sharpening questions, then call evaluate_decision with stage='deep_analysis'",
        "deep_analysis": "Call evaluate_decision with stage='final_check' for a pre-mortem stress test",
        "final_check": "You now have a comprehensive analysis. Make your decision with confidence.",
    }.get(stage, "")

    response = {
        "models_applied": models_applied,
        "synthesis": synthesis,
        "stage": stage,
        "suggested_next": suggested_next,
    }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def _search_models(
    query: str,
    discipline: str | None = None,
    top_k: int = 5,
) -> list[TextContent]:
    search = _get_search()
    top_k = min(max(top_k, 1), 20)

    results = search.search(
        query=query,
        top_k=top_k,
        discipline_filter=discipline,
    )

    output = []
    for m in results:
        output.append({
            "name": m["name"],
            "discipline": m["discipline"],
            "summary": m.get("summary", ""),
            "relevance_score": round(m.get("relevance_score", 0), 4),
        })

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _get_model(name: str) -> list[TextContent]:
    index = _get_index()
    model = index.get_by_name(name)

    if model is None:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Model '{name}' not found"}),
        )]

    return [TextContent(type="text", text=json.dumps(model, indent=2))]


async def _list_disciplines() -> list[TextContent]:
    index = _get_index()
    disciplines = index.get_disciplines()
    return [TextContent(type="text", text=json.dumps(disciplines, indent=2))]


@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="mental-models://index",
            name="Mental Models Index",
            description="All 700 mental model names grouped by discipline",
            mimeType="application/json",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    if str(uri) == "mental-models://index":
        index = _get_index()
        listing = index.get_index_listing()
        return json.dumps(listing, indent=2)
    raise ValueError(f"Unknown resource: {uri}")


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

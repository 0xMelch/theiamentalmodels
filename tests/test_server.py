"""Tests for MCP server tool handlers."""

import json

import pytest

from mental_models_mcp.models import ModelIndex, parse_raw_model
from mental_models_mcp.search import ModelSearch
from mental_models_mcp import server as server_module

SAMPLE_MODELS_RAW = [
    {
        "Discipline": "Game Theory",
        "Chapter": "General",
        "Name": "Nash Equilibrium",
        "Properties": "Concept",
        "Description": "Title\nNash Equilibrium\nModel\nA stable state where no player can improve their outcome.\nHeart of the Model\n• Best response equilibrium\nLimitations\n• Assumes rationality",
    },
    {
        "Discipline": "Behavioral Economics",
        "Chapter": "Biases",
        "Name": "Loss Aversion",
        "Properties": "Concept",
        "Description": "Title\nLoss Aversion\nModel\nLosses loom larger than equivalent gains.\nHeart of the Model\n• 2x loss sensitivity\nLimitations\n• Cultural variation",
    },
    {
        "Discipline": "Investing",
        "Chapter": "Value",
        "Name": "Margin of Safety",
        "Properties": "Principle",
        "Description": "Title\nMargin of Safety\nModel\nBuy at a discount to intrinsic value.\nHeart of the Model\n• Buffer against errors\nLimitations\n• Requires valuation skill",
    },
]


@pytest.fixture(autouse=True)
def setup_server():
    """Set up the server with test data."""
    parsed = [parse_raw_model(r) for r in SAMPLE_MODELS_RAW]
    server_module._index = ModelIndex(parsed)
    server_module._search = ModelSearch(parsed_models=parsed)
    yield
    server_module._index = None
    server_module._search = None


@pytest.mark.asyncio
async def test_list_tools():
    tools = await server_module.list_tools()
    names = [t.name for t in tools]
    assert "evaluate_decision" in names
    assert "search_models" in names
    assert "get_model" in names
    assert "list_disciplines" in names


@pytest.mark.asyncio
async def test_evaluate_decision_initial():
    result = await server_module.call_tool("evaluate_decision", {
        "situation": "Negotiating a business partnership",
        "decision_type": "negotiation",
        "stage": "initial_framing",
    })
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert data["stage"] == "initial_framing"
    assert "models_applied" in data
    assert "synthesis" in data
    assert "suggested_next" in data
    assert isinstance(data["models_applied"], list)


@pytest.mark.asyncio
async def test_evaluate_decision_deep():
    result = await server_module.call_tool("evaluate_decision", {
        "situation": "Should I invest in this startup?",
        "decision_type": "investment",
        "stage": "deep_analysis",
    })
    data = json.loads(result[0].text)
    assert data["stage"] == "deep_analysis"
    assert isinstance(data["models_applied"], list)


@pytest.mark.asyncio
async def test_evaluate_decision_final():
    result = await server_module.call_tool("evaluate_decision", {
        "situation": "Closing a major deal",
        "decision_type": "deal_evaluation",
        "stage": "final_check",
    })
    data = json.loads(result[0].text)
    assert data["stage"] == "final_check"


@pytest.mark.asyncio
async def test_search_models():
    result = await server_module.call_tool("search_models", {
        "query": "game theory equilibrium",
    })
    data = json.loads(result[0].text)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0]
    assert "relevance_score" in data[0]


@pytest.mark.asyncio
async def test_search_models_with_discipline():
    result = await server_module.call_tool("search_models", {
        "query": "strategy",
        "discipline": "Game Theory",
        "top_k": 3,
    })
    data = json.loads(result[0].text)
    for m in data:
        assert m["discipline"] == "Game Theory"


@pytest.mark.asyncio
async def test_get_model_exact():
    result = await server_module.call_tool("get_model", {
        "name": "Nash Equilibrium",
    })
    data = json.loads(result[0].text)
    assert data["name"] == "Nash Equilibrium"
    assert data["discipline"] == "Game Theory"


@pytest.mark.asyncio
async def test_get_model_fuzzy():
    result = await server_module.call_tool("get_model", {
        "name": "Nash",
    })
    data = json.loads(result[0].text)
    assert data["name"] == "Nash Equilibrium"


@pytest.mark.asyncio
async def test_get_model_not_found():
    result = await server_module.call_tool("get_model", {
        "name": "Nonexistent Model XYZ",
    })
    data = json.loads(result[0].text)
    assert "error" in data


@pytest.mark.asyncio
async def test_list_disciplines():
    result = await server_module.call_tool("list_disciplines", {})
    data = json.loads(result[0].text)
    assert isinstance(data, list)
    assert len(data) == 3  # Game Theory, Behavioral Economics, Investing
    disciplines = [d["discipline"] for d in data]
    assert "Game Theory" in disciplines


@pytest.mark.asyncio
async def test_list_resources():
    resources = await server_module.list_resources()
    assert len(resources) == 1
    assert str(resources[0].uri) == "mental-models://index"


@pytest.mark.asyncio
async def test_read_resource():
    content = await server_module.read_resource("mental-models://index")
    data = json.loads(content)
    assert "Game Theory" in data
    assert "Nash Equilibrium" in data["Game Theory"]


@pytest.mark.asyncio
async def test_unknown_tool():
    result = await server_module.call_tool("nonexistent_tool", {})
    assert "Unknown tool" in result[0].text

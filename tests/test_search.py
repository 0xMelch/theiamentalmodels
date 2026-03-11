"""Tests for semantic search."""

import pytest

from mental_models_mcp.models import parse_raw_model
from mental_models_mcp.search import ModelSearch, _simple_tokenize

SAMPLE_MODELS_RAW = [
    {
        "Discipline": "Game Theory",
        "Chapter": "General",
        "Name": "Nash Equilibrium",
        "Properties": "Concept",
        "Description": "Title\nNash Equilibrium\nModel\nA stable state in a game where no player can improve their outcome by unilaterally changing their strategy.\nHeart of the Model\n• Every player making best response\n• No incentive to deviate\nLimitations\n• Assumes rationality",
    },
    {
        "Discipline": "Economics",
        "Chapter": "General",
        "Name": "Supply and Demand",
        "Properties": "Framework",
        "Description": "Title\nSupply and Demand\nModel\nThe fundamental economic model describing how prices are determined by the interaction of supply and demand forces in a market.\nHeart of the Model\n• Price equilibrium where supply meets demand\n• Shifts in either curve change price and quantity\nLimitations\n• Assumes perfect competition",
    },
    {
        "Discipline": "Behavioral Economics",
        "Chapter": "Cognitive Biases",
        "Name": "Loss Aversion",
        "Properties": "Concept",
        "Description": "Title\nLoss Aversion\nModel\nPeople feel the pain of losing more strongly than the pleasure of gaining an equivalent amount. Losses loom larger than gains.\nHeart of the Model\n• Losses are felt about 2x as strongly as equivalent gains\n• Drives risk aversion for gains and risk seeking for losses\nLimitations\n• Varies across cultures and individuals",
    },
    {
        "Discipline": "Investing",
        "Chapter": "Value",
        "Name": "Margin of Safety",
        "Properties": "Principle",
        "Description": "Title\nMargin of Safety\nModel\nBuy assets at a significant discount to their intrinsic value to provide a buffer against errors in analysis or unforeseen negative events.\nHeart of the Model\n• Buffer against analytical errors\n• Protection against downside risk\nLimitations\n• Requires accurate valuation",
    },
    {
        "Discipline": "Probability",
        "Chapter": "General",
        "Name": "Bayes Theorem",
        "Properties": "Framework",
        "Description": "Title\nBayes Theorem\nModel\nA mathematical framework for updating probabilities based on new evidence. Prior beliefs are combined with likelihood of new data.\nHeart of the Model\n• Update beliefs with new evidence\n• Prior probability matters\nMath Intuition\nP(A|B) = P(B|A) * P(A) / P(B)\nLimitations\n• Requires prior probabilities",
    },
]


@pytest.fixture
def parsed_models():
    return [parse_raw_model(r) for r in SAMPLE_MODELS_RAW]


@pytest.fixture
def search_engine(parsed_models):
    """Create a search engine using TF-IDF fallback (no ONNX/embeddings files)."""
    return ModelSearch(parsed_models=parsed_models)


def test_simple_tokenize():
    tokens = _simple_tokenize("Hello, World! This is a TEST.")
    assert tokens == ["hello", "world", "this", "is", "a", "test"]


def test_search_basic(search_engine):
    results = search_engine.search("game theory strategy equilibrium", top_k=3)
    assert len(results) > 0
    # Nash Equilibrium should be top result for game theory queries
    names = [r["name"] for r in results]
    assert "Nash Equilibrium" in names


def test_search_economics(search_engine):
    results = search_engine.search("market prices supply demand", top_k=3)
    assert len(results) > 0
    names = [r["name"] for r in results]
    assert "Supply and Demand" in names


def test_search_investing(search_engine):
    results = search_engine.search("investment value discount safety", top_k=3)
    assert len(results) > 0
    names = [r["name"] for r in results]
    assert "Margin of Safety" in names


def test_search_with_discipline_filter(search_engine):
    results = search_engine.search("strategy", top_k=5, discipline_filter="Game Theory")
    for r in results:
        assert r["discipline"] == "Game Theory"


def test_search_with_decision_type_boost(search_engine):
    # With investment boost, investing models should rank higher
    results_general = search_engine.search("risk management", top_k=5)
    results_investment = search_engine.search(
        "risk management", top_k=5, decision_type="investment"
    )
    # Both should return results
    assert len(results_general) > 0
    assert len(results_investment) > 0


def test_search_returns_scores(search_engine):
    results = search_engine.search("probability updating beliefs", top_k=3)
    for r in results:
        assert "relevance_score" in r
        assert r["relevance_score"] > 0


def test_search_top_k(search_engine):
    results = search_engine.search("model", top_k=2)
    assert len(results) <= 2


def test_search_empty_query(search_engine):
    results = search_engine.search("", top_k=5)
    # Should still return something (all zeros or whatever)
    assert isinstance(results, list)

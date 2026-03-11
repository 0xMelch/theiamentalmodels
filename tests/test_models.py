"""Tests for model parsing and indexing."""

import json
import tempfile
from pathlib import Path

import pytest

from mental_models_mcp.models import (
    ModelIndex,
    _slugify,
    parse_all_models,
    parse_description,
    parse_raw_model,
)

SAMPLE_DESCRIPTION = """Title
Nash Equilibrium
Model
A Nash Equilibrium is a stable state in a game where no player can improve their outcome by unilaterally changing their strategy. This is a fundamental concept in game theory.
Heart of the Model
• Every player is making the best response to what others are doing
• No player has incentive to deviate unilaterally
• Multiple equilibria can exist in the same game
Limitations
• Assumes perfect rationality
• Does not account for learning or adaptation over time"""

SAMPLE_RAW = {
    "Discipline": "Game Theory",
    "Chapter": "General",
    "Name": "Nash Equilibrium",
    "Properties": "Concept",
    "Description": SAMPLE_DESCRIPTION,
}


def test_slugify():
    assert _slugify("Game Theory", "Nash Equilibrium") == "game-theory-nash-equilibrium"
    assert _slugify("Economics", "Supply & Demand") == "economics-supply-demand"


def test_parse_description():
    sections = parse_description(SAMPLE_DESCRIPTION)
    assert "Title" in sections
    assert "Model" in sections
    assert "Heart of the Model" in sections
    assert "Limitations" in sections
    assert "Nash Equilibrium" in sections["Title"]
    assert "stable state" in sections["Model"]
    assert "best response" in sections["Heart of the Model"]
    assert "perfect rationality" in sections["Limitations"]


def test_parse_raw_model():
    parsed = parse_raw_model(SAMPLE_RAW)
    assert parsed["name"] == "Nash Equilibrium"
    assert parsed["discipline"] == "Game Theory"
    assert parsed["chapter"] == "General"
    assert parsed["id"] == "game-theory-nash-equilibrium"
    assert parsed["summary"] is not None
    assert parsed["heart"] is not None
    assert "best response" in parsed["heart"]
    assert parsed["limitations"] is not None
    assert parsed["math"] is None  # Not in sample


def test_parse_description_with_math():
    desc = """Title
Bayes Theorem
Model
Bayes theorem describes how to update probabilities.
Math Intuition
P(A|B) = P(B|A) * P(A) / P(B)
Limitations
• Requires prior probabilities"""
    sections = parse_description(desc)
    assert "Math Intuition" in sections
    assert "P(A|B)" in sections["Math Intuition"]


def test_model_index():
    models = [
        parse_raw_model(SAMPLE_RAW),
        parse_raw_model({
            **SAMPLE_RAW,
            "Name": "Prisoner's Dilemma",
            "Description": "Title\nPrisoner's Dilemma\nModel\nA game theory scenario.\nLimitations\n• Simplified",
        }),
    ]
    index = ModelIndex(models)

    assert len(index.all_models) == 2

    # get_by_name exact
    m = index.get_by_name("Nash Equilibrium")
    assert m is not None
    assert m["name"] == "Nash Equilibrium"

    # get_by_name case insensitive
    m = index.get_by_name("nash equilibrium")
    assert m is not None

    # get_by_name fuzzy
    m = index.get_by_name("Nash")
    assert m is not None

    # get_disciplines
    discs = index.get_disciplines()
    assert len(discs) == 1
    assert discs[0]["discipline"] == "Game Theory"
    assert discs[0]["model_count"] == 2

    # get_index_listing
    listing = index.get_index_listing()
    assert "Game Theory" in listing
    assert len(listing["Game Theory"]) == 2


def test_parse_all_models_from_file():
    """Test parsing from a temporary file."""
    raw_data = [SAMPLE_RAW]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(raw_data, f)
        tmp_path = Path(f.name)

    try:
        models = parse_all_models(tmp_path)
        assert len(models) == 1
        assert models[0]["name"] == "Nash Equilibrium"
    finally:
        tmp_path.unlink()

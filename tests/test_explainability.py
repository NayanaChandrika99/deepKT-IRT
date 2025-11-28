# ABOUTME: Tests explanation generation from attention patterns.
# ABOUTME: Ensures insights and formatting include key details.

from src.common.explainability import (
    analyze_attention_pattern,
    generate_explanation,
    format_explanation,
    MasteryExplanation,
)
import pandas as pd


def test_analyze_attention_pattern_low_mastery_recent_failures():
    factors = [
        {"correct": False, "position": 4, "weight": 0.35},
        {"correct": False, "position": 3, "weight": 0.25},
        {"correct": True, "position": 1, "weight": 0.1},
    ]
    insight, rec = analyze_attention_pattern(factors, mastery_score=0.3)
    assert "struggle" in insight.lower() or "recent" in insight.lower()
    assert rec


def test_analyze_attention_pattern_high_mastery_success():
    factors = [
        {"correct": True, "position": 4, "weight": 0.3},
        {"correct": True, "position": 3, "weight": 0.25},
        {"correct": True, "position": 2, "weight": 0.2},
        {"correct": True, "position": 1, "weight": 0.1},
    ]
    insight, rec = analyze_attention_pattern(factors, mastery_score=0.85)
    assert "strong" in insight.lower() or "confident" in insight.lower()
    assert rec


def test_generate_and_format_explanation_handles_missing_attention():
    attention_df = pd.DataFrame({"user_id": [], "top_influences": []})
    events_df = pd.DataFrame({"user_id": [], "item_id": [], "skill_ids": []})

    explanation = generate_explanation(
        user_id="u1",
        skill_id="s1",
        mastery_score=0.5,
        attention_data=attention_df,
        events_df=events_df,
        interaction_count=0,
    )
    output = format_explanation(explanation)
    assert "u1" in output
    assert "s1" in output
    assert "No attention" in output or "insufficient" in output.lower()

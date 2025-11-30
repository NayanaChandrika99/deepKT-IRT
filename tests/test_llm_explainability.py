# ABOUTME: Tests for LLM-based explanation generation with mocked API calls.
# ABOUTME: Verifies fallback behavior, error handling, and prompt construction.

from unittest.mock import Mock, patch, AsyncMock
import pytest

from src.common.llm_explainability import (
    AttentionContext,
    build_attention_context,
    format_prompt_for_llm,
    generate_llm_insight_recommendation_sync,
    generate_llm_rl_reason_sync,
    generate_llm_rule_based_reason_sync,
)
from src.common.bandit import StudentContext, ItemArm


def test_build_attention_context():
    """Test building attention context from attention data."""
    key_factors = [
        {"item_id": "Q1", "correct": True, "weight": 0.3, "skill": "7.RP.A.1"},
        {"item_id": "Q2", "correct": False, "weight": 0.4, "skill": "7.RP.A.1"},
        {"item_id": "Q3", "correct": True, "weight": 0.2, "skill": "7.RP.A.1"},
    ]

    context = build_attention_context(
        user_id="student1",
        skill_id="7.RP.A.1",
        mastery_score=0.65,
        key_factors=key_factors,
        interaction_count=25,
    )

    assert context.user_id == "student1"
    assert context.skill_id == "7.RP.A.1"
    assert context.mastery_score == 0.65
    assert context.confidence == "high"  # 25 interactions
    assert context.correct_count == 2
    assert context.incorrect_count == 1
    assert context.total_attention_on_incorrect == 0.4


def test_build_attention_context_low_confidence():
    """Test confidence levels based on interaction count."""
    context_low = build_attention_context(
        "u1", "s1", 0.5, [], interaction_count=3
    )
    assert context_low.confidence == "low"

    context_med = build_attention_context(
        "u1", "s1", 0.5, [], interaction_count=15
    )
    assert context_med.confidence == "medium"

    context_high = build_attention_context(
        "u1", "s1", 0.5, [], interaction_count=25
    )
    assert context_high.confidence == "high"


def test_format_prompt_for_llm():
    """Test that prompt includes key context fields."""
    context = AttentionContext(
        user_id="student1",
        skill_id="7.RP.A.1",
        skill_name="Ratios & Proportional Relationships",
        mastery_score=0.45,
        confidence="medium",
        interaction_count=12,
        key_interactions=[
            {"item_id": "Q1", "correct": False, "weight": 0.35, "skill": "7.RP.A.1", "position": 5}
        ],
        correct_count=0,
        incorrect_count=1,
        total_attention_on_incorrect=0.35,
    )

    prompt = format_prompt_for_llm(context)

    assert "student1" in prompt
    assert "7.RP.A.1" in prompt
    assert "Ratios & Proportional Relationships" in prompt
    assert "45%" in prompt or "0.45" in prompt
    assert "Q1" in prompt
    assert "incorrect" in prompt.lower()


@patch("src.common.llm_explainability.openai.AsyncOpenAI")
def test_generate_llm_insight_recommendation_sync_openai(mock_openai_class):
    """Test LLM explanation generation with mocked OpenAI API."""
    # Setup mock
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"insight": "Test insight", "recommendation": "Test recommendation"}'

    mock_completion = AsyncMock(return_value=mock_response)
    mock_client.chat.completions.create = mock_completion

    # Create context
    context = build_attention_context(
        "u1", "s1", 0.5, [], interaction_count=10
    )

    # Call function
    insight, recommendation = generate_llm_insight_recommendation_sync(
        context, api_key="test_key", provider="openai"
    )

    assert insight == "Test insight"
    assert recommendation == "Test recommendation"
    assert mock_completion.called


@patch("src.common.llm_explainability.anthropic.AsyncAnthropic")
def test_generate_llm_insight_recommendation_sync_anthropic(mock_anthropic_class):
    """Test LLM explanation generation with mocked Anthropic API."""
    # Setup mock
    mock_client = Mock()
    mock_anthropic_class.return_value = mock_client

    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = '{"insight": "Claude insight", "recommendation": "Claude recommendation"}'

    mock_completion = AsyncMock(return_value=mock_response)
    mock_client.messages.create = mock_completion

    # Create context
    context = build_attention_context(
        "u1", "s1", 0.5, [], interaction_count=10
    )

    # Call function
    insight, recommendation = generate_llm_insight_recommendation_sync(
        context, api_key="test_key", provider="anthropic"
    )

    assert insight == "Claude insight"
    assert recommendation == "Claude recommendation"
    assert mock_completion.called


def test_generate_llm_insight_fallback_on_error():
    """Test that LLM generation falls back to template on error."""
    context = build_attention_context(
        "u1", "s1", 0.3,
        [{"item_id": "Q1", "correct": False, "weight": 0.5}],
        interaction_count=10
    )

    # Should fall back to template-based without API key
    insight, recommendation = generate_llm_insight_recommendation_sync(
        context, api_key=None, provider="openai"
    )

    # Should get something back (fallback)
    assert isinstance(insight, str)
    assert isinstance(recommendation, str)
    assert len(insight) > 0
    assert len(recommendation) > 0


@patch("src.common.llm_explainability.openai.AsyncOpenAI")
def test_generate_llm_rl_reason_sync(mock_openai_class):
    """Test RL reason generation with mocked OpenAI API."""
    # Setup mock
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"reason": "Test RL reason"}'

    mock_completion = AsyncMock(return_value=mock_response)
    mock_client.chat.completions.create = mock_completion

    student = StudentContext(
        user_id="s1",
        mastery=0.6,
        recent_accuracy=0.7,
        recent_speed=0.5,
        help_tendency=0.1,
        skill_gap=0.2,
    )
    item = ItemArm("Q1", "7.RP.A.1", difficulty=0.65)

    reason = generate_llm_rl_reason_sync(
        student, item, expected=0.7, uncertainty=0.15, is_exploration=False,
        api_key="test_key"
    )

    assert reason == "Test RL reason"
    assert mock_completion.called


@patch("src.common.llm_explainability.openai.AsyncOpenAI")
def test_generate_llm_rule_based_reason_sync(mock_openai_class):
    """Test rule-based reason generation with mocked OpenAI API."""
    # Setup mock
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"reason": "Test rule reason"}'

    mock_completion = AsyncMock(return_value=mock_response)
    mock_client.chat.completions.create = mock_completion

    reason = generate_llm_rule_based_reason_sync(
        user_id="s1",
        target_skill="7.RP.A.1",
        mastery_mean=0.65,
        item_id="Q1",
        item_difficulty=0.7,
        api_key="test_key"
    )

    assert reason == "Test rule reason"
    assert mock_completion.called


def test_llm_rl_reason_fallback():
    """Test RL reason falls back to template on error."""
    student = StudentContext(
        user_id="s1",
        mastery=0.6,
        recent_accuracy=0.7,
        recent_speed=0.5,
        help_tendency=0.1,
        skill_gap=0.2,
    )
    item = ItemArm("Q1", "7.RP.A.1", difficulty=0.65)

    # Should fall back without API key
    reason = generate_llm_rl_reason_sync(
        student, item, expected=0.7, uncertainty=0.15, is_exploration=False,
        api_key=None
    )

    assert isinstance(reason, str)
    assert len(reason) > 0

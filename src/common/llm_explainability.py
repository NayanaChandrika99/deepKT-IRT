# ABOUTME: Generates natural language explanations using LLM from attention data.
# ABOUTME: Formats attention patterns into prompts and parses LLM responses.

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

from .explainability import MasteryExplanation

if TYPE_CHECKING:
    from .bandit import StudentContext, ItemArm


def sanitize_for_prompt(text: str, max_length: int = 100) -> str:
    """
    Sanitize user-provided text for safe inclusion in LLM prompts.

    Removes newlines, non-printable characters, and truncates to prevent
    prompt injection attacks.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length (default 100)

    Returns:
        Sanitized text safe for prompt inclusion
    """
    if not isinstance(text, str):
        text = str(text)

    # Replace newlines with spaces to prevent prompt structure manipulation
    text = text.replace("\n", " ").replace("\r", " ")

    # Remove control characters except space
    text = "".join(char for char in text if char.isprintable() or char == " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Truncate to max length
    text = text[:max_length].strip()

    return text


@dataclass
class AttentionContext:
    """Structured context for LLM explanation generation."""
    
    user_id: str
    skill_id: str
    skill_name: str  # Human-readable skill name (e.g., "Ratios & Proportional Relationships")
    mastery_score: float
    confidence: str  # low/medium/high
    interaction_count: int
    
    # Top influential interactions from attention
    key_interactions: List[Dict]  # [{item_id, correct, weight, skill, position}]
    
    # Summary stats
    correct_count: int
    incorrect_count: int
    total_attention_on_incorrect: float  # Sum of weights on incorrect items


def build_attention_context(
    user_id: str,
    skill_id: str,
    mastery_score: float,
    key_factors: List[Dict],
    interaction_count: int,
    skill_name_map: Optional[Dict[str, str]] = None,
) -> AttentionContext:
    """Build structured context from attention data for LLM."""
    
    correct_count = sum(1 for f in key_factors if f.get("correct"))
    incorrect_count = len(key_factors) - correct_count
    total_incorrect_weight = sum(
        f.get("weight", 0) for f in key_factors if not f.get("correct")
    )
    
    # Map skill code to human-readable name
    skill_name = skill_id
    if skill_name_map and skill_id in skill_name_map:
        skill_name = skill_name_map[skill_id]
    
    # Determine confidence
    if interaction_count < 5:
        confidence = "low"
    elif interaction_count < 20:
        confidence = "medium"
    else:
        confidence = "high"
    
    return AttentionContext(
        user_id=user_id,
        skill_id=skill_id,
        skill_name=skill_name,
        mastery_score=mastery_score,
        confidence=confidence,
        interaction_count=interaction_count,
        key_interactions=key_factors,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        total_attention_on_incorrect=total_incorrect_weight,
    )


def format_prompt_for_llm(context: AttentionContext) -> str:
    """
    Format attention context into a prompt for LLM explanation generation.

    The prompt includes:
    - Student's mastery score and confidence
    - Top influential interactions (item ID, correct/incorrect, attention weight)
    - Instructions for generating explanation

    All user-provided fields are sanitized to prevent prompt injection.
    """

    # Sanitize user-provided fields
    safe_user_id = sanitize_for_prompt(context.user_id, max_length=50)
    safe_skill_id = sanitize_for_prompt(context.skill_id, max_length=50)
    safe_skill_name = sanitize_for_prompt(context.skill_name, max_length=100)

    interactions_text = []
    for i, factor in enumerate(context.key_interactions, 1):
        status = "✓ correct" if factor.get("correct") else "✗ incorrect"
        weight = factor.get("weight", 0) * 100
        safe_skill = sanitize_for_prompt(str(factor.get("skill", "unknown")), max_length=50)
        safe_item_id = sanitize_for_prompt(str(factor.get("item_id", "unknown")), max_length=20)
        interactions_text.append(
            f"  {i}. Item {safe_item_id} ({status}, {weight:.0f}% attention weight, skill: {safe_skill})"
        )

    prompt = f"""You are an educational analytics assistant explaining a student's mastery prediction.

## Student Context
- Student ID: {safe_user_id}
- Skill: {safe_skill_name} ({safe_skill_id})
- Predicted Mastery: {context.mastery_score:.0%}
- Confidence: {context.confidence} ({context.interaction_count} interactions)

## Attention Analysis
The AI model focused on these past interactions when making this prediction:
{chr(10).join(interactions_text)}

Summary:
- {context.correct_count} correct interactions received attention
- {context.incorrect_count} incorrect interactions received attention
- {context.total_attention_on_incorrect:.0%} of attention was on incorrect answers

## Your Task
Generate a 2-3 sentence explanation for the student that:
1. Explains WHY their mastery is at this level (based on which interactions the model focused on)
2. Provides a SPECIFIC, ACTIONABLE recommendation (mention actual item IDs or skills to review)

Be encouraging but honest. Use simple language suitable for a student.

Respond with JSON in this format:
{{"insight": "...", "recommendation": "..."}}
"""
    return prompt


async def generate_llm_explanation(
    context: AttentionContext,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> MasteryExplanation:
    """
    Generate explanation using OpenAI API.
    
    Args:
        context: Structured attention context
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model: Model to use (default: gpt-4o-mini for cost efficiency)
    
    Returns:
        MasteryExplanation with LLM-generated insight and recommendation
    """
    try:
        import openai
    except ImportError:
        # Fallback to template-based if openai not installed
        from .explainability import analyze_attention_pattern
        insight, recommendation = analyze_attention_pattern(
            context.key_interactions, context.mastery_score
        )
        return MasteryExplanation(
            user_id=context.user_id,
            skill_id=context.skill_id,
            mastery_score=context.mastery_score,
            key_factors=context.key_interactions,
            insight=insight,
            recommendation=recommendation,
            confidence=context.confidence,
        )
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
    
    client = openai.AsyncOpenAI(api_key=api_key)
    prompt = format_prompt_for_llm(context)
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    
    result = json.loads(response.choices[0].message.content)
    
    return MasteryExplanation(
        user_id=context.user_id,
        skill_id=context.skill_id,
        mastery_score=context.mastery_score,
        key_factors=context.key_interactions,
        insight=result.get("insight", "Unable to generate insight."),
        recommendation=result.get("recommendation", "Continue practicing."),
        confidence=context.confidence,
    )


def generate_llm_explanation_sync(
    context: AttentionContext,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> MasteryExplanation:
    """Synchronous wrapper for generate_llm_explanation."""
    import asyncio
    return asyncio.run(generate_llm_explanation(context, api_key, model))


# Alternative: Anthropic Claude
async def generate_claude_explanation(
    context: AttentionContext,
    api_key: Optional[str] = None,
    model: str = "claude-3-haiku-20240307",
) -> MasteryExplanation:
    """
    Generate explanation using Anthropic Claude API.
    Uses claude-3-haiku for cost efficiency.
    """
    try:
        import anthropic
    except ImportError:
        from .explainability import analyze_attention_pattern
        insight, recommendation = analyze_attention_pattern(
            context.key_interactions, context.mastery_score
        )
        return MasteryExplanation(
            user_id=context.user_id,
            skill_id=context.skill_id,
            mastery_score=context.mastery_score,
            key_factors=context.key_interactions,
            insight=insight,
            recommendation=recommendation,
            confidence=context.confidence,
        )
    
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key.")
    
    client = anthropic.AsyncAnthropic(api_key=api_key)
    prompt = format_prompt_for_llm(context)
    
    response = await client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    
    # Parse JSON from response
    content = response.content[0].text
    # Extract JSON if wrapped in markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    result = json.loads(content.strip())
    
    return MasteryExplanation(
        user_id=context.user_id,
        skill_id=context.skill_id,
        mastery_score=context.mastery_score,
        key_factors=context.key_interactions,
        insight=result.get("insight", "Unable to generate insight."),
        recommendation=result.get("recommendation", "Continue practicing."),
        confidence=context.confidence,
    )


# LLM configuration
LLM_ENABLED = os.environ.get("USE_LLM_EXPLANATIONS", "false").lower() == "true"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")  # "openai" or "anthropic"
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")  # or "claude-3-haiku-20240307"


async def generate_llm_insight_recommendation(
    context: AttentionContext,
    api_key: Optional[str] = None,
    provider: str = None,
    model: str = None,
) -> Tuple[str, str]:
    """
    Generate insight and recommendation using LLM.
    Returns (insight, recommendation) tuple.

    All user inputs are sanitized to prevent prompt injection.
    """
    provider = provider or LLM_PROVIDER
    model = model or LLM_MODEL

    # Sanitize user inputs
    safe_user_id = sanitize_for_prompt(context.user_id, max_length=50)
    safe_skill_name = sanitize_for_prompt(context.skill_name, max_length=100)
    safe_skill_id = sanitize_for_prompt(context.skill_id, max_length=50)

    prompt = f"""You are an educational analytics assistant explaining a student's mastery prediction.

## Student Context
- Student ID: {safe_user_id}
- Skill: {safe_skill_name} ({safe_skill_id})
- Predicted Mastery: {context.mastery_score:.0%}
- Confidence: {context.confidence} ({context.interaction_count} interactions)

## Attention Analysis
The AI model focused on these past interactions when making this prediction:
{chr(10).join([f"  {i+1}. Item {f.get('item_id', 'unknown')[:12]} ({'✓ correct' if f.get('correct') else '✗ incorrect'}, {f.get('weight', 0)*100:.0f}% attention weight, skill: {f.get('skill', 'unknown')})" for i, f in enumerate(context.key_interactions)])}

Summary:
- {context.correct_count} correct interactions received attention
- {context.incorrect_count} incorrect interactions received attention
- {context.total_attention_on_incorrect:.0%} of attention was on incorrect answers

## Your Task
Generate a 2-3 sentence explanation that:
1. Explains WHY their mastery is at this level (based on which interactions the model focused on)
2. Provides a SPECIFIC, ACTIONABLE recommendation (mention actual item IDs or skills to review)

Be encouraging but honest. Use simple language suitable for a student.

Respond with JSON: {{"insight": "...", "recommendation": "..."}}
"""
    
    try:
        if provider == "anthropic":
            import anthropic
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model=model or "claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
        else:  # OpenAI
            import openai
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            client = openai.AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
        
        return (
            result.get("insight", "Unable to generate insight."),
            result.get("recommendation", "Continue practicing."),
        )
    except Exception as e:
        # Fallback to template-based
        from .explainability import analyze_attention_pattern
        return analyze_attention_pattern(context.key_interactions, context.mastery_score)


def generate_llm_insight_recommendation_sync(
    context: AttentionContext,
    api_key: Optional[str] = None,
    provider: str = None,
    model: str = None,
) -> Tuple[str, str]:
    """Synchronous wrapper."""
    import asyncio
    return asyncio.run(generate_llm_insight_recommendation(context, api_key, provider, model))


async def generate_llm_rl_reason(
    student: "StudentContext",
    item: "ItemArm",
    expected: float,
    uncertainty: float,
    is_exploration: bool,
    api_key: Optional[str] = None,
    provider: str = None,
    model: str = None,
) -> str:
    """
    Generate natural language reason for RL recommendation using LLM.

    All user inputs are sanitized to prevent prompt injection.

    Args:
        student: Student context (mastery, recent_accuracy, etc.)
        item: Item being recommended
        expected: Expected reward (0-1)
        uncertainty: Uncertainty/confidence width
        is_exploration: Whether this is exploration vs exploitation
        api_key: API key (defaults to env var)
        provider: "openai" or "anthropic"
        model: Model name

    Returns:
        Natural language reason for recommendation
    """
    # Import here to avoid circular dependency
    from .bandit import StudentContext, ItemArm

    provider = provider or LLM_PROVIDER
    model = model or LLM_MODEL

    # Sanitize user inputs
    safe_item_id = sanitize_for_prompt(item.item_id, max_length=50)
    safe_skill = sanitize_for_prompt(item.skill, max_length=50)

    prompt = f"""You are an educational recommendation system explaining why an item is recommended.

## Student Profile
- Mastery Level: {student.mastery:.0%}
- Recent Accuracy: {student.recent_accuracy:.0%}
- Recent Speed: {student.recent_speed:.1f} (normalized)
- Help Tendency: {student.help_tendency:.0%}
- Skill Gap: {student.skill_gap:.2f}

## Item Details
- Item ID: {safe_item_id}
- Skill: {safe_skill}
- Difficulty: {item.difficulty:.2f}
- Discrimination: {item.discrimination:.2f}

## Recommendation Metrics
- Expected Success Rate: {expected:.0%}
- Uncertainty: {uncertainty:.2f}
- Strategy: {"Exploration (learning preferences)" if is_exploration else "Exploitation (high confidence)"}
- Difficulty Match: {"Good match" if abs(student.mastery - item.difficulty) < 0.15 else "Challenging" if item.difficulty > student.mastery else "Confidence-building"}

## Your Task
Generate a 1-2 sentence natural language explanation for why this item is recommended.
Be specific about the difficulty match and strategy (exploration vs exploitation).
Use simple, encouraging language.

Respond with JSON: {{"reason": "..."}}
"""
    
    try:
        if provider == "anthropic":
            import anthropic
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model=model or "claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
        else:  # OpenAI
            import openai
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            client = openai.AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
        
        return result.get("reason", "Recommended based on your learning profile.")
    except Exception as e:
        # Fallback to template-based
        from .bandit import generate_rl_reason as template_reason
        return template_reason(student, item, expected, uncertainty, is_exploration)


def generate_llm_rl_reason_sync(
    student: "StudentContext",
    item: "ItemArm",
    expected: float,
    uncertainty: float,
    is_exploration: bool,
    api_key: Optional[str] = None,
    provider: str = None,
    model: str = None,
) -> str:
    """Synchronous wrapper."""
    import asyncio
    return asyncio.run(generate_llm_rl_reason(student, item, expected, uncertainty, is_exploration, api_key, provider, model))


async def generate_llm_rule_based_reason(
    user_id: str,
    target_skill: str,
    mastery_mean: float,
    item_id: str,
    item_difficulty: float,
    api_key: Optional[str] = None,
    provider: str = None,
    model: str = None,
) -> str:
    """
    Generate natural language reason for rule-based recommendation using LLM.

    All user inputs are sanitized to prevent prompt injection.

    Args:
        user_id: Student identifier
        target_skill: Skill being targeted
        mastery_mean: Student's mastery for this skill
        item_id: Item being recommended
        item_difficulty: Item difficulty level
        api_key: API key (defaults to env var)
        provider: "openai" or "anthropic"
        model: Model name

    Returns:
        Natural language reason for recommendation
    """
    provider = provider or LLM_PROVIDER
    model = model or LLM_MODEL

    # Sanitize user inputs
    safe_user_id = sanitize_for_prompt(user_id, max_length=50)
    safe_target_skill = sanitize_for_prompt(target_skill, max_length=50)
    safe_item_id = sanitize_for_prompt(item_id, max_length=50)

    prompt = f"""You are an educational recommendation system explaining why an item is recommended.

## Student Profile
- Student ID: {safe_user_id}
- Skill: {safe_target_skill}
- Mastery Level: {mastery_mean:.0%}

## Item Details
- Item ID: {safe_item_id}
- Difficulty: {item_difficulty:.2f}

## Recommendation Logic
This item is recommended because its difficulty ({item_difficulty:.2f}) is well-matched to the student's mastery level ({mastery_mean:.0%}).

## Your Task
Generate a 1 sentence natural language explanation for why this item is recommended.
Be specific about the difficulty match. Use simple, encouraging language.

Respond with JSON: {{"reason": "..."}}
"""
    
    try:
        if provider == "anthropic":
            import anthropic
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model=model or "claude-3-haiku-20240307",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
        else:  # OpenAI
            import openai
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            client = openai.AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
        
        return result.get("reason", f"Recommended for skill {target_skill} based on mastery level.")
    except Exception as e:
        # Fallback to template-based
        return f"Skill {target_skill} mastery {mastery_mean:.2f}, item difficulty {item_difficulty:.2f}"


def generate_llm_rule_based_reason_sync(
    user_id: str,
    target_skill: str,
    mastery_mean: float,
    item_id: str,
    item_difficulty: float,
    api_key: Optional[str] = None,
    provider: str = None,
    model: str = None,
) -> str:
    """Synchronous wrapper."""
    import asyncio
    return asyncio.run(generate_llm_rule_based_reason(user_id, target_skill, mastery_mean, item_id, item_difficulty, api_key, provider, model))


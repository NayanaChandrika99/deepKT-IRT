# ABOUTME: Azure OpenAI client wrapper for Databricks environment.
# ABOUTME: Provides LLM explainability functions compatible with Azure-managed APIs.

"""
Azure OpenAI Explainability Module

Replaces the original `llm_explainability.py` with Azure-specific implementation.
Uses Azure OpenAI Service for enterprise-compliant LLM access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import json


@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI Service."""
    endpoint: str
    api_key: str
    api_version: str = "2024-02-15-preview"
    chat_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-ada-002"
    

def get_azure_openai_config() -> AzureOpenAIConfig:
    """
    Get Azure OpenAI configuration.
    
    In Databricks, use secrets. Locally, use environment variables.
    """
    # Try Databricks secrets first
    try:
        from databricks.sdk.runtime import dbutils
        return AzureOpenAIConfig(
            endpoint=dbutils.secrets.get(scope="deepkt-secrets", key="azure-openai-endpoint"),
            api_key=dbutils.secrets.get(scope="deepkt-secrets", key="azure-openai-key"),
        )
    except (ImportError, Exception):
        # Fall back to environment variables
        return AzureOpenAIConfig(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_KEY", ""),
            chat_model=os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o"),
            embedding_model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        )


def get_azure_openai_client():
    """Create Azure OpenAI client."""
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai>=1.0")
    
    config = get_azure_openai_config()
    
    return AzureOpenAI(
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
        api_version=config.api_version,
    )


@dataclass
class LLMExplanation:
    """Structured LLM explanation output."""
    text: str
    tokens_used: int
    model: str
    cost_estimate_usd: float


class AzureOpenAIExplainer:
    """
    Explainability module using Azure OpenAI Service.
    
    Provides enterprise-compliant LLM explanations for:
    - Mastery insights
    - Recommendation reasoning
    - Gaming behavior alerts
    """
    
    def __init__(self, config: Optional[AzureOpenAIConfig] = None):
        self.config = config or get_azure_openai_config()
        self.client = get_azure_openai_client()
    
    def explain_mastery(
        self,
        user_id: str,
        skill_id: str,
        mastery_score: float,
        key_factors: List[Dict],
        interaction_count: int,
    ) -> LLMExplanation:
        """
        Generate natural language mastery explanation.
        
        Replaces template-based explanations from `explainability.py`.
        """
        prompt = self._build_mastery_prompt(
            user_id, skill_id, mastery_score, key_factors, interaction_count
        )
        
        response = self.client.chat.completions.create(
            model=self.config.chat_model,
            messages=[
                {"role": "system", "content": self._mastery_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        
        return LLMExplanation(
            text=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=self.config.chat_model,
            cost_estimate_usd=self._estimate_cost(response.usage.total_tokens),
        )
    
    def explain_recommendation(
        self,
        user_id: str,
        item_id: str,
        skill: str,
        expected_reward: float,
        uncertainty: float,
        is_exploration: bool,
        student_mastery: float,
        item_difficulty: float,
    ) -> LLMExplanation:
        """
        Generate explanation for bandit recommendation.
        
        Replaces `generate_rl_reason_with_llm` from `bandit.py`.
        """
        prompt = self._build_recommendation_prompt(
            user_id, item_id, skill, expected_reward, uncertainty,
            is_exploration, student_mastery, item_difficulty
        )
        
        response = self.client.chat.completions.create(
            model=self.config.chat_model,
            messages=[
                {"role": "system", "content": self._recommendation_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        
        return LLMExplanation(
            text=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=self.config.chat_model,
            cost_estimate_usd=self._estimate_cost(response.usage.total_tokens),
        )
    
    def explain_gaming_alert(
        self,
        user_id: str,
        rapid_guess_rate: float,
        help_abuse_rate: float,
        severity: str,
    ) -> LLMExplanation:
        """Generate explanation for gaming behavior detection."""
        prompt = self._build_gaming_prompt(
            user_id, rapid_guess_rate, help_abuse_rate, severity
        )
        
        response = self.client.chat.completions.create(
            model=self.config.chat_model,
            messages=[
                {"role": "system", "content": self._gaming_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.5,
        )
        
        return LLMExplanation(
            text=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=self.config.chat_model,
            cost_estimate_usd=self._estimate_cost(response.usage.total_tokens),
        )
    
    def _mastery_system_prompt(self) -> str:
        return """You are an educational analytics assistant providing mastery insights.
Your explanations should be:
- Clear and encouraging
- Focused on learning patterns, not just scores
- Actionable with specific recommendations
Keep responses concise (2-3 sentences max)."""
    
    def _recommendation_system_prompt(self) -> str:
        return """You are an adaptive learning system explaining item recommendations.
Explain why a practice item was selected for students in a helpful, non-technical way.
Keep responses brief (1-2 sentences)."""
    
    def _gaming_system_prompt(self) -> str:
        return """You are a learning behavior analyst providing intervention guidance.
When explaining gaming behavior, be:
- Non-judgmental and constructive
- Focused on helping students improve their learning approach
Keep responses brief and actionable."""
    
    def _build_mastery_prompt(
        self,
        user_id: str,
        skill_id: str,
        mastery_score: float,
        key_factors: List[Dict],
        interaction_count: int,
    ) -> str:
        mastery_pct = mastery_score * 100
        trend = "improving" if mastery_score > 0.6 else "developing" if mastery_score > 0.4 else "emerging"
        
        factors_text = "\n".join([
            f"- {f.get('item_id', 'item')}: weight {f.get('attention_weight', 0):.2f}, "
            f"{'correct' if f.get('correct') else 'incorrect'}"
            for f in key_factors[:3]
        ])
        
        return f"""Student mastery analysis:
- Skill: {skill_id}
- Mastery Score: {mastery_pct:.0f}%
- Trend: {trend}
- Practice Sessions: {interaction_count}
- Key Contributing Interactions:
{factors_text}

Provide a brief insight and one recommendation."""
    
    def _build_recommendation_prompt(
        self,
        user_id: str,
        item_id: str,
        skill: str,
        expected_reward: float,
        uncertainty: float,
        is_exploration: bool,
        student_mastery: float,
        item_difficulty: float,
    ) -> str:
        mode = "exploration" if is_exploration else "exploitation"
        success_chance = expected_reward * 100
        
        return f"""Recommendation context:
- Item: {item_id} (skill: {skill})
- Student Mastery: {student_mastery*100:.0f}%
- Item Difficulty: {item_difficulty:.2f}
- Expected Success: {success_chance:.0f}%
- Mode: {mode} (uncertainty: {uncertainty:.2f})

Explain why this item is a good choice right now."""
    
    def _build_gaming_prompt(
        self,
        user_id: str,
        rapid_guess_rate: float,
        help_abuse_rate: float,
        severity: str,
    ) -> str:
        return f"""Gaming behavior detected:
- Rapid Guess Rate: {rapid_guess_rate*100:.0f}%
- Help Abuse Rate: {help_abuse_rate*100:.0f}%
- Severity: {severity}

Provide a constructive explanation and intervention suggestion."""
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate Azure OpenAI cost."""
        # GPT-4o pricing (approximate)
        input_cost = 0.0025 / 1000  # per token
        output_cost = 0.01 / 1000
        
        # Assume 60% input, 40% output
        return (tokens * 0.6 * input_cost) + (tokens * 0.4 * output_cost)


# Convenience functions matching original API

def generate_explanation_with_llm(
    user_id: str,
    skill_id: str,
    mastery_score: float,
    key_factors: List[Dict],
    interaction_count: int,
) -> str:
    """
    Drop-in replacement for template-based explanations.
    
    Uses Azure OpenAI when available, falls back to templates.
    """
    try:
        explainer = AzureOpenAIExplainer()
        result = explainer.explain_mastery(
            user_id, skill_id, mastery_score, key_factors, interaction_count
        )
        return result.text
    except Exception as e:
        # Fall back to template-based
        from .explainability import analyze_attention_pattern
        insight, recommendation = analyze_attention_pattern(key_factors, mastery_score)
        return f"{insight}\n\nRecommendation: {recommendation}"


def generate_rl_reason_with_llm(
    user_id: str,
    item_id: str,
    skill: str,
    expected_reward: float,
    uncertainty: float,
    is_exploration: bool,
    student_mastery: float = 0.5,
    item_difficulty: float = 0.0,
) -> str:
    """
    Drop-in replacement for RL recommendation reasoning.
    """
    try:
        explainer = AzureOpenAIExplainer()
        result = explainer.explain_recommendation(
            user_id, item_id, skill, expected_reward, uncertainty,
            is_exploration, student_mastery, item_difficulty
        )
        return result.text
    except Exception:
        # Fall back to template
        mode = "exploring" if is_exploration else "exploiting"
        return f"{mode.capitalize()} {skill}: expected success {expected_reward:.0%}"


# Environment check utility
def is_azure_openai_available() -> bool:
    """Check if Azure OpenAI is configured and available."""
    try:
        config = get_azure_openai_config()
        return bool(config.endpoint and config.api_key)
    except Exception:
        return False


if __name__ == "__main__":
    # Test configuration
    print(f"Azure OpenAI available: {is_azure_openai_available()}")
    
    if is_azure_openai_available():
        explainer = AzureOpenAIExplainer()
        
        # Test mastery explanation
        result = explainer.explain_mastery(
            user_id="test_user",
            skill_id="algebra",
            mastery_score=0.75,
            key_factors=[
                {"item_id": "q1", "attention_weight": 0.4, "correct": True},
                {"item_id": "q2", "attention_weight": 0.3, "correct": False},
            ],
            interaction_count=25,
        )
        
        print(f"Mastery Explanation:\n{result.text}")
        print(f"Tokens: {result.tokens_used}, Cost: ${result.cost_estimate_usd:.4f}")

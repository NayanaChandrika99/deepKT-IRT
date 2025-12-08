# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - RAG Explainer
# MAGIC 
# MAGIC Implements RAG-based explanations using Databricks Vector Search and Azure OpenAI.
# MAGIC 
# MAGIC **Use Cases:**
# MAGIC 1. Personalized Explanations: Explain why a student got an answer wrong
# MAGIC 2. Mastery Insights: Explain student's skill mastery patterns
# MAGIC 3. Recommendation Reasoning: Explain why an item was recommended
# MAGIC 
# MAGIC **Input:**
# MAGIC - Vector Search index (content library)
# MAGIC - Student/item context from Gold layer
# MAGIC - Azure OpenAI GPT-4o
# MAGIC 
# MAGIC **Output:**
# MAGIC - `gold.rag_explanations`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from openai import AzureOpenAI
from pyspark.sql import functions as F
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

CATALOG = "deepkt_irt"
GOLD_SCHEMA = "gold"

spark.sql(f"USE CATALOG {CATALOG}")

# Vector Search configuration
VS_ENDPOINT_NAME = "deepkt-vector-search"
VS_INDEX_NAME = f"{CATALOG}.{GOLD_SCHEMA}.content_embeddings_index"

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope="deepkt-secrets", key="azure-openai-endpoint")
AZURE_OPENAI_KEY = dbutils.secrets.get(scope="deepkt-secrets", key="azure-openai-key")
GPT_MODEL = "gpt-4o"  # Or "gpt-35-turbo" for cost savings
EMBEDDING_MODEL = "text-embedding-ada-002"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Initialize Clients

# COMMAND ----------

# Initialize Azure OpenAI
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-15-preview"
)

# Initialize Vector Search
vsc = VectorSearchClient()

def get_embedding(text: str) -> list:
    """Get embedding from Azure OpenAI."""
    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def search_content(query: str, num_results: int = 5) -> List[Dict]:
    """Search content library."""
    try:
        query_embedding = get_embedding(query)
        index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
        
        results = index.similarity_search(
            query_vector=query_embedding,
            columns=["content_id", "content_type", "title", "text", "skill_id"],
            num_results=num_results
        )
        
        return results.get('result', {}).get('data_array', [])
    except Exception as e:
        print(f"Vector search error: {e}")
        return []

print("Clients initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. RAG Explanation Generator

# COMMAND ----------

@dataclass
class ExplanationContext:
    """Context for generating an explanation."""
    user_id: str
    skill_id: Optional[str]
    item_id: Optional[str]
    question_text: Optional[str]
    student_answer: Optional[str]
    correct_answer: Optional[str]
    mastery_score: float
    explanation_type: str  # "wrong_answer", "mastery_insight", "recommendation"

@dataclass
class RAGExplanation:
    """Generated explanation with metadata."""
    explanation_text: str
    sources_used: List[str]
    confidence: float
    explanation_type: str
    tokens_used: int


class RAGExplainer:
    """
    RAG-based explanation generator using Databricks Vector Search + Azure OpenAI.
    
    Implements the UWorld "Personalized Explanations" use case:
    Takes a student's wrong answer and generates a custom explanation
    based on trusted content library (not generic GPT answers).
    """
    
    def __init__(self, openai_client, vsc, vs_endpoint: str, vs_index: str):
        self.openai = openai_client
        self.vsc = vsc
        self.vs_endpoint = vs_endpoint
        self.vs_index = vs_index
    
    def retrieve_context(self, context: ExplanationContext, num_results: int = 5) -> List[Dict]:
        """Retrieve relevant content from vector store."""
        
        # Build search query based on context
        query_parts = []
        
        if context.skill_id:
            query_parts.append(f"skill: {context.skill_id}")
        
        if context.item_id:
            query_parts.append(f"problem: {context.item_id}")
        
        if context.question_text:
            query_parts.append(context.question_text[:200])
        
        if context.explanation_type == "wrong_answer":
            query_parts.append("common mistakes misconceptions solving strategy")
        elif context.explanation_type == "mastery_insight":
            query_parts.append("learning progression skill development")
        elif context.explanation_type == "recommendation":
            query_parts.append("difficulty level practice recommendation")
        
        query = " ".join(query_parts)
        return search_content(query, num_results)
    
    def generate_explanation(self, context: ExplanationContext) -> RAGExplanation:
        """Generate personalized explanation using RAG."""
        
        # Step 1: Retrieve relevant content
        retrieved_content = self.retrieve_context(context)
        
        # Build context from retrieved content
        if retrieved_content:
            content_texts = [item[3] for item in retrieved_content]  # text column
            source_ids = [item[0] for item in retrieved_content]  # content_id column
            rag_context = "\n\n".join(content_texts[:3])  # Use top 3
        else:
            rag_context = "No specific content available. Use general educational principles."
            source_ids = []
        
        # Step 2: Build prompt based on explanation type
        if context.explanation_type == "wrong_answer":
            prompt = self._build_wrong_answer_prompt(context, rag_context)
        elif context.explanation_type == "mastery_insight":
            prompt = self._build_mastery_prompt(context, rag_context)
        elif context.explanation_type == "recommendation":
            prompt = self._build_recommendation_prompt(context, rag_context)
        else:
            prompt = self._build_general_prompt(context, rag_context)
        
        # Step 3: Generate with GPT-4o
        response = self.openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        explanation_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        return RAGExplanation(
            explanation_text=explanation_text,
            sources_used=source_ids,
            confidence=min(0.9, 0.5 + 0.1 * len(source_ids)),
            explanation_type=context.explanation_type,
            tokens_used=tokens_used
        )
    
    def _get_system_prompt(self) -> str:
        return """You are an expert educational tutor providing personalized explanations.

Your explanations should:
1. Be clear, concise, and encouraging
2. Focus on understanding, not just the correct answer
3. Connect to broader learning concepts
4. Provide actionable next steps for improvement
5. Use the provided content context to give accurate, curriculum-aligned guidance

Always ground your explanations in the educational content provided. Do not make up information."""
    
    def _build_wrong_answer_prompt(self, context: ExplanationContext, rag_context: str) -> str:
        return f"""A student answered a question incorrectly. Help them understand their mistake.

**Skill Area:** {context.skill_id or 'Not specified'}
**Student's Current Mastery:** {context.mastery_score:.0%}
**Question:** {context.question_text or 'Not provided'}
**Student's Answer:** {context.student_answer or 'Not provided'}
**Correct Answer:** {context.correct_answer or 'Not provided'}

**Relevant Educational Content:**
{rag_context}

Based on the above, provide:
1. A brief explanation of why the student's answer was incorrect
2. The key concept they may have misunderstood
3. A step-by-step approach to solve this type of problem
4. One practice tip to reinforce the correct understanding"""
    
    def _build_mastery_prompt(self, context: ExplanationContext, rag_context: str) -> str:
        return f"""Explain a student's mastery progress and provide learning insights.

**User ID:** {context.user_id}
**Skill Area:** {context.skill_id or 'Not specified'}  
**Current Mastery Score:** {context.mastery_score:.0%}

**Relevant Educational Content:**
{rag_context}

Based on the mastery level, provide:
1. An interpretation of what this mastery score means
2. Strengths the student has demonstrated
3. Areas that still need development
4. Recommended next steps for continued improvement"""
    
    def _build_recommendation_prompt(self, context: ExplanationContext, rag_context: str) -> str:
        return f"""Explain why a specific practice item was recommended for a student.

**User ID:** {context.user_id}
**Current Mastery:** {context.mastery_score:.0%}
**Recommended Item:** {context.item_id or 'Not specified'}
**Target Skill:** {context.skill_id or 'Not specified'}

**Educational Context:**
{rag_context}

Explain:
1. Why this particular item is a good fit for the student right now
2. How practicing this will help their learning progression
3. What the student should focus on when attempting this problem"""
    
    def _build_general_prompt(self, context: ExplanationContext, rag_context: str) -> str:
        return f"""Provide educational guidance for the following context:

**User ID:** {context.user_id}
**Skill:** {context.skill_id or 'Not specified'}
**Mastery:** {context.mastery_score:.0%}

**Educational Content:**
{rag_context}

Provide helpful guidance and next steps for this student."""

# COMMAND ----------

# Initialize explainer
explainer = RAGExplainer(openai_client, vsc, VS_ENDPOINT_NAME, VS_INDEX_NAME)
print("RAG Explainer initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Sample Explanations

# COMMAND ----------

def generate_explanations_for_students(limit: int = 20):
    """Generate explanations for a sample of students."""
    
    # Get sample predictions (wrong answers)
    wrong_answers = (
        spark.table(f"{GOLD_SCHEMA}.sakt_predictions")
        .filter(F.col("actual") == 0)  # Incorrect answers
        .filter(F.col("predicted") > 0.5)  # Model expected correct
        .limit(limit)
    ).toPandas()
    
    # Get skill vocab
    skill_vocab = spark.table(f"{GOLD_SCHEMA}.skill_vocabulary").toPandas()
    skill_map = dict(zip(skill_vocab['skill_idx'], skill_vocab['skill_id']))
    
    explanations = []
    
    for _, row in wrong_answers.iterrows():
        skill_id = skill_map.get(int(row['skill_id']), str(row['skill_id']))
        
        context = ExplanationContext(
            user_id=str(row['user_idx']),
            skill_id=skill_id,
            item_id=None,
            question_text=None,
            student_answer=None,
            correct_answer=None,
            mastery_score=float(row['predicted']),
            explanation_type="wrong_answer"
        )
        
        try:
            explanation = explainer.generate_explanation(context)
            
            explanations.append({
                'user_id': context.user_id,
                'skill_id': context.skill_id,
                'mastery_score': context.mastery_score,
                'explanation_type': int(context.explanation_type == "wrong_answer"),
                'explanation_text': explanation.explanation_text,
                'sources_used': json.dumps(explanation.sources_used),
                'confidence': explanation.confidence,
                'tokens_used': explanation.tokens_used
            })
            
            print(f"Generated explanation for user {context.user_id}")
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
    
    return explanations

# Generate explanations
explanations = generate_explanations_for_students(limit=10)
print(f"\nGenerated {len(explanations)} explanations")

# COMMAND ----------

# Save explanations to Gold layer
if explanations:
    explanations_df = (
        spark.createDataFrame(explanations)
        .withColumn("generated_at", F.current_timestamp())
    )
    
    (
        explanations_df
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(f"{GOLD_SCHEMA}.rag_explanations")
    )
    
    print("Explanations saved to Gold layer")
    explanations_df.show(3, truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. API Function for Real-Time Explanations

# COMMAND ----------

def get_personalized_explanation(
    user_id: str,
    skill_id: str = None,
    item_id: str = None,
    explanation_type: str = "mastery_insight",
    question_text: str = None,
    student_answer: str = None,
    correct_answer: str = None
) -> Dict:
    """
    API function for generating real-time personalized explanations.
    
    This can be exposed via Databricks Model Serving for real-time access.
    """
    
    # Get student's latest mastery
    mastery = 0.5  # Default
    try:
        mastery_df = (
            spark.table(f"{GOLD_SCHEMA}.sakt_predictions")
            .filter(F.col("user_idx") == user_id)
            .agg(F.avg("predicted"))
            .collect()
        )
        if mastery_df:
            mastery = float(mastery_df[0][0])
    except:
        pass
    
    context = ExplanationContext(
        user_id=user_id,
        skill_id=skill_id,
        item_id=item_id,
        question_text=question_text,
        student_answer=student_answer,
        correct_answer=correct_answer,
        mastery_score=mastery,
        explanation_type=explanation_type
    )
    
    explanation = explainer.generate_explanation(context)
    
    return {
        "user_id": user_id,
        "explanation": explanation.explanation_text,
        "sources": explanation.sources_used,
        "confidence": explanation.confidence,
        "tokens_used": explanation.tokens_used
    }

# Test the API function
test_result = get_personalized_explanation(
    user_id="test_user",
    skill_id="algebra",
    explanation_type="mastery_insight"
)

print("API Test Result:")
print(json.dumps(test_result, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cost Tracking

# COMMAND ----------

def estimate_costs(total_tokens: int) -> Dict:
    """Estimate Azure OpenAI costs."""
    
    # Pricing (approximate, check Azure portal for current rates)
    GPT4O_INPUT_COST = 0.0025 / 1000  # per 1K tokens
    GPT4O_OUTPUT_COST = 0.01 / 1000
    EMBEDDING_COST = 0.0001 / 1000
    
    # Assume 60% input, 40% output for chat
    input_tokens = int(total_tokens * 0.6)
    output_tokens = int(total_tokens * 0.4)
    
    gpt_cost = (input_tokens * GPT4O_INPUT_COST) + (output_tokens * GPT4O_OUTPUT_COST)
    embedding_cost = total_tokens * EMBEDDING_COST * 0.2  # Embedding queries are smaller
    
    return {
        "total_tokens": total_tokens,
        "estimated_gpt_cost_usd": round(gpt_cost, 4),
        "estimated_embedding_cost_usd": round(embedding_cost, 4),
        "total_estimated_cost_usd": round(gpt_cost + embedding_cost, 4)
    }

# Calculate costs for generated explanations
if explanations:
    total_tokens = sum(e['tokens_used'] for e in explanations)
    costs = estimate_costs(total_tokens)
    print(f"Cost estimate for {len(explanations)} explanations:")
    print(json.dumps(costs, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC RAG Explainer implementation complete:
# MAGIC 
# MAGIC **Components:**
# MAGIC - `RAGExplainer` class for personalized explanations
# MAGIC - Databricks Vector Search integration for content retrieval
# MAGIC - Azure OpenAI GPT-4o for generation
# MAGIC 
# MAGIC **Explanation Types:**
# MAGIC - `wrong_answer`: Help students understand mistakes
# MAGIC - `mastery_insight`: Explain skill progression
# MAGIC - `recommendation`: Justify item recommendations
# MAGIC 
# MAGIC **Output:**
# MAGIC - `gold.rag_explanations` - Generated explanations with metadata
# MAGIC 
# MAGIC **For Production:**
# MAGIC - Deploy `get_personalized_explanation` via Databricks Model Serving
# MAGIC - Implement caching for frequent queries
# MAGIC - Add feedback loop for explanation quality improvement

# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Build Vector Search Index
# MAGIC 
# MAGIC Creates Databricks Vector Search index for RAG-based explanations.
# MAGIC 
# MAGIC **Input:**
# MAGIC - Content library (skills, items, explanations)
# MAGIC - Azure OpenAI embeddings
# MAGIC 
# MAGIC **Output:**
# MAGIC - Vector Search endpoint
# MAGIC - Content embeddings index

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from openai import AzureOpenAI
import json
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType

CATALOG = "deepkt_irt"
GOLD_SCHEMA = "gold"
SILVER_SCHEMA = "silver"

spark.sql(f"USE CATALOG {CATALOG}")

# Vector Search configuration
VS_ENDPOINT_NAME = "deepkt-vector-search"
VS_INDEX_NAME = f"{CATALOG}.{GOLD_SCHEMA}.content_embeddings_index"

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope="deepkt-secrets", key="azure-openai-endpoint")
AZURE_OPENAI_KEY = dbutils.secrets.get(scope="deepkt-secrets", key="azure-openai-key")
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Content Library
# MAGIC 
# MAGIC Build a library of educational content for RAG retrieval.

# COMMAND ----------

def create_content_library():
    """
    Create content library from skills, items, and explanations.
    
    In production, this would include:
    - Skill descriptions and learning objectives
    - Item explanations and solution strategies
    - Common misconceptions and remediation tips
    """
    
    # Skills with descriptions (mock data - in production from CMS)
    skills = spark.table(f"{SILVER_SCHEMA}.skills").toPandas()
    
    skill_content = []
    for _, row in skills.iterrows():
        skill_id = row['skill_id']
        # In production, these would come from a content management system
        skill_content.append({
            'content_id': f"skill_{skill_id}",
            'content_type': 'skill_description',
            'title': f"Skill: {skill_id}",
            'text': f"This skill covers the learning objective: {skill_id}. Students should understand the core concepts and be able to apply them in various problem contexts.",
            'skill_id': skill_id,
            'item_id': None
        })
    
    # Items with explanations
    items = spark.table(f"{SILVER_SCHEMA}.items").toPandas()
    item_params = spark.table(f"{GOLD_SCHEMA}.item_parameters").toPandas()
    
    item_content = []
    for _, row in items.head(500).iterrows():  # Limit for demo
        item_id = row['item_id']
        skills_list = row['skill_ids'] if isinstance(row['skill_ids'], list) else []
        
        # Get difficulty from parameters
        param_row = item_params[item_params['item_id'] == item_id]
        difficulty = param_row['difficulty'].values[0] if len(param_row) > 0 else 0.0
        
        difficulty_label = "easy" if difficulty < -0.5 else "moderate" if difficulty < 0.5 else "challenging"
        
        item_content.append({
            'content_id': f"item_{item_id}",
            'content_type': 'item_explanation',
            'title': f"Item: {item_id}",
            'text': f"This {difficulty_label} problem tests understanding of {', '.join(skills_list[:3])}. The key to solving this problem is identifying the relevant concepts and applying systematic problem-solving strategies.",
            'skill_id': skills_list[0] if skills_list else None,
            'item_id': item_id
        })
    
    # Combine content
    all_content = skill_content + item_content
    
    return spark.createDataFrame(all_content)

# Create content library
content_df = create_content_library()
print(f"Content library size: {content_df.count()}")
content_df.show(5, truncate=50)

# COMMAND ----------

# Save content library to Gold layer
(
    content_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{GOLD_SCHEMA}.content_library")
)

print("Content library saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate Embeddings with Azure OpenAI

# COMMAND ----------

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-15-preview"
)

def get_embedding(text: str) -> list:
    """Get embedding vector from Azure OpenAI."""
    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

# Test embedding
test_embedding = get_embedding("This is a test sentence.")
print(f"Embedding dimension: {len(test_embedding)}")

# COMMAND ----------

# Create UDF for batch embedding
from pyspark.sql.functions import udf, pandas_udf
import pandas as pd

@pandas_udf(ArrayType(FloatType()))
def embed_text_batch(texts: pd.Series) -> pd.Series:
    """Batch embedding UDF for Spark."""
    embeddings = []
    
    # Process in batches of 100 for API efficiency
    batch_size = 100
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size].tolist()
        
        # Filter empty texts
        batch = [t if t else "empty" for t in batch]
        
        try:
            response = openai_client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error embedding batch: {e}")
            # Return zero vectors on error
            embeddings.extend([[0.0] * EMBEDDING_DIMENSION] * len(batch))
    
    return pd.Series(embeddings)

# COMMAND ----------

# Generate embeddings for content library
content_with_embeddings = (
    spark.table(f"{GOLD_SCHEMA}.content_library")
    .withColumn("embedding", embed_text_batch(F.col("text")))
)

# Save with embeddings
(
    content_with_embeddings
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{GOLD_SCHEMA}.content_embeddings")
)

print("Content embeddings generated and saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Vector Search Endpoint

# COMMAND ----------

# Initialize Vector Search client
vsc = VectorSearchClient()

# Create endpoint (if not exists)
try:
    endpoint = vsc.get_endpoint(VS_ENDPOINT_NAME)
    print(f"Endpoint {VS_ENDPOINT_NAME} already exists")
except Exception as e:
    print(f"Creating endpoint {VS_ENDPOINT_NAME}...")
    vsc.create_endpoint(
        name=VS_ENDPOINT_NAME,
        endpoint_type="STANDARD"
    )
    print("Endpoint created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Vector Search Index

# COMMAND ----------

# Create Delta Sync index from the embeddings table
SOURCE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.content_embeddings"

try:
    index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    print(f"Index {VS_INDEX_NAME} already exists")
except Exception as e:
    print(f"Creating index {VS_INDEX_NAME}...")
    
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        source_table_name=SOURCE_TABLE,
        index_name=VS_INDEX_NAME,
        pipeline_type="TRIGGERED",  # Manual sync for now
        primary_key="content_id",
        embedding_dimension=EMBEDDING_DIMENSION,
        embedding_vector_column="embedding",
        embedding_source_column="text"
    )
    
    print("Index created successfully")

# COMMAND ----------

# Sync index with latest data
try:
    vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME).sync()
    print("Index sync triggered")
except Exception as e:
    print(f"Sync status: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test Vector Search

# COMMAND ----------

def search_content(query: str, num_results: int = 5) -> list:
    """Search content library using vector similarity."""
    
    # Get query embedding
    query_embedding = get_embedding(query)
    
    # Search index
    index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    
    results = index.similarity_search(
        query_vector=query_embedding,
        columns=["content_id", "content_type", "title", "text", "skill_id"],
        num_results=num_results
    )
    
    return results.get('result', {}).get('data_array', [])

# Test search
test_results = search_content("algebra equations solving")
print("Search results:")
for result in test_results[:3]:
    print(f"  - {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC Vector Search index created:
# MAGIC - Content library: `gold.content_library`
# MAGIC - Embeddings: `gold.content_embeddings`
# MAGIC - Vector Search endpoint: `deepkt-vector-search`
# MAGIC - Vector Search index: `content_embeddings_index`
# MAGIC 
# MAGIC Next: Use for RAG explanations in `08_rag_explainer`.

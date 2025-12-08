# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Transform Silver Layer
# MAGIC 
# MAGIC Transforms raw Bronze data into canonical Silver layer format.
# MAGIC 
# MAGIC **Input:** Bronze tables
# MAGIC - `bronze.raw_action_logs`
# MAGIC - `bronze.raw_assignment_details`
# MAGIC - `bronze.raw_problem_details`
# MAGIC 
# MAGIC **Output:** Silver tables
# MAGIC - `silver.canonical_learning_events` - Standardized LearningEvent schema
# MAGIC - `silver.users` - User dimension table
# MAGIC - `silver.items` - Item dimension table
# MAGIC - `silver.skills` - Skill dimension table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, IntegerType, BooleanType, TimestampType
from pyspark.sql.window import Window

# Catalog configuration
# UPDATED: Using 'workspace' catalog and 'default' schema where data is uploaded
CATALOG = "workspace"
BRONZE_SCHEMA = "default"  # Your data is here!
SILVER_SCHEMA = "default"  # We'll save results here too

# Note: If you want to organize data better, you can:
# 1. Create bronze/silver/gold schemas: spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")
# 2. Move tables there, or
# 3. Just use 'default' schema for everything (simpler for testing)

try:
    spark.sql(f"USE CATALOG {CATALOG}")
    print(f"✓ Using catalog: {CATALOG}")
except Exception as e:
    # If workspace catalog doesn't exist, fall back to hive_metastore
    print(f"Workspace catalog not available, using hive_metastore")
    CATALOG = "hive_metastore"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER_SCHEMA}")

print(f"Configuration: {CATALOG}.{BRONZE_SCHEMA} (source) → {CATALOG}.{SILVER_SCHEMA} (output)")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define Canonical Schema
# MAGIC 
# MAGIC Maps to the `LearningEvent` dataclass from the original system.

# COMMAND ----------

# Canonical LearningEvent schema matching src/common/schemas.py
LEARNING_EVENT_SCHEMA = """
    user_id STRING,
    item_id STRING, 
    skill_ids ARRAY<STRING>,
    timestamp TIMESTAMP,
    correct INT,
    action_sequence_id STRING,
    latency_ms INT,
    help_requested BOOLEAN,
    -- Metadata columns
    _source STRING,
    _processed_at TIMESTAMP
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Transform EDM Cup 2023 Data
# MAGIC 
# MAGIC Replicates logic from `src/common/data_pipeline.py::_prepare_edm_events`

# COMMAND ----------

# Response actions indicating answer submission
RESPONSE_ACTIONS = ["submit_answer", "submit", "answer"]
HELP_ACTIONS = ["request_hint", "view_hint", "request_help"]
PROBLEM_START_ACTION = "problem_start"

@F.udf(returnType=ArrayType(StringType()))
def parse_skill_list(raw_value):
    """Parse skill list from JSON or comma-separated string."""
    if raw_value is None:
        return []
    
    import json
    try:
        # Try JSON array first
        skills = json.loads(raw_value)
        if isinstance(skills, list):
            return [str(s) for s in skills]
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Fall back to comma-separated
    return [s.strip() for s in str(raw_value).split(",") if s.strip()]


def transform_edm_to_canonical():
    """
    Transform EDM Cup 2023 raw data to canonical LearningEvent format.
    
    Replicates logic from data_pipeline.py but in PySpark.
    """
    # Load Bronze tables
    action_logs = spark.table(f"{BRONZE_SCHEMA}.raw_action_logs")
    assignment_details = spark.table(f"{BRONZE_SCHEMA}.raw_assignment_details")
    problem_details = spark.table(f"{BRONZE_SCHEMA}.raw_problem_details")
    
    # Filter to response actions only
    responses = action_logs.filter(
        F.col("action").isin(RESPONSE_ACTIONS)
    )
    
    # Calculate latency from problem_start to submit
    # Window to get previous action timestamp per user/problem
    window_spec = Window.partitionBy("user_id", "problem_id").orderBy("timestamp")
    
    responses = (
        responses
        .withColumn("prev_timestamp", F.lag("timestamp").over(window_spec))
        .withColumn(
            "latency_ms",
            F.when(
                F.col("prev_timestamp").isNotNull(),
                (F.unix_timestamp("timestamp") - F.unix_timestamp("prev_timestamp")) * 1000
            ).otherwise(None).cast(IntegerType())
        )
    )
    
    # Check if help was requested before this response
    help_actions_df = action_logs.filter(F.col("action").isin(HELP_ACTIONS))
    
    help_per_problem = (
        help_actions_df
        .groupBy("user_id", "problem_id", "action_sequence_id")
        .agg(F.lit(True).alias("help_requested"))
    )
    
    responses = responses.join(
        help_per_problem,
        on=["user_id", "problem_id", "action_sequence_id"],
        how="left"
    ).fillna({"help_requested": False})
    
    # Join with assignment details for additional metadata
    responses = responses.join(
        assignment_details.select(
            "assignment_id", 
            F.col("assignment_name").alias("assignment_name")
        ),
        on="assignment_id",
        how="left"
    )
    
    # Join with problem details for skill information
    responses = responses.join(
        problem_details.select(
            F.col("problem_id"),
            F.col("skill_ids").alias("raw_skill_ids"),
            F.col("is_multiple_choice").alias("is_mc")
        ),
        on="problem_id",
        how="left"
    )
    
    # Transform to canonical schema
    canonical_df = (
        responses
        .select(
            F.col("user_id").cast(StringType()).alias("user_id"),
            F.col("problem_id").cast(StringType()).alias("item_id"),
            parse_skill_list("raw_skill_ids").alias("skill_ids"),
            F.col("timestamp").cast(TimestampType()).alias("timestamp"),
            F.when(F.col("correct") == True, 1).otherwise(0).alias("correct"),
            F.col("action_sequence_id").cast(StringType()).alias("action_sequence_id"),
            F.col("latency_ms"),
            F.col("help_requested"),
            F.lit("edm_cup_2023").alias("_source"),
            F.current_timestamp().alias("_processed_at")
        )
        .filter(F.col("user_id").isNotNull())
        .filter(F.col("item_id").isNotNull())
    )
    
    return canonical_df

# COMMAND ----------

# Run transformation
canonical_events = transform_edm_to_canonical()

# Show sample
print(f"Transformed {canonical_events.count()} canonical events")
canonical_events.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write to Silver Layer

# COMMAND ----------

# Write canonical events
(
    canonical_events
    .write
    .format("delta")
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(f"{SILVER_SCHEMA}.canonical_learning_events")
)

print("Wrote canonical_learning_events to Silver layer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Dimension Tables

# COMMAND ----------

# Users dimension
users_df = (
    canonical_events
    .select("user_id")
    .distinct()
    .withColumn("created_at", F.current_timestamp())
)

(
    users_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{SILVER_SCHEMA}.users")
)

print(f"Created users dimension: {users_df.count()} users")

# Items dimension
items_df = (
    canonical_events
    .select("item_id", "skill_ids")
    .distinct()
    .withColumn("created_at", F.current_timestamp())
)

(
    items_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{SILVER_SCHEMA}.items")
)

print(f"Created items dimension: {items_df.count()} items")

# Skills dimension (explode skill arrays)
skills_df = (
    canonical_events
    .select(F.explode("skill_ids").alias("skill_id"))
    .distinct()
    .withColumn("created_at", F.current_timestamp())
)

(
    skills_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{SILVER_SCHEMA}.skills")
)

print(f"Created skills dimension: {skills_df.count()} skills")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate User Splits
# MAGIC 
# MAGIC Deterministic train/val/test splits matching original logic.

# COMMAND ----------

def generate_user_splits(train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Create deterministic user splits for model training.
    Replicates logic from data_pipeline.py::generate_user_splits
    """
    import hashlib
    
    users = spark.table(f"{SILVER_SCHEMA}.users").select("user_id").collect()
    user_ids = sorted([row.user_id for row in users])
    
    # Deterministic shuffle using hash
    def hash_user(user_id):
        h = hashlib.md5(f"{seed}_{user_id}".encode()).hexdigest()
        return int(h, 16)
    
    user_ids_sorted = sorted(user_ids, key=hash_user)
    
    n = len(user_ids_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    splits = {
        "train": user_ids_sorted[:train_end],
        "val": user_ids_sorted[train_end:val_end],
        "test": user_ids_sorted[val_end:]
    }
    
    # Create splits DataFrame
    rows = []
    for split_name, users_in_split in splits.items():
        for user_id in users_in_split:
            rows.append({"user_id": user_id, "split": split_name})
    
    splits_df = spark.createDataFrame(rows)
    
    (
        splits_df
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(f"{SILVER_SCHEMA}.user_splits")
    )
    
    return {split: len(users) for split, users in splits.items()}

# Generate splits
split_counts = generate_user_splits(seed=42)
print(f"User splits: {split_counts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data Quality Validation

# COMMAND ----------

def validate_silver_tables():
    """Run data quality checks on Silver tables."""
    
    # Check canonical events
    events = spark.table(f"{SILVER_SCHEMA}.canonical_learning_events")
    
    checks = {
        "total_events": events.count(),
        "null_user_ids": events.filter(F.col("user_id").isNull()).count(),
        "null_item_ids": events.filter(F.col("item_id").isNull()).count(),
        "correct_values": events.groupBy("correct").count().collect(),
        "date_range": events.agg(
            F.min("timestamp").alias("min_ts"),
            F.max("timestamp").alias("max_ts")
        ).collect()[0],
        "avg_latency_ms": events.agg(F.avg("latency_ms")).collect()[0][0],
        "help_request_rate": events.filter(F.col("help_requested") == True).count() / events.count()
    }
    
    return checks

validation = validate_silver_tables()
for check, value in validation.items():
    print(f"{check}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC Silver layer transformation complete:
# MAGIC - `silver.canonical_learning_events` - Main fact table
# MAGIC - `silver.users` - User dimension
# MAGIC - `silver.items` - Item dimension
# MAGIC - `silver.skills` - Skill dimension
# MAGIC - `silver.user_splits` - Train/val/test splits
# MAGIC 
# MAGIC Next step: Feature engineering in `03_feature_engineering` notebook.

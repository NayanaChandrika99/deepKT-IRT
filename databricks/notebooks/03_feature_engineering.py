# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Feature Engineering
# MAGIC 
# MAGIC Engineers features for both SAKT and Wide & Deep IRT models.
# MAGIC 
# MAGIC **Input:** Silver tables
# MAGIC - `silver.canonical_learning_events`
# MAGIC - `silver.user_splits`
# MAGIC 
# MAGIC **Output:** Gold tables
# MAGIC - `gold.sakt_sequences` - Prepared sequences for SAKT training
# MAGIC - `gold.wdirt_features` - Clickstream features for WD-IRT

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, IntegerType, FloatType, StringType

CATALOG = "deepkt_irt"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. SAKT Sequence Preparation
# MAGIC 
# MAGIC Converts canonical events to PyKT-compatible sequences.
# MAGIC Replicates logic from `src/sakt_kt/adapters.py`.

# COMMAND ----------

# Configuration matching sakt_edm.yaml
SEQ_LEN = 200
MIN_SEQ_LEN = 3

def prepare_sakt_sequences(seq_len=SEQ_LEN, min_seq_len=MIN_SEQ_LEN):
    """
    Prepare SAKT training sequences from canonical events.
    
    Output format matches pyKT requirements:
    - uid: user identifier
    - questions: skill sequence (1-indexed)  
    - responses: correctness sequence
    - timestamps: event timestamps
    """
    events = spark.table(f"{SILVER_SCHEMA}.canonical_learning_events")
    splits = spark.table(f"{SILVER_SCHEMA}.user_splits")
    
    # Create skill vocabulary (1-indexed for pyKT)
    skills = spark.table(f"{SILVER_SCHEMA}.skills")
    skill_vocab = skills.select("skill_id").distinct().orderBy("skill_id")
    skill_vocab = skill_vocab.withColumn("skill_idx", F.row_number().over(Window.orderBy("skill_id")))
    
    # For multi-skill items, take first skill
    events_with_skill = (
        events
        .withColumn("primary_skill", F.col("skill_ids").getItem(0))
        .filter(F.col("primary_skill").isNotNull())
    )
    
    # Join with skill vocab
    events_indexed = events_with_skill.join(
        skill_vocab,
        events_with_skill.primary_skill == skill_vocab.skill_id,
        "inner"
    )
    
    # Order events per user
    window = Window.partitionBy("user_id").orderBy("timestamp")
    events_ordered = events_indexed.withColumn("seq_pos", F.row_number().over(window))
    
    # Aggregate into sequences per user
    user_sequences = (
        events_ordered
        .groupBy("user_id")
        .agg(
            F.collect_list(F.struct(
                F.col("seq_pos"),
                F.col("skill_idx"),
                F.col("correct"),
                F.col("timestamp")
            )).alias("events_struct")
        )
        .withColumn("seq_len", F.size("events_struct"))
        .filter(F.col("seq_len") >= min_seq_len)
    )
    
    # Extract ordered arrays
    @F.udf(returnType=ArrayType(IntegerType()))
    def extract_skills(events_struct):
        sorted_events = sorted(events_struct, key=lambda x: x.seq_pos)
        return [e.skill_idx for e in sorted_events][:seq_len]
    
    @F.udf(returnType=ArrayType(IntegerType()))
    def extract_responses(events_struct):
        sorted_events = sorted(events_struct, key=lambda x: x.seq_pos)
        return [e.correct for e in sorted_events][:seq_len]
    
    sequences = (
        user_sequences
        .withColumn("questions", extract_skills("events_struct"))
        .withColumn("responses", extract_responses("events_struct"))
        .withColumn("selectmasks", F.array_repeat(F.lit(1), F.size("questions")))
        .select("user_id", "questions", "responses", "selectmasks", "seq_len")
    )
    
    # Join with splits
    sequences = sequences.join(splits, on="user_id", how="inner")
    
    return sequences, skill_vocab

# COMMAND ----------

# Prepare sequences
sakt_sequences, skill_vocab = prepare_sakt_sequences()

print(f"Total sequences: {sakt_sequences.count()}")
print(f"Skill vocabulary size: {skill_vocab.count()}")

# Show sample
sakt_sequences.show(3, truncate=50)

# COMMAND ----------

# Save SAKT sequences
(
    sakt_sequences
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("split")
    .saveAsTable(f"{GOLD_SCHEMA}.sakt_sequences")
)

# Save skill vocabulary
(
    skill_vocab
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{GOLD_SCHEMA}.skill_vocabulary")
)

print("Saved SAKT sequences and skill vocabulary to Gold layer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Wide & Deep IRT Feature Engineering
# MAGIC 
# MAGIC Replicates logic from `src/wd_irt/features.py`.

# COMMAND ----------

# Feature configuration matching wd_irt_edm.yaml
LATENCY_BUCKETS = [0, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 60000, float('inf')]
HISTORY_WINDOW = 20

def engineer_wdirt_features(history_window=HISTORY_WINDOW):
    """
    Engineer clickstream features for Wide & Deep IRT.
    
    Features:
    - latency_bucket: Discretized response latency
    - recency: Time since first interaction (normalized)
    - item_success_rate: Historical success rate on this item
    - help_tendency: User's help request frequency
    - history_actions: Recent action sequence
    """
    events = spark.table(f"{SILVER_SCHEMA}.canonical_learning_events")
    splits = spark.table(f"{SILVER_SCHEMA}.user_splits")
    
    # Calculate latency bucket
    @F.udf(returnType=IntegerType())
    def get_latency_bucket(latency_ms):
        if latency_ms is None:
            return 0  # Unknown bucket
        for i, upper in enumerate(LATENCY_BUCKETS[1:], 1):
            if latency_ms < upper:
                return i
        return len(LATENCY_BUCKETS) - 1
    
    # Add window for per-user calculations
    user_window = Window.partitionBy("user_id").orderBy("timestamp")
    
    # Calculate features
    features = (
        events
        .withColumn("latency_bucket", get_latency_bucket("latency_ms"))
        .withColumn("event_order", F.row_number().over(user_window))
        .withColumn("first_timestamp", F.min("timestamp").over(Window.partitionBy("user_id")))
        .withColumn(
            "recency",
            (F.unix_timestamp("timestamp") - F.unix_timestamp("first_timestamp")).cast(FloatType())
        )
    )
    
    # Normalize recency per user
    user_max_recency = features.groupBy("user_id").agg(F.max("recency").alias("max_recency"))
    features = features.join(user_max_recency, on="user_id", how="left")
    features = features.withColumn(
        "recency_normalized",
        F.when(F.col("max_recency") > 0, F.col("recency") / F.col("max_recency")).otherwise(0.0)
    )
    
    # Item success rate (historical average)
    item_window = Window.partitionBy("item_id").orderBy("timestamp").rowsBetween(Window.unboundedPreceding, -1)
    features = features.withColumn(
        "item_success_rate",
        F.coalesce(F.avg("correct").over(item_window), F.lit(0.5))  # Default 0.5 for first attempt
    )
    
    # User help tendency (rolling average)
    help_window = Window.partitionBy("user_id").orderBy("timestamp").rowsBetween(-history_window, -1)
    features = features.withColumn(
        "help_tendency",
        F.coalesce(
            F.avg(F.when(F.col("help_requested"), 1.0).otherwise(0.0)).over(help_window),
            F.lit(0.0)
        )
    )
    
    # Recent accuracy (rolling)
    features = features.withColumn(
        "recent_accuracy",
        F.coalesce(F.avg("correct").over(help_window.rowsBetween(-10, -1)), F.lit(0.5))
    )
    
    # Select final features
    wdirt_features = (
        features
        .select(
            "user_id",
            "item_id",
            "timestamp",
            "correct",
            "latency_bucket",
            "recency_normalized",
            "item_success_rate",
            "help_tendency",
            "recent_accuracy",
            F.when(F.col("help_requested"), 1).otherwise(0).alias("help_requested_int"),
            "event_order"
        )
    )
    
    # Join with splits
    wdirt_features = wdirt_features.join(splits, on="user_id", how="inner")
    
    return wdirt_features

# COMMAND ----------

# Engineer features
wdirt_features = engineer_wdirt_features()

print(f"Total feature records: {wdirt_features.count()}")
wdirt_features.show(5)

# COMMAND ----------

# Save WD-IRT features
(
    wdirt_features
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("split")
    .saveAsTable(f"{GOLD_SCHEMA}.wdirt_features")
)

print("Saved WD-IRT features to Gold layer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Item Vocabulary

# COMMAND ----------

# Item vocabulary for embeddings
items = spark.table(f"{SILVER_SCHEMA}.items")
item_vocab = (
    items
    .select("item_id")
    .distinct()
    .orderBy("item_id")
    .withColumn("item_idx", F.row_number().over(Window.orderBy("item_id")))
)

(
    item_vocab
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{GOLD_SCHEMA}.item_vocabulary")
)

print(f"Item vocabulary size: {item_vocab.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature Statistics for Model Config

# COMMAND ----------

def compute_feature_stats():
    """Compute statistics needed for model configuration."""
    
    sakt_seqs = spark.table(f"{GOLD_SCHEMA}.sakt_sequences")
    wdirt_feats = spark.table(f"{GOLD_SCHEMA}.wdirt_features")
    skill_vocab = spark.table(f"{GOLD_SCHEMA}.skill_vocabulary")
    item_vocab = spark.table(f"{GOLD_SCHEMA}.item_vocabulary")
    
    stats = {
        # SAKT config
        "n_skills": skill_vocab.count(),
        "max_seq_len": sakt_seqs.agg(F.max("seq_len")).collect()[0][0],
        "total_sequences": sakt_seqs.count(),
        "train_sequences": sakt_seqs.filter(F.col("split") == "train").count(),
        "val_sequences": sakt_seqs.filter(F.col("split") == "val").count(),
        "test_sequences": sakt_seqs.filter(F.col("split") == "test").count(),
        
        # WD-IRT config
        "n_items": item_vocab.count(),
        "n_latency_buckets": 10,
        "total_wdirt_records": wdirt_feats.count(),
        
        # Feature distributions
        "avg_help_tendency": wdirt_feats.agg(F.avg("help_tendency")).collect()[0][0],
        "avg_item_success_rate": wdirt_feats.agg(F.avg("item_success_rate")).collect()[0][0],
    }
    
    return stats

stats = compute_feature_stats()
print("Feature Statistics for Model Config:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC Feature engineering complete:
# MAGIC 
# MAGIC **Gold Tables Created:**
# MAGIC - `gold.sakt_sequences` - PyKT-format sequences for SAKT
# MAGIC - `gold.wdirt_features` - Clickstream features for WD-IRT
# MAGIC - `gold.skill_vocabulary` - Skill ID to index mapping
# MAGIC - `gold.item_vocabulary` - Item ID to index mapping
# MAGIC 
# MAGIC Next step: Train models in `04_train_sakt` and `05_train_wd_irt` notebooks.

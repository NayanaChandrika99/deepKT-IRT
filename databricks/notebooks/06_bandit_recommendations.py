# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - LinUCB Bandit for Recommendations
# MAGIC 
# MAGIC Implements the LinUCB contextual bandit for adaptive item recommendations.
# MAGIC 
# MAGIC **Input:**
# MAGIC - `gold.sakt_predictions` (student mastery)
# MAGIC - `gold.item_parameters` (item difficulty/discrimination)
# MAGIC - `silver.canonical_learning_events` (historical outcomes)
# MAGIC 
# MAGIC **Output:**
# MAGIC - `gold.bandit_recommendations`
# MAGIC - MLflow registered bandit state

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pyspark.sql import functions as F

CATALOG = "deepkt_irt"
GOLD_SCHEMA = "gold"
SILVER_SCHEMA = "silver"

spark.sql(f"USE CATALOG {CATALOG}")

# Bandit configuration
BANDIT_CONFIG = {
    "n_features": 8,  # Student + item context features
    "alpha": 1.0,  # Exploration parameter
    "exploration_threshold": 0.5,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. LinUCB Implementation
# MAGIC 
# MAGIC Replicates `src/common/bandit.py` for Databricks.

# COMMAND ----------

@dataclass
class StudentContext:
    """Student features for contextual bandit."""
    user_id: str
    mastery: float
    recent_accuracy: float
    recent_speed: float
    help_tendency: float
    skill_gap: float

@dataclass
class ItemArm:
    """Item (arm) in the bandit."""
    item_id: str
    skill: str
    difficulty: float
    discrimination: float = 1.0

@dataclass
class BanditRecommendation:
    """Recommendation with bandit metadata."""
    item: ItemArm
    expected_reward: float
    uncertainty: float
    ucb_score: float
    reason: str
    is_exploration: bool


class LinUCBBandit:
    """
    Linear Upper Confidence Bound (LinUCB) contextual bandit.
    
    UCB Score = E[reward] + alpha * sqrt(uncertainty)
    """
    
    def __init__(self, n_features: int = 8, alpha: float = 1.0, exploration_threshold: float = 0.5):
        self.n_features = n_features
        self.alpha = alpha
        self.exploration_threshold = exploration_threshold
        
        # Model parameters
        self.A = np.eye(n_features)  # Design matrix
        self.b = np.zeros(n_features)  # Response vector
    
    def get_context_vector(self, student: StudentContext, item: ItemArm) -> np.ndarray:
        """Create feature vector from student-item pair."""
        return np.array([
            student.mastery,
            student.recent_accuracy,
            student.recent_speed,
            student.help_tendency,
            student.skill_gap,
            item.difficulty,
            item.discrimination,
            student.mastery - item.difficulty,  # Ability gap
        ])
    
    def predict(self, student: StudentContext, item: ItemArm) -> Tuple[float, float]:
        """Predict expected reward and uncertainty."""
        x = self.get_context_vector(student, item)
        
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        
        expected = float(np.dot(theta, x))
        uncertainty = float(np.sqrt(x @ A_inv @ x))
        
        return expected, uncertainty
    
    def select_best(self, student: StudentContext, items: List[ItemArm]) -> BanditRecommendation:
        """Select best item using UCB."""
        best_score = float('-inf')
        best_rec = None
        
        for item in items:
            expected, uncertainty = self.predict(student, item)
            ucb_score = expected + self.alpha * uncertainty
            
            if ucb_score > best_score:
                best_score = ucb_score
                is_exploration = uncertainty > self.exploration_threshold * abs(expected)
                
                best_rec = BanditRecommendation(
                    item=item,
                    expected_reward=max(0, min(1, expected)),  # Clip to [0, 1]
                    uncertainty=uncertainty,
                    ucb_score=ucb_score,
                    reason=self._generate_reason(expected, uncertainty, is_exploration, item),
                    is_exploration=is_exploration
                )
        
        return best_rec
    
    def update(self, student: StudentContext, item: ItemArm, reward: float):
        """Update model with observed reward."""
        x = self.get_context_vector(student, item)
        self.A += np.outer(x, x)
        self.b += reward * x
    
    def _generate_reason(self, expected: float, uncertainty: float, is_exploration: bool, item: ItemArm) -> str:
        """Generate explanation for recommendation."""
        if is_exploration:
            return f"Exploring {item.skill} (difficulty={item.difficulty:.2f}) - uncertainty {uncertainty:.2f} is high"
        else:
            return f"Exploiting known good match for {item.skill} (expected success={expected:.2f})"
    
    def save_state(self) -> dict:
        """Serialize bandit state for MLflow."""
        return {
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "n_features": self.n_features,
            "alpha": self.alpha,
            "exploration_threshold": self.exploration_threshold
        }
    
    @classmethod
    def load_state(cls, state: dict) -> 'LinUCBBandit':
        """Load bandit from saved state."""
        bandit = cls(
            n_features=state["n_features"],
            alpha=state["alpha"],
            exploration_threshold=state["exploration_threshold"]
        )
        bandit.A = np.array(state["A"])
        bandit.b = np.array(state["b"])
        return bandit

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build Student Contexts from Data

# COMMAND ----------

def build_student_contexts(limit: int = 100) -> List[Tuple[StudentContext, pd.DataFrame]]:
    """
    Build student contexts from SAKT predictions and event history.
    
    Returns list of (StudentContext, user_events) tuples.
    """
    # Get recent mastery from SAKT predictions
    mastery = (
        spark.table(f"{GOLD_SCHEMA}.sakt_predictions")
        .groupBy("user_idx")
        .agg(
            F.avg("predicted").alias("avg_mastery"),
            F.avg("actual").alias("actual_accuracy")
        )
    ).toPandas()
    
    # Get behavioral features from events
    events = spark.table(f"{SILVER_SCHEMA}.canonical_learning_events")
    
    user_features = (
        events
        .groupBy("user_id")
        .agg(
            F.avg(F.when(F.col("latency_ms") < 3000, 1.0).otherwise(0.0)).alias("speed_score"),
            F.avg(F.when(F.col("help_requested"), 1.0).otherwise(0.0)).alias("help_tendency"),
            F.count("*").alias("total_attempts")
        )
        .filter(F.col("total_attempts") >= 10)
        .limit(limit)
    ).toPandas()
    
    contexts = []
    for _, row in user_features.iterrows():
        user_id = row['user_id']
        
        # Find mastery (approximate by matching index)
        user_mastery = 0.5  # Default
        if len(mastery) > 0:
            user_mastery = mastery['avg_mastery'].mean()
        
        ctx = StudentContext(
            user_id=user_id,
            mastery=float(user_mastery),
            recent_accuracy=0.5,  # Will be refined with more data
            recent_speed=float(row['speed_score']),
            help_tendency=float(row['help_tendency']),
            skill_gap=0.0
        )
        
        contexts.append(ctx)
    
    return contexts

# Build contexts
student_contexts = build_student_contexts(limit=50)
print(f"Built {len(student_contexts)} student contexts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build Item Arms from Parameters

# COMMAND ----------

def build_item_arms(skill_filter: Optional[str] = None) -> List[ItemArm]:
    """Build item arms from Gold layer item parameters."""
    
    params = spark.table(f"{GOLD_SCHEMA}.item_parameters")
    items = spark.table(f"{SILVER_SCHEMA}.items")
    
    # Join for skill info
    item_data = (
        params
        .join(items, on="item_id", how="inner")
        .select(
            "item_id",
            F.col("skill_ids").getItem(0).alias("primary_skill"),
            "difficulty",
            "discrimination"
        )
    ).toPandas()
    
    arms = []
    for _, row in item_data.iterrows():
        skill = row['primary_skill'] or "unknown"
        if skill_filter and skill != skill_filter:
            continue
            
        arms.append(ItemArm(
            item_id=row['item_id'],
            skill=skill,
            difficulty=float(row['difficulty']),
            discrimination=float(row['discrimination'])
        ))
    
    return arms

# Build arms
item_arms = build_item_arms()
print(f"Built {len(item_arms)} item arms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Warm-Start Bandit from Historical Data

# COMMAND ----------

def warmstart_bandit(bandit: LinUCBBandit, sample_size: int = 5000) -> int:
    """
    Warm-start bandit from historical outcomes.
    
    Uses past student-item-outcome triples to initialize the model.
    """
    # Sample historical data
    events = (
        spark.table(f"{SILVER_SCHEMA}.canonical_learning_events")
        .sample(fraction=min(1.0, sample_size / 1000000))
        .limit(sample_size)
    )
    
    # Join with item parameters
    params = spark.table(f"{GOLD_SCHEMA}.item_parameters")
    items_df = spark.table(f"{SILVER_SCHEMA}.items")
    
    warmstart_data = (
        events
        .join(params, on="item_id", how="inner")
        .join(items_df.select("item_id", F.col("skill_ids").getItem(0).alias("skill")), on="item_id")
        .select(
            "user_id",
            "item_id",
            "skill",
            "difficulty",
            "discrimination",
            "correct",
            "latency_ms",
            "help_requested"
        )
    ).toPandas()
    
    updates = 0
    for _, row in warmstart_data.iterrows():
        # Create simplified context
        student = StudentContext(
            user_id=row['user_id'],
            mastery=0.5,
            recent_accuracy=0.5,
            recent_speed=0.5 if row['latency_ms'] and row['latency_ms'] < 5000 else 0.3,
            help_tendency=1.0 if row['help_requested'] else 0.0,
            skill_gap=0.0
        )
        
        item = ItemArm(
            item_id=row['item_id'],
            skill=row['skill'] or "unknown",
            difficulty=float(row['difficulty']),
            discrimination=float(row['discrimination'])
        )
        
        bandit.update(student, item, float(row['correct']))
        updates += 1
    
    return updates

# Initialize and warm-start bandit
bandit = LinUCBBandit(**BANDIT_CONFIG)
n_updates = warmstart_bandit(bandit, sample_size=10000)
print(f"Warm-started bandit with {n_updates} historical outcomes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate Recommendations

# COMMAND ----------

def generate_recommendations(bandit: LinUCBBandit, students: List[StudentContext], items: List[ItemArm], top_k: int = 5):
    """Generate recommendations for each student."""
    
    recommendations = []
    
    for student in students:
        # Select top-k items for this student
        student_recs = []
        remaining_items = items.copy()
        
        for rank in range(min(top_k, len(items))):
            if not remaining_items:
                break
                
            rec = bandit.select_best(student, remaining_items)
            if rec:
                student_recs.append({
                    'user_id': student.user_id,
                    'item_id': rec.item.item_id,
                    'skill': rec.item.skill,
                    'rank': rank + 1,
                    'expected_reward': rec.expected_reward,
                    'uncertainty': rec.uncertainty,
                    'ucb_score': rec.ucb_score,
                    'is_exploration': rec.is_exploration,
                    'reason': rec.reason,
                    'student_mastery': student.mastery,
                    'item_difficulty': rec.item.difficulty
                })
                
                # Remove selected item
                remaining_items = [i for i in remaining_items if i.item_id != rec.item.item_id]
        
        recommendations.extend(student_recs)
    
    return recommendations

# Generate recommendations
recs = generate_recommendations(bandit, student_contexts[:20], item_arms[:100], top_k=5)
print(f"Generated {len(recs)} recommendations")

# COMMAND ----------

# Save recommendations to Gold layer
recs_df = spark.createDataFrame(recs)

(
    recs_df
    .withColumn("generated_at", F.current_timestamp())
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{GOLD_SCHEMA}.bandit_recommendations")
)

print("Recommendations saved to Gold layer")
recs_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Bandit State to MLflow

# COMMAND ----------

import json

with mlflow.start_run(run_name="linucb-warmstart"):
    # Log configuration
    mlflow.log_params(BANDIT_CONFIG)
    mlflow.log_param("warmstart_updates", n_updates)
    
    # Save state as artifact
    state = bandit.save_state()
    state_json = json.dumps(state)
    
    # Write to temp file and log
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(state_json)
        temp_path = f.name
    
    mlflow.log_artifact(temp_path, "bandit_state")
    
    # Log metrics
    mlflow.log_metric("total_recommendations", len(recs))
    exploration_rate = sum(1 for r in recs if r['is_exploration']) / len(recs) if recs else 0
    mlflow.log_metric("exploration_rate", exploration_rate)
    
    print(f"Bandit state saved to MLflow. Exploration rate: {exploration_rate:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC LinUCB bandit implementation complete:
# MAGIC - Warm-started from historical data
# MAGIC - Recommendations saved to `gold.bandit_recommendations`
# MAGIC - Bandit state saved to MLflow as artifact
# MAGIC 
# MAGIC Next: Create Vector Search index for RAG explanations in `07_build_vector_index`.

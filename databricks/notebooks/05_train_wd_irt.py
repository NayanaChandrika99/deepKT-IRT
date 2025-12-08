# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Train Wide & Deep IRT Model
# MAGIC 
# MAGIC Trains the Wide & Deep IRT model on Databricks GPU cluster.
# MAGIC 
# MAGIC **Input:** Gold tables
# MAGIC - `gold.wdirt_features`
# MAGIC - `gold.item_vocabulary`
# MAGIC 
# MAGIC **Output:**
# MAGIC - MLflow registered model
# MAGIC - `gold.item_parameters` (difficulty, discrimination, guessing)
# MAGIC - `gold.item_drift` (temporal drift flags)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score
from pyspark.sql import functions as F

# MLflow experiment
EXPERIMENT_NAME = "/Shared/deepkt-irt/wd-irt-training"
mlflow.set_experiment(EXPERIMENT_NAME)

# Catalog configuration
CATALOG = "deepkt_irt"
GOLD_SCHEMA = "gold"
spark.sql(f"USE CATALOG {CATALOG}")

# Model hyperparameters (matching configs/wd_irt_edm.yaml)
CONFIG = {
    "wide_units": 256,
    "deep_units": [512, 256, 128],
    "embedding_dim": 128,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "ability_regularizer": 0.01,
    "batch_size": 256,
    "max_epochs": 30,
    "patience": 5,
    "n_latency_buckets": 10,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data from Delta Lake

# COMMAND ----------

def load_wdirt_data(split: str):
    """Load WD-IRT features from Delta table."""
    df = spark.table(f"{GOLD_SCHEMA}.wdirt_features").filter(f"split = '{split}'")
    return df.toPandas()

train_df = load_wdirt_data("train")
val_df = load_wdirt_data("val")

# Get vocabulary sizes
item_vocab = spark.table(f"{GOLD_SCHEMA}.item_vocabulary")
n_items = item_vocab.count()

print(f"Training records: {len(train_df)}")
print(f"Validation records: {len(val_df)}")
print(f"Number of items: {n_items}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. PyTorch Dataset

# COMMAND ----------

class WDIRTDataset(Dataset):
    """Dataset for Wide & Deep IRT model."""
    
    def __init__(self, df, item_vocab_df):
        # Create item index mapping
        self.item_to_idx = dict(zip(item_vocab_df['item_id'], item_vocab_df['item_idx']))
        
        self.data = []
        for _, row in df.iterrows():
            item_idx = self.item_to_idx.get(row['item_id'], 0)
            
            self.data.append({
                'item_idx': item_idx,
                'latency_bucket': int(row['latency_bucket']),
                'recency': float(row['recency_normalized']),
                'item_success_rate': float(row['item_success_rate']),
                'help_tendency': float(row['help_tendency']),
                'recent_accuracy': float(row['recent_accuracy']),
                'correct': int(row['correct'])
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'item_idx': torch.tensor(item['item_idx'], dtype=torch.long),
            'latency_bucket': torch.tensor(item['latency_bucket'], dtype=torch.long),
            'recency': torch.tensor(item['recency'], dtype=torch.float32),
            'item_success_rate': torch.tensor(item['item_success_rate'], dtype=torch.float32),
            'help_tendency': torch.tensor(item['help_tendency'], dtype=torch.float32),
            'recent_accuracy': torch.tensor(item['recent_accuracy'], dtype=torch.float32),
            'correct': torch.tensor(item['correct'], dtype=torch.float32)
        }

# Load item vocabulary
item_vocab_pd = item_vocab.toPandas()

# Create datasets
train_dataset = WDIRTDataset(train_df, item_vocab_pd)
val_dataset = WDIRTDataset(val_df, item_vocab_pd)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Wide & Deep IRT Model

# COMMAND ----------

class WideDeepIRT(pl.LightningModule):
    """
    Wide & Deep IRT model.
    
    Wide component: Traditional IRT (item difficulty, discrimination, guessing)
    Deep component: Clickstream feature encoding for ability estimation
    """
    
    def __init__(self, n_items, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Deep component: Feature embeddings and MLP
        embed_dim = config['embedding_dim']
        
        self.latency_emb = nn.Embedding(config['n_latency_buckets'] + 1, embed_dim, padding_idx=0)
        self.recency_proj = nn.Linear(1, embed_dim)
        self.success_rate_proj = nn.Linear(1, embed_dim)
        self.help_proj = nn.Linear(1, embed_dim)
        self.accuracy_proj = nn.Linear(1, embed_dim)
        
        # Deep MLP
        deep_input_dim = embed_dim * 5
        layers = []
        in_dim = deep_input_dim
        for units in config['deep_units']:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config['dropout']))
            in_dim = units
        layers.append(nn.Linear(in_dim, 1))  # Output: ability
        self.deep_mlp = nn.Sequential(*layers)
        
        # Wide component: IRT parameters
        self.item_difficulty = nn.Parameter(torch.zeros(n_items + 1))
        self.item_discrimination = nn.Parameter(torch.ones(n_items + 1))
        self.item_guessing = nn.Parameter(torch.zeros(n_items + 1))
        
        self.criterion = nn.BCELoss()
        
    def forward(self, batch):
        """Forward pass returning probability and ability."""
        
        # Deep component: estimate ability from features
        lat_emb = self.latency_emb(batch['latency_bucket'])
        rec_emb = self.recency_proj(batch['recency'].unsqueeze(-1))
        suc_emb = self.success_rate_proj(batch['item_success_rate'].unsqueeze(-1))
        help_emb = self.help_proj(batch['help_tendency'].unsqueeze(-1))
        acc_emb = self.accuracy_proj(batch['recent_accuracy'].unsqueeze(-1))
        
        deep_input = torch.cat([lat_emb, rec_emb, suc_emb, help_emb, acc_emb], dim=-1)
        ability = self.deep_mlp(deep_input).squeeze(-1)
        
        # Wide component: 3PL IRT model
        item_idx = batch['item_idx']
        beta = self.item_difficulty[item_idx]
        alpha = torch.abs(self.item_discrimination[item_idx])  # Ensure positive
        gamma = torch.sigmoid(self.item_guessing[item_idx])  # Ensure [0, 1]
        
        # 3PL formula: P(correct) = gamma + (1 - gamma) * sigmoid(alpha * (ability - beta))
        logits = alpha * (ability - beta)
        base_prob = torch.sigmoid(logits)
        prob = gamma + (1 - gamma) * base_prob
        
        return prob, ability
    
    def training_step(self, batch, batch_idx):
        prob, ability = self(batch)
        target = batch['correct']
        
        loss = self.criterion(prob, target)
        # Regularize ability to prevent explosion
        loss = loss + self.config['ability_regularizer'] * ability.pow(2).mean()
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        prob, _ = self(batch)
        target = batch['correct']
        
        loss = self.criterion(prob, target)
        self.log('val_loss', loss, prog_bar=True)
        
        return {'preds': prob.cpu(), 'targets': target.cpu()}
    
    def validation_epoch_end(self, outputs):
        all_preds = torch.cat([o['preds'] for o in outputs])
        all_targets = torch.cat([o['targets'] for o in outputs])
        
        auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())
        self.log('val_auc', auc, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Training with MLflow

# COMMAND ----------

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

# Initialize model
model = WideDeepIRT(n_items, CONFIG)

# MLflow logger
mlflow_logger = MLFlowLogger(
    experiment_name=EXPERIMENT_NAME,
    tracking_uri="databricks"
)

# Callbacks
early_stop = EarlyStopping(monitor='val_auc', patience=CONFIG['patience'], mode='max')
checkpoint = ModelCheckpoint(monitor='val_auc', mode='max', save_top_k=1)

# Trainer
trainer = pl.Trainer(
    max_epochs=CONFIG['max_epochs'],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    callbacks=[early_stop, checkpoint],
    logger=mlflow_logger,
    enable_progress_bar=True
)

# Train
trainer.fit(model, train_loader, val_loader)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Extract and Export Item Parameters

# COMMAND ----------

def extract_item_parameters(model, item_vocab_df):
    """Extract IRT parameters for all items."""
    
    model.eval()
    
    # Get parameters from model
    difficulty = model.item_difficulty.detach().cpu().numpy()
    discrimination = torch.abs(model.item_discrimination).detach().cpu().numpy()
    guessing = torch.sigmoid(model.item_guessing).detach().cpu().numpy()
    
    # Create DataFrame
    params_data = []
    for _, row in item_vocab_df.iterrows():
        idx = int(row['item_idx'])
        params_data.append({
            'item_id': row['item_id'],
            'item_idx': idx,
            'difficulty': float(difficulty[idx]),
            'discrimination': float(discrimination[idx]),
            'guessing': float(guessing[idx])
        })
    
    return spark.createDataFrame(params_data)

# Extract parameters
item_params = extract_item_parameters(model, item_vocab_pd)
print(f"Item parameters extracted: {item_params.count()}")
item_params.show(5)

# COMMAND ----------

# Save to Gold layer
(
    item_params
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{GOLD_SCHEMA}.item_parameters")
)

print("Item parameters saved to Gold layer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Detect Item Drift
# MAGIC 
# MAGIC Compare recent vs historical difficulty to flag drifting items.

# COMMAND ----------

def detect_item_drift(events_table, item_params_df, lookback_days=30):
    """
    Detect items with significant difficulty drift.
    
    Compares recent success rate to estimated difficulty.
    """
    events = spark.table(f"silver.canonical_learning_events")
    
    # Calculate recent success rate per item
    cutoff_date = F.date_sub(F.current_date(), lookback_days)
    
    recent_stats = (
        events
        .filter(F.col("timestamp") >= cutoff_date)
        .groupBy("item_id")
        .agg(
            F.avg("correct").alias("recent_success_rate"),
            F.count("*").alias("recent_attempts")
        )
        .filter(F.col("recent_attempts") >= 30)  # Sufficient sample
    )
    
    # Join with item parameters
    drift_analysis = recent_stats.join(
        item_params_df.select("item_id", "difficulty"),
        on="item_id",
        how="inner"
    )
    
    # Calculate implied difficulty from success rate
    # Simple approximation: implied_difficulty ~ -log(success_rate / (1 - success_rate))
    drift_analysis = drift_analysis.withColumn(
        "implied_difficulty",
        -F.log(F.col("recent_success_rate") / (1 - F.col("recent_success_rate")))
    )
    
    # Calculate drift
    drift_analysis = drift_analysis.withColumn(
        "drift_score",
        F.abs(F.col("implied_difficulty") - F.col("difficulty"))
    )
    
    # Flag significant drift (threshold = 0.5)
    drift_analysis = drift_analysis.withColumn(
        "has_drift",
        F.col("drift_score") > 0.5
    )
    
    return drift_analysis.select(
        "item_id",
        "difficulty",
        "recent_success_rate",
        "implied_difficulty", 
        "drift_score",
        "has_drift",
        "recent_attempts"
    )

# COMMAND ----------

# Calculate drift
item_drift = detect_item_drift("silver.canonical_learning_events", item_params)

(
    item_drift
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{GOLD_SCHEMA}.item_drift")
)

drifting_count = item_drift.filter(F.col("has_drift") == True).count()
print(f"Items with significant drift: {drifting_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Register Model in MLflow

# COMMAND ----------

# Register best model
with mlflow.start_run(run_id=mlflow_logger.run_id):
    mlflow.pytorch.log_model(model, "wdirt_model")
    
    model_uri = f"runs:/{mlflow_logger.run_id}/wdirt_model"
    mlflow.register_model(model_uri, "wide-deep-irt")

print("Model registered as 'wide-deep-irt' in MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC Wide & Deep IRT training complete:
# MAGIC - Model registered in MLflow Model Registry as `wide-deep-irt`
# MAGIC - Item parameters saved to `gold.item_parameters`
# MAGIC - Item drift analysis saved to `gold.item_drift`

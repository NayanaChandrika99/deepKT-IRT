# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Train SAKT Model
# MAGIC 
# MAGIC Trains the SAKT (Self-Attentive Knowledge Tracing) model on Databricks GPU cluster.
# MAGIC 
# MAGIC **Input:** Gold tables
# MAGIC - `gold.sakt_sequences`
# MAGIC - `gold.skill_vocabulary`
# MAGIC 
# MAGIC **Output:**
# MAGIC - MLflow registered model
# MAGIC - `gold.sakt_predictions`
# MAGIC - `gold.sakt_student_state`
# MAGIC - `gold.sakt_attention`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score

# MLflow experiment
EXPERIMENT_NAME = "/Shared/deepkt-irt/sakt-training"
mlflow.set_experiment(EXPERIMENT_NAME)

# Catalog configuration
CATALOG = "deepkt_irt"
GOLD_SCHEMA = "gold"
spark.sql(f"USE CATALOG {CATALOG}")

# Model hyperparameters (matching configs/sakt_edm.yaml)
CONFIG = {
    "seq_len": 200,
    "emb_size": 64,
    "num_attn_heads": 4,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 5,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data from Delta Lake

# COMMAND ----------

def load_sequences_from_delta(split: str):
    """Load SAKT sequences from Delta table."""
    df = spark.table(f"{GOLD_SCHEMA}.sakt_sequences").filter(f"split = '{split}'")
    return df.toPandas()

train_df = load_sequences_from_delta("train")
val_df = load_sequences_from_delta("val")

# Get vocabulary size
skill_vocab = spark.table(f"{GOLD_SCHEMA}.skill_vocabulary")
n_skills = skill_vocab.count()

print(f"Training sequences: {len(train_df)}")
print(f"Validation sequences: {len(val_df)}")
print(f"Number of skills: {n_skills}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. PyTorch Dataset

# COMMAND ----------

class SAKTDataset(Dataset):
    """Dataset for SAKT model training."""
    
    def __init__(self, df, seq_len, n_skills):
        self.seq_len = seq_len
        self.n_skills = n_skills
        self.data = []
        
        for _, row in df.iterrows():
            questions = row['questions']
            responses = row['responses']
            
            # Pad or truncate
            length = min(len(questions), seq_len)
            
            q = np.zeros(seq_len, dtype=np.int64)
            r = np.zeros(seq_len, dtype=np.int64)
            mask = np.zeros(seq_len, dtype=np.float32)
            
            q[:length] = questions[:length]
            r[:length] = responses[:length]
            mask[:length] = 1.0
            
            self.data.append({
                'questions': q,
                'responses': r,
                'mask': mask,
                'length': length
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['questions']),
            torch.tensor(item['responses']),
            torch.tensor(item['mask']),
            item['length']
        )

# Create datasets
train_dataset = SAKTDataset(train_df, CONFIG['seq_len'], n_skills)
val_dataset = SAKTDataset(val_df, CONFIG['seq_len'], n_skills)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SAKT Model Architecture

# COMMAND ----------

class SAKT(nn.Module):
    """
    Self-Attentive Knowledge Tracing model.
    
    Architecture matches the pyKT SAKT implementation.
    """
    
    def __init__(self, n_skills, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.n_skills = n_skills
        self.emb_size = emb_size
        self.seq_len = seq_len
        
        # Embeddings: interaction = (skill, response) pair
        # skill embedding + response embedding
        self.skill_emb = nn.Embedding(n_skills + 1, emb_size, padding_idx=0)
        self.response_emb = nn.Embedding(2, emb_size)  # 0=incorrect, 1=correct
        
        # Position embedding
        self.pos_emb = nn.Embedding(seq_len, emb_size)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 4, emb_size),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        
        # Output projection
        self.fc_out = nn.Linear(emb_size, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, questions, responses, mask=None):
        """
        Args:
            questions: (batch, seq_len) skill indices
            responses: (batch, seq_len) correct/incorrect labels
            mask: (batch, seq_len) attention mask
            
        Returns:
            predictions: (batch, seq_len) probability of correct response
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = questions.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=questions.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed skills and responses
        skill_embed = self.skill_emb(questions)
        resp_embed = self.response_emb(responses)
        pos_embed = self.pos_emb(positions)
        
        # Combine embeddings: x = skill + response + position
        x = skill_embed + resp_embed + pos_embed
        x = self.dropout(x)
        
        # Create causal mask (can't attend to future)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        
        # Key padding mask from input mask
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)
        
        # Self-attention with attention weights output
        attn_out, attn_weights = self.attn(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # Get per-head weights
        )
        
        # Residual + layer norm
        x = self.ln1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        # Output prediction
        logits = self.fc_out(x).squeeze(-1)
        predictions = torch.sigmoid(logits)
        
        return predictions, attn_weights

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Training Loop with MLflow

# COMMAND ----------

def train_sakt_model(model, train_loader, val_loader, config, device):
    """Train SAKT model with early stopping and MLflow logging."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss(reduction='none')
    
    best_auc = 0
    patience_counter = 0
    best_state = None
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(config)
        
        for epoch in range(config['max_epochs']):
            # Training phase
            model.train()
            train_losses = []
            
            for questions, responses, mask, lengths in train_loader:
                questions = questions.to(device)
                responses = responses.to(device)
                mask = mask.to(device)
                
                optimizer.zero_grad()
                
                # Shift responses for prediction (predict next response from current state)
                target = responses[:, 1:].float()
                pred_mask = mask[:, 1:]
                
                predictions, _ = model(questions[:, :-1], responses[:, :-1], mask[:, :-1])
                
                # Masked loss
                loss = criterion(predictions, target)
                loss = (loss * pred_mask).sum() / pred_mask.sum()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation phase
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for questions, responses, mask, lengths in val_loader:
                    questions = questions.to(device)
                    responses = responses.to(device)
                    mask = mask.to(device)
                    
                    predictions, _ = model(questions[:, :-1], responses[:, :-1], mask[:, :-1])
                    
                    pred_mask = mask[:, 1:]
                    
                    # Flatten and filter masked positions
                    for i in range(predictions.shape[0]):
                        length = int(pred_mask[i].sum().item())
                        all_preds.extend(predictions[i, :length].cpu().numpy())
                        all_targets.extend(responses[i, 1:length+1].cpu().numpy())
            
            val_auc = roc_auc_score(all_targets, all_preds)
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_auc": val_auc,
                "epoch": epoch
            }, step=epoch)
            
            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_auc={val_auc:.4f}")
            
            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and log
        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_auc", best_auc)
        mlflow.pytorch.log_model(model, "sakt_model")
        
        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/sakt_model"
        mlflow.register_model(model_uri, "sakt-knowledge-tracing")
        
    return model, best_auc

# COMMAND ----------

# Initialize and train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = SAKT(
    n_skills=n_skills,
    emb_size=CONFIG['emb_size'],
    num_heads=CONFIG['num_attn_heads'],
    seq_len=CONFIG['seq_len'],
    dropout=CONFIG['dropout']
)

trained_model, best_auc = train_sakt_model(model, train_loader, val_loader, CONFIG, device)
print(f"Training complete. Best validation AUC: {best_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Export Predictions and Attention

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType

def export_sakt_artifacts(model, dataset, split_name, device):
    """Export predictions, student states, and attention weights."""
    
    model.eval()
    
    predictions_data = []
    attention_data = []
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (questions, responses, mask, lengths) in enumerate(loader):
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)
            
            preds, attn_weights = model(questions, responses, mask)
            
            for i in range(questions.shape[0]):
                user_idx = batch_idx * 32 + i
                length = lengths[i]
                
                # Predictions
                for t in range(length - 1):
                    predictions_data.append({
                        "user_idx": user_idx,
                        "position": t + 1,
                        "skill_id": int(questions[i, t].item()),
                        "actual": int(responses[i, t + 1].item()),
                        "predicted": float(preds[i, t].item()),
                        "split": split_name
                    })
                
                # Attention weights (average across heads)
                avg_attn = attn_weights[i].mean(dim=0).cpu().numpy()
                attention_data.append({
                    "user_idx": user_idx,
                    "attention_matrix": avg_attn[:length, :length].tolist(),
                    "seq_length": length,
                    "split": split_name
                })
    
    return predictions_data, attention_data

# COMMAND ----------

# Export for validation set
val_preds, val_attn = export_sakt_artifacts(trained_model, val_dataset, "val", device)

# Convert to Spark DataFrames
preds_df = spark.createDataFrame(val_preds)
attn_df = spark.createDataFrame(val_attn)

print(f"Predictions: {preds_df.count()}, Attention records: {attn_df.count()}")

# COMMAND ----------

# Save to Gold layer
(
    preds_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{GOLD_SCHEMA}.sakt_predictions")
)

(
    attn_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{GOLD_SCHEMA}.sakt_attention")
)

print("Exported SAKT artifacts to Gold layer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC SAKT training complete:
# MAGIC - Model registered in MLflow Model Registry as `sakt-knowledge-tracing`
# MAGIC - Predictions saved to `gold.sakt_predictions`
# MAGIC - Attention weights saved to `gold.sakt_attention`
# MAGIC 
# MAGIC Best validation AUC: See MLflow experiment for details.

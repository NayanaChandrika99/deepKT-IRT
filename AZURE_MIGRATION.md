# ABOUTME: README for Azure/Databricks migration scaffolding.
# ABOUTME: Documents the new infrastructure and deployment instructions.

# deepKT+IRT: Azure/Databricks Migration

This document describes the Azure/Databricks migration scaffold for the deepKT+IRT learning analytics system.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Azure Data Platform                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────────────────────────────────┐    │
│  │ Azure Data  │    │           Databricks Lakehouse          │    │
│  │   Factory   │───▶│  ┌────────┐  ┌────────┐  ┌────────┐    │    │
│  │ (Orchestr.) │    │  │ Bronze │  │ Silver │  │  Gold  │    │    │
│  └─────────────┘    │  │  Layer │─▶│  Layer │─▶│  Layer │    │    │
│                     │  └────────┘  └────────┘  └────────┘    │    │
│  ┌─────────────┐    │       │                       │        │    │
│  │   ADLS      │    │       ▼                       ▼        │    │
│  │   Gen2      │◀───│  ┌─────────┐           ┌──────────┐    │    │
│  │ (Storage)   │    │  │ MLflow  │           │  Vector  │    │    │
│  └─────────────┘    │  │Registry │           │  Search  │    │    │
│                     │  └─────────┘           └──────────┘    │    │
│  ┌─────────────┐    └────────────────────────────────────────┘    │
│  │Azure OpenAI │                      │                           │
│  │  (GPT-4o)   │◀─────────────────────┘                           │
│  └─────────────┘                                                   │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐                                │
│  │  Tableau    │    │  Power BI   │                                │
│  │     BI      │    │     BI      │                                │
│  └─────────────┘    └─────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
deepKT+IRT/
├── terraform/                    # Infrastructure as Code
│   ├── main.tf                  # Resource group, storage
│   ├── databricks.tf            # Databricks workspace, clusters
│   ├── openai.tf                # Azure OpenAI deployments
│   ├── adf.tf                   # Data Factory configuration
│   ├── variables.tf             # Input variables
│   └── terraform.tfvars.example # Example variable values
│
├── databricks/
│   ├── notebooks/               # Databricks notebooks
│   │   ├── 01_ingest_bronze.py      # Raw data ingestion
│   │   ├── 02_transform_silver.py   # Canonical transformation
│   │   ├── 03_feature_engineering.py # Feature extraction
│   │   ├── 04_train_sakt.py         # SAKT model training
│   │   ├── 05_train_wd_irt.py       # WD-IRT model training
│   │   ├── 06_bandit_recommendations.py # LinUCB bandit
│   │   ├── 07_build_vector_index.py # Vector Search index
│   │   └── 08_rag_explainer.py      # RAG explanations
│   │
│   └── jobs/
│       └── job_definitions.yml  # Scheduled job configs
│
├── adf/
│   └── pipelines/               # Azure Data Factory pipelines
│       ├── daily_ingestion.json
│       └── weekly_training.json
│
└── src/common/
    └── azure_openai_explainability.py  # Azure OpenAI wrapper
```

## Deployment Guide

### Prerequisites

1. **Azure Subscription** with permissions to create:
   - Resource groups
   - Storage accounts (ADLS Gen2)
   - Databricks workspaces
   - Cognitive Services (OpenAI)
   - Data Factory

2. **Local Tools**:
   ```bash
   # Install Terraform
   brew install terraform
   
   # Install Azure CLI
   brew install azure-cli
   
   # Install Databricks CLI
   pip install databricks-cli
   ```

### Step 1: Deploy Infrastructure

```bash
cd terraform

# Copy and configure variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy infrastructure
terraform apply
```

### Step 2: Configure Databricks

```bash
# Set Databricks CLI config
databricks configure --host https://<workspace-url>

# Upload notebooks
databricks workspace import_dir databricks/notebooks /Shared/deepkt-irt/notebooks

# Create scheduled jobs
databricks jobs create --json-file databricks/jobs/job_definitions.yml
```

### Step 3: Configure Azure Data Factory

```bash
# Deploy ADF pipelines (via Azure Portal or ARM templates)
az datafactory pipeline create \
  --resource-group deepkt-irt-dev-rg \
  --factory-name deepkt-irt-dev-adf \
  --name daily_ingestion \
  --pipeline @adf/pipelines/daily_ingestion.json
```

### Step 4: Upload Initial Data

```bash
# Upload CSV files to Bronze layer
az storage blob upload-batch \
  --destination bronze/uploads \
  --source data/raw \
  --account-name deepktirtdevlake
```

## Component Mapping

| Original Component | Azure/Databricks Equivalent |
|-------------------|----------------------------|
| `data_pipeline.py` | `01_ingest_bronze.py` + `02_transform_silver.py` |
| `src/wd_irt/features.py` | `03_feature_engineering.py` |
| `src/sakt_kt/train.py` | `04_train_sakt.py` |
| `src/wd_irt/train.py` | `05_train_wd_irt.py` |
| `src/common/bandit.py` | `06_bandit_recommendations.py` |
| `llm_explainability.py` | `azure_openai_explainability.py` + `08_rag_explainer.py` |
| Parquet files | Delta Lake tables (Unity Catalog) |
| Local model files | MLflow Model Registry |
| GitHub Pages dashboard | Tableau / Power BI |

## Data Lake Schema

### Bronze Layer (Raw)
- `bronze.raw_action_logs`
- `bronze.raw_assignment_details`
- `bronze.raw_problem_details`

### Silver Layer (Canonical)
- `silver.canonical_learning_events`
- `silver.users`
- `silver.items`
- `silver.skills`
- `silver.user_splits`

### Gold Layer (Analytics)
- `gold.sakt_sequences`
- `gold.sakt_predictions`
- `gold.sakt_attention`
- `gold.wdirt_features`
- `gold.item_parameters`
- `gold.item_drift`
- `gold.skill_vocabulary`
- `gold.item_vocabulary`
- `gold.bandit_recommendations`
- `gold.content_library`
- `gold.content_embeddings`
- `gold.rag_explanations`

## Scheduled Jobs

| Job | Schedule | Purpose |
|-----|----------|---------|
| Daily Ingestion | 2 AM UTC daily | Bronze → Silver → Gold pipeline |
| Weekly Training | 4 AM UTC Sunday | SAKT + WD-IRT model training |

## Cost Estimates

| Resource | Monthly Estimate |
|----------|-----------------|
| Databricks (Standard) | $500 - $2,000 |
| Databricks GPU (training) | $500 - $1,500 |
| ADLS Gen2 Storage | $20 - $50 |
| Azure OpenAI | $30 - $100 |
| Azure Data Factory | $50 - $200 |
| **Total** | **$1,100 - $3,850** |

*Note: Costs vary based on usage patterns. Use job clusters that auto-terminate to reduce expenses.*

## Development Workflow

### Local Development
The original `src/` code continues to work for local development and testing.

```bash
# Run locally
python scripts/demo_trace.py --student-id test --topic algebra

# Run tests
pytest tests/ -v
```

### Databricks Development
Use Databricks Repos to sync notebooks with Git.

```bash
# Link to Git repo
databricks repos create \
  --url https://github.com/your-org/deepKT-IRT \
  --path /Repos/your-email/deepKT-IRT
```

## Monitoring

### MLflow Experiments
- `/Shared/deepkt-irt/sakt-training`
- `/Shared/deepkt-irt/wd-irt-training`

### Key Metrics to Track
- `val_auc` - Model validation AUC
- `tokens_used` - Azure OpenAI token consumption
- Pipeline run times and success rates

## Security Considerations

1. **Secrets Management**: All credentials stored in Databricks Secret Scopes
2. **Network**: Consider VNet integration for private endpoints
3. **Data Access**: Unity Catalog for fine-grained access control
4. **Compliance**: Azure OpenAI provides enterprise-compliant LLM access

## Next Steps

1. [ ] Configure Unity Catalog metastore
2. [ ] Set up VNet for production
3. [ ] Configure Tableau/Power BI connections
4. [ ] Implement real-time streaming (optional)
5. [ ] Set up alerting and monitoring dashboards

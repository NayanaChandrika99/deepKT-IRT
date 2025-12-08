# ABOUTME: Terraform configuration for Azure Databricks workspace.
# ABOUTME: Creates workspace, clusters, and Unity Catalog configuration.

# Databricks Workspace
resource "azurerm_databricks_workspace" "main" {
  name                        = "${local.resource_prefix}-dbx"
  resource_group_name         = azurerm_resource_group.main.name
  location                    = azurerm_resource_group.main.location
  sku                         = "premium"  # Required for Unity Catalog
  managed_resource_group_name = "${local.resource_prefix}-dbx-managed-rg"
  
  tags = local.tags
}

# Configure Databricks provider after workspace is created
provider "databricks" {
  host                        = azurerm_databricks_workspace.main.workspace_url
  azure_workspace_resource_id = azurerm_databricks_workspace.main.id
}

# Cluster for data processing (auto-terminating)
resource "databricks_cluster" "data_processing" {
  cluster_name            = "data-processing-cluster"
  spark_version           = "13.3.x-scala2.12"
  node_type_id            = "Standard_DS3_v2"
  autotermination_minutes = 30
  
  autoscale {
    min_workers = 1
    max_workers = 4
  }
  
  spark_conf = {
    "spark.databricks.delta.preview.enabled" = "true"
  }
  
  custom_tags = local.tags
  
  depends_on = [azurerm_databricks_workspace.main]
}

# GPU cluster for ML training (auto-terminating, job cluster preferred)
resource "databricks_cluster" "ml_training" {
  cluster_name            = "ml-training-gpu"
  spark_version           = "13.3.x-gpu-ml-scala2.12"
  node_type_id            = "Standard_NC6s_v3"  # NVIDIA V100 GPU
  autotermination_minutes = 20
  num_workers             = 1  # Single node for SAKT/WD-IRT training
  
  spark_conf = {
    "spark.databricks.delta.preview.enabled" = "true"
  }
  
  custom_tags = merge(local.tags, {
    Purpose = "ML Training"
  })
  
  depends_on = [azurerm_databricks_workspace.main]
}

# Secret scope for credentials
resource "databricks_secret_scope" "main" {
  name = "deepkt-secrets"
  
  depends_on = [azurerm_databricks_workspace.main]
}

# Unity Catalog metastore assignment (requires existing metastore)
# Uncomment when Unity Catalog metastore is available
# resource "databricks_metastore_assignment" "main" {
#   metastore_id = var.unity_catalog_metastore_id
#   workspace_id = azurerm_databricks_workspace.main.workspace_id
# }

# Catalog for deepKT-IRT
resource "databricks_catalog" "deepkt" {
  name    = "deepkt_irt"
  comment = "Catalog for deepKT+IRT learning analytics"
  
  depends_on = [azurerm_databricks_workspace.main]
}

# Schemas for medallion architecture
resource "databricks_schema" "bronze" {
  catalog_name = databricks_catalog.deepkt.name
  name         = "bronze"
  comment      = "Raw ingested data"
}

resource "databricks_schema" "silver" {
  catalog_name = databricks_catalog.deepkt.name
  name         = "silver"
  comment      = "Cleaned canonical data"
}

resource "databricks_schema" "gold" {
  catalog_name = databricks_catalog.deepkt.name
  name         = "gold"
  comment      = "Analytics-ready aggregates"
}

# MLflow experiment for model tracking
resource "databricks_mlflow_experiment" "sakt" {
  name        = "/Shared/deepkt-irt/sakt-training"
  description = "SAKT knowledge tracing experiments"
  
  depends_on = [azurerm_databricks_workspace.main]
}

resource "databricks_mlflow_experiment" "wdirt" {
  name        = "/Shared/deepkt-irt/wd-irt-training"
  description = "Wide & Deep IRT experiments"
  
  depends_on = [azurerm_databricks_workspace.main]
}

# Outputs
output "databricks_workspace_url" {
  value = azurerm_databricks_workspace.main.workspace_url
}

output "databricks_workspace_id" {
  value = azurerm_databricks_workspace.main.workspace_id
}

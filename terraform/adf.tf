# ABOUTME: Terraform configuration for Azure Data Factory.
# ABOUTME: Creates ADF instance with linked services for data sources.

# Azure Data Factory
resource "azurerm_data_factory" "main" {
  name                = "${local.resource_prefix}-adf"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  identity {
    type = "SystemAssigned"
  }
  
  tags = local.tags
}

# Linked Service: ADLS Gen2
resource "azurerm_data_factory_linked_service_data_lake_storage_gen2" "datalake" {
  name                 = "ADLSGen2LinkedService"
  data_factory_id      = azurerm_data_factory.main.id
  url                  = azurerm_storage_account.datalake.primary_dfs_endpoint
  use_managed_identity = true
}

# Grant ADF access to storage
resource "azurerm_role_assignment" "adf_storage_contributor" {
  scope                = azurerm_storage_account.datalake.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_data_factory.main.identity[0].principal_id
}

# Linked Service: Azure Databricks
# NOTE: Uncomment after Databricks clusters are created
# resource "azurerm_data_factory_linked_service_azure_databricks" "main" {
#   name            = "DatabricksLinkedService"
#   data_factory_id = azurerm_data_factory.main.id
#   description     = "Linked service to Databricks workspace"
#   
#   adb_domain = "https://${azurerm_databricks_workspace.main.workspace_url}"
#   
#   # Use existing cluster
#   existing_cluster_id = databricks_cluster.data_processing.id
#   
#   # Use MSI for authentication
#   msi_work_space_resource_id = azurerm_databricks_workspace.main.id
# }

# NOTE: Additional linked services for source systems should be configured
# based on your specific requirements. Examples below are placeholders.

# Linked Service: SQL Server (for legacy data warehouse)
# Uncomment and configure when SQL Server details are available
# resource "azurerm_data_factory_linked_service_sql_server" "legacy_dwh" {
#   name              = "LegacySQLServerLinkedService"
#   data_factory_id   = azurerm_data_factory.main.id
#   connection_string = "Server=${var.sql_server_host};Database=${var.sql_server_db};User Id=${var.sql_server_user};Password=${var.sql_server_password};"
# }

# Linked Service: REST API (for web app data)
# resource "azurerm_data_factory_linked_service_rest" "webapp_api" {
#   name            = "WebAppAPILinkedService"
#   data_factory_id = azurerm_data_factory.main.id
#   url             = var.webapp_api_url
#   authentication  = "Anonymous"  # Or OAuth2, etc.
# }

# Pipeline trigger - daily at 2 AM UTC
# NOTE: Uncomment when ADF pipelines are created
# resource "azurerm_data_factory_trigger_schedule" "daily_ingestion" {
#   name            = "daily-ingestion-trigger"
#   data_factory_id = azurerm_data_factory.main.id
#   
#   interval  = 1
#   frequency = "Day"
#   
#   schedule {
#     hours   = [2]
#     minutes = [0]
#   }
#   
#   activated = false  # Set to true in production
# }

# Outputs
output "data_factory_name" {
  value = azurerm_data_factory.main.name
}

output "data_factory_id" {
  value = azurerm_data_factory.main.id
}

# ABOUTME: Terraform configuration for Azure OpenAI Service.
# ABOUTME: Provisions cognitive services account with GPT-4o deployment.

# Azure Cognitive Services account for OpenAI
resource "azurerm_cognitive_account" "openai" {
  name                = "${local.resource_prefix}-openai"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "OpenAI"
  sku_name            = "S0"
  
  custom_subdomain_name = "${local.resource_prefix}-openai"
  
  tags = local.tags
}

# GPT-4o deployment for explanations
resource "azurerm_cognitive_deployment" "gpt4o" {
  name                 = "gpt-4o"
  cognitive_account_id = azurerm_cognitive_account.openai.id
  
  model {
    format  = "OpenAI"
    name    = "gpt-4o"
    version = "2024-05-13"
  }
  
  scale {
    type     = "Standard"
    capacity = 10  # 10K tokens per minute
  }
}

# GPT-3.5 Turbo for cost-effective explanations
# NOTE: Skipping - model versions keep getting deprecated. Use GPT-4o instead.
# resource "azurerm_cognitive_deployment" "gpt35" {
#   name                 = "gpt-35-turbo"
#   cognitive_account_id = azurerm_cognitive_account.openai.id
#   
#   model {
#     format  = "OpenAI"
#     name    = "gpt-35-turbo"
#     version = "0125"
#   }
#   
#   scale {
#     type     = "Standard"
#     capacity = 30
#   }
# }

# Text embedding model for vector search
resource "azurerm_cognitive_deployment" "embedding" {
  name                 = "text-embedding-ada-002"
  cognitive_account_id = azurerm_cognitive_account.openai.id
  
  model {
    format  = "OpenAI"
    name    = "text-embedding-ada-002"
    version = "2"
  }
  
  scale {
    type     = "Standard"
    capacity = 30
  }
}

# Store OpenAI key in Databricks secret scope
# NOTE: Create these manually via Databricks CLI after workspace is configured
# resource "databricks_secret" "openai_key" {
#   key          = "azure-openai-key"
#   string_value = azurerm_cognitive_account.openai.primary_access_key
#   scope        = databricks_secret_scope.main.name
#   
#   depends_on = [databricks_secret_scope.main]
# }
# 
# resource "databricks_secret" "openai_endpoint" {
#   key          = "azure-openai-endpoint"
#   string_value = azurerm_cognitive_account.openai.endpoint
#   scope        = databricks_secret_scope.main.name
#   
#   depends_on = [databricks_secret_scope.main]
# }

# Outputs
output "openai_endpoint" {
  value     = azurerm_cognitive_account.openai.endpoint
  sensitive = true
}

output "openai_key" {
  value     = azurerm_cognitive_account.openai.primary_access_key
  sensitive = true
}

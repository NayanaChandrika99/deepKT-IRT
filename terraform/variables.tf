# ABOUTME: Terraform variable definitions for Azure infrastructure.
# ABOUTME: Centralizes configurable parameters for the deployment.

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "location" {
  description = "Azure region for resource deployment"
  type        = string
  default     = "eastus2"
}

variable "project_name" {
  description = "Project identifier used in resource naming"
  type        = string
  default     = "deepkt-irt"
}

# Unity Catalog (optional - requires existing metastore)
variable "unity_catalog_metastore_id" {
  description = "ID of existing Unity Catalog metastore"
  type        = string
  default     = ""
}

# SQL Server connection (for legacy DWH migration)
variable "sql_server_host" {
  description = "Legacy SQL Server hostname"
  type        = string
  default     = ""
}

variable "sql_server_db" {
  description = "Legacy SQL Server database name"
  type        = string
  default     = ""
}

variable "sql_server_user" {
  description = "Legacy SQL Server username"
  type        = string
  default     = ""
  sensitive   = true
}

variable "sql_server_password" {
  description = "Legacy SQL Server password"
  type        = string
  default     = ""
  sensitive   = true
}

# Web App API (for REST ingestion)
variable "webapp_api_url" {
  description = "Base URL for web application API"
  type        = string
  default     = ""
}

# Salesforce connection
variable "salesforce_client_id" {
  description = "Salesforce OAuth client ID"
  type        = string
  default     = ""
  sensitive   = true
}

variable "salesforce_client_secret" {
  description = "Salesforce OAuth client secret"
  type        = string
  default     = ""
  sensitive   = true
}

# MongoDB connection
variable "mongodb_connection_string" {
  description = "MongoDB Atlas connection string"
  type        = string
  default     = ""
  sensitive   = true
}

# GPU cluster configuration
variable "gpu_node_type" {
  description = "Azure VM type for GPU cluster"
  type        = string
  default     = "Standard_NC6s_v3"  # NVIDIA V100
}

variable "gpu_cluster_autoterminate_minutes" {
  description = "Minutes of inactivity before GPU cluster terminates"
  type        = number
  default     = 20
}

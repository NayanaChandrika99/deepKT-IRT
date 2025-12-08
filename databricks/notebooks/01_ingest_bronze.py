# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingest Bronze Layer
# MAGIC 
# MAGIC Ingests raw data from multiple sources into the Bronze layer of our Delta Lake.
# MAGIC 
# MAGIC **Sources:**
# MAGIC - Local CSV files (EDM Cup 2023, ASSISTments)
# MAGIC - SQL Server (legacy data warehouse)
# MAGIC - REST APIs (web application data)
# MAGIC 
# MAGIC **Output:**
# MAGIC - `bronze.raw_action_logs`
# MAGIC - `bronze.raw_assignment_details`
# MAGIC - `bronze.raw_problem_details`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Storage configuration
STORAGE_ACCOUNT = dbutils.secrets.get(scope="deepkt-secrets", key="storage-account-name")
BRONZE_PATH = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"

# Catalog configuration
CATALOG = "deepkt_irt"
BRONZE_SCHEMA = "bronze"

# Set default catalog and schema
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {BRONZE_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingest from Local/Uploaded Files
# MAGIC 
# MAGIC For initial migration, upload EDM Cup 2023 and ASSISTments CSV files to ADLS.

# COMMAND ----------

def ingest_edm_cup_2023(raw_path: str):
    """
    Ingest EDM Cup 2023 dataset files.
    
    Expected files:
    - action_logs.csv
    - assignment_details.csv
    - problem_details.csv
    """
    # Read action logs
    action_logs_df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{raw_path}/edm_cup_2023/action_logs.csv")
    )
    
    # Add ingestion metadata
    from pyspark.sql.functions import current_timestamp, lit
    
    action_logs_df = (
        action_logs_df
        .withColumn("_ingested_at", current_timestamp())
        .withColumn("_source", lit("edm_cup_2023"))
    )
    
    # Write to Bronze layer
    (
        action_logs_df
        .write
        .format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .saveAsTable("raw_action_logs")
    )
    
    print(f"Ingested {action_logs_df.count()} action log records")
    
    # Read assignment details
    assignment_df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{raw_path}/edm_cup_2023/assignment_details.csv")
        .withColumn("_ingested_at", current_timestamp())
        .withColumn("_source", lit("edm_cup_2023"))
    )
    
    (
        assignment_df
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable("raw_assignment_details")
    )
    
    print(f"Ingested {assignment_df.count()} assignment records")
    
    # Read problem details
    problem_df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{raw_path}/edm_cup_2023/problem_details.csv")
        .withColumn("_ingested_at", current_timestamp())
        .withColumn("_source", lit("edm_cup_2023"))
    )
    
    (
        problem_df
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable("raw_problem_details")
    )
    
    print(f"Ingested {problem_df.count()} problem records")
    
    return {
        "action_logs": action_logs_df.count(),
        "assignments": assignment_df.count(),
        "problems": problem_df.count()
    }

# COMMAND ----------

# Run ingestion
raw_upload_path = f"{BRONZE_PATH}/uploads"
counts = ingest_edm_cup_2023(raw_upload_path)
print(f"Total records ingested: {sum(counts.values())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ingest from SQL Server (Legacy Data Warehouse)
# MAGIC 
# MAGIC For production, use ADF pipelines. This is for ad-hoc ingestion.

# COMMAND ----------

def ingest_from_sql_server(table_name: str, target_table: str):
    """
    Ingest a table from legacy SQL Server data warehouse.
    Uses JDBC connection via Databricks secrets.
    """
    jdbc_url = dbutils.secrets.get(scope="deepkt-secrets", key="sql-server-jdbc-url")
    jdbc_user = dbutils.secrets.get(scope="deepkt-secrets", key="sql-server-user")
    jdbc_password = dbutils.secrets.get(scope="deepkt-secrets", key="sql-server-password")
    
    from pyspark.sql.functions import current_timestamp, lit
    
    df = (
        spark.read
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table_name)
        .option("user", jdbc_user)
        .option("password", jdbc_password)
        .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver")
        .load()
        .withColumn("_ingested_at", current_timestamp())
        .withColumn("_source", lit("sql_server"))
    )
    
    (
        df
        .write
        .format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .saveAsTable(target_table)
    )
    
    return df.count()

# Example usage (uncomment when SQL Server is configured):
# count = ingest_from_sql_server("dbo.StudentResponses", "raw_legacy_responses")
# print(f"Ingested {count} records from SQL Server")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ingest from REST API
# MAGIC 
# MAGIC For real-time or near-real-time data from web applications.

# COMMAND ----------

import requests
from pyspark.sql.functions import current_timestamp, lit, explode
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, TimestampType

def ingest_from_api(api_url: str, target_table: str, params: dict = None):
    """
    Ingest data from a REST API endpoint.
    Expects JSON array response.
    """
    api_key = dbutils.secrets.get(scope="deepkt-secrets", key="webapp-api-key")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(api_url, headers=headers, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    if not data:
        print("No data returned from API")
        return 0
    
    # Create DataFrame from JSON
    df = spark.createDataFrame(data)
    
    df = (
        df
        .withColumn("_ingested_at", current_timestamp())
        .withColumn("_source", lit("rest_api"))
    )
    
    (
        df
        .write
        .format("delta")
        .mode("append")  # Append for incremental API data
        .option("mergeSchema", "true")
        .saveAsTable(target_table)
    )
    
    return df.count()

# Example usage (uncomment when API is configured):
# count = ingest_from_api(
#     "https://api.yourapp.com/v1/learning-events",
#     "raw_api_events",
#     params={"since": "2024-01-01"}
# )
# print(f"Ingested {count} records from API")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data Quality Checks

# COMMAND ----------

def validate_bronze_tables():
    """Run basic data quality checks on Bronze tables."""
    
    tables = ["raw_action_logs", "raw_assignment_details", "raw_problem_details"]
    results = {}
    
    for table in tables:
        try:
            df = spark.table(table)
            count = df.count()
            null_counts = {col: df.filter(df[col].isNull()).count() for col in df.columns[:5]}
            
            results[table] = {
                "row_count": count,
                "sample_null_counts": null_counts,
                "status": "OK" if count > 0 else "EMPTY"
            }
        except Exception as e:
            results[table] = {"status": "ERROR", "error": str(e)}
    
    return results

# Run validation
validation_results = validate_bronze_tables()
for table, result in validation_results.items():
    print(f"{table}: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC Bronze layer ingestion complete. Next step: Transform to Silver layer using `02_transform_silver` notebook.

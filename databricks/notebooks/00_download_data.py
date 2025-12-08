# Databricks notebook source
# MAGIC %md
# MAGIC # Download EDM Cup 2023 Dataset
# MAGIC 
# MAGIC Downloads the dataset directly from the internet - no uploads needed!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Download the Dataset
# MAGIC 
# MAGIC **Where to get EDM Cup 2023 data:**
# MAGIC - **Kaggle**: https://www.kaggle.com/competitions/edm-cup-2023
# MAGIC - **Official Site**: Check EDM Cup website
# MAGIC - **OSF**: https://osf.io/... (if hosted there)
# MAGIC 
# MAGIC ### For Kaggle datasets:
# MAGIC 1. Get your Kaggle API token from https://www.kaggle.com/settings → "Create New Token"
# MAGIC 2. Use the code below

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 3A: Download from Kaggle

# COMMAND ----------

# Install kaggle package
%pip install kaggle

# COMMAND ----------

# Set Kaggle credentials
# Get these from Kaggle → Settings → API → Create New Token
import os
os.environ['KAGGLE_USERNAME'] = "your-username"  # ← Replace with your Kaggle username
os.environ['KAGGLE_KEY'] = "your-api-key"        # ← Replace with your Kaggle API key

# Download dataset
import kaggle
kaggle.api.competition_download_files(
    'edm-cup-2023',  # Competition name
    path='/tmp/edm_data/',
    quiet=False
)

print("✓ Download complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 3B: Download from Direct URL (If you have one)

# COMMAND ----------

import urllib.request
import zipfile
from pathlib import Path

# If you have a direct download link (Google Drive, Dropbox, etc.)
DATA_URL = "YOUR_DOWNLOAD_URL_HERE"  # ← Paste your URL here

download_path = "/tmp/edm_cup_2023.zip"
extract_path = "/tmp/edm_data/"

print("Downloading dataset...")
try:
    urllib.request.urlretrieve(DATA_URL, download_path)
    print(f"✓ Downloaded to {download_path}")
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"✓ Extracted to {extract_path}")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure to set DATA_URL to a valid download link")

# COMMAND ----------

# Check what files we have
import os
for root, dirs, files in os.walk("/tmp/edm_data/"):
    for file in files:
        filepath = os.path.join(root, file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"📄 {file} ({size_mb:.1f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load CSV Files into Delta Tables

# COMMAND ----------

# Find the CSV files (adjust paths based on what you see above)
DATA_DIR = "/tmp/edm_data/"  # Adjust if needed

# Read action_logs.csv
print("Loading action_logs...")
action_logs = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(f"{DATA_DIR}action_logs.csv")  # Adjust filename if different

print(f"✓ Found {action_logs.count():,} action log records")
print(f"Columns: {action_logs.columns}")

# Save as Delta table
action_logs.write.format("delta").mode("overwrite").saveAsTable("workspace.default.action_logs")
print("✓ Saved to workspace.default.action_logs")

# COMMAND ----------

# Read assignment_details.csv
print("Loading assignment_details...")
assignment_details = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(f"{DATA_DIR}assignment_details.csv")

print(f"✓ Found {assignment_details.count():,} assignment records")
assignment_details.write.format("delta").mode("overwrite").saveAsTable("workspace.default.assignment_details")
print("✓ Saved to workspace.default.assignment_details")

# COMMAND ----------

# Read problem_details.csv
print("Loading problem_details...")
problem_details = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(f"{DATA_DIR}problem_details.csv")

print(f"✓ Found {problem_details.count():,} problem records")
problem_details.write.format("delta").mode("overwrite").saveAsTable("workspace.default.problem_details")
print("✓ Saved to workspace.default.problem_details")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Verify Data is Loaded

# COMMAND ----------

# List all tables
tables = spark.sql("SHOW TABLES IN workspace.default")
display(tables)

# COMMAND ----------

# Preview action_logs
print("Preview of action_logs:")
display(spark.table("workspace.default.action_logs").limit(10))

# COMMAND ----------

# Preview assignment_details
print("Preview of assignment_details:")
display(spark.table("workspace.default.assignment_details").limit(10))

# COMMAND ----------

# Preview problem_details
print("Preview of problem_details:")
display(spark.table("workspace.default.problem_details").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Data Summary

# COMMAND ----------

print("=" * 60)
print("DATA LOADED SUCCESSFULLY!")
print("=" * 60)

# Get counts
action_count = spark.table("workspace.default.action_logs").count()
assignment_count = spark.table("workspace.default.assignment_details").count()
problem_count = spark.table("workspace.default.problem_details").count()

print(f"\n📊 Records loaded:")
print(f"   • action_logs:         {action_count:,}")
print(f"   • assignment_details:  {assignment_count:,}")
print(f"   • problem_details:     {problem_count:,}")

print(f"\n📁 Tables created in: workspace.default")
print(f"   • workspace.default.action_logs")
print(f"   • workspace.default.assignment_details")
print(f"   • workspace.default.problem_details")

print(f"\n🎯 Next step: Run notebook 02_transform_silver to transform this data!")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC 
# MAGIC **If download fails:**
# MAGIC 1. For Kaggle: Make sure you've joined the competition and accepted rules
# MAGIC 2. For direct URL: Make sure the URL is a direct download link (not a webpage)
# MAGIC 3. Check if the file is on Google Drive - you may need to use `gdown` package
# MAGIC 
# MAGIC **If CSV files have different names:**
# MAGIC - Run the cell that lists files to see actual filenames
# MAGIC - Update the `load()` paths accordingly
# MAGIC 
# MAGIC **If you get memory errors:**
# MAGIC - Increase cluster size or
# MAGIC - Load files in smaller chunks

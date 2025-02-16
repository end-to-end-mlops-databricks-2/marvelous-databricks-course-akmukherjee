# Databricks notebook source
# MAGIC %restart_python
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

# MAGIC %pip show databricks-sdk

# COMMAND ----------

import mlflow
from pyspark.sql import SparkSession
from typing import List, Dict
from typing import List, Dict
import os
import requests
import time
from src.hotel_reservation.config import ProjectConfig, Tags
from src.hotel_reservation.serving.feature_serving import FeatureLookupServing

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})
endpoint_name="hotel-reservations-model-serving"

# COMMAND ----------

feature_model_service = FeatureLookupServing(
    model_name=f"{catalog_name}.{schema_name}.random_forest_model_fe",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.customer_features"
)

# COMMAND ----------

feature_model_service.create_online_table()

# COMMAND ----------

feature_model_service.deploy_or_update_serving_endpoint()

# COMMAND ----------

# Initialise the Model Serving manager
model_serving = ModelServing(model_name=f"{catalog_name}.{schema_name}.random_forest_model_week2",
                             endpoint_name=endpoint_name)

# COMMAND ----------

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------

# Sample 100 records from the training set
sampled_data = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas().drop(columns=[config.target,"update_timestamp_utc"]).sample(n=100).to_dict(orient="records")

# COMMAND ----------

dataframe_records = [[record] for record in sampled_data]
print(dataframe_records[0])

# COMMAND ----------

def call_endpoint(record: List[Dict]):
    """
    Call the model serving endpoint with a given input
    """
    serving_uri = f"https://{os.environ['DBR_HOST']}/serving-endpoints/hotel-reservations-model-serving/invocations"
    
    response = requests.post(
        serving_uri,
        headers={
            "Authorization": f"Bearer {os.environ['DBR_TOKEN']}"
        },
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

# COMMAND ----------

status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")


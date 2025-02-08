# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

import mlflow
from pyspark.sql import SparkSession

from src.hotel_reservation.config import ProjectConfig, Tags
from src.hotel_reservation.models.model_rf import RandomForestModel

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------

# Initialize model with the config path
rf_model = RandomForestModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

rf_model.load_data()
rf_model.prepare_features()

# COMMAND ----------

rf_model.train()

# COMMAND ----------

rf_model.log_model()

# COMMAND ----------

rf_model.register_model()

# COMMAND ----------

rf_model.retrieve_current_run_metadata()

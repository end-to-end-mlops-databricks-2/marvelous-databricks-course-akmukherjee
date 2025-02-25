# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC %pip install mlflow
# MAGIC %pip install --upgrade typing_extensions
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

# MAGIC %restart_python
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip show databricks-sdk

# COMMAND ----------

import mlflow
from pyspark.sql import SparkSession

from src.hotel_reservation.config import ProjectConfig, Tags
from src.hotel_reservation.models.model_fe import FeatureLookUpModel

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------

# Initialize model with the config path
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

fe_model.create_feature_table()
fe_model.define_feature_function()

# COMMAND ----------

fe_model.load_data()
fe_model.prepare_features()

# COMMAND ----------

fe_model.train()

# COMMAND ----------

fe_model.log_model()

# COMMAND ----------

fe_model.register_model()

# COMMAND ----------

fe_model.retrieve_current_run_metadata()

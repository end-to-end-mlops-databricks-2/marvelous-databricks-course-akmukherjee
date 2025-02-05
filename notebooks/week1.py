# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# MAGIC %pip install /Volumes/marvelous_mlops/pre_processed/wheels/hotel_reservation-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import yaml
import logging

# COMMAND ----------

from hotel_reservation.data_processor import DataProcessor
from hotel_reservation.config import ProjectConfig
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Construct the path to config file
config_path = '../project_config.yml'

# Load configuration
config = ProjectConfig.from_yaml(config_path=str(config_path))

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("/Volumes/marvelous-mlops/pre-processed/data/Hotel Reservations.csv", header=True,inferSchema=True).toPandas()

# COMMAND ----------

# Initialize DataProcessor
data_processor = DataProcessor(df, config)

# Preprocess the data
data_processor.preprocess()

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

data_processor.save_to_catalog(X_train, X_test,spark)

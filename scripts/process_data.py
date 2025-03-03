import argparse
import logging
import yaml
import sys
import os
from pyspark.sql import SparkSession

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True
)
parser.add_argument(
    "--data_path",
    action="store",
    default="/Volumes/marvelous-mlops/pre-processed/data/Hotel Reservations.csv",
    type=str,
    required=False,
    help="Path to the input CSV data file"
)

# Parse arguments
args = parser.parse_args()
root_path = args.root_path
data_path = args.data_path

# Add the root path to Python's path
logger.info(f"Adding {root_path} to Python path")
sys.path.append(root_path)

# Import required modules after adding to path
try:
    from src.hotel_reservation.config import ProjectConfig
    from src.hotel_reservation.data_processor import DataProcessor
    logger.info("Successfully imported modules")
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    raise

# Set config path and load configuration
config_path = f"{root_path}/project_config.yml"
logger.info(f"Loading configuration from {config_path}")

try:
    # Load configuration
    config = ProjectConfig.from_yaml(config_path=config_path)
    logger.info("Configuration loaded:")
    logger.info(yaml.dump(config, default_flow_style=False))
    
    # Create Spark session
    spark = SparkSession.builder.getOrCreate()
    logger.info("Created Spark session")
    
    # Read data
    logger.info(f"Reading data from {data_path}")
    df = spark.read.csv(
        data_path, 
        header=True, 
        inferSchema=True
    ).toPandas()
    
    # Initialize DataProcessor
    logger.info("Initializing DataProcessor")
    data_processor = DataProcessor(df, config)
    
    # Preprocess the data
    logger.info("Preprocessing data")
    data_processor.preprocess()
    
    # Split the data
    logger.info("Splitting data into train and test sets")
    X_train, X_test = data_processor.split_data()
    
    # Save to catalog
    logger.info("Saving data to catalog")
    data_processor.save_to_catalog(X_train, X_test, spark)
    
    logger.info("Process completed successfully")
    
except Exception as e:
    logger.error(f"Error in processing: {e}")
    raise
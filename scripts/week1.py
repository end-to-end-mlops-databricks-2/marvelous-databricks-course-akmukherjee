import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pyspark.sql import SparkSession
from src.hotel_reservation.config import ProjectConfig, Tags
from src.hotel_reservation.data_processor import DataProcessor
#from src.hotel_reservation.data_processor import DataProcessor
# Configure logging
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Construct the path to config file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up one level to the project root
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Add the project root to the Python path
sys.path.append(project_root)
config_path = os.path.join(project_root, "project_config.yml")

# Load configuration
config = ProjectConfig.from_yaml(config_path=str(config_path))

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("/Volumes/marvelous-mlops/pre-processed/data/Hotel Reservations.csv", header=True,inferSchema=True).toPandas()

# Initialize DataProcessor
data_processor = DataProcessor(df, config)

# Preprocess the data
data_processor.preprocess()
# Split the data and save
X_train, X_test = data_processor.split_data()
spark = SparkSession.builder.getOrCreate()
data_processor.save_to_catalog(X_train, X_test,spark)
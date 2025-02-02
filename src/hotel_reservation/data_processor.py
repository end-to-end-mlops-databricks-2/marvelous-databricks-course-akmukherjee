import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

from hotel_reservation.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        print(f"Number of rows before dropping duplicates: {len(self.df)}")
        # Dropping duplicate columns
        self.df.drop_duplicates(inplace=True)
        print(f"Number of rows before after dropping duplicates: {len(self.df)}")

        # Drop Columns which are not of interest
        columns_to_drop = self.config.columns_to_drop
        self.df.drop(columns=columns_to_drop, inplace=True)
        ####### Handle numeric features #######
        # Removing Outliers
        self.df = self.df[self.df['lead_time'] <= 400]
        print(f"Number of rows after removing outliers for Lead Time: {len(self.df)}")

        self.df = self.df[self.df['avg_price_per_room'] <= 300]
        print(f"Number of rows after removing outliers for average room price: {len(self.df)}")

        scaler = MinMaxScaler()
        self.df[num_features] = scaler.fit_transform(self.df[num_features])

        # Convert categorical features to the appropriate type
        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")
        # One-hot encode with cleaned column names
        df_encoded = pd.get_dummies(
            self.df, 
            columns=cat_features,
            drop_first=True,
            dtype=bool
        )
        
        # Clean column names: replace spaces and special characters with underscores
        clean_columns = {col: col.replace(' ', '_').replace('-', '_') 
                        for col in df_encoded.columns}
        self.df = df_encoded.rename(columns=clean_columns)
    
        # For the target variable
        lb = LabelEncoder()
        target = self.config.target
        self.df[target] = lb.fit_transform(self.df[target])
        
        return self.df
        

            
            

    
    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
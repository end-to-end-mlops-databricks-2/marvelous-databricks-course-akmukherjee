from sklearn.metrics import accuracy_score  # Keep only what's used in the code
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier 
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from src.hotel_reservation.config import ProjectConfig, Tags
import random
import string

class RandomForestModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.tags = tags.dict()

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        Splits data into:
        Features (X_train, X_test)
        Target (y_train, y_test)
        """
        print("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0" #describe history -> retrieve

        self.X_train = self.train_set.drop([self.target,'update_timestamp_utc'], axis=1)
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set.drop([self.target,'update_timestamp_utc'], axis=1)
        self.y_test = self.test_set[self.target]
        print("✅ Data successfully loaded.")
    
    def prepare_features(self):
        """
        Encodes categorical features with OneHotEncoder (ignores unseen categories).
        Passes numerical features as-is (remainder='passthrough').
        Defines a pipeline combining:
            Features processing
            Random Forest Classifier model
        """
        print("🔄 Defining preprocessing pipeline...")
        cat_cols=self.X_train.select_dtypes(include='object').columns.tolist()
        cat_cols = [self.X_train.columns.get_loc(col) for col in cat_cols]
        model=RandomForestClassifier()
        smote = SMOTENC(categorical_features=cat_cols, random_state=42) if cat_cols else SMOTE(random_state=42)
        preprocessor=ColumnTransformer(transformers=[('encoder',OneHotEncoder(drop='first'),cat_cols)],remainder='passthrough')
        # Create pipeline with SMOTE
        self.pipeline = ImbPipeline(steps=[
            ('preprocessing', preprocessor),  # Step 1: Encoding
            ('smote', smote),  # Step 2: Apply SMOTE
            ('model', model)  # Step 3: Train Model
        ])
        print("✅ Preprocessing pipeline defined.")

    def train(self):
        """
        Train the model.
        """
        print("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)
        print("✅ Training completed.")
    
    def log_model(self):
        """
        Log the model.
        """
        # Adding randomness to the experiment name to avoid overwriting
        #randstr =(''.join(random.choices(string.ascii_lowercase, k=5)))
        experiment_name =(f'{self.experiment_name}')
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(name=experiment_name)

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            # Perform cross-validation
            cv_scores = cross_val_score(self.pipeline, self.X_train, self.y_train, scoring='accuracy')
            
            # Predict on test set
            y_pred = self.pipeline.predict(self.X_test)
            # Evaluate metrics
            acc_score = accuracy_score(self.y_test,y_pred)
            print(f"Accuracy: {acc_score}")
            # Log parameters and metrics
            mlflow.log_param("model_type", "Random Forest with SMOTE")
           
            mlflow.log_metric("cv_scores_mean", cv_scores.mean())
            mlflow.log_metric("cv_scores_standard_deviation", cv_scores.std())
            mlflow.log_metric("accuracy", acc_score)
            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version
            )
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="random_forest",
                signature=signature
            )
        
    def register_model(self):
        """
        Register model in UC
        """
        print("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f'runs:/{self.run_id}/random_forest',
            name=f"{self.catalog_name}.{self.schema_name}.random_forest_model_week2",
            tags=self.tags
        )
        print(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.random_forest_model_week2",
            alias="latest-model",
            version=latest_version
        )
    
    def retrieve_current_run_dataset(self):
        """
        Retrieve MLflow run dataset.
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        print("✅ Dataset source loaded.")
        return dataset_source.load()
    
    def retrieve_current_run_metadata(self):
        """
        Retrieve MLflow run metadata.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        print("✅ Dataset metadata loaded.")
        return metrics, params

            
# Databricks notebook source
train_set_df = spark.table("marvelous_mlops.pre_processed.train_set")
display(train_set_df)

# COMMAND ----------

x_train = train_set_df.drop("booking_status")
y_train = train_set_df.select("booking_status")

# COMMAND ----------

pip install imbalanced-learn

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC  # Use SMOTENC for categorical features
from imblearn.over_sampling import SMOTE  #
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# COMMAND ----------

type(x_train)

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

def prepare_spark_data(x_train, y_train):
    # Convert Spark DataFrames to pandas, then to numpy
    x_array = x_train.drop("update_timestamp_utc").toPandas().to_numpy()
    y_array = y_train.drop("update_timestamp_utc").toPandas().to_numpy()
    
    # Ensure y is 1D
    if len(y_array.shape) > 1 and y_array.shape[1] == 1:
        y_array = y_array.ravel()
    
    return x_array, y_array

# Convert the data
x_train_array, y_train_array = prepare_spark_data(x_train, y_train)

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

def prepare_spark_data(x_train, y_train):
    # Convert Spark DataFrames to pandas, then to numpy
    x_array = x_train.drop("update_timestamp_utc").toPandas().to_numpy()
    y_array = y_train.drop("update_timestamp_utc").toPandas().to_numpy()
    
    # Ensure y is 1D
    if len(y_array.shape) > 1 and y_array.shape[1] == 1:
        y_array = y_array.ravel()
    
    return x_array, y_array

# Convert the data
x_train_array, y_train_array = prepare_spark_data(x_train, y_train)

# Now proceed with the model
model = DecisionTreeClassifier()
path = model.cost_complexity_pruning_path(x_train_array, y_train_array)
alphas = path['ccp_alphas']
alphas_list = alphas.tolist()
random_values = random.sample(alphas_list, min(10, len(alphas_list)))

# COMMAND ----------

params = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 4, 5, 6, 7, 8, 9],
    'ccp_alpha': random_values,
}

grid = GridSearchCV(
    estimator= model,
    param_grid= params,
    scoring= 'f1',
    cv=5,
)
grid.fit(x_train_array, y_train_array)

grid.best_params_

# COMMAND ----------

model = DecisionTreeClassifier(criterion=grid.best_params_['criterion'],
                               max_depth=grid.best_params_['max_depth'],
                               ccp_alpha=grid.best_params_['ccp_alpha'])
model.fit(x_train_array, y_train_array)
yTrain_pred = model.predict(x_train_array)
f1_score(y_train_array, yTrain_pred)

# COMMAND ----------

import pandas as pd
import numpy as np
df=pd.read_csv('/Workspace/Users/akmukherjee@gmail.com/Week2-MarvelousMLOps/marvelous-databricks-course-akmukherjee/data/Hotel Reservations.csv')
df.drop(columns='Booking_ID',inplace=True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop(columns='booking_status'),df['booking_status'],test_size=0.2,random_state=42)
cat_cols=X_train.select_dtypes(include='object').columns.tolist()
cat_cols = [X_train.columns.get_loc(col) for col in cat_cols]

# COMMAND ----------



# COMMAND ----------

model=RandomForestClassifier()
smote = SMOTENC(categorical_features=cat_cols, random_state=42) if cat_cols else SMOTE(random_state=42)
preprocessor=ColumnTransformer(transformers=[('encoder',OneHotEncoder(drop='first'),cat_cols)],remainder='passthrough')
# Create pipeline with SMOTE
pipeline = ImbPipeline(steps=[
    ('preprocessing', preprocessor),  # Step 1: Encoding
    ('smote', smote),  # Step 2: Apply SMOTE
    ('model', model)  # Step 3: Train Model
])

# Perform cross-validation
#cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

# COMMAND ----------

df = spark.table("marvelous_mlops.pre_processed.train_set").toPandas()

X_train,X_test,y_train,y_test=train_test_split(df.drop(columns=['booking_status','update_timestamp_utc']),df['booking_status'],test_size=0.2,random_state=42)

# COMMAND ----------

# Display results
#print(f"Cross-validation scores: {cv_scores}")
#print(f"Mean Accuracy: {cv_scores.mean():.4f}")
#print(f"Standard Deviation: {cv_scores.std():.4f}")
cat_cols=X_train.select_dtypes(include='object').columns.tolist()
cat_cols = [X_train.columns.get_loc(col) for col in cat_cols]
model=RandomForestClassifier()
smote = SMOTENC(categorical_features=cat_cols, random_state=42) if cat_cols else SMOTE(random_state=42)
preprocessor=ColumnTransformer(transformers=[('encoder',OneHotEncoder(drop='first'),cat_cols)],remainder='passthrough')
# Create pipeline with SMOTE
pipeline = ImbPipeline(steps=[
    ('preprocessing', preprocessor),  # Step 1: Encoding
    ('smote', smote),  # Step 2: Apply SMOTE
    ('model', model)  # Step 3: Train Model
])

pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_test)
print()
print("Test Results")
print("Accuracy: ",accuracy_score(y_test,y_pred))

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
catalog_name ='marvelous_mlops'
schema_name = 'pre_processed'
df_spark =spark.table("marvelous_mlops.pre_processed.train_set")
#ARTIFACT_PATH = f"dbfs:/Volumes/{catalog_name}/{schema_name}/{VOLUME}"
EXP_NAME = "/Users/akmukherjee@gmail.com/my-random-forest-experiment-1"
if mlflow.get_experiment_by_name(EXP_NAME) is None:
    mlflow.create_experiment(name=EXP_NAME)
mlflow.set_experiment(EXP_NAME)
with mlflow.start_run() as run:
    run_id = run.info.run_id
    y_pred = pipeline.predict(X_test)
    mlflow.log_param("model_type", "RandomForestClassifier with SMOTE")
    # Log the model
    signature = infer_signature(model_input=X_train, model_output=y_pred)
    catalog_name ='marvelous_mlops'
    schema_name = 'pre_processed'
    dataset = mlflow.data.from_spark(
                df_spark,
                table_name=f"{catalog_name}.{schema_name}.train_set",
                version='0'
            )
    mlflow.log_input(dataset, context="training")
    mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="random_forest",
                signature=signature
            )
    registered_model = mlflow.register_model(
            model_uri=f'runs:/{run_id}/random_forest',
            name=f"{catalog_name}.{schema_name}.hotel_reservation_model-test"
        )

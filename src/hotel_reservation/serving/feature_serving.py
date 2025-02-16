import mlflow

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *
from databricks.sdk.service.serving import ServedModelInput, EndpointCoreConfigInput,ServedModelInputWorkloadSize

class FeatureLookupServing:
    def __init__(self, model_name: str, endpoint_name: str, feature_table_name: str):
        """
        Initialise Feature Lookup Server Manager
        """
        self.workspace = WorkspaceClient()
        self.feature_table = feature_table_name
        self.online_table = f"{self.feature_table}_online"
        self.model_name = model_name
        self.endpoint_name = endpoint_name
        
    def create_online_table(self):
        """
        Create an online table for hotel features
        """
        spec = OnlineTableSpec(
            primary_key_columns=["Booking_ID"],
            source_table_full_name=self.feature_table,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )
        
        # Create an OnlineTable object with the spec
        online_table = OnlineTable(
            name=self.online_table,
            spec=spec
        )
        
        # Create the online table
        self.workspace.online_tables.create(
            table=online_table
        )
        
    def get_latest_model_version(self):
        """
        Gets the latest model version from the DB
        Model Registry. Assumes model is tagged with
        the 'latest-model' alias.
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(
            self.model_name, alias="latest-model"
        ).version
        print(f"Latest model version: {latest_version}")
        return latest_version
    
    def deploy_or_update_serving_endpoint(self,
        version: str = "latest", workload_size: str = "Small",
        scale_to_zero: bool = True):
        """
        Deploy a model serving endpoint on Databricks
        """
        endpoint_exists = any(
            item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list()
        )
        if version == "latest":
            entity_version = self.get_latest_model_version()
        else:
            entity_version = version
            
        served_models = [
            ServedModelInput(
                model_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=ServedModelInputWorkloadSize.SMALL,
                model_version=entity_version
            )
        ]
        
        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_models=served_models
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(
                name=self.endpoint_name,
                served_models=served_models
            )
import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
from app import app

logger = logging.getLogger(__name__)

def initialize_mlflow():
    """Initialize MLFlow tracking."""
    tracking_uri = app.config.get('MLFLOW_TRACKING_URI')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLFlow tracking URI set to: {tracking_uri}")
    else:
        logger.warning("No MLFlow tracking URI configured")

def create_mlflow_experiment(experiment_name, parameters):
    """
    Create a new MLFlow experiment or use an existing one.
    
    Args:
        experiment_name: Name of the experiment
        parameters: Dict containing experiment parameters
        
    Returns:
        tuple: (experiment_id, run_id)
    """
    initialize_mlflow()
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    # Start a new run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        
        # Log parameters
        for key, value in parameters.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
        
        logger.info(f"Created MLFlow run: {run_id} for experiment: {experiment_id}")
        
    return experiment_id, run_id

def log_metrics_to_mlflow(run_id, metrics):
    """
    Log metrics to an MLFlow run.
    
    Args:
        run_id: ID of the MLFlow run
        metrics: Dict containing metrics to log
    """
    initialize_mlflow()
    
    with mlflow.start_run(run_id=run_id):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        logger.info(f"Logged metrics to MLFlow run: {run_id}")

def log_model_to_mlflow(run_id, model_path, model_type):
    """
    Log a model to MLFlow.
    
    Args:
        run_id: ID of the MLFlow run
        model_path: Path to the model file
        model_type: Type of the model (yolo, rf-detr)
    """
    initialize_mlflow()
    
    with mlflow.start_run(run_id=run_id):
        if model_type.startswith('yolo'):
            mlflow.pytorch.log_model(model_path, artifact_path="model")
        elif model_type.startswith('rf-detr'):
            mlflow.pytorch.log_model(model_path, artifact_path="model")
        
        logger.info(f"Logged model to MLFlow run: {run_id}")

def get_mlflow_run_info(run_id):
    """
    Get information about an MLFlow run.
    
    Args:
        run_id: ID of the MLFlow run
        
    Returns:
        dict: Run information including metrics and parameters
    """
    initialize_mlflow()
    
    client = MlflowClient()
    run = client.get_run(run_id)
    
    return {
        'metrics': run.data.metrics,
        'parameters': run.data.params,
        'status': run.info.status,
        'start_time': run.info.start_time,
        'end_time': run.info.end_time
    }

def list_mlflow_experiments():
    """
    List all MLFlow experiments.
    
    Returns:
        list: List of experiment information
    """
    initialize_mlflow()
    
    client = MlflowClient()
    experiments = client.search_experiments()
    
    return [{
        'id': exp.experiment_id,
        'name': exp.name,
        'artifact_location': exp.artifact_location,
        'lifecycle_stage': exp.lifecycle_stage
    } for exp in experiments]

def list_mlflow_runs(experiment_id):
    """
    List all runs for an MLFlow experiment.
    
    Args:
        experiment_id: ID of the MLFlow experiment
        
    Returns:
        list: List of run information
    """
    initialize_mlflow()
    
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    return [{
        'run_id': run.info.run_id,
        'status': run.info.status,
        'start_time': run.info.start_time,
        'end_time': run.info.end_time,
        'metrics': run.data.metrics
    } for run in runs]

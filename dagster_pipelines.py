import os
import logging
import json
import requests
from app import app

logger = logging.getLogger(__name__)

def initialize_dagster():
    """Initialize Dagster connection."""
    dagster_api_url = app.config.get('DAGSTER_API_URL')
    if not dagster_api_url:
        logger.warning("No Dagster API URL configured")
        return None
    
    return dagster_api_url

def submit_dagster_pipeline(job_id, parameters):
    """
    Submit a training pipeline to Dagster.
    
    Args:
        job_id: ID of the training job
        parameters: Dict containing pipeline parameters
        
    Returns:
        str: Dagster run ID
    """
    dagster_api_url = initialize_dagster()
    if not dagster_api_url:
        raise ValueError("Dagster API URL not configured")
    
    # Choose pipeline based on model type
    pipeline_name = "yolo_training_pipeline"
    if parameters.get('model_type') == 'rf-detr':
        pipeline_name = "rf_detr_training_pipeline"
    
    # Prepare pipeline configuration
    config = {
        "ops": {
            "load_dataset": {
                "config": {
                    "job_id": job_id
                }
            },
            "train_model": {
                "config": {
                    "job_id": job_id,
                    "model_type": parameters.get('model_type'),
                    "model_variant": parameters.get('model_variant'),
                    "epochs": parameters.get('epochs'),
                    "batch_size": parameters.get('batch_size'),
                    "learning_rate": parameters.get('learning_rate'),
                    "augmentation": parameters.get('augmentation', 'default'),
                    "transfer_learning": parameters.get('transfer_learning', False),
                    "pretrained_weights": parameters.get('pretrained_weights', 'coco'),
                    "validation_split": parameters.get('validation_split', 0.2)
                }
            },
            "evaluate_model": {
                "config": {
                    "job_id": job_id
                }
            },
            "save_model": {
                "config": {
                    "job_id": job_id
                }
            }
        }
    }
    
    # Submit pipeline run
    try:
        endpoint = f"{dagster_api_url}/api/v1/pipelines/{pipeline_name}/execute"
        response = requests.post(
            endpoint,
            json={
                "mode": "default",
                "runConfigData": config
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        run_id = result.get('run', {}).get('runId')
        if not run_id:
            raise ValueError("No run ID returned from Dagster")
        
        logger.info(f"Submitted Dagster pipeline: {pipeline_name}, run ID: {run_id}")
        return run_id
        
    except Exception as e:
        logger.error(f"Error submitting Dagster pipeline: {str(e)}")
        raise

def get_dagster_run_status(run_id):
    """
    Get the status of a Dagster pipeline run.
    
    Args:
        run_id: ID of the Dagster run
        
    Returns:
        dict: Run status information
    """
    dagster_api_url = initialize_dagster()
    if not dagster_api_url:
        raise ValueError("Dagster API URL not configured")
    
    try:
        endpoint = f"{dagster_api_url}/api/v1/runs/{run_id}"
        response = requests.get(endpoint)
        response.raise_for_status()
        
        run_info = response.json()
        return {
            'run_id': run_info.get('runId'),
            'status': run_info.get('status'),
            'start_time': run_info.get('startTime'),
            'end_time': run_info.get('endTime')
        }
        
    except Exception as e:
        logger.error(f"Error getting Dagster run status: {str(e)}")
        raise

def cancel_dagster_pipeline(run_id):
    """
    Cancel a running Dagster pipeline.
    
    Args:
        run_id: ID of the Dagster run to cancel
    """
    dagster_api_url = initialize_dagster()
    if not dagster_api_url:
        raise ValueError("Dagster API URL not configured")
    
    try:
        endpoint = f"{dagster_api_url}/api/v1/runs/{run_id}/cancel"
        response = requests.post(endpoint)
        response.raise_for_status()
        
        logger.info(f"Cancelled Dagster run: {run_id}")
        
    except Exception as e:
        logger.error(f"Error cancelling Dagster run: {str(e)}")
        raise

def list_dagster_pipelines():
    """
    List all available Dagster pipelines.
    
    Returns:
        list: List of pipeline information
    """
    dagster_api_url = initialize_dagster()
    if not dagster_api_url:
        raise ValueError("Dagster API URL not configured")
    
    try:
        endpoint = f"{dagster_api_url}/api/v1/pipelines"
        response = requests.get(endpoint)
        response.raise_for_status()
        
        pipelines = response.json()
        return [{
            'name': p.get('name'),
            'description': p.get('description')
        } for p in pipelines]
        
    except Exception as e:
        logger.error(f"Error listing Dagster pipelines: {str(e)}")
        raise

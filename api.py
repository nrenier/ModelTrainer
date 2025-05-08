import logging
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
from app import db
from models import Dataset, TrainingJob, Model
from ml_utils import process_dataset, validate_training_config
from mlflow_utils import create_mlflow_experiment
from dagster_pipelines import submit_dagster_pipeline

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

@api_bp.route('/datasets', methods=['GET'])
def get_datasets():
    """Get all datasets."""
    datasets = Dataset.query.all()
    result = [{
        'id': dataset.id,
        'name': dataset.name,
        'format': dataset.format,
        'num_classes': dataset.num_classes,
        'num_images': dataset.num_images,
        'created_at': dataset.created_at.isoformat() if dataset.created_at else None
    } for dataset in datasets]
    
    return jsonify(result)

@api_bp.route('/datasets/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """Get details of a specific dataset."""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    result = {
        'id': dataset.id,
        'name': dataset.name,
        'description': dataset.description,
        'format': dataset.format,
        'path': dataset.path,
        'created_at': dataset.created_at.isoformat() if dataset.created_at else None,
        'size': dataset.size,
        'num_classes': dataset.num_classes,
        'num_images': dataset.num_images,
        'class_names': dataset.class_names
    }
    
    return jsonify(result)

@api_bp.route('/jobs', methods=['GET'])
def get_jobs():
    """Get all training jobs."""
    jobs = TrainingJob.query.all()
    result = [{
        'id': job.id,
        'name': job.name,
        'model_type': job.model_type,
        'status': job.status,
        'created_at': job.created_at.isoformat() if job.created_at else None,
        'dataset_id': job.dataset_id
    } for job in jobs]
    
    return jsonify(result)

@api_bp.route('/jobs/<int:job_id>', methods=['GET'])
def get_job(job_id):
    """Get details of a specific training job."""
    job = TrainingJob.query.get_or_404(job_id)
    
    result = {
        'id': job.id,
        'name': job.name,
        'description': job.description,
        'model_type': job.model_type,
        'status': job.status,
        'created_at': job.created_at.isoformat() if job.created_at else None,
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'parameters': job.parameters,
        'dataset_id': job.dataset_id,
        'mlflow_experiment_id': job.mlflow_experiment_id,
        'mlflow_run_id': job.mlflow_run_id,
        'dagster_run_id': job.dagster_run_id
    }
    
    return jsonify(result)

@api_bp.route('/jobs', methods=['POST'])
def create_job():
    """Create a new training job."""
    data = request.json
    
    # Validate required fields
    required_fields = ['name', 'model_type', 'dataset_id', 'parameters']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Get the dataset
    dataset = Dataset.query.get(data['dataset_id'])
    if not dataset:
        return jsonify({'error': f'Dataset with ID {data["dataset_id"]} not found'}), 404
    
    # Validate training configuration
    validation_error = validate_training_config(data['parameters'])
    if validation_error:
        return jsonify({'error': validation_error}), 400
    
    try:
        # Create MLFlow experiment
        mlflow_exp_id, mlflow_run_id = create_mlflow_experiment(
            data['name'], 
            data['parameters']
        )
        
        # Create a new training job
        new_job = TrainingJob(
            name=data['name'],
            description=data.get('description', ''),
            model_type=data['model_type'],
            status='pending',
            created_at=datetime.utcnow(),
            parameters=data['parameters'],
            dataset_id=data['dataset_id'],
            mlflow_experiment_id=mlflow_exp_id,
            mlflow_run_id=mlflow_run_id
        )
        
        db.session.add(new_job)
        db.session.commit()
        
        # Submit to Dagster pipeline
        dagster_run_id = submit_dagster_pipeline(new_job.id, data['parameters'])
        
        # Update job with Dagster run ID
        new_job.dagster_run_id = dagster_run_id
        new_job.status = 'running'
        new_job.started_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'id': new_job.id,
            'name': new_job.name,
            'status': new_job.status,
            'dagster_run_id': dagster_run_id
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating training job: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/jobs/<int:job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a specific training job."""
    job = TrainingJob.query.get_or_404(job_id)
    
    result = {
        'id': job.id,
        'status': job.status,
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'completed_at': job.completed_at.isoformat() if job.completed_at else None
    }
    
    return jsonify(result)

@api_bp.route('/jobs/<int:job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a running training job."""
    job = TrainingJob.query.get_or_404(job_id)
    
    if job.status != 'running':
        return jsonify({'error': 'Can only cancel running jobs'}), 400
    
    try:
        # Cancel Dagster pipeline run
        from dagster_pipelines import cancel_dagster_pipeline
        cancel_dagster_pipeline(job.dagster_run_id)
        
        # Update job status
        job.status = 'cancelled'
        db.session.commit()
        
        return jsonify({'status': 'cancelled'})
        
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/models', methods=['GET'])
def get_models():
    """Get all trained models."""
    models = Model.query.all()
    result = [{
        'id': model.id,
        'name': model.name,
        'type': model.type,
        'version': model.version,
        'created_at': model.created_at.isoformat() if model.created_at else None,
        'training_job_id': model.training_job_id
    } for model in models]
    
    return jsonify(result)

@api_bp.route('/models/<int:model_id>', methods=['GET'])
def get_model(model_id):
    """Get details of a specific model."""
    model = Model.query.get_or_404(model_id)
    
    result = {
        'id': model.id,
        'name': model.name,
        'type': model.type,
        'version': model.version,
        'path': model.path,
        'created_at': model.created_at.isoformat() if model.created_at else None,
        'metrics': model.metrics,
        'training_job_id': model.training_job_id
    }
    
    return jsonify(result)

@api_bp.route('/models/<int:model_id>/metrics', methods=['GET'])
def get_model_metrics(model_id):
    """Get metrics for a specific model."""
    model = Model.query.get_or_404(model_id)
    
    return jsonify(model.metrics or {})

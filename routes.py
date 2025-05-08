import os
import logging
from flask import render_template, request, flash, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
from datetime import datetime
from app import app, db
from models import Dataset, TrainingJob, Model
from ml_utils import process_dataset, validate_training_config
from mlflow_utils import create_mlflow_experiment, log_model_to_mlflow
from dagster_pipelines import submit_dagster_pipeline

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Home page with overview stats and links to main sections."""
    datasets_count = Dataset.query.count()
    jobs_count = TrainingJob.query.count()
    running_jobs = TrainingJob.query.filter_by(status='running').count()
    completed_jobs = TrainingJob.query.filter_by(status='completed').count()
    
    return render_template('index.html', 
                          datasets_count=datasets_count,
                          jobs_count=jobs_count,
                          running_jobs=running_jobs,
                          completed_jobs=completed_jobs)

@app.route('/upload', methods=['GET', 'POST'])
def upload_dataset():
    """Page for uploading new datasets."""
    logger.info("Upload dataset route accessed with method: %s", request.method)
    
    if request.method == 'POST':
        logger.info("Processing POST request for dataset upload")
        # Check if the post request has the file part
        if 'dataset_file' not in request.files:
            logger.error("No file part in request")
            flash('No file part', 'error')
            return redirect(request.url)
            
        file = request.files['dataset_file']
        logger.info("File received: %s", file.filename)
        
        # If user does not select file, browser may submit an empty file
        if file.filename == '':
            logger.error("Empty filename submitted")
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file:
            dataset_name = request.form.get('dataset_name', 'Unnamed Dataset')
            dataset_format = request.form.get('dataset_format')
            description = request.form.get('description', '')
            
            logger.info("Processing dataset: %s, format: %s", dataset_name, dataset_format)
            
            # Create a secure filename and save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info("Saving file to: %s", file_path)
            
            try:
                file.save(file_path)
                logger.info("File saved successfully")
                
                # Process the dataset (extract, validate, etc.)
                logger.info("Starting dataset processing")
                dataset_info = process_dataset(file_path, dataset_format)
                logger.info("Dataset processed successfully: %s", dataset_info)
                
                # Create a new dataset record
                new_dataset = Dataset(
                    name=dataset_name,
                    description=description,
                    format=dataset_format,
                    path=file_path,
                    created_at=datetime.utcnow(),
                    size=os.path.getsize(file_path),
                    num_classes=dataset_info.get('num_classes', 0),
                    num_images=dataset_info.get('num_images', 0),
                    class_names=dataset_info.get('class_names', [])
                )
                
                logger.info("Adding dataset to database")
                db.session.add(new_dataset)
                db.session.commit()
                logger.info("Dataset saved to database with ID: %s", new_dataset.id)
                
                flash(f'Dataset "{dataset_name}" uploaded successfully!', 'success')
                logger.info("Redirecting to configure_training")
                return redirect(url_for('configure_training', dataset_id=new_dataset.id))
                
            except Exception as e:
                logger.exception("Error processing dataset")
                logger.error(f"Error details: {str(e)}")
                flash(f'Error processing dataset: {str(e)}', 'error')
                return redirect(request.url)
    
    # GET request - show the upload form
    logger.info("Showing upload form")
    supported_formats = app.config.get('SUPPORTED_FORMATS', ['COCO', 'YOLO', 'Pascal VOC'])
    return render_template('upload.html', supported_formats=supported_formats)

@app.route('/datasets')
def list_datasets():
    """Display all available datasets."""
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    return render_template('datasets.html', datasets=datasets)

@app.route('/dataset/<int:dataset_id>')
def view_dataset(dataset_id):
    """View details of a specific dataset."""
    dataset = Dataset.query.get_or_404(dataset_id)
    return render_template('dataset_detail.html', dataset=dataset)

@app.route('/configure/<int:dataset_id>', methods=['GET', 'POST'])
def configure_training(dataset_id):
    """Configure and start a training job for a dataset."""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if request.method == 'POST':
        job_name = request.form.get('job_name', f'Training job - {datetime.utcnow().strftime("%Y%m%d%H%M%S")}')
        description = request.form.get('description', '')
        model_type = request.form.get('model_type')
        model_variant = request.form.get('model_variant')
        
        # Collect training parameters
        parameters = {
            'model_type': model_type,
            'model_variant': model_variant,
            'epochs': int(request.form.get('epochs', app.config.get('DEFAULT_EPOCHS'))),
            'batch_size': int(request.form.get('batch_size', app.config.get('DEFAULT_BATCH_SIZE'))),
            'learning_rate': float(request.form.get('learning_rate', app.config.get('DEFAULT_LEARNING_RATE'))),
            'augmentation': request.form.get('augmentation', 'default'),
            'transfer_learning': 'transfer_learning' in request.form,
            'pretrained_weights': request.form.get('pretrained_weights', 'coco'),
            'validation_split': float(request.form.get('validation_split', 0.2)),
        }
        
        # Validate the configuration
        validation_error = validate_training_config(parameters)
        if validation_error:
            flash(validation_error, 'error')
            return redirect(request.url)
        
        try:
            # Create MLFlow experiment
            mlflow_exp_id, mlflow_run_id = create_mlflow_experiment(job_name, parameters)
            
            # Create a new training job
            new_job = TrainingJob(
                name=job_name,
                description=description,
                model_type=f"{model_type}_{model_variant}",
                status='pending',
                created_at=datetime.utcnow(),
                parameters=parameters,
                dataset_id=dataset.id,
                mlflow_experiment_id=mlflow_exp_id,
                mlflow_run_id=mlflow_run_id
            )
            
            db.session.add(new_job)
            db.session.commit()
            
            # Submit to Dagster pipeline
            dagster_run_id = submit_dagster_pipeline(new_job.id, parameters)
            
            # Update job with Dagster run ID
            new_job.dagster_run_id = dagster_run_id
            new_job.status = 'running'
            new_job.started_at = datetime.utcnow()
            db.session.commit()
            
            flash(f'Training job "{job_name}" started successfully!', 'success')
            return redirect(url_for('view_job', job_id=new_job.id))
            
        except Exception as e:
            logger.error(f"Error creating training job: {str(e)}")
            flash(f'Error starting training job: {str(e)}', 'error')
            return redirect(request.url)
    
    # GET request - show the configuration form
    supported_models = app.config.get('SUPPORTED_MODELS', {
        'yolo': ['yolov5s', 'yolov8m'],
        'rf-detr': ['rf_detr_r50']
    })
    
    return render_template('configure.html', 
                          dataset=dataset,
                          supported_models=supported_models,
                          default_epochs=app.config.get('DEFAULT_EPOCHS', 100),
                          default_batch_size=app.config.get('DEFAULT_BATCH_SIZE', 16),
                          default_learning_rate=app.config.get('DEFAULT_LEARNING_RATE', 0.001))

@app.route('/jobs')
def list_jobs():
    """Display all training jobs."""
    jobs = TrainingJob.query.order_by(TrainingJob.created_at.desc()).all()
    return render_template('jobs.html', jobs=jobs)

@app.route('/job/<int:job_id>')
def view_job(job_id):
    """View details of a specific training job."""
    job = TrainingJob.query.get_or_404(job_id)
    return render_template('job_detail.html', job=job)

@app.route('/results/<int:job_id>')
def view_results(job_id):
    """View training results and metrics for a job."""
    job = TrainingJob.query.get_or_404(job_id)
    models = Model.query.filter_by(training_job_id=job_id).all()
    
    return render_template('results.html', job=job, models=models)

@app.route('/mlflow')
def mlflow_dashboard():
    """Interface for MLFlow integration."""
    mlflow_url = app.config.get('MLFLOW_TRACKING_URI')
    return render_template('mlflow.html', mlflow_url=mlflow_url)

@app.route('/dagster')
def dagster_dashboard():
    """Interface for Dagster integration."""
    dagster_url = app.config.get('DAGSTER_API_URL')
    return render_template('dagster.html', dagster_url=dagster_url)

@app.route('/job_status/<int:job_id>')
def job_status(job_id):
    """Return the current status of a job as JSON."""
    job = TrainingJob.query.get_or_404(job_id)
    return jsonify({
        'status': job.status,
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'duration': str(job.completed_at - job.started_at) if job.completed_at and job.started_at else None
    })

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

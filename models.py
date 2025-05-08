import datetime
from app import db
from flask_login import UserMixin
from sqlalchemy.dialects.sqlite import JSON

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    
    # Relationships
    datasets = db.relationship('Dataset', backref='owner', lazy='dynamic')
    jobs = db.relationship('TrainingJob', backref='owner', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    format = db.Column(db.String(64))  # COCO, YOLO, Pascal VOC, etc.
    path = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    size = db.Column(db.Integer)  # Size in bytes
    num_classes = db.Column(db.Integer)
    num_images = db.Column(db.Integer)
    class_names = db.Column(JSON)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    # Relationships
    jobs = db.relationship('TrainingJob', backref='dataset', lazy='dynamic')
    
    def __repr__(self):
        return f'<Dataset {self.name}>'

class TrainingJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    model_type = db.Column(db.String(64), nullable=False)  # YOLO, RF-DETR
    status = db.Column(db.String(32), default='pending')  # pending, running, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    parameters = db.Column(JSON)  # Training hyperparameters
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    mlflow_experiment_id = db.Column(db.String(128))
    mlflow_run_id = db.Column(db.String(128))
    dagster_run_id = db.Column(db.String(128))
    
    # Relationships
    models = db.relationship('Model', backref='job', lazy='dynamic')
    
    def __repr__(self):
        return f'<TrainingJob {self.name} ({self.status})>'

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    type = db.Column(db.String(64), nullable=False)  # YOLO, RF-DETR
    version = db.Column(db.String(32))
    path = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    metrics = db.Column(JSON)  # Performance metrics
    training_job_id = db.Column(db.Integer, db.ForeignKey('training_job.id'))
    
    def __repr__(self):
        return f'<Model {self.name} v{self.version}>'

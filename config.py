import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SESSION_SECRET', 'dev_key')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///ml_pipeline.db')
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max upload size
    
    # MLFlow configuration
    MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    MLFLOW_EXPERIMENT_NAME = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'object-detection-training')
    
    # Dagster configuration
    DAGSTER_HOME = os.environ.get('DAGSTER_HOME', os.path.join(os.getcwd(), 'dagster_home'))
    DAGSTER_API_URL = os.environ.get('DAGSTER_API_URL', 'http://localhost:3000')
    
    # Model training defaults
    DEFAULT_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_LEARNING_RATE = 0.001
    
    # Model architectures
    SUPPORTED_MODELS = {
        'yolo': ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        'rf-detr': ['rf_detr_r50', 'rf_detr_r101']
    }
    
    # Dataset formats
    SUPPORTED_FORMATS = ['COCO', 'YOLO', 'Pascal VOC']
    
    # File upload settings
    ALLOWED_EXTENSIONS = {'zip', 'tar', 'gz', 'json', 'yaml', 'yml'}

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    
# Dictionary for environment-based configuration
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get the appropriate configuration based on environment."""
    return config[os.environ.get('FLASK_ENV', 'default')]

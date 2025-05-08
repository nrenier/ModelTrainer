import os
import logging
import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///ml_pipeline.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with SQLAlchemy
db.init_app(app)

# Configure file uploads
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload size
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Add context processor for templates
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# Import and register routes after app is created
with app.app_context():
    # Import models and routes
    from models import User, Dataset, TrainingJob, Model  # noqa: F401
    import routes  # noqa: F401
    from api import api_bp  # noqa: F401

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Create database tables
    db.create_all()
    
    logger.info("Database tables created successfully")

"""
Main application initialization module.
Sets up Flask app with all necessary configurations and extensions.
"""
from flask import Flask
from flask_cors import CORS
from pathlib import Path
import logging
from pythonjsonlogger import jsonlogger
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging():
    """
    Set up JSON logging configuration
    @returns: None
    """
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    log_handler.setFormatter(formatter)
    logging.getLogger().addHandler(log_handler)
    logging.getLogger().setLevel(logging.INFO)

def create_app(config_name=None):
    """
    Create and configure the Flask application
    @param config_name: str - Name of the configuration to use
    @returns: Flask - Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Setup logging
    setup_logging()
    
    # Ensure upload directory exists
    upload_dir = Path(app.root_path) / 'static' / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure upload folder
    app.config['UPLOAD_FOLDER'] = str(upload_dir)
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 16MB max file size
    
    # Register blueprints with error handling
    try:
        from .controllers.rag_controller import rag_bp
        app.register_blueprint(rag_bp, url_prefix='/api')
        
        # Add a simple health check route
        @app.route('/health', methods=['GET'])
        def health_check():
            return {'status': 'healthy'}, 200
        
    except Exception as e:
        logging.error(f"Error registering blueprints: {str(e)}")
        raise
    
    return app

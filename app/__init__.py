"""
Main application initialization module.
Sets up Flask app with all necessary configurations and extensions.
"""
from flask import Flask
from flask_cors import CORS
from pathlib import Path
import os
from dotenv import load_dotenv
import sys
import logging
from flask_caching import Cache


# Load environment variables
load_dotenv()

def create_app(config_name=None):
    """
    Create and configure the Flask application
    @param config_name: str - Name of the configuration to use
    @returns: Flask - Configured Flask application instance
    """
    app = Flask(__name__)
    
    CORS(app, resources={
        r"/api/*": {
            "origins": ["https://booktube-opal.vercel.app", "http://localhost:5173"]
        }
        })
    
    # Ensure upload directory exists
    upload_dir = Path(app.root_path) / 'static' / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure upload folder
    app.config['UPLOAD_FOLDER'] = str(upload_dir)
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  
    
    # Configure watchdog to ignore PyTorch library files
    if sys.platform == 'win32':
        import watchdog.observers
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class IgnorePyTorchHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if 'torch' in event.src_path:
                    return
                super().on_modified(event)
        
        observer = Observer()
        observer.schedule(IgnorePyTorchHandler(), path=sys.prefix, recursive=True)
        observer.start()
    
    # Register blueprints with error handling
    try:
        from .controllers.rag_controller import rag_bp
        app.register_blueprint(rag_bp, url_prefix='/api')
        logging.info("Successfully registered RAG blueprint")
        
        from .controllers.course_controller import course_bp
        app.register_blueprint(course_bp, url_prefix='/api')
        logging.info("Successfully registered course blueprint")
        
        from .controllers.directory_controller import directory_bp
        app.register_blueprint(directory_bp, url_prefix='/api')
        logging.info("Successfully registered directory blueprint")
        
        from .controllers.rag_generation_controller import rag_generation_bp
        app.register_blueprint(rag_generation_bp, url_prefix='/api')
        logging.info("Successfully registered rag generation blueprint")
        
        from .controllers.recommendation_controller import recommendation_bp
        app.register_blueprint(recommendation_bp, url_prefix='/api')
        logging.info("Successfully registered recommendation blueprint")
        
        # Add a simple health check route
        @app.route('/health', methods=['GET'])
        def health_check():
            return {'status': 'healthy'}, 200
        
    except Exception as e:
        logging.error(f"Error registering blueprints: {str(e)}")
        raise
    
    return app

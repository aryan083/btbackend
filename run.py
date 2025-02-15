"""
Main entry point for the Flask application.
Supports both development and production server configurations.
"""
import os
import sys
import colorlog
import logging
from app import create_app
from config import Config

def setup_logging():
    """
    Configure colored logging for the application
    @returns: None
    """
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        Config.LOG_FORMAT,
        log_colors=Config.LOG_COLORS
    ))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

def run_development_server():
    """
    Run the Flask development server
    @returns: None
    """
    app = create_app()
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=True
    )

def run_production_server():
    """
    Run the production server using gunicorn/uvicorn
    @returns: None
    """
    try:
        import gunicorn.app.base
        
        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        app = create_app()
        
        options = {
            'bind': f"{Config.HOST}:{Config.PORT}",
            'workers': 4,
            'worker_class': 'uvicorn.workers.UvicornWorker',
            'timeout': 120,
            'reload': False,
            'accesslog': '-',
            'errorlog': '-',
            'loglevel': 'info'
        }
        
        StandaloneApplication(app, options).run()
        
    except ImportError as e:
        logging.error(f"Failed to import gunicorn: {str(e)}")
        logging.error("Please install gunicorn and uvicorn for production deployment")
        sys.exit(1)

if __name__ == '__main__':
    # Setup logging
    setup_logging()
    
    # Determine environment
    env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'production':
        logging.info("Starting production server...")
        run_production_server()
    else:
        logging.info("Starting development server...")
        run_development_server()

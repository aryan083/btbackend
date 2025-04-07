"""
Application entry point with environment-specific server configuration
"""
import os
import sys
import logging
import colorlog
from app import create_app

def setup_logging():
    """Configure colored logging"""
    # Create a color formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a stream handler
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def run_development_server():
    """Run the development server with debug mode"""
    try:
        app = create_app()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logging.error(f"Failed to start development server: {str(e)}")
        sys.exit(1)

def run_production_server():
    """Run the production server based on the operating system"""
    try:
        app = create_app()
        
        if sys.platform == 'win32':
            # Windows: Use waitress
            try:
                from waitress import serve
                logging.info("Starting production server with waitress...")
                serve(app, host='0.0.0.0', port=5000)
            except ImportError:
                logging.error("Please install waitress for Windows production deployment")
                logging.error("Run: pip install waitress")
                sys.exit(1)
        else:
            # Unix/Linux: Use gunicorn
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

                options = {
                    'bind': '0.0.0.0:5000',
                    'workers': 4,
                    'worker_class': 'uvicorn.workers.UvicornWorker'
                }

                logging.info("Starting production server with gunicorn...")
                StandaloneApplication(app, options).run()
                
            except ImportError:
                logging.error("Failed to import gunicorn or uvicorn")
                logging.error("Please install gunicorn and uvicorn for Unix/Linux production deployment")
                logging.error("Run: pip install gunicorn uvicorn")
                sys.exit(1)
                
    except Exception as e:
        logging.error(f"Failed to start production server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    # Setup colored logging
    setup_logging()
    
    if os.environ.get('FLASK_ENV') == 'development':
        logging.info("Starting development server...")
        run_development_server()
    else:
        logging.info("Starting production server...")
        run_production_server()

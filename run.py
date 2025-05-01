"""
Application entry point with environment-specific server configuration
"""
import os
import sys
import logging
import colorlog
import functools
import inspect
import time
# from datetime import datetime
from app import create_app  
from flask_cors import CORS

class CustomLogger:
    """
    Custom logger class to handle detailed function logging with terminal output only
    """
    def __init__(self):
        # Create logger
        self.logger = logging.getLogger('DetailedLogger')
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler with color formatting
        console_handler = colorlog.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Create detailed color formatter
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s "
            "%(blue)s%(name)s %(bold_white)s%(funcName)s:%(lineno)d%(reset)s - "
            "%(message_log_color)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={
                'message': {
                    'DEBUG': 'cyan',
                    'INFO': 'white',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            }
        )
        
        console_handler.setFormatter(console_formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)

    def log_function_call(self, func):
        """Decorator to log function calls with timing and parameters"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function details
            func_name = func.__name__
            module_name = func.__module__
            file_name = inspect.getfile(func)
            line_no = inspect.getsourcelines(func)[1]
            
            # Get caller information
            caller_frame = inspect.currentframe().f_back
            caller_info = ""
            if caller_frame:
                caller_info = f"called from {caller_frame.f_code.co_name} at line {caller_frame.f_lineno}"
            
            # Log function entry
            self.logger.info(
                f"→ Entering {func_name} "
                f"[{os.path.basename(file_name)}:{line_no}] {caller_info}"
            )
            
            # Log parameters if any
            if args or kwargs:
                params = []
                if args:
                    params.append(f"args: {args}")
                if kwargs:
                    params.append(f"kwargs: {kwargs}")
                self.logger.debug(f"Parameters: {', '.join(params)}")
            
            # Execute function and time it
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # Log successful execution
                self.logger.info(
                    f"← Completed {func_name} in {execution_time:.2f}ms"
                )
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self.logger.error(
                    f"✕ Error in {func_name} after {execution_time:.2f}ms: "
                    f"{str(e)}", exc_info=True
                )
                raise
            
        return wrapper

# Initialize the custom logger
custom_logger = CustomLogger()

# @custom_logger.log_function_call
def setup_logging():
    """Configure colored logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

@custom_logger.log_function_call
def run_development_server():
    """Run the development server with debug mode and hot reloading"""
    try:
        app = create_app()
        
        # Watch all Python files in the app directory
        extra_dirs = ['app']
        extra_files = []
        for extra_dir in extra_dirs:
            for dirname, dirs, files in os.walk(extra_dir):
                for filename in files:
                    filename = os.path.join(dirname, filename)
                    if os.path.isfile(filename):
                        extra_files.append(filename)
        
        custom_logger.logger.info("Starting development server with hot reloading enabled...")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False,
            extra_files=extra_files
        )
    except Exception as e:
        custom_logger.logger.error(f"Failed to start development server: {str(e)}")
        sys.exit(1)

@custom_logger.log_function_call
def run_production_server():
    """Run the production server based on the operating system"""
    try:
        app = create_app()
        
        
        if sys.platform == 'win32':
            # Windows: Use waitress
            try:
                from waitress import serve
                custom_logger.logger.info("Starting production server with waitress...")
                serve(app, host='0.0.0.0', port=5000)
            except ImportError:
                custom_logger.logger.error("Please install waitress for Windows production deployment")
                custom_logger.logger.error("Run: pip install waitress")
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
                
                
                custom_logger.logger.info("Starting production server with gunicorn...")
                StandaloneApplication(app, options).run()
                
            except ImportError:
                custom_logger.logger.error("Failed to import gunicorn or uvicorn")
                custom_logger.logger.error("Please install gunicorn and uvicorn for Unix/Linux production deployment")
                custom_logger.logger.error("Run: pip install gunicorn uvicorn")
                sys.exit(1)
                
    except Exception as e:
        custom_logger.logger.error(f"Failed to start production server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    setup_logging()
    run_development_server()

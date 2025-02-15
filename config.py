"""
Configuration settings for the application
"""
from pathlib import Path

class Config:
    # Base directory
    BASE_DIR = Path(__file__).parent
    
    # Output directory for processed files
    OUTPUT_FOLDER = BASE_DIR / "output"
    UPLOAD_FOLDER = BASE_DIR / "app" / "static" / "uploads"
    
    # Create necessary directories
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 5000
    
    # Logging settings
    LOG_FORMAT = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"
    LOG_COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }

"""
Configuration file for the RAG system application.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration class."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    
    # Storage Configuration
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    BOOK_PDF_FOLDER = os.getenv('BOOK_PDF_FOLDER', './bookpdf')
    SCANNED_PDF_FOLDER = os.getenv('SCANNED_PDF_FOLDER', './scannedpdf')
    TEMP_FOLDER = os.getenv('TEMP_FOLDER', './temp')
    
    # Create necessary folders if they don't exist
    os.makedirs(BOOK_PDF_FOLDER, exist_ok=True)
    os.makedirs(SCANNED_PDF_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 
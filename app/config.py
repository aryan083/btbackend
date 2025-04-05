"""
Configuration settings for the application
"""
import os
from dotenv import load_dotenv
from .env_config import BACKEND_URL, ENVIRONMENT, DEBUG

# Load environment variables
load_dotenv()

# Module-level configuration variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SUPABASE_URL = os.getenv('VITE_SUPABASE_URL')
SUPABASE_KEY = os.getenv('VITE_SUPABASE_ANON_KEY')

# File paths
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'uploads')

# RAG settings
MATCH_THRESHOLD = float(os.getenv('MATCH_THRESHOLD', '0.7'))
MAX_RESULTS = int(os.getenv('MAX_RESULTS', '5'))

class Config:
    """
    Configuration class for the application.
    Contains all necessary settings and environment variables.
    """
    
    # API Keys
    GEMINI_API_KEY = GEMINI_API_KEY
    SUPABASE_URL = SUPABASE_URL
    SUPABASE_KEY = SUPABASE_KEY
    
    # Environment settings
    ENVIRONMENT = ENVIRONMENT
    DEBUG = DEBUG
    API_BASE_URL = BACKEND_URL
    
    # File paths
    UPLOAD_FOLDER = UPLOAD_FOLDER
    OUTPUT_FOLDER = OUTPUT_FOLDER
    
    # RAG settings
    MATCH_THRESHOLD = MATCH_THRESHOLD
    MAX_RESULTS = MAX_RESULTS
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate that all required configuration values are set.
        Raises ValueError if any required value is missing.
        """
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        if not cls.SUPABASE_URL:
            raise ValueError("VITE_SUPABASE_URL environment variable is not set")
        if not cls.SUPABASE_KEY:
            raise ValueError("VITE_SUPABASE_ANON_KEY environment variable is not set")
        
        # Create necessary directories
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.OUTPUT_FOLDER, exist_ok=True)

# Validate configuration on import
Config.validate() 
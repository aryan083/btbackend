"""Environment-based configuration settings"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Backend URL configuration
BACKEND_URLS = {
    'development': 'http://localhost:5000',
    'production': 'https://btbackend.onrender.com'  # Replace with your actual render.com URL
}

# Get the appropriate backend URL based on environment
BACKEND_URL = BACKEND_URLS.get(ENVIRONMENT, BACKEND_URLS['development'])

# Other environment-specific configurations can be added here
DEBUG = ENVIRONMENT == 'development'
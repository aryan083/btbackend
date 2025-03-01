"""
Configuration settings for the application
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
if not os.getenv('VITE_SUPABASE_URL'):
    raise ValueError("VITE_SUPABASE_URL environment variable is not set")
# if not os.getenv('VITE_SUPABASE_ANON_KEY'):
#     raise ValueError("VITE_SUPABASE_ANON_KEY environment variable is not set")

# from http import client
from supabase import create_client, Client
import requests 
import json
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
from app.config import Config
import logging
import colorlog

# Set up colored logging
def setup_module_logging():
    """Configure colored logging for the module"""
    logger = logging.getLogger(__name__)
    
    # Only add handler if logger doesn't have handlers
    if not logger.handlers:
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

# Initialize logger
logger = setup_module_logging()

def init_gemini():
    """
    Initialize the Gemini AI model
    
    Returns:
        GenerativeModel: Initialized Gemini model instance
    """
    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {str(e)}")
        raise

def init_supabase():
    """
    Initialize Supabase client with proper error handling
    
    Returns:
        Client: Initialized Supabase client
    
    Raises:
        ValueError: If Supabase credentials are missing
        Exception: For other initialization errors
    """
    try:
        if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
            raise ValueError("Supabase URL or key is missing in configuration")
            
        logger.info(f"Initializing Supabase with URL: {Config.SUPABASE_URL}")
        supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        
        # Test the connection
        supabase.auth.get_session()  # This will raise an error if authentication fails
        return supabase
        
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {str(e)}")
        raise

def get_article_content(supabase, article_id):
    """
    Retrieve article content from Supabase
    
    Args:
        supabase (Client): Supabase client instance
        article_id (str): ID of the article to retrieve
        
    Returns:
        str: Article content or empty string if not found
    """
    try:
        if not article_id:
            raise ValueError("Article ID is required")
            
        response = (supabase.table("articles")
            .select("*")
            .eq("article_id", article_id)
            .execute())
        
        if not response.data:
            logger.warning(f"No article found with ID: {article_id}")
            return ""
            
        article_content = response.data[0].get('content_text', '')
        if not article_content:
            logger.warning(f"Article {article_id} has no content")
            
        return article_content
        
    except Exception as e:
        logger.error(f"Error getting article content: {str(e)}")
        raise

def get_image_query(client, article_content):
    """
    Generate image search query from article content
    
    Args:
        client: Gemini AI client
        article_content (str): Article content to analyze
        
    Returns:
        str: Generated search query
    """
    try:
        if not article_content:
            raise ValueError("Article content is required")
            
        prompt = """Generate a two to three word summary that can be used to query a relevant thumbnail image from the Unsplash API for the given content. 
        The output should be descriptive and focused on visually representing the concept in a way that aligns with themes when applicable. 
        Avoid literal interpretations that may return unrelated results. 
        Do not use any formatting or special characters in the output—only return the words."""
        
        response = client.generate_content([prompt, article_content])
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Failed to generate image query: {str(e)}")
        raise

def query_unsplash(query, num_images=1):
    """
    Query Unsplash API for images
    
    Args:
        query (str): Search query
        num_images (int): Number of images to retrieve
        
    Returns:
        str: URL of the first matching image
    """
    try:
        if not Config.UNSPLASH_API_KEY:
            raise ValueError("Unsplash API key is missing")
            
        url = f"https://api.unsplash.com/search/photos"
        params = {
            "query": query,
            "per_page": num_images,
            "client_id": Config.UNSPLASH_API_KEY
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        if not data.get("results"):
            logger.warning(f"No images found for query: {query}")
            return ""
            
        return data["results"][0]["urls"]["regular"]
        
    except Exception as e:
        logger.error(f"Unsplash API error: {str(e)}")
        raise

def search_pexels(query):
    """
    Search for images on Pexels
    
    Args:
        query (str): Search query
        
    Returns:
        str: URL of the matching image
    """
    try:
        if not Config.PEXELS_API_KEY:
            raise ValueError("Pexels API key is missing")
            
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": Config.PEXELS_API_KEY}
        params = {"query": query, "per_page": 1}
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        if not data.get("photos"):
            logger.warning(f"No Pexels images found for query: {query}")
            return ""
            
        return data["photos"][0]["src"]["original"]
        
    except Exception as e:
        logger.error(f"Pexels API error: {str(e)}")
        raise

def get_best_image(client, image_links, article_content):
    """
    Determine the best matching image for the article
    
    Args:
        client: Gemini AI client
        image_links (list): List of image URLs to compare
        article_content (str): Article content to match against
        
    Returns:
        str: Index of the best matching image
    """
    try:
        if not image_links or len(image_links) < 2:
            return "0"
            
        prompt = """You are given two image URLs and one HTML article or text passage. 
        Analyze the content of the article and determine which image is the most contextually and semantically relevant to the article. 
        Return "0" if the first image is the best match, "1" if the second image is the best match. 
        Return only the number as a string — "0" or "1" — with no additional text or explanation."""
        
        response = client.generate_content([prompt, article_content, image_links[0], image_links[1]])
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error determining best image: {str(e)}")
        return "0"  # Default to first image on error

def append_link_in_supabase(supabase, article_id, image_url):
    """
    Update article with image URL in Supabase
    
    Args:
        supabase (Client): Supabase client
        article_id (str): Article ID to update
        image_url (str): Image URL to add
    """
    try:
        if not article_id or not image_url:
            raise ValueError("Article ID and image URL are required")
            
        response = (supabase.table("articles")
            .update({"content_img": image_url})
            .eq("article_id", article_id)
            .execute())
            
        if not response.data:
            raise Exception("Failed to update article")
            
        logger.info(f"Successfully updated article {article_id} with image")
        
    except Exception as e:
        logger.error(f"Failed to update article in Supabase: {str(e)}")
        raise

def unsplash_api_fetcher(article_id):
    """
    Main function to fetch and attach images to articles
    
    Args:
        article_id (str): ID of the article to process
    """
    try:
        # Initialize clients
        supabase = init_supabase()
        client = init_gemini()
        
        # Get article content
        article_content = get_article_content(supabase, article_id)
        if not article_content:
            raise ValueError(f"No content found for article {article_id}")
        
        # Generate image search query
        query = get_image_query(client, article_content)
        logger.info(f"Generated image query: {query}")
        
        # Get images from different sources
        image_links = []
        
        # Get Unsplash image
        unsplash_link = query_unsplash(query)
        if unsplash_link:
            image_links.append(unsplash_link)
            
        # Get Pexels image
        pexels_link = search_pexels(query)
        if pexels_link:
            image_links.append(pexels_link)
            
        if not image_links:
            raise ValueError("No images found from any source")
            
        # Determine best image
        index = get_best_image(client, image_links, article_content)
        selected_image = image_links[int(index)]
        
        # Update article with selected image
        append_link_in_supabase(supabase, article_id, selected_image)
        logger.info(f"Successfully processed article {article_id}")
        
    except Exception as e:
        logger.error(f"Failed to process article {article_id}: {str(e)}")
        raise


def search_image(query):
    # Your Shutterstock API credentials
    api_key = Config.SHUTTERSTOCK_API_KEY
    api_secret = Config.SHUTTERSTOCK_API_SECRET
    
    # Get access token
    token_url = 'https://api.shutterstock.com/v2/oauth/access_token'
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': api_key,
        'client_secret': api_secret
    }
    token_response = requests.post(token_url, data=token_data)
    token_response.raise_for_status()
    access_token = token_response.json().get('access_token')

    # Search for images
    search_url = 'https://api.shutterstock.com/v2/images/search'
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'query': query, 'per_page': 1}
    search_response = requests.get(search_url, headers=headers, params=params)
    search_response.raise_for_status()
    data = search_response.json()

    if data['data']:
        # print(data)
        return data['data'][0]['assets']['huge_thumb']['url']
    else:
        return 'No images found.'











    
        









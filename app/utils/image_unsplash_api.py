# from http import client
from supabase import create_client, Client
import requests 
import json
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
from app.config import Config
import logging
from run import custom_logger
from pydantic import BaseModel
from typing import List, Optional

from app.utils.course_utils import get_course_articles
from app.utils.supabase_utils import bulk_update

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@custom_logger.log_function_call
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

@custom_logger.log_function_call
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

# def get_article_content(supabase, article_id):
#     """
#     Retrieve article content from Supabase
    
#     Args:
#         supabase (Client): Supabase client instance
#         article_id (str): ID of the article to retrieve
        
#     Returns:
#         str: Article content or empty string if not found
#     """
#     try:
#         if not article_id:
#             raise ValueError("Article ID is required")
            
#         response = (supabase.table("articles")
#             .select("*")
#             .eq("article_id", article_id)
#             .execute())
        
#         if not response.data:
#             logger.warning(f"No article found with ID: {article_id}")
#             return ""
            
#         article_content = response.data[0].get('content_text', '')
#         if not article_content:
#             logger.warning(f"Article {article_id} has no content")
            
#         return article_content
        
#     except Exception as e:
#         logger.error(f"Error getting article content: {str(e)}")
#         raise

@custom_logger.log_function_call
def get_image_query(client, article_content: str) -> Optional[str]:
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
            
        prompt = """Analyze the following article content and generate a two to three word summary that can be used to query a relevant thumbnail image from the Unsplash API.
        The output should be descriptive and focused on visually representing the concept.
        Return ONLY the search query words, nothing else.
        
        Article content:
        {content}
        """.format(content=article_content)
        
        response = client.generate_content(
            contents=[prompt],
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        
        if not response.text:
            logger.error("Empty response from Gemini for image query")
            return None
            
        # Clean the response to get just the search query
        query = response.text.strip()
        # Remove any markdown formatting or quotes
        query = query.replace('"', '').replace('*', '').replace('#', '').strip()
        # If multiple lines, take the first one
        if '\n' in query:
            query = query.split('\n')[0].strip()
            
        logger.info(f"Generated image query: {query}")
        return query
        
    except Exception as e:
        logger.error(f"Failed to generate image query: {str(e)}")
        return None

@custom_logger.log_function_call
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

# def search_pexels(query):
#     """
#     Search for images on Pexels
    
#     Args:
#         query (str): Search query
        
#     Returns:
#         str: URL of the matching image
#     """
#     try:
#         if not Config.PEXELS_API_KEY:
#             logger.warning("Pexels API key is missing")
#             return ""
            
#         url = "https://api.pexels.com/v1/search"
#         headers = {"Authorization": Config.PEXELS_API_KEY}
#         params = {"query": query, "per_page": 1}
        
#         response = requests.get(url, headers=headers, params=params)
        
#         # Check if the request was successful
#         if response.status_code != 200:
#             logger.warning(f"Pexels API returned status code {response.status_code}")
#             return ""
            
#         data = response.json()
#         if not data.get("photos"):
#             logger.warning(f"No Pexels images found for query: {query}")
#             return ""
            
#         return data["photos"][0]["src"]["original"]
        
#     except Exception as e:
#         logger.warning(f"Pexels API error: {str(e)}")
#         return ""

# def get_best_image(client, image_links, article_content):
#     """
#     Determine the best matching image for the article
    
#     Args:
#         client: Gemini AI client
#         image_links (list): List of image URLs to compare
#         article_content (str): Article content to match against
        
#     Returns:
#         str: Index of the best matching image
#     """
#     try:
#         if not image_links or len(image_links) < 2:
#             return "0"
            
#         prompt = """You are given two image URLs and one HTML article or text passage. 
#         Analyze the content of the article and determine which image is the most contextually and semantically relevant to the article. 
#         Return "0" if the first image is the best match, "1" if the second image is the best match. 
#         Return only the number as a string — "0" or "1" — with no additional text or explanation."""
        
#         response = client.generate_content([prompt, article_content, image_links[0], image_links[1]])
#         return response.text.strip()
        
#     except Exception as e:
#         logger.error(f"Error determining best image: {str(e)}")
#         return "0"  # Default to first image on error

@custom_logger.log_function_call
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
            
        # Use bulk update with a single record
        result = bulk_update(
            supabase,
            "articles",
            [{"article_id": article_id, "content_img": image_url}],
            id_field="article_id"
        )
            
        if result["success_count"] > 0:
            logger.info(f"Successfully updated article {article_id} with image")
        else:
            raise Exception("Failed to update article")
            
    except Exception as e:
        logger.error(f"Failed to update article in Supabase: {str(e)}")
        raise

@custom_logger.log_function_call
def bulk_append_links_in_supabase(supabase, article_updates):
    """
    Update multiple articles with image URLs in Supabase in bulk
    
    Args:
        supabase (Client): Supabase client
        article_updates (List[Dict[str, str]]): List of article updates with format
            [{"article_id": "id1", "image_url": "url1"}, ...]
            
    Returns:
        Dict[str, Any]: Result of the operation with success count and errors
    """
    try:
        if not article_updates:
            logger.warning("No article updates provided")
            return {"success_count": 0, "errors": ["No article updates provided"]}
            
        # Format updates for bulk operation
        formatted_updates = [
            {"article_id": update["article_id"], "content_img": update["image_url"]}
            for update in article_updates
        ]
        
        # Perform bulk update
        result = bulk_update(
            supabase,
            "articles",
            formatted_updates,
            id_field="article_id"
        )
        
        if result["success_count"] > 0:
            logger.info(f"Successfully updated {result['success_count']} articles with images")
        else:
            logger.warning("No articles were updated")
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to bulk update articles in Supabase: {str(e)}")
        return {"success_count": 0, "errors": [str(e)]}

class ArticleTitle(BaseModel):
    title: str
    keywords: List[str]

@custom_logger.log_function_call
def generate_article_title(client, content: str) -> Optional[str]:
    """
    Generate an engaging and SEO-friendly title for an article using Gemini.
    
    Args:
        client: Gemini AI client
        content (str): Article content to analyze
        
    Returns:
        Optional[str]: Generated title or None if generation fails
    """
    try:
        if not content:
            raise ValueError("Article content is required")
            
        prompt = """Generate an engaging and SEO-friendly title for the following article content.
        The title should be around max 10 words long and include relevant keywords.
        Make it attention-grabbing while maintaining accuracy.
        make the article title such that from outside no one can figure out what the article is about but it will relate very well to the article content once the user starts reading it, try to make it some what cool and less in length.
        Return ONLY the title text, nothing else.
        
        Article content:
        {content}
        """.format(content=content)
        
        response = client.generate_content(
            contents=[prompt],
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        
        if not response.text:
            logger.error("Empty response from Gemini for title")
            return None
            
        # Clean the response to get just the title
        title = response.text.strip()
        # Remove any markdown formatting or quotes
        title = title.replace('"', '').replace('*', '').replace('#', '').strip()
        # If multiple lines, take the first one
        if '\n' in title:
            title = title.split('\n')[0].strip()
            
        logger.info(f"Generated title: {title}")
        return title
            
    except Exception as e:
        logger.error(f"Failed to generate article title: {str(e)}")
        return None

@custom_logger.log_function_call
def unsplash_api_fetcher(course_id: str):
    """
    Main function to fetch and attach images to articles using existing data
    
    Args:
        course_id (str): ID of the course to process 
    """
    try:
        # Initialize clients
        supabase = init_supabase()
        client = init_gemini()
        
        # Get all articles for the course in a single query
        response = get_course_articles(course_id)

        if not response:
            logger.warning(f"No articles found for course {course_id}")
            return

        # Process articles in batch
        success_count = 0
        article_updates = []
        
        for article in response:
            article_id = None
            try:
                # Unpack tuple if necessary
                if isinstance(article, tuple):
                    article = dict(article)

                article_id = article.get('article_id')
                content = article.get('content_text', '')
                
                if not content:
                    logger.warning(f"Skipping empty content for {article_id}")
                    continue

                # Generate and apply image and title
                query = get_image_query(client, content)
                if not query:
                    logger.warning(f"Failed to generate image query for article {article_id}")
                    continue
                    
                unsplash_link = query_unsplash(query)
                if not unsplash_link:
                    logger.warning(f"Failed to get image for article {article_id}")
                    continue
                    
                # Generate title
                new_title = generate_article_title(client, content)
                if not new_title:
                    logger.warning(f"Failed to generate title for article {article_id}")
                    continue
                    
                # Add to batch for bulk update
                article_updates.append({
                    "article_id": article_id,
                    "article_name": new_title,
                    "content_img": unsplash_link
                })
                success_count += 1
                logger.info(f"Prepared update for {article_id} with title: {new_title} and image: {unsplash_link}")

            except Exception as e:
                logger.error(f"Failed article {article_id}: {str(e)}")
                continue
        
        # Perform bulk update if we have updates
        if article_updates:
            try:
                result = bulk_update(
                    supabase,
                    "articles",
                    article_updates,
                    "article_id"
                )
                
                if result["success_count"] > 0:
                    logger.info(f"Successfully updated {result['success_count']} articles with titles and images")
                else:
                    logger.error("No articles were updated - Check if article_id exists and is valid")
                
                if result.get("errors"):
                    for error in result["errors"]:
                        logger.error(f"Bulk update error: {error}")
                        
            except Exception as e:
                logger.error(f"Failed to perform bulk update: {str(e)}")

        logger.info(f"Processed {len(response)} articles | {success_count} successes")

    except Exception as e:
        logger.error(f"Course processing failed: {str(e)}")
        raise

# def unsplash_api_fetcher(article_id):
#     """
#     Main function to fetch and attach images to articles
    
#     Args:
#         article_id (str): ID of the article to process
#     """
#     try:
#         # Initialize clients
#         supabase = init_supabase()
#         client = init_gemini()
        
#         # Get article content
#         article_content = get_article_content(supabase, article_id)
#         if not article_content:
#             logger.warning(f"No content found for article {article_id}")
#             return
        
#         # Generate image search query
#         query = get_image_query(client, article_content)
#         logger.info(f"Generated image query: {query}")
        
#         # Get images from different sources
#         image_links = []
        
#         # Get Unsplash image
#         unsplash_link = query_unsplash(query)
#         if unsplash_link:
#             image_links.append(unsplash_link)
#             logger.info(f"Found Unsplash image for article {article_id}")
            
#         # Get Pexels image
#         # pexels_link = search_pexels(query)
#         # if pexels_link:
#         #     image_links.append(pexels_link)
#         #     logger.info(f"Found Pexels image for article {article_id}")
            
#         if not image_links:
#             logger.warning(f"No images found for article {article_id}")
#             return
            
#         # If we only have one image, use it directly
#         if len(image_links) == 1:
#             append_link_in_supabase(supabase, article_id, image_links[0])
#             logger.info(f"Successfully processed article {article_id} with single image")
#             return
            
#         # Determine best image
#         # index = get_best_image(client, image_links, article_content)
#         # selected_image = image_links[int(index)]
        
#         # # Update article with selected image
#         # append_link_in_supabase(supabase, article_id, selected_image)
#         # logger.info(f"Successfully processed article {article_id}")
        
#     except Exception as e:
#         logger.error(f"Failed to process article {article_id}: {str(e)}")
#         # Don't raise the exception to prevent the entire process from failing


# def search_image(query):
#     # Your Shutterstock API credentials
#     api_key = Config.SHUTTERSTOCK_API_KEY
#     api_secret = Config.SHUTTERSTOCK_API_SECRET
    
#     # Get access token
#     token_url = 'https://api.shutterstock.com/v2/oauth/access_token'
#     token_data = {
#         'grant_type': 'client_credentials',
#         'client_id': api_key,
#         'client_secret': api_secret
#     }
#     token_response = requests.post(token_url, data=token_data)
#     token_response.raise_for_status()
#     access_token = token_response.json().get('access_token')

#     # Search for images
#     search_url = 'https://api.shutterstock.com/v2/images/search'
#     headers = {'Authorization': f'Bearer {access_token}'}
#     params = {'query': query, 'per_page': 1}
#     search_response = requests.get(search_url, headers=headers, params=params)
#     search_response.raise_for_status()
#     data = search_response.json()

#     if data['data']:
#         # print(data)
#         return data['data'][0]['assets']['huge_thumb']['url']
#     else:
#         return 'No images found.'


# if __name__ == "__main__":
#     unsplash_api_fetcher("b6f57120-039a-464c-94b2-9a4533cc811a")
    
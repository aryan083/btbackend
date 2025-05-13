from app.config import Config
# from app.controllers.recommendation_controller import recommendation
import pandas as pd
from supabase import Client,create_client
import logging
import os 
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import random

logger = logging.getLogger(__name__)

class RShomeService:
    def __init__(self):
        self.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        return  
      
    def recommendation_algorithm(self, data: pd.DataFrame, article_id: str, top_k: int = 1) -> list:
        """
        Generate article recommendations based on content similarity.
        
        Args:
            data (pd.DataFrame): DataFrame containing article data with 'article_id' and 'content_text' columns
            article_id (str): ID of the reference article
            top_k (int): Number of recommendations to return (default: 1)
            
        Returns:
            list: List of recommended article IDs
        """
        try:
            logger.info(f"Starting recommendation algorithm for article_id: {article_id}")
            
            # Ensure data is not empty
            if data.empty:
                logger.error("Input DataFrame is empty")
                return []
            
            # Ensure required columns exist
            required_columns = ['article_id', 'content_text']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Required columns missing. Found columns: {data.columns.tolist()}")
                return []
            
            # Rename columns to match expected format
            data = data.rename(columns={
                'article_id': 'id',
                'content_text': 'description'
            })
            
            logger.info(f"Input DataFrame columns: {data.columns.tolist()}")
            logger.info(f"Input DataFrame shape: {data.shape}")
            
            if article_id not in data['id'].values:
                logger.error(f"Article ID {article_id} not found in DataFrame")
                return []
            
            # Initialize the model
            logger.info("Initializing SentenceTransformer model...")
            model_name = 'all-mpnet-base-v2'
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, model_name)
            
            # Try loading from local cache first
            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading embedding model from local cache: {model_path}")
                    model = SentenceTransformer(model_path, device='cpu')
                    # Verify model is loaded correctly
                    test_embedding = model.encode("test")
                    if test_embedding is not None and len(test_embedding) > 0:
                        logger.info("Successfully loaded embedding model from local cache")
                    else:
                        logger.warning("Local model returned empty embeddings, will try downloading fresh copy")
                        model = SentenceTransformer(model_name, device='cpu', cache_folder=cache_dir)
                except Exception as e:
                    logger.warning(f"Error loading model from local cache: {str(e)}. Will try downloading.")
                    model = SentenceTransformer(model_name, device='cpu', cache_folder=cache_dir)
            else:
                # If local loading failed or model doesn't exist locally, try downloading
                logger.info("Downloading embedding model from internet")
                model = SentenceTransformer(model_name, device='cpu', cache_folder=cache_dir)
            
            # Clean HTML content and fill NaNs or empty strings with placeholder
            data['description'] = data['description'].apply(lambda x: self._clean_html_content(x) if isinstance(x, str) else "")
            data['description'] = data['description'].fillna("")
            
            # Compute embeddings for all articles
            logger.info("Computing embeddings for all articles...")
            embeddings = model.encode(data['description'].tolist(), convert_to_tensor=True)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            
            # Find the index of the reference article
            reference_idx = data[data['id'] == article_id].index[0]
            reference_embedding = embeddings[reference_idx]
            logger.info(f"Found reference article at index: {reference_idx}")
            
            # Calculate cosine similarity with all other articles
            cosine_scores = util.cos_sim(reference_embedding, embeddings)[0]
            logger.info(f"Generated cosine scores shape: {cosine_scores.shape}")
            
            # Add similarity scores to DataFrame
            data['similarity_score'] = cosine_scores.cpu().numpy()
            
            # Sort by similarity (descending) and exclude the reference article
            data_sorted = data[data['id'] != article_id].sort_values(by='similarity_score', ascending=False)
            logger.info(f"Sorted DataFrame shape: {data_sorted.shape}")
            
            # Get top-k recommendations
            recommendation_list = data_sorted.head(top_k)['id'].tolist()
            logger.info(f"Generated recommendation list: {recommendation_list}")
            
            # Log the recommendations and their similarity scores
            logger.info("Top 5 recommended articles:")
            for idx, rec_id in enumerate(recommendation_list, 1):
                similarity = data_sorted[data_sorted['id'] == rec_id]['similarity_score'].values[0]
                logger.info(f"{idx}. Article ID: {rec_id}, Similarity Score: {similarity:.4f}")
            
            # Ensure we return a list even if empty
            return recommendation_list if recommendation_list else []
            
        except Exception as e:
            logger.error(f"Error in recommendation algorithm: {str(e)}")
            logger.error(f"DataFrame info:\n{data.info()}")
            logger.error(f"DataFrame head:\n{data.head()}")
            return []

    def _clean_html_content(self, html_content: str) -> str:
        """
        Clean HTML content by removing HTML tags and extra whitespace.
        
        Args:
            html_content (str): HTML content to clean
            
        Returns:
            str: Cleaned text content
        """
        try:
            # Remove HTML tags
            import re
            clean_text = re.sub(r'<[^>]+>', ' ', html_content)
            # Remove extra whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            return clean_text
        except Exception as e:
            logger.error(f"Error cleaning HTML content: {str(e)}")
            return ""

    def get_user_history(self, supabase, user_id):
        try:
            user_history = supabase.table('users')\
                .select('history')\
                .eq('user_id', user_id)\
                .execute()
            logger.info(f"User history: {user_history.data}")
            if not user_history.data:
                logger.info(f"No history found for user {user_id}")
                return []
                
            # Handle nested history structure
            history_data = user_history.data[0].get('history', {})
            if isinstance(history_data, dict) and 'history' in history_data:
                history_data = history_data['history']
            
            if not isinstance(history_data, dict):
                return []
                
            # Convert history dict to list of article IDs
            return [{'article_id': article_id} for article_id in history_data.values()]
        except Exception as e:
            logger.error(f"Error getting user history for user {user_id}: {str(e)}")
            return []
        
    def get_user_read_later(self, supabase, user_id):
        """
        Get user's read later articles from Supabase.
        
        Args:
            supabase: Supabase client instance
            user_id (str): User ID to fetch read later articles for
            
        Returns:
            list: List of read later articles
        """
        try:
            user_read_later = supabase.table('users')\
                .select('watch_later')\
                .eq('user_id', user_id)\
                .execute()
            logger.info(f"User read later: {user_read_later.data}")
            if not user_read_later.data:
                logger.info(f"No read later found for user {user_id}")
                return []
            watch_later_data = user_read_later.data[0].get('watch_later', [])
            if not isinstance(watch_later_data, list):
                return []
            # Ensure each item has article_id
            return [{'article_id': item['article_id']} if isinstance(item, dict) and 'article_id' in item 
                   else {'article_id': item} if isinstance(item, str) 
                   else {'article_id': str(item)} for item in watch_later_data]
        except Exception as e:
            logger.error(f"Error getting user read later for user {user_id}: {str(e)}")
            return []
        
    def get_article_data(self,supabase,user_id):
        try:
            article_data = supabase.table('articles')\
                .select('article_id,content_text')\
                .eq('user_id',user_id)\
                .execute()
            logger.info(f"Article data: {article_data.data}")
            if not article_data.data:
                logger.info(f"No article data found for user {user_id}")
                return []
            return article_data.data
        except Exception as e:
            logger.error(f"Error getting article data for user {user_id}: {str(e)}")
            return []
        
    def get_user_bookmarked(self, supabase, user_id):
        """
        Get user's bookmarked articles from Supabase.
        
        Args:
            supabase: Supabase client instance
            user_id (str): User ID to fetch bookmarked articles for
            
        Returns:
            list: List of bookmarked articles
        """
        try:
            user_bookmarked = supabase.table('users')\
                .select('bookmarked_articles')\
                .eq('user_id', user_id)\
                .execute()
            if not user_bookmarked.data:
                logger.info(f"No bookmarked articles found for user {user_id}")
                return []
            bookmarked_data = user_bookmarked.data[0].get('bookmarked_articles', [])
            if not isinstance(bookmarked_data, list):
                return []
            # Ensure each item has article_id
            return [{'article_id': item['article_id']} if isinstance(item, dict) and 'article_id' in item 
                   else {'article_id': item} if isinstance(item, str) 
                   else {'article_id': str(item)} for item in bookmarked_data]
        except Exception as e:
            logger.error(f"Error getting user bookmarked articles for user {user_id}: {str(e)}")
            return []

    def clean_recommendation(self,recommendation_list):
        #remove dublicate 
        seen = set()
        unique_recommendation = []
        for item in recommendation_list:
            if item not in seen:
                seen.add(item)
                unique_recommendation.append(item)
        return unique_recommendation
    
    def save_to_reccomendat_jsonindb(self, recommendation_list, user_id):
        """
        Save recommendations to database.
        
        Args:
            recommendation_list (list): List of article IDs to save
            user_id (str): User ID to save recommendations for
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not recommendation_list:
                logger.warning("No recommendations to save")
                return False
                
            supabase = self.supabase
            supabase.table('users').update({
                'recommended_json': recommendation_list
            }).eq('user_id', user_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error saving recommendations to database: {str(e)}")
            return False
    
    def remove_duplicates(self, bookmarked_articles, history, watch_later):
        """
        Remove duplicates from user's articles and combine them.
        
        Args:
            bookmarked_articles (list): List of bookmarked articles
            history (list): List of history items
            watch_later (list): List of watch later items
            
        Returns:
            list: Combined and deduplicated list of articles
        """
        try:
            # Ensure all inputs are lists
            bookmarked_articles = bookmarked_articles if isinstance(bookmarked_articles, list) else []
            history = history if isinstance(history, list) else []
            watch_later = watch_later if isinstance(watch_later, list) else []

            # Extract article IDs from all sources
            article_ids = set()
            unique_articles = []

            # Process bookmarked articles
            for article in bookmarked_articles:
                if isinstance(article, dict) and 'article_id' in article:
                    article_id = article['article_id']
                    if article_id not in article_ids:
                        article_ids.add(article_id)
                        unique_articles.append({'article_id': article_id})

            # Process history
            for article in history:
                if isinstance(article, dict) and 'article_id' in article:
                    article_id = article['article_id']
                    if article_id not in article_ids:
                        article_ids.add(article_id)
                        unique_articles.append({'article_id': article_id})

            # Process watch later
            for article in watch_later:
                if isinstance(article, dict) and 'article_id' in article:
                    article_id = article['article_id']
                    if article_id not in article_ids:
                        article_ids.add(article_id)
                        unique_articles.append({'article_id': article_id})

            # Take up to 20 articles
            return unique_articles[:20]
        except Exception as e:
            logger.error(f"Error in remove_duplicates: {str(e)}")
            return []

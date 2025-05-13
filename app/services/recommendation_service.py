from app.config import Config
# from app.controllers.recommendation_controller import recommendation
import pandas as pd
from supabase import Client,create_client
import logging
import os 
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Recommendation:
    def __init__(self):
        return
    @staticmethod
    def supabase_init():  
        supabase_url = Config.SUPABASE_URL
        supabase_key = Config.SUPABASE_KEY
        supabase = create_client(supabase_url, supabase_key)
        return supabase


    
    
    def recommendation_algorithm(self,data: pd.DataFrame, article_id: str, top_k: int = 3) -> list:
        try:
            logger.info(f"Starting recommendation algorithm for article_id: {article_id}")
            logger.info(f"Input DataFrame columns: {data.columns.tolist()}")
            logger.info(f"Input DataFrame shape: {data.shape}")
            logger.info(f"DataFrame head:\n{data.head()}")
            
            if data.empty:
                logger.error("Input DataFrame is empty")
                return []
                
            if article_id not in data['id'].values:
                logger.error(f"Article ID {article_id} not found in DataFrame")
                return []
            
            # Initialize the model
            logger.info("Initializing SentenceTransformer model...")
            model_name = 'all-mpnet-base-v2'
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_file_))), 'models')
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
            
            # Fill NaNs or empty strings with placeholder
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
            logger.info(f"Top 5 similarity scores:\n{data_sorted[['id', 'similarity_score']].head()}")
            
            # Get top-k recommendations
            recommendation_list = data_sorted.head(top_k)['id'].tolist()
            logger.info(f"Generated recommendation list: {recommendation_list}")
            
            # Log the recommendations and their similarity scores
            logger.info("Top 5 recommended articles:")
            for idx, rec_id in enumerate(recommendation_list, 1):
                similarity = data_sorted[data_sorted['id'] == rec_id]['similarity_score'].values[0]
                logger.info(f"{idx}. Article ID: {rec_id}, Similarity Score: {similarity:.4f}")
            
            return recommendation_list
            
        except Exception as e:
            logger.error(f"Error in recommendation algorithm: {str(e)}")
            logger.error(f"DataFrame info:\n{data.info()}")
            logger.error(f"DataFrame head:\n{data.head()}")
            return []

                
    def get_articles(self,supabase,user_id):
        try:
            # Get articles for the user
            articles = supabase.table('articles')\
                .select('article_id,content_text')\
                .eq('user_id', user_id)\
                .execute()
                
            if not articles.data:
                return []
                
            return articles.data
            
        except Exception as e:
            logger.error(f"Error getting articles for user {user_id}: {str(e)}")
            return []
        
    def format_articles(self,response):
        
        id = []
        description = []
        for article in response:

            id.append(article['article_id'])
            description.append(article['content_text'])

        formatted_response = pd.DataFrame({
            'id': id,
            'description': description
        })

        return formatted_response
    



from app.config import Config
# from app.controllers.recommendation_controller import recommendation
import pandas as pd
from supabase import Client,create_client
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Recommendation:
    def __init__(self):
        return

    def supabase_init():  
        supabase_url = Config.SUPABASE_URL
        supabase_key = Config.SUPABASE_KEY
        supabase = create_client(supabase_url, supabase_key)
        return supabase

    def recommendation_algorithm( data: pd.DataFrame, article_id: str) -> list:
        recommendation_list = ['1','2','3','4','5']
        print(article_id)
        return recommendation_list

    def get_articles(supabase,user_id):
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
        
    def format_articles(response):
        
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



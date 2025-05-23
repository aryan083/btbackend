import logging
from supabase import create_client
from app.config import Config

logger = logging.getLogger(__name__)

class ArticleService:
    @staticmethod
    def supabase_init():
        try:
            supabase_url = Config.SUPABASE_URL
            supabase_key = Config.SUPABASE_KEY
            supabase = create_client(supabase_url, supabase_key)
            return supabase
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {str(e)}")
            raise e

    def get_article_content(self, article_id):
        """
        Get the content of a specific article by its ID
        """
        try:
            supabase = self.supabase_init()
            
            # Get the specific article
            article = supabase.table('articles')\
                .select('content_text')\
                .eq('article_id', article_id)\
                .execute()
                
            if not article.data:
                logger.warning(f"No article found with ID {article_id}")
                return None
                
            # Return just the content text
            return article.data[0]['content_text']
            
        except Exception as e:
            logger.error(f"Error getting article content for article {article_id}: {str(e)}")
            return None

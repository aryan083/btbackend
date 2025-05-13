"""Article Recommendation Controller Module"""

from flask import Blueprint, request, jsonify
from app.services.article_recommendation_service import ArticleRecommendationService
import logging

logger = logging.getLogger(__name__)
article_recommendation_bp = Blueprint('article_recommendation', __name__)
article_recommendation_service = ArticleRecommendationService()
@article_recommendation_bp.route('/recommend_articles_by_user_activity', methods=['POST'])
@article_recommendation_bp.route('/recommend_articles_by_course', methods=['POST'])
@article_recommendation_bp.route('/recommend_articles', methods=['POST'])
@article_recommendation_bp.route('/recommend_articles_by_user_preferences', methods=['POST'])

def recommend_articles():
    """
    Recommend articles based on the current open article
    @param request: Flask request object containing user preferences
    @returns: JSON response with recommended articles or error
    """            
    try:
        data = request.get_json()
        user_preferences = data.get('user_preferences')
        recommended_articles = article_recommendation_service.recommend_articles(user_preferences)
        return jsonify(recommended_articles), 200
    except Exception as e:
        logger.error(f"Error recommending articles: {e}")
        return jsonify({"error": str(e)}), 500

                
                
                    
                    
                    
            

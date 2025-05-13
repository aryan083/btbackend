from flask import Blueprint, request, jsonify, current_app
from app.services.course_service import CourseService
import logging
from pathlib import Path
from markdownify import markdownify as md
from app.config import Config
from supabase import create_client, Client
from run import custom_logger
from app.services.recommendation_service import Recommendation
from flask_caching import Cache
from supabase import Client,create_client



logger = logging.getLogger(__name__)
recommendation_bp = Blueprint('recommendation', __name__)
@recommendation_bp.route('/recommendation', methods=['POST'])
@custom_logger.log_function_call




def recommendation():


    try:
        top_k = 5
        user_id = request.form.get("user_id")
        article_id = request.form.get("article_id")

        supabase = Recommendation.supabase_init()

        response = Recommendation.get_articles(supabase,user_id)
        formatted_response = Recommendation.format_articles(response)
        print(f"{formatted_response}")
        top_responses = Recommendation.recommendation_algorithm(formatted_response, article_id, top_k)

        return jsonify({
            'message': 'TOP IDs Fetched successfully',
            'data': top_responses
        }), 201
        
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        return jsonify({
            'error': 'Failed to run recommendation code.',
            'details': str(e)
        }), 500
        





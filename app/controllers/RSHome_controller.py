from flask import Blueprint, request, jsonify, current_app
from app.services.course_service import CourseService
import logging
from pathlib import Path
from markdownify import markdownify as md
from app.config import Config
from supabase import create_client, Client
from run import custom_logger
from app.services.RShome_service import RShomeService
import pandas as pd
# from flask_caching import Cache
from supabase import Client,create_client

logger = logging.getLogger(__name__)
recommendationHome_bp = Blueprint('recommendationHome', __name__)


@recommendationHome_bp.route('/recommendationHome', methods=['POST'])
@custom_logger.log_function_call
def recommendationHome():
    try:
        user_id = request.form.get("user_id") or "56a3cf1a-4914-4387-93c2-a6bbce6e236c"
        rs_service = RShomeService()
        supabase = rs_service.supabase
        user_history = rs_service.get_user_history(supabase,user_id) or []
        user_read_later = rs_service.get_user_read_later(supabase,user_id) or []
        user_bookmarked = rs_service.get_user_bookmarked(supabase,user_id) or []
        article_data = rs_service.get_article_data(supabase,user_id) or []
        data = pd.DataFrame(article_data)
        cleaned_user_data = rs_service.remove_duplicates(user_bookmarked,user_history,user_read_later)
        
        # Initialize empty list for recommendations
        rs_list = []
        
        # Process each item in cleaned_user_data
        for item in cleaned_user_data:
            if isinstance(item, dict) and 'article_id' in item:
                recommendations = rs_service.recommendation_algorithm(data, article_id=item['article_id'])
                if recommendations:  # Only append if we got recommendations
                    rs_list.extend(recommendations)
        
        # Remove duplicates from final list using article IDs
        seen_ids = set()
        unique_rs_list = []
        for article_id in rs_list:
            if article_id not in seen_ids:
                seen_ids.add(article_id)
                unique_rs_list.append(article_id)
        
        # Save to database
        if unique_rs_list:
            rs_service.save_to_reccomendat_jsonindb(unique_rs_list, user_id)
        
        return jsonify({
            'message': 'Recommendation Home Page Fetched successfully',
            'data': unique_rs_list
        }), 201
    except Exception as e:
        logger.error(f"Error in recommendationHome: {str(e)}")
        return jsonify({'error': str(e)}), 500

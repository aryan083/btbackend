from flask import Blueprint, request, jsonify
import logging
from app.services.QuestionAns import QuestionAnswerService, find_answer

logger = logging.getLogger(__name__)
qa_bp = Blueprint('qa', __name__)

@qa_bp.route('/question', methods=['POST'])
def answer_question():
    try:
        # Get article_id and question from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
            
        article_id = data.get('article_id')
        question = data.get('question')
        
        if not article_id:
            return jsonify({
                'error': 'article_id is required'
            }), 400
            
        if not question:
            return jsonify({
                'error': 'question is required'
            }), 400
            
        # Initialize QuestionAnswerService and set article content
        qa_service = QuestionAnswerService()
        if not qa_service.set_article_content_by_id(article_id):
            return jsonify({
                'error': 'Failed to load article content'
            }), 404
            
        # Get the article content
        article_content = qa_service.get_article_content()
        
        # Find answer
        answer = find_answer(question, article_content)
        
        return jsonify({
            'message': 'Answer generated successfully',
            'data': {
                'question': question,
                'answer': answer,
                'article_id': article_id
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return jsonify({
            'error': 'Failed to generate answer',
            'details': str(e)
        }), 500

"""
Controller module for handling RAG (Retrieval Augmented Generation) HTTP requests.
This module provides endpoints for generating HTML content using RAG techniques.
"""
import logging
from typing import Dict, Any, List
from flask import Blueprint, request, jsonify,current_app
from werkzeug.exceptions import HTTPException
from app.services.rag_generation_service import RAGGenerationService
from app.config import Config
from pathlib import Path
from werkzeug.utils import secure_filename
from app.services.pdf_processor import PDFParser as DocumentService
from app.utils.course_utils import get_course_detailed_data, get_course_summary,get_course_generation_data
from app.utils.image_unsplash_api import unsplash_api_fetcher
from run import custom_logger
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
rag_generation_bp = Blueprint('rag_generation', __name__)

# Initialize service
rag_service = RAGGenerationService(
    supabase_url=Config.SUPABASE_URL,
    supabase_key=Config.SUPABASE_KEY,
    gemini_api_key=Config.GEMINI_API_KEY
)

# @rag_generation_bp.route('/generate', methods=['POST'])
@custom_logger.log_function_call
def generate_content():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No request body provided"}), 400
            
        required_fields = ['document_dir', 'course_id', 'user_id']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # Dynamic paths based on course and user
        document_dir = data['document_dir']
        course_id = data['course_id']
        user_id = data['user_id']
        
        output_dir = Path(current_app.config['UPLOAD_FOLDER']) / course_id / user_id / 'generated'
        
        # Get topic IDs for the course
        topic_ids = get_course_detailed_data(user_id, course_id)
        logger.info(f"Topic IDs: {topic_ids}")
        if not topic_ids:
            return jsonify({
                "error": "No topics found for this course",
                "details": "Please check course configuration"
            }), 404
        course_summary = get_course_summary(user_id, course_id)
        logger.info(f"Course summary: {course_summary}")
        # Generate content with topic IDs
        generated_files = rag_service.generate_html_content(
            document_dir=document_dir,
            course_id=course_id,
            output_dir=str(output_dir),
            topic_ids=topic_ids,
            match_threshold=float(data.get('match_threshold', 0.7)),
            max_results=int(data.get('max_results', 5))
        )

        return jsonify({
            "message": "Content generated successfully",
            "generated_files": list(generated_files.keys()),
            "topic_ids": topic_ids
        }), 200

    except Exception as e:
        logger.error(f"Content generation failed: {str(e)}")
        return jsonify({
            "error": "Content generation failed",
            "details": str(e)
        }), 500

# @rag_generation_bp.route('/process', methods=['POST'])
@custom_logger.log_function_call
def process_content() -> tuple[Dict[str, Any], int]:
    """
    Process course content (text and images) for RAG generation.
    
    Request body:
        base_path (str): Base path containing Text and Images directories
        course_id (str): ID of the course
    
    Returns:
        tuple[Dict[str, Any], int]: Response containing processing results and status code
    """
    try:
        # Validate request body
        data = request.get_json()
        if not data:
            return jsonify({"error": "No request body provided"}), 400
            
        required_fields = ['base_path', 'course_id']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # Process content
        results = rag_service.process_content(
            base_path=data['base_path'],
            course_id=data['course_id']
        )

        if not results:
            return jsonify({
                "error": "Content processing failed",
                "details": "Check if the base path exists and contains valid content"
            }), 404

        return jsonify({
            "message": "Content processed successfully",
            "results": results
        }), 200

    except ValueError as e:
        logger.error(f"Invalid parameter value: {str(e)}")
        return jsonify({
            "error": "Invalid parameter value",
            "details": str(e)
        }), 400
    except HTTPException as e:
        logger.error(f"HTTP error: {str(e)}")
        return jsonify({
            "error": e.name,
            "details": e.description
        }), e.code
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@custom_logger.log_function_call
def process_article_images(article_ids: List[Dict[str, Any]], course_id: str) -> Dict[str, Any]:
    """
    Process article IDs and attach images to them using the unsplash_api_fetcher function.
    
    Args:
        article_ids (List[Dict[str, Any]]): List of article data including article_id and content_text
        course_id (str): ID of the course these articles belong to
        
    Returns:
        Dict[str, Any]: Results of image processing
    """
    results = {
        "processed": 0,
        "failed": 0,
        "errors": []
    }

    try:
        # Use the unsplash_api_fetcher to process all articles for this course
        logger.info(f"Processing images for course ID: {course_id}")
        unsplash_api_fetcher(course_id=course_id)
        
        # Since unsplash_api_fetcher processes all articles for the course,
        # we'll consider all articles as processed
        results["processed"] = len(article_ids)
        
    except Exception as e:
        error_msg = f"Failed to process images for course {course_id}: {str(e)}"
        logger.error(error_msg)
        results["errors"].append(error_msg)
        results["failed"] = len(article_ids)
    
    return results

@rag_generation_bp.route('/upload_and_process2', methods=['POST'])
@custom_logger.log_function_call
def upload_and_process_pdf():
    try:
        # Get required parameters
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file part in the request',
                'error_code': 'MISSING_FILE'
            }), 400

        file = request.files['file']
        course_id = request.form.get('course_id')
        user_id = request.form.get('user_id')
        document_type = request.form.get('document_type', 'book')
        document_name = request.form.get('document_name', '')

        document_dir = Path(current_app.config['UPLOAD_FOLDER']) / (document_name or secure_filename(file.filename).rsplit('.', 1)[0])

        # Validate required parameters
        if not all([course_id, user_id]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameters: course_id and user_id',
                'error_code': 'MISSING_PARAMS'
            }), 400

        # Create dynamic directory structure
        upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
        course_dir = upload_dir / course_id / user_id
        document_dir.mkdir(parents=True, exist_ok=True)

        # Save and process PDF
        file_path = document_dir / secure_filename(file.filename)
        file.save(str(file_path))
        
        # Process PDF
        service = DocumentService(str(file_path), str(document_dir))
        
        # Get course generation data including topic metadata and user preferences
        generation_data = get_course_generation_data(user_id, course_id)
        if not generation_data:
            return jsonify({
                'status': 'error',
                'message': 'No course generation data found',
                'error_code': 'NO_COURSE_DATA'
            }), 404
            
        # Extract necessary data
        topic_metadata = generation_data.get('topic_name_to_id', {})
        user_prompt = generation_data.get('user_prompt', '')
        teaching_pattern_list = generation_data.get('teaching_pattern', [])
        teaching_pattern = ', '.join(teaching_pattern_list)  # Convert list to string
        skill_level = generation_data.get('skill_level', '')

        if not topic_metadata:
            return jsonify({
                'status': 'error',
                'message': 'No topics found for this course',
                'error_code': 'NO_TOPICS'
            }), 404
            
        # Process the document
        result = service.process_document(
            # extract_images=request.form.get('extract_images', 'true').lower() == 'true',
            extract_images=False,
            extract_text=request.form.get('extract_text', 'true').lower() == 'true',
            save_json=request.form.get('save_json', 'true').lower() == 'true'
        )

        if result['status'] == 'success':
            try:
                # Process content for RAG
                process_result = rag_service.process_content(
                    base_path=str(document_dir),
                    course_id=course_id
                )

                # Generate HTML content with all necessary parameters
                generated_result = rag_service.generate_html_content(
                document_dir=str(document_dir),
                course_id=course_id,
                output_dir=str(document_dir / 'generated'),
                user_prompt=user_prompt,
                teaching_pattern=teaching_pattern,
                skill_level=skill_level,
                topic_metadata=topic_metadata,
                user_id=user_id,
                allow_images=False 
                )
                
                # Extract article generation results
                generated_articles = generated_result.get('articles', [])
                generation_errors = generated_result.get('errors', [])
                
                if generation_errors:
                    logger.warning(f"Some articles had generation errors: {generation_errors}")
                
                if not generated_articles:
                    logger.warning("No articles were generated successfully")
                
                # Process images for generated articles
                image_results = {}
                if generated_articles:
                    logger.info(f"Processing images for {len(generated_articles)} articles")
                    image_results = process_article_images(generated_articles, course_id)
                    logger.info(f"Image processing results: {image_results}")
                
                return jsonify({
                    'status': 'success',
                    'file_path': str(file_path),
                    'document_dir': str(document_dir),
                    'document_name': document_name,
                    'document_type': document_type,
                    'course_id': course_id,
                    'user_id': user_id,
                    'generation_data': {
                        'topic_metadata': topic_metadata,
                        'user_prompt': user_prompt,
                        'teaching_pattern': teaching_pattern,
                        'skill_level': skill_level
                    },
                    'process_result': process_result,
                    'generated_result': {
                        'articles': generated_articles,
                        'errors': generation_errors,
                        'files': generated_result.get('files', {})
                    },
                    'image_results': image_results
                }), 200

            except Exception as e:
                logger.error(f"Pipeline processing failed: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'Pipeline processing failed: {str(e)}',
                    'error_code': 'PIPELINE_FAILED',
                    'details': {
                        'document_dir': str(document_dir),
                        'course_id': course_id,
                        'user_id': user_id
                    }
                }), 500

        return jsonify({
            'status': 'error',
            'message': 'Document processing failed',
            'error_code': 'PROCESSING_FAILED',
            'result': result
        }), 500

    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_code': 'PROCESSING_FAILED'
        }), 500
@custom_logger.log_function_call
@rag_generation_bp.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException) -> tuple[Dict[str, Any], int]:
    """
    Handle HTTP exceptions globally.
    
    Args:
        e (HTTPException): The HTTP exception that occurred
        
    Returns:
        tuple[Dict[str, Any], int]: Error response and status code
    """
    logger.error(f"HTTP error: {str(e)}")
    return jsonify({
        "error": e.name,
        "details": e.description
    }), e.code
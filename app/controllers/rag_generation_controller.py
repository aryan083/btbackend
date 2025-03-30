"""
Controller module for handling RAG (Retrieval Augmented Generation) HTTP requests.
This module provides endpoints for generating HTML content using RAG techniques.
"""

import logging
from typing import Dict, Any
from flask import Blueprint, request, jsonify
from werkzeug.exceptions import HTTPException
from app.services.rag_generation_service import RAGGenerationService
from app.config import Config

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

@rag_generation_bp.route('/generate', methods=['POST'])
def generate_content() -> tuple[Dict[str, Any], int]:
    """
    Generate HTML content for course materials using RAG.
    
    Request body:
        document_dir (str): Directory containing course documents
        course_id (str): ID of the course
        output_dir (str, optional): Directory to save generated HTML files
        match_threshold (float, optional): Threshold for vector similarity matching
        max_results (int, optional): Maximum number of results to retrieve
    
    Returns:
        tuple[Dict[str, Any], int]: Response containing generated file paths and status code
    """
    try:
        # Validate request body
        data = request.get_json()
        if not data:
            return jsonify({"error": "No request body provided"}), 400
            
        required_fields = ['document_dir', 'course_id']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # Extract parameters with defaults
        document_dir = data['document_dir']
        course_id = data['course_id']
        output_dir = data.get('output_dir', 'output')
        match_threshold = float(data.get('match_threshold', 0.7))
        max_results = int(data.get('max_results', 5))

        # Generate content
        generated_files = rag_service.generate_html_content(
            document_dir=document_dir,
            course_id=course_id,
            output_dir=output_dir,
            match_threshold=match_threshold,
            max_results=max_results
        )

        if not generated_files:
            return jsonify({
                "error": "No content was generated",
                "details": "Check if the course structure is valid and documents exist"
            }), 404

        return jsonify({
            "message": "Content generated successfully",
            "generated_files": list(generated_files.keys())
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

@rag_generation_bp.route('/process', methods=['POST'])
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
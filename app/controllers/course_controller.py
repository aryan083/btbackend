"""
Course Controller Module
Handles course-related HTTP requests and responses
"""
from flask import Blueprint, request, jsonify, current_app
from app.services.course_service import CourseService
import logging
import fitz  # PyMuPDF
from pathlib import Path
logger = logging.getLogger(__name__)
course_bp = Blueprint('course', __name__)
course_service = CourseService()

@course_bp.route('/courses', methods=['POST'])
def create_course():
    """
    Create a new course
    @param request: Flask request object containing course data
    @returns: JSON response with created course data or error
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'course_name', 'skill_level', 'teaching_pattern', 
            'user_prompt'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create course with optional fields defaulting to empty values
        course = course_service.create_course(
            course_name=data['course_name'],
            tags=data.get('tags', {}),
            metadata=data.get('metadata', ''),
            chapters_json=data.get('chapters_json', {}),
            skill_level=data['skill_level'],
            teaching_pattern=data['teaching_pattern'],
            user_prompt=data['user_prompt'],
            progress=data.get('progress', 0.0)
        )
        
        return jsonify({
            'message': 'Course created successfully',
            'data': course
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating course: {str(e)}")
        return jsonify({
            'error': 'Failed to create course',
            'details': str(e)
        }), 500




@course_bp.route('/send_pdf_to_gemini', methods=['POST'])
def send_pdf_to_gemini():
    """
    Send PDF content directly to Gemini API for analysis
    @returns: JSON response with course content, keywords, and welcome message
    @description: Processes PDF content through Gemini API without additional processing
    """
    try:
        # Validate file in request
        if 'course_pdf' not in request.files:
            error_msg = "No course PDF file provided in request"
            logger.warning(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'NO_FILE'
            }), 400

        course_pdf = request.files['course_pdf']
        if not course_pdf or course_pdf.filename == '':
            error_msg = "No course PDF file selected"
            logger.warning(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'NO_FILE_SELECTED'
            }), 400
            
        # Get output directory from request data
        data = request.form.to_dict()
        upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
        
        output_dir = data.get('book_pdf_name', None)
        
        if not output_dir:
            error_msg = "No output directory specified"
            logger.warning(error_msg)
            return jsonify({
               'status': 'error',
               'message': error_msg,
                'error_code': 'NO_OUTPUT_DIR'
            }), 400
        #remove .pdf
        output_dir = output_dir.replace('.pdf', '')
        output_dir = Path(upload_dir) / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the service with the model
        pdf_service = CourseService()
        
        # Process the PDF file with output directory
        result = pdf_service.process_pdf(course_pdf.read(), output_dir)
        
        # Return the processed results
        return jsonify(result), 200

    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_code': 'PARSING_FAILED'
        }), 500

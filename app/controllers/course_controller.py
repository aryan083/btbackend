"""
Course Controller Module
Handles course-related HTTP requests and responses
"""
from flask import Blueprint, request, jsonify
from app.services.course_service import CourseService
import logging

logger = logging.getLogger(__name__)
course_bp = Blueprint('course', __name__)
course_service = CourseService()

@course_bp.route('/api/courses', methods=['POST'])
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

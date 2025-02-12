"""
Course handling routes for the API.
"""
from flask import Blueprint

bp = Blueprint('course', __name__, url_prefix='/api/course')

# Placeholder for course-specific routes
@bp.route('/health', methods=['GET'])
def health_check():
    return {'status': 'ok'}, 200 
"""
Book handling routes for the API.
"""
from flask import Blueprint

bp = Blueprint('book', __name__, url_prefix='/api/book')

# Placeholder for book-specific routes
@bp.route('/health', methods=['GET'])
def health_check():
    return {'status': 'ok'}, 200 
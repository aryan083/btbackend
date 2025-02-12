"""
Authentication routes for the API.
"""
from flask import Blueprint

bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# Placeholder for authentication routes
@bp.route('/health', methods=['GET'])
def health_check():
    return {'status': 'ok'}, 200 
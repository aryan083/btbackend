from flask import Blueprint, request, jsonify
from app.services.directory_service import create_directory
from run import custom_logger
directory_bp = Blueprint('directory', __name__)

@custom_logger.log_function_call
@directory_bp.route('/create', methods=['POST'])
def create_dir():
    """
    Create a new directory
    @body: {"dir_path": "path/to/directory"}
    @returns: JSON response indicating success or failure
    """
    try:
        data = request.get_json()
        
        if not data or 'dir_path' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameter: dir_path'
            }), 400
            
        dir_path = data['dir_path']
        
        # Attempt to create the directory
        create_directory(dir_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Directory created successfully at {dir_path}'
        }), 201
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
        
    except PermissionError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 403
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500
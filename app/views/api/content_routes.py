"""
API routes for content processing and generation.
"""
import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from ...controllers.content_controller import ContentController
from ...config import Config

bp = Blueprint('content', __name__, url_prefix='/api/content')

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/process-book', methods=['POST'])
def process_book():
    """
    Process uploaded book PDF.
    
    Returns:
        JSON response with extracted content and output file path
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.BOOK_PDF_FOLDER, filename)
        
        # Ensure directory exists
        os.makedirs(Config.BOOK_PDF_FOLDER, exist_ok=True)
        
        # Save uploaded file
        file.save(filepath)
        
        result = ContentController.process_book(filepath)
        
        if result['status'] == 'success':
            return jsonify({
                'message': 'PDF processed successfully',
                'data': result['data'],
                'output_path': result['data'].get('output_path')
            }), 200
        return jsonify({'error': result['message']}), 500
        
    return jsonify({'error': 'Invalid file type'}), 400

@bp.route('/process-course', methods=['POST'])
def process_course():
    """
    Process uploaded course content PDF.
    
    Returns:
        JSON response with extracted course structure
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        result = ContentController.process_course_content(filepath)
        
        if result['status'] == 'success':
            return jsonify(result['data']), 200
        return jsonify({'error': result['message']}), 500
        
    return jsonify({'error': 'Invalid file type'}), 400 
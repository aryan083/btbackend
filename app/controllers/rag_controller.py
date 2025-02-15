"""
Controller for handling document processing and RAG operations
"""
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import logging
from typing import Dict, Any
from ..services.rag_service import DocumentService

logger = logging.getLogger(__name__)

# Create a blueprint with a more descriptive name
rag_bp = Blueprint('document_processing', __name__)

def allowed_file(filename: str) -> bool:
    """
    Check if file has an allowed extension
    @param filename: str - Name of the file to check
    @returns: bool - True if file extension is allowed
    @description: Validates file extensions for processing
    """
    allowed_extensions = {'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@rag_bp.route('/process_book', methods=['POST'])
def process_book() -> Dict[str, Any]:
    """
    Process a book PDF file
    @returns: Dict containing processing results or error message
    @description: Handles file upload, saves it, and processes it for text and images
    """
    try:
        # Check if file is present in the request
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({
                'status': 'error',
                'message': 'No file part in the request',
                'error_code': 'MISSING_FILE'
            }), 400

        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({
                'status': 'error',
                'message': 'No file selected',
                'error_code': 'EMPTY_FILENAME'
            }), 400

        # Validate file type
        if not file or not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Only PDF files are allowed.',
                'error_code': 'INVALID_FILE_TYPE'
            }), 400

        # Get upload directory from app config
        upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique directory for this book
        filename = secure_filename(file.filename)
        book_name = filename.rsplit('.', 1)[0]
        book_dir = upload_dir / book_name
        book_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the PDF file
        file_path = book_dir / filename
        file.save(str(file_path))
        
        logger.info(f"Book file saved: {file_path}")

        # Process the book
        service = DocumentService(str(book_dir))
        try:
            result = service.process_book(str(file_path))
            
            if result['status'] == 'success':
                # Update paths to be relative to static directory
                if 'image_info_path' in result:
                    result['image_info_path'] = f"/static/uploads/{book_name}/{result['image_info_path']}"
                if 'metadata' in result and 'full_text_path' in result['metadata']:
                    result['metadata']['full_text_path'] = f"/static/uploads/{book_name}/{result['metadata']['full_text_path']}"
                
                logger.info("Book processing completed successfully")
                return jsonify(result), 200
            else:
                logger.error(f"Book processing failed: {result.get('message', 'Unknown error')}")
                return jsonify(result), 500

        except Exception as e:
            error_msg = f"Error processing book: {str(e)}"
            logger.error(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'PROCESSING_ERROR'
            }), 500

    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_code': 'SERVER_ERROR'
        }), 500

@rag_bp.route('/process_syllabus', methods=['POST'])
def process_syllabus() -> Dict[str, Any]:
    """
    Process a syllabus PDF file
    @returns: Dict containing processing results or error message
    @description: Handles syllabus upload and processing
    """
    try:
        # Check if file is present in the request
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({
                'status': 'error',
                'message': 'No file part in the request',
                'error_code': 'MISSING_FILE'
            }), 400

        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({
                'status': 'error',
                'message': 'No file selected',
                'error_code': 'EMPTY_FILENAME'
            }), 400

        # Validate file type
        if not file or not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Only PDF files are allowed.',
                'error_code': 'INVALID_FILE_TYPE'
            }), 400

        # Get upload directory from app config
        upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique directory for this syllabus
        filename = secure_filename(file.filename)
        syllabus_name = filename.rsplit('.', 1)[0]
        syllabus_dir = upload_dir / syllabus_name
        syllabus_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the PDF file
        file_path = syllabus_dir / filename
        file.save(str(file_path))
        
        logger.info(f"Syllabus file saved: {file_path}")

        # Process the syllabus
        service = DocumentService(str(syllabus_dir))
        try:
            result = service.process_syllabus(str(file_path))
            
            if result['status'] == 'success':
                # Update paths to be relative to static directory
                if 'image_info_path' in result:
                    result['image_info_path'] = f"/static/uploads/{syllabus_name}/{result['image_info_path']}"
                if 'metadata' in result and 'full_text_path' in result['metadata']:
                    result['metadata']['full_text_path'] = f"/static/uploads/{syllabus_name}/{result['metadata']['full_text_path']}"
                
                logger.info("Syllabus processing completed successfully")
                return jsonify(result), 200
            else:
                logger.error(f"Syllabus processing failed: {result.get('message', 'Unknown error')}")
                return jsonify(result), 500

        except Exception as e:
            error_msg = f"Error processing syllabus: {str(e)}"
            logger.error(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'PROCESSING_ERROR'
            }), 500

    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_code': 'SERVER_ERROR'
        }), 500

@rag_bp.route('/uploads/<path:filename>')
def serve_file(filename: str):
    """
    Serve uploaded files
    @param filename: str - Path to the file relative to upload directory
    @returns: File response or error
    @description: Provides access to uploaded files via a secure route
    """
    try:
        return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'File not found',
            'error_code': 'FILE_NOT_FOUND'
        }), 404

@rag_bp.route('/upload', methods=['POST'])
def upload_pdf():
    """
    Upload a PDF file
    @returns: Dict containing upload results or error message
    @description: Handles PDF file upload and saves it to the upload directory
    """
    try:
        # Check if file is present in the request
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({
                'status': 'error',
                'message': 'No file part in the request',
                'error_code': 'MISSING_FILE'
            }), 400

        file = request.files['file']
        book_name = request.form.get('book_name', '')
        
        # Check if filename is empty
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({
                'status': 'error',
                'message': 'No file selected',
                'error_code': 'EMPTY_FILENAME'
            }), 400

        # Validate file type
        if not file or not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Only PDF files are allowed.',
                'error_code': 'INVALID_FILE_TYPE'
            }), 400

        # Get upload directory from app config
        upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique directory for this book
        filename = secure_filename(file.filename)
        book_name = book_name or filename.rsplit('.', 1)[0]
        book_dir = upload_dir / book_name
        book_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the PDF file
        file_path = book_dir / filename
        file.save(str(file_path))
        
        logger.info(f"PDF file saved: {file_path}")

        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'file_path': str(file_path),
            'book_name': book_name
        }), 200

    except Exception as e:
        error_msg = f"Error uploading PDF: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_code': 'UPLOAD_FAILED'
        }), 500

@rag_bp.route('/process', methods=['POST'])
def process_pdf():
    """
    Process a PDF file
    @returns: Dict containing processing results or error message
    @description: Handles PDF processing with configurable options
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'file_path' not in data:
            logger.warning("Missing file path in request")
            return jsonify({
                'status': 'error',
                'message': 'Missing file path',
                'error_code': 'MISSING_FILE_PATH'
            }), 400

        file_path = data['file_path']
        
        # Optional processing flags
        extract_images = data.get('extract_images', True)
        extract_text = data.get('extract_text', True)
        save_json = data.get('save_json', True)

        # Process the PDF
        service = DocumentService(os.path.dirname(file_path))
        result = service.process_book(file_path, 
                                      extract_images=extract_images, 
                                      extract_text=extract_text, 
                                      save_json=save_json)
        
        if result['status'] == 'success':
            logger.info("PDF processing completed successfully")
            return jsonify(result), 200
        else:
            logger.error(f"PDF processing failed: {result.get('message', 'Unknown error')}")
            return jsonify(result), 500

    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_code': 'PROCESSING_FAILED'
        }), 500

try:
    # Register the blueprint
    current_app.register_blueprint(rag_bp)
    logger.info("RAG blueprint registered successfully")
except Exception as e:
    logger.error(f"Error registering RAG blueprint: {str(e)}")

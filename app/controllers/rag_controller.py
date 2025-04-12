"""
Controller for handling document processing and RAG operations
"""
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import logging
from typing import Dict, Any, List
import json
import google.generativeai as genai
from ..config import GEMINI_API_KEY
from ..services.pdf_processor import PDFParser as DocumentService
import fitz  # PyMuPDF
from run import custom_logger
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Create a blueprint with a more descriptive name
rag_bp = Blueprint('document_processing', __name__)

@custom_logger.log_function_call
def allowed_file(filename: str) -> bool:
    """
    Check if file has an allowed extension
    @param filename: str - Name of the file to check
    @returns: bool - True if file extension is allowed
    @description: Validates file extensions for processing
    """
    allowed_extensions = {'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@custom_logger.log_function_call
@rag_bp.route('/upload_and_process', methods=['POST'])
def upload_and_process_pdf():
    """
    Upload and process a PDF file in one step
    @returns: Dict containing processing results or error message
    @description: Handles file upload and processing in a single API call.
                 Supports both book and syllabus PDFs.
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
        document_type = request.form.get('document_type', 'book')  # 'book' or 'syllabus'
        document_name = request.form.get('document_name', '')

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
        user_id = request.form.get('user_id')
        course_id = request.form.get('course_id')
        logger.info(f"User ID: {user_id}, Course ID: {course_id}")
        
        # Create a unique directory for this document
        filename = secure_filename(file.filename)
        document_name = document_name or filename.rsplit('.', 1)[0]
        document_dir = upload_dir / document_name
        document_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the PDF file
        file_path = document_dir / filename
        file.save(str(file_path))
        
        logger.info(f"PDF file saved: {file_path}")

        # Process the PDF
        service = DocumentService(str(file_path), str(document_dir))
        
        # Get processing options from request
        extract_images = request.form.get('extract_images', 'true').lower() == 'true'
        extract_text = request.form.get('extract_text', 'true').lower() == 'true'
        save_json = request.form.get('save_json', 'true').lower() == 'true'

        # Process based on document type
        if document_type == 'syllabus':
            service = DocumentService(str(file_path), str(document_dir))
        else:
            result = service.process_document(
            extract_images=extract_images,
            extract_text=extract_text,
            save_json=save_json)
        
        if result['status'] == 'success':
            logger.info(f"{document_type.capitalize()} processing completed successfully")
            return jsonify({
                **result,
                'file_path': str(file_path),
                'document_name': document_name,
                'document_type': document_type
            }), 200
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


@rag_bp.route('/process_book', methods=['POST'])
@custom_logger.log_function_call
def process_book():
    """
    Process a book PDF file
    @returns: Dict containing processing results or error message
    @description: Handles file upload, saves it, and processes it for text and images
    """
    # Add document_type to the request
    request.form = dict(request.form)
    request.form['document_type'] = 'book'
    return upload_and_process_pdf()

@custom_logger.log_function_call
@rag_bp.route('/process_syllabus', methods=['POST'])
def process_syllabus():
    """
    Process a syllabus PDF file
    @returns: Dict containing processing results or error message
    @description: Handles syllabus upload and processing
    """
    # Add document_type to the request
    request.form = dict(request.form)
    request.form['document_type'] = 'syllabus'
    return upload_and_process_pdf()

@custom_logger.log_function_call
@rag_bp.route('/uploads/<path:filename>')
def serve_file(filename: str):
    """
    Serve uploaded files
    @param filename: str - Path to the file relative to upload directory
    @returns: File or error response
    @description: Serves files from the upload directory
    """
    try:
        return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({
            'status': 'error',
            'message': 'File not found',
            'error_code': 'FILE_NOT_FOUND'
        }), 404

@custom_logger.log_function_call
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

# @rag_bp.route('/process', methods=['POST'])
# def process_pdf():
#     """
#     Process a PDF file
#     @returns: Dict containing processing results or error message
#     @description: Handles PDF processing with configurable options
#     """
#     try:
#         # Get request data
#         data = request.get_json()
        
#         if not data or 'file_path' not in data:
#             logger.warning("Missing file path in request")
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Missing file path',
#                 'error_code': 'MISSING_FILE_PATH'
#             }), 400

#         file_path = data['file_path']
        
#         # Optional processing flags
#         extract_images = data.get('extract_images', True)
#         extract_text = data.get('extract_text', True)
#         save_json = data.get('save_json', True)

#         # Process the PDF
#         service = DocumentService(os.path.dirname(file_path))
#         result = service.process_document(file_path, 
#                                       extract_images=extract_images, 
#                                       extract_text=extract_text, 
#                                       save_json=save_json)
        
#         if result['status'] == 'success':
#             logger.info("PDF processing completed successfully")
#             return jsonify(result), 200
#         else:
#             logger.error(f"PDF processing failed: {result.get('message', 'Unknown error')}")
#             return jsonify(result), 500

#     except Exception as e:
#         error_msg = f"Error processing PDF: {str(e)}"
#         logger.error(error_msg)
#         return jsonify({
#             'status': 'error',
#             'message': error_msg,
#             'error_code': 'PROCESSING_FAILED'
#         }), 500

@custom_logger.log_function_call
@rag_bp.route('/process_course_json', methods=['POST'])
def process_course_json():
    """
    Process a Course.json file with Gemini
    @returns: Dict containing processing results or error message
    @description: Handles Course.json processing with Gemini
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
        
        # Load the Course.json file
        with open(file_path, 'r') as f:
            course_data = json.load(f)
        
        # Process the Course.json file with Gemini
        result = model.generate(course_data)
        
        if result['status'] == 'success':
            logger.info("Course.json processing completed successfully")
            return jsonify(result), 200
        else:
            logger.error(f"Course.json processing failed: {result.get('message', 'Unknown error')}")
            return jsonify(result), 500

    except Exception as e:
        error_msg = f"Error processing Course.json: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_code': 'PROCESSING_FAILED'
        }), 500



@custom_logger.log_function_call
@rag_bp.route('/parse_document', methods=['POST'])
def parse_document():
    """
    Parse the document and extract relevant information
    @returns: Dict containing parsed data or error message
    @description: Handles the parsing of documents and returns structured information
    """
    try:
        # Implementation for parsing the document goes here
        pass  # Placeholder for actual parsing logic
    except Exception as e:
        error_msg = f"Error parsing document: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_code': 'PARSING_FAILED'
        }), 500
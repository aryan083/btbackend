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
import requests
from app.services import rag_service
from ..config import GEMINI_API_KEY
from ..services.pdf_processor import PDFParser as DocumentService
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

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
            
        # Get callback URL from request
        callback_url = request.form.get('callback_url') or 'localhost:5173/api/callback/'
        if not callback_url:
            logger.warning("No callback URL provided")
            return jsonify({
                'status': 'error',
                'message': 'Callback URL is required',
                'error_code': 'MISSING_CALLBACK'
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
        
        # Create a unique directory for this document
        filename = secure_filename(file.filename)
        document_name = document_name or filename.rsplit('.', 1)[0]
        document_dir = upload_dir / document_name
        
        # Create the main directory and its subdirectories
        document_dir.mkdir(parents=True, exist_ok=True)
        (document_dir / "Text").mkdir(exist_ok=True)
        (document_dir / "Images").mkdir(exist_ok=True)
        
        # Save the PDF file in the main directory
        file_path = document_dir / filename
        file.save(str(file_path))
        
        logger.info(f"PDF file saved: {file_path}")

        # Process the PDF - Pass the document_dir as the output directory
        service = DocumentService(str(file_path), str(document_dir))
        print(document_dir)
        
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
                save_json=save_json
            )
        
        if result['status'] == 'success':
            logger.info(f"{document_type.capitalize()} processing completed successfully")
            
            # Prepare callback data
            callback_data = {
                'status': 'success',
                'message': 'Document processed successfully',
                'data': {
                    'file_path': str(file_path),
                    'document_name': document_name,
                    'document_type': document_type,
                    'base_path': str(document_dir),  # This will be needed for next step
                    'processing_results': result
                }
            }

            # Make callback request
            try:
                response = requests.post(
                    callback_url,
                    json=callback_data,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                logger.info(f"Callback successful to {callback_url}")
            except Exception as e:
                logger.error(f"Callback failed: {str(e)}")
                # Still return success to client as processing was successful
                return jsonify({
                    'status': 'success',
                    'message': 'Document processed but callback failed',
                    'data': callback_data
                }), 200

            return jsonify({
                'status': 'success',
                'message': 'Document processed and callback sent',
                'callback_url': callback_url
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


@rag_bp.route('/process_callback', methods=['POST'])
def process_callback():
    """
    Handle callback from document processing and continue with course processing
    @returns: Dict containing processing results or error message
    @description: Receives callback data from document processing and continues
                 with course content processing and generation.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided in callback'
            }), 400

        # Extract required data
        processing_data = data.get('data', {})
        base_path = processing_data.get('base_path')
        document_type = processing_data.get('document_type')

        if not base_path:
            return jsonify({
                'status': 'error',
                'message': 'Missing base_path in callback data'
            }), 400

        # Initialize processing results
        processing_results = {
            'status': 'processing',
            'steps': {
                'text_processing': {'status': 'pending', 'count': 0},
                'image_processing': {'status': 'pending', 'count': 0},
                'content_generation': {'status': 'pending'}
            }
        }

        # Process content (text and images)
        try:
            results = rag_service.process_content(
                base_path=base_path,
                course_id=processing_data.get('course_id')
            )
            
            # Update processing results
            processing_results['steps']['text_processing'] = {
                'status': 'completed',
                'count': results.get('text_processed', 0)
            }
            processing_results['steps']['image_processing'] = {
                'status': 'completed',
                'count': results.get('image_processed', 0)
            }

            if results.get('errors'):
                processing_results['errors'] = results['errors']

        except Exception as e:
            logger.error(f"Content processing failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Content processing failed: {str(e)}',
                'processing_results': processing_results
            }), 500

        # Generate HTML content
        try:
            generated_files = rag_service.generate_html_content(
                document_dir=base_path,
                course_id=processing_data.get('course_id')
            )
            
            processing_results['steps']['content_generation'] = {
                'status': 'completed',
                'files_generated': len(generated_files)
            }

        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Content generation failed: {str(e)}',
                'processing_results': processing_results
            }), 500

        # Return final results
        return jsonify({
            'status': 'success',
            'message': 'Processing completed successfully',
            'processing_results': processing_results,
            'results': {
                'text_processed': results.get('text_processed', 0),
                'image_processed': results.get('image_processed', 0),
                'generated_files': list(generated_files.keys())
            }
        }), 200

    except Exception as e:
        logger.error(f"Callback processing failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Callback processing failed: {str(e)}'
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
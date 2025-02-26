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
from ..services.rag_service import DocumentService
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

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
        document_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the PDF file
        file_path = document_dir / filename
        file.save(str(file_path))
        
        logger.info(f"PDF file saved: {file_path}")

        # Process the PDF
        service = DocumentService(str(document_dir))
        
        # Get processing options from request
        extract_images = request.form.get('extract_images', 'true').lower() == 'true'
        extract_text = request.form.get('extract_text', 'true').lower() == 'true'
        save_json = request.form.get('save_json', 'true').lower() == 'true'

        # Process based on document type
        if document_type == 'syllabus':
            result = service.process_syllabus(str(file_path))
        else:
            result = service.process_book(
                str(file_path),
                extract_images=extract_images,
                extract_text=extract_text,
                save_json=save_json
            )
        
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

@rag_bp.route('/extract_topics', methods=['POST'])
def extract_topics():
    """
    Send Course.json content to Gemini AI for processing
    @returns: Dict containing the response from Gemini or error message
    @description: Processes a Course.json file to send to Gemini AI without extracting topics
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
        if not file.filename.endswith('.json'):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Only JSON files are allowed.',
                'error_code': 'INVALID_FILE_TYPE'
            }), 400

        # Read the JSON content
        with open("your_file.pdf", "rb") as file:
            pdf_document = fitz.open(file)  # Open PDF file
            pdf_content = "\n".join([page.get_text() for page in pdf_document])  # Extract text

        
        
        # Create prompt for Gemini
        prompt = """RETURN ALL THE TOPICS COVERED IN COURSE.JSON
        ALSO RETURN THEM IN A CHAPTER-WISE FORMAT
        WITH SUBTOPICS IN A NUMBERED SEQUENCE.THE SEQUENCE IS LIKE CH 1 
        THEN ALL THE SUBTOPICS IN CH 1 THAN CH 2 AND SO ON GIVE ME LIEK THAT
        PLEASE GIVE ME A NESTED JSON FOR CH AND SUBTOPICS
        THE PDF CAN ALSO TABLE,SIMPLE TEXT AND ALL THOSE THINGS HELP ME GET THE TOPICS!
        THE FORMAT SHOULD BE CH 1 WITH NAME :{ITS SUBTOPIC},CH2 WITH NAME:{ITS SUBTOPICS}
        """ + pdf_content 
        # Get response from Gemini
        response = model.generate_content(prompt)
        
        logger.info("Prompt sent to Gemini successfully")
        return jsonify({
            'status': 'success',
            'data': response.text
        }), 200

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_code': 'PROCESSING_FAILED'
        }), 500

@rag_bp.route('/send_pdf_to_gemini', methods=['POST'])
def send_pdf_to_gemini():
    """
    Send PDF content to Gemini AI for processing
    @returns: Dict containing the response from Gemini or error message
    @description: Processes a PDF file to send to Gemini AI for analysis
    """
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded', 'error_code': 'MISSING_FILE'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected', 'error_code': 'EMPTY_FILENAME'}), 400

        if not file.filename.endswith('.pdf'):
            return jsonify({'status': 'error', 'message': 'Invalid file type. Only PDFs are allowed.', 'error_code': 'INVALID_FILE_TYPE'}), 400

        # Read and extract text from PDF using PyMuPDF
        pdf_data = file.read()
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        pdf_text = ""
        for page in pdf_document:
            pdf_text += page.get_text()
        pdf_document.close()

        # Create prompt for Gemini
        prompt = """
RETURN ALL THE TOPICS COVERED IN COURSE.JSON
ALSO RETURN THEM IN A CHAPTER-WISE FORMAT
WITH SUBTOPICS IN A NUMBERED SEQUENCE.
"""
        content_parts = [prompt, pdf_text]

        # Get response from Gemini
        response = model.generate_content(content_parts)
        
        logger.info("PDF sent to Gemini successfully")
        return jsonify({
            'status': 'success',
            'data': response.text
        }), 200

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'status': 'error', 'message': str(e), 'error_code': 'PROCESSING_FAILED'}), 500


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

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

def clean_gemini_json_response(response_text: str) -> str:
    """
    Clean Gemini API JSON response by removing markdown code block markers
    @param response_text: Raw response text from Gemini
    @returns: Cleaned JSON string
    """
    # Remove ```json and ``` markers
    cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
    return cleaned_text

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
        if not file.filename.endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Only PDF files are allowed.',
                'error_code': 'INVALID_FILE_TYPE'
            }), 400

        # Read the PDF content
        pdf_document = fitz.open(file)  # Open PDF file
        pdf_content = "\n".join([page.get_text() for page in pdf_document])  # Extract text

        # Extract course content (topics and subtopics)
        content_prompt = """Extract only the course content (topics and subtopics) from the given syllabus PDF. Ignore any extra details such as unit hours,Contact Hours, examination schemes, objectives, and references.  
                Format the extracted content in a structured JSON format as follows:  
                - Each chapter should be labeled as **"Chapter X: Chapter Name"**.  
                - Each chapter should contain a list of subtopics in a numbered sequence, such as **"X.Y: Subtopic Name"**.  
                - Maintain the hierarchical structure of topics and subtopics.  
                - The output should be clean and formatted as nested JSON.

                Example format:  
                ```json
                {
                "Chapters": {
                    "Chapter 1: Introduction": {
                    "1.1": "Subtopic Name",
                    "1.2": "Subtopic Name"
                    },
                    "Chapter 2: Next Chapter Name": {
                    "2.1": "Subtopic Name"
                    }
                }
                } NOTE: ONLY RETURN THE JSON NO OTHER TEXT IS NEEDED AND NOT WANTED 
                I REPEAT THAT WE NOT NO SUCH THINGS AS HERE'S THE BACKDOWN   """

        content_parts = [content_prompt, pdf_content]
        content_response = model.generate_content(content_parts)
        course_content = json.loads(content_response.text)

        # Extract keywords and skills
        keyword_prompt = """Analyze this course syllabus and extract:
        1. Key technical terms and concepts that will be covered
        2. Main skills students will learn
        3. Core technologies or tools that will be taught

        Format the response as a JSON with the following structure:
        {
            "technical_terms": ["term1", "term2", ...],
            "skills": ["skill1", "skill2", ...],
            "technologies": ["tech1", "tech2", ...]
        }
        NOTE: ONLY RETURN THE JSON NO OTHER TEXT IS NEEDED"""

        keyword_parts = [keyword_prompt, pdf_content]
        keyword_response = model.generate_content(keyword_parts)
        keywords = json.loads(keyword_response.text)

        # Generate welcome message
        welcome_prompt = f"""Create an engaging and professional welcome message for a course with the following details:
        Course Content: {json.dumps(course_content)}
        Key Learning Outcomes: {json.dumps(keywords)}

        The message should:
        1. Be warm and encouraging
        2. Highlight the value and relevance of the course
        3. Mention 2-3 key skills or technologies they'll learn
        4. Keep it concise (max 4-5 sentences)
        
        NOTE: Return only the welcome message, no additional text."""

        welcome_response = model.generate_content(welcome_prompt)
        welcome_message = welcome_response.text

        # Close the PDF document
        pdf_document.close()

        return jsonify({
            'status': 'success',
            'course_content': course_content,
            'keywords': keywords,
            'welcome_message': welcome_message
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
    Send PDF content directly to Gemini API for analysis
    @returns: JSON response with course content, keywords, and welcome message
    @description: Processes PDF content through Gemini API without additional processing
    """
    try:
        if 'file' not in request.files:
            error_msg = "No file provided in request"
            logger.warning(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'NO_FILE'
            }), 400

        file = request.files['file']
        if not file or file.filename == '':
            error_msg = "No file selected"
            logger.warning(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'NO_FILE_SELECTED'
            }), 400

        # Read PDF content directly from request
        
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        pdf_text = "\n".join([page.get_text() for page in pdf_document])
        pdf_document.close()
        if pdf_document == None:
            error_msg = "Unable to open PDF document"
            logger.error(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'PDF_DOCUMENT_OPEN_FAILED',
         
            }), 400

        # Validate PDF text extraction
        if not pdf_text or len(pdf_text.strip()) < 10:
            error_msg = "Unable to extract text from PDF. The document might be empty or unreadable."
            logger.error(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'PDF_TEXT_EXTRACTION_FAILED'
            }), 400

        # Extract course content (topics and subtopics)
        content_prompt = """Extract only the course content (topics and subtopics) from the given syllabus PDF. Ignore any extra details such as unit hours,Contact Hours, examination schemes, objectives, and references.  
                Format the extracted content in a structured JSON format as follows:  
                - Each chapter should be labeled as **"Chapter X: Chapter Name"**.  
                - Each chapter should contain a list of subtopics in a numbered sequence, such as **"X.Y: Subtopic Name"**.  
                - Maintain the hierarchical structure of topics and subtopics.  
                - The output should be clean and formatted as nested JSON.

                Example format:  
                ```json
                {
                "Chapters": {
                    "Chapter 1: Introduction": {
                    "1.1": "Subtopic Name",
                    "1.2": "Subtopic Name"
                    },
                    "Chapter 2: Next Chapter Name": {
                    "2.1": "Subtopic Name"
                    }
                }
                } NOTE: ONLY RETURN THE JSON NO OTHER TEXT IS NEEDED AND NOT WANTED 
                I REPEAT THAT WE NOT NO SUCH THINGS AS HERE'S THE BACKDOWN   """

        content_parts = [content_prompt, pdf_text]
        content_response = model.generate_content(content_parts)
        
        # Validate course content response
        try:
            # Clean the JSON response
            cleaned_content_text = clean_gemini_json_response(content_response.text)
            course_content = json.loads(cleaned_content_text)
            logger.info("Course content extracted successfully")
            logger.info(json.dumps(course_content, indent=4))
        except json.JSONDecodeError:
            logger.error(f"Failed to parse course content JSON. Raw response: {content_response.text}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to parse course content from Gemini response',
                'raw_response': content_response.text,
                'error_code': 'COURSE_CONTENT_PARSING_FAILED',
                
            }), 500

        # Extract keywords and skills
        keyword_prompt = """Analyze this course syllabus and extract:
        1. Key technical terms and concepts that will be covered
        2. Main skills students will learn
        3. Core technologies or tools that will be taught
        4. The technical terms should be concise
        5. Technical terms array should not exceed size of 10
        6. Technical terms should be covering the course content completely

        Format the response as a JSON with the following structure:
        {
            "technical_terms": ["term1", "term2", ...],
            "skills": ["skill1", "skill2", ...],
            "technologies": ["tech1", "tech2", ...]
        }
        NOTE: ONLY RETURN THE JSON NO OTHER TEXT IS NEEDED"""

        keyword_parts = [keyword_prompt, pdf_text]
        keyword_response = model.generate_content(keyword_parts)
        
        # Validate keywords response
        try:
            # Clean the JSON response
            cleaned_keyword_text = clean_gemini_json_response(keyword_response.text)
            keywords = json.loads(cleaned_keyword_text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse keywords JSON. Raw response: {keyword_response.text}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to parse keywords from Gemini response',
                'raw_response': keyword_response.text,
                'error_code': 'KEYWORDS_PARSING_FAILED'
            }), 500

        # Generate welcome message
        welcome_prompt = f"""Create an engaging and professional welcome message for a course with the following details:
        Course Content: {json.dumps(course_content)}
        Key Learning Outcomes: {json.dumps(keywords)}

        The message should:
        1. Be warm and encouraging
        2. Highlight the value and relevance of the course
        3. Mention 2-3 key skills or technologies they'll learn
        4. Keep it concise (max 4-5 sentences)
        
        NOTE: Return only the welcome message, no additional text."""

        welcome_response = model.generate_content(welcome_prompt)
        welcome_message = welcome_response.text.strip()

        # Validate welcome message
        if not welcome_message:
            welcome_message = "Welcome to the course! We're excited to help you learn and grow."

        return jsonify({
            'status': 'success',
            'course_content': course_content,
            'keywords': keywords,
            'welcome_message': welcome_message
        }), 200

    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_code': 'PARSING_FAILED'
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

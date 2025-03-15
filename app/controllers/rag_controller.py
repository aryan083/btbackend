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
import uuid
import shutil
from ..services.pdf_processor import PDFParser

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

@rag_bp.route('/send_pdf_to_gemini', methods=['POST'])
def send_pdf_to_gemini():
    """
    Process the PDF, extract course structure, keywords, skills, and generate a title & welcome message.
    @returns: JSON response with all extracted information.
    """
    try:
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({'status': 'error', 'message': 'No file part in the request', 'error_code': 'MISSING_FILE'}), 400

        file = request.files['file']
        document_name = request.form.get('document_name', '')

        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({'status': 'error', 'message': 'No file selected', 'error_code': 'EMPTY_FILENAME'}), 400

        if not file.filename.endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'status': 'error', 'message': 'Only PDF files are allowed.', 'error_code': 'INVALID_FILE_TYPE'}), 400

        # Read PDF content
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        pdf_text = "\n".join([page.get_text() for page in pdf_document])
        pdf_document.close()

        if not pdf_text or len(pdf_text.strip()) < 10:
            logger.error("Failed to extract text from PDF.")
            return jsonify({'status': 'error', 'message': 'Failed to extract text from PDF.', 'error_code': 'PDF_TEXT_EXTRACTION_FAILED'}), 400

        # Single prompt for all extractions
        full_prompt = f"""
        Analyze the given course syllabus and extract:
        
        1. Course Content (Chapters & Subtopics)  
            - Format: JSON with "Chapter X: Chapter Name" as keys, and "X.Y: Subtopic Name" as values.  
        
        2. Key Technical Terms & Concepts (Max 10)  
            - These should be **concise and relevant** to the syllabus.
        
        3. Skills that students will learn  
            - Short list of skills based on the syllabus.
        
        4. Technologies or Tools taught in the course  
            - If applicable, list them.
        
        5. Welcome Message (Max 2 sentences)  
            - Warm and encouraging.
            - Highlights value & key skills learned.
        
        6. Course Title (Max 30 characters)  
            - Short, engaging, and relevant.

        ### Example Output Format:
        ```json
        {{
            "course_content": {{
                "Chapter 1: Introduction": {{
                    "1.1": "Subtopic Name",
                    "1.2": "Subtopic Name"
                }}
            }},
            "technical_terms": ["term1", "term2"],
            "skills": ["skill1", "skill2"],
            "technologies": ["tech1", "tech2"],
            "welcome_message": "Welcome to the course!",
            "course_title": "AI & ML Basics"
        }}
        ```

        Process the following text and return **ONLY the JSON output**:
        ```text
        {pdf_text}
        ```
        """

        # Send a single request to Gemini AI
        response = model.generate_content(full_prompt)

        # Parse JSON response
        try:
            course_data = json.loads(clean_gemini_json_response(response.text))
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response JSON. Raw response: {response.text}")
            return jsonify({'status': 'error', 'message': 'Failed to parse JSON response', 'error_code': 'JSON_PARSE_ERROR'}), 500

        # Save JSON file
        document_dir = Path(current_app.config['UPLOAD_FOLDER']) / (document_name or secure_filename(file.filename).rsplit('.', 1)[0])
        document_dir.mkdir(parents=True, exist_ok=True)
        json_path = document_dir / "course_structure.json"

        with open(json_path, 'w') as f:
            json.dump(course_data, f, indent=2)

        return jsonify({
            'status': 'success',
            'course_content': course_data.get("course_content", {}),
            'technical_terms': course_data.get("technical_terms", []),
            'skills': course_data.get("skills", []),
            'technologies': course_data.get("technologies", []),
            'welcome_message': course_data.get("welcome_message", ""),
            'course_title': course_data.get("course_title", ""),
            'saved_path': str(json_path)
        })

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e), 'error_code': 'PROCESSING_FAILED'}), 500


# Add these new endpoints
@rag_bp.route('/start_processing', methods=['POST'])
def start_processing():
    """Initiate processing and return process ID"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        document_name = secure_filename(file.filename)
        upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
        
        # Create processing directory
        process_id = str(uuid.uuid4())
        temp_dir = upload_dir / "processing" / process_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PDF
        pdf_path = temp_dir / document_name
        file.save(pdf_path)
        
        # Initialize processor
        processor = PDFParser(str(pdf_path), str(temp_dir))
        processor.process_document(extract_images=False, extract_text=True, save_json=False)
        
        # Initialize processing_tasks if not exists
        if not hasattr(current_app, 'processing_tasks'):
            current_app.processing_tasks = {}
            
        # Store processor in temporary storage
        current_app.processing_tasks[process_id] = {
            "processor": processor,
            "pdf_path": str(pdf_path)
        }
        
        return jsonify({
            "status": "success",
            "process_id": process_id,
            "temp_dir": str(temp_dir)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@rag_bp.route('/finalize_processing', methods=['POST'])
def finalize_processing():
    """Complete processing with course JSON"""
    try:
        data = request.get_json()
        process_id = data.get('process_id')
        course_json = data.get('course_json')
        
        if not process_id or not course_json:
            return jsonify({"error": "Missing parameters"}), 400
            
        # Retrieve processor
        task = current_app.processing_tasks.get(process_id)
        if not task:
            return jsonify({"error": "Invalid process ID"}), 400
            
        processor = task["processor"]
        processor.load_course_json(course_json)
        
        # Complete processing
        result = processor.process_document(
            extract_images=True,
            extract_text=True,
            save_json=True
        )
        
        # Move to final location
        final_dir = Path(current_app.config['UPLOAD_FOLDER']) / processor.book_name
        final_dir.mkdir(exist_ok=True)
        
        # Move files from temp to final
        shutil.move(task["pdf_path"], final_dir)
        shutil.move(processor.text_dir, final_dir)
        shutil.move(processor.images_dir, final_dir)
        
        # Cleanup
        del current_app.processing_tasks[process_id]
        shutil.rmtree(Path(task["pdf_path"]).parent)
        
        return jsonify({
            "status": "success",
            "output_dir": str(final_dir),
            "chunks": result.get("chunks", [])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
"""
Service for processing PDFs and extracting content
This module handles the business logic for processing PDF documents,
including books and syllabi, extracting text and images, and organizing
the content in a structured format.
"""
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from .pdf_processor import PDFParser

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Service for processing documents and extracting content
    Handles PDF processing, content extraction, and metadata organization
    """
    def __init__(self, upload_dir: str):
        """
        Initialize the document service
        @param upload_dir: str - Base directory for storing uploaded and processed files
        @returns: None
        @description: Creates a DocumentService instance with the specified upload directory
                     The upload directory will contain subdirectories for each processed document
        """
        self.upload_dir = Path(upload_dir)

    def _validate_pdf_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate the response from PDF processor
        @param response: Dict[str, Any] - Response from PDF processor
        @returns: bool - True if response is valid, False otherwise
        @description: Checks if the response contains all required fields and has valid status
        """
    
        if not isinstance(response, dict):
            logger.error("Response is not a dictionary")
            return False
            
        if response.get('status') != 'success':
            logger.error(f"Response status is not success: {response.get('status')}")
            return False
            
        # Validate common metadata
        required_metadata = ['title', 'page_count']
        if not all(key in response.get('metadata', {}) for key in required_metadata):
            logger.error("Missing required metadata fields")
            return False
            
        # Validate chunk data when using course JSON
        if response.get('course_json', False):
            if not isinstance(response.get('chunks', []), list):
                logger.error("Invalid chunks format")
                return False
        else:
            # Validate chapter structure for default processing
            if not isinstance(response.get('chapters', []), list):
                logger.error("Invalid chapters format")
                return False
                
        return True

    def process_book(self, book_path: str, extract_images: bool = True, 
                     extract_text: bool = True, save_json: bool = True) -> Dict[str, Any]:
        """
        Process a book PDF and extract its content
        @param book_path: str - Absolute path to the book PDF file
        @param extract_images: bool - Whether to extract images from the PDF
        @param extract_text: bool - Whether to extract text from the PDF
        @param save_json: bool - Whether to save extracted content as JSON
        @returns: Dict[str, Any] - Processing results including metadata and content paths
        @description: Extracts text and images from the book, organizes content by chapters,
                     and returns structured metadata including title, author, and page count
        @raises: ValueError if book_path is invalid
                RuntimeError if processing fails
        """
        try:
            if not os.path.isfile(book_path):
                error_msg = f"Invalid book path: {book_path}"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg
                }

            # Process PDF
            processor = PDFParser(book_path, self.upload_dir)
            try:
                document_structure = processor.process_document(
                    extract_images=extract_images,
                    extract_text=extract_text,
                    save_json=save_json
                )
                
                # Log the response for debugging
                logger.debug(f"PDF processor response: {document_structure}")
                
                # Validate response
                if not self._validate_pdf_response(document_structure):
                    raise RuntimeError("Invalid response from PDF processor")

                # Return processed data
                return document_structure

            except Exception as e:
                error_msg = f"PDF processing error: {str(e)}"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg,
                    'error_details': str(e)
                }
            finally:
                processor.close()

        except Exception as e:
            error_msg = f"Book processing failed: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg,
                'error_details': str(e)
            }

    def process_with_course_json(self, pdf_path: str, course_json: Dict) -> Dict[str, Any]:
        """
        Process PDF with provided course JSON structure
        @param pdf_path: str - Path to the PDF file
        @param course_json: Dict - Course structure JSON
        @returns: Dict[str, Any] - Processing results with JSON-guided chunking
        @description: Processes PDF using provided course structure for chunking
        @raises: ValueError if pdf_path is invalid
                RuntimeError if processing fails
        """
        try:
            if not os.path.isfile(pdf_path):
                error_msg = f"Invalid PDF path: {pdf_path}"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg
                }

            processor = PDFParser(pdf_path, self.upload_dir)
            try:
                processor.course_json = course_json
                return processor.process_document()
            except Exception as e:
                error_msg = f"PDF processing error: {str(e)}"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg,
                    'error_details': str(e)
                }
            finally:
                processor.close()

        except Exception as e:
            error_msg = f"Processing with course JSON failed: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg,
                'error_details': str(e)
            }

    def process_syllabus(self, syllabus_path: str) -> Dict[str, Any]:
        """
        Process a syllabus PDF to extract course structure
        @param syllabus_path: str - Absolute path to the syllabus PDF file
        @returns: Dict[str, Any] - Extracted syllabus structure and metadata
        @description: Processes syllabus PDF to extract course information,
                     including chapters, sections, and any embedded images
        @raises: ValueError if syllabus_path is invalid
                RuntimeError if processing fails
        """
        try:
            if not os.path.isfile(syllabus_path):
                error_msg = f"Invalid syllabus path: {syllabus_path}"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg
                }

            # Process PDF
            processor = PDFParser(syllabus_path, self.upload_dir)
            try:
                document_structure = processor.process_document()
                
                # Log the response for debugging
                logger.debug(f"PDF processor response: {document_structure}")
                
                # Validate response
                if not self._validate_pdf_response(document_structure):
                    raise RuntimeError("Invalid response from PDF processor")

                # Return processed data
                return document_structure

            except Exception as e:
                error_msg = f"PDF processing error: {str(e)}"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg,
                    'error_details': str(e)
                }
            finally:
                processor.close()

        except Exception as e:
            error_msg = f"Syllabus processing failed: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg,
                'error_details': str(e)
            }

        """
        Process PDF with provided course JSON structure
        @param pdf_path: str - Path to the PDF file
        @param course_json: Dict - Course structure JSON
        @returns: Dict[str, Any] - Processing results with JSON-guided chunking
        @description: Processes PDF using provided course structure for chunking
        @raises: ValueError if pdf_path is invalid
                RuntimeError if processing fails
        """
    def process_with_json(self, pdf_path: str, course_json: Dict) -> Dict[str, Any]:
        """Process PDF with provided course JSON"""
        try:
            processor = PDFParser(pdf_path, self.upload_dir)
            processor.load_course_json(course_json)
            result = processor.process_document()
            return {
                "status": "success",
                "chunks": result.get("chunks", []),
                "images": result.get("images", []),
                "metadata": result.get("metadata", {})
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

        """
        Save course JSON to document directory
        @param document_dir: str - Directory path for the document
        @param course_json: Dict - Course structure JSON
        @returns: str - Path to the saved JSON file
        @description: Saves course JSON to the specified document directory
        """
    def save_course_json(self, document_dir: str, course_json: Dict):
        """Save course JSON to document directory"""
        json_path = Path(document_dir) / "course_structure.json"
        with open(json_path, 'w') as f:
            json.dump(course_json, f, indent=2)
        return str(json_path)
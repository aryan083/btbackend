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
            
        metadata = response.get('metadata')
        if not isinstance(metadata, dict):
            logger.error("Metadata is not a dictionary")
            return False
            
        required_fields = ['title', 'author', 'page_count']
        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            logger.error(f"Missing required metadata fields: {missing_fields}")
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

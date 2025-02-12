"""
Controller for handling content processing and generation.
"""
import logging
from typing import Dict, List
from ..services.pdf_parser import PDFParser

class ContentController:
    """Handles content processing and generation logic."""
    
    @staticmethod
    def process_book(file_path: str) -> Dict[str, List]:
        """
        Process book PDF and extract content.
        
        Args:
            file_path (str): Path to the book PDF file
            
        Returns:
            Dict[str, List]: Extracted content from the book
        """
        try:
            parser = PDFParser()
            content = parser.parse_pdf(file_path)
            return {
                'status': 'success',
                'data': content
            }
        except Exception as e:
            logging.error(f"Error processing book: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    @staticmethod
    def process_course_content(file_path: str) -> Dict[str, List[str]]:
        """
        Process course content PDF and extract structure.
        
        Args:
            file_path (str): Path to the course content PDF
            
        Returns:
            Dict[str, List[str]]: Extracted course structure
        """
        try:
            parser = PDFParser()
            content = parser.extract_course_content(file_path)
            return {
                'status': 'success',
                'data': content
            }
        except Exception as e:
            logging.error(f"Error processing course content: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            } 
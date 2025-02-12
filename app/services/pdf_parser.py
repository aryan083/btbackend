"""
Service for parsing PDF documents using unstructured library.
Extracts text content from PDFs and organizes by chapters.
"""
import logging
import json
import os
import re
from typing import Dict, List
from pathlib import Path
from datetime import datetime
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text
from ..config import Config

class PDFParser:
    """Handles PDF document parsing and text extraction."""
    
    @staticmethod
    def _clean_temp_files():
        """Clean up temporary files in the temp directory."""
        try:
            temp_dir = Config.TEMP_FOLDER
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logging.info("Temporary files cleaned successfully")
        except Exception as e:
            logging.error(f"Error cleaning temporary files: {str(e)}")

    @staticmethod
    def _identify_chapter(text: str) -> bool:
        """
        Check if text is a chapter heading.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is a chapter heading
        """
        # Common chapter patterns
        chapter_patterns = [
            r'^chapter\s+\d+',
            r'^chapter\s+[ivxlcdm]+',
            r'^\d+\.\s+',
            r'^[ivxlcdm]+\.\s+'
        ]
        
        text_lower = text.lower().strip()
        return any(re.match(pattern, text_lower) for pattern in chapter_patterns)

    @staticmethod
    def parse_pdf(file_path: str) -> Dict[str, List[str]]:
        """
        Parse PDF and extract text content organized by chapters.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Dict[str, List[str]]: Dictionary containing text content by chapters and metadata
        """
        try:
            logging.info(f"Starting PDF parsing for file: {file_path}")
            
            if not Path(file_path).exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            # Parse PDF using unstructured
            elements = partition_pdf(
                filename=file_path,
                strategy='auto',
                infer_table_structure=True,
                extract_image_block_types=['Image,table'],
                extract_forms=True,
                form_extraction_skip_tables=False              
            #extract Imges and tables two
            )
            # Extract text content and organize by chapters
            chapters = {}
            current_chapter = "Introduction"  # Default chapter
            current_text = []
            
            for element in elements:
                if isinstance(element, Text):
                    text = str(element).strip()
                    if text:
                        if PDFParser._identify_chapter(text):
                            # Save previous chapter
                            if current_text:
                                chapters[current_chapter] = current_text
                            # Start new chapter
                            current_chapter = text
                            current_text = []
                        else:
                            current_text.append(text)
            
            # Save last chapter
            if current_text:
                chapters[current_chapter] = current_text
            
            # Generate output filename
            input_filename = Path(file_path).stem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{input_filename}_text_{timestamp}.json"
            output_path = os.path.join(Config.SCANNED_PDF_FOLDER, output_filename)
            
            # Prepare response data
            parsed_content = {
                'chapters': chapters,
                'metadata': {
                    'filename': input_filename,
                    'timestamp': timestamp,
                    'total_chapters': len(chapters),
                    'chapter_names': list(chapters.keys())
                }
            }
            
            # Save parsed content to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_content, f, indent=4, ensure_ascii=False)
            
            parsed_content['output_path'] = output_path
            
            logging.info(f"Successfully extracted text from PDF: {file_path}")
            logging.info(f"Output saved to: {output_path}")
            
            # Clean up temp files
            PDFParser._clean_temp_files()
            
            return parsed_content
            
        except Exception as e:
            logging.error(f"Error parsing PDF {file_path}: {str(e)}")
            PDFParser._clean_temp_files()  # Clean up even if there's an error
            raise

    @staticmethod
    def extract_course_content(pdf_path: str) -> Dict[str, List[str]]:
        """
        Extract course syllabus and structure from PDF.
        
        Args:
            pdf_path (str): Path to the course content PDF
            
        Returns:
            Dict[str, List[str]]: Dictionary containing course structure
                {
                    'chapters': List[str],
                    'topics': Dict[str, List[str]]
                }
        """
        try:
            elements = partition_pdf(filename=pdf_path)
            
            course_structure = {
                'chapters': [],
                'topics': {}
            }
            
            current_chapter = None
            
            for element in elements:
                if isinstance(element, Title):
                    current_chapter = str(element)
                    course_structure['chapters'].append(current_chapter)
                    course_structure['topics'][current_chapter] = []
                elif isinstance(element, ListItem) and current_chapter:
                    course_structure['topics'][current_chapter].append(str(element))
            
            return course_structure
            
        except Exception as e:
            logging.error(f"Error extracting course content: {str(e)}")
            raise 
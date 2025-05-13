"""
Enhanced PDF parser that combines PyMuPDF for image extraction and unstructured for text extraction
"""
import json
import logging
import colorlog
from pathlib import Path
from typing import Dict, List, Any, Optional
import fitz  # PyMuPDF
from PIL import Image
import io
from unstructured.partition.pdf import partition_pdf
from datetime import datetime
from config import Config
import re
from run import custom_logger
# Set up colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    Config.LOG_FORMAT,
    log_colors=Config.LOG_COLORS
))
logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DocumentService:
    """High-level service for document processing operations"""
    @custom_logger.log_function_call
    def __init__(self, output_base_dir: str):
        """Initialize the document service
        
        Args:
            output_base_dir: Base directory for storing processed documents
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
    @custom_logger.log_function_call
    def process_document(self, pdf_file_path: str, extract_images: bool = True,
                        extract_text: bool = True, save_json: bool = True) -> Dict[str, Any]:
        """Process a PDF document and extract its content
        
        Args:
            pdf_file_path: Path to the PDF file
            extract_images: Whether to extract images
            extract_text: Whether to extract text
            save_json: Whether to save results as JSON
            
        Returns:
            Dict containing processing results and extracted content
        """
        try:
            parser = PDFParser(pdf_file_path, str(self.output_base_dir))
            
            try:
                result = parser.process_document(
                    extract_images=extract_images,
                    extract_text=extract_text,
                    save_json=save_json
                )
                
                return {
                    'status': 'success',
                    'message': 'Document processed successfully',
                    'data': result
                }
                
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                return {
                    'status': 'error',
                    'message': f'Failed to process document: {str(e)}'
                }
                
            finally:
                parser.close()
                
        except Exception as e:
            logger.error(f"Error initializing document processing: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to initialize document processing: {str(e)}'
            }

class PDFParser:
    """Enhanced PDF parser for text and image extraction"""
    @custom_logger.log_function_call
    def __init__(self, pdf_path: str, output_base_dir: str):
        """
        Initialize the PDF parser
        @param pdf_path: str - Path to the PDF file
        @param output_base_dir: str - Base directory to store extracted content
        @returns: None
        @description: Creates directories for text and images, initializes PDF document
        """
        self.pdf_path = Path(pdf_path)
        self.book_name = self.pdf_path.stem
        self.output_dir = Path(output_base_dir) / self.book_name
        self.text_dir = self.output_dir / "Text"
        self.images_dir = self.output_dir / "Images"
        
        # Create directories
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.doc = fitz.open(pdf_path)
            self.metadata = self.doc.metadata
            logger.info(f"Opened PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to open PDF: {str(e)}")
            raise

    @custom_logger.log_function_call
    def _process_toc(self) -> List[Dict[str, Any]]:
        """
        Process table of contents into chapter structure
        @returns: List of chapter dictionaries with start and end pages
        @description: Extracts chapter information from PDF table of contents
        """
        toc = self.doc.get_toc()
        chapters = []
        top_level = [entry for entry in toc if entry[0] == 1]
        
        if not top_level:
            return [{'title': 'Full Document', 'start_page': 1, 'end_page': len(self.doc)}]
        
        for i, entry in enumerate(top_level):
            _, title, page = entry[:3]
            start = page  # PyMuPDF uses 0-based page numbers
            end = top_level[i+1][2] if i < len(top_level)-1 else len(self.doc)
            end = end if isinstance(end, int) else len(self.doc)
            chapters.append({
                'title': title,
                'start_page': start,
                'end_page': end
            })
        
        return chapters
    @custom_logger.log_function_call
    def _extract_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from PDF document
        @returns: Dict containing document metadata
        @description: Extracts title, author, creation date, and page count
        """
        try:
            metadata = {
                'title': self.metadata.get('title', self.book_name),
                'author': self.metadata.get('author', ''),
                'creation_date': self._parse_date(self.metadata.get('creationDate')),
                'page_count': len(self.doc)
            }
            logger.info("Metadata extraction complete")
            return metadata
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            raise
    @custom_logger.log_function_call
    def _parse_date(self, pdf_date: Optional[str]) -> Optional[str]:
        """
        Parse PDF date format to ISO-8601
        @param pdf_date: Optional PDF date string
        @returns: Formatted date string or None
        @description: Converts PDF date format to standard ISO format
        """
        if pdf_date:
            try:
                return datetime.strptime(pdf_date[2:16], "%Y%m%d%H%M%S").isoformat()
            except Exception:
                return None
        return None
    @custom_logger.log_function_call
    def _extract_images(self) -> List[Dict[str, Any]]:
        """
        Extract images from PDF pages
        @returns: List[Dict[str, Any]] - List of image information dictionaries
        @description: Extracts images from each page and saves them with metadata
        """
        images = []
        try:
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                page_images = page.get_images()
                
                if page_images:
                    logger.info(f"Found {len(page_images)} images on page {page_num + 1}")
                
                for img_idx, img in enumerate(page_images):
                    try:
                        xref = img[0]
                        base_image = self.doc.extract_image(xref)
                        
                        if base_image:
                            # Generate unique filename
                            image_filename = f"page_{page_num + 1}_img_{img_idx + 1}.{base_image['ext']}"
                            image_path = self.images_dir / image_filename
                            
                            # Save image file
                            with open(image_path, "wb") as f:
                                f.write(base_image["image"])
                            
                            # Store image metadata
                            image_info = {
                                'path': str(image_path.relative_to(self.output_dir)),
                                'page_number': page_num + 1,
                                'dimensions': {
                                    'width': base_image['width'],
                                    'height': base_image['height']
                                },
                                'filename': image_filename,
                                'format': base_image['ext'],
                                'colorspace': base_image['colorspace'],
                                'size_bytes': len(base_image["image"])
                            }
                            images.append(image_info)
                            logger.info(f"Saved image: {image_path}")
                            
                    except Exception as e:
                        logger.error(f"Failed to extract image {img_idx} from page {page_num + 1}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Image extraction failed: {str(e)}")
            raise
            
        return images
    @custom_logger.log_function_call
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into smaller chunks with overlap
        @param text: str - Text to split
        @param chunk_size: int - Target size for each chunk
        @param overlap: int - Number of characters to overlap between chunks
        @returns: List[str] - List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size
            
            if end < len(text):
                # Try to find a natural break point (sentence or paragraph)
                break_chars = ['.', '\n', '!', '?']
                for i in range(min(end + 100, len(text) - 1), start, -1):
                    if text[i] in break_chars:
                        end = i + 1
                        break
            else:
                end = len(text)
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Move start position for next chunk, including overlap
            start = max(end - overlap, 0)
            
        return chunks
    @custom_logger.log_function_call
    def _extract_text_with_unstructured(self) -> List[Dict[str, Any]]:
        """
        Extract text from PDF pages using unstructured
        @returns: List of dictionaries containing text elements
        @description: Extracts text from each page and returns structured data
        """
        text_elements = []
        try:
            elements = partition_pdf(
                filename=str(self.pdf_path)
            )

            structured = []
            for el in elements:
                # Split the text into smaller chunks
                chunks = self._chunk_text(el.text)
                
                # Create an entry for each chunk
                for chunk_idx, chunk_text in enumerate(chunks):
                    entry = {
                        'type': el.category,
                        'text': chunk_text,
                        'metadata': {
                            'page_number': el.metadata.page_number if hasattr(el.metadata, 'page_number') else None,
                            'coordinates': el.metadata.coordinates.to_dict() if hasattr(el.metadata, 'coordinates') else None,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks)
                        }
                    }
                    structured.append(entry)

            text_elements.extend(structured)
            
            # Save text elements to JSON
            if text_elements:
                json_path = self.text_dir / f"{self.book_name}_text_elements.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(text_elements, f, ensure_ascii=False, indent=2)
                logger.info(f"Text elements saved to {json_path}")
            
            return text_elements

        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise
    @custom_logger.log_function_call
    def _save_json_files(self, metadata: Dict[str, Any], chapters: List[Dict[str, Any]], 
                        text_elements: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Save extracted content to JSON files
        @param metadata: Dict[str, Any] - Document metadata
        @param chapters: List[Dict[str, Any]] - Chapter information
        @param text_elements: List[Dict[str, Any]] - Extracted text elements
        @param images: List[Dict[str, Any]] - Extracted image information
        @returns: Dict[str, str] - Paths to saved JSON files
        @description: Saves metadata, chapters, and content to structured JSON files
        """
        try:
            # Save metadata and table of contents
            index_data = {
                'metadata': metadata,
                'table_of_contents': [
                    {
                        'title': chapter['title'],
                        'page_range': {
                            'start': chapter['start_page'],
                            'end': chapter['end_page']
                        }
                    }
                    for chapter in chapters
                ]
            }
            
            index_path = self.text_dir / "index.json"
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=4, ensure_ascii=False)
            
            # Save chapter content
            chapter_files = []
            for idx, chapter in enumerate(chapters):
                chapter_data = {
                    'metadata': {
                        'number': idx + 1,
                        'title': chapter['title'],
                        'page_range': {
                            'start': chapter['start_page'],
                            'end': chapter['end_page']
                        }
                    },
                    'content': []
                }
                
                # Add page content for this chapter
                for page_num in range(chapter['start_page'], chapter['end_page'] + 1):
                    page_content = {
                        'page_number': page_num,
                        'text': [],
                        'images': []
                    }
                    
                    # Add text elements for this page
                    page_text = [elem for elem in text_elements 
                               if elem['metadata']['page_number'] == page_num]
                    page_content['text'] = page_text
                    
                    # Add images for this page
                    page_images = [img for img in images if img['page_number'] == page_num]
                    page_content['images'] = page_images
                    
                    chapter_data['content'].append(page_content)
                
                # Save chapter file
                safe_title = re.sub(r'[^\w-]', '_', chapter['title']).lower()
                chapter_path = self.text_dir / f"chapter_{idx+1}_{safe_title[:50]}.json"
                with open(chapter_path, 'w', encoding='utf-8') as f:
                    json.dump(chapter_data, f, indent=4, ensure_ascii=False)
                chapter_files.append(str(chapter_path))
            
            # Save image metadata
            if images:
                image_meta_path = self.images_dir / "image_metadata.json"
                with open(image_meta_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'total_images': len(images),
                        'images': images
                    }, f, indent=4, ensure_ascii=False)
            
            return {
                'index': str(index_path),
                'chapters': chapter_files,
                'images_meta': str(image_meta_path) if images else None
            }
            
        except Exception as e:
            logger.error(f"Failed to save JSON files: {str(e)}")
            raise
    @custom_logger.log_function_call
    def process_document(self, extract_images: bool = True, 
                         extract_text: bool = True, 
                         save_json: bool = True) -> Dict[str, Any]:
        """
        Process the PDF document and extract content based on specified options
        @param extract_images: bool - Whether to extract images from the PDF
        @param extract_text: bool - Whether to extract text from the PDF
        @param save_json: bool - Whether to save extracted content as JSON
        @returns: Dict[str, Any] - Document structure with metadata and content
        @description: Extracts metadata, processes table of contents, and extracts
                     content based on specified options. Returns structured data
                     including metadata, chapters, and paths to extracted content.
        """
        try:
            # Extract metadata
            metadata = self._extract_metadata()
            logger.info(f"Metadata extracted for {self.book_name}")

            # Process table of contents
            chapters = self._process_toc()
            logger.info(f"Table of contents processed: {len(chapters)} chapters found")

            # Initialize result structure
            result = {
                'status': 'success',
                'metadata': metadata,
                'chapters': chapters,
                'images': [],
                'text_elements': []
            }

            # Extract images if requested
            if extract_images:
                try:
                    images = self._extract_images()
                    result['images'] = images
                    logger.info(f"Images extracted: {len(images)} images found")
                except Exception as e:
                    logger.error(f"Image extraction failed: {str(e)}")
                    result['image_extraction_error'] = str(e)

            # Extract text if requested
            if extract_text:
                try:
                    text_elements = self._extract_text_with_unstructured()
                    result['text_elements'] = text_elements
                    logger.info(f"Text extracted: {len(text_elements)} elements found")
                except Exception as e:
                    logger.error(f"Text extraction failed: {str(e)}")
                    result['text_extraction_error'] = str(e)

            # Save to JSON if requested
            if save_json:
                try:
                    json_files = self._save_json_files(metadata, chapters, text_elements, result['images'])
                    result['json_files'] = json_files
                    logger.info("JSON files saved successfully")
                except Exception as e:
                    logger.error(f"JSON saving failed: {str(e)}")
                    result['json_save_error'] = str(e)

            return result

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {
                'status': 'error',
                'message': f"Document processing failed: {str(e)}"
            }
    @custom_logger.log_function_call
    def close(self):
        """
        Close the PDF document and clean up resources
        @returns: None
        """
        try:
            self.doc.close()
        except Exception as e:
            logger.error(f"Error closing document: {str(e)}")
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
from fuzzywuzzy import fuzz

# Set up colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    Config.LOG_FORMAT,
    log_colors=Config.LOG_COLORS
))
logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class PDFParser:
    """Enhanced PDF parser for text and image extraction"""

    def __init__(self, pdf_path: str, output_base_dir: str):
        """
        Initialize the PDF parser
        @param pdf_path: str - Path to the PDF file
        @param output_base_dir: str - Base directory to store extracted content
        @returns: None
        @description: Creates directories for text and images, initializes PDF document
        @raises: ValueError if pdf_path is invalid or not a PDF file
                RuntimeError if PDF cannot be opened
        """
        self.pdf_path = Path(pdf_path)
        
        # Validate PDF file
        if not self.pdf_path.exists():
            raise ValueError(f"PDF file does not exist: {pdf_path}")
        if self.pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
            
        self.book_name = self.pdf_path.stem
        self.output_dir = Path(output_base_dir)
        self.text_dir = self.output_dir / "Text"
        self.images_dir = self.output_dir / "Images"
        self.course_json = None
        self.chapter_map = {}
        self.doc = None

        try:
            # Create directories
            self.text_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)
            
            # Open and validate PDF
            self.doc = fitz.open(pdf_path)
            if not self.doc.is_pdf:
                raise ValueError("Document is not a valid PDF")
                
            self.metadata = self.doc.metadata
            logger.info(f"Opened PDF: {pdf_path}")
            
        except Exception as e:
            # Clean up if initialization fails
            if self.doc:
                self.doc.close()
            logger.error(f"Failed to initialize PDF parser: {str(e)}")
            raise RuntimeError(f"Failed to open PDF: {str(e)}")

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

    def _extract_images(self) -> List[Dict[str, Any]]:
        """
        Extract images from PDF pages
        @returns: List[Dict[str, Any]] - List of image information dictionaries
        @description: Extracts images from each page and saves them with metadata
        """
        images = []
        image_refs = set()  # Track unique image references
        
        try:
            if not self.doc:
                raise ValueError("PDF document not properly initialized")
                
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                page_images = page.get_images()
                
                if page_images:
                    logger.info(f"Found {len(page_images)} images on page {page_num + 1}")
                
                for img_idx, img in enumerate(page_images):
                    try:
                        xref = img[0]
                        # Skip if we've already processed this image
                        if xref in image_refs:
                            continue
                            
                        image_refs.add(xref)
                        base_image = self.doc.extract_image(xref)
                        
                        if not base_image:
                            logger.warning(f"Failed to extract image data for xref {xref} on page {page_num + 1}")
                            continue
                            
                        # Generate unique filename
                        image_filename = f"page_{page_num + 1}_img_{img_idx + 1}.{base_image['ext']}"
                        image_path = self.images_dir / image_filename
                        
                        try:
                            # Save image file with proper error handling
                            with open(image_path, "wb") as f:
                                f.write(base_image["image"])
                                
                            # Validate saved image
                            if not image_path.exists() or image_path.stat().st_size == 0:
                                raise IOError("Failed to save image or empty file created")
                                
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
                            
                        except IOError as e:
                            logger.error(f"Failed to save image {image_filename}: {str(e)}")
                            # Clean up partially written file
                            if image_path.exists():
                                image_path.unlink()
                            continue
                            
                    except Exception as e:
                        logger.error(f"Failed to extract image {img_idx} from page {page_num + 1}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Image extraction failed: {str(e)}")
            raise
            
        return images

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
                entry = {
                    'type': el.category,
                    'text': el.text,
                    'metadata': {
                        'page_number': el.metadata.page_number if hasattr(el.metadata, 'page_number') else None,
                        'coordinates': el.metadata.coordinates.to_dict() if hasattr(el.metadata, 'coordinates') else None
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

    def _create_chapter_map(self):
        """Map JSON chapters to PDF structure"""
        try:
            toc = self._get_processed_toc()
            for json_chapter in self.course_json.get("Chapters", {}).keys():
                best_match = max(toc, 
                    key=lambda x: fuzz.token_sort_ratio(x['title'], json_chapter))
                self.chapter_map[json_chapter] = {
                    'start_page': best_match['start_page'],
                    'end_page': best_match['end_page']
                }
        except Exception as e:
            logger.error(f"Chapter mapping failed: {str(e)}")
            raise

 
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
        
        # Add subtopic chunks if using JSON
        if self.course_json:
            chunks = self._chunk_by_subtopics(text_elements, images)
            chunk_dir = self.text_dir / "subtopic_chunks"
            chunk_dir.mkdir(exist_ok=True)
            
            chunk_files = []
            for chunk in chunks:
                safe_code = chunk['subtopic_code'].replace('.', '_')
                chunk_path = chunk_dir / f"chunk_{safe_code}.json"
                with open(chunk_path, 'w') as f:
                    json.dump(chunk, f, indent=2)
                chunk_files.append(str(chunk_path))
            
            result['subtopic_chunks'] = chunk_files
        
        return result

  
        
    # def process_book(self) -> Dict[str, Any]:
    #     """
    #     Process a book PDF document
    #     @returns: Dict[str, Any] - Document structure with metadata and content
    #     @description: Processes the PDF as a book, extracting metadata, text, and images
    #     """
    #     return self.process_document(extract_images=True, extract_text=True, save_json=True)

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
            # Initialize result structure with default values
            result = {
                'status': 'success',
                'metadata': {},
                'chapters': [],
                'images': [],
                'text_elements': [],
                'chunks': [],
                'course_json': False
            }
            
            # Extract core components
            result['metadata'] = self._extract_metadata()
            logger.info(f"Metadata extracted for {self.book_name}")

            if extract_text:
                result['text_elements'] = self._extract_text_with_unstructured()
                logger.info(f"Text extracted: {len(result['text_elements'])} elements found")

            if extract_images:
                result['images'] = self._extract_images()
                logger.info(f"Images extracted: {len(result['images'])} images found")

            # Process structure
            result['chapters'] = self._process_toc()
            logger.info(f"Table of contents processed: {len(result['chapters'])} chapters found")

            # Handle course JSON chunking
            if self.course_json:
                try:
                    result['chunks'] = self._chunk_by_subtopics(result['text_elements'], result['images'])
                    result['course_json'] = True
                    logger.info(f"Created {len(result['chunks'])} subtopic chunks")
                except Exception as chunk_error:
                    logger.error(f"Subtopic chunking failed: {str(chunk_error)}")
                    result['chunking_error'] = str(chunk_error)

            # Save files
            if save_json:
                try:
                    json_files = self._save_json_files(
                        result['metadata'],
                        result['chapters'],
                        result['text_elements'],
                        result['images']
                    )
                    result['json_files'] = json_files
                    logger.info("JSON files saved successfully")
                except Exception as save_error:
                    logger.error(f"JSON saving failed: {str(save_error)}")
                    result['json_save_error'] = str(save_error)

            return result

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {
                'status': 'error',
                'message': f"Document processing failed: {str(e)}"
            }

    def close(self):
        """
        Close the PDF document and clean up resources
        @returns: None
        """
        try:
            self.doc.close()
        except Exception as e:
            logger.error(f"Error closing document: {str(e)}")

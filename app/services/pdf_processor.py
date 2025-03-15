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
        self.output_dir = Path(output_base_dir) / self.book_name
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
                            # Clean up any partially written file
                            if image_path.exists():
                                try:
                                    image_path.unlink()
                                except Exception as cleanup_error:
                                    logger.warning(f"Failed to clean up partial image file: {str(cleanup_error)}")
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

    def _save_course_json(self, course_json: Dict):
        """Save course JSON to output directory"""
        json_path = self.output_dir / "course_structure.json"
        with open(json_path, 'w') as f:
            json.dump(course_json, f, indent=2)
        return json_path

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

    def _load_course_json(self):
        """Load course JSON from output directory"""
        try:
            json_path = self.output_dir / "course_structure.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    self.course_json = json.load(f)
                self._map_chapters_to_toc()
            else:
                logger.warning(f"Course JSON not found at {json_path}")
                self.course_json = None
        except Exception as e:
            logger.error(f"Failed to load course JSON: {str(e)}")
            self.course_json = None

    def load_course_json(self, course_data: Dict):
        """Load course JSON structure for chunking"""
        self.course_json = course_data
        self._map_chapters_to_toc()
        
        
    def _map_chapters_to_toc(self):
        """Map course JSON chapters to PDF table of contents"""
        try:
            if not self.course_json:
                return
            
            if not self.doc:
                logger.warning("Cannot map chapters to TOC: PDF document not initialized")
                return
                
            # Clear existing TOC
            self.doc.set_toc([])
            
            # Add new entries from course JSON
            new_toc = []
            for chapter in self.course_json.get('chapters', []):
                if 'title' not in chapter or 'start_page' not in chapter:
                    logger.warning(f"Skipping invalid chapter entry: {chapter}")
                    continue
                    
                new_toc.append([1, chapter['title'], chapter['start_page'] - 1])  # Convert 1-based to 0-based page index
                
            self.doc.set_toc(new_toc)
            logger.info(f"Updated TOC with {len(new_toc)} chapters from course JSON")
        except Exception as e:
            logger.error(f"Failed to map chapters to TOC: {str(e)}")
        
        
    def process_document(self) -> Dict:
        """Main document processing method"""
        try:
            if self.course_json:
                self._map_chapters_to_toc()
            
            return {
                'status': 'success',
                'metadata': self._extract_metadata(),
                'chapters': self._process_toc(),
                'images': self._extract_images(),
                'text_elements': self._extract_text_with_unstructured()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    def _chunk_by_subtopics(self, text_elements: List[Dict], images: List[Dict]) -> List[Dict]:
        """Create subtopic chunks using JSON structure"""
        try:
            if not self.course_json:
                logger.warning("No course JSON available for chunking")
                return []
                
            # Validate chapter_map is populated
            if not self.chapter_map:
                logger.warning("Chapter map is empty, cannot create subtopic chunks")
                return []
                
            chunks = []
            
            for json_chap, chap_data in self.chapter_map.items():
                # Get elements for this chapter
                chap_elements = [
                    e for e in text_elements
                    if chap_data['start_page'] <= e['metadata']['page_number'] <= chap_data['end_page']
                ]
                
                current_chunk = []
                current_subtopic = None
                buffer_images = []
                
                for elem in chap_elements:
                    if self._is_subtopic_header(elem, chap_data['subtopics']):
                        if current_subtopic:
                            chunks.append(self._create_subtopic_chunk(
                                current_chunk,
                                current_subtopic,
                                buffer_images,
                                images
                            ))
                            buffer_images = []
                        current_subtopic = self._match_subtopic(elem['text'], chap_data['subtopics'])
                        current_chunk = [elem]
                    else:
                        current_chunk.append(elem)
                    
                    if elem['type'] == 'Image':
                        buffer_images.append(elem['metadata'])
                
                if current_subtopic:
                    chunks.append(self._create_subtopic_chunk(
                        current_chunk,
                        current_subtopic,
                        buffer_images,
                        images
                    ))
                    
            
            return chunks
            
        except Exception as e:
            # Handle chunking errors
            logger.error(f"Subtopic chunking failed: {str(e)}")
            # Fallback to chapter-based chunking
            
            return self.process_document(True,True,True)

    def _is_subtopic_header(self, element: Dict, subtopics: Dict) -> bool:
        """Check if element is a subtopic header"""
        text = element['text'].lower()
        return any(
            fuzz.partial_ratio(text, f"{num} {title.lower()}") > 85
            for num, title in subtopics.items()
        )

    def _match_subtopic(self, text: str, subtopics: Dict) -> Optional[str]:
        """Find best matching subtopic code"""
        best_match = max(
            subtopics.items(),
            key=lambda item: fuzz.partial_ratio(text.lower(), f"{item[0]} {item[1].lower()}"),
            default=None
        )
        return best_match[0] if best_match else None

    def _create_subtopic_chunk(self, elements: List, subtopic_code: str, 
                             buffer_images: List, all_images: List) -> Dict:
        """Create chunk JSON with image associations"""
        pages = {e['metadata']['page_number'] for e in elements}
        chunk_images = [
            img for img in all_images
            if img['page_number'] in pages
        ] + buffer_images
        
        return {
            "subtopic_code": subtopic_code,
            "content": " ".join(e['text'] for e in elements),
            "images": [img['path'] for img in chunk_images],
            "page_range": {
                "start": min(e['metadata']['page_number'] for e in elements),
                "end": max(e['metadata']['page_number'] for e in elements)
            }
        }

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

            # Load existing course JSON if available
            try:
                self._load_course_json()
            except Exception as e:
                logger.warning(f"Failed to load course JSON, proceeding without it: {str(e)}")
                self.course_json = None
            
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

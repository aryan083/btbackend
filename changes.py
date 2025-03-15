 # pdf_processor.py
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from fuzzywuzzy import fuzz
import fitz
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)

class AsyncPDFProcessor:
    """Handles PDF processing with async JSON integration"""
    
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.processing_data = {}  # {process_id: {elements, images}}

    def start_processing(self, pdf_path: str) -> str:
        """Initial PDF processing, returns process ID"""
        process_id = str(uuid.uuid4())
        output_dir = self.upload_dir / process_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract raw elements and images
            elements = partition_pdf(pdf_path, strategy="fast")
            images = self._extract_images(pdf_path, output_dir)
            
            # Store for async processing
            self.processing_data[process_id] = {
                "elements": [self._element_to_dict(e) for e in elements],
                "images": images,
                "output_dir": output_dir
            }
            
            return process_id
            
        except Exception as e:
            logger.error(f"Initial processing failed: {str(e)}")
            raise

    def finalize_with_json(self, process_id: str, course_json: Dict) -> Dict:
        """Final processing with received JSON"""
        if process_id not in self.processing_data:
            raise ValueError("Invalid process ID")
            
        data = self.processing_data[process_id]
        output_dir = data['output_dir']
        
        try:
            # Create final directory structure
            book_name = Path(data['elements'][0]['metadata']['filename']).stem
            final_dir = self.upload_dir / book_name
            final_text_dir = final_dir / "text"
            final_images_dir = final_dir / "images"
            final_text_dir.mkdir(parents=True, exist_ok=True)
            final_images_dir.mkdir(parents=True, exist_ok=True)

            # Process chunks
            chunks = []
            for chap_title, subtopics in course_json['course_content']['Chapters'].items():
                chap_chunks = self._process_chapter(
                    data['elements'], 
                    subtopics,
                    final_images_dir
                )
                chunks.extend(chap_chunks)
                
                # Save chapter chunks
                for chunk in chap_chunks:
                    chunk_file = final_text_dir / f"{chap_title}_{chunk['subtopic_code']}.json"
                    with open(chunk_file, 'w') as f:
                        json.dump(chunk, f, indent=2)

            # Create index
            index_data = {
                "chunks": [{
                    "chapter": c['chapter'],
                    "subtopic": c['subtopic_code'],
                    "file": str(final_text_dir / f"{c['chapter']}_{c['subtopic_code']}.json")
                } for c in chunks]
            }
            with open(final_dir / "index.json", 'w') as f:
                json.dump(index_data, f, indent=2)

            return {
                "status": "success",
                "output_dir": str(final_dir),
                "chunk_count": len(chunks)
            }
            
        finally:
            # Cleanup temporary data
            del self.processing_data[process_id]

    def _process_chapter(self, elements: List[Dict], subtopics: Dict, img_dir: Path) -> List[Dict]:
        """Process a chapter into subtopic chunks"""
        chunks = []
        current_chunk = []
        current_subtopic = None
        buffer_images = []
        
        for elem in elements:
            if self._is_subtopic_header(elem, subtopics):
                if current_subtopic:
                    chunks.append(self._finalize_chunk(
                        current_chunk,
                        current_subtopic,
                        buffer_images,
                        img_dir
                    ))
                    buffer_images = []
                current_subtopic = self._match_subtopic(elem['text'], subtopics)
                current_chunk = [elem]
            else:
                current_chunk.append(elem)
            
            # Track nearby images
            if elem['category'] == 'Image':
                buffer_images.append(elem['metadata']['image_path'])
                
        return chunks

    def _is_subtopic_header(self, element: Dict, subtopics: Dict) -> bool:
        """Match element to subtopic headers using fuzzy matching"""
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

    def _finalize_chunk(self, elements: List, subtopic_code: str, images: List[str], img_dir: Path) -> Dict:
        """Create chunk JSON structure"""
        return {
            "chapter": elements[0]['metadata']['chapter'],
            "subtopic_code": subtopic_code,
            "content": " ".join([e['text'] for e in elements]),
            "images": [str(img_dir / img) for img in images],
            "page_range": {
                "start": elements[0]['metadata']['page_number'],
                "end": elements[-1]['metadata']['page_number']
            }
        }

    def _extract_images(self, pdf_path: str, output_dir: Path) -> List[Dict]:
        """Extract images with positions"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_idx, img in enumerate(page.get_images()):
                xref = img[0]
                base_img = doc.extract_image(xref)
                img_path = output_dir / f"page_{page_num+1}_img_{img_idx+1}.{base_img['ext']}"
                
                with open(img_path, 'wb') as f:
                    f.write(base_img["image"])
                
                images.append({
                    "path": str(img_path),
                    "page": page_num + 1,
                    "bbox": img[1:5]  # (x0, y0, x1, y1)
                })
                
        return images

    def _element_to_dict(self, element) -> Dict:
        """Convert unstructured element to serializable dict"""
        return {
            "text": element.text,
            "category": element.category,
            "metadata": {
                "page_number": element.metadata.page_number,
                "coordinates": element.metadata.coordinates.to_dict() if element.metadata.coordinates else None,
                "filename": element.metadata.filename
            }
        }
        
        
# rag_controller.py
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid

rag_bp = Blueprint('rag', __name__)
processor = AsyncPDFProcessor(Path("static/uploads"))

@rag_bp.route('/init_processing', methods=['POST'])
def start_processing():
    """Initiate PDF processing, returns process ID"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Save to temp location
        temp_dir = Path("/tmp") / str(uuid.uuid4())
        temp_dir.mkdir()
        pdf_path = temp_dir / secure_filename(file.filename)
        file.save(pdf_path)
        
        process_id = processor.start_processing(str(pdf_path))
        return jsonify({"process_id": process_id})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@rag_bp.route('/complete_processing', methods=['POST'])
def complete_processing():
    """Complete processing with course JSON"""
    data = request.get_json()
    if not data or 'process_id' not in data or 'course_json' not in data:
        return jsonify({"error": "Missing parameters"}), 400
        
    try:
        result = processor.finalize_with_json(
            data['process_id'],
            data['course_json']
        )
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
    
    

# rag_service.py
from typing import Dict, Any
from pathlib import Path

class DocumentService:
    """Service wrapper for async processing"""
    
    def __init__(self, upload_dir: str):
        self.processor = AsyncPDFProcessor(Path(upload_dir))
        
    def process_async(self, pdf_path: str) -> Dict[str, Any]:
        """Initiate async processing flow"""
        process_id = self.processor.start_processing(pdf_path)
        return {"process_id": process_id}
        
    def finalize_processing(self, process_id: str, course_json: Dict) -> Dict[str, Any]:
        """Complete processing with course JSON"""
        return self.processor.finalize_with_json(process_id, course_json)
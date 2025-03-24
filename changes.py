'll help you modify the code to handle the JSON from send_pdf_to_gemini and integrate it with the chunking process. Here are the key changes needed:

1. In PDFParser class (pdf_processor.py):

python
Copy
# Add these new methods
def _save_course_json(self, course_json: Dict):
    """Save course JSON to output directory"""
    json_path = self.output_dir / "course_structure.json"
    with open(json_path, 'w') as f:
        json.dump(course_json, f, indent=2)
    return json_path

def _load_course_json(self):
    """Load course JSON if exists"""
    json_path = self.output_dir / "course_structure.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            self.course_json = json.load(f)
        self._map_chapters_to_toc()

# Modify process_document method
def process_document(self, extract_images: bool = True, 
                     extract_text: bool = True, 
                     save_json: bool = True) -> Dict[str, Any]:
    # Add this at the start of the method
    self._load_course_json()  # Auto-load course JSON if exists
    
    # Rest of your existing code...
    
    # Add this before saving JSON files
    if self.course_json and save_json:
        self._save_course_json(self.course_json)
    
    # Modify the return value
    result.update({
        'course_json': bool(self.course_json),
        'chunk_strategy': 'subtopics' if self.course_json else 'chapters'
    })
    return result
2. In send_pdf_to_gemini endpoint (rag_controller.py):

python
Copy
@rag_bp.route('/send_pdf_to_gemini', methods=['POST'])
def send_pdf_to_gemini():
    try:
        # ... existing code until course_content is generated ...
        
        # Save the generated JSON to document directory
        document_dir = Path(current_app.config['UPLOAD_FOLDER']) / document_name
        json_path = document_dir / "course_structure.json"
        with open(json_path, 'w') as f:
            json.dump({
                'course_content': course_content,
                'keywords': keywords,
                'welcome_message': welcome_message,
                'course_title': course_title
            }, f, indent=2)
        
        # Update response
        return jsonify({
            'status': 'success',
            'course_content': course_content,
            'saved_path': str(json_path),
            # ... rest of fields ...
        })
3. In DocumentService class (rag_service.py):

python
Copy
# Add this method
def process_with_course_json(self, pdf_path: str, course_json: Dict) -> Dict[str, Any]:
    processor = PDFParser(pdf_path, self.upload_dir)
    try:
        processor.load_course_json(course_json)
        return processor.process_document()
    finally:
        processor.close()
4. Add new endpoint in rag_controller.py:

python
Copy
@rag_bp.route('/process_with_json', methods=['POST'])
def process_with_json():
    try:
        data = request.get_json()
        if not data or 'pdf_path' not in data or 'course_json' not in data:
            return jsonify({"error": "Missing parameters"}), 400
            
        service = DocumentService(os.path.dirname(data['pdf_path']))
        result = service.process_with_course_json(
            data['pdf_path'],
            data['course_json']
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
Key Flow Changes:

When PDF is sent to Gemini:

Generated JSON is saved as course_structure.json in the document directory

Response includes path to saved JSON

During normal processing:

PDFParser auto-detects and loads course_structure.json if present

Uses JSON for subtopic chunking when available

New endpoint allows explicit JSON processing:

python
Copy
POST /process_with_json
{
    "pdf_path": "/uploads/book.pdf",
    "course_json": { ... }
}
Modified Chunking Behavior:

When course JSON exists:

Chunks are created per subtopic (1.1, 1.2, etc.)

Images are associated using page number ranges

Output files follow chapter_X_subtopic_Y.json format

Without course JSON:

Falls back to original chapter-based chunking

This maintains your existing file structure while adding JSON-guided processing. The system will automatically use the course JSON if present in the document directory.
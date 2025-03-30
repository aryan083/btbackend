"""
Service module for handling RAG (Retrieval Augmented Generation) operations.
This module provides functionality for generating HTML content using RAG techniques
with Supabase vector store and Google's Gemini model.
"""

import re
import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGGenerationService:
    """
    Service class for handling RAG generation operations.
    
    This class provides methods for generating HTML content using RAG techniques,
    combining vector search with LLM generation.
    
    Attributes:
        supabase_client (Client): Supabase client instance for vector store operations
        _gemini_model: Google's Gemini model instance (lazy loaded)
        _embedding_model: Sentence transformer model for generating embeddings (lazy loaded)
    """
    
    def __init__(self, supabase_url: str, supabase_key: str, gemini_api_key: str):
        """
        Initialize the RAG generation service.
        
        Args:
            supabase_url (str): URL of the Supabase instance
            supabase_key (str): API key for Supabase
            gemini_api_key (str): API key for Google's Gemini model
        """
        self.supabase_client = create_client(supabase_url, supabase_key)
        self._gemini_api_key = gemini_api_key
        self._gemini_model = None
        self._embedding_model = None

    @property
    def gemini_model(self):
        """
        Lazy load the Gemini model.
        
        Returns:
            The initialized Gemini model
        """
        if self._gemini_model is None:
            genai.configure(api_key=self._gemini_api_key)
            self._gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        return self._gemini_model

    @property
    def embedding_model(self):
        """
        Lazy load the embedding model.
        
        Returns:
            The initialized embedding model
        """
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
        return self._embedding_model

    def sanitize_path(self, name: str) -> str:
        """
        Sanitize names for Windows paths.
        
        Args:
            name (str): The name to sanitize
            
        Returns:
            str: Sanitized name safe for use in file paths
        """
        return re.sub(r'[<>:"/\\|?*:]', '_', name).strip()

    def get_course_structure(self, document_dir: str) -> Dict:
        """
        Load the course structure from JSON file.
        
        Args:
            document_dir (str): Directory containing the course structure JSON
            
        Returns:
            Dict: Course structure data
        """
        path = os.path.join(document_dir, 'course_structure.json')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading course structure: {str(e)}")
            return {"Chapters": {}}

    def generate_html_content(self, 
                            document_dir: str, 
                            course_id: str,
                            output_dir: str = "output",
                            match_threshold: float = 0.7, 
                            max_results: int = 5) -> Dict[str, str]:
        """
        Generate HTML content for all subtopics using RAG.
        
        Args:
            document_dir (str): Directory containing course documents
            course_id (str): ID of the course
            output_dir (str): Directory to save generated HTML files
            match_threshold (float): Threshold for vector similarity matching
            max_results (int): Maximum number of results to retrieve
            
        Returns:
            Dict[str, str]: Dictionary mapping file paths to generated content
        """
        try:
            course_structure = self.get_course_structure(document_dir)
            
            if not course_structure.get("Chapters"):
                logger.error("Invalid course structure format")
                return {}

            os.makedirs(output_dir, exist_ok=True)
            generated_files = {}

            for chapter_name, chapter_data in course_structure["Chapters"].items():
                if not isinstance(chapter_data, dict):
                    continue

                safe_chapter = self.sanitize_path(chapter_name).replace(" ", "_")
                chapter_dir = os.path.join(output_dir, safe_chapter)
                os.makedirs(chapter_dir, exist_ok=True)

                for subtopic_code, subtopic_name in chapter_data.items():
                    if not (isinstance(subtopic_code, str) and subtopic_code.replace(".", "").isdigit()):
                        continue

                    try:
                        safe_subtopic = self.sanitize_path(subtopic_code).replace(".", "_")
                        logger.info(f"Processing {safe_chapter}/{safe_subtopic}...")

                        # Generate embedding and retrieve context
                        query_text = f"Comprehensive technical explanation of {subtopic_name}"
                        query_embedding = self.embedding_model.encode(query_text).tolist()

                        context = self._retrieve_context(
                            query_embedding, 
                            course_id, 
                            match_threshold, 
                            max_results
                        )

                        # Generate content
                        content = self._generate_content(subtopic_name, subtopic_code, context)
                        
                        # Save content
                        filename = f"{safe_subtopic}.html"
                        filepath = os.path.normpath(os.path.join(chapter_dir, filename))
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        generated_files[filepath] = content
                        logger.info(f"Created: {filepath}")

                    except Exception as subtopic_error:
                        logger.error(f"Failed {subtopic_code}: {str(subtopic_error)}")

            return generated_files

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {}

    def _retrieve_context(self, 
                         query_embedding: List[float], 
                         course_id: str,
                         match_threshold: float,
                         max_results: int) -> str:
        """
        Retrieve relevant context from vector store.
        
        Args:
            query_embedding (List[float]): Query embedding vector
            course_id (str): ID of the course
            match_threshold (float): Threshold for matching
            max_results (int): Maximum number of results
            
        Returns:
            str: Retrieved context
        """
        try:
            response = self.supabase_client.rpc('match_documents', {
                'query_embedding': query_embedding,
                'match_threshold': match_threshold,
                'match_count': max_results,
                'course_filter': course_id
            }).execute()

            if response.data:
                return "\n".join(
                    f"<source>{r.get('content', '')}</source>" 
                    for r in response.data[:3]
                )
            return ""
        except Exception as e:
            logger.warning(f"Vector search failed: {str(e)}")
            return ""

    def _generate_content(self, subtopic_name: str, subtopic_code: str, context: str) -> str:
        """
        Generate HTML content using Gemini model.
        
        Args:
            subtopic_name (str): Name of the subtopic
            subtopic_code (str): Code of the subtopic
            context (str): Retrieved context
            
        Returns:
            str: Generated HTML content
        """
        prompt = f"""Generate exhaustive HTML body content about {subtopic_name} ({subtopic_code}).
        Use ALL available token capacity for depth and detail. Include:
        
        - Comprehensive definition section
        - Detailed technical breakdown of components
        - Multiple practical examples
        - Real-world implementation scenarios
        - Common pitfalls and troubleshooting
        - Advanced usage patterns
        - Cross-references to related concepts
        
        Context materials:
        {context}
        
        Requirements:
        - Strictly HTML body content only
        - No CSS/JS/head/meta tags
        - Semantic HTML5 elements
        - Technical depth over brevity
        - Tables for comparative analysis
        - Code samples in <pre><code>
        """

        response = self.gemini_model.generate_content(prompt)
        return response.text 

    def process_content(self, base_path: str, course_id: str) -> Dict[str, Any]:
        """
        Process both text and image content for a course.
        
        Args:
            base_path (str): Base path containing Text and Images directories
            course_id (str): ID of the course
            
        Returns:
            Dict[str, Any]: Processing results and statistics
        """
        try:
            results = {
                "text_processed": 0,
                "image_processed": 0,
                "errors": []
            }
            
            # Process text content
            text_results = self._process_text_content(base_path, course_id)
            results["text_processed"] = text_results["processed"]
            results["errors"].extend(text_results["errors"])
            
            # Process image content
            image_results = self._process_image_content(base_path, course_id)
            results["image_processed"] = image_results["processed"]
            results["errors"].extend(image_results["errors"])
            
            return results
            
        except Exception as e:
            logger.error(f"Content processing failed: {str(e)}")
            return {
                "text_processed": 0,
                "image_processed": 0,
                "errors": [str(e)]
            }

    def _process_text_content(self, base_path: str, course_id: str) -> Dict[str, Any]:
        """
        Process text content from JSON files.
        
        Args:
            base_path (str): Base path containing Text directory
            course_id (str): ID of the course
            
        Returns:
            Dict[str, Any]: Processing results
        """
        results = {"processed": 0, "errors": []}
        text_dir = Path(base_path) / "Text"
        
        if not text_dir.exists():
            logger.warning(f"Text directory not found: {text_dir}")
            return results

        for text_file in text_dir.glob("*.json"):
            try:
                with open(text_file, 'r') as f:
                    data = json.load(f)
                    content = data.get('content', '')
                    
                    if not content:
                        continue
                    
                    # Generate summary and embedding
                    summary = self._generate_text_summary(content)
                    if not summary:
                        continue
                    
                    embedding = self._generate_embedding(summary)
                    if not embedding:
                        continue
                    
                    # Create record
                    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text_file.stem))
                    record = {
                        'course_id': course_id,
                        'chunk_id': chunk_id,
                        'content': content,
                        'content_summary': summary,
                        'summary_vector': embedding
                    }
                    
                    # Store in database
                    if self._store_record('material_text', record):
                        results["processed"] += 1
                    else:
                        results["errors"].append(f"Failed to store text record: {chunk_id}")
                        
            except Exception as e:
                results["errors"].append(f"Error processing text file {text_file}: {str(e)}")
                
        return results

    def _process_image_content(self, base_path: str, course_id: str) -> Dict[str, Any]:
        """
        Process image content.
        
        Args:
            base_path (str): Base path containing Images directory
            course_id (str): ID of the course
            
        Returns:
            Dict[str, Any]: Processing results
        """
        results = {"processed": 0, "errors": []}
        image_dir = Path(base_path) / "Images"
        
        if not image_dir.exists():
            logger.warning(f"Images directory not found: {image_dir}")
            return results

        for image_file in image_dir.glob("*"):
            if image_file.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
                
            try:
                # Generate summary and embedding
                summary = self._generate_image_summary(str(image_file))
                if not summary:
                    continue
                
                embedding = self._generate_embedding(summary)
                if not embedding:
                    continue
                
                # Create record
                chunk_id = image_file.stem
                record = {
                    'course_id': course_id,
                    'chunk_id': chunk_id,
                    'content_path': str(image_file),
                    'content_summary': summary,
                    'summary_vector': embedding
                }
                
                # Store in database
                if self._store_record('material_image', record):
                    results["processed"] += 1
                else:
                    results["errors"].append(f"Failed to store image record: {chunk_id}")
                    
            except Exception as e:
                results["errors"].append(f"Error processing image {image_file}: {str(e)}")
                
        return results

    def _generate_text_summary(self, text: str) -> Optional[str]:
        """
        Generate text summary using Gemini model.
        
        Args:
            text (str): Text to summarize
            
        Returns:
            Optional[str]: Generated summary or None if failed
        """
        try:
            response = self.gemini_model.generate_content(
                f"Generate a concise summary of this text: {text}"
            )
            return response.text
        except Exception as e:
            logger.error(f"Text summary error: {e}")
            return None

    def _generate_image_summary(self, image_path: str) -> Optional[str]:
        """
        Generate image summary using Gemini model.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            Optional[str]: Generated summary or None if failed
        """
        try:
            img = Image.open(image_path)
            response = self.gemini_model.generate_content(
                ["Describe this image concisely:", img]
            )
            return response.text
        except Exception as e:
            logger.error(f"Image summary error: {e}")
            return None

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using sentence transformer.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            Optional[List[float]]: Generated embedding or None if failed
        """
        try:
            instruction = "Represent the summary for retrieval: "
            return self.embedding_model.encode([instruction + text])[0].tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    def _store_record(self, table: str, data: Dict[str, Any]) -> bool:
        """
        Store record in Supabase.
        
        Args:
            table (str): Table name
            data (Dict[str, Any]): Record data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.supabase_client.table(table).insert(data).execute()
            return True
        except Exception as e:
            logger.error(f"Supabase insert error: {e}")
            return False 
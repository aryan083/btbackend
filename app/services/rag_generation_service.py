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
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from run import custom_logger
from app.utils.supabase_utils import bulk_insert, bulk_update, bulk_upsert
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
    @custom_logger.log_function_call
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
        Load the embedding model, with fallback mechanisms:
        1. Try to load from local cache directory first
        2. If not found locally, download from internet and save locally
        3. If download fails, attempt to use a previously downloaded version
        
        This property initializes the SentenceTransformer model for generating embeddings
        only when needed, avoiding unnecessary memory usage until required.
        
        Returns:
            SentenceTransformer: The initialized embedding model ready for generating text embeddings
            
        Raises:
            RuntimeError: If there's an issue loading the model from any source
        """
        if self._embedding_model is None:
            # Define model name and local cache directory
            model_name = 'all-mpnet-base-v2'
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, model_name)
            
            # Try loading from local cache first
            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading embedding model from local cache: {model_path}")
                    self._embedding_model = SentenceTransformer(model_path, device='cpu')
                    
                    # Verify model is loaded correctly
                    test_embedding = self._embedding_model.encode("test")
                    if test_embedding is not None and len(test_embedding) > 0:
                        logger.info("Successfully loaded embedding model from local cache")
                        return self._embedding_model
                    else:
                        logger.warning("Local model returned empty embeddings, will try downloading fresh copy")
                except Exception as e:
                    logger.warning(f"Error loading model from local cache: {str(e)}. Will try downloading.")
            
            # If local loading failed or model doesn't exist locally, try downloading
            try:
                logger.info("Downloading embedding model from internet")
                # Download with cache_folder parameter to save locally
                self._embedding_model = SentenceTransformer(model_name, device='cpu', cache_folder=cache_dir)
                
                # Verify model is loaded correctly
                test_embedding = self._embedding_model.encode("test")
                if test_embedding is None or len(test_embedding) == 0:
                    raise ValueError("Model returned empty embeddings during validation")
                    
                logger.info(f"Successfully downloaded and cached embedding model to {model_path}")
            except Exception as e:
                logger.error(f"Error downloading embedding model: {str(e)}")
                # Last resort: try to find any previously downloaded model files
                try:
                    # Look for any partial downloads or previous versions
                    if os.path.exists(cache_dir):
                        model_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
                        if model_dirs:
                            fallback_path = os.path.join(cache_dir, model_dirs[0])
                            logger.warning(f"Attempting to load fallback model from: {fallback_path}")
                            self._embedding_model = SentenceTransformer(fallback_path, device='cpu')
                            
                            # Verify fallback model
                            test_embedding = self._embedding_model.encode("test")
                            if test_embedding is not None and len(test_embedding) > 0:
                                logger.info("Successfully loaded fallback embedding model")
                                return self._embedding_model
                except Exception as fallback_error:
                    logger.error(f"Fallback loading also failed: {str(fallback_error)}")
                
                # If we get here, all attempts have failed
                raise RuntimeError(f"Failed to initialize embedding model: {str(e)}") from e
                
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


    @custom_logger.log_function_call
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

    @custom_logger.log_function_call
    def generate_html_content(
        self,
        document_dir: str,
        course_id: str,
        output_dir: str,
        match_threshold: float = 0.3,
        max_results: int = 5,
        user_prompt: str = "",
        teaching_pattern: Union[str, List[str]] = "",
        skill_level: str = "",
        topic_metadata: Dict[str, str] = None,
        user_id: str = None,
        allow_images: bool = False  
    ) -> Dict[str, Any]:
        """
        Generate HTML content for all subtopics using RAG.
        
        Args:
            document_dir (str): Directory containing course documents
            course_id (str): ID of the course
            output_dir (str): Directory to save generated HTML files
            match_threshold (float): Threshold for vector similarity matching
            max_results (int): Maximum number of results to retrieve
            user_prompt (str): User's specific requirements for content
            teaching_pattern (Union[str, List[str]]): Preferred teaching pattern (e.g., case studies, stories)
            skill_level (str): User's skill level (1=beginner, 2=intermediate, 3=advanced)
            topic_metadata (Dict[str, str]): Mapping of topic names to their IDs
            user_id (str): ID of the user requesting the content
            allow_images (bool): Whether to allow images in the content
            
        Returns:
            Dict[str, Any]: Dictionary containing generated files and their metadata
        """
        try:
            course_structure = self.get_course_structure(document_dir)
            
            if not course_structure.get("Chapters"):
                logger.error("Invalid course structure format")
                return {"files": {}, "articles": [], "errors": ["Invalid course structure format"]}

            os.makedirs(output_dir, exist_ok=True)
            generated_files = {}
            generated_articles = []
            errors = []
            
            # Collect all articles for bulk insert
            articles_to_insert = []
            topic_updates = []

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

                        # Generate content with user preferences
                        content = self._generate_article_content(
                            subtopic_name=subtopic_name,
                            subtopic_code=subtopic_code,
                            course_id=course_id,
                            match_threshold=match_threshold,
                            max_results=max_results,
                            skill_level=skill_level,
                            user_prompt=user_prompt,
                            teaching_pattern=teaching_pattern,
                            allow_images=allow_images  # Pass new parameter
                        )
                        
                        # Save content to file
                        filename = f"{safe_subtopic}.html"
                        filepath = os.path.normpath(os.path.join(chapter_dir, filename))
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        # Find topic_id from metadata
                        topic_id = topic_metadata.get(subtopic_name) if topic_metadata else None
                        if not topic_id:
                            error_msg = f"No topic ID found for subtopic: {subtopic_name}"
                            errors.append(error_msg)
                            logger.error(error_msg)
                            continue
                        
                        # Create article record for bulk insert
                        article_data = {
                            "article_name": f"{subtopic_name} - {subtopic_code}",
                            "tags": {
                                "course_id": course_id,
                                "chapter_name": chapter_name,
                                "subtopic_code": subtopic_code,
                                "skill_level": skill_level,
                                "teaching_pattern": teaching_pattern
                            },
                            "content_text": content,
                            "topic_id": topic_id,
                            "is_completed": False,
                            "user_id": user_id if user_id else None
                        }
                        
                        articles_to_insert.append(article_data)
                        generated_files[filepath] = content
                        
                        # Prepare topic update
                        topic_updates.append({
                            "topic_id": topic_id,
                            "articles_json": {"article_ids": []}  # Will be populated after insert
                        })

                    except Exception as subtopic_error:
                        error_msg = f"Failed {subtopic_code}: {str(subtopic_error)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
            
            # Perform bulk insert for all articles
            if articles_to_insert:
                bulk_result = self._save_articles_bulk(articles_to_insert, topic_updates)
                
                # Process results
                if bulk_result["articles"]["success_count"] > 0:
                    # Map article IDs to topics
                    article_ids = bulk_result["articles"]["article_ids"]
                    for i, article_id in enumerate(article_ids):
                        if i < len(topic_updates):
                            topic_id = topic_updates[i]["topic_id"]
                            generated_articles.append({
                                "article_id": article_id,
                                "article_name": articles_to_insert[i]["article_name"],
                                "filepath": list(generated_files.keys())[i] if i < len(generated_files) else None,
                                "topic_id": topic_id
                            })
                
                # Add any errors from bulk operation
                errors.extend(bulk_result["articles"]["errors"])
                errors.extend(bulk_result["topics"]["errors"])

            return {
                "files": generated_files,
                "articles": generated_articles,
                "errors": errors
            }

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg)
            return {
                "files": {},
                "articles": [],
                "errors": [error_msg]
            }

    @custom_logger.log_function_call
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

            # Log the response data
            logger.info(f"Vector search response: {response.data}")


            if response.data==None or response.data==[] or response.data=={} or response.data==['']:
                response = self.supabase_client.table('material_text').select('*').match('course_id', course_id).execute()
                # logger.info(f"Material text response: {response.data}")
                logger.info("FIND DATATATATATATATTATATAATATTAT",response.data)
                return response.data
            if response.data:
                return "\n".join(
                    f"<source>{r.get('content', '')}</source>"                    
                    for r in response.data
                )
            return "\n".join(
                    f"<source>{r.get('content', '')}</source>"                    
                    for r in response.data
                )
        except Exception as e:
            logger.warning(f"Vector search failed: {str(e)}")
            return ""

    @custom_logger.log_function_call
    def _generate_article_content(
        self, 
        subtopic_name: str,
        subtopic_code: str,
        course_id: str,
        match_threshold: float = 0.3,
        max_results: int = 5,
        skill_level: str = "",
        user_prompt: str = "",
        teaching_pattern: Union[str, List[str]] = "",
        allow_images: bool = False
    ) -> str:
        """
        Generate HTML content for a single article.
        
        Args:
            subtopic_name (str): Name of the subtopic
            subtopic_code (str): Code of the subtopic
            course_id (str): ID of the course
            match_threshold (float): Threshold for vector similarity matching
            max_results (int): Maximum number of results to retrieve
            skill_level (str): User's skill level (1=beginner, 2=intermediate, 3=advanced)
            user_prompt (str): User's specific requirements for content
            teaching_pattern (Union[str, List[str]]): Preferred teaching pattern (e.g., case studies, stories)
            allow_images (bool): Whether to allow images in the content
            
        Returns:
            str: Generated HTML content
        """
        try:
            # Generate embedding and retrieve context
            query_text = f"Comprehensive technical explanation of {subtopic_name}"
            query_embedding = self.embedding_model.encode(query_text).tolist()
            context = self._retrieve_context(
                query_embedding=query_embedding,
                course_id=course_id,
                match_threshold=match_threshold,
                max_results=max_results
            )

            # Build unified prompt
            prompt = self._build_generation_prompt(
                subtopic_name=subtopic_name,
                subtopic_code=subtopic_code,
                context=context,
                skill_level=skill_level,
                user_prompt=user_prompt,
                teaching_pattern=teaching_pattern,
                allow_images=allow_images
            )

            response = self.gemini_model.generate_content(prompt)
            return self._clean_html_content(response.text)
            
        except Exception as e:
            logger.error(f"Failed to generate content: {str(e)}")
            return f"<p>Error generating content: {str(e)}</p>"
    @custom_logger.log_function_call
    def process_content(self, base_path: str, course_id: str) -> Dict[str, Any]:
        """
        Process both text and image content for a course and generate HTML articles.
        
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
                "articles_generated": 0,
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
            
            # Generate and store HTML articles
            try:
                # Create articles directory if it doesn't exist
                articles_dir = Path(base_path) / "Articles"
                articles_dir.mkdir(exist_ok=True)
                
                # Generate articles from processed content
                article_results = self._generate_articles(
                    base_path=base_path,
                    course_id=course_id,
                    output_dir=str(articles_dir)
                )
                
                results["articles_generated"] = article_results["generated"]
                results["errors"].extend(article_results["errors"])
                
            except Exception as e:
                logger.error(f"Article generation failed: {str(e)}")
                results["errors"].append(f"Article generation failed: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Content processing failed: {str(e)}")
            return {
                "text_processed": 0,
                "image_processed": 0,
                "articles_generated": 0,
                "errors": [str(e)]
            }

    @custom_logger.log_function_call
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

        # Collect all records for bulk insert
        records_to_insert = []
        
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
                    
                    # Add to batch for bulk insert
                    records_to_insert.append(record)
                        
            except Exception as e:
                results["errors"].append(f"Error processing text file {text_file}: {str(e)}")
        
        # Perform bulk insert if we have records
        if records_to_insert:
            bulk_result = self._store_records_bulk('material_text', records_to_insert)
            results["processed"] = bulk_result["success_count"]
            results["errors"].extend(bulk_result["errors"])
                
        return results

    @custom_logger.log_function_call
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

        # Collect all records for bulk insert
        records_to_insert = []

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
                
                # Add to batch for bulk insert
                records_to_insert.append(record)
                    
            except Exception as e:
                results["errors"].append(f"Error processing image {image_file}: {str(e)}")
                
        # Perform bulk insert if we have records
        if records_to_insert:
            bulk_result = self._store_records_bulk('material_image', records_to_insert)
            results["processed"] = bulk_result["success_count"]
            results["errors"].extend(bulk_result["errors"])
                
        return results

    @custom_logger.log_function_call
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

    @custom_logger.log_function_call
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

    @custom_logger.log_function_call
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

    @custom_logger.log_function_call
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
            # Use the bulk insert function with a single record
            result = bulk_insert(self.supabase_client, table, [data])
            return result["success_count"] > 0
        except Exception as e:
            logger.error(f"Supabase insert error: {e}")
            return False
            
    @custom_logger.log_function_call
    def _store_records_bulk(self, table: str, records: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
        """
        Store multiple records in Supabase in bulk.
        
        Args:
            table (str): Table name
            records (List[Dict[str, Any]]): List of records to store
            batch_size (int): Number of records to insert in each batch
            
        Returns:
            Dict[str, Any]: Result of the operation with success count and errors
        """
        try:
            return bulk_insert(self.supabase_client, table, records, batch_size)
        except Exception as e:
            logger.error(f"Supabase bulk insert error: {e}")
            return {"success_count": 0, "errors": [str(e)]}

    @custom_logger.log_function_call
    def _generate_articles(
        self,
        base_path: str,
        course_id: str,
        output_dir: str
    ) -> Dict[str, Any]:        
        """
        Generate HTML articles from processed content.
        
        Args:
            base_path (str): Base path containing processed content
            course_id (str): ID of the course
            output_dir (str): Directory to store generated articles
            
        Returns:
            Dict[str, Any]: Generation results and statistics
        """
        try:
            results = {
                "generated": 0,
                "errors": []
            }
            
            # Load course structure
            course_structure = self.get_course_structure(base_path)
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process each chapter
            for chapter_name, chapter_data in course_structure.get("Chapters", {}).items():
                try:
                    # Create chapter directory
                    chapter_dir = output_path / self.sanitize_path(chapter_name)
                    chapter_dir.mkdir(exist_ok=True)
                    
                    # Process each subtopic
                    for subtopic_code, subtopic_name in chapter_data.items():
                        try:
                            # Generate article content using the unified method
                            article_content = self._generate_article_content(
                                subtopic_name=subtopic_name,
                                subtopic_code=subtopic_code,
                                course_id=course_id,
                                allow_images=False  # Enable images for batch
                            )
                            
                            # Save article
                            article_filename = f"{self.sanitize_path(subtopic_code)}.html"
                            article_path = chapter_dir / article_filename
                            
                            with open(article_path, 'w', encoding='utf-8') as f:
                                f.write(article_content)
                            
                            results["generated"] += 1
                            logger.info(f"Generated article: {article_path}")
                            
                        except Exception as e:
                            error_msg = f"Failed to generate article for {subtopic_code}: {str(e)}"
                            logger.error(error_msg)
                            results["errors"].append(error_msg)
                            
                except Exception as e:
                    error_msg = f"Failed to process chapter {chapter_name}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            return results
            
        except Exception as e:
            logger.error(f"Article generation failed: {str(e)}")
            return {
                "generated": 0,
                "errors": [str(e)]
            }

    @custom_logger.log_function_call
    def _save_article_to_supabase(self, 
                                 course_id: str,
                                 chapter_name: str,
                                 subtopic_code: str,
                                 subtopic_name: str,
                                 content: str,
                                 filepath: str,
                                 user_id: str = "",
                                 skill_level: str = "",
                                 teaching_pattern: Union[str, List[str]] = "",
                                 topic_metadata: Dict[str, str] = {}) -> Optional[str]:
        """
        Save generated article to Supabase and update topic relationship.
        
        Args:
            course_id (str): ID of the course
            chapter_name (str): Name of the chapter
            subtopic_code (str): Code of the subtopic
            subtopic_name (str): Name of the subtopic
            content (str): Generated HTML content
            filepath (str): Path to the saved file
            user_id (str): ID of the user requesting the content
            skill_level (str): User's skill level
            teaching_pattern (Union[str, List[str]]): Preferred teaching pattern
            topic_metadata (Dict[str, str]): Mapping of topic names to their IDs
            
        Returns:
            Optional[str]: Article ID if successful, None otherwise
        """
        try:
            # Find topic_id from metadata
            topic_id = topic_metadata.get(subtopic_name)
            if not topic_id:
                logger.error(f"No topic ID found for subtopic: {subtopic_name}")
                return None

            # Clean the content before saving
            cleaned_content = self._clean_html_content(content)

            # Create article record
            article_data = {
                "article_name": f"{subtopic_name} - {subtopic_code}",
                "tags": {
                    "course_id": course_id,
                    "chapter_name": chapter_name,
                    "subtopic_code": subtopic_code,
                    "skill_level": skill_level,
                    "teaching_pattern": teaching_pattern
                },
                "content_text": cleaned_content,
                "topic_id": topic_id,
                "is_completed": False,
                "user_id": user_id if user_id else None
            }
            
            # Use bulk insert with a single record
            result = bulk_insert(self.supabase_client, "articles", [article_data])
            
            if not result["success_count"]:
                logger.error("Failed to save article: No data returned")
                return None
                
            article_id = result["data"][0].get('article_id') if result.get("data") else None
            if not article_id:
                logger.error("No article ID returned from insert")
                return None
                
            # Update topic's articles_json
            # self._update_topic_articles(topic_id, article_id)
            
            logger.info(f"Article saved successfully with ID: {article_id}")
            return article_id
                
        except Exception as e:
            logger.error(f"Error saving article to Supabase: {str(e)}")
            return None
            
    def _save_articles_bulk(self, 
                           articles_data: List[Dict[str, Any]], 
                           topic_updates: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save multiple articles to Supabase in bulk and optionally update topics.
        
        Args:
            articles_data (List[Dict[str, Any]]): List of article data to save
            topic_updates (List[Dict[str, Any]], optional): List of topic updates to perform
            
        Returns:
            Dict[str, Any]: Result of the operation with success count and errors
        """
        result = {
            "articles": {"success_count": 0, "errors": [], "article_ids": []},
            "topics": {"success_count": 0, "errors": []}
        }
        
        try:
            # Clean content for all articles
            for article in articles_data:
                if "content_text" in article:
                    article["content_text"] = self._clean_html_content(article["content_text"])
            
            # Bulk insert articles
            articles_result = bulk_insert(self.supabase_client, "articles", articles_data)
            result["articles"]["success_count"] = articles_result["success_count"]
            result["articles"]["errors"] = articles_result["errors"]
            
            # Extract article IDs for topic updates
            if articles_result.get("data"):
                result["articles"]["article_ids"] = [
                    article.get("article_id") for article in articles_result["data"] 
                    if article.get("article_id")
                ]
            
            # Update topics if needed
            if topic_updates and result["articles"]["article_ids"]:
                topics_result = bulk_update(
                    self.supabase_client, 
                    "topics", 
                    topic_updates,
                    id_field="topic_id"
                )
                result["topics"]["success_count"] = topics_result["success_count"]
                result["topics"]["errors"] = topics_result["errors"]
                
            return result
            
        except Exception as e:
            logger.error(f"Error in bulk article save: {str(e)}")
            result["articles"]["errors"].append(str(e))
            return result
        

    @custom_logger.log_function_call
    def _clean_html_content(self, content: str) -> str:
        """
        Clean and format HTML content.
        
        Args:
            content (str): Raw HTML content
            
        Returns:
            str: Cleaned HTML content
        """
        # Remove any backticks that might be in the content
        content = re.sub(r'```html?\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        
        # Remove any extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Ensure proper HTML formatting
        content = content.strip()
        
        return content 
    

    def _get_skill_description(self, level: str) -> str:
        """Convert skill level to descriptive text."""
        return {
            "1": "beginner-friendly with basic concepts and simple explanations",
            "2": "intermediate level with balanced theory and practical applications",
            "3": "advanced level with complex concepts and in-depth technical details",
            "": ""  # Default empty
        }.get(level, "")

    @custom_logger.log_function_call
    def _build_teaching_instructions(self, pattern: Union[str, List[str]]) -> str:
        """Convert teaching pattern to prompt instructions."""
        instructions = []
        if isinstance(pattern, list):
            pattern_str = ', '.join(pattern)
        else:
            pattern_str = str(pattern)

        patterns = {
            "case studies": "- Include multiple real-world case studies\n",
            "stories": "- Use storytelling approach to explain concepts\n",
            "examples": "- Provide numerous practical examples\n",
            "visual": "- Focus on visual explanations and diagrams\n",
            "hands-on": "- Include hands-on exercises and tutorials\n"
        }

        for key, instruction in patterns.items():
            if key in pattern_str.lower():
                instructions.append(instruction)

        return "".join(instructions) if instructions else \
            "- Multiple practical examples\n- Real-world implementation scenarios\n"
            
            
    def _build_generation_prompt(
        self,
        subtopic_name: str,
        subtopic_code: str,
        context: str,
        skill_level: str = "",
        user_prompt: str = "",
        teaching_pattern: Union[str, List[str]] = "",
        allow_images: bool = False
    ) -> str:
        """Build generation prompt with conditional sections.
        Args:
            subtopic_name (str): Name of the subtopic
            subtopic_code (str): Code of the subtopic
            context (str): Context of the subtopic
            skill_level (str): Skill level of the user
            user_prompt (str): User prompt
            teaching_pattern (Union[str, List[str]]): Teaching pattern
            allow_images (bool): Whether to allow images in the content
            
        Returns:
            str: Generation prompt
        """
         # Skill level description
        skill_desc = self._get_skill_description(skill_level)
        skill_section = f"**Target Audience**: {skill_desc}." if skill_desc else ""

        # Teaching instructions
        teaching_instructions = self._build_teaching_instructions(teaching_pattern)

        # Image policy
        image_policy = ("- STRICTLY NO IMAGES ALLOWED" if not allow_images 
                        else "- Include relevant <figure> elements with proper <figcaption>")
# - Minimum 3 code samples in <pre><code> blocks
        logger.info(f"Generating prompt for {subtopic_name} ({subtopic_code}  )")
        logger.info(f"Context: {context}")
        
        if context == "":
            return f"""Generate exhaustive professional technical documentation about {subtopic_name} ({subtopic_code}).
            Use 100% of available token capacity for maximum depth and quality. {skill_section}

            **Mandatory Structure:**
            1. <h1>Single Title</h1> (Only one H1 heading at beginning)
            2. Table of Contents (Linked to section IDs)
            3. Comprehensive Definition Section with Etymology
            4. Technical Deep Dive with Mathematical Notation (if applicable)
            5. Implementation Examples:
            
            - Error handling examples
            6. Comparative Analysis Table:
            <table class="comparative">
                <thead><tr><th>Feature</th><th>Implementation A</th><th>Implementation B</th></tr></thead>
                <tbody>...</tbody>
            </table>
            7. Best Practices & Anti-Patterns
            8. Security Considerations
            9. Performance Characteristics
            10. Cross-References to Related Concepts

            **Context Materials:**
            {context}

            **Quality Requirements:**
            - PhD-level technical depth prioritized
            - All claims must be backed by context or examples
            - No filler content - maximize information density
            - Strict technical accuracy over readability
            - Include industry-specific terminology
            - No token preservation - use full capacity

            **HTML Requirements:**
            - Only <body> content allowed (no head/styles)
            - Semantic HTML5 elements required (article, section, etc.)
            - Tables must use proper <thead>/<tbody> structure
            - Code samples require syntax highlighting hints and must have word wrapping to fit display
            - External links open in new tab
            - All sections must have ID attributes
            - No markdown - only pure HTML
            - Error handling examples in red bordered divs

            **Special Instructions:**
            If the gethered content from the context is not relevant to the subtopic, then just return "No relevant content found", do not generate any content.
            strictly follow this instruction at any cost.
            {teaching_instructions}
            {user_prompt}

            **Prohibitions:**
            - No "Note:" or "Tip:" boxes
            - No colloquial language
            - No placeholder comments
            - No unfinished sections
            - No markdown formatting
            - No duplicated content

            Output MUST use 100% of available tokens while maintaining technical precision."""
            
        else:
            return "No relevant content found I'm sorry, I couldn't find any reliable information to answer your query."

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
        Lazy load the embedding model from local directory.
        If the model doesn't exist or is incomplete, it will be downloaded.
        
        Returns:
            The initialized embedding model
        """
        if self._embedding_model is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'all-mpnet-base-v2')
            
            try:
                # Try to load the model if it exists
                if os.path.exists(model_path):
                    try:
                        self._embedding_model = SentenceTransformer(model_path)
                        # Verify model is loaded correctly by encoding a test string
                        self._embedding_model.encode("test")
                        logger.info(f"Successfully loaded model from {model_path}")
                        return self._embedding_model
                    except Exception as e:
                        logger.warning(f"Failed to load existing model: {str(e)}")
                        # If loading fails, delete the incomplete/corrupted model directory
                        import shutil
                        shutil.rmtree(model_path, ignore_errors=True)
                        logger.info(f"Removed incomplete model directory: {model_path}")
                
                # Download fresh copy of the model
                logger.info(f"Downloading model to {model_path}")
                self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
                
                # Verify the model works before saving
                self._embedding_model.encode("test")
                
                # Save the verified model
                self._embedding_model.save(model_path)
                logger.info(f"Model downloaded and saved to {model_path}")
                
            except Exception as e:
                logger.error(f"Error initializing embedding model: {str(e)}")
                raise
                
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
        match_threshold: float = 0.7,
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
                        
                        # Save article to Supabase
                        article_id = self._save_article_to_supabase(
                            course_id=course_id,
                            chapter_name=chapter_name,
                            subtopic_code=subtopic_code,
                            subtopic_name=subtopic_name,
                            content=content,
                            filepath=filepath,
                            user_id=user_id,
                            skill_level=skill_level,
                            teaching_pattern=teaching_pattern,
                            topic_metadata=topic_metadata
                        )
                        
                        if article_id:
                            generated_articles.append({
                                "article_id": article_id,
                                "article_name": f"{subtopic_name} - {subtopic_code}",
                                "filepath": filepath,
                                "topic_id": topic_metadata.get(subtopic_name) if topic_metadata else None
                            })
                            generated_files[filepath] = content
                            logger.info(f"Created and saved article: {filepath}")
                        else:
                            error_msg = f"Failed to save article for {subtopic_code}"
                            errors.append(error_msg)
                            logger.error(error_msg)

                    except Exception as subtopic_error:
                        error_msg = f"Failed {subtopic_code}: {str(subtopic_error)}"
                        errors.append(error_msg)
                        logger.error(error_msg)

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

            if response.data==None:
                return "No relevant content found I'm sorry, I couldn't find any reliable information to answer your query."
            if response.data:
                return "\n".join(
                    f"<source>{r.get('content', '')}</source>" 
                    for r in response.data[:3]
                )
            return ""
        except Exception as e:
            logger.warning(f"Vector search failed: {str(e)}")
            return ""

    @custom_logger.log_function_call
    def _generate_article_content(
        self, 
        subtopic_name: str,
        subtopic_code: str,
        course_id: str,
        match_threshold: float = 0.7,
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
            self.supabase_client.table(table).insert(data).execute()
            return True
        except Exception as e:
            logger.error(f"Supabase insert error: {e}")
            return False

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
            
            # Insert article into Supabase
            response = self.supabase_client.table("articles").insert(article_data).execute()
            
            if not response.data:
                logger.error("Failed to save article: No data returned")
                return None
                
            article_id = response.data[0].get('article_id')
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
            
    @custom_logger.log_function_call
    def _update_topic_articles(self, topic_id: str, article_id: str) -> bool:
        """
        Update topic's articles_json field with new article ID using atomic update.
        
        Args:
            topic_id (str): ID of the topic
            article_id (str): ID of the article to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use PostgreSQL's jsonb_array_append function
            update_query = """
            UPDATE topics 
            SET articles_json = 
                COALESCE(articles_json, '[]'::jsonb) || jsonb_build_array(%(article_id)s)
            WHERE topic_id = %(topic_id)s
            """
            
            # Execute raw SQL
            response = self.supabase_client.rpc('execute', {
                "query": update_query,
                "params": {
                    "article_id": article_id,
                    "topic_id": topic_id
                }
            }).execute()
            
            # Check affected rows
            if response.data and response.data[0]['affected_rows'] > 0:
                logger.info(f"Updated topic {topic_id} with article {article_id}")
                return True
            
            logger.error(f"No rows affected for topic {topic_id}")
            return False
            
        except Exception as e:
            logger.error(f"Topic update error: {str(e)}")
            return False

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
        
        if context =="":
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
            {image_policy}
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

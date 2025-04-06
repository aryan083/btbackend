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
        Lazy load the embedding model from local directory.
        
        Returns:
            The initialized embedding model
        """
        if self._embedding_model is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'all-mpnet-base-v2')
            self._embedding_model = SentenceTransformer(model_path)
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
                            max_results: int = 5,
                            user_prompt: str = "",
                            teaching_pattern: Union[str, List[str]] = "",
                            skill_level: str = "",
                            topic_metadata: Dict[str, str] = {},
                            user_id: str = "",                            
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

                        # Generate embedding and retrieve context
                        query_text = f"Comprehensive technical explanation of {subtopic_name}"
                        query_embedding = self.embedding_model.encode(query_text).tolist()

                        context = self._retrieve_context(
                            query_embedding, 
                            course_id, 
                            match_threshold, 
                            max_results
                        )

                        # Generate content with user preferences
                        content = self._generate_content(
                            subtopic_name, 
                            subtopic_code, 
                            context,
                            skill_level,
                            user_prompt,
                            teaching_pattern
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
                                "topic_id": topic_metadata.get(subtopic_name)
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

    def _generate_content(self, subtopic_name: str, subtopic_code: str, context: str, 
                         skill_level: str = "", user_prompt: str = "", 
                         teaching_pattern: Union[str, List[str]] = "") -> str:
        """
        Generate HTML content using Gemini model.
        
        Args:
            subtopic_name (str): Name of the subtopic
            subtopic_code (str): Code of the subtopic
            context (str): Retrieved context
            skill_level (str): User's skill level (1=beginner, 2=intermediate, 3=advanced)
            user_prompt (str): User's specific requirements for content
            teaching_pattern (Union[str, List[str]]): Preferred teaching pattern (e.g., case studies, stories)
            
        Returns:
            str: Generated HTML content
        """
        # Determine skill level description
        skill_level_desc = ""
        if skill_level == "1":
            skill_level_desc = "beginner-friendly with basic concepts and simple explanations"
        elif skill_level == "2":
            skill_level_desc = "intermediate level with balanced theory and practical applications"
        elif skill_level == "3":
            skill_level_desc = "advanced level with complex concepts and in-depth technical details"
        
        # Build teaching pattern instructions
        teaching_instructions = ""
        if teaching_pattern:
            # Handle both string and list types for teaching_pattern
            if isinstance(teaching_pattern, list):
                teaching_pattern_str = ', '.join(teaching_pattern)
            else:
                teaching_pattern_str = teaching_pattern
                
            if "case studies" in teaching_pattern_str.lower():
                teaching_instructions += "- Include multiple real-world case studies\n"
            if "stories" in teaching_pattern_str.lower():
                teaching_instructions += "- Use storytelling approach to explain concepts\n"
            if "examples" in teaching_pattern_str.lower():
                teaching_instructions += "- Provide numerous practical examples\n"
            if "visual" in teaching_pattern_str.lower():
                teaching_instructions += "- Focus on visual explanations and diagrams\n"
            if "hands-on" in teaching_pattern_str.lower():
                teaching_instructions += "- Include hands-on exercises and tutorials\n"
        
        # If no specific teaching patterns were matched, use default
        if not teaching_instructions:
            teaching_instructions = "- Multiple practical examples\n- Real-world implementation scenarios\n"
        
        prompt = f"""Generate exhaustive HTML body content about {subtopic_name} ({subtopic_code}).
        Use ALL available token capacity for depth and detail. The content should be {skill_level_desc}.
        
        Include:
        
        - Comprehensive definition section
        - Detailed technical breakdown of components
        {teaching_instructions}
        - Common pitfalls and troubleshooting
        - Advanced usage patterns
        - Cross-references to related concepts
        
        {user_prompt}
        
        Context materials:
        {context}
        
        Requirements:
        - Strictly HTML body content only
        - No CSS/JS/head/meta tags
        - Semantic HTML5 elements
        - Technical depth over brevity
        - Tables for comparative analysis
        - Code samples in <pre><code>
        - DO NOT include any images in the generated content
        - Use the full token capacity for maximum detail and comprehensiveness
        """

        response = self.gemini_model.generate_content(prompt)
        return response.text

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

    def _generate_articles(self, base_path: str, course_id: str, output_dir: str) -> Dict[str, Any]:
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
                            # Generate article content
                            article_content = self._generate_article_content(
                                subtopic_name=subtopic_name,
                                subtopic_code=subtopic_code,
                                course_id=course_id
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

    def _generate_article_content(self, subtopic_name: str, subtopic_code: str, course_id: str) -> str:
        """
        Generate HTML content for a single article.
        
        Args:
            subtopic_name (str): Name of the subtopic
            subtopic_code (str): Code of the subtopic
            course_id (str): ID of the course
            
        Returns:
            str: Generated HTML content
        """
        try:
            # Generate embedding for the subtopic
            query_text = f"Comprehensive technical explanation of {subtopic_name}"
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            # Retrieve relevant context
            context = self._retrieve_context(
                query_embedding=query_embedding,
                course_id=course_id,
                match_threshold=0.7,
                max_results=5
            )
            
            # Generate content using Gemini
            prompt = f"""Generate a comprehensive HTML article about {subtopic_name} ({subtopic_code}).
            Include:
            
            - Detailed explanation
            - Technical concepts
            - Examples
            - Implementation details
            - Best practices
            
            Context materials:
            {context}
            
            Requirements:
            - Use semantic HTML5
            - Include code samples in <pre><code>
            - Use tables for comparisons
            - Include relevant images
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Failed to generate article content for {subtopic_code}: {str(e)}")
            raise 

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
                "content_text": content,
                "topic_id": topic_id,
                "is_completed": True,
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
            self._update_topic_articles(topic_id, article_id)
            
            logger.info(f"Article saved successfully with ID: {article_id}")
            return article_id
                
        except Exception as e:
            logger.error(f"Error saving article to Supabase: {str(e)}")
            return None
            
    def _update_topic_articles(self, topic_id: str, article_id: str) -> bool:
        """
        Update topic's articles_json field with new article ID.
        
        Args:
            topic_id (str): ID of the topic
            article_id (str): ID of the article to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First get current articles_json
            response = self.supabase_client.table("topics").select("articles_json").eq("topic_id", topic_id).execute()
            
            if not response.data:
                logger.error(f"Topic not found: {topic_id}")
                return False
                
            current_articles = response.data[0].get('articles_json', []) or []
            if not isinstance(current_articles, list):
                current_articles = []
                
            # Add new article ID if not already present
            if article_id not in current_articles:
                current_articles.append(article_id)
                
            # Update topic
            update_response = self.supabase_client.table("topics").update(
                {"articles_json": current_articles}
            ).eq("topic_id", topic_id).execute()
            
            if update_response.data:
                logger.info(f"Updated topic {topic_id} with article {article_id}")
                return True
            else:
                logger.error(f"Failed to update topic {topic_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating topic articles: {str(e)}")
            return False 
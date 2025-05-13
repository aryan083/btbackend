"""
Course Service Module
Handles business logic for course-related operations
"""
from typing import Dict
import logging
from datetime import datetime
from app.models.course import Course
import google.generativeai as genai
from ..config import GEMINI_API_KEY
import fitz  # PyMuPDF
import json
from run import custom_logger
logger = logging.getLogger(__name__)

class CourseService:
    """
    Service class for handling course-related operations
    """
    def __init__(self):
        """
        Initialize the service with a Gemini model instance
        """
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def create_course(
        self,
        course_name: str,
        tags: Dict,
        metadata: str,
        chapters_json: Dict,
        skill_level: int,
        teaching_pattern: Dict,
        user_prompt: str,
        progress: float = 0.0
    ) -> Dict:
        """
        Create a new course
        @param course_name: Name of the course
        @param tags: JSON object containing course tags
        @param metadata: Additional metadata string
        @param chapters_json: JSON object containing chapter information
        @param skill_level: Integer representing the skill level
        @param teaching_pattern: JSON object containing teaching methods
        @param user_prompt: User's input prompt
        @param progress: Course progress (default: 0.0)
        @returns: Dictionary containing the created course data
        """
        try:
            # Create course object
            course = Course(
                course_name=course_name,
                tags=tags,
                metadata=metadata,
                chapters_json=chapters_json,
                skill_level=skill_level,
                teaching_pattern=teaching_pattern,
                user_prompt=user_prompt,
                progress=progress,
                created_at=datetime.utcnow()
            )
            
            # Save to database (implement database logic here)
            # For now, we'll just return the course data
            return {
                'course_id': course.course_id,
                'course_name': course.course_name,
                'tags': course.tags,
                'metadata': course.metadata,
                'chapters_json': course.chapters_json,
                'skill_level': course.skill_level,
                'teaching_pattern': course.teaching_pattern,
                'user_prompt': course.user_prompt,
                'progress': course.progress,
                'created_at': course.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in create_course: {str(e)}")
            raise

    @custom_logger.log_function_call
    def extract_text_from_pdf(self, pdf_file_stream):
        """
        Extract text content from a PDF file stream
        
        Args:
            pdf_file_stream: The raw PDF file data
            
        Returns:
            str: Extracted text from PDF
            
        Raises:
            Exception: If PDF extraction fails
        """
        pdf_document = fitz.open(stream=pdf_file_stream, filetype="pdf")
        
        if pdf_document is None:
            raise Exception("Unable to open PDF document")
            
        pdf_text = "\n".join([page.get_text() for page in pdf_document])
        pdf_document.close()
        
        if not pdf_text or len(pdf_text.strip()) < 10:
            raise Exception("Unable to extract text from PDF. The document might be empty or unreadable.")
            
        return pdf_text
    
    @custom_logger.log_function_call
    def clean_gemini_json_response(self, response_text):
        """
        Clean the JSON response from Gemini to ensure it's properly formatted
        
        Args:
            response_text: Raw text response from Gemini
            
        Returns:
            str: Cleaned JSON string
        """
        # Assuming this function exists elsewhere - implement based on your needs
        # This would typically strip markdown formatting, extract json blocks, etc.
        # Simple placeholder implementation:
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        return response_text.strip()
        
    @custom_logger.log_function_call
    def extract_course_content(self, pdf_text):
        """
        Extract structured course content from PDF text
        
        Args:
            pdf_text: Extracted text from PDF
            
        Returns:
            dict: JSON structure of course content
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        content_prompt = """Extract only the course content (topics and subtopics) from the given syllabus PDF. Ignore any extra details such as unit hours,Contact Hours, examination schemes, objectives, and references.  
                Format the extracted content in a structured JSON format as follows:  
                - Each chapter should be labeled as **"Chapter X: Chapter Name"**.  
                - Each chapter should contain a list of subtopics in a numbered sequence, such as **"X.Y: Subtopic Name"**.  
                - Maintain the hierarchical structure of topics and subtopics.  
                - The output should be clean and formatted as nested JSON.

                Example format:  
                ```json
                {
                "Chapters": {
                    "Chapter 1: Introduction": {
                    "1.1": "Subtopic Name",
                    "1.2": "Subtopic Name"
                    },
                    "Chapter 2: Next Chapter Name": {
                    "2.1": "Subtopic Name"
                    }
                }
                } NOTE: ONLY RETURN THE JSON NO OTHER TEXT IS NEEDED AND NOT WANTED 
                I REPEAT THAT WE NOT NO SUCH THINGS AS HERE'S THE BACKDOWN   """

        content_parts = [content_prompt, pdf_text]
        content_response = self.model.generate_content(content_parts)
        
        # Clean and parse the JSON response
        cleaned_content_text = self.clean_gemini_json_response(content_response.text)
        course_content = json.loads(cleaned_content_text)
        
        logger.info("Course content extracted successfully")
        logger.info(json.dumps(course_content, indent=4))
        
        return course_content
        
    @custom_logger.log_function_call
    def extract_keywords(self, pdf_text):
        """
        Extract keywords, skills and technologies from PDF text
        
        Args:
            pdf_text: Extracted text from PDF
            
        Returns:
            dict: JSON structure of keywords and related information
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        keyword_prompt = """Analyze this course syllabus and extract:
        1. Key technical terms and concepts that will be covered
        2. Main skills students will learn
        3. Core technologies or tools that will be taught
        4. The technical terms should be concise
        5. Technical terms array should not exceed size of 10
        6. Technical terms should be covering the course content completely

        Format the response as a JSON with the following structure:
        {
            "technical_terms": ["term1", "term2", ...],
            "skills": ["skill1", "skill2", ...],
            "technologies": ["tech1", "tech2", ...]
        }
        NOTE: ONLY RETURN THE JSON NO OTHER TEXT IS NEEDED"""

        keyword_parts = [keyword_prompt, pdf_text]
        keyword_response = self.model.generate_content(keyword_parts)
        
        # Clean and parse the JSON response
        cleaned_keyword_text = self.clean_gemini_json_response(keyword_response.text)
        keywords = json.loads(cleaned_keyword_text)
        
        return keywords
        
    @custom_logger.log_function_call
    def generate_welcome_message(self, course_content, keywords):
        """
        Generate a welcome message for the course
        
        Args:
            course_content: Structured course content
            keywords: Extracted keywords and skills
            
        Returns:
            str: Welcome message
        """
        welcome_prompt = f"""Create an engaging and professional welcome message for a course with the following details:
        Course Content: {json.dumps(course_content)}
        Key Learning Outcomes: {json.dumps(keywords)}

        The message should:
        1. Be warm and encouraging
        2. Highlight the value and relevance of the course
        3. Mention 2-3 key skills or technologies they'll learn
        4. Keep it concise (max 2 sentences)
        
        NOTE: Return only the welcome message, no additional text."""

        welcome_response = self.model.generate_content(welcome_prompt)
        welcome_message = welcome_response.text.strip()
        
        # Fallback if empty
        if not welcome_message:
            welcome_message = "Welcome to the course! We're excited to help you learn and grow."
            
        return welcome_message
        
    @custom_logger.log_function_call
    def generate_course_title(self, course_content, keywords):
        """
        Generate a title for the course
        
        Args:
            course_content: Structured course content
            keywords: Extracted keywords and skills
            
        Returns:
            str: Course title
        """
        title_prompt = f"""Create an engaging and professional course title based on the following course content and key learning outcomes:
        Course Content: {json.dumps(course_content)}
        Key Learning Outcomes: {json.dumps(keywords)}

        The title should:
        1. Be informative and engaging
        2. Highlight the value and relevance of the course
        3. Be brief max 30 letters
        
        NOTE: Return only the course title, no additional text."""

        title_response = self.model.generate_content(title_prompt)
        course_title = title_response.text.strip()
        
        # Fallback if empty
        if not course_title:
            course_title = "Course title."
            
        return course_title

    @custom_logger.log_function_call
    def save_course_json(self, course_json, output_dir: str):
        """
        Save course JSON to output directory
        
        Args:
            course_json: The JSON content to save
            output_dir: Directory path where to save the JSON file
            
        Returns:
            Path: Path to the saved JSON file
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_path = output_path / "course_structure.json"
        with open(json_path, 'w') as f:
            json.dump(course_json, f, indent=2)
        return json_path
        
    @custom_logger.log_function_call
    def process_pdf(self, pdf_file_stream, output_dir: str = None):
        """
        Main method to process PDF and return all required information
        
        Args:
            pdf_file_stream: The raw PDF file data
            output_dir: Optional directory path to save the course JSON
            
        Returns:
            dict: All course information including content, keywords, welcome message and title
            
        Raises:
            Exception: If any step fails
        """
        try:
            # Extract text from PDF
            pdf_text = self.extract_text_from_pdf(pdf_file_stream)
            
            # Extract course content
            course_content = self.extract_course_content(pdf_text)
            
            # Save course JSON if output directory is provided
            if output_dir:
                output_dir=self.save_course_json(course_content, output_dir)
            
            # Extract keywords and skills
            keywords = self.extract_keywords(pdf_text)
            
            # Generate welcome message
            welcome_message = self.generate_welcome_message(course_content, keywords)
            
            # Generate course title
            course_title = self.generate_course_title(course_content, keywords)
            
            return {
                'status': 'success',
                'course_content': course_content,
                'keywords': keywords,
                'welcome_message': welcome_message,
                'course_title': course_title,
                'course_path': str(output_dir) if output_dir else None
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            raise
    


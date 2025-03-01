"""
Course Service Module
Handles business logic for course-related operations
"""
from typing import Dict
import logging
from datetime import datetime
from app.models.course import Course

logger = logging.getLogger(__name__)

class CourseService:
    """
    Service class for handling course-related operations
    """
    
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

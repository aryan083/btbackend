# app/utils/course_utils.py

import logging
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from supabase import create_client, Client
from app.config import Config

logger = logging.getLogger(__name__)



class TopicData(TypedDict):
    topic_id: str
    topic_name: str

class ChapterData(TypedDict):
    chapter_id: str
    chapter_name: str
    topics: List[TopicData]
    created_at: datetime
    tags: Dict[str, Any]

class CourseDetailedData(TypedDict):
    skill_level: int
    teaching_pattern: List[str]
    user_prompt: str
    topic_name_to_id: Dict[str, str]

class CourseGenerationData(TypedDict):
    skill_level: int
    teaching_pattern: List[str]
    user_prompt: str
    topic_name_to_id: Dict[str, str]


def get_course_generation_data(user_id: str, course_id: str) -> Optional[CourseGenerationData]:
    """
    Get minimal course data needed for content generation with flat topic mapping.
    
    Returns:
        CourseGenerationData: {
            'skill_level': int,
            'teaching_pattern': List[str],
            'user_prompt': str,
            'topic_name_to_id': Dict[str, str]  # {topic_name: topic_id}
        }
    """
    try:
        supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        
        # Verify user access
        user_response = supabase.table('users')\
            .select('courses_json')\
            .eq('user_id', user_id)\
            .single()\
            .execute()
        
        if not user_response.data or course_id not in user_response.data.get('courses_json', []):
            logger.error(f"User {user_id} doesn't have access to course {course_id}")
            return None

        # Get course metadata
        course_response = supabase.table('courses')\
            .select('skill_level, teaching_pattern, user_prompt, chapters_json')\
            .eq('course_id', course_id)\
            .single()\
            .execute()
        
        if not course_response.data:
            logger.error(f"Course {course_id} not found")
            return None

        course_data = course_response.data
        chapters_json = course_data.get('chapters_json', {}).get('chapters', [])
        topic_name_to_id = {}

        # Get all topics across all chapters
        for chapter_id in chapters_json:
            chapter_response = supabase.table('chapters')\
                .select('topics_json')\
                .eq('chapter_id', chapter_id)\
                .single()\
                .execute()
            
            if chapter_response.data:
                topic_ids = chapter_response.data.get('topics_json', {}).get('topic_ids', [])
                
                topics_response = supabase.table('topics')\
                    .select('topic_id, topic_name')\
                    .in_('topic_id', topic_ids)\
                    .execute()
                for topic in topics_response.data:
                    topic_name_to_id[topic['topic_name']] = topic['topic_id']

        return {
            'skill_level': course_data['skill_level'],
            'teaching_pattern': course_data.get('teaching_pattern', []),
            'user_prompt': course_data.get('user_prompt', ''),
            'topic_name_to_id': topic_name_to_id
        }
    
    except Exception as e:
        logger.error(f"Error getting course generation data: {str(e)}")
        return None

def get_course_detailed_data(user_id: str, course_id: str) -> Optional[CourseDetailedData]:
    """
    Get detailed course data including all chapters, topics, and course metadata.
    
    Args:
        user_id (str): The ID of the user
        course_id (str): The ID of the course
        
    Returns:
        Optional[CourseDetailedData]: Detailed course data including chapters and topics,
                                    or None if course not found or error occurs
    
    Example:
        course_data = get_course_detailed_data(
            "15eee4cb-ad17-4818-88d4-5de9f15b55a7",
            "e26098b5-a504-4b13-88a5-02c0502886d5"
        )
        if course_data:
            print(f"Course: {course_data['course_name']}")
            print(f"Skill Level: {course_data['skill_level']}")
            for chapter in course_data['chapters']:
                print(f"\nChapter: {chapter['chapter_name']}")
                for topic in chapter['topics']:
                    print(f"  - Topic: {topic['topic_name']}")
    """
    try:
        # Initialize Supabase client
        supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        
        # First, verify user has access to this course
        user_response = supabase.table('users')\
            .select('courses_json')\
            .eq('user_id', user_id)\
            .single()\
            .execute()
            
        if not user_response.data or course_id not in user_response.data.get('courses_json', []):
            logger.error(f"User {user_id} does not have access to course {course_id}")
            return None
            
        # Get course data
        course_response = supabase.table('courses')\
            .select('*')\
            .eq('course_id', course_id)\
            .single()\
            .execute()
            
        if not course_response.data:
            logger.error(f"Course {course_id} not found")
            return None
            
        course_data = course_response.data
        chapters_json = course_data.get('chapters_json', {}).get('chapters', [])
        
        # Get all chapters data
        chapters_data: List[ChapterData] = []
        for chapter_id in chapters_json:
            # Get chapter details
            chapter_response = supabase.table('chapters')\
                .select('*')\
                .eq('chapter_id', chapter_id)\
                .single()\
                .execute()
                
            if chapter_response.data:
                chapter = chapter_response.data
                topic_ids = chapter.get('topics_json', {}).get('topic_ids', [])
                
                # Get topics for this chapter
                topics_response = supabase.table('topics')\
                    .select('topic_id, topic_name')\
                    .in_('topic_id', topic_ids)\
                    .execute()
                
                topics_data = [
                    {
                        'topic_id': topic['topic_id'],
                        'topic_name': topic['topic_name']
                    }
                    for topic in topics_response.data
                ]
                
                chapters_data.append({
                    'chapter_id': chapter['chapter_id'],
                    'chapter_name': chapter['chapter_name'],
                    'topics': topics_data,
                    'created_at': chapter['created_at'],
                    'tags': chapter.get('tags', {})
                })
        
        # Construct the final response
        detailed_data: CourseDetailedData = {
            'course_id': course_data['course_id'],
            'course_name': course_data['course_name'],
            'skill_level': course_data['skill_level'],
            'teaching_pattern': course_data.get('teaching_pattern', []),
            'user_prompt': course_data.get('user_prompt', ''),
            'chapters': chapters_data,
            'created_at': course_data['created_at'],
            'tags': course_data.get('tags')
        }
        
        return detailed_data
        
    except Exception as e:
        logger.error(f"Error getting detailed course data: {str(e)}")
        return None

def get_course_summary(user_id: str, course_id: str) -> Dict[str, Any]:
    """
    Get a summary of course data without detailed chapter and topic information.
    
    Args:
        user_id (str): The ID of the user
        course_id (str): The ID of the course
        
    Returns:
        Dict[str, Any]: Course summary data or empty dict if not found
    """
    try:
        supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        
        # Get course data
        course_response = supabase.table('courses')\
            .select('course_name, skill_level, teaching_pattern, user_prompt, created_at')\
            .eq('course_id', course_id)\
            .single()\
            .execute()
            
        if not course_response.data:
            return {}
            
        return {
            'course_id': course_id,
            'course_name': course_response.data['course_name'],
            'skill_level': course_response.data['skill_level'],
            'teaching_pattern': course_response.data.get('teaching_pattern', []),
            'user_prompt': course_response.data.get('user_prompt', ''),
            'created_at': course_response.data['created_at']
        }
        
    except Exception as e:
        logger.error(f"Error getting course summary: {str(e)}")
        return {}
    
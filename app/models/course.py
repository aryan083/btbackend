"""
Course Model Module
Defines the Course data model
"""
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
import uuid

@dataclass
class Course:
    """
    Course Model
    Represents a course in the system with the Supabase schema
    """
    course_name: str
    tags: Dict
    metadata: str
    chapters_json: Dict
    skill_level: int
    teaching_pattern: Dict
    user_prompt: str
    progress: float
    created_at: datetime
    course_id: str = str(uuid.uuid4())  # Primary key in Supabase

import os
from pathlib import Path
import logging
from run import custom_logger
logger = logging.getLogger(__name__)
@custom_logger.log_function_call
def create_directory(dir_path: str) -> bool:
    """
    Create a directory at the specified path
    @param dir_path: str - Path where the directory should be created
    @returns: bool - True if directory was created or already exists, False if creation failed
    @raises: ValueError if the path is invalid
    """
    try:
        # Convert string path to Path object for better handling
        path = upload_dir +Path(dir_path)
        
        # Create directory and any necessary parent directories
        path.mkdir(parents=True, exist_ok=True)
        return True
        
    except ValueError as e:
        raise ValueError(f"Invalid path format: {str(e)}")
    except PermissionError:
        raise PermissionError(f"Permission denied: Cannot create directory at {dir_path}")
    except Exception as e:
        raise Exception(f"Failed to create directory: {str(e)}")
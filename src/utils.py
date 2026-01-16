"""
Utility functions
"""

import os
from typing import List


def create_directories(dirs: List[str]) -> None:
    """Create multiple directories if they don't exist"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)


def cleanup_old_files(folder: str, max_age_hours: int = 24) -> int:
    """
    Delete files older than max_age_hours
    Returns number of deleted files
    """
    import time
    
    deleted = 0
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            age = now - os.path.getmtime(filepath)
            if age > max_age_seconds:
                os.remove(filepath)
                deleted += 1
    
    return deleted

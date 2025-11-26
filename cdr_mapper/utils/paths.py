"""
Path resolution utilities for CDR Mapper
"""
from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Get the cdr_mapper project root directory."""
    return Path(__file__).parent.parent


def get_data_base_path() -> Path:
    """Get the shared data directory path."""
    return get_project_root().parent / "data"


def get_cache_path() -> Path:
    """Get the cache directory path."""
    return get_project_root() / "data" / "cache"


def resolve_data_path(relative_path: Union[str, Path]) -> Path:
    """
    Resolve a relative data path to absolute path.
    
    Parameters:
    -----------
    relative_path : str or Path
        Relative path from data base directory
    
    Returns:
    --------
    Path: Absolute path to the data file
    """
    base_path = get_data_base_path()
    full_path = base_path / relative_path
    return full_path


def ensure_cache_dir(category: str) -> Path:
    """
    Ensure cache directory exists for a category.
    
    Parameters:
    -----------
    category : str
        Category name ('storage' or 'energy')
    
    Returns:
    --------
    Path: Cache directory path
    """
    cache_dir = get_cache_path() / category
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

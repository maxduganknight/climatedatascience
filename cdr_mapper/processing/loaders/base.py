"""
Base data loader class for CDR Mapper
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import geopandas as gpd
import pandas as pd


class DataLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    All loaders should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, config: dict, data_base_path: Path, cache_path: Path):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        config : dict
            Layer configuration from layers.yaml
        data_base_path : Path
            Base path to shared data directory
        cache_path : Path
            Path to cache directory
        """
        self.config = config
        self.data_base_path = Path(data_base_path)
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def load(self, use_cache: bool = True, **kwargs) -> Any:
        """
        Load and return the data.
        
        Parameters:
        -----------
        use_cache : bool
            If True, use cached data if available
        **kwargs : dict
            Additional loader-specific parameters
            
        Returns:
        --------
        Data in appropriate format (GeoDataFrame, ndarray, etc.)
        """
        pass
        
    @abstractmethod
    def get_cache_key(self) -> str:
        """
        Return unique cache identifier for this data.
        
        Returns:
        --------
        str: Cache key (filename without extension)
        """
        pass
        
    def get_cache_path(self, extension: str = 'gpkg') -> Path:
        """
        Get full path to cache file.
        
        Parameters:
        -----------
        extension : str
            File extension (default: 'gpkg')
            
        Returns:
        --------
        Path: Full path to cache file
        """
        return self.cache_path / f"{self.get_cache_key()}.{extension}"
        
    def cache_exists(self, extension: str = 'gpkg') -> bool:
        """
        Check if cached data exists.
        
        Parameters:
        -----------
        extension : str
            File extension to check
            
        Returns:
        --------
        bool: True if cache file exists
        """
        return self.get_cache_path(extension).exists()
        
    def get_data_path(self) -> Path:
        """
        Get full path to source data file.
        
        Returns:
        --------
        Path: Full path to data file
        """
        return self.data_base_path / self.config['data_path']

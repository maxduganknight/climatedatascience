"""
Land and country masking utilities
"""
import numpy as np
from shapely.geometry import Point
import cartopy.io.shapereader as shpreader


def create_land_mask(lon_grid: np.ndarray, lat_grid: np.ndarray, 
                     resolution: str = '110m') -> np.ndarray:
    """
    Create a boolean mask for land areas.
    
    Parameters:
    -----------
    lon_grid : ndarray
        2D array of longitude coordinates
    lat_grid : ndarray
        2D array of latitude coordinates
    resolution : str
        Resolution for natural earth data ('110m', '50m', '10m')
        
    Returns:
    --------
    ndarray: Boolean mask where True = land
    """
    from cartopy.io import shapereader
    from shapely.geometry import Point
    from shapely.prepared import prep
    
    # Load land polygons from Natural Earth
    land_shp = shapereader.natural_earth(resolution=resolution,
                                         category='physical',
                                         name='land')
    
    # Create prepared geometry for faster contains checks
    land_geom = shapereader.Reader(land_shp).geometries()
    land_union = None
    for geom in land_geom:
        if land_union is None:
            land_union = geom
        else:
            land_union = land_union.union(geom)
    
    land_prep = prep(land_union)
    
    # Create mask
    mask = np.zeros(lon_grid.shape, dtype=bool)
    
    # Vectorized point creation and testing
    shape = lon_grid.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            mask[i, j] = land_prep.contains(point)
    
    return mask

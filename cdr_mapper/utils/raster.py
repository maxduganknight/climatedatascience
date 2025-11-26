"""
Raster processing utilities.

Common raster operations like reprojection, resampling, and masking.
"""

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject


def reproject_raster(
    raster: np.ndarray,
    src_transform,
    src_crs: str,
    dst_crs: str,
    src_bounds: tuple = None,
    resampling_method=Resampling.bilinear,
    nodata_value=None,
) -> tuple:
    """
    Reproject a raster array from one CRS to another.

    Parameters:
    -----------
    raster : np.ndarray
        Source raster array
    src_transform : affine.Affine
        Source affine transform
    src_crs : str
        Source CRS (e.g., "EPSG:4326")
    dst_crs : str
        Destination CRS (e.g., "EPSG:3857")
    src_bounds : tuple, optional
        (left, bottom, right, top) in source CRS. If None, calculated from transform.
    resampling_method : Resampling, optional
        Resampling method (default: bilinear)
    nodata_value : float, optional
        NoData value (default: None for NaN handling)

    Returns:
    --------
    tuple : (reprojected_raster, dst_transform, dst_bounds)
    """
    height, width = raster.shape[:2]

    # Calculate source bounds if not provided
    if src_bounds is None:
        src_bounds = rasterio.transform.array_bounds(height, width, src_transform)

    # Calculate destination transform and dimensions
    dst_transform, dst_width, dst_height = calculate_default_transform(
        CRS.from_string(src_crs),
        CRS.from_string(dst_crs),
        width,
        height,
        *src_bounds,
    )

    # Create destination array with same dtype
    dst_raster = np.zeros((dst_height, dst_width), dtype=raster.dtype)

    # Perform reprojection
    reproject(
        source=raster,
        destination=dst_raster,
        src_transform=src_transform,
        src_crs=CRS.from_string(src_crs),
        dst_transform=dst_transform,
        dst_crs=CRS.from_string(dst_crs),
        resampling=resampling_method,
        src_nodata=nodata_value,
        dst_nodata=nodata_value,
    )

    # Calculate destination bounds
    dst_bounds = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)

    return dst_raster, dst_transform, dst_bounds


def downsample_raster(
    raster: np.ndarray,
    factor: int,
    method: str = "mean",
) -> np.ndarray:
    """
    Downsample a raster by an integer factor.

    Parameters:
    -----------
    raster : np.ndarray
        Input raster
    factor : int
        Downsampling factor (e.g., 2 = half resolution)
    method : str
        Downsampling method: "mean", "median", "min", "max"

    Returns:
    --------
    np.ndarray : Downsampled raster
    """
    if factor == 1:
        return raster

    from scipy.ndimage import zoom

    if method == "mean":
        # Use zoom with bilinear interpolation
        return zoom(raster, 1.0 / factor, order=1, mode="nearest")
    else:
        # For other methods, use block reduction
        from skimage.measure import block_reduce

        func_map = {
            "median": np.nanmedian,
            "min": np.nanmin,
            "max": np.nanmax,
        }

        return block_reduce(
            raster,
            block_size=(factor, factor),
            func=func_map.get(method, np.nanmean),
        )


def create_raster_transform(bounds: tuple, shape: tuple):
    """
    Create an affine transform for a raster.

    Parameters:
    -----------
    bounds : tuple
        (left, bottom, right, top)
    shape : tuple
        (height, width)

    Returns:
    --------
    affine.Affine : Affine transform
    """
    left, bottom, right, top = bounds
    height, width = shape
    return from_bounds(left, bottom, right, top, width, height)


def apply_mask_to_raster(
    raster: np.ndarray,
    mask: np.ndarray,
    keep_value: int = 1,
    fill_value=np.nan,
) -> np.ndarray:
    """
    Apply a binary mask to a raster.

    Parameters:
    -----------
    raster : np.ndarray
        Input raster
    mask : np.ndarray
        Binary mask (same shape as raster)
    keep_value : int
        Value in mask that indicates pixels to keep (default: 1)
    fill_value : float
        Value to fill masked pixels (default: np.nan)

    Returns:
    --------
    np.ndarray : Masked raster
    """
    result = raster.copy()
    result = np.where(mask == keep_value, result, fill_value)
    return result

"""
Distance calculation utilities for spatial analysis.

Provides efficient distance calculations for raster-based spatial operations.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def calculate_distance_from_mask(
    binary_mask: np.ndarray,
    pixel_size_meters: float,
) -> np.ndarray:
    """
    Calculate distance in meters from True pixels in a binary mask.

    Uses Euclidean distance transform and converts pixel distances to meters.

    Parameters:
    -----------
    binary_mask : np.ndarray
        Binary mask (True/1 = feature, False/0 = background)
    pixel_size_meters : float
        Size of each pixel in meters

    Returns:
    --------
    np.ndarray : Distance in meters from nearest True pixel
    """
    # Invert mask if needed (distance_transform_edt measures from False/0)
    if binary_mask.dtype == bool:
        distance_mask = ~binary_mask
    else:
        distance_mask = 1 - binary_mask

    # Calculate distance in pixels
    distance_pixels = distance_transform_edt(distance_mask)

    # Convert to meters
    distance_meters = distance_pixels * pixel_size_meters

    return distance_meters


def create_distance_buffer_mask(
    binary_mask: np.ndarray,
    pixel_size_meters: float,
    max_distance_meters: float,
) -> np.ndarray:
    """
    Create a buffer mask around features within a maximum distance.

    Parameters:
    -----------
    binary_mask : np.ndarray
        Binary mask of features (True/1 = feature)
    pixel_size_meters : float
        Size of each pixel in meters
    max_distance_meters : float
        Maximum buffer distance in meters

    Returns:
    --------
    np.ndarray : Binary mask (True within buffer distance, False beyond)
    """
    distance_meters = calculate_distance_from_mask(binary_mask, pixel_size_meters)
    buffer_mask = distance_meters <= max_distance_meters
    return buffer_mask.astype(np.uint8)


def miles_to_meters(miles: float) -> float:
    """Convert miles to meters."""
    return miles * 1609.34


def meters_to_miles(meters: float) -> float:
    """Convert meters to miles."""
    return meters / 1609.34


def km_to_meters(km: float) -> float:
    """Convert kilometers to meters."""
    return km * 1000


def meters_to_km(meters: float) -> float:
    """Convert meters to kilometers."""
    return meters / 1000

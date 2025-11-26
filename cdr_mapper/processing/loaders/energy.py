"""
Energy potential data loaders (solar PV and geothermal)
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling

from .base import DataLoader


class SolarLoader(DataLoader):
    """Loader for solar PV potential data from Global Solar Atlas."""

    def __init__(self, config: dict, data_base_path: Path, cache_path: Path):
        super().__init__(config, data_base_path, cache_path)
        self.target_resolution = config.get("target_resolution", 0.5)

    def load(
        self, use_cache: bool = True, target_resolution: float = None
    ) -> Tuple[np.ndarray, tuple]:
        """
        Load solar PV potential data and reproject to Web Mercator.

        Parameters:
        -----------
        use_cache : bool
            Use cached data if available
        target_resolution : float, optional
            Override target resolution in degrees

        Returns:
        --------
        tuple: (data_array, extent) where:
            - data_array is in Web Mercator projection (EPSG:3857)
            - extent is (lon_min, lat_min, lon_max, lat_max) in EPSG:4326
        """
        if target_resolution is not None:
            self.target_resolution = target_resolution

        # Check for Web Mercator cache
        cache_file = self.cache_path / f"{self.get_cache_key()}_3857.npz"
        if use_cache and cache_file.exists():
            print(
                f"Loading {self.config['name']} from Web Mercator cache...", flush=True
            )
            cached = np.load(cache_file)
            return cached["data"], tuple(cached["extent"])

        # Load from GeoTIFF
        print(f"Loading {self.config['name']} from GeoTIFF...", flush=True)
        data_path = self.get_data_path()

        if not data_path.exists():
            error_msg = f"Data file not found: {data_path}\nPlease ensure data is in the correct location."
            print(f"ERROR: {error_msg}", flush=True)
            raise FileNotFoundError(error_msg)

        with rasterio.open(data_path) as src:
            print(f"  Original shape: {src.shape}", flush=True)
            print(f"  Original resolution: {src.res}", flush=True)

            # Calculate downsampling factor
            downsample_factor = int(self.target_resolution / src.res[0])
            print(f"  Downsample factor: {downsample_factor}", flush=True)

            # Read and downsample
            data = src.read(
                1,
                out_shape=(
                    src.height // downsample_factor,
                    src.width // downsample_factor,
                ),
                resampling=Resampling.average,
            )

            # Get extent in standard format: (lon_min, lat_min, lon_max, lat_max)
            bounds = src.bounds
            extent = (bounds.left, bounds.bottom, bounds.right, bounds.top)

        # Clean data
        data = data.astype(float)
        data[data <= 0] = np.nan
        data[data > 10000] = np.nan

        print(f"  Final shape: {data.shape}", flush=True)
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            print(
                f"  Value range: {valid_data.min():.1f} - {valid_data.max():.1f} kWh/kWp/day",
                flush=True,
            )

        # Reproject to Web Mercator
        print(f"  Reprojecting solar data to Web Mercator...", flush=True)
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds as rast_from_bounds
        from rasterio.warp import Resampling as RioResampling
        from rasterio.warp import calculate_default_transform
        from rasterio.warp import reproject as rio_reproject

        lon_min, lat_min, lon_max, lat_max = extent
        height, width = data.shape

        # Clip to Web Mercator valid range
        MERCATOR_MAX_LAT = 85.05112878
        lat_min_clip = max(lat_min, -MERCATOR_MAX_LAT)
        lat_max_clip = min(lat_max, MERCATOR_MAX_LAT)

        if lat_min_clip != lat_min or lat_max_clip != lat_max:
            print(
                f"    Clipping latitude from [{lat_min}, {lat_max}] to [{lat_min_clip}, {lat_max_clip}]",
                flush=True,
            )
            pixel_height = (lat_max - lat_min) / height
            row_min = int((lat_max - lat_max_clip) / pixel_height)
            row_max = height - int((lat_min_clip - lat_min) / pixel_height)
            data = data[row_min:row_max, :]
            lat_min, lat_max = lat_min_clip, lat_max_clip
            height = data.shape[0]

        # Create transforms
        src_transform = rast_from_bounds(
            lon_min, lat_min, lon_max, lat_max, width, height
        )
        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_epsg(3857)

        # Calculate destination dimensions
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            dst_crs,
            width,
            height,
            left=lon_min,
            bottom=lat_min,
            right=lon_max,
            top=lat_max,
        )

        # Create destination array
        data_3857 = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        # Reproject with bilinear interpolation
        rio_reproject(
            source=data.astype(np.float32),
            destination=data_3857,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=RioResampling.bilinear,
        )

        extent_3857 = (lon_min, lat_min, lon_max, lat_max)
        print(
            f"    Reprojected: {width}x{height} -> {dst_width}x{dst_height}", flush=True
        )

        # Save Web Mercator version to cache
        print(f"  Saving Web Mercator cache: {cache_file}", flush=True)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, data=data_3857, extent=np.array(extent_3857))

        return data_3857, extent_3857

    def get_cache_key(self) -> str:
        """Generate cache key based on resolution."""
        return f"solar_pvout_res{self.target_resolution:.3f}"


class GeothermalLoader(DataLoader):
    """Loader for geothermal heat flow data from IHFC database."""

    def __init__(self, config: dict, data_base_path: Path, cache_path: Path):
        super().__init__(config, data_base_path, cache_path)

    def load(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load geothermal heat flow point data.

        Parameters:
        -----------
        use_cache : bool
            Use cached data if available

        Returns:
        --------
        DataFrame: Heat flow measurements with columns ['q', 'lat_NS', 'long_EW']
        """
        # Check cache
        cache_file = self.get_cache_path("parquet")
        if use_cache and cache_file.exists():
            print(f"Loading {self.config['name']} from cache...", flush=True)
            return pd.read_parquet(cache_file)

        # Load from Excel
        print(f"Loading {self.config['name']} from Excel...", flush=True)
        data_path = self.get_data_path()

        if not data_path.exists():
            error_msg = f"Data file not found: {data_path}\nPlease ensure data is in the correct location."
            print(f"ERROR: {error_msg}", flush=True)
            raise FileNotFoundError(error_msg)

        # Load with first row as header
        df = pd.read_excel(data_path, skiprows=4, nrows=1)
        col_names = df.iloc[0].tolist()

        # Load actual data
        df_data = pd.read_excel(data_path, skiprows=5)
        df_data.columns = col_names

        # Convert to numeric
        df_data["q"] = pd.to_numeric(df_data["q"], errors="coerce")
        df_data["lat_NS"] = pd.to_numeric(df_data["lat_NS"], errors="coerce")
        df_data["long_EW"] = pd.to_numeric(df_data["long_EW"], errors="coerce")

        # Filter valid data
        df_clean = df_data[["q", "lat_NS", "long_EW"]].dropna()

        # Remove extreme outliers (0-500 mW/m² is reasonable range)
        df_clean = df_clean[(df_clean["q"] > 0) & (df_clean["q"] < 500)]

        print(f"  Loaded {len(df_clean):,} valid measurements", flush=True)
        print(
            f"  Heat flow range: {df_clean['q'].min():.1f} - {df_clean['q'].max():.1f} mW/m²",
            flush=True,
        )

        # Save to cache
        print(f"  Saving to cache: {cache_file}", flush=True)
        df_clean.to_parquet(cache_file)

        return df_clean

    def get_cache_key(self) -> str:
        """Generate cache key."""
        return "geothermal_ihfc_2024"

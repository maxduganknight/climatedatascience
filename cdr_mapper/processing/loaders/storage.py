"""
Storage potential data loaders (geological formations)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np

from ..lithology_filters import (
    StorageTier,
    get_all_mineralization_tiers,
    get_all_sedimentary_tiers,
    get_tiered_storage_data,
    print_tier_summary,
)
from .base import DataLoader


class GeologicalLoader(DataLoader):
    """Base loader for geological data from GLiM database."""

    def __init__(
        self,
        config: dict,
        data_base_path: Path,
        cache_path: Path,
        lithology_codes: Optional[List[str]] = None,
        storage_tiers: Optional[List[StorageTier]] = None,
    ):
        super().__init__(config, data_base_path, cache_path)
        self.lithology_codes = lithology_codes
        self.storage_tiers = storage_tiers
        self.use_tiered_filtering = storage_tiers is not None
        self.simplify_tolerance = config.get("simplify_tolerance", 0.3)
        self.rasterize = config.get("rasterize", True)  # Default to rasterized
        self.raster_resolution = config.get("raster_resolution", 0.5)  # degrees

    def load(
        self,
        use_cache: bool = True,
        simplify_tolerance: float = None,
        as_raster: bool = None,
    ) -> Union[gpd.GeoDataFrame, Tuple[np.ndarray, tuple]]:
        """
        Load geological data from GLiM database.

        Parameters:
        -----------
        use_cache : bool
            Use cached data if available
        simplify_tolerance : float, optional
            Override default simplification tolerance
        as_raster : bool, optional
            Return as rasterized data instead of vector. If None, uses config setting.

        Returns:
        --------
        GeoDataFrame or tuple:
            - If as_raster=False: GeoDataFrame of geological formations
            - If as_raster=True: (raster_array, extent) tuple
        """
        if simplify_tolerance is not None:
            self.simplify_tolerance = simplify_tolerance

        if as_raster is None:
            as_raster = self.rasterize

        # Check cache for vector data
        if use_cache and self.cache_exists():
            print(f"Loading {self.config['name']} from cache...", flush=True)
            gdf = gpd.read_file(self.get_cache_path())

            # Rasterize if requested
            if as_raster:
                return self._rasterize_data(gdf, use_cache)

            return gdf

        # Load from geodatabase
        print(f"Loading {self.config['name']} from geodatabase...", flush=True)
        data_path = self.get_data_path()

        if not data_path.exists():
            error_msg = f"Data file not found: {data_path}\nPlease ensure data is in the correct location."
            print(f"ERROR: {error_msg}", flush=True)
            raise FileNotFoundError(error_msg)

        gdf = gpd.read_file(data_path)
        print(f"  Loaded {len(gdf):,} total features", flush=True)

        # Reproject to WGS84 if needed
        if gdf.crs != "EPSG:4326":
            print("  Reprojecting to WGS84...", flush=True)
            gdf = gdf.to_crs("EPSG:4326")

        # Apply filtering based on mode
        if self.use_tiered_filtering:
            # Use advanced tiered filtering with subclass support
            print(f"  Applying tiered lithology filtering...", flush=True)
            gdf = get_tiered_storage_data(
                gdf, self.storage_tiers, simplify_tolerance=self.simplify_tolerance
            )
            print_tier_summary(gdf)
        else:
            # Use simple xx-level filtering (legacy mode)
            print(
                f"  Filtering for lithology codes: {', '.join(self.lithology_codes)}",
                flush=True,
            )
            gdf = gdf[gdf["xx"].isin(self.lithology_codes)].copy()
            print(f"  Filtered to {len(gdf):,} features", flush=True)

            # Simplify geometries
            if self.simplify_tolerance > 0:
                print(
                    f"  Simplifying geometries (tolerance={self.simplify_tolerance})...",
                    flush=True,
                )
                gdf["geometry"] = gdf["geometry"].simplify(self.simplify_tolerance)

        # Save to cache
        print(f"  Saving to cache: {self.get_cache_path()}", flush=True)
        gdf.to_file(self.get_cache_path(), driver="GPKG")

        # Rasterize if requested
        if as_raster:
            return self._rasterize_data(gdf, use_cache)

        return gdf

    def _rasterize_data(
        self, gdf: gpd.GeoDataFrame, use_cache: bool
    ) -> Tuple[np.ndarray, tuple]:
        """
        Convert vector data to raster format and reproject to Web Mercator.

        This performs two critical steps:
        1. Rasterize vector geometries to a grid in EPSG:4326
        2. Reproject the raster to Web Mercator (EPSG:3857)

        The reprojection is required for proper alignment with the base map.
        Without it, features appear vertically stretched and horizontally displaced.
        """
        import sys
        from pathlib import Path

        # Add project root to path if not already there
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from processing.rasterize import (
            rasterize_geological_layer,
            reproject_raster_to_web_mercator,
        )

        # Check for Web Mercator raster cache
        raster_cache_path = (
            self.cache_path
            / f"{self.get_cache_key()}_raster_3857_{self.raster_resolution:.3f}.npz"
        )

        if use_cache and raster_cache_path.exists():
            print(f"  Loading Web Mercator rasterized data from cache...", flush=True)
            cached = np.load(raster_cache_path)
            return cached["data"], tuple(cached["extent"])

        # Rasterize the vector data in EPSG:4326
        raster_4326, extent_4326 = rasterize_geological_layer(
            gdf,
            resolution=self.raster_resolution,
            cache_path=None,  # Don't cache EPSG:4326 version
        )

        # Reproject to Web Mercator (EPSG:3857)
        raster_3857, extent_3857 = reproject_raster_to_web_mercator(
            raster_4326, extent_4326
        )

        # Cache the Web Mercator version
        if use_cache:
            print(
                f"  Saving Web Mercator raster to cache: {raster_cache_path}",
                flush=True,
            )
            raster_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raster_cache_path,
                data=raster_3857,
                extent=np.array(extent_3857),
            )

        return raster_3857, extent_3857

    def get_cache_key(self) -> str:
        """Generate cache key based on lithology codes/tiers and tolerance."""
        if self.use_tiered_filtering:
            # Create key from tier names
            tier_names = "_".join(
                [
                    tier.name.split(":")[0].replace(" ", "")
                    for tier in self.storage_tiers
                ]
            )
            return f"glim_tiered_{tier_names}_tol{self.simplify_tolerance:.3f}"
        else:
            # Legacy mode: use lithology codes
            codes_str = "_".join(sorted(self.lithology_codes))
            return f"glim_{codes_str}_tol{self.simplify_tolerance:.3f}"


class BasalticLoader(GeologicalLoader):
    """Loader for basaltic and ultramafic formations (mineralization potential)."""

    def __init__(self, config: dict, data_base_path: Path, cache_path: Path):
        # Check if tiered filtering is enabled
        use_tiered = config.get("use_tiered_filtering", False)

        if use_tiered:
            # Use advanced tiered filtering
            storage_tiers = get_all_mineralization_tiers()
            super().__init__(
                config, data_base_path, cache_path, storage_tiers=storage_tiers
            )
        else:
            # Legacy mode: simple xx-level filtering
            lithology_codes = config.get("lithology_codes", ["vb", "vi", "pb", "mt"])
            super().__init__(
                config, data_base_path, cache_path, lithology_codes=lithology_codes
            )


class SedimentaryLoader(GeologicalLoader):
    """Loader for sedimentary formations (porous storage potential)."""

    def __init__(self, config: dict, data_base_path: Path, cache_path: Path):
        # Check if tiered filtering is enabled
        use_tiered = config.get("use_tiered_filtering", False)

        if use_tiered:
            # Use advanced tiered filtering
            storage_tiers = get_all_sedimentary_tiers()
            super().__init__(
                config, data_base_path, cache_path, storage_tiers=storage_tiers
            )
        else:
            # Legacy mode: simple xx-level filtering
            lithology_codes = config.get("lithology_codes", ["ss"])
            super().__init__(
                config, data_base_path, cache_path, lithology_codes=lithology_codes
            )


class SedimentaryThicknessLoader(DataLoader):
    """Loader for sedimentary thickness raster (Pilorge et al., CDR Primer)."""

    def __init__(self, config: dict, data_base_path: Path, cache_path: Path):
        super().__init__(config, data_base_path, cache_path)
        self.min_thickness = config.get(
            "min_thickness", 100
        )  # meters, minimum viable thickness
        self.downsample_factor = config.get(
            "downsample_factor", 4
        )  # Reduce resolution for performance (1=full, 2=half, 4=quarter)
        # Kearns offshore accessibility criteria
        self.apply_offshore_mask = config.get(
            "apply_offshore_mask", True
        )  # Apply Kearns accessibility criteria
        self.max_offshore_distance_miles = config.get(
            "max_offshore_distance_miles", 200
        )  # Maximum distance from shore
        self.min_landmass_km2 = config.get(
            "min_landmass_km2", 10000
        )  # Minimum landmass size

        # self.latitude_bounds = tuple(
        #     config.get("latitude_bounds", [-66.0, 66.0])
        # )  # Arctic/Antarctic exclusion

    def load(self, use_cache: bool = True) -> Tuple[np.ndarray, tuple]:
        """
        Load sedimentary thickness raster and reproject to Web Mercator.

        Returns:
        --------
        tuple: (thickness_array, extent)
            - thickness_array: 2D numpy array in EPSG:3857 with thickness values in meters
            - extent: (lon_min, lat_min, lon_max, lat_max) in EPSG:4326 (for Folium)
        """
        import rasterio
        from rasterio.warp import Resampling, calculate_default_transform, reproject

        # Check cache
        cache_path = self.cache_path / f"{self.get_cache_key()}_thickness_3857.npz"
        if use_cache and cache_path.exists():
            print(f"Loading {self.config['name']} from cache...", flush=True)
            cached = np.load(cache_path)
            return cached["data"], tuple(cached["extent"])

        print(f"Loading {self.config['name']} from GeoTIFF...", flush=True)
        data_path = self.get_data_path()

        if not data_path.exists():
            error_msg = f"Data file not found: {data_path}\nPlease ensure data is in the correct location."
            print(f"ERROR: {error_msg}", flush=True)
            raise FileNotFoundError(error_msg)

        with rasterio.open(data_path) as src:
            print(f"  Source CRS: {src.crs}", flush=True)
            print(f"  Source shape: {src.shape}", flush=True)

            # Read the raster data
            thickness = src.read(1)

            # Mask out nodata and values below minimum threshold
            nodata = src.nodata
            if nodata is not None:
                thickness = np.where(thickness == nodata, np.nan, thickness)

            # Apply minimum thickness filter (< 100m not viable for storage)
            thickness = np.where(thickness < self.min_thickness, np.nan, thickness)

            print(
                f"  Valid thickness range: {np.nanmin(thickness):.0f}m - {np.nanmax(thickness):.0f}m",
                flush=True,
            )

            # Store original info for mask generation later
            src_crs = str(src.crs)
            src_bounds = src.bounds
            src_shape = thickness.shape

            # Downsample for performance if requested
            if self.downsample_factor > 1:
                from scipy.ndimage import zoom

                print(
                    f"  Downsampling by factor of {self.downsample_factor}...",
                    flush=True,
                )
                original_shape = thickness.shape
                thickness = zoom(
                    thickness,
                    1.0 / self.downsample_factor,
                    order=1,  # Bilinear interpolation
                    mode="nearest",
                )
                print(f"    {original_shape} → {thickness.shape}", flush=True)

                # Update transform for downsampled data
                downsampled_transform = src.transform * src.transform.scale(
                    self.downsample_factor, self.downsample_factor
                )
            else:
                downsampled_transform = src.transform

            # Reproject to EPSG:4326 (WGS84 lat/lon) - same as geological layers before they're reprojected
            dst_crs = "EPSG:4326"

            print(f"  Reprojecting to {dst_crs} (WGS84)...", flush=True)
            transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                thickness.shape[1],
                thickness.shape[0],
                *rasterio.transform.array_bounds(
                    thickness.shape[0], thickness.shape[1], downsampled_transform
                ),
            )

            # Create output array
            thickness_4326 = np.empty((height, width), dtype=np.float32)

            reproject(
                source=thickness,
                destination=thickness_4326,
                src_transform=downsampled_transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

            # Get geographic extent
            bounds_4326 = rasterio.transform.array_bounds(height, width, transform)
            extent_4326 = bounds_4326  # (left, bottom, right, top) in EPSG:4326

            print(f"  Output shape: {thickness_4326.shape}", flush=True)
            print(f"  Geographic extent: {extent_4326}", flush=True)

        # Apply latitude bounds (66°N to 66°S) in EPSG:4326 to exclude polar regions
        # This is done in geographic coordinates to avoid distortion
        print(
            "\n  Applying latitude constraint (±66° Arctic/Antarctic exclusion)...",
            flush=True,
        )

        # Create latitude grid for the EPSG:4326 raster
        height, width = thickness_4326.shape
        lon_min, lat_min, lon_max, lat_max = (
            extent_4326  # extent is (left, bottom, right, top)
        )

        # Create arrays of latitude values for each row
        lat_values = np.linspace(lat_max, lat_min, height)  # Top to bottom
        lat_grid = np.repeat(lat_values[:, np.newaxis], width, axis=1)

        # Apply latitude mask (keep only between -66° and +66°)
        latitude_mask = (lat_grid >= -66.0) & (lat_grid <= 66.0)

        valid_before_lat = np.sum(~np.isnan(thickness_4326))
        thickness_4326 = np.where(latitude_mask, thickness_4326, np.nan)
        valid_after_lat = np.sum(~np.isnan(thickness_4326))

        excluded_by_lat = valid_before_lat - valid_after_lat
        print(
            f"  Excluded {excluded_by_lat:,} pixels in polar regions "
            f"({excluded_by_lat / valid_before_lat * 100:.1f}% of valid data)",
            flush=True,
        )

        # Clean up
        del lat_grid, latitude_mask

        # Apply Kearns offshore accessibility mask AFTER reprojection
        # This ensures the mask and data are in the same CRS and aligned correctly
        if self.apply_offshore_mask:
            from ..offshore_mask import create_offshore_accessibility_mask

            # Get path to world countries GeoJSON
            countries_path = (
                self.data_base_path.parent
                / "data/shapefiles/downloaded/world_countries.geojson"
            )

            if not countries_path.exists():
                print(
                    f"  WARNING: World countries GeoJSON not found at {countries_path}",
                    flush=True,
                )
                print("  Skipping offshore accessibility mask", flush=True)
            else:
                print(
                    "\n  Applying offshore accessibility mask (distance only)...",
                    flush=True,
                )

                # Create mask in EPSG:4326 to match the reprojected thickness data
                # Use ±90° latitude bounds to disable latitude filtering - only apply distance criterion
                accessibility_mask = create_offshore_accessibility_mask(
                    raster_shape=thickness_4326.shape,
                    raster_bounds=extent_4326,
                    raster_crs="EPSG:4326",
                    countries_geojson_path=str(countries_path),
                    max_distance_miles=self.max_offshore_distance_miles,
                    min_landmass_km2=self.min_landmass_km2,
                    latitude_bounds=(-90.0, 90.0),  # Disable latitude filtering
                )

                # Apply mask: set inaccessible offshore areas to NaN
                valid_before_mask = np.sum(~np.isnan(thickness_4326))
                thickness_4326 = np.where(
                    accessibility_mask == 0, np.nan, thickness_4326
                )
                valid_after_mask = np.sum(~np.isnan(thickness_4326))

                print(
                    f"  Valid pixels after offshore mask: {valid_after_mask:,} "
                    f"(excluded {valid_before_mask - valid_after_mask:,} pixels)",
                    flush=True,
                )

        # Reproject to Web Mercator (EPSG:3857) - same as geological layers
        # This ensures proper alignment with the base map
        from ..rasterize import reproject_raster_to_web_mercator

        print("\n  Reprojecting to Web Mercator for proper alignment...", flush=True)
        thickness_3857, extent = reproject_raster_to_web_mercator(
            thickness_4326, extent_4326, is_boolean=False
        )

        # Cache the result
        if use_cache:
            print(f"  Saving to cache: {cache_path}", flush=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                data=thickness_3857,
                extent=np.array(extent),
            )

        return thickness_3857, extent

    def get_cache_key(self) -> str:
        """Generate cache key."""
        key = f"pilorge_sediment_thickness_min{self.min_thickness}_ds{self.downsample_factor}"
        if self.apply_offshore_mask:
            key += f"_offshore{self.max_offshore_distance_miles}"
        return key

"""
Advanced lithology filtering for GLiM geological data.

This module provides fine-grained filtering of geological formations based on
the GLiM classification system's three-level hierarchy (xxyyzz):
- xx: First level (dominant lithology)
- yy: Second level (subclass)
- zz: Third level (special attributes)

References:
    Hartmann, J., & Moosdorf, N. (2012). The new global lithological map database
    GLiM: A representation of rock properties at the Earth surface.
    Geochemistry, Geophysics, Geosystems, 13(12).
"""

from dataclasses import dataclass
from typing import List, Optional, Set

import geopandas as gpd
import pandas as pd


@dataclass
class LithologyFilter:
    """
    Define a lithology filter with inclusion/exclusion rules.

    Attributes:
        xx_codes: First-level codes to include (e.g., ['vb', 'pb'])
        exclude_yy: Second-level codes to exclude (e.g., ['py', 'ev'])
        exclude_zz: Third-level codes to exclude (e.g., ['pr'])
        require_yy: If set, only include these second-level codes
        require_zz: If set, only include these third-level codes
    """

    xx_codes: List[str]
    exclude_yy: Optional[List[str]] = None
    exclude_zz: Optional[List[str]] = None
    require_yy: Optional[List[str]] = None
    require_zz: Optional[List[str]] = None


@dataclass
class StorageTier:
    """
    Define a storage potential tier with geological criteria.

    Attributes:
        name: Tier name (e.g., "Tier 1: Highest Potential")
        description: Description of the tier
        filters: List of lithology filters for this tier
        color: Hex color code for visualization
    """

    name: str
    description: str
    filters: List[LithologyFilter]
    color: str


# ==============================================================================
# TIER DEFINITIONS FOR CDR STORAGE POTENTIAL
# ==============================================================================

MINERALIZATION_TIER_1 = StorageTier(
    name="Mineralization: Tier 1 (Highest Potential)",
    description="Basaltic volcanics with high reactivity and Ca-Mg-Fe content",
    filters=[
        LithologyFilter(
            xx_codes=["vb"],  # Basic volcanic rocks (basalt)
            exclude_yy=["py"],  # Exclude pyroclastics (variable properties)
        )
    ],
    color="#1B4D3E",  # Dark teal
)

MINERALIZATION_TIER_2 = StorageTier(
    name="Mineralization: Tier 2 (Moderate Potential)",
    description="Basic plutonic and intermediate volcanic rocks with moderate reactivity",
    filters=[
        LithologyFilter(
            xx_codes=["pb"],  # Basic plutonic (gabbro, diorite)
            # No exclusions - all pb subtypes acceptable
        ),
        LithologyFilter(
            xx_codes=["vi"],  # Intermediate volcanic rocks
            exclude_yy=["py"],  # Exclude pyroclastics
        ),
    ],
    color="#3F6B6F",  # Medium teal
)

MINERALIZATION_TIER_3 = StorageTier(
    name="Mineralization: Tier 3 (Case-by-Case)",
    description="Metamorphic rocks with potential mafic/ultramafic protolith",
    filters=[
        LithologyFilter(
            xx_codes=["mt"],  # Metamorphic rocks
            # Include amphibolite (am) and pyroxene (pu) metamorphics
            # These often indicate mafic/ultramafic protoliths
            require_yy=["am", "pu"],  # Amphibolite, pyroxenite metamorphics
        ),
    ],
    color="#6B9B9E",  # Light teal
)

SEDIMENTARY_TIER_1 = StorageTier(
    name="Sedimentary: Tier 1 (Highest Potential)",
    description="Sandstone formations, excluding pyroclastics and evaporites",
    filters=[
        LithologyFilter(
            xx_codes=["ss"],  # Siliciclastic sedimentary
            exclude_yy=["py"],  # Exclude pyroclastics
            exclude_zz=["ev"],  # Exclude evaporites (dissolution risk)
        )
    ],
    color="#C27D1B",  # Dark orange
)

# SEDIMENTARY_TIER_2 = StorageTier(
#     name="Sedimentary: Tier 2 (Moderate Potential)",
#     description="Unconsolidated sediments and carbonate rocks",
#     filters=[
#         LithologyFilter(
#             xx_codes=["su"],  # Unconsolidated sediments
#             exclude_yy=["py"],  # Exclude pyroclastics
#             exclude_zz=["ev"],  # Exclude evaporites
#         ),
#         LithologyFilter(
#             xx_codes=["sc"],  # Carbonate sedimentary
#             exclude_yy=["py"],  # Exclude pyroclastics
#             exclude_zz=["ev"],  # Exclude evaporites
#         ),
#     ],
#     color="#E8A33D",  # Medium orange
# )

# Explicitly excluded formations (for documentation)
EXCLUDED_FORMATIONS = {
    "va": "Acidic volcanic (rhyolite) - low Ca-Mg-Fe content",
    "pa": "Acidic plutonic (granite) - low reactivity",
    "pi": "Intermediate plutonic - lower reactivity than basic rocks",
    "sm": "Mixed sedimentary - too heterogeneous without detailed data",
    "ev": "Evaporites - dissolution risk, poor seals",
    "py": "Pyroclastics - variable properties, not typical reservoir rocks",
}


# ==============================================================================
# FILTERING FUNCTIONS
# ==============================================================================


def parse_litho_code(litho: str) -> tuple:
    """
    Parse a 6-character GLiM lithology code into its components.

    Args:
        litho: 6-character code (e.g., "vb____", "mtam__", "sspyev")

    Returns:
        (xx, yy, zz) tuple where underscores indicate "not specified"
    """
    litho = str(litho).strip()

    # Ensure it's 6 characters
    if len(litho) != 6:
        raise ValueError(f"Litho code must be 6 characters, got: {litho}")

    xx = litho[0:2]
    yy = litho[2:4] if litho[2:4] != "__" else None
    zz = litho[4:6] if litho[4:6] != "__" else None

    return xx, yy, zz


def apply_lithology_filter(
    gdf: gpd.GeoDataFrame, litho_filter: LithologyFilter
) -> gpd.GeoDataFrame:
    """
    Apply a single lithology filter to a GeoDataFrame.

    Args:
        gdf: GeoDataFrame with 'Litho' column (6-char codes)
        litho_filter: Filter specification

    Returns:
        Filtered GeoDataFrame
    """
    # Start with xx-level filter
    mask = gdf["xx"].isin(litho_filter.xx_codes)

    # Parse litho codes for yy and zz filtering
    if (
        litho_filter.exclude_yy
        or litho_filter.require_yy
        or litho_filter.exclude_zz
        or litho_filter.require_zz
    ):
        # Extract yy and zz from Litho column
        litho_parsed = gdf["Litho"].apply(parse_litho_code)
        gdf_temp = gdf.copy()
        gdf_temp["yy"] = litho_parsed.apply(lambda x: x[1])
        gdf_temp["zz"] = litho_parsed.apply(lambda x: x[2])

        # Apply yy-level filters
        if litho_filter.require_yy:
            # Only keep rows with specified yy codes (exclude None/unspecified)
            yy_mask = gdf_temp["yy"].isin(litho_filter.require_yy)
            mask = mask & yy_mask

        if litho_filter.exclude_yy:
            # Exclude rows with specified yy codes
            yy_mask = ~gdf_temp["yy"].isin(litho_filter.exclude_yy)
            mask = mask & yy_mask

        # Apply zz-level filters
        if litho_filter.require_zz:
            zz_mask = (
                gdf_temp["zz"].isin(litho_filter.require_zz) | gdf_temp["zz"].isna()
            )
            mask = mask & zz_mask

        if litho_filter.exclude_zz:
            zz_mask = ~gdf_temp["zz"].isin(litho_filter.exclude_zz)
            mask = mask & zz_mask

    return gdf[mask].copy()


def apply_storage_tier(gdf: gpd.GeoDataFrame, tier: StorageTier) -> gpd.GeoDataFrame:
    """
    Apply all filters for a storage tier and add tier metadata.

    Args:
        gdf: GeoDataFrame with GLiM lithology data
        tier: Storage tier definition

    Returns:
        Filtered GeoDataFrame with added 'tier' and 'tier_name' columns
    """
    # Apply each filter and combine results
    filtered_gdfs = []

    for litho_filter in tier.filters:
        filtered = apply_lithology_filter(gdf, litho_filter)
        if len(filtered) > 0:
            filtered_gdfs.append(filtered)

    if not filtered_gdfs:
        # Return empty GeoDataFrame with same schema
        result = gdf.iloc[0:0].copy()
        result["tier_name"] = pd.Series(dtype=str)
        result["tier_color"] = pd.Series(dtype=str)
        return result

    # Combine all filtered results
    result = gpd.GeoDataFrame(pd.concat(filtered_gdfs, ignore_index=True))

    # Add tier metadata
    result["tier_name"] = tier.name
    result["tier_color"] = tier.color

    return result


def get_tiered_storage_data(
    gdf: gpd.GeoDataFrame, tiers: List[StorageTier], simplify_tolerance: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Apply multiple storage tiers and return combined dataset with tier labels.

    Args:
        gdf: GeoDataFrame with GLiM lithology data
        tiers: List of storage tier definitions
        simplify_tolerance: Simplification tolerance for geometries

    Returns:
        GeoDataFrame with all tiers combined, including tier metadata
    """
    tier_gdfs = []

    print(f"Applying {len(tiers)} storage tiers...", flush=True)

    for i, tier in enumerate(tiers, 1):
        print(f"  Tier {i}: {tier.name}", flush=True)
        tier_gdf = apply_storage_tier(gdf, tier)
        print(f"    → {len(tier_gdf):,} features", flush=True)

        if len(tier_gdf) > 0:
            tier_gdfs.append(tier_gdf)

    if not tier_gdfs:
        print("  WARNING: No features matched any tier", flush=True)
        result = gdf.iloc[0:0].copy()
        result["tier_name"] = pd.Series(dtype=str)
        result["tier_color"] = pd.Series(dtype=str)
        return result

    # Combine all tiers
    result = gpd.GeoDataFrame(pd.concat(tier_gdfs, ignore_index=True))

    # Simplify geometries if requested
    if simplify_tolerance > 0:
        print(
            f"  Simplifying geometries (tolerance={simplify_tolerance})...", flush=True
        )
        result["geometry"] = result["geometry"].simplify(simplify_tolerance)

    print(f"  Total features across all tiers: {len(result):,}", flush=True)

    return result


def get_all_mineralization_tiers() -> List[StorageTier]:
    """Get all mineralization storage tiers."""
    return [
        MINERALIZATION_TIER_1,
        MINERALIZATION_TIER_2,
        MINERALIZATION_TIER_3,
    ]


def get_all_sedimentary_tiers() -> List[StorageTier]:
    """Get all sedimentary storage tiers."""
    return [
        SEDIMENTARY_TIER_1,
        # SEDIMENTARY_TIER_2,  # Disabled - too extensive globally
    ]


def print_tier_summary(gdf: gpd.GeoDataFrame):
    """
    Print a summary of features by tier.

    Args:
        gdf: GeoDataFrame with 'tier_name' column
    """
    if "tier_name" not in gdf.columns:
        print("No tier information in GeoDataFrame")
        return

    print("\n" + "=" * 70)
    print("STORAGE POTENTIAL TIER SUMMARY")
    print("=" * 70)

    tier_counts = gdf["tier_name"].value_counts()

    for tier_name, count in tier_counts.items():
        print(f"\n{tier_name}")
        print(f"  Features: {count:,}")

        # Show lithology code breakdown
        tier_data = gdf[gdf["tier_name"] == tier_name]
        litho_counts = tier_data["Litho"].value_counts().head(5)
        print(f"  Top lithology codes:")
        for litho, litho_count in litho_counts.items():
            print(f"    {litho}: {litho_count:,}")

    print("\n" + "=" * 70)

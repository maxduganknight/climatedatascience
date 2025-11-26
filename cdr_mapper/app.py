"""
CDR Mapper - Interactive Carbon Removal Resource Explorer

Main Streamlit application for visualizing CDR storage and energy potential.
"""

import sys
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import load_config
from processing.loaders import get_loader
from utils import get_cache_path, get_data_base_path, get_logger
from visualization import (
    add_geological_layer,
    add_geothermal_layer,
    add_legend_to_map,
    add_solar_layer,
    add_thickness_layer,
    create_base_map,
)

# Page configuration
st.set_page_config(page_title="CDR Mapper", page_icon="🌍", layout="wide")


# Initialize logger
@st.cache_resource
def get_app_logger():
    """Get the application logger."""
    log_dir = project_root / "logs"
    return get_logger(log_dir)


logger = get_app_logger()


# Load configurations
@st.cache_resource
def load_all_configs():
    """Load all configuration files."""
    return {
        "layers": load_config("layers"),
        "settings": load_config("settings"),
    }


configs = load_all_configs()
layer_config = configs["layers"]
settings = configs["settings"]

# App title
st.title(settings["app"]["title"])

selected_layers = {}

# Storage layers
st.sidebar.subheader("Storage Potential")
for layer_id, layer_info in layer_config["layers"]["storage"].items():
    key = f"storage.{layer_id}"
    selected_layers[key] = st.sidebar.checkbox(
        layer_info["name"],
        value=layer_info["enabled_by_default"],
        help=layer_info["description"],
    )

# Energy layers
st.sidebar.subheader("Energy Resources")
for layer_id, layer_info in layer_config["layers"]["energy"].items():
    key = f"energy.{layer_id}"
    selected_layers[key] = st.sidebar.checkbox(
        layer_info["name"],
        value=layer_info["enabled_by_default"],
        help=layer_info["description"],
    )

# Main content area
st.markdown("---")

# Filter to get only selected layers
active_layers = {k: v for k, v in selected_layers.items() if v}

# Initialize session state for caching
if "last_active_layers" not in st.session_state:
    st.session_state.last_active_layers = None
if "cached_map" not in st.session_state:
    st.session_state.cached_map = None

# Check if layers have changed
layers_changed = st.session_state.last_active_layers != set(active_layers.keys())

# Log session start and layer selection
if layers_changed or st.session_state.cached_map is None:
    logger.log_session_start()
    logger.log_layer_selection(selected_layers)

# Only reload data and rebuild map if layers have changed
if layers_changed or st.session_state.cached_map is None:
    layer_data = {}
    data_base_path = get_data_base_path()
    cache_base_path = get_cache_path()

    with st.spinner("Loading data..."):
        for layer_key in active_layers.keys():
            category, layer_id = layer_key.split(".")
            layer_info = layer_config["layers"][category][layer_id]

            try:
                logger.log_layer_load_start(layer_key, layer_info["name"])

                loader = get_loader(
                    layer_key, layer_info, data_base_path, cache_base_path
                )
                data = loader.load(use_cache=True)

                # Log data info
                if isinstance(data, tuple):
                    data_info = f"raster {data[0].shape}, extent: {data[1]}"
                    print(
                        f"  Loaded as raster: {data[0].shape}, extent: {data[1]}",
                        flush=True,
                    )
                    data_size = data[0].nbytes / 1024 / 1024
                    print(f"  Raster size: {data_size:.2f} MB", flush=True)
                else:
                    data_info = f"vector with {len(data)} features"
                    print(f"  Loaded as vector: {len(data)} features", flush=True)
                    # Estimate size (rough)
                    import sys

                    data_size = sys.getsizeof(data) / 1024 / 1024
                    print(f"  Vector size estimate: {data_size:.2f} MB", flush=True)

                logger.log_layer_load_success(layer_key, data_info)
                layer_data[layer_key] = {"data": data, "config": layer_info}
                # st.success(f"✓ Loaded {layer_info['name']}")

            except FileNotFoundError as e:
                logger.log_layer_load_error(layer_key, layer_info["name"], e)
                st.error(f"❌ Data file not found for {layer_info['name']}")
                st.error(str(e))
                st.info(
                    "Please ensure data files are in the correct location (../data/)"
                )
                continue
            except Exception as e:
                logger.log_layer_load_error(layer_key, layer_info["name"], e)
                st.error(f"❌ Error loading {layer_info['name']}: {str(e)}")
                continue

    # Create map

    # Set global map center and zoom
    map_center = [20, 0]  # Global center
    map_zoom = 2  # Global zoom level

    # Create base map
    print("Creating base map...", flush=True)
    m = create_base_map(
        center=map_center,
        zoom=map_zoom,
        tile_layer=settings["map"]["default_tile_layer"],
    )
    print(f"  Map created at center {map_center}, zoom {map_zoom}", flush=True)

    # Add layers to map
    logger.log_map_render_start(len(layer_data))
    print(f"Adding {len(layer_data)} layers to map...", flush=True)

    for layer_key, layer_info in layer_data.items():
        category = layer_key.split(".")[0]
        layer_name = layer_key.split(".")[1]
        print(
            f"  Adding layer: {layer_info['config']['name']} ({category})", flush=True
        )

        try:
            if category == "storage":
                # Check if this is the sedimentary thickness layer
                if layer_name == "sedimentary_thickness":
                    # Sedimentary thickness is a raster with graduated colors
                    data_array, extent = layer_info["data"]
                    m = add_thickness_layer(
                        m,
                        data_array,
                        extent,
                        layer_info["config"]["name"],
                        layer_info["config"]["thresholds"],
                        layer_info["config"]["colors"],
                        layer_info["config"]["opacity"],
                    )
                    print(f"    ✓ Thickness layer added", flush=True)
                else:
                    # Add geological layer (rasterized)
                    m = add_geological_layer(
                        m,
                        layer_info["data"],
                        layer_info["config"]["name"],
                        layer_info["config"]["color"],
                        layer_info["config"]["opacity"],
                    )
                    print(f"    ✓ Layer added", flush=True)

            elif category == "energy":
                if layer_name == "solar":
                    # Solar data is a raster (array, extent)
                    data_array, extent = layer_info["data"]
                    m = add_solar_layer(
                        m,
                        data_array,
                        extent,
                        layer_info["config"]["name"],
                        layer_info["config"]["opacity"],
                    )
                    print(f"    ✓ Solar layer added", flush=True)

                elif layer_name == "geothermal":
                    # Geothermal data is a DataFrame
                    m = add_geothermal_layer(
                        m,
                        layer_info["data"],
                        layer_info["config"]["name"],
                        layer_info["config"]["opacity"],
                    )
                    print(f"    ✓ Geothermal layer added", flush=True)

        except Exception as e:
            logger.log_map_render_error(e)
            print(f"    ❌ Error adding layer: {e}", flush=True)
            import traceback

            traceback.print_exc()

    # Add dynamic legend
    print("Adding dynamic legend...", flush=True)
    m = add_legend_to_map(m, layer_data)
    print("  ✓ Legend added", flush=True)

    # Add layer control
    import folium

    print("Adding layer control...", flush=True)
    folium.LayerControl().add_to(m)

    # Log successful map render
    logger.log_map_render_success()

    # Cache the map and update session state
    st.session_state.cached_map = m
    st.session_state.last_active_layers = set(active_layers.keys())
    print("  ✓ Map cached in session state", flush=True)

else:
    # Use cached map
    print("Using cached map (no layer changes detected)...", flush=True)
    m = st.session_state.cached_map

# Display map with optimized settings
print("Rendering map to Streamlit...", flush=True)
try:
    st_folium(
        m,
        width=settings["map"]["width"],
        height=settings["map"]["height"],
        returned_objects=[],  # Don't return map state on interactions
        key="cdr_map",  # Static key prevents re-rendering
    )
    print("  ✓ Map rendered successfully", flush=True)
except Exception as e:
    logger.log_map_render_error(e)
    print(f"  ✗ Error rendering map: {e}", flush=True)
    import traceback

    traceback.print_exc()
    st.error(f"Error displaying map: {e}")
    raise

# Methodology Section
st.markdown("---")
st.markdown("## Methodology", unsafe_allow_html=True)
st.markdown('<a id="methodology"></a>', unsafe_allow_html=True)

st.markdown("""
This interactive map visualizes global CDR (Carbon Dioxide Removal) storage potential and
energy resources based on peer-reviewed geological and climate datasets. Below we detail our
data sources, processing methodology, and geological classification criteria.
""")

# Storage Potential Section
st.markdown("### 🗺️ Geological Storage Potential")

st.markdown("""
**Data Source:** Global Lithological Map (GLiM)
**Reference:** Hartmann, J., & Moosdorf, N. (2012). *The new global lithological map database GLiM:
A representation of rock properties at the Earth surface.* Geochemistry, Geophysics, Geosystems, 13(12).
**DOI:** [10.1029/2012GC004370](https://doi.org/10.1029/2012GC004370)
""")

# Mineralization Storage Details
with st.expander("Mineralization Storage"):
    st.markdown("""
    #### GLiM Classification System

    GLiM uses a three-level hierarchical lithology classification:
    - **Level 1 (xx)**: Dominant lithology type
    - **Level 2 (yy)**: Subclass refinement (optional)
    - **Level 3 (zz)**: Special attributes (optional)

    Format: `xxyyzz` where `__` indicates "not specified"

    #### Our Filtering Approach: Tiered Classification

    We implement a **tiered filtering system** that selects formations based on their potential for
    CO₂ mineralization (reaction with Ca-Mg-Fe minerals to form stable carbonates).

    **Tier 1: Highest Potential** 🟢
    - **Code:** `vb` (Basic volcanic rocks - basalt)
    - **Exclusions:** Pyroclastic rocks (`py` subcode)
    - **Rationale:** Basalts have high reactivity and abundant Ca-Mg-Fe minerals. Excluded pyroclastics
      have variable properties unsuitable for reliable storage.
    - **Global extent:** ~67,000 formations

    **Tier 2: Moderate Potential** 🟡
    - **Codes:**
      - `pb` (Basic plutonic rocks - gabbro, diorite)
      - `vi` (Intermediate volcanic rocks)
    - **Exclusions:** Pyroclastics for `vi`
    - **Rationale:** Good mineralization potential but slower reaction kinetics than basalt
    - **Global extent:** ~42,000 formations

    **Tier 3: Case-by-Case Assessment** 🟠
    - **Code:** `mt` (Metamorphic rocks)
    - **Requirements:** Only amphibolite (`am`) or pyroxenite (`pu`) subclasses
    - **Rationale:** These metamorphic rocks indicate mafic/ultramafic protoliths that may contain
      serpentinite—excellent for carbonation. Generic metamorphics excluded as they may be felsic origin.
    - **Global extent:** ~24,000 formations meeting criteria
    - **Note:** Requires site-specific geochemical assessment

    #### Explicitly Excluded Formations

    | Code | Type | Reason |
    |------|------|--------|
    | `va` | Acidic volcanic (rhyolite, dacite) | Low Ca-Mg-Fe content; poor mineralization potential |
    | `pa` | Acidic plutonic (granite) | Low reactivity; insufficient divalent cations |
    | `pi` | Intermediate plutonic | Lower reactivity than basic rocks |
    | `py` | Pyroclastic subclass | Variable properties; not typical reservoir rocks |

    #### Processing Steps

    1. **Load GLiM geodatabase** (1.2M global geological features)
    2. **Apply tiered filters** at primary (xx) and subclass (yy) levels
    3. **Exclude unsuitable subclasses** (pyroclastics, acidic compositions)
    4. **Reproject to WGS84** for web mapping
    5. **Simplify geometries** (0.05° tolerance) for performance
    6. **Rasterize to 0.05° grid** (~5.5km at equator) for rendering
    7. **Reproject to Web Mercator** for accurate map display
    """)

# Sedimentary Storage Details
with st.expander("Sedimentary Storage"):
    st.markdown("""
    #### Our Filtering Approach: Conservative Selection

    We select only high-quality sedimentary formations suitable for CO₂ injection into deep saline
    aquifers (>800m depth required, though depth is not mapped here).

    **Selected Formations** 🟢
    - **Code:** `ss` (Siliciclastic sedimentary - sandstone)
    - **Rationale:** Sandstones provide excellent porosity (15-30%) and permeability, making them
      ideal for CO₂ injection. These are well-characterized formations with predictable reservoir
      properties.
    - **Global extent:** ~41,000 formations

    While we use a tiered approach for mineralization storage, we apply a conservative,
    single-tier filter for sedimentary storage because tier 2 formations are too extensive:
    Including unconsolidated sediments (`su`) and carbonates (`sc`) would show potential
    storage almost everywhere globally, making the map less useful for identifying priority areas.

    As with mineralization, surface geology alone cannot distinguish between shallow/deep
    sediments or assess consolidation state—critical factors for storage viability.

    #### Formations NOT Shown (But May Have Potential)

    | Code | Type | Why Not Shown |
    |------|------|---------------|
    | `su` | Unconsolidated sediments | Too extensive globally; depth/consolidation unknown |
    | `sc` | Carbonate sedimentary | May have good storage potential but requires detailed assessment |
    | `sm` | Mixed sedimentary | Too heterogeneous; insufficient data for storage assessment |

    #### Processing Steps

    1. **Load GLiM geodatabase** (1.2M global geological features)
    2. **Filter for sandstone formations** (`ss` primary code only)
    3. **Reproject to WGS84** for web mapping
    4. **Simplify geometries** (0.05° tolerance) for performance
    5. **Rasterize to 0.05° grid** (~5.5km at equator) for rendering
    6. **Reproject to Web Mercator** for accurate map display

    #### Limitations

    - **Surface geology only:** Map shows surface/near-surface formations; subsurface assessment required
    - **Depth not included:** Actual storage requires formations >800m depth
    - **No porosity/permeability data:** Site-specific reservoir characterization needed
    - **Seal integrity not assessed:** Requires caprock evaluation
    - **Conservative filter:** Many viable storage sites may exist in formations not shown (e.g., deep carbonates, consolidated sediments)
    """)

# Sedimentary Thickness Details
with st.expander("Sedimentary Thickness"):
    st.markdown("""
    **Data Source:** CDR Primer Chapter 3
    **Reference:** Pilorge, H., et al. (2020). *Cost analysis and carbon removal potential of
    geological carbon storage.*

    #### Data Description

    Global sediment thickness dataset at 3km resolution (interpolated from point measurements and
    regional models).

    #### Processing Steps

    1. **Load GeoTIFF raster** (global sediment thickness in meters)
    2. **Apply minimum threshold:** Mask out areas <500m thickness (insufficient for viable storage)
    3. **Downsample by 3x** (3km → ~9km effective resolution) for performance
    4. **Reproject to Web Mercator** for map display
    5. **Apply color ramp:**
       - Light yellow: 500-1,000m (marginal)
       - Orange: 1,000-2,000m (moderate)
       - Dark red: >2,000m (excellent)

    #### Interpretation

    Thicker sedimentary basins generally indicate:
    - Greater storage capacity potential
    - More extensive aquifer systems
    - Better developed seal formations

    **Note:** Thickness alone does not guarantee storage suitability. Requires assessment of
    lithology, depth, porosity, and structural trapping.
    """)

# Energy Resources Section
st.markdown("### ⚡ Energy Resources")

# Solar Details
with st.expander("Solar Photovoltaic Potential"):
    st.markdown("""
    **Data Source:** Global Solar Atlas 2.0 (Solargis)
    **URL:** [globalsolaratlas.info](https://globalsolaratlas.info/)

    #### Data Description

    Photovoltaic power output potential (kWh/kWp/day) - the daily electricity generation per
    kilowatt-peak of installed PV capacity.

    #### Processing Steps

    1. **Load GeoTIFF** (PVOUT long-term yearly average)
    2. **Downsample by 10x** for web performance
    3. **Apply graduated color scale:**
       - 2.0-3.0 kWh/kWp/day: Poor (high latitudes)
       - 3.0-4.0: Moderate
       - 4.0-5.0: Good
       - 5.0-6.0: Excellent
       - >6.0: Outstanding (deserts, low latitudes)
    4. **Reproject to Web Mercator**

    #### Relevance to CDR

    Solar resources are critical for powering energy-intensive CDR methods:
    - Direct Air Capture (DAC) facilities
    - Enhanced weathering crushing/transport
    - Mineralization plant operations

    **Note:** Areas with >4.5 kWh/kWp/day provide economical solar energy for CDR applications.
    """)

# Geothermal Details
with st.expander("Geothermal Potential"):
    st.markdown("""
    **Data Source:** International Heat Flow Commission (IHFC) Global Heat Flow Database 2024
    **URL:** [ihfc-iugg.org/products/global-heat-flow-database](https://ihfc-iugg.org/products/global-heat-flow-database)

    #### Data Description

    Point measurements of terrestrial heat flow (mW/m²) from boreholes, mines, and tunnels worldwide.

    #### Processing & Interpolation

    1. **Load heat flow point data** (~35,000 measurements globally)
    2. **Calculate drilling depth to 150°C:**
       - Target temperature: 150°C (minimum for efficient EGS)
       - Surface temperature: 15°C (global average)
       - Thermal gradient: (Heat flow) / (Thermal conductivity)
       - Depth = (150 - 15) / gradient
    3. **Interpolate to grid** using Inverse Distance Weighting (IDW)
    4. **Classify into tiers:**
       - <4 km depth: Excellent (conventional geothermal)
       - 4-6 km: Good (enhanced geothermal systems)
       - 6-8 km: Moderate (deep EGS)
       - >8 km: Poor (economically challenging)

    #### Relevance to CDR

    Geothermal energy can power:
    - Direct Air Capture thermal processes
    - Mineralization heating requirements
    - Process heat for carbonate calcination

    **Note:** Areas with <6 km drilling depth to 150°C are economically viable for EGS-powered CDR.
    """)

# Data Quality & Limitations
st.markdown("### ⚠️ Data Quality & Limitations")

st.markdown("""
**Spatial Resolution:**
- GLiM geological data: Variable (1:1M to 1:5M scale depending on region)
- Sediment thickness: ~9km effective resolution
- Solar data: ~11km (downsampled from 250m)
- Geothermal: Sparse point measurements

**Temporal Coverage:**
- GLiM: Published 2012 (based on pre-2010 geological surveys)
- Solar: 1999-2018 long-term averages
- Geothermal: Measurements from 1950s-2024
- Sediment thickness: Based on models as of 2020

**Known Gaps:**
- Geothermal data sparse in Africa, South America, and parts of Asia
- No subsurface geological data (depth, stratigraphy)
- No economic or regulatory feasibility assessment

**Important Notes:**
- This map shows *potential* resource distribution, not *viable* CDR project sites
- Site-specific assessment required for any deployment
- Depth to suitable formations not shown (critical for CO₂ storage)
- Infrastructure, land use, and permitting not considered
- Does not account for competing land/resource uses
""")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit by Deep Sky Research*")

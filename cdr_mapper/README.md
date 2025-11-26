# CDR Mapper 🌍

Interactive tool for exploring carbon removal resource potential globally.

## Overview

CDR Mapper is an internal tool for visualizing and analyzing geospatial data related to carbon dioxide removal (CDR) potential. It provides interactive maps showing:

- **Storage Potential**: Geological formations suitable for CO2 storage
  - Basaltic & ultramafic rocks (mineralization)
  - Sedimentary basins (porous storage)

- **Energy Resources**: Renewable energy potential for powering CDR operations
  - Solar photovoltaic potential
  - Enhanced geothermal systems

## Quick Start

### Prerequisites

- Python 3.9+
- Data files in `../data/cdr_mapper/` directory (see Data Setup below)

### Installation

```bash
# Navigate to cdr_mapper directory
cd datascience-platform/cdr_mapper

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Data Setup

### Directory Structure

Ensure data files are organized in the shared data directory:

```
datascience-platform/
├── data/
│   └── cdr_mapper/
│       ├── storage/
│       │   └── glim/
│       │       └── LiMW_GIS 2015.gdb
│       │   └── pilorge/
│       │       └── Primer-data-sharing/
│       └── energy/
│           ├── solar/
│           │   └── World_PVOUT_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/
│           │       └── PVOUT.tif
│           └── geothermal/
│               └── IHFC_global/
│                   └── IHFC_2024_GHFDB.xlsx
└── cdr_mapper/
    └── (this directory)
```

### Data Sources

1. **GLiM (Global Lithological Map)**
   - Source: Hartmann & Moosdorf (2012)
   - File: `LiMW_GIS 2015.gdb`
   - Place in: `../data/cdr_mapper/storage/glim/`

2. **Global Solar Atlas**
   - Source: Solargis
   - File: `PVOUT.tif`
   - Place in: `../data/cdr_mapper/energy/solar/World_PVOUT_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/`

3. **IHFC Global Heat Flow Database**
   - Source: International Heat Flow Commission
   - File: `IHFC_2024_GHFDB.xlsx`
   - Place in: `../data/cdr_mapper/energy/geothermal/IHFC_global/`
   
4. **CDR Primer Chapter 3**
   - Source: CDR Primer - Hélène Pilorgé
   - Directory: `Primer-data-sharing/`
   - Place in: `../data/cdr_mapper/storage/pilorge/`

## Architecture

```
cdr_mapper/
├── config/              # YAML configuration files
│   ├── layers.yaml     # Data layer definitions
│   └── settings.yaml   # App settings
├── data/
│   ├── cache/          # Processed data cache
├── processing/         # Data processing utilities
│   ├── rasterize.py    # Rasterization and projection handling
│   └── loaders/        # Data loading modules
├── visualization/      # Map and chart creation
├── utils/              # Helper functions
└── app.py              # Main Streamlit application
```

### Projection System (CRITICAL)

**Web maps use Web Mercator projection (EPSG:3857)**. To ensure proper alignment:

1. **Vector data** is loaded in EPSG:4326 (WGS84 lat/lon)
2. **Rasterized** at 0.5° resolution in EPSG:4326
3. **Reprojected** to Web Mercator (EPSG:3857) 
4. **Latitude clipped** to ±85.05° (Web Mercator's valid range)
5. **Displayed** on Web Mercator base map

**⚠️ DO NOT:**
- Set `crs='EPSG4326'` on the base map (causes tile misalignment)
- Skip the Web Mercator reprojection (causes vertical stretching)
- Attempt to "fix" alignment by flipping arrays (not the issue)

**Why this matters:**
Web Mercator has vertical distortion that increases with latitude. Without reprojection:
- Iceland's geology appears north of Siberia
- Australia's features appear in the wrong ocean
- Only equatorial data aligns correctly

See `processing/rasterize.py` for implementation details.

## Adding New Data Layers

1. **Add configuration** to `config/layers.yaml`:

```yaml
layers:
  energy:
    wind:  # New layer
      name: "Wind Potential"
      description: "Wind power potential"
      data_path: "energy/wind/wind_speed.tif"
      loader: "energy.WindLoader"
      color: "#1E88E5"
      opacity: 0.6
      enabled_by_default: true
```

2. **Create loader** in `processing/loaders/`:

```python
class WindLoader(DataLoader):
    def load(self, use_cache=True):
        # Implementation
        pass
```

## Troubleshooting

### "Data file not found" error

Ensure data files are in the correct location (`../data/`). Check file paths in `config/layers.yaml`.

### Cache issues

Delete cache and reload:
```bash
rm -rf data/cache/*
```

### Slow loading

First load downloads and processes data (can take 2-5 minutes). Subsequent loads use cache and are fast (<10 seconds).

## Development

### Running tests

```bash
pytest tests/
```

## Data Attribution

- **Geological Data**: Hartmann, J., & Moosdorf, N. (2012). Global Lithological Map Database
- **Solar Data**: Global Solar Atlas 2.0, Solargis
- **Geothermal Data**: IHFC Global Heat Flow Database 2024
- **Sediment Thickness**: Pilorgé et al. (2021). CDR Primer Chapter 3

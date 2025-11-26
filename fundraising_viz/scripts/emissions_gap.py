import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import datetime

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot

# Import data loading functions from cdr_scaleup.py
from cdr_scaleup import (
    load_historical_emissions,
    load_2c_pathway_from_pgr,
    load_pgr_fossil_fuel_data,
    join_historical_to_projections,
    create_2c_pathway_series,
    interpolate_with_cubic_spline
)

def load_global_emissions_data():
    """
    Load global emissions data directly from source files.
    Uses the same data loading approach as cdr_scaleup.py.
    Returns historical emissions and pathway data in format expected by plotting functions.
    """
    # Load source data
    historical_emissions = load_historical_emissions()
    pathway_2c_dict = load_2c_pathway_from_pgr()
    fossil_fuel_proj = load_pgr_fossil_fuel_data()

    # Join historical to projections
    historical_adjusted = join_historical_to_projections(historical_emissions, fossil_fuel_proj)

    # Create complete year range (2000-2050)
    all_years = list(range(2000, 2051))
    df = pd.DataFrame({'year': all_years})

    # Add historical emissions (2000-2023)
    df = df.merge(historical_adjusted, on='year', how='left')

    # Add fossil fuel breakdown for projection years
    df = df.merge(
        fossil_fuel_proj[['year', 'coal_gtco2', 'oil_gtco2', 'gas_gtco2']],
        on='year',
        how='left'
    )

    # For projection years (2024-2050), calculate total emissions from fossil fuel sum
    # Only calculate where we have fossil fuel data (not for all years)
    for idx, row in df.iterrows():
        year = row['year']
        if year >= 2024:
            df.loc[idx, 'global_emissions'] = (
                row['coal_gtco2'] + row['oil_gtco2'] + row['gas_gtco2']
            )

    # Convert to Gt
    df['emissions_gt'] = df['global_emissions'] / 1e9

    # Interpolate projected emissions to fill gaps between years (2024-2050)
    # This creates a smooth curve through the available data points
    df = interpolate_with_cubic_spline(df, 'emissions_gt')

    # Create emissions dataframe
    emissions_df = df[df['emissions_gt'].notna()][['year', 'emissions_gt']].copy()

    # Add 2°C pathway (using cubic spline interpolation)
    df['2_degree_pathway'] = create_2c_pathway_series(df['year'].values, pathway_2c_dict)
    df['2_degree_pathway_gt'] = df['2_degree_pathway'] / 1e9

    # For 1.5°C pathway, we'll load it from the PGR data
    # Read the 1.5C pathway from the same Excel file
    pgr_file = 'data/needed_removal_capacity/PGR2025_data.xlsx'
    df_pgr = pd.read_excel(pgr_file, sheet_name='Figure ES-1 and 2.1', header=None)

    # Row 4 (index 4) has years, Row 14 (index 14) has 1.5C median pathway
    years_row = df_pgr.iloc[4, 3:9]
    pathway_1_5_row = df_pgr.iloc[14, 3:9]

    pathway_1_5_dict = {}
    for i, year in enumerate(years_row.values):
        if not pd.isna(year) and not pd.isna(pathway_1_5_row.values[i]):
            pathway_1_5_dict[int(year)] = float(pathway_1_5_row.values[i]) * 1e9

    df['1_5_degree_pathway'] = create_2c_pathway_series(df['year'].values, pathway_1_5_dict)
    df['1_5_degree_pathway_gt'] = df['1_5_degree_pathway'] / 1e9

    # Pathway data (years where pathway values exist)
    pathway_df = df[df['2_degree_pathway_gt'].notna()][['year', '2_degree_pathway_gt', '1_5_degree_pathway_gt']].copy()

    return emissions_df, pathway_df


def load_canada_emissions_data(csv_path):
    """
    Load Canada emissions data from 440 Megatonnes data provided here: https://dashboard.440megatonnes.ca/?_gl=1*1vk8sh9*_gcl_au*MTUyNzgzNDIzNy4xNzU4NDY3MTAx*_ga*MTA5NTAyMzkzNC4xNzU4NDY3MTAw*_ga_DVTX0HL4Z5*czE3NTg0NjcxMDAkbzEkZzEkdDE3NTg0NjczNTEkajYwJGwwJGgw
    The early estimate of 2024 emissions comes from here: https://440megatonnes.ca/early-estimate-of-national-emissions/
    The numbers on the second page are higher than the first page, but we don't get all years just the IPCC selected years. 
    So to add 2024 data I took the % change in emissions from the second page and applied it to the first page's 2023 emissions total. 
    """
    df = pd.read_excel(csv_path, sheet_name='Data')

    # Filter for national sector data
    national_data = df[df['sector'] == 'national'].copy()

    # Filter for years 2005-2023 (historical data)
    historical_data = national_data[(national_data['year'] >= 2005) & (national_data['year'] <= 2023)]

    # Select only year and ghg columns, and ensure unique years
    emissions_df = historical_data[['year', 'ghg']].copy()
    emissions_df.rename(columns={'ghg': 'emissions_mt'}, inplace=True)
    emissions_df = emissions_df.drop_duplicates(subset=['year']).reset_index(drop=True)

    # Add 2024 value (0.1% higher than 2023)
    emissions_2023 = emissions_df[emissions_df['year'] == 2023]['emissions_mt'].iloc[0]
    emissions_2024 = emissions_2023 * 1.001  # 0.1% increase

    # Add 2024 row
    new_row = pd.DataFrame({'year': [2024], 'emissions_mt': [emissions_2024]})
    emissions_df = pd.concat([emissions_df, new_row], ignore_index=True)

    # MDK I got Canada's emissions projections from the Canada's Energy Future Report which is also what the Production Gap Report Cites
    # https://www.cer-rec.gc.ca/en/data-analysis/canada-energy-future/2023/canada-energy-futures-2023.pdf
    # Figure R.3 has total emissions for all scenarios until 2050.
    # We select "Current Measures" 
    canada_emissions_2030 = 613.41
    canada_emissions_2050 = 567.27
    canada_projected_emissions = pd.DataFrame({'year': [2030, 2050], 'emissions_mt': [canada_emissions_2030, canada_emissions_2050]})
    emissions_df = pd.concat([emissions_df, canada_projected_emissions], ignore_index=True)


    # Load NZP pathway data (2025-2050)
    nzp_data = national_data[national_data['scenario'].isin(['NZP_lower', 'NZP_upper'])]
    nzp_data = nzp_data[(nzp_data['year'] >= 2025) & (nzp_data['year'] <= 2050)]

    # Pivot NZP data to get upper and lower bounds
    nzp_pivot = nzp_data.pivot(index='year', columns='scenario', values='ghg')
    nzp_pivot = nzp_pivot.reset_index()
    nzp_pivot.columns.name = None

    return emissions_df, nzp_pivot


def load_all_cams_wildfire_emissions(data_dir, shapefile_path, cache_dir='data/wildfire_emissions'):
    """
    Load and process all CAMS wildfire emissions data for Canada from 2005.
    Each file contains 1 year of data based on filename (e.g., 2005.nc has 2005 data).
    Uses CSV cache to speed up subsequent runs.
    Returns a DataFrame with year and wildfire_emissions columns.
    """
    from rasterio.transform import from_bounds
    from rasterio.features import rasterize
    from shapely import affinity
    import glob
    import os
    import re

    # Check for cached CSV files first
    cache_file = os.path.join(cache_dir, 'canada_cams_wildfire_emissions.csv')
    provincial_cache_file = os.path.join(cache_dir, 'canada_cams_wildfire_emissions_provincial.csv')

    if os.path.exists(cache_file):
        print(f"Loading cached wildfire emissions data from {cache_file}")
        return pd.read_csv(cache_file)

    print("No cache found. Processing CAMS NetCDF files...")

    # Get all CAMS files
    cams_files = glob.glob(os.path.join(data_dir, 'cams_wildfire_emissions_*.nc'))
    cams_files.sort()

    print(f"Processing {len(cams_files)} CAMS files for Canada wildfire emissions...")

    # Load Canada shapefile once
    canada_gdf = gpd.read_file(shapefile_path)
    canada_gdf_360 = canada_gdf.copy()
    canada_gdf_360['geometry'] = canada_gdf_360['geometry'].apply(
        lambda geom: affinity.translate(geom, xoff=360)
    )
    bounds = canada_gdf_360.total_bounds

    yearly_emissions = {}
    provincial_emissions = []

    # Process each file
    for filepath in cams_files:
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")

        try:
            # Extract year from filename (e.g., "2005" from "cams_wildfire_emissions_2005.nc")
            year_match = re.search(r'(\d{4})\.nc$', filename)
            if year_match:
                year = int(year_match.group(1))
                years_to_process = [year]
            else:
                print(f"  Could not parse year from {filename}")
                continue

            # Load the dataset once
            ds = xr.open_dataset(filepath)
            # Subset to Canada region once
            ds_canada = ds.sel(
                longitude=slice(bounds[0], bounds[2]),
                latitude=slice(bounds[3], bounds[1])
            )

            # Create Canada mask once per file
            transform = from_bounds(
                ds_canada.longitude.min().values, ds_canada.latitude.min().values,
                ds_canada.longitude.max().values, ds_canada.latitude.max().values,
                len(ds_canada.longitude), len(ds_canada.latitude)
            )

            canada_mask = rasterize(
                canada_gdf_360.geometry,
                out_shape=(len(ds_canada.latitude), len(ds_canada.longitude)),
                transform=transform, fill=0, default_value=1, dtype='uint8'
            ).astype(bool)

            # Calculate grid cell areas once per file
            _, lat_grid = np.meshgrid(ds_canada.longitude.values, ds_canada.latitude.values)
            lat_rad = np.radians(lat_grid)
            cell_areas = (6371000 ** 2) * np.radians(0.1) ** 2 * np.cos(lat_rad)

            # Process the year in this file
            year = years_to_process[0]

            # Calculate national emissions for this year
            emissions_masked = ds_canada.co2fire.values * canada_mask * cell_areas
            daily_totals = np.nansum(emissions_masked, axis=(1, 2)) * 86400
            total_emissions_mt = np.sum(daily_totals) / 1e9 * GWIS_CALIBRATION_FACTOR

            yearly_emissions[year] = total_emissions_mt

            # Calculate provincial emissions for this year
            for _, province in canada_gdf_360.iterrows():
                province_mask = rasterize(
                    [province.geometry],
                    out_shape=(len(ds_canada.latitude), len(ds_canada.longitude)),
                    transform=transform, fill=0, default_value=1, dtype='uint8'
                ).astype(bool)

                province_emissions_masked = ds_canada.co2fire.values * province_mask * cell_areas
                province_daily_totals = np.nansum(province_emissions_masked, axis=(1, 2)) * 86400
                province_total_mt = np.sum(province_daily_totals) / 1e9 * GWIS_CALIBRATION_FACTOR

                # Remove accents from province name for clean CSV output
                import unicodedata
                province_name = unicodedata.normalize('NFD', province['name']).encode('ascii', 'ignore').decode('ascii')

                provincial_emissions.append({
                    'year': year,
                    'province': province_name,
                    'wildfire_emissions': province_total_mt
                })

            ds.close()

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    # Convert to DataFrames
    wildfire_df = pd.DataFrame([
        {'year': year, 'wildfire_emissions': emissions}
        for year, emissions in yearly_emissions.items()
    ])
    wildfire_df = wildfire_df.sort_values('year').reset_index(drop=True)

    provincial_df = pd.DataFrame(provincial_emissions)
    provincial_df = provincial_df.sort_values(['year', 'province']).reset_index(drop=True)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    wildfire_df.to_csv(cache_file, index=False)
    provincial_df.to_csv(provincial_cache_file, index=False)
    print(f"Saved processed data to cache: {cache_file}")
    print(f"Saved provincial data to cache: {provincial_cache_file}")

    return wildfire_df

def annotate_emissions_gap(ax, year, projected_emissions, pathway_emissions, units='Gt'):
    """
    Annotate the emissions gap between current trajectory and pathways for a given year.
    """

    gap_2_degree = projected_emissions - pathway_emissions

    # Calculate offset based on scale of emissions
    offset = max(projected_emissions, pathway_emissions) * 0.02

    # Draw vertical lines showing the gaps
    ax.plot([year, year], [projected_emissions - offset, pathway_emissions + offset],
            color='#E74C3C', linewidth=2, alpha=0.8)

    ax.plot([year - 0.5, year + 0.5], [projected_emissions - offset, projected_emissions - offset],
            color='#E74C3C', linewidth=2, alpha=0.8)

    ax.plot([year - 0.5, year + 0.5], [pathway_emissions + offset, pathway_emissions + offset],
            color='#E74C3C', linewidth=2, alpha=0.8)

    # Add gap annotation
    midpoint = (pathway_emissions + projected_emissions) / 2
    gap_text = f'{gap_2_degree:.0f} {units} OVER\nBY {year}'

    ax.annotate(gap_text,
                xy=(year - 5, midpoint), xytext=(year - 5, midpoint),
                fontsize=10, ha='left', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='#E74C3C', alpha=0.9))


def create_emissions_gap_plot(emissions_df, pathway_df):
    """
    Create global emissions gap visualization with 1.5 and 2 degree pathways.
    Similar to Canada plot but using global pathways instead of NZP.
    Flexible to handle different column naming conventions:
    - emissions_df: 'emissions_gt' or 'emissions_mt'
    - pathway_df: ('2_degree_pathway_gt', '1_5_degree_pathway_gt') or ('NZP_lower', 'NZP_upper')
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 8))

    # Detect column names and units
    # Check emissions column
    if 'emissions_gt' in emissions_df.columns:
        emissions_col = 'emissions_gt'
        units = 'Gt'
    elif 'emissions_mt' in emissions_df.columns:
        emissions_col = 'emissions_mt'
        units = 'Mt'
    else:
        raise ValueError("emissions_df must contain either 'emissions_gt' or 'emissions_mt' column")

    # Check pathway columns and validate consistency
    if 'NZP_lower' in pathway_df.columns and 'NZP_upper' in pathway_df.columns:
        if units != 'Mt':
            raise ValueError("When pathway_df has NZP columns, emissions_df must have 'emissions_mt' column")
        pathway_lower_col = 'NZP_lower'
        pathway_upper_col = 'NZP_upper'
        pathway_type = 'nzp'
    elif '2_degree_pathway_gt' in pathway_df.columns and '1_5_degree_pathway_gt' in pathway_df.columns:
        if units != 'Gt':
            raise ValueError("When pathway_df has degree pathway columns, emissions_df must have 'emissions_gt' column")
        pathway_lower_col = '2_degree_pathway_gt'
        pathway_upper_col = '1_5_degree_pathway_gt'
        pathway_type = 'degree'
    else:
        raise ValueError("pathway_df must contain either ('NZP_lower', 'NZP_upper') or ('2_degree_pathway_gt', '1_5_degree_pathway_gt') columns")

    # Split historical and projected emissions at 2025
    # Historical goes up to and includes 2024, projected starts from 2025
    # But we need to include 2024 in projected to ensure line connection
    historical_df = emissions_df[emissions_df['year'] <= 2024].copy()
    projected_df = emissions_df[emissions_df['year'] >= 2024].copy()
    pathway_df = pathway_df[pathway_df['year']>=2030]

    # Plot historical data (solid line)
    ax.plot(historical_df['year'], historical_df[emissions_col],
            color='#2C3E50', linewidth=3, marker='o', markersize=6, solid_capstyle='round')

    # Plot projected data (dashed line)
    if len(projected_df) > 0:
        ax.plot(projected_df['year'], projected_df[emissions_col],
                color='#2C3E50', linewidth=3, linestyle='--', marker='', markersize=6,
                solid_capstyle='round', alpha=0.7)

    # Plot pathway data based on type
    if pathway_type == 'degree':
        ax.plot(pathway_df['year'], pathway_df[pathway_lower_col],
                color='#F39C12', linewidth=3, marker='o', markersize=6,
                solid_capstyle='round')

        ax.plot(pathway_df['year'], pathway_df[pathway_upper_col],
            color='#27AE60', linewidth=3, marker='o', markersize=6,
            solid_capstyle='round')
        
    else:  # nzp type
        # Fill between upper and lower bounds for NZP
        ax.fill_between(pathway_df['year'], pathway_df[pathway_lower_col], pathway_df[pathway_upper_col],
                       color='#3498DB', alpha=0.3)

        # Plot the bounds
        ax.plot(pathway_df['year'], pathway_df[pathway_lower_col],
                color='#3498DB', linewidth=2, linestyle='--', alpha=0.8)
        ax.plot(pathway_df['year'], pathway_df[pathway_upper_col],
                color='#3498DB', linewidth=2, linestyle='--', alpha=0.8)

    ax.set_xlim(2000, 2050 + 0.5)

    # Calculate max for y-axis
    max_emissions = emissions_df[emissions_col].max()
    ax.set_ylim(0, max_emissions * 1.1)

    # X-axis formatting
    ax.set_xticks(range(2000, 2051, 10))
    ax.tick_params(axis='both', labelsize=12)

    # Add pathway-specific annotations
    if pathway_type == 'degree':
        # Position annotations based on data scale
        pathway_2_2040 = pathway_df[pathway_df['year'] == 2040][pathway_lower_col].iloc[0] if len(pathway_df[pathway_df['year'] == 2040]) > 0 else max_emissions * 0.6
        pathway_1_5_2040 = pathway_df[pathway_df['year'] == 2040][pathway_upper_col].iloc[0] if len(pathway_df[pathway_df['year'] == 2040]) > 0 else max_emissions * 0.3

        ax.text(2015, max_emissions*.9, 'PAST\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', color='#2C3E50')

        ax.text(2040, max_emissions, 'PROJECTED EMISSIONS',
                fontsize=12, ha='center', va='center',
                fontweight='bold', alpha=0.6, color='#2C3E50')

        ax.text(2040, pathway_2_2040 + (max_emissions * 0.075), '2°C\nPATHWAY',
                fontsize=12, ha='center', va='center',
                fontweight='bold', alpha=0.7, color='#E67E22')

        ax.text(2040, pathway_1_5_2040 - (max_emissions * 0.075), '1.5°C\nPATHWAY',
                fontsize=12, ha='center', va='center',
                fontweight='bold', alpha=0.7, color='#27AE60')
        
        # Check if we have 2030 data
        if len(emissions_df[emissions_df['year'] == 2030]) > 0 and len(pathway_df[pathway_df['year'] == 2030]) > 0:
            projected_emissions_2030 = emissions_df[emissions_df['year'] == 2030][emissions_col].iloc[0]
            pathway_2_emissions_2030 = pathway_df[pathway_df['year'] == 2030][pathway_lower_col].iloc[0]
            annotate_emissions_gap(ax, 2030, projected_emissions_2030, pathway_2_emissions_2030, units)

        # Check if we have 2050 data
        if len(emissions_df[emissions_df['year'] == 2050]) > 0 and len(pathway_df[pathway_df['year'] == 2050]) > 0:
            projected_emissions_2050 = emissions_df[emissions_df['year'] == 2050][emissions_col].iloc[0]
            pathway_2_emissions_2050 = pathway_df[pathway_df['year'] == 2050][pathway_lower_col].iloc[0]
            annotate_emissions_gap(ax, 2050, projected_emissions_2050, pathway_2_emissions_2050, units)

    else:  # nzp type
        # For NZP, add annotation for the pathway range
        nzp_middle_2040 = (pathway_df[pathway_df['year'] == 2040][pathway_lower_col].iloc[0] +
                       pathway_df[pathway_df['year'] == 2040][pathway_upper_col].iloc[0]) / 2 if len(pathway_df[pathway_df['year'] == 2040]) > 0 else max_emissions * 0.3

        ax.text(2040, nzp_middle_2040, 'NET ZERO\nPATHWAY',
                fontsize=12, ha='center', va='center',
                fontweight='bold', alpha=0.7, color='#3498DB')
        
        ax.text(2020, max_emissions, 'HISTORICAL\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', color='#2C3E50')

        ax.text(2040, max_emissions * .9, 'PROJECTED EMISSIONS',
                fontsize=12, ha='center', va='center',
                fontweight='bold', alpha=0.6, color='#2C3E50')

                # Check if we have 2030 data
        if len(emissions_df[emissions_df['year'] == 2030]) > 0 and len(pathway_df[pathway_df['year'] == 2030]) > 0:
            projected_emissions_2030 = emissions_df[emissions_df['year'] == 2030][emissions_col].iloc[0]
            pathway_2_emissions_2030 = pathway_df[pathway_df['year'] == 2030][pathway_upper_col].iloc[0]
            annotate_emissions_gap(ax, 2030, projected_emissions_2030, pathway_2_emissions_2030, units)

        # Check if we have 2050 data
        if len(emissions_df[emissions_df['year'] == 2050]) > 0 and len(pathway_df[pathway_df['year'] == 2050]) > 0:
            projected_emissions_2050 = emissions_df[emissions_df['year'] == 2050][emissions_col].iloc[0]
            pathway_2_emissions_2050 = pathway_df[pathway_df['year'] == 2050][pathway_upper_col].iloc[0]
            annotate_emissions_gap(ax, 2050, projected_emissions_2050, pathway_2_emissions_2050, units)


    return fig


def main():
    
    # Canada

    emissions_file_path = 'data/canada_emissions_gap/canada_emissions_440Mt.xlsx'
    cams_data_dir = 'data/wildfire_emissions'
    canada_shapefile_path = 'data/canada_emissions_gap/ca_shp/ca.shp'
    canada_emissions_df, canada_nzp_df = load_canada_emissions_data(emissions_file_path)
    fig = create_emissions_gap_plot(canada_emissions_df, canada_nzp_df)
    format_plot_title(plt.gca(),
                        "",
                        "Canada's CO\N{SUBSCRIPT TWO}e Emissions (Mt)",
                        None)
    add_deep_sky_branding(plt.gca(), None,
                            "DATA: CANADIAN CLIMATE INSTITUTE, CANADA'S ENERGY FUTURE (2023).\nProjected emissions from CEF (2023) Current Measures scenario.",
                            analysis_date=datetime.datetime.now())
    save_path = 'figures/canada_emissions_gap_line.png'
    os.makedirs('figures', exist_ok=True)
    save_plot(fig, save_path)
    print(f"Canada emissions gap line plot saved to {save_path}")

    # Global

    emissions_df, pathway_df = load_global_emissions_data()
    fig = create_emissions_gap_plot(emissions_df, pathway_df)
    format_plot_title(plt.gca(),
                        "",
                        "GLOBAL CO\N{SUBSCRIPT TWO} EMISSIONS (GIGATONNES)",
                        None)
    add_deep_sky_branding(plt.gca(), None,
                            "DATA: GLOBAL CARBON PROJECT (2024), THE PRODUCTION GAP REPORT (2025)",
                            analysis_date=datetime.datetime.now())

    # Save the plot
    save_path = 'figures/global_emissions_gap.png'
    os.makedirs('figures', exist_ok=True)
    save_plot(fig, save_path)

    print(f"Global emissions gap plot saved to {save_path}")


if __name__ == "__main__":
    main()


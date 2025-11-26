import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
import os
import sys

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
reports_dir = os.path.join(os.path.dirname(project_root), 'reports')
sys.path.append(project_root)
sys.path.append(reports_dir)

from creds import gfw_api_key
from utils import (
    setup_enhanced_plot, format_plot_title, add_deep_sky_branding,
    save_plot, COLORS
)

# Data and figures directories
DATADIR = os.path.join(project_root, 'data/carbon_cycle')
FIGDIR = os.path.join(project_root, 'figures')
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# GFW API configuration
GFW_BASE_URL = 'https://data-api.globalforestwatch.org'
DATASET_ID = 'carbonflux_iso_change'
DATASET_VERSION = 'v20250515'


def get_country_iso_code(country_name):
    """
    Convert country name to ISO 3-letter code using the shapefile.

    Args:
        country_name: Name of country (e.g., 'Bolivia')

    Returns:
        ISO 3-letter code (e.g., 'BOL')
    """
    import geopandas as gpd

    shapefile_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'data/shapefiles/downloaded/world_countries.geojson'
    )

    gdf = gpd.read_file(shapefile_path)

    # Try different name columns
    for name_col in ['name', 'NAME', 'name_long', 'admin', 'sovereignt']:
        if name_col in gdf.columns:
            match = gdf[gdf[name_col].str.contains(country_name, case=False, na=False)]
            if len(match) > 0:
                # Get ISO code - try different columns
                for iso_col in ['iso_a3', 'ISO_A3', 'adm0_a3', 'sov_a3']:
                    if iso_col in match.columns:
                        iso_code = match.iloc[0][iso_col]
                        if iso_code and iso_code != '-99':
                            return iso_code

    raise ValueError(f"Could not find ISO code for country: {country_name}")


def query_gfw_emissions(iso_code, tree_cover_threshold=30):
    """
    Query GFW API for annual forest emissions data.

    Args:
        iso_code: ISO 3-letter country code
        tree_cover_threshold: Tree cover density threshold (default 30%)

    Returns:
        DataFrame with year and gross_emissions columns
    """
    headers = {'x-api-key': gfw_api_key}
    query_url = f'{GFW_BASE_URL}/dataset/{DATASET_ID}/{DATASET_VERSION}/query/csv'

    sql = f"""
    SELECT
        "umd_tree_cover_loss__year" as year,
        SUM("gfw_full_extent_gross_emissions_biomass_soil__Mg_CO2e") as gross_emissions
    FROM data
    WHERE "iso" = '{iso_code}'
    AND "umd_tree_cover_density_2000__threshold" = {tree_cover_threshold}
    GROUP BY "umd_tree_cover_loss__year"
    ORDER BY "umd_tree_cover_loss__year"
    """

    print(f"Querying emissions for {iso_code}...")
    response = requests.get(query_url, headers=headers, params={'sql': sql})

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    df = pd.read_csv(io.StringIO(response.text))
    return df


def query_gfw_annual_removals(iso_code, tree_cover_threshold=30):
    """
    Query GFW API for annual forest removals (constant across all years).

    Note: GFW removals represent "long-term average carbon uptake by intact
    and regrowing forests" rather than year-specific measurements. This is
    a constant value that should be applied to all years.

    Args:
        iso_code: ISO 3-letter country code
        tree_cover_threshold: Tree cover density threshold (default 30%)

    Returns:
        Annual removals value in Mg CO2/year
    """
    headers = {'x-api-key': gfw_api_key}

    # Use the summary dataset to get total annual removals
    summary_dataset = 'carbonflux_iso_summary'
    query_url = f'{GFW_BASE_URL}/dataset/{summary_dataset}/{DATASET_VERSION}/query/csv'

    sql = f"""
    SELECT
        SUM("gfw_full_extent_annual_removals__Mg_C") as annual_removals_C
    FROM data
    WHERE "iso" = '{iso_code}'
    AND "umd_tree_cover_density_2000__threshold" = {tree_cover_threshold}
    """

    print(f"Querying annual removals for {iso_code}...")
    response = requests.get(query_url, headers=headers, params={'sql': sql})

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    df = pd.read_csv(io.StringIO(response.text))

    # Convert from Mg C to Mg CO2 (multiply by 44/12 = molecular weight ratio)
    annual_removals_C = df['annual_removals_c'].values[0]
    annual_removals_CO2 = annual_removals_C * (44/12)

    print(f"  Annual removals: {annual_removals_CO2 / 1e6:.2f} Mt CO2/year (constant across all years)")

    return annual_removals_CO2


def retrieve_country_carbon_flux(country_name, tree_cover_threshold=30, region_name=None):
    """
    Retrieve forest carbon flux data for one or more countries and save to CSV.

    Args:
        country_name: Name of country (e.g., 'Bolivia') OR list of countries (e.g., ['Brazil', 'Bolivia'])
        tree_cover_threshold: Tree cover density threshold to define "forest" (default 30%)
        region_name: Optional name for multi-country regions (e.g., 'Amazon')

    Returns:
        Tuple of (output_file_path, list_of_countries)
    """
    # Handle both single country and list of countries
    if isinstance(country_name, str):
        countries = [country_name]
        is_multi_country = False
    else:
        countries = country_name
        is_multi_country = True

    # Display header
    if is_multi_country:
        display_name = region_name if region_name else f"{len(countries)} countries"
        print(f"\n{'='*70}")
        print(f"Retrieving forest carbon flux data for {display_name}")
        print(f"Countries: {', '.join(countries)}")
        print(f"Tree cover threshold: {tree_cover_threshold}% (defines 'forest')")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*60}")
        print(f"Retrieving forest carbon flux data for {countries[0]}")
        print(f"Tree cover threshold: {tree_cover_threshold}% (defines 'forest')")
        print(f"{'='*60}\n")

    # Query data for each country
    all_emissions = []
    total_removals = 0
    successful_countries = []
    failed_countries = []

    for country in countries:
        print(f"\n--- Processing {country} ---")

        try:
            # Get ISO code
            print(f"Looking up ISO code for {country}...")
            iso_code = get_country_iso_code(country)
            print(f"  ISO code: {iso_code}")

            # Query emissions (varies by year)
            df_emissions = query_gfw_emissions(iso_code, tree_cover_threshold)
            df_emissions['country'] = country  # Track which country for debugging
            all_emissions.append(df_emissions)

            # Query annual removals (constant value for all years)
            annual_removals_Mg_CO2 = query_gfw_annual_removals(iso_code, tree_cover_threshold)
            total_removals += annual_removals_Mg_CO2

            successful_countries.append(country)

        except Exception as e:
            error_msg = str(e)
            if "504" in error_msg or "timed out" in error_msg.lower():
                print(f"  ⚠️  API timeout for {country} (dataset too large)")
                print(f"  Skipping {country} and continuing with other countries...")
                failed_countries.append(country)
            else:
                # Re-raise other errors
                raise

    # Check if we got any successful countries
    if not successful_countries:
        raise Exception("Failed to retrieve data for all countries. Try with smaller countries or single countries.")

    if failed_countries:
        print(f"\n⚠️  Warning: Skipped {len(failed_countries)} countries due to API timeouts:")
        for country in failed_countries:
            print(f"    - {country}")
        print(f"Continuing with {len(successful_countries)} successful countries...")
        countries = successful_countries  # Update countries list for output

    # Aggregate emissions across all countries by year
    print(f"\n{'='*70}")
    print("Aggregating data across countries...")

    df_combined = pd.concat(all_emissions, ignore_index=True)
    df_aggregated = df_combined.groupby('year').agg({
        'gross_emissions': 'sum'
    }).reset_index()

    # Add total removals (sum across all countries)
    df_aggregated['gross_removals'] = total_removals

    # Convert from Mg (megagrams) to million tonnes CO2e
    print("Converting units to million tonnes CO2e...")
    df_aggregated['gross_emissions_Mt_CO2e'] = df_aggregated['gross_emissions'] / 1_000_000
    df_aggregated['gross_removals_Mt_CO2'] = df_aggregated['gross_removals'] / 1_000_000

    # Calculate net flux (emissions - removals)
    df_aggregated['net_flux_Mt_CO2e'] = df_aggregated['gross_emissions_Mt_CO2e'] - df_aggregated['gross_removals_Mt_CO2']

    # Select final columns
    df_final = df_aggregated[['year', 'gross_emissions_Mt_CO2e', 'gross_removals_Mt_CO2', 'net_flux_Mt_CO2e']].copy()

    # Round to 2 decimal places
    for col in ['gross_emissions_Mt_CO2e', 'gross_removals_Mt_CO2', 'net_flux_Mt_CO2e']:
        df_final[col] = df_final[col].round(2)

    # Determine output filename
    if is_multi_country:
        if region_name:
            filename_base = region_name.lower().replace(' ', '_')
        else:
            # Use concatenation of country names (truncate if too long)
            filename_base = '_'.join([c.lower().replace(' ', '_') for c in countries[:3]])
            if len(countries) > 3:
                filename_base += f'_plus_{len(countries)-3}_more'
    else:
        filename_base = countries[0].lower().replace(' ', '_')

    output_file = os.path.join(DATADIR, f'{filename_base}_forest_net_flux.csv')
    df_final.to_csv(output_file, index=False)

    # Display summary
    summary_name = region_name if (is_multi_country and region_name) else ', '.join(countries)
    print(f"\n{'='*70}")
    print(f"Results Summary for {summary_name}")
    print(f"{'='*70}")
    print(f"\nYears covered: {int(df_final['year'].min())} - {int(df_final['year'].max())}")
    print(f"Total years: {len(df_final)}")

    if is_multi_country:
        print(f"Countries included: {len(countries)}")
        for i, country in enumerate(countries, 1):
            print(f"  {i}. {country}")

    print(f"\nSample data (first 5 years):")
    print(df_final.head().to_string(index=False))
    print(f"\nSample data (last 5 years):")
    print(df_final.tail().to_string(index=False))

    print(f"\n{'='*70}")
    print(f"Data saved to: {output_file}")
    print(f"{'='*70}\n")

    return output_file, countries


def plot_forest_carbon_flux(data_file, countries, region_name=None, tree_cover_threshold=30):
    """
    Create a plot showing forest carbon flux over time with emissions bars,
    removals bars, and net flux line.

    Args:
        data_file: Path to CSV file with forest carbon flux data
        countries: List of country names included in the data
        region_name: Optional name for multi-country regions (e.g., 'Amazon')
        tree_cover_threshold: Tree cover threshold used for the data
    """
    # Load data
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Determine display name
    if region_name:
        display_name = region_name
    elif len(countries) == 1:
        display_name = countries[0]
    else:
        display_name = f"{len(countries)} countries"

    print(f"\n{'='*70}")
    print(f"Creating plot for {display_name}")
    print(f"Loaded data: {len(df)} years ({df['year'].min()}-{df['year'].max()})")
    print(f"{'='*70}\n")

    # Set up the plot with Deep Sky styling
    fig, ax, font_props = setup_enhanced_plot(figsize=(16, 10))

    # Define colors
    emissions_color = COLORS['primary']  # Red for emissions
    removals_color = COLORS['secondary']  # Green for removals
    net_flux_color = '#2C2C2C'  # Dark grey/black for net flux line

    # Bar width
    bar_width = 0.7

    # Plot emissions bars (positive values going up)
    ax.bar(df['year'], df['gross_emissions_Mt_CO2e'],
           width=bar_width,
           color=emissions_color,
           label='Gross emissions',
           alpha=0.85,
           zorder=2)

    # Plot removals bars (negative values going down)
    ax.bar(df['year'], -df['gross_removals_Mt_CO2'],
           width=bar_width,
           color=removals_color,
           label='Gross removals',
           alpha=0.85,
           zorder=2)

    # Plot net flux as a line
    ax.plot(df['year'], df['net_flux_Mt_CO2e'],
            color=net_flux_color,
            linewidth=3.5,
            marker='o',
            markersize=5,
            label='Net flux (emissions - removals)',
            zorder=3)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='#666666', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)

    # Format axes
    font_prop = font_props.get('regular') if font_props else None

    # X-axis: show every few years to avoid crowding
    year_ticks = df['year'].values
    if len(year_ticks) > 15:
        # Show every 3rd year
        year_labels = [str(int(y)) if i % 3 == 0 else '' for i, y in enumerate(year_ticks)]
    else:
        year_labels = [str(int(y)) for y in year_ticks]

    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, fontsize=13, fontproperties=font_prop)
    ax.set_xlabel('')

    # Y-axis
    # ax.set_ylabel('Million tonnes CO₂e per year', fontsize=16, fontproperties=font_prop, labelpad=15)
    ax.tick_params(axis='y', labelsize=13)

    # Format y-axis labels with comma separator
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Add legend
    font_prop_legend = font_props.get('regular') if font_props else None
    ax.legend(
        fontsize=14,
        frameon=True,
        facecolor=COLORS['background'],
        edgecolor='#DDDDDD',
        loc='upper left',
        prop=font_prop_legend
    )

    title = ''
    subtitle = f"{region_name.upper()} FOREST CARBON FLUX (MILLION TONNES CO2e)"
    format_plot_title(ax, title, subtitle, font_props)

    # Data note with country list
    latest_year = int(df['year'].max())

    country_list = ", ".join(countries)

    data_note = (
        f"SOURCE: Global Forest Watch (GFW) | WRI, {latest_year}\n"
        f"Includes data for {country_list} where tree cover is greater than {tree_cover_threshold}%."
    )
    add_deep_sky_branding(ax, font_props, data_note=data_note)

    # Save the plot
    filename_base = region_name.lower().replace(' ', '_') if region_name else countries[0].lower().replace(' ', '_')
    output_file = os.path.join(FIGDIR, f'{filename_base}_forest_flux.png')
    save_plot(fig, output_file)

    # Print summary
    print(f"{'='*70}")
    print(f"Plot Summary for {display_name}")
    print(f"{'='*70}")
    print(f"Years: {int(df['year'].min())} - {int(df['year'].max())}")
    print(f"Emissions range: {df['gross_emissions_Mt_CO2e'].min():.1f} - {df['gross_emissions_Mt_CO2e'].max():.1f} Mt CO₂e/year")
    print(f"Removals (constant): {df['gross_removals_Mt_CO2'].iloc[0]:.1f} Mt CO₂/year")
    print(f"Net flux range: {df['net_flux_Mt_CO2e'].min():.1f} to {df['net_flux_Mt_CO2e'].max():.1f} Mt CO₂e/year")

    # Find transition year
    sink_years = df[df['net_flux_Mt_CO2e'] < 0]
    source_years = df[df['net_flux_Mt_CO2e'] > 0]
    if len(sink_years) > 0 and len(source_years) > 0:
        transition_year = source_years['year'].min()
        print(f"\nTransition from sink to source: ~{int(transition_year)}")

    print(f"\nFigure saved to: {output_file}")
    print(f"{'='*70}\n")

    return fig


def main():
    """
    Main execution function.
    Retrieve and plot forest carbon flux data for specified country or region.
    """
    # =========================================================================
    # CONFIGURATION: Define your region/country here
    # =========================================================================

    # EXAMPLE 1: Single country
    # country = 'Bolivia'
    # region_name = None

    # EXAMPLE 2: Multiple countries (Amazon region - temporarily disabled due to API timeout for large datasets)
    country = [
        'Brazil',
        'Peru',
        'Colombia',
        'Venezuela',
        'Bolivia',
        'Ecuador',
        'Guyana',
        'Suriname'
    ]
    region_name = 'Amazon'

    # Tree cover threshold
    tree_cover_threshold = 30

    # =========================================================================
    # END CONFIGURATION
    # =========================================================================

    try:
        # Step 1: Retrieve data
        print("\n" + "="*70)
        print("STEP 1: RETRIEVING DATA")
        print("="*70)

        if isinstance(country, list):
            data_file, countries = retrieve_country_carbon_flux(
                country, tree_cover_threshold, region_name
            )
        else:
            data_file, countries = retrieve_country_carbon_flux(
                country, tree_cover_threshold
            )

        # Step 2: Create plot
        print("\n" + "="*70)
        print("STEP 2: CREATING PLOT")
        print("="*70)

        fig = plot_forest_carbon_flux(
            data_file, countries, region_name, tree_cover_threshold
        )

        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"Data file: {data_file}")
        print(f"Plot saved to figures/")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

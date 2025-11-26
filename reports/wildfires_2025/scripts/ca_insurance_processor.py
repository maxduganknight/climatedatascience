import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import geopandas as gpd
import matplotlib.colors as mcolors
import contextily as ctx
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
import pyproj


# Import shared utilities
from utils import (
    setup_space_mono_font, process_whp_risk_data, add_risk_categories,
    setup_enhanced_plot, format_plot_title, add_deep_sky_branding,
    plot_risk_lines, add_legend, save_plot, calculate_policy_count_growth,
    calculate_average_premium_by_risk_category, plot_lines_by_risk_category,
    filter_by_state_zip_codes,
    COLORS, RISK_COLORS
)

def process_non_renewal_data(raw_non_renewal_df):
    print("Processing non-renewal data...")
    clean_non_renewal_df = raw_non_renewal_df.copy()

    numeric_cols = ['new', 'renewed', 'non_renewed']
    for col in numeric_cols:
        # Check if the column is already numeric
        if pd.api.types.is_numeric_dtype(clean_non_renewal_df[col]):
            # Already numeric, no conversion needed
            continue
        else:
            # It's a string, so filter and convert
            clean_non_renewal_df = clean_non_renewal_df[clean_non_renewal_df[col].str.contains(r'^[\d,]+$', regex=True, na=False)]
            clean_non_renewal_df[col] = clean_non_renewal_df[col].str.replace(',', '').astype(int)

    # Filter out zip codes with less than 5 policies in force
    clean_non_renewal_df['policies_in_force'] = clean_non_renewal_df['new'] + clean_non_renewal_df['renewed']
    # clean_non_renewal_df = clean_non_renewal_df[clean_non_renewal_df['policies_in_force'] > 5]

    # Convert ZIP_CODE to string and ensure it's 5 digits
    clean_non_renewal_df['zip'] = clean_non_renewal_df['zip'].astype(str).str.zfill(5)

    clean_non_renewal_df = filter_by_state_zip_codes(
        clean_non_renewal_df,
        state_code='CA',
        zip_column='zip'
    )

    return clean_non_renewal_df

def process_roa_premiums_data(raw_roa_premiums_df, cpi):
    print("Processing ROA premiums data...")
    clean_roa_premiums_df = raw_roa_premiums_df.copy()

    # Look only at homeowner insurance
    clean_roa_premiums_df = clean_roa_premiums_df[clean_roa_premiums_df['POLICY_FORM'] == 'HO']
    
    # Exclude FAIR plan
    clean_roa_premiums_df = clean_roa_premiums_df[clean_roa_premiums_df['NAIC_CODE'] != 33665]

    # Filter out rows with missing premium or exposure data
    clean_roa_premiums_df = clean_roa_premiums_df.dropna(subset=['EARNED_PREMIUM', 'EARNED_EXPOSURE'])
    clean_roa_premiums_df = clean_roa_premiums_df[clean_roa_premiums_df['EARNED_EXPOSURE'] > 0]

    clean_roa_premiums_df['year'] = (2000 + pd.to_numeric(clean_roa_premiums_df['EXP_YEAR'], errors='coerce')).astype(int)

    # make zip code 5 digit string
    clean_roa_premiums_df['ZIP_CODE'] = clean_roa_premiums_df['ZIP_CODE'].astype(str).str.zfill(5)

    clean_roa_premiums_df = filter_by_state_zip_codes(
        clean_roa_premiums_df,
        state_code='CA',
        zip_column='ZIP_CODE'
    )

    # adjust for inflation. Calculate premiums in 2020 dollars
    # First, get the CPI value for 2020 as the reference point
    cpi_2020 = cpi[cpi['year'] == 2020]['cpi'].iloc[0]

    # Create a dictionary mapping each year to its inflation adjustment factor
    cpi_factors = {row['year']: cpi_2020 / row['cpi'] for _, row in cpi.iterrows()}

    # Apply the adjustment factor to each premium based on its year
    clean_roa_premiums_df['EARNED_PREMIUM_2020'] = clean_roa_premiums_df.apply(
        lambda row: row['EARNED_PREMIUM'] * cpi_factors.get(row['year'], 1.0), 
        axis=1
    )

    return clean_roa_premiums_df

def annotate_locations(ax, locations_dict, font_props=None, transform_coords=False):
    """
    Add location annotations to a map with consistent styling.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add annotations to
    locations_dict : dict
        Dictionary with location names as keys and (lat, lon) coordinates as values
    font_props : dict, optional
        Font properties from setup_space_mono_font()
    transform_coords : bool, optional
        Whether to transform coordinates (for 0-360 longitude systems)
    """
    # Create consistent styling for annotations
    marker_style = {
        'marker': 'o',
        'markersize': 6,
        'markerfacecolor': 'white',
        'markeredgecolor': 'black',
        'markeredgewidth': 1.0,
        'zorder': 10
    }
    
    text_style = {
        'ha': 'center',
        'va': 'center',
        'fontsize': 10,
        'fontweight': 'bold',
        'color': 'black',
        'path_effects': [
            plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='white')
        ],
        'zorder': 11,
        'fontproperties': font_props.get('bold') if font_props else None
    }
    
    # Add annotations for each location
    for name, (lat, lon) in locations_dict.items():
        # Transform longitude if needed
        if transform_coords and lon < 0:
            display_lon = lon + 360
        else:
            display_lon = lon
            
        # Plot marker
        ax.plot(display_lon, lat, **marker_style)
        
        # # Add text label with offset
        # ax.text(
        #     display_lon, lat - 0.03,
        #     name, 
        #     **text_style
        # )

def create_california_fire_risk_map(whp_data, shapefile_path, output_path, 
                                  title="CALIFORNIA WILDFIRE RISK BY ZIP CODE",
                                  legend_labels=None, data_note=None, fire_label=None):
    """
    Create a map of California zip codes colored by their fire risk categories.
    """
    # Set default values for English
    if legend_labels is None:
        legend_labels = ['Extreme Fire Risk', 'High Fire Risk', 'Low Fire Risk', 'No Data']
    if data_note is None:
        data_note = "DATA: US FOREST SERVICE"
    if fire_label is None:
        fire_label = "2025 LA FIRES"

    fire_locations_to_label = {
        'Palisades Fire': (34.043, -118.690),
        'Eaton Fire': (34.183, -118.103),
        'Hurst Fire': (34.327, -118.484)
    }

    # Set up the figure and font properties
    fig, ax, font_props = setup_enhanced_plot(figsize=(16, 18))
    
    # First load California state boundary for proper masking and focus
    try:
        print("Loading California state boundary")
        states_gdf = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip')
        ca_boundary = states_gdf[states_gdf['STUSPS'] == 'CA']
        
        if len(ca_boundary) == 0:
            print("Warning: California boundary not found in shapefile")
            ca_boundary = None
        else:
            print(f"California boundary loaded successfully")
            # Convert to Web Mercator projection
            ca_boundary = ca_boundary.to_crs(epsg=3857)
    except Exception as e:
        print(f"Could not load state boundary data: {e}")
        ca_boundary = None
    
    # Load the ZIP code shapefile
    print(f"Loading shapefile from {shapefile_path}")
    zip_gdf = gpd.read_file(shapefile_path)
    
    # Ensure zip code is string format for proper joining
    zip_gdf['ZCTA5CE20'] = zip_gdf['ZCTA5CE20'].astype(str)
    whp_data['zip'] = whp_data['zip'].astype(str)
    
    # Filter to California zip codes
    if ca_boundary is not None:
        # Ensure both are in the same CRS
        if zip_gdf.crs != ca_boundary.crs:
            zip_gdf = zip_gdf.to_crs(ca_boundary.crs)
        
        print("Filtering ZIP codes to those within California boundaries")
        # Use spatial join to find ZIP codes within California
        zip_gdf = gpd.sjoin(zip_gdf, ca_boundary[['geometry']], predicate='intersects', how='inner')
        print(f"Found {len(zip_gdf)} ZIP codes within California boundaries")
    else:
        # Fallback to postal code prefix filtering if we couldn't load the boundary
        print("Using ZIP code prefix as fallback filter for California")
        ca_prefixes = [str(prefix) for prefix in range(900, 962)]
        zip_gdf = zip_gdf[zip_gdf['ZCTA5CE20'].str[:3].isin(ca_prefixes)]
        print(f"Found {len(zip_gdf)} ZIP codes matching California prefixes")
    
    # Merge the zip code shapes with WHP risk data
    print(f"Merging shapefile with WHP data ({len(whp_data)} records)")
    map_data = zip_gdf.merge(
        whp_data[['zip', 'risk_category']], 
        left_on='ZCTA5CE20', 
        right_on='zip',
        how='left'
    )
    
    # Convert to Web Mercator projection for basemap compatibility
    if not map_data.crs:
        map_data = map_data.set_crs(epsg=4326)
    map_data = map_data.to_crs(epsg=3857)
    
    # Calculate coverage statistics
    total_zips = len(zip_gdf)
    matched_zips = map_data['risk_category'].notna().sum()
    coverage_pct = (matched_zips / total_zips) * 100
    print(f"Matched {matched_zips} of {total_zips} zip codes ({coverage_pct:.1f}%)")
    
    # Define the color mapping for risk categories
    risk_colors = {
        'Extreme Fire Risk': RISK_COLORS['Extreme Fire Risk'],
        'High Fire Risk': RISK_COLORS['High Fire Risk'],
        'Low Fire Risk': RISK_COLORS['Low Fire Risk']
    }
    
    # --------- PLOTTING SECTION ---------
    # First plot California boundary for context
    if ca_boundary is not None:
        ca_boundary.boundary.plot(ax=ax, linewidth=1.5, color='#555555')
    
    # Plot base layer with all zip codes in light gray
    base = map_data.plot(
        ax=ax,
        color='#EEEEEE',  # Default color for all shapes
        linewidth=0.1,
        edgecolor='#DDDDDD'
    )
    
    # Plot each category separately with its color
    for category, color in risk_colors.items():
        category_data = map_data[map_data['risk_category'] == category]
        if len(category_data) > 0:
            category_data.plot(
                ax=ax,
                color=color,
                linewidth=0.1,
                edgecolor='#DDDDDD'
            )
    
    # Get the bounds
    if ca_boundary is not None and not ca_boundary.empty:
        bounds = ca_boundary.total_bounds
    else:
        bounds = map_data.total_bounds
    
    # Set the axis limits
    ax.set_xlim([bounds[0], bounds[2]])
    ax.set_ylim([bounds[1], bounds[3]])
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    
    # Format the map title, subtitle and branding
    format_plot_title(ax, title, "", font_props)
    
    # Add the Deep Sky branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    # Create custom legend
    legend_elements = [
        Patch(facecolor=risk_colors['Extreme Fire Risk'], edgecolor='white', label=legend_labels[0]),
        Patch(facecolor=risk_colors['High Fire Risk'], edgecolor='white', label=legend_labels[1]),
        Patch(facecolor=risk_colors['Low Fire Risk'], edgecolor='white', label=legend_labels[2]),
        Patch(facecolor='#EEEEEE', edgecolor='white', label=legend_labels[3])
    ]
    
    legend = ax.legend(
        handles=legend_elements, 
        loc='center left',
        bbox_to_anchor=(-0.05, 0.5),
        frameon=True,
        facecolor=COLORS['background'],
        edgecolor='#DDDDDD',
        fontsize=16,
        prop=font_props.get('regular') if font_props else None,
        markerscale=2.0,
        handlelength=2.5,
        handleheight=2.0
    )
    
    # Annotate fire locations
    # First convert coordinates to Web Mercator projection to match the map
    
    # Create transformer function for the coordinate conversion
    wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 - standard lat/lon
    web_mercator = pyproj.CRS('EPSG:3857')  # Web Mercator
    transformer = pyproj.Transformer.from_crs(wgs84, web_mercator, always_xy=True)
    
    # Transform fire location coordinates and create new dictionary
    fire_locations_mercator = {}
    for name, (lat, lon) in fire_locations_to_label.items():
        x, y = transformer.transform(lon, lat)  # Note the order: lon, lat
        fire_locations_mercator[name] = (x, y)
    
    # Use the transformed coordinates to annotate the map
    # Create annotation styling
    for name, (x, y) in fire_locations_mercator.items():
        # Plot marker
        ax.plot(x, y, 'o', markersize=10, markerfacecolor='black', 
                markeredgecolor='black', markeredgewidth=0.5, zorder=10)
    
    label_anchor_x = min([loc[0] for loc in fire_locations_mercator.values()]) - 300000
    label_anchor_y = sum([loc[1] for loc in fire_locations_mercator.values()]) / len(fire_locations_mercator)

    ax.text(x - 130000, y - 90000,  # Offset in meters for Web Mercator
            fire_label,
            ha='center', va='center', fontsize=16, fontweight='bold',
            color='black', zorder=11,
            # path_effects=[path_effects.withStroke(linewidth=3, foreground='white')],
            fontproperties=font_props.get('bold') if font_props else None)
    
    # Save the map
    save_plot(fig, output_path)
    
    return fig


if __name__ == "__main__":
    
    # Inputs
    roa_premiums_df = pd.read_csv('insurance/CA_Request/PRA-2025-00261 - Premiums and Exposures by Company.csv')
    public_policies_df_1 = pd.read_excel('insurance/Residential-Property-Voluntary-Market-New-Renew-NonRenew-by-ZIP-2015-2021.xlsx')
    public_policies_df_1 = public_policies_df_1[public_policies_df_1['Year'] < 2020]
    public_policies_df_1['Non-Renewed'] = public_policies_df_1['Insured-Initiated Nonrenewed'] + public_policies_df_1['Insurer-Initiated Nonrenewed']
    public_policies_df_1 = public_policies_df_1[['ZIP Code', 'Year', 'New', 'Renewed', 'Non-Renewed']]

    public_policies_df_2 = pd.read_excel('insurance/Residential-Insurance-Voluntary-Market-New-Renew-NonRenew-by-ZIP-2020-2023.xlsx')
    public_policies_df_2 = public_policies_df_2[public_policies_df_2['Year'] >= 2020]
    public_policies_df_2 = public_policies_df_2[['ZIP Code', 'Year', 'New', 'Renewed', 'Non-Renewed']]

    public_policies_df = pd.concat([public_policies_df_1, public_policies_df_2], ignore_index=True)
    public_policies_df = public_policies_df.rename(columns={'ZIP Code': 'zip', 'Year': 'year', 'New': 'new', 'Renewed': 'renewed', 'Non-Renewed': 'non_renewed'})
    public_policies_df['zip'] = public_policies_df['zip'].astype(str).str.zfill(5)

    whp_df = pd.read_csv('whp_fs/whp_clean.csv')
    cpi = pd.read_csv('../data/cpi/cpi.csv')

    output_dir = 'figures'

    # Processing
    processed_roa_premiums_df = process_roa_premiums_data(roa_premiums_df, cpi)
    processed_policies_df = process_non_renewal_data(public_policies_df)
    
    processed_whp_df = process_whp_risk_data(whp_df, state_code='CA')

    # Merge with WHP Fire Risk Data
    premiums_merged = pd.merge(processed_roa_premiums_df, processed_whp_df, left_on='ZIP_CODE', right_on='zip', how='inner')
    policy_counts_merged = pd.merge(processed_policies_df, processed_whp_df, left_on='zip', right_on='zip', how='inner')

    # Add WHP Risk Categories
    premiums_merged = add_risk_categories(premiums_merged)
    policy_totals = add_risk_categories(policy_counts_merged)
    policy_totals['zip'] = policy_totals['zip'].astype(str).str.zfill(5)

    yearly_premiums_by_risk = calculate_average_premium_by_risk_category(premiums_merged)
    yearly_premiums_by_risk.to_csv('insurance/ca_premiums.csv')

    # Create the premiums visualization using plot_lines_by_risk_category
    output_file_premiums = f'{output_dir}/ca_premiums_vs_whp_risk.png'
    fig_premiums = plot_lines_by_risk_category(
        yearly_premiums_by_risk,
        y_val='average_premium_adj',
        title='PRICE OF CALIFORNIA HOME INSURANCE IS SOARING IN FIRE PRONE AREAS',
        subtitle='AVERAGE HOMEOWNERS INSURANCE PREMIUM (2020 DOLLARS)',
        data_note='DATA: CALIFORNIA DEPARTMENT OF INSURANCE AND US FOREST SERVICE',
        unit='dollar',
        save_path=output_file_premiums,
        legend_placement='upper left',
        figsize=(16, 14)
    )

    # incorporate census data to calculate insurance rates as percentage of number of households per zip code
    census_df = pd.read_csv('census_data/zip_census_data_TX_CA_multi_year.csv')
    census_df = census_df[census_df['state'] == 'CA']
    census_df['zip_code'] = census_df['zip_code'].astype(str).str.zfill(5)
    census_df = census_df[['zip_code', 'year', 'households', 'population']]

    insurance_census_merged = pd.merge(
        policy_totals,
        census_df,
        left_on=['zip', 'year'],
        right_on=['zip_code', 'year'],
        how='inner'
    )

    policies_per_household_grouped = insurance_census_merged.groupby(['year', 'risk_category'], observed=False)[['policies_in_force', 'households']].sum().reset_index()
    policies_per_household_grouped['policies_per_household'] = policies_per_household_grouped['policies_in_force'] / policies_per_household_grouped['households']
    policies_per_household_grouped['insured_pct'] = policies_per_household_grouped['policies_per_household'] * 100
    
    policies_per_household_grouped.to_csv('insurance/ca_policies_per_household.csv', index=False)

    output_file_policies_per_household = f'figures/ca_policies_per_household_vs_whp_risk.png'
    fig_policies_per_household = plot_lines_by_risk_category(
        policies_per_household_grouped,
        y_val='insured_pct',
        title='INSURERS ARE LEAVING WILDFIRE PRONE AREAS OF CALIFORNIA',
        subtitle='PERCENT OF HOUSEHOLDS WITH HOME INSURANCE',
        unit='percent',
        data_note='DATA: CALIFORNIA DEPARTMENT OF INSURANCE, US FOREST SERVICE, US CENSUS',
        save_path=output_file_policies_per_household,
        figsize=(16, 14)
    )

    policies_grouped = policy_totals.groupby(['year', 'risk_category'], observed=False)[['policies_in_force']].sum().reset_index()
    policies_baseline = policies_grouped[policies_grouped['year'] == 2015]
    policies_baseline = policies_baseline[['risk_category', 'policies_in_force']].rename(columns={'policies_in_force': 'policies_baseline'})
    policies_growth = pd.merge(
        policies_grouped,
        policies_baseline[['risk_category', 'policies_baseline']],
        on='risk_category',
        how='left'
    )

    policies_growth['policies_growth'] = (policies_growth['policies_in_force'] / policies_growth['policies_baseline']) * 100 - 100
    policies_growth.to_csv('insurance/ca_policies_growth_test.csv', index=False)
    output_file_policies_in_force = f'figures/ca_policies_in_force.png'
    fig_policies_per_household = plot_lines_by_risk_category(
        policies_growth,
        y_val='policies_growth',
        title='INSURERS ARE LEAVING WILDFIRE PRONE AREAS OF CALIFORNIA',
        subtitle='PERCENT CHANGE IN NUMBER OF HOME INSURANCE POLICIES FROM 2015',
        unit='percent',
        data_note='DATA: CALIFORNIA DEPARTMENT OF INSURANCE, US FOREST SERVICE',
        save_path=output_file_policies_in_force,
        legend_placement='center left',
        show_change_labels=True,
        figsize=(16, 14)
    )

    census_fire = pd.merge(census_df, processed_whp_df, left_on='zip_code', right_on='zip', how='inner')
    census_fire = add_risk_categories(census_fire)

    population_by_risk_category = census_fire.groupby(['year', 'risk_category'], observed=False)[['population']].sum().reset_index()
    population_baseline = population_by_risk_category[population_by_risk_category['year'] == 2011]
    population_baseline = population_baseline[['risk_category', 'population']].rename(columns={'population': 'population_baseline'})
    population_by_risk_category = pd.merge(
        population_by_risk_category,
        population_baseline[['risk_category', 'population_baseline']],
        on='risk_category',
        how='left'
    )
    population_by_risk_category['population_growth'] = (population_by_risk_category['population'] / population_by_risk_category['population_baseline']) * 100 - 100    

    # Plot population growth by fire risk category
    population_growth_fig = plot_lines_by_risk_category(
        population_by_risk_category,
        y_val='population_growth',
        title='POPULATION GROWTH BY FIRE RISK CATEGORY',
        subtitle='POPULATION GROWTH BY FIRE RISK CATEGORY',
        data_note='DATA: US FOREST SERVICE, US CENSUS',
        unit='percent',
        save_path='figures/ca_population_growth_by_risk.png',
        legend_placement='upper left'
    )

    shapefile_path = '../data/shapefiles/us/zip_codes/tl_2020_us_zcta520/tl_2020_us_zcta520.shp'
    map_output_path = f'{output_dir}/ca_fire_risk_map.png'
    
    # Create the fire risk map for California
    whp_categorized = add_risk_categories(processed_whp_df)
    fig_map = create_california_fire_risk_map(
        whp_categorized,
        shapefile_path,
        map_output_path,
        title="CALIFORNIA WILDFIRE RISK BY ZIP CODE"
    )

    # =====================
    # FRENCH LANGUAGE CHARTS
    # =====================
    print("\nGenerating French language charts...")

    french_risk_translations = {
        'Extreme Fire Risk': 'Risque d\'Incendie Extrême',
        'High Fire Risk': 'Risque d\'Incendie Élevé', 
        'Low Fire Risk': 'Risque d\'Incendie Faible'
    }
    
    # 1. French version of premiums chart
    output_file_premiums_fr = f'{output_dir}/ca_premiums_vs_whp_risk_fr.png'
    fig_premiums_fr = plot_lines_by_risk_category(
        yearly_premiums_by_risk,
        y_val='average_premium_adj',
        title='LE PRIX DE L\'ASSURANCE HABITATION EN CALIFORNIE MONTE EN FLÈCHE\n DANS LES ZONES À RISQUE D\'INCENDIE',
        subtitle='PRIME MOYENNE D\'ASSURANCE HABITATION (DOLLARS 2020)',
        data_note='DONNÉES: CALIFORNIA DEPARTMENT OF INSURANCE, US FOREST SERVICE',
        unit='dollar',
        save_path=output_file_premiums_fr,
        legend_placement='upper left',
        figsize=(16, 14),
        risk_category_translations=french_risk_translations
    )

    # 2. French version of policies per household chart
    output_file_policies_per_household_fr = f'figures/ca_policies_per_household_vs_whp_risk_fr.png'
    fig_policies_per_household_fr = plot_lines_by_risk_category(
        policies_per_household_grouped,
        y_val='insured_pct',
        title='LES ASSUREURS QUITTENT LES ZONES À RISQUE D\'INCENDIE DE CALIFORNIE',
        subtitle='POURCENTAGE DE MÉNAGES AVEC ASSURANCE HABITATION',
        unit='percent',
        data_note='DONNÉES: CALIFORNIA DEPARTMENT OF INSURANCE, US FOREST SERVICE, US CENSUS',
        save_path=output_file_policies_per_household_fr,
        figsize=(16, 14),
        risk_category_translations=french_risk_translations
    )

    # French translations for risk categories
    french_risk_translations = {
        'Extreme Fire Risk': 'Risque d\'Incendie Extrême',
        'High Fire Risk': 'Risque d\'Incendie Élevé', 
        'Low Fire Risk': 'Risque d\'Incendie Faible'
    }

    # 3. French version of policies growth chart
    output_file_policies_in_force_fr = f'figures/ca_policies_in_force_fr.png'
    fig_policies_growth_fr = plot_lines_by_risk_category(
        policies_growth,
        y_val='policies_growth',
        title='LES ASSUREURS QUITTENT LES ZONES À RISQUE D\'INCENDIE DE CALIFORNIE',
        subtitle='VARIATION EN POURCENTAGE DU NOMBRE DE POLICES D\'ASSURANCE HABITATION DEPUIS 2015',
        unit='percent',
        data_note='DONNÉES: CALIFORNIA DEPARTMENT OF INSURANCE, US FOREST SERVICE',
        save_path=output_file_policies_in_force_fr,
        legend_placement='center left',
        show_change_labels=True,
        figsize=(16, 14),
        risk_category_translations=french_risk_translations
    )

    # 4. French version of population growth chart
    population_growth_fig_fr = plot_lines_by_risk_category(
        population_by_risk_category,
        y_val='population_growth',
        title='CROISSANCE DÉMOGRAPHIQUE PAR CATÉGORIE DE RISQUE D\'INCENDIE',
        subtitle='CROISSANCE DÉMOGRAPHIQUE PAR CATÉGORIE DE RISQUE D\'INCENDIE',
        data_note='DONNÉES : US FOREST SERVICE, US CENSUS',
        unit='percent',
        save_path='figures/ca_population_growth_by_risk_fr.png',
        legend_placement='upper left',
        risk_category_translations=french_risk_translations
    )

    # 5. French version of fire risk map
    map_output_path_fr = f'{output_dir}/ca_fire_risk_map_fr.png'
    fig_map_fr = create_california_fire_risk_map(
        whp_categorized,
        shapefile_path,
        map_output_path_fr,
        title="RISQUE D'INCENDIE DE FORÊT EN CALIFORNIE PAR ZIP CODE",
        legend_labels=['Risque d\'Incendie Extrême', 'Risque d\'Incendie Élevé', 'Risque d\'Incendie Faible', 'Aucune Donnée'],
        data_note="DONNÉES : US FOREST SERVICE",
        fire_label="INCENDIES LA 2025"
    )
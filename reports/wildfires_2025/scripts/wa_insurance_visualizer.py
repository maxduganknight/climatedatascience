import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
import pyproj

from utils import (
    setup_space_mono_font, process_whp_risk_data, add_risk_categories,
    setup_enhanced_plot, format_plot_title, add_deep_sky_branding,
    plot_risk_lines, add_legend, save_plot, calculate_policy_count_growth,
    plot_lines_by_risk_category, filter_by_state_zip_codes,
    COLORS, RISK_COLORS
)

def process_wa_insurance_data(raw_wa_df, cpi):
    """
    Process Washington insurance data to calculate average premiums.
    """
    clean_wa_df = raw_wa_df.copy()
        
    # Convert year to numeric
    clean_wa_df['year'] = clean_wa_df['Year'].astype(int)

    # adjust for inflation
    clean_wa_df = pd.merge(clean_wa_df, cpi, on='year', how='left')
    cpi_2020 = cpi[cpi['year'] == 2020]['cpi'].iloc[0]
    clean_wa_df['PREMIUM_2020'] = clean_wa_df['Premiums Per Policy'] * (cpi_2020 / clean_wa_df['cpi'])
    
    percent_cols = [
        'Nonrenewal Rate', 'Nonpayment Cancellation Rate', 'Other than Nonpayment Cancellation Rate'
    ]
    # Convert percentage columns to numeric and handle errors
    for col in percent_cols:
        clean_wa_df[col] = pd.to_numeric(clean_wa_df[col], errors='coerce')
        clean_wa_df[col] = clean_wa_df[col].replace(0, np.nan)  # Replace 0 with NaN for non-renewal rates
        clean_wa_df[col] = clean_wa_df[col] * 100  # Convert to percentage

    clean_wa_df['all_non_renewals'] = clean_wa_df['Nonrenewal Rate'] + clean_wa_df['Nonpayment Cancellation Rate'] + clean_wa_df['Other than Nonpayment Cancellation Rate']

    # Ensure ZIP codes are strings with 5 characters
    clean_wa_df['ZIP Code'] = clean_wa_df['ZIP Code'].astype(str).str.zfill(5)

    clean_wa_df = filter_by_state_zip_codes(
        clean_wa_df,
        state_code='WA',
        zip_column='ZIP Code'
    )
    
    return clean_wa_df


def create_washington_fire_risk_map(whp_data, shapefile_path, output_path, 
                                  title="WASHINGTON WILDFIRE RISK BY ZIP CODE",
                                  legend_labels=None, data_note=None):
    """
    Create a map of Washington zip codes colored by their fire risk categories.
    """
    # Set default values for English
    if legend_labels is None:
        legend_labels = ['Extreme Fire Risk', 'High Fire Risk', 'Low Fire Risk', 'No Data']
    if data_note is None:
        data_note = "DATA: US FOREST SERVICE"

    # Set up the figure and font properties
    fig, ax, font_props = setup_enhanced_plot(figsize=(16, 12))
    
    # First load California state boundary for proper masking and focus
    try:
        print("Loading Wasington state boundary")
        states_gdf = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip')
        wa_boundary = states_gdf[states_gdf['STUSPS'] == 'WA']
        
        if len(wa_boundary) == 0:
            print("Warning: Washington boundary not found in shapefile")
            wa_boundary = None
        else:
            print(f"Washington boundary loaded successfully")
            # Convert to Web Mercator projection
            wa_boundary = wa_boundary.to_crs(epsg=3857)
    except Exception as e:
        print(f"Could not load state boundary data: {e}")
        wa_boundary = None
    
    # Load the ZIP code shapefile
    print(f"Loading shapefile from {shapefile_path}")
    zip_gdf = gpd.read_file(shapefile_path)
    
    # Ensure zip code is string format for proper joining
    zip_gdf['ZCTA5CE20'] = zip_gdf['ZCTA5CE20'].astype(str)
    whp_data['zip'] = whp_data['zip'].astype(str)
    
    # Filter to Washington zip codes
    if wa_boundary is not None:
        # Ensure both are in the same CRS
        if zip_gdf.crs != wa_boundary.crs:
            zip_gdf = zip_gdf.to_crs(wa_boundary.crs)
        
        print("Filtering ZIP codes to those within Washington boundaries")
        # Use spatial join to find ZIP codes within Washington
        zip_gdf = gpd.sjoin(zip_gdf, wa_boundary[['geometry']], predicate='intersects', how='inner')
        print(f"Found {len(zip_gdf)} ZIP codes within Washington boundaries")
    else:
        # Fallback to postal code prefix filtering if we couldn't load the boundary
        print("Using ZIP code prefix as fallback filter for Washington")
        zip_gdf = zip_gdf[zip_gdf['ZCTA5CE20'].str.startswith(('980', '981', '982', '983', '984', '985', '986', '988', '989', '990', '991', '992', '993', '994'))]
        print(f"Found {len(zip_gdf)} ZIP codes matching Washington prefixes")
    
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
    if wa_boundary is not None:
        wa_boundary.boundary.plot(ax=ax, linewidth=1.5, color='#555555')
    
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
    if wa_boundary is not None and not wa_boundary.empty:
        bounds = wa_boundary.total_bounds
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
    
    # Create custom legend with provided labels
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
    
    # Save the map
    save_plot(fig, output_path)
    
    return fig



if __name__ == "__main__":
    print("Processing Washington insurance data...")
    
    # Load the data
    file_path = 'insurance/WA_Request/Supporting_Underlying_Metrics_and_Disclaimer_for_Analyses_of_US_Homeowners_Insurance_Markets_2018-2022.xlsx'
    wa_insurance_df = pd.read_excel(file_path, sheet_name='Supporting Underlying Metrics')

    # Load WHP and CPI data with error handling
    try:
        whp_df = pd.read_csv('whp_fs/whp_clean.csv')
        print(f"WHP data loaded: {whp_df.shape[0]} records")
    except Exception as e:
        print(f"Error loading WHP data: {e}")
        whp_df = pd.DataFrame()
    
    try:
        cpi = pd.read_csv('../data/cpi/cpi.csv')
        print(f"CPI data loaded for years: {cpi['year'].min()}-{cpi['year'].max()}")
    except Exception as e:
        print(f"Error loading CPI data: {e}")
        cpi = pd.DataFrame({'year': [2020], 'cpi': [1.0]})
    
    # Process data
    processed_wa_df = process_wa_insurance_data(wa_insurance_df, cpi)
    processed_whp_df = process_whp_risk_data(whp_df, state_code='WA')
    
    # Merge datasets
    merged_data = pd.merge(processed_wa_df, processed_whp_df, left_on='ZIP Code', right_on='zip', how='inner')
    risk_categorized_data = add_risk_categories(merged_data)
    risk_categorized_data.to_csv('insurance/wa_insurance_risk_categorized_test.csv', index=False)
    yearly_premiums_by_risk = risk_categorized_data.groupby(['year', 'risk_category']).agg(
        average_premium=('PREMIUM_2020', 'mean'),
        non_renewal_rate=('Nonrenewal Rate', 'mean'),
        all_non_renewal_rate=('all_non_renewals', 'mean'),
        loss_ratio=('Loss Ratio', 'mean')
    ).reset_index()

    # Create output directory
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot average premiums by risk category
    output_file_premiums = f'{output_dir}/wa_premiums_vs_whp_risk.png'
    fig_premiums = plot_lines_by_risk_category(
        yearly_premiums_by_risk,
        y_val='average_premium',
        title='WASHINGTOM HOME INSURANCE RATES VARY BY WILDFIRE RISK',
        subtitle='AVERAGE HOMEOWNERS INSURANCE PREMIUM (2020 DOLLARS)',
        data_note='DATA: WASHINGTON STATE OFFICE OF THE INSURANCE COMMISSIONER AND US FOREST SERVICE',
        save_path=output_file_premiums,
        legend_placement='upper left'
    )
    
    # Plot average premiums by risk category
    output_file_nonrenewals = f'{output_dir}/wa_nonrenewals_vs_whp_risk.png'
    fig_premiums = plot_lines_by_risk_category(
        yearly_premiums_by_risk,
        y_val='non_renewal_rate',
        title='WASHINGTOM HOME INSURANCE NON-RENEWALS VARY BY WILDFIRE RISK',
        subtitle='AVERAGE HOMEOWNERS INSURANCE NON_RENEWAL RATE',
        data_note='DATA: WASHINGTON STATE OFFICE OF THE INSURANCE COMMISSIONER AND US FOREST SERVICE',
        unit = 'percent',
        save_path=output_file_nonrenewals,
        legend_placement='upper left'
    )

    # Plot average loss ratio by risk category
    output_file_loss_ratio = f'{output_dir}/wa_loss_ratio_vs_whp_risk.png'
    fig_loss_ratio = plot_lines_by_risk_category(
        yearly_premiums_by_risk,
        y_val='loss_ratio',
        title='WASHINGTOM HOME INSURANCE LOSS RATIOS VARY BY WILDFIRE RISK',
        subtitle='AVERAGE HOMEOWNERS INSURANCE LOSS RATIO',
        data_note='DATA: WASHINGTON STATE OFFICE OF THE INSURANCE COMMISSIONER AND US FOREST SERVICE',
        unit = 'percent',
        save_path=output_file_loss_ratio,
        legend_placement='upper left'
    )

    shapefile_path = '../data/shapefiles/us/zip_codes/tl_2020_us_zcta520/tl_2020_us_zcta520.shp'
    map_output_path = f'{output_dir}/wa_fire_risk_map.png'
    
    # Create the fire risk map for Washington
    whp_categorized = add_risk_categories(processed_whp_df)
    fig_map = create_washington_fire_risk_map(
        whp_categorized,
        shapefile_path,
        map_output_path,
        title="WASHINGTON WILDFIRE RISK BY ZIP CODE"
    )

    # =====================
    # FRENCH LANGUAGE CHARTS
    # =====================
    print("\nGenerating French language Washington charts...")

    french_risk_translations = {
        'Extreme Fire Risk': 'Risque d\'Incendie Extrême',
        'High Fire Risk': 'Risque d\'Incendie Élevé', 
        'Low Fire Risk': 'Risque d\'Incendie Faible'
    }

    # 1. French version of premiums chart
    output_file_premiums_fr = f'{output_dir}/wa_premiums_vs_whp_risk_fr.png'
    fig_premiums_fr = plot_lines_by_risk_category(
        yearly_premiums_by_risk,
        y_val='average_premium',
        title='LES TARIFS D\'ASSURANCE HABITATION DE WASHINGTON VARIENT SELON LE RISQUE D\'INCENDIE',
        subtitle='PRIME MOYENNE D\'ASSURANCE HABITATION (DOLLARS 2020)',
        data_note='DONNÉES : BUREAU DU COMMISSAIRE AUX ASSURANCES DE L\'ÉTAT DE WASHINGTON ET SERVICE FORESTIER AMÉRICAIN',
        save_path=output_file_premiums_fr,
        legend_placement='upper left',
        risk_category_translations=french_risk_translations
    )
    
    # 2. French version of non-renewals chart
    output_file_nonrenewals_fr = f'{output_dir}/wa_nonrenewals_vs_whp_risk_fr.png'
    fig_nonrenewals_fr = plot_lines_by_risk_category(
        yearly_premiums_by_risk,
        y_val='non_renewal_rate',
        title='LES NON-RENOUVELLEMENTS D\'ASSURANCE HABITATION DE WASHINGTON VARIENT SELON LE RISQUE D\'INCENDIE',
        subtitle='TAUX MOYEN DE NON-RENOUVELLEMENT D\'ASSURANCE HABITATION',
        data_note='DONNÉES : BUREAU DU COMMISSAIRE AUX ASSURANCES DE L\'ÉTAT DE WASHINGTON ET SERVICE FORESTIER AMÉRICAIN',
        unit='percent',
        save_path=output_file_nonrenewals_fr,
        legend_placement='upper left',
        risk_category_translations=french_risk_translations
    )

    # 3. French version of loss ratio chart
    output_file_loss_ratio_fr = f'{output_dir}/wa_loss_ratio_vs_whp_risk_fr.png'
    fig_loss_ratio_fr = plot_lines_by_risk_category(
        yearly_premiums_by_risk,
        y_val='loss_ratio',
        title='LES RATIOS DE SINISTRES D\'ASSURANCE HABITATION DE WASHINGTON VARIENT SELON LE RISQUE D\'INCENDIE',
        subtitle='RATIO MOYEN DE SINISTRES D\'ASSURANCE HABITATION',
        data_note='DONNÉES : BUREAU DU COMMISSAIRE AUX ASSURANCES DE L\'ÉTAT DE WASHINGTON ET SERVICE FORESTIER AMÉRICAIN',
        unit='percent',
        save_path=output_file_loss_ratio_fr,
        legend_placement='upper left',
        risk_category_translations=french_risk_translations
    )

    # 4. French version of fire risk map
    map_output_path_fr = f'{output_dir}/wa_fire_risk_map_fr.png'
    fig_map_fr = create_washington_fire_risk_map(
        whp_categorized,
        shapefile_path,
        map_output_path_fr,
        title="RISQUE D'INCENDIE DE FORÊT À WASHINGTON PAR CODE POSTAL",
        legend_labels=['Risque d\'Incendie Extrême', 'Risque d\'Incendie Élevé', 'Risque d\'Incendie Faible', 'Aucune Donnée'],
        data_note="DONNÉES : SERVICE FORESTIER AMÉRICAIN"
    )

    print("French language Washington charts generated successfully!")

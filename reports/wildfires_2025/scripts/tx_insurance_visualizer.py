import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Add this import for handling images
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # Make sure this is imported
import os

# Import shared utilities
from utils import (
    setup_space_mono_font, process_whp_risk_data, add_risk_categories,
    setup_enhanced_plot, format_plot_title, add_deep_sky_branding,
    plot_risk_lines, add_legend, save_plot, calculate_policy_count_growth,
    plot_lines_by_risk_category, plot_line, calculate_average_premium_by_risk_category,
    filter_by_state_zip_codes,
    COLORS, RISK_COLORS
)

def process_tx_insurance_data(raw_tx_df, cpi):
    """
    Process Texas insurance data to calculate average premiums.
    """
    clean_tx_df = raw_tx_df.copy()
    
    # Filter for homeowner insurance (HO line)
    clean_tx_df = clean_tx_df[clean_tx_df['LINE'] == 'HO ']

    # Filter for only Texas data
    clean_tx_df = filter_by_state_zip_codes(
        clean_tx_df,
        state_code='TX',
        zip_column='ZIP'
    )
    
    # Convert year to numeric
    clean_tx_df['year'] = clean_tx_df['YEAR'].astype(int)
    
    # Filter out rows with missing premium or exposure data
    clean_tx_df = clean_tx_df.dropna(subset=[
        'PREMIUM IN FORCE AT END OF QTR', 'EXPOSURE IN FORCE AT END OF QTR ($000)', 'POLICIES IN FORCE AT END OF QTR'
        ])
    
    # Filter out zero/negative exposure values
    clean_tx_df = clean_tx_df[clean_tx_df['EXPOSURE IN FORCE AT END OF QTR ($000)'] > 0]
    # Filter out zero/negative premium values
    clean_tx_df = clean_tx_df[clean_tx_df['PREMIUM IN FORCE AT END OF QTR'] > 0]
    # Filter out zero/negative policies in force
    clean_tx_df = clean_tx_df[clean_tx_df['POLICIES IN FORCE AT END OF QTR'] > 0]

    # Remove zip codes with less than 10 policies in force
    clean_tx_df = clean_tx_df[clean_tx_df['POLICIES IN FORCE AT END OF QTR'] > 5]

    # Adjust for inflation: Calculate premiums in 2020 dollars
    cpi_2020 = cpi[cpi['year'] == 2020]['cpi'].iloc[0] if not cpi[cpi['year'] == 2020].empty else 1.0
    
    # Create a dictionary mapping each year to its inflation adjustment factor
    cpi_factors = {row['year']: cpi_2020 / row['cpi'] for _, row in cpi.iterrows()}
    
    dollar_cols = [
        'PREMIUM IN FORCE AT END OF QTR', 'TOTAL_PAID_LOSS', 'FIRE_LOSS'
    ]

    for col in dollar_cols:
        clean_tx_df[col] = clean_tx_df.apply(
            lambda row: row[col] * cpi_factors.get(row['year'], 1.0), 
            axis=1
        )
    
    clean_tx_df = clean_tx_df.rename(columns={
        'PREMIUM IN FORCE AT END OF QTR': 'PREMIUM_2020',
        'TOTAL_PAID_LOSS': 'TOTAL_PAID_LOSS_2020',
        'FIRE_LOSS': 'FIRE_LOSS_2020'
    })

    clean_tx_df['ZIP'] = clean_tx_df['ZIP'].astype(str).str.zfill(5)
    return clean_tx_df
    
def plot_zip_code_premiums(data, year_filter=2023, top_n=10, 
                           output_path='figures/tx_highest_premium_zip_codes.png', plot_cities=False):
    """
    Plot premium values for each zip code with the highest premium zip codes labeled.
    
    Parameters:
    -----------
    data : DataFrame
        Data containing zip codes and premium information at individual policy level
    metric : str
        Column name for the premium metric to plot
    year_filter : int
        Year to use for identifying the top zip codes
    top_n : int
        Number of top zip codes to label
    output_path : str
        Path to save the output figure
    plot_cities : bool
        Whether to plot major Texas cities instead of individual zip codes
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Create a copy to avoid modifying original data
    raw_data = data.copy()
    
    if plot_cities:
        # Dictionary mapping Texas cities to their zip code prefixes/lists
        texas_cities = {
            'Houston': ['770', '771', '772', '773', '774', '775'],
            'Dallas': ['752', '753'],
            'San Antonio': ['782', '78201', '78202', '78203', '78204', '78205', '78207'],
            'Austin': ['787'],
            'Fort Worth': ['761', '762'],
            'El Paso': ['799'],
            'Arlington': ['760'],
            'Corpus Christi': ['784']
        }
        
        # Create city-based aggregation
        city_data = {}
        
        # Create a column to identify which city each ZIP code belongs to
        raw_data['city'] = 'Other'
        for city, zip_prefixes in texas_cities.items():
            for prefix in zip_prefixes:
                if len(prefix) == 5:
                    # Exact match
                    mask = raw_data['ZIP'] == prefix
                else:
                    # Prefix match
                    mask = raw_data['ZIP'].str.startswith(prefix)
                raw_data.loc[mask, 'city'] = city
        
        # Calculate proper city-level aggregation directly from policy-level data
        city_agg = raw_data.groupby(['city', 'year']).agg(
            total_premium=('PREMIUM_2020', 'sum'),
            total_policies=('POLICIES IN FORCE AT END OF QTR', 'sum')
        ).reset_index()

        city_agg['average_premium'] = city_agg['total_premium'] / city_agg['total_policies']

        
        # Filter out the 'Other' category
        city_agg = city_agg[city_agg['city'] != 'Other']
        
        # Calculate state-wide aggregation the same way
        state_agg = raw_data.groupby(['year']).agg(
            total_premium=('PREMIUM_2020', 'sum'),
            total_policies=('POLICIES IN FORCE AT END OF QTR', 'sum')
        ).reset_index()
        
        state_agg['average_premium'] = state_agg['total_premium'] / state_agg['total_policies']
        
        # Create figure with clean aesthetics
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        setup_space_mono_font()
        
        # Clean up the plot area - remove borders and set light background
        ax.set_facecolor('white')
        
        # Remove all spines (borders)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(False)
        
        # Add light gray spines just for bottom and left
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.spines['left'].set_color('#DDDDDD')
        
        # Add only horizontal light gridlines
        ax.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.8)
        ax.grid(axis='x', visible=False)
        ax.set_axisbelow(True)
        
        # Plot state-wide average first as a thicker, black dashed line
        ax.plot(state_agg['year'], state_agg['average_premium'], 
                color='black', linestyle='--', linewidth=2, 
                marker='o', markersize=0, label='TX Average')
        
        # Define city colors - using the same colors as the zip code view
        city_colors = {
            'Dallas': (0/255, 53/255, 148/255),  # Dallas blue
            'Houston': (167/255, 25/255, 48/255),  # Houston red
            'San Antonio': (0/255, 107/255, 166/255),  # San Antonio blue
            'Austin': (87/255, 196/255, 195/255),  # Austin teal
            'Fort Worth': (254/255, 80/255, 0/255),  # Fort Worth orange
            'El Paso': (0/255, 122/255, 51/255),  # El Paso green
            'Arlington': (178/255, 58/255, 158/255),  # Arlington purple
            'Corpus Christi': (99/255, 99/255, 99/255)  # Corpus Christi gray
        }
        
        # Plot city lines with distinct colors
        legend_handles = []
        legend_labels = []
        
        cities = city_agg['city'].unique()
        for i, city in enumerate(cities):
            city_data = city_agg[city_agg['city'] == city]
            color = city_colors.get(city, (0.1, 0.7, 0.8))  # default cyan if city not in dictionary
            line = ax.plot(city_data['year'], city_data['average_premium'], color=color, linewidth=2.5, 
                       marker='o', markersize=5, label=city)
            
            # Add annotation for the last data point
            if not city_data.empty:
                last_point = city_data.sort_values('year').iloc[-1]
                ax.annotate(f"{city}: ${last_point['average_premium']:,.0f}", 
                            xy=(last_point['year'], last_point['average_premium']),
                            xytext=(10, 0), 
                            textcoords="offset points", 
                            fontsize=11, 
                            fontweight='bold',
                            color=color)
            
            legend_handles.append(line[0])
            legend_labels.append(city)
                
        # Add Texas average to legend too
        legend_handles.insert(0, ax.get_lines()[0])
        legend_labels.insert(0, 'Texas Average')
        
        # Add legend with clean styling
        legend = ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='upper left',
            fontsize=10,
            frameon=True,
            facecolor='white',
            edgecolor='#DDDDDD'
        )
                
        title_text = "HOME INSURANCE PREMIUMS IN TEXAS CITIES"
        subtitle_text = f"AVERAGE PREMIUM PER POLICY (2020 DOLLARS)"

    else:
        # For ZIP code view, we need to group by ZIP code first
        zip_agg = raw_data.groupby(['ZIP', 'year']).agg(
            total_premium=('PREMIUM_2020', 'sum'),
            total_policies=('POLICIES IN FORCE AT END OF QTR', 'sum')
        ).reset_index()

        zip_agg['average_premium'] = zip_agg['total_premium'] / zip_agg['total_policies']

        
        # Add city information to each ZIP code
        # Dictionary mapping Texas cities to their zip code prefixes/lists
        texas_cities = {
            'Houston': ['770', '771', '772', '773', '774', '775'],
            'Dallas': ['752', '753'],
            'San Antonio': ['782'],
            'Austin': ['787'],
            'Fort Worth': ['761', '762'],
            'El Paso': ['799'],
            'Arlington': ['760'],
            'Corpus Christi': ['784']
        }
        
        # Create a column to identify which city each ZIP code belongs to
        zip_agg['city'] = 'Other'
        for city, zip_prefixes in texas_cities.items():
            for prefix in zip_prefixes:
                mask = zip_agg['ZIP'].str.startswith(prefix)
                zip_agg.loc[mask, 'city'] = city
        
        # Filter for zip codes that have data for the specified year
        latest_year_zips = zip_agg[zip_agg['year'] == year_filter]['ZIP'].unique()
        plot_data = zip_agg[zip_agg['ZIP'].isin(latest_year_zips)]
        
        # Create figure with clean aesthetics
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        setup_space_mono_font()
        
        # Clean up the plot area - remove borders and set light background
        ax.set_facecolor('white')
        
        # Remove all spines (borders)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(False)
        
        # Add light gray spines just for bottom and left
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.spines['left'].set_color('#DDDDDD')
        
        # Add only horizontal light gridlines
        ax.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.8)
        ax.grid(axis='x', visible=False)
        ax.set_axisbelow(True)
        
        # Identify top zip codes based on the most recent year
        top_zips_df = plot_data[plot_data['year'] == year_filter].sort_values(
            by='average_premium', ascending=False).head(top_n)
        top_zips = top_zips_df['ZIP'].tolist()
        
        # Calculate Texas average across all zip codes
        texas_avg_by_year = zip_agg.groupby('year')['average_premium'].mean().reset_index()
        
        # Plot Texas average as a thicker dashed black line
        tx_line = ax.plot(texas_avg_by_year['year'], texas_avg_by_year['average_premium'], 
                color='black', linestyle='--', linewidth=2.0, 
                marker='o', markersize=0, label='TX\nAverage')
        
        # Store handles for legend
        legend_handles = [tx_line[0]]  # Start with Texas average
        legend_labels = ['TX Average']
        
        # Create a dictionary to group zip codes by city
        city_zip_codes = {}
        for zip_code in top_zips:
            city = plot_data[plot_data['ZIP'] == zip_code]['city'].iloc[0]
            # Skip ZIP codes not in Dallas or Houston
            if city not in ['Dallas', 'Houston']:
                continue
            if city not in city_zip_codes:
                city_zip_codes[city] = []
            city_zip_codes[city].append(zip_code)
        
        # Define colors for each city
        city_colors = {
            'Dallas': (0/255, 53/255, 148/255),  # Dallas blue
            'Houston': (167/255, 25/255, 48/255),  # Houston red
        }
                # Create empty lists for legend with the desired order
        legend_handles = []
        legend_labels = []
        
        # Plot top zip codes grouped by city with consistent colors
        # First track lines by city
        city_lines = {}
        
        # Plot Dallas first
        if 'Dallas' in city_zip_codes:
            added_to_legend = False
            for zip_code in city_zip_codes['Dallas']:
                zip_data = plot_data[plot_data['ZIP'] == zip_code]
                line = ax.plot(zip_data['year'], zip_data['average_premium'], 
                           color=city_colors['Dallas'], linewidth=2.0, 
                           marker='o', markersize=0)
                
                # Add small text label near the last point
                last_point = zip_data[zip_data['year'] == zip_data['year'].max()]
                if not last_point.empty:
                    x = last_point['year'].values[0]
                    y = last_point['average_premium'].values[0]
                    ax.text(x, y, f" {zip_code}", fontsize=8, color=city_colors['Dallas'])
                
                if not added_to_legend:
                    city_lines['Dallas'] = line[0]
                    added_to_legend = True
        
        # Then plot Houston
        if 'Houston' in city_zip_codes:
            added_to_legend = False
            for zip_code in city_zip_codes['Houston']:
                zip_data = plot_data[plot_data['ZIP'] == zip_code]
                line = ax.plot(zip_data['year'], zip_data['average_premium'], 
                           color=city_colors['Houston'], linewidth=2.0, 
                           marker='o', markersize=0)
                
                # Add small text label near the last point
                last_point = zip_data[zip_data['year'] == zip_data['year'].max()]
                if not last_point.empty:
                    x = last_point['year'].values[0]
                    y = last_point['average_premium'].values[0]
                    ax.text(x, y, f" {zip_code}", fontsize=8, color=city_colors['Houston'])
                
                if not added_to_legend:
                    city_lines['Houston'] = line[0]
                    added_to_legend = True
        
        # Add all lines to legend in desired order
        if 'Dallas' in city_lines:
            legend_handles.append(city_lines['Dallas'])
            legend_labels.append('Dallas')
            
        if 'Houston' in city_lines:
            legend_handles.append(city_lines['Houston'])
            legend_labels.append('Houston')
            
        # Texas average is added last in the legend
        legend_handles.append(tx_line[0])
        legend_labels.append('TX Average')
        
        # Add annotation for Texas average
        last_year = texas_avg_by_year['year'].max()
        last_avg = texas_avg_by_year[texas_avg_by_year['year'] == last_year]['average_premium'].values[0]
        # ax.annotate("Texas Average", 
        #         # xy=(last_year, last_avg),
        #         # xytext=(10, 0), 
        #         textcoords="offset points", 
        #         fontsize=8,
        #         color='black')
                
        title_text = "HOME INSURANCE PRICES SOARING IN TEXAS URBAN AREAS"
        subtitle_text = f"AVERAGE HOME INSURANCE PREMIUM (2020 DOLLARS)"
        
        # Add legend with clean styling
        legend = ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='upper left',
            fontsize=10,
            frameon=True,
            facecolor='white',
            edgecolor='#DDDDDD'
        )
    
    # Format the plot title with consistent styling
    format_plot_title(ax, title_text, subtitle_text)
    
    # Add the data source note directly to match CA style
    data_note = "DATA: TEXAS DEPARTMENT OF INSURANCE. Displaying 9 of the 10 top zip codes by home insurance premium in 2023."
    
    # Add data source note and Deep Sky attribution directly
    plt.figtext(0.1, 0.01, f'ANALYSIS: DEEP SKY RESEARCH\n{data_note}', 
               fontsize=12, color='#505050', 
               ha='left', va='bottom')
    
    # Add Deep Sky icon at the bottom-right, correctly aligned with chart edge
    icon_path = '/Users/max/Deep_Sky/design/Favicon/favicon_for_charts.png'
    if os.path.exists(icon_path):
        icon = mpimg.imread(icon_path)
        imagebox = OffsetImage(icon, zoom=0.03)
        
        # Position right-aligned with the plot area
        ab = AnnotationBbox(imagebox, (0.95, 0.01),
                          xycoords='figure fraction',
                          frameon=False,
                          box_alignment=(1.0, 0.0))
        ax.add_artist(ab)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    # Set x-axis to show only years (integers)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    padding = y_range * 0.05
    y_lower = max(0, y_min - padding)  # Don't go below 0 unless data does
    y_upper = y_max + padding
    ax.set_ylim(bottom=y_lower, top=y_upper)
    
    # Add additional information about the plot
    # if plot_cities:
    #     ax.text(0.01, 0.02, f"Showing average premiums for major Texas cities", 
    #             transform=ax.transAxes, fontsize=9, alpha=0.7)
    # else:
    #     filtered_count = sum([len(zips) for city, zips in city_zip_codes.items()])
    #     ax.text(0.01, 0.02, f"Showing Texas average and top premiums from {filtered_count} high-cost ZIP codes in urban areas", 
    #             transform=ax.transAxes, fontsize=9, alpha=0.7)
    
    # Tight layout for proper spacing
    plt.tight_layout()
    
    # Save the plot
    save_plot(fig, output_path)
    
    return fig

if __name__ == "__main__":
    print("Processing Texas insurance data...")
    
    # Load the data
    tx_insurance_df = pd.read_csv('insurance/TX_Request/combined_insurance_data.csv')
    
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
    processed_tx_df = process_tx_insurance_data(tx_insurance_df, cpi)
    processed_whp_df = process_whp_risk_data(whp_df, state_code='TX')

    merged_data = pd.merge(processed_tx_df, processed_whp_df, left_on='ZIP', right_on='zip', how='inner')
    risk_categorized_data = add_risk_categories(merged_data)
    yearly_premiums_by_risk = calculate_average_premium_by_risk_category(
        risk_categorized_data, 
        premium_col='PREMIUM_2020', 
        denom='POLICIES IN FORCE AT END OF QTR'
    )
    
    # Create output directory
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot average premiums by risk category (premium per exposure)
    output_file_premiums_exposure = f'{output_dir}/tx_premiums_vs_whp_risk.png'
    fig_premiums_exposure = plot_lines_by_risk_category(
        yearly_premiums_by_risk,
        y_val='average_premium_adj',
        title='TEXAS HOME INSURANCE RATES VARY BY WILDFIRE RISK',
        subtitle='AVERAGE PREMIUM PER POLICY (2020 DOLLARS)',  # Updated subtitle
        data_note='DATA: TEXAS DEPARTMENT OF INSURANCE AND US FOREST SERVICE',
        unit='dollar',
        save_path=output_file_premiums_exposure,
        legend_placement='upper left'
    )

    # incorporate census data to calculate insurance rates as percentage of number of households per zip code
    census_df = pd.read_csv('census_data/zip_census_data_TX_CA_multi_year.csv')
    census_df = census_df[census_df['state'] == 'TX']
    census_df['zip_code'] = census_df['zip_code'].astype(str).str.zfill(5)
    census_df = census_df[['zip_code', 'year', 'households', 'population']]

    insurance_census_merged = pd.merge(
        risk_categorized_data,
        census_df,
        left_on=['ZIP', 'year'],
        right_on=['zip_code', 'year'],
        how='inner'
    )

    # remove zip codes with no households
    insurance_census_merged = insurance_census_merged[insurance_census_merged['households'] > 5]
    insurance_census_merged.to_csv('insurance/tx_insurance_census_merged.csv', index=False)

    # Calculate policies per household
    policies_per_household_grouped = insurance_census_merged.groupby(['year', 'risk_category'], observed=False)[['POLICIES IN FORCE AT END OF QTR', 'households']].sum().reset_index()
    policies_per_household_grouped['policies_per_household'] = policies_per_household_grouped['POLICIES IN FORCE AT END OF QTR'] / policies_per_household_grouped['households'] 
    policies_per_household_grouped['insured_pct'] = policies_per_household_grouped['policies_per_household'] * 100
    
    policies_per_household_grouped.to_csv('insurance/tx_policies_per_household_grouped.csv', index=False)

    # Plot policies per household
    output_file_policies_per_household = f'figures/tx_policies_per_household_vs_whp_risk.png'
    fig_policies_per_household = plot_lines_by_risk_category(
        policies_per_household_grouped,
        y_val='insured_pct',
        title='TEXAS POLICIES PER HOUSEHOLD BY WILDFIRE RISK',
        subtitle='NUMBER OF HOME INSURANCE POLICIES IN FORCE / NUMBER OF HOUSEHOLDS',
        data_note='DATA: TEXAS DEPARTMENT OF INSURANCE, US FOREST SERVICE, US CENSUS',
        unit= 'percent',
        save_path=output_file_policies_per_household,
        legend_placement='upper right'
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
    population_by_risk_category['population_growth'] = (population_by_risk_category['population'] / population_by_risk_category['population_baseline']) * 100    

    # Plot population growth by fire risk category
    population_growth_fig = plot_lines_by_risk_category(
        population_by_risk_category,
        y_val='population_growth',
        title='POPULATION GROWTH BY FIRE RISK CATEGORY',
        subtitle='POPULATION GROWTH BY FIRE RISK CATEGORY',
        data_note='DATA: US FOREST SERVICE, US CENSUS',
        unit='percent',
        save_path='figures/tx_population_growth_by_risk.png',
        legend_placement='upper left'
    )

    # After the other plots, add this section
    # Plot zip codes with highest premiums
    highest_premium_zip_fig = plot_zip_code_premiums(
        processed_tx_df,
        year_filter=processed_tx_df['year'].max(),
        top_n=10,  # Now showing top 10
        output_path='figures/tx_highest_premium_zip_codes_by_city.png'
    )
    
    # For city analysis, also use the original processed data
    city_premium_fig = plot_zip_code_premiums(
        processed_tx_df,  # Use the original processed data that has all the original column names
        year_filter=processed_tx_df['year'].max(),
        output_path='figures/tx_city_premiums.png',
        plot_cities=True
    )
    
    yearly_data_by_risk = risk_categorized_data.groupby(['year', 'risk_category']).agg(
        total_loss=('TOTAL_PAID_LOSS_2020', 'sum'),
        fire_loss=('FIRE_LOSS_2020', 'sum'),
        policies_in_force=('POLICIES IN FORCE AT END OF QTR', 'sum'),
    ).reset_index()
    yearly_data_by_risk['losses_per_policy'] = yearly_data_by_risk['total_loss'] / yearly_data_by_risk['policies_in_force']
    yearly_data_by_risk['fire_loss_per_policy'] = yearly_data_by_risk['fire_loss'] / yearly_data_by_risk['policies_in_force']

    losses_per_policy = plot_lines_by_risk_category(
        yearly_data_by_risk,
        y_val='losses_per_policy',
        title='AVERAGE LOSSES BY FIRE RISK CATEGORY',
        subtitle='AVERAGE LOSSES BY FIRE RISK CATEGORY',
        data_note='DATA: TEXAS DEPARTMENT OF INSURANCE AND US FOREST SERVICE',
        unit='dollar',
        save_path='figures/tx_average_losses_by_risk.png',
        legend_placement='upper left'
    )

    losses_per_policy = plot_lines_by_risk_category(
        yearly_data_by_risk,
        y_val='fire_loss_per_policy',
        title='AVERAGE FIRE LOSSES BY FIRE RISK CATEGORY',
        subtitle='AVERAGE FIRE LOSSES BY FIRE RISK CATEGORY',
        data_note='DATA: TEXAS DEPARTMENT OF INSURANCE AND US FOREST SERVICE',
        unit='dollar',
        save_path='figures/tx_average_fire_losses_by_risk.png',
        legend_placement='upper left'
    )

    print("All visualizations complete.")

    average_premiums = processed_tx_df.groupby('year').agg(
        total_premiums=('PREMIUM_2020', 'sum'),
        total_policies=('POLICIES IN FORCE AT END OF QTR', 'sum')
    ).reset_index()
    average_premiums['average_premium'] = average_premiums['total_premiums'] / average_premiums['total_policies']

    overall_premiums = plot_line(
        average_premiums,
        y_val='average_premium',
        title='TEXAS HOME INSURANCE PREMIUMS ARE SOARING',
        subtitle='AVERAGE TEXAS HOME INSURANCE PREMIUMS (2020 DOLLARS)',
        data_note='DATA: TEXAS DEPARTMENT OF INSURANCE',
        unit='dollar',
        save_path='figures/tx_overall_premiums.png',
        legend_placement='upper left'
    )


import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import datetime
import math

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot


def load_company_emissions_data(csv_path, company_name):
    """
    Load company emissions data from CSV file.
    Returns emissions dataframe (historical) and target pathway dataframe.

    Parameters:
    -----------
    csv_path : str
        Path to the company_emissions.csv file
    company_name : str
        Name of the company to extract (e.g., 'Air Canada')
    """
    df = pd.read_csv(csv_path)

    # Filter for the specific company
    company_df = df[df['Company'] == company_name].copy()

    if len(company_df) == 0:
        raise ValueError(f"Company '{company_name}' not found in the data")

    # Clean the Total column - remove commas and convert to float
    company_df['Total_clean'] = company_df['Total'].astype(str).str.replace(',', '').replace('', None)
    company_df['Total_clean'] = pd.to_numeric(company_df['Total_clean'], errors='coerce')

    # Split into historical and target data based on 'Historica/target' column
    historical_df = company_df[company_df['Historica/target'] == 'Historical'][['Year', 'Total_clean']].copy()
    historical_df.columns = ['year', 'actual']
    historical_df = historical_df.dropna(subset=['actual'])
    target_df = company_df[company_df['Historica/target'] == 'Target'][['Year', 'Total_clean']].copy()
    target_df.columns = ['year', 'target']
    target_df = target_df.dropna(subset=['target'])

    return historical_df, target_df


def annotate_emissions_gap(ax, year, projected_emissions, target_emissions, units='tCO2e'):
    """
    Annotate the emissions gap between current trajectory and target pathway for a given year.
    """
    gap = projected_emissions - target_emissions

    # Calculate offset based on scale of emissions
    offset = max(projected_emissions, abs(target_emissions)) * 0.02

    # Draw vertical lines showing the gaps
    ax.plot([year, year], [projected_emissions - offset, target_emissions + offset],
            color='#E74C3C', linewidth=2, alpha=0.8)

    ax.plot([year - 0.5, year + 0.5], [projected_emissions - offset, projected_emissions - offset],
            color='#E74C3C', linewidth=2, alpha=0.8)

    ax.plot([year - 0.5, year + 0.5], [target_emissions + offset, target_emissions + offset],
            color='#E74C3C', linewidth=2, alpha=0.8)

    # Add gap annotation
    midpoint = (target_emissions + projected_emissions) / 2

    # Format gap text - convert to millions for readability
    gap_millions = gap / 1e6
    gap_text = f'{gap_millions:.1f}M tCO\N{SUBSCRIPT TWO}e\nOVER BY {year}'

    ax.annotate(gap_text,
                xy=(year + 1, midpoint), xytext=(year + 1, midpoint),
                fontsize=10, ha='left', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='#E74C3C', alpha=0.9))


def create_company_emissions_gap_plot(emissions_df, pathway_df, company_name):
    """
    Create company emissions gap visualization.
    Shows actual emissions, projected trend, and target pathway.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 8))

    # Find the last year with actual data
    last_actual_year = emissions_df['year'].max()
    last_actual_value = emissions_df[emissions_df['year'] == last_actual_year]['actual'].iloc[0]

    # Calculate growth rate from recent actual data (last 3 years if available)
    actual_data = emissions_df.copy()
    if len(actual_data) >= 2:
        recent_years = min(3, len(actual_data))
        recent_data = actual_data.tail(recent_years)

        years_span = recent_data['year'].iloc[-1] - recent_data['year'].iloc[0]
        if years_span > 0:
            annual_change = (recent_data['actual'].iloc[-1] - recent_data['actual'].iloc[0]) / years_span
        else:
            annual_change = 0
    else:
        annual_change = 0

    # Create projected emissions based on trend
    future_years = pathway_df[pathway_df['year'] >= last_actual_year]['year'].values
    projected_data = []

    for year in future_years:
        years_ahead = year - last_actual_year
        projected_value = last_actual_value + (annual_change * years_ahead)
        projected_data.append({'year': year, 'projected': projected_value})

    projected_df = pd.DataFrame(projected_data)

    # Combine last actual with projected for continuous line
    last_actual_df = pd.DataFrame([{'year': last_actual_year, 'projected': last_actual_value}])
    projected_with_connection = pd.concat([last_actual_df, projected_df], ignore_index=True)

    # Split historical and projected at 2024
    historical_df = emissions_df[emissions_df['year'] <= 2024].copy()

    # Plot historical data (solid line)
    ax.plot(historical_df['year'], historical_df['actual'],
            color='#2C3E50', linewidth=3, marker='o', markersize=6, solid_capstyle='round')

    # Plot projected data (dashed line)
    if len(projected_with_connection) > 0:
        ax.plot(projected_with_connection['year'], projected_with_connection['projected'],
                color='#2C3E50', linewidth=3, linestyle='--', marker='', markersize=6,
                solid_capstyle='round', alpha=0.7)

    # Plot target pathway
    ax.plot(pathway_df['year'], pathway_df['target'],
            color='#F39C12', linewidth=3, marker='o', markersize=6,
            solid_capstyle='round')

    # Determine x-axis limits based on data
    min_year = min(emissions_df['year'].min(), pathway_df['year'].min())
    max_year = max(emissions_df['year'].max(), pathway_df['year'].max())
    ax.set_xlim(min_year - 2, max_year + 2)

    # Calculate max for y-axis - handle target being 0
    all_values = list(emissions_df['actual'].values) + list(projected_df['projected'].values)
    max_value = max(all_values)
    min_value = 0  # Since target is 0

    # Set y-limits with some padding
    ax.set_ylim(min_value, max_value * 1.2)

    # X-axis formatting
    x_range = max_year - min_year
    if x_range > 30:
        tick_interval = 10
    else:
        tick_interval = 5
    ax.set_xticks(range(int(min_year), int(max_year) + 1, tick_interval))
    ax.tick_params(axis='both', labelsize=12)

    # Y-axis formatting - convert to millions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))

    # Add text annotations
    mid_historical_year = int((min_year + last_actual_year) / 2)
    ax.text(mid_historical_year, max_value * 1.1, 'HISTORICAL\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', color='#2C3E50')

    mid_projection_year = int((last_actual_year + max_year) / 2)
    ax.text(mid_projection_year, max_value * 1.1, 'PROJECTED EMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.6, color='#2C3E50')

    # Position target pathway label
    target_year = pathway_df['year'].iloc[0]
    target_value = pathway_df['target'].iloc[0]
    ax.text(target_year, target_value + max_value * 0.05, 'TARGET\nPATHWAY',
            fontsize=12, ha='center', va='bottom',
            fontweight='bold', alpha=0.7, color='#E67E22')

    # Annotate the gap at the target year
    if len(projected_df) > 0:
        target_year = pathway_df['year'].iloc[0]
        target_value = pathway_df['target'].iloc[0]

        # Find projected value for the target year
        projected_at_target = projected_df[projected_df['year'] == target_year]
        if len(projected_at_target) > 0:
            projected_value = projected_at_target['projected'].iloc[0]
            annotate_emissions_gap(ax, target_year, projected_value, target_value)

    return fig


def load_aviation_industry_data_bergero(xlsx_path):
    """
    Load aviation industry emissions data from Bergero 2023 Excel file.
    Returns historical emissions and projection dataframes with averaged scenarios.

    The Excel file contains:
    - Column A: Year
    - Columns AJ, AN, AR (indices 36, 40, 44): Carbon Intensive scenarios
    - Columns AK, AO, AS (indices 37, 41, 45): Reduced Carbon scenarios
    - Columns AL, AP, AT (indices 38, 42, 46): Net-zero scenarios
    """
    # ============================================================================
    # RECENT EMISSIONS DATA (NOT IN BERGERO 2023) - UPDATE THESE VALUES AS NEEDED
    # ============================================================================
    EMISSIONS_2022_MT = 800   # Mt CO2 - Update this value as new data becomes available
    EMISSIONS_2023_MT = 950   # Mt CO2 - Update this value as new data becomes available
    # ============================================================================

    # Read Excel file without headers
    df_raw = pd.read_excel(xlsx_path, sheet_name='Fig1_future_Kaya', header=None)

    # The actual data starts from row 3 (index 3)
    # Row 0: "Emissions (F)", NaN, ...
    # Row 1: "Business-as-usual (GtCO2)", NaN, ..., "Industry Projections (GtCO2)", ...
    # Row 2: "Year", "Carbon Intensive", "Reduced Carbon", "Net-zero", NaN, "Carbon Intensive", ...
    # Row 3: 1990, 0.542698, ...  <- First data row

    # Extract year column (column A, index 0) - starts at row 3
    years = df_raw.iloc[3:, 0].copy()

    # Based on debugging, the correct columns are:
    # Column 35, 39, 43: "Carbon Intensive" (Business-as-usual, Industry Projections, Ambitious Projections)
    # Column 36, 40, 44: "Reduced Carbon"
    # Column 37, 41, 45: "Net-zero"
    # Column 38, 42, 46: separator (NaN)

    # MDK I have stopped using the Ambitious Projection values because lead author Candelaria Bergero 
    # said via email that these pathways assume no return to pre-COVID air travel demand and that has already
    # been proven wrong. 

    # carbon_intensive_cols = [35, 39, 43]
    # reduced_carbon_cols = [36, 40, 44]
    # net_zero_cols = [37, 41, 45]

    carbon_intensive_cols = [35, 39]
    reduced_carbon_cols = [36, 40]
    net_zero_cols = [37, 41]

    # Extract data starting from row 3
    data_rows = df_raw.iloc[3:].copy()

    # Calculate means for each scenario - convert to numeric first
    carbon_intensive_mean = data_rows.iloc[:, carbon_intensive_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    reduced_carbon_mean = data_rows.iloc[:, reduced_carbon_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    net_zero_mean = data_rows.iloc[:, net_zero_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    # Create projections dataframe
    projections_df = pd.DataFrame({
        'year': years.values,
        'carbon_intensive_gt': carbon_intensive_mean.values,
        'reduced_carbon_gt': reduced_carbon_mean.values,
        'net_zero_gt': net_zero_mean.values
    })

    # Convert to numeric and drop any NaN rows
    projections_df = projections_df.apply(pd.to_numeric, errors='coerce')
    projections_df = projections_df.dropna()

    # Split into historical (2000-2021) and future projections
    historical_df = projections_df[(projections_df['year'] >= 2000) & (projections_df['year'] <= 2021)][['year', 'carbon_intensive_gt']].copy()
    historical_df.columns = ['year', 'emissions_gt']

    # Add hardcoded recent years (2022-2023) that are not in Bergero 2023 data
    recent_years_df = pd.DataFrame({
        'year': [2022, 2023],
        'emissions_gt': [EMISSIONS_2022_MT / 1000, EMISSIONS_2023_MT / 1000]  # Convert Mt to Gt
    })
    historical_df = pd.concat([historical_df, recent_years_df], ignore_index=True)

    # Get the 2023 historical value for interpolation
    historical_2023_value = historical_df[historical_df['year'] == 2023]['emissions_gt'].iloc[0]

    # Future projections (from 2030 onwards, we'll interpolate 2024-2029)
    future_df = projections_df[projections_df['year'] >= 2030].copy()

    # Create interpolated years 2024-2029 with all three scenarios
    interpolated_years = pd.DataFrame({'year': range(2023, 2030)})

    # Interpolate each scenario from 2023 to 2030
    for scenario_col in ['carbon_intensive_gt', 'reduced_carbon_gt', 'net_zero_gt']:
        # Get 2030 value for this scenario
        value_2030 = future_df[future_df['year'] == 2030][scenario_col].iloc[0]

        # Linear interpolation for 2024-2029
        interpolated_years[scenario_col] = interpolated_years['year'].apply(
            lambda y: historical_2023_value + (value_2030 - historical_2023_value) * (y - 2023) / (2030 - 2023)
        )

    # Combine interpolated years with future projections
    future_df = pd.concat([interpolated_years, future_df], ignore_index=True)

    # Sort by year to ensure proper ordering
    future_df = future_df.sort_values('year').reset_index(drop=True)

    # Net-zero pathway serves as the target
    target_df = future_df[['year', 'net_zero_gt']].copy()
    target_df.columns = ['year', 'target_gt']

    return historical_df, target_df, future_df


def create_aviation_industry_gap_plot(historical_df, target_df, projections_df):
    """
    Create aviation industry emissions gap visualization using Bergero 2023 data.
    Shows historical emissions, three projection scenarios (carbon-intensive, reduced-carbon, net-zero).
    Annotates the gap between reduced-carbon projection and net-zero pathway.

    Parameters:
    -----------
    historical_df : DataFrame with columns ['year', 'emissions_gt']
    target_df : DataFrame with columns ['year', 'target_gt'] (net-zero pathway)
    projections_df : DataFrame with columns ['year', 'carbon_intensive_gt', 'reduced_carbon_gt', 'net_zero_gt']
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 8))

    # Detect column suffix (either _gt or _mt)
    emissions_col = 'emissions_gt' if 'emissions_gt' in historical_df.columns else 'emissions_mt'
    target_col = 'target_gt' if 'target_gt' in target_df.columns else 'target_mt'

    # Determine unit label
    units = 'Gt' if '_gt' in emissions_col else 'Mt'

    # Get last historical point for connection
    last_historical_year = historical_df['year'].max()
    last_historical_value = historical_df.iloc[-1][emissions_col]

    # Plot historical data (solid line)
    ax.plot(historical_df['year'], historical_df[emissions_col],
            color='#2C3E50', linewidth=3, marker='o', markersize=6, solid_capstyle='round')

    # Plot carbon-intensive projection (dashed line)
    if 'carbon_intensive_gt' in projections_df.columns:
        carbon_intensive_col = 'carbon_intensive_gt'
    else:
        carbon_intensive_col = 'aviation_carbon_intensive'

    # carbon_intensive_with_connection = pd.concat([
    #     pd.DataFrame([{'year': last_historical_year, 'value': last_historical_value}]),
    #     pd.DataFrame({'year': projections_df['year'], 'value': projections_df[carbon_intensive_col]})
    # ], ignore_index=True)

    ax.plot(projections_df['year'], projections_df[carbon_intensive_col],
            color='#E74C3C', linewidth=3, linestyle='--', marker='', markersize=6,
            solid_capstyle='round', alpha=0.7)

    # Plot reduced-carbon projection (dashed line)
    if 'reduced_carbon_gt' in projections_df.columns:
        reduced_carbon_col = 'reduced_carbon_gt'
    else:
        reduced_carbon_col = 'aviation_reduced_carbon'

    # reduced_carbon_with_connection = pd.concat([
    #     pd.DataFrame([{'year': last_historical_year, 'value': last_historical_value}]),
    #     pd.DataFrame({'year': projections_df['year'], 'value': projections_df[reduced_carbon_col]})
    # ], ignore_index=True)

    ax.plot(projections_df['year'], projections_df[reduced_carbon_col],
            color='#F39C12', linewidth=3, linestyle='--', marker='', markersize=6,
            solid_capstyle='round', alpha=0.7)

    # Plot net-zero/target pathway
    ax.plot(target_df['year'], target_df[target_col],
            color='#27AE60', linewidth=3, marker='', markersize=6,
            solid_capstyle='round')

    # Set axis limits
    min_year = int(historical_df['year'].min())
    max_year = int(target_df['year'].max())
    ax.set_xlim(min_year, max_year + 0.5)

    # Calculate max for y-axis
    all_values = list(historical_df[emissions_col].values) + \
                 list(projections_df[carbon_intensive_col].values) + \
                 list(projections_df[reduced_carbon_col].values) + \
                 list(target_df[target_col].values)
    max_value = max(all_values)

    ax.set_ylim(0, max_value * 1.1)

    # X-axis formatting
    tick_interval = 10 if (max_year - min_year) > 30 else 5
    ax.set_xticks(range(min_year, max_year + 1, tick_interval))
    ax.tick_params(axis='both', labelsize=12)

    # Y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

    # Add text annotations
    mid_historical = int((min_year + last_historical_year) / 2)
    ax.text(mid_historical, max_value*.85, 'PAST\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', color='#2C3E50')

    mid_projection = int((last_historical_year + max_year) / 2)
    ax.text(mid_projection, max_value*.85, 'PROJECTED\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.6, color='#2C3E50')

    ax.text(2045, max_value * .77, 'CARBON INTENSIVE\nPATHWAY',
        fontsize=12, ha='center', va='center',
        fontweight='bold', alpha=.8, color='#E74C3C')

    ax.text(2045, max_value * .48, 'REDUCED CARBON\nPATHWAY',
        fontsize=12, ha='center', va='center',
        fontweight='bold', alpha=.8, color='#F39C12')

    ax.text(2020, max_value * .28, 'PANDEMIC\nSLOWDOWN',
        fontsize=12, ha='center', va='center',
        fontweight='bold', color='#2C3E50')    

    # Position pathway label
    target_mid = target_df[target_df['year'] == int((last_historical_year + max_year) / 2)]
    if len(target_mid) > 0:
        target_mid_value = target_mid[target_col].iloc[0]
    else:
        target_mid_value = target_df[target_col].mean()

    ax.text(2040, max_value * 0.1, 'NET-ZERO PATHWAY',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#27AE60')

    # Annotate the gap between reduced-carbon and net-zero pathways
    gap_years = [2030, 2050]

    for year in gap_years:
        reduced_data = projections_df[projections_df['year'] == year]
        target_data = target_df[target_df['year'] == year]

        if len(reduced_data) > 0 and len(target_data) > 0:
            reduced_value = reduced_data[reduced_carbon_col].iloc[0]
            target_value = target_data[target_col].iloc[0]

            gap = reduced_value - target_value
            gap_mt = gap * 1000  # Convert Gt to Mt
            gap_mt_rounded = math.ceil(gap_mt / 10) * 10  # Round up to nearest 10
            units = 'Mt'
            offset = max_value * 0.02

            # Draw vertical lines showing the gap
            ax.plot([year, year], [reduced_value - offset, target_value + offset],
                    color='#3498DB', linewidth=2, alpha=0.8)

            ax.plot([year - 0.5, year + 0.5], [reduced_value - offset, reduced_value - offset],
                    color='#3498DB', linewidth=2, alpha=0.8)

            ax.plot([year - 0.5, year + 0.5], [target_value + offset, target_value + offset],
                    color='#3498DB', linewidth=2, alpha=0.8)

            # Add gap annotation
            midpoint = (target_value + reduced_value) / 2
            gap_text = f'{gap_mt_rounded:.0f} {units} GAP\nIN {year}'

            ax.annotate(gap_text,
                        xy=(year - 2, midpoint), xytext=(year - 2, midpoint),
                        fontsize=10, ha='left', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#3498DB', alpha=0.9))

    return fig


def load_shipping_industry_data_cat(xlsx_path):
    """
    Load international shipping emissions data from Climate Action Tracker Excel file.
    Returns historical emissions and projection dataframes.

    Tab 1 (Assessment):
    - Row 20 (index 19): Historical emissions (voyage-based) 1990-2021
    - Row 22 (index 21): Current policy projections Max 2021-2050
    - Row 23 (index 22): Current policy projections Min 2021-2050

    Net-zero pathway:
    - Created as linear interpolation from 2021 historical value to 0 in 2050
    """
    # Read Tab 1 - Assessment
    df_assessment = pd.read_excel(xlsx_path, sheet_name='Assessment', header=None)

    # Row 20 (index 19): Historical voyage-based emissions 1990-2021
    historical_row = df_assessment.iloc[19, 4:].values
    # Years are 1990-2021 (32 years)
    historical_years = list(range(1990, 2022))
    historical_values = [val for val in historical_row if pd.notna(val)][:32]

    historical_df = pd.DataFrame({
        'year': historical_years,
        'emissions_mt': historical_values
    })

    # Convert to Gt (divide by 1000)
    historical_df['emissions_gt'] = historical_df['emissions_mt'] / 1000
    historical_df = historical_df[['year', 'emissions_gt']]

    # Get the 2021 historical value for interpolation
    historical_2021_value = historical_df[historical_df['year'] == 2021]['emissions_gt'].iloc[0]

    # Row 22-23 (indices 21-22): Current policy projections (Max and Min)
    # These start from column 35 (index 35) which corresponds to 2021
    max_row = df_assessment.iloc[21, 35:].values
    min_row = df_assessment.iloc[22, 35:].values

    # Years are 2021-2050 (30 years)
    projection_years = list(range(2021, 2051))
    max_values = [val for val in max_row if pd.notna(val)][:30]
    min_values = [val for val in min_row if pd.notna(val)][:30]

    # Calculate mean of max and min for current trajectory
    current_trajectory_mt = [(max_val + min_val) / 2 for max_val, min_val in zip(max_values, min_values)]

    # Filter to 2030 onwards, we'll interpolate 2021-2029
    future_years = list(range(2030, 2051))
    future_trajectory_mt = current_trajectory_mt[9:]  # Skip 2021-2029 (first 9 years)

    future_projections = pd.DataFrame({
        'year': future_years,
        'current_trajectory_mt': future_trajectory_mt
    })
    future_projections['current_trajectory_gt'] = future_projections['current_trajectory_mt'] / 1000

    # Create interpolated years 2021-2029 for current trajectory
    interpolated_years = pd.DataFrame({'year': range(2021, 2030)})

    # Interpolate current trajectory from 2021 to 2030
    current_trajectory_2030 = future_projections[future_projections['year'] == 2030]['current_trajectory_gt'].iloc[0]
    interpolated_years['current_trajectory_gt'] = interpolated_years['year'].apply(
        lambda y: historical_2021_value + (current_trajectory_2030 - historical_2021_value) * (y - 2021) / (2030 - 2021)
    )

    # Combine interpolated years with future projections
    projections_df = pd.concat([
        interpolated_years[['year', 'current_trajectory_gt']],
        future_projections[['year', 'current_trajectory_gt']]
    ], ignore_index=True)

    # Create net-zero pathway: linear from 2021 historical value to 0 in 2050
    netzero_years = list(range(2021, 2051))
    projections_df['net_zero_gt'] = projections_df['year'].apply(
        lambda y: historical_2021_value * (1 - (y - 2021) / (2050 - 2021))
    )

    # Create target dataframe (net-zero pathway)
    target_df = projections_df[['year', 'net_zero_gt']].copy()
    target_df.columns = ['year', 'target_gt']

    return historical_df, target_df, projections_df


def create_shipping_industry_gap_plot(historical_df, target_df, projections_df):
    """
    Create international shipping emissions gap visualization.
    Shows historical emissions, current trajectory projection, and net-zero pathway.
    Annotates the gap between current trajectory and net-zero pathway.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 8))

    # Get last historical point for connection
    last_historical_year = historical_df['year'].max()
    last_historical_value = historical_df.iloc[-1]['emissions_gt']

    # Plot historical data (solid line)
    ax.plot(historical_df['year'], historical_df['emissions_gt'],
            color='#2C3E50', linewidth=3, marker='o', markersize=6, solid_capstyle='round')

    # Plot current trajectory (dashed line)
    ax.plot(projections_df['year'], projections_df['current_trajectory_gt'],
            color='#E74C3C', linewidth=3, linestyle='--', marker='', markersize=6,
            solid_capstyle='round', alpha=0.7)

    # Plot net-zero pathway
    ax.plot(target_df['year'], target_df['target_gt'],
            color='#27AE60', linewidth=3, marker='', markersize=6,
            solid_capstyle='round')

    # Set axis limits
    min_year = 2000
    max_year = int(target_df['year'].max())
    ax.set_xlim(min_year, max_year + 0.5)

    # Calculate max for y-axis
    all_values = list(historical_df['emissions_gt'].values) + \
                 list(projections_df['current_trajectory_gt'].values) + \
                 list(target_df['target_gt'].values)
    max_value = max(all_values)

    ax.set_ylim(0, max_value * 1.15)

    # X-axis formatting
    tick_interval = 10
    ax.set_xticks(range(min_year, max_year + 1, tick_interval))
    ax.tick_params(axis='both', labelsize=12)

    # Y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

    # Add text annotations
    mid_historical = int((min_year + last_historical_year) / 2)
    ax.text(mid_historical, max_value, 'PAST\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', color='#2C3E50')

    mid_projection = int((last_historical_year + max_year) / 2)
    ax.text(mid_projection - 5, max_value, 'PROJECTED\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.6, color='#2C3E50')

    ax.text(2040, max_value * 0.88, 'CURRENT POLICIES',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#E74C3C')

    ax.text(2040, max_value * 0.45, 'NET-ZERO PATHWAY',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#27AE60')

    # Annotate the gap between current trajectory and net-zero pathway
    gap_years = [2030, 2050]

    for year in gap_years:
        trajectory_data = projections_df[projections_df['year'] == year]
        target_data = target_df[target_df['year'] == year]

        if len(trajectory_data) > 0 and len(target_data) > 0:
            trajectory_value = trajectory_data['current_trajectory_gt'].iloc[0]
            target_value = target_data['target_gt'].iloc[0]

            gap = trajectory_value - target_value
            gap_mt = gap * 1000  # Convert Gt to Mt
            gap_mt_rounded = math.ceil(gap_mt / 10) * 10  # Round up to nearest 10
            units = 'Mt'
            offset = max_value * 0.02

            # Draw vertical lines showing the gap
            ax.plot([year, year], [trajectory_value - offset, target_value + offset],
                    color='#E74C3C', linewidth=2, alpha=0.8)

            ax.plot([year - 0.5, year + 0.5], [trajectory_value - offset, trajectory_value - offset],
                    color='#E74C3C', linewidth=2, alpha=0.8)

            ax.plot([year - 0.5, year + 0.5], [target_value + offset, target_value + offset],
                    color='#E74C3C', linewidth=2, alpha=0.8)

            # Add gap annotation
            midpoint = (target_value + trajectory_value) / 2
            gap_text = f'{gap_mt_rounded:.0f} {units} GAP\nIN {year}'

            ax.annotate(gap_text,
                        xy=(year - 2, midpoint), xytext=(year - 2, midpoint),
                        fontsize=10, ha='left', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#E74C3C', alpha=0.9))

    return fig


def main():
    """
    Main execution function for company emissions gap visualization.
    """

    # Aviation industry emissions gap using Bergero 2023 data
    print("\n" + "="*60)
    print("Creating Aviation Industry Emissions Gap visualization...")
    print("="*60)

    aviation_xlsx_path = 'data/corporate_targets/bergero_2023_emissions.xlsx'
    historical_df, target_df, projections_df = load_aviation_industry_data_bergero(aviation_xlsx_path)

    print(f"\nHistorical emissions data: {len(historical_df)} years ({historical_df['year'].min():.0f} - {historical_df['year'].max():.0f})")
    print(f"Target pathway data: {len(target_df)} years ({target_df['year'].min():.0f} - {target_df['year'].max():.0f})")
    print(f"Projection data: {len(projections_df)} years ({projections_df['year'].min():.0f} - {projections_df['year'].max():.0f})")

    # Create aviation industry visualization
    print("\nCreating aviation industry visualization...")
    fig_aviation = create_aviation_industry_gap_plot(historical_df, target_df, projections_df)

    # Format plot
    format_plot_title(plt.gca(),
                     "",
                     "GLOBAL AVIATION INDUSTRY CO\N{SUBSCRIPT TWO} EMISSIONS (GIGATONNES)",
                     None)
    add_deep_sky_branding(plt.gca(), None,
                         "DATA: BERGERO ET AL. (2023) PATHWAYS TO NET-ZERO EMISSIONS FROM AVIATION.\nProjections are means of Business-As-Usual and Industry Projection scenarios.",
                         analysis_date=datetime.datetime.now())

    # Save the plot
    aviation_save_path = 'figures/aviation_industry_emissions_gap.png'
    save_plot(fig_aviation, aviation_save_path)

    print(f"\nAviation industry emissions gap plot saved to {aviation_save_path}")

    # International Shipping industry emissions gap using CAT data
    print("\n" + "="*60)
    print("Creating International Shipping Emissions Gap visualization...")
    print("="*60)

    shipping_xlsx_path = 'data/corporate_targets/cat_international_shipping.xlsx'
    shipping_historical_df, shipping_target_df, shipping_projections_df = load_shipping_industry_data_cat(shipping_xlsx_path)

    print(f"\nHistorical emissions data: {len(shipping_historical_df)} years ({shipping_historical_df['year'].min():.0f} - {shipping_historical_df['year'].max():.0f})")
    print(f"Target pathway data: {len(shipping_target_df)} years ({shipping_target_df['year'].min():.0f} - {shipping_target_df['year'].max():.0f})")
    print(f"Projection data: {len(shipping_projections_df)} years ({shipping_projections_df['year'].min():.0f} - {shipping_projections_df['year'].max():.0f})")

    # Create shipping industry visualization
    print("\nCreating shipping industry visualization...")
    fig_shipping = create_shipping_industry_gap_plot(shipping_historical_df, shipping_target_df, shipping_projections_df)

    # Format plot
    format_plot_title(plt.gca(),
                     "",
                     "MARITIME SHIPPING CO\N{SUBSCRIPT TWO} EMISSIONS (GIGATONNES)",
                     None)
    add_deep_sky_branding(plt.gca(), None,
                         "DATA: CLIMATE ACTION TRACKER - INTERNATIONAL SHIPPING ASSESSMENT (2023).",
                         analysis_date=datetime.datetime.now())

    # Save the plot
    shipping_save_path = 'figures/shipping_industry_emissions_gap.png'
    save_plot(fig_shipping, shipping_save_path)

    print(f"\nShipping industry emissions gap plot saved to {shipping_save_path}")


if __name__ == "__main__":
    main()

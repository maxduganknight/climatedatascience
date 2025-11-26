import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob

# Add reports directory to Python path to import shared utilities
reports_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, reports_dir)

# Import shared utilities
from utils import (
    setup_enhanced_plot, format_plot_title, add_deep_sky_branding,
    save_plot, COLORS
)

def load_cpi_data():
    """
    Load CPI data for inflation adjustment.

    Returns:
    --------
    pandas DataFrame
        DataFrame with year and cpi columns
    """
    cpi_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'cpi', 'cpi.csv')
    cpi_df = pd.read_csv(cpi_path)

    # Clean up any BOM characters
    cpi_df.columns = [col.replace('\ufeff', '') for col in cpi_df.columns]

    return cpi_df

def adjust_for_inflation(df, amount_cols, year_col='year', base_year=2020):
    """
    Adjust dollar amounts for inflation using CPI data.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data to adjust
    amount_cols : list
        List of column names containing dollar amounts
    year_col : str
        Name of the year column
    base_year : int
        Base year to adjust all values to (default: 2020)

    Returns:
    --------
    pandas DataFrame
        DataFrame with new inflation-adjusted columns
    """
    # Load CPI data
    cpi_df = load_cpi_data()

    # Get base year CPI
    base_cpi = cpi_df[cpi_df['year'] == base_year]['cpi'].iloc[0]

    # Merge CPI data with the main dataframe
    df_with_cpi = df.merge(cpi_df, on=year_col, how='left')

    # Create inflation-adjusted columns
    for col in amount_cols:
        if col in df.columns:
            # Calculate inflation factor (base_year_cpi / current_year_cpi)
            inflation_factor = base_cpi / df_with_cpi['cpi']
            # Create new column name
            adj_col_name = f"{col}_adj_{base_year}"
            # Apply adjustment
            df_with_cpi[adj_col_name] = df_with_cpi[col] * inflation_factor

    return df_with_cpi

def find_latest_nfip_file(data_type):
    """
    Find the most recent NFIP data file based on timestamp in filename.

    Parameters:
    -----------
    data_type : str
        Either 'policies' or 'claims'

    Returns:
    --------
    str
        Path to the most recent file
    """
    pattern = f"nfip/florida_nfip_{data_type}_*.csv"
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No {data_type} files found matching pattern: {pattern}")

    # Sort files by name (which includes timestamp) to get the most recent
    latest_file = sorted(files)[-1]
    print(f"Using latest {data_type} file: {latest_file}")

    return latest_file

def filter_owner_occupied_residential(df, occupancy_col='occupancyType'):
    """
    Filter NFIP data to approximate owner-occupied residential homeowners,
    excluding tenant and condo policies to match the scope of fl_insurance_processor.py.
    
    1 = single family residence; 
    2 = 2 to 4 unit residential building; 
    3 = residential building with more than 4 units; 
    4 = Non-residential building; 
    6 = Non Residential - Business; 
    11 = Single-family residential building with the exception of a mobile home or a single residential unit within a multi unit building; 
    12 = A residential non-condo building with 2, 3, or 4 units seeking insurance on all units; 
    13 = A residential non-condo building with 5 or more units seeking insurance on all units; 
    14 = Residential mobile/manufactured home; 
    15 = Residential condo association seeking coverage on a building with one or more units; 
    16 = Single residential unit within a multi-unit building; 
    17 = Non-residential mobile/manufactured home; 
    18 = A non-residential building; 
    19 = a non-residential unit within a multi-unit building;	
    """
    print(f"Original dataset shape: {df.shape}")

    # Focus on owner-occupied residential - primary residence single family homes
    # Type 1 (Single Family) are most similar to owner-occupied residential homeowners
    residential_types = [1, 2, 3, 11, 12, 13, 16]

    filtered_df = df[df[occupancy_col].isin(residential_types)].copy()

    # For policies, also filter by primaryResidenceIndicator if available
    if 'primaryResidenceIndicator' in df.columns:
        filtered_df = filtered_df[filtered_df['primaryResidenceIndicator'] == True].copy()
        print(f"Filtered to primary residence single family homes: {filtered_df.shape}")
    else:
        print(f"Filtered to single family homes: {filtered_df.shape}")

    return filtered_df

def load_and_process_nfip_policies(file_path):
    """
    Load and process NFIP policies data.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing NFIP policies data

    Returns:
    --------
    pandas DataFrame
        Processed DataFrame with cleaned data and extracted years
    """
    print(f"Loading NFIP policies data from: {file_path}")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Filter to owner-occupied residential (similar to fl_insurance_processor focus)
    df = filter_owner_occupied_residential(df)

    # Extract year from policyEffectiveDate
    df['policyEffectiveDate'] = pd.to_datetime(df['policyEffectiveDate'])
    df['year'] = df['policyEffectiveDate'].dt.year

    # Filter out 2025 data (incomplete year)
    df = df[df['year'] < 2025].copy()

    # Clean numeric columns
    numeric_columns = [
        'policyCost', 'totalInsurancePremiumOfThePolicy', 'policyCount',
        'totalBuildingInsuranceCoverage', 'totalContentsInsuranceCoverage'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate total coverage if both building and contents are available
    if 'totalBuildingInsuranceCoverage' in df.columns and 'totalContentsInsuranceCoverage' in df.columns:
        df['total_coverage'] = df['totalBuildingInsuranceCoverage'].fillna(0) + df['totalContentsInsuranceCoverage'].fillna(0)

    # Apply inflation adjustment to dollar amounts
    dollar_columns = [
        'policyCost', 'totalInsurancePremiumOfThePolicy',
        'totalBuildingInsuranceCoverage', 'totalContentsInsuranceCoverage', 'total_coverage'
    ]
    available_dollar_columns = [col for col in dollar_columns if col in df.columns]

    if available_dollar_columns:
        df = adjust_for_inflation(df, available_dollar_columns)

    print(f"Processed {len(df)} NFIP policies from {df['year'].min()} to {df['year'].max()}")

    return df

def load_and_process_nfip_claims(file_path):
    """
    Load and process NFIP claims data.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing NFIP claims data

    Returns:
    --------
    pandas DataFrame
        Processed DataFrame with cleaned data and extracted years
    """
    print(f"Loading NFIP claims data from: {file_path}")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Filter to owner-occupied residential (similar to fl_insurance_processor focus)
    df = filter_owner_occupied_residential(df)

    # Use yearOfLoss for grouping
    df['year'] = df['yearOfLoss']

    # Filter out 2025 data (incomplete year)
    df = df[df['year'] < 2025].copy()

    # Clean numeric columns
    numeric_columns = [
        'amountPaidOnBuildingClaim', 'amountPaidOnContentsClaim',
        'buildingDamageAmount', 'contentsDamageAmount'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculate total paid and total damage
    df['total_paid'] = df['amountPaidOnBuildingClaim'] + df['amountPaidOnContentsClaim']
    df['total_damage'] = df['buildingDamageAmount'] + df['contentsDamageAmount']

    # Apply inflation adjustment to dollar amounts
    dollar_columns = [
        'amountPaidOnBuildingClaim', 'amountPaidOnContentsClaim',
        'buildingDamageAmount', 'contentsDamageAmount', 'total_paid', 'total_damage'
    ]
    available_dollar_columns = [col for col in dollar_columns if col in df.columns]

    if available_dollar_columns:
        df = adjust_for_inflation(df, available_dollar_columns)

    print(f"Processed {len(df)} NFIP claims from {df['year'].min()} to {df['year'].max()}")

    return df

def aggregate_policies_by_year(df):
    """
    Aggregate NFIP policies by year.
    """
    agg_dict = {
        'policyCount': 'sum',
        'policyCost': ['sum', 'mean'],
        'totalInsurancePremiumOfThePolicy': ['sum', 'mean']
    }

    # Add inflation-adjusted columns if available
    if 'policyCost_adj_2020' in df.columns:
        agg_dict['policyCost_adj_2020'] = ['sum', 'mean']
    if 'totalInsurancePremiumOfThePolicy_adj_2020' in df.columns:
        agg_dict['totalInsurancePremiumOfThePolicy_adj_2020'] = ['sum', 'mean']

    # Add coverage aggregation if available
    if 'total_coverage' in df.columns:
        agg_dict['total_coverage'] = 'sum'
    if 'total_coverage_adj_2020' in df.columns:
        agg_dict['total_coverage_adj_2020'] = 'sum'

    yearly_stats = df.groupby('year').agg(agg_dict).reset_index()

    # Flatten column names
    base_columns = [
        'year', 'total_policies', 'total_policy_cost', 'avg_policy_cost',
        'total_premiums', 'avg_premium'
    ]

    # Add inflation-adjusted columns
    if 'policyCost_adj_2020' in df.columns:
        base_columns.extend(['total_policy_cost_adj_2020', 'avg_policy_cost_adj_2020'])
    if 'totalInsurancePremiumOfThePolicy_adj_2020' in df.columns:
        base_columns.extend(['total_premiums_adj_2020', 'avg_premium_adj_2020'])

    if 'total_coverage' in df.columns:
        base_columns.append('total_coverage_amount')
    if 'total_coverage_adj_2020' in df.columns:
        base_columns.append('total_coverage_amount_adj_2020')

    yearly_stats.columns = base_columns

    return yearly_stats

def aggregate_claims_by_year(df):
    """
    Aggregate NFIP claims by year.
    """
    agg_dict = {
        'amountPaidOnBuildingClaim': 'sum',
        'amountPaidOnContentsClaim': 'sum',
        'total_paid': 'sum',
        'buildingDamageAmount': 'sum',
        'contentsDamageAmount': 'sum',
        'total_damage': 'sum',
        'id': 'count'  # Count of claims
    }

    # Add inflation-adjusted columns if available
    if 'amountPaidOnBuildingClaim_adj_2020' in df.columns:
        agg_dict['amountPaidOnBuildingClaim_adj_2020'] = 'sum'
    if 'amountPaidOnContentsClaim_adj_2020' in df.columns:
        agg_dict['amountPaidOnContentsClaim_adj_2020'] = 'sum'
    if 'total_paid_adj_2020' in df.columns:
        agg_dict['total_paid_adj_2020'] = 'sum'
    if 'buildingDamageAmount_adj_2020' in df.columns:
        agg_dict['buildingDamageAmount_adj_2020'] = 'sum'
    if 'contentsDamageAmount_adj_2020' in df.columns:
        agg_dict['contentsDamageAmount_adj_2020'] = 'sum'
    if 'total_damage_adj_2020' in df.columns:
        agg_dict['total_damage_adj_2020'] = 'sum'

    yearly_stats = df.groupby('year').agg(agg_dict).reset_index()

    # Flatten column names
    base_columns = [
        'year', 'building_paid', 'contents_paid', 'total_paid',
        'building_damage', 'contents_damage', 'total_damage', 'claim_count'
    ]

    # Add inflation-adjusted columns
    if 'amountPaidOnBuildingClaim_adj_2020' in df.columns:
        base_columns.append('building_paid_adj_2020')
    if 'amountPaidOnContentsClaim_adj_2020' in df.columns:
        base_columns.append('contents_paid_adj_2020')
    if 'total_paid_adj_2020' in df.columns:
        base_columns.append('total_paid_adj_2020')
    if 'buildingDamageAmount_adj_2020' in df.columns:
        base_columns.append('building_damage_adj_2020')
    if 'contentsDamageAmount_adj_2020' in df.columns:
        base_columns.append('contents_damage_adj_2020')
    if 'total_damage_adj_2020' in df.columns:
        base_columns.append('total_damage_adj_2020')

    yearly_stats.columns = base_columns

    return yearly_stats

def create_nfip_line_chart(df, y_col, title, subtitle, data_note, unit='count', save_path=None):
    """
    Create a line chart for NFIP data following the same pattern as fl_insurance_processor.
    """
    # Set up the plot with common styling
    fig, ax, font_props = setup_enhanced_plot()

    # For large dollar amounts, convert to billions for better display
    plot_data = df.copy()
    y_values = plot_data[y_col]

    if unit == 'dollar_billions':
        # Convert to billions
        plot_data[y_col] = y_values / 1e9
        y_values = plot_data[y_col]

    # Create line plot with markers
    plt.plot(plot_data['year'], plot_data[y_col],
             color=COLORS['primary'],
             linewidth=3,
             marker='o',
             markersize=8,
             markerfacecolor=COLORS['primary'],
             markeredgecolor='white',
             markeredgewidth=2)

    # Format the plot title
    format_plot_title(ax, title, subtitle, font_props)

    # Format axes
    font_prop = font_props.get('regular') if font_props else None
    ax.set_xticks(plot_data['year'])
    ax.set_xticklabels(plot_data['year'], fontproperties=font_prop, fontsize=14)

    # Format y-axis
    if unit == 'dollar':
        ax.yaxis.set_major_formatter('${x:,.0f}')
    elif unit == 'dollar_billions':
        ax.yaxis.set_major_formatter('${x:.1f}B')
    elif unit == 'count':
        ax.yaxis.set_major_formatter('{x:,.0f}')

    ax.tick_params(axis='y', labelsize=14)

    # Add branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)

    # Save the plot if a path is provided
    save_plot(fig, save_path)

    return fig

def create_policies_with_coverage_chart(df, title, subtitle, data_note, save_path=None, use_inflation_adjusted=False):
    """
    Create a dual-axis chart showing policy count (line) and total coverage amount (bars).

    Parameters:
    -----------
    use_inflation_adjusted : bool, optional
        Whether to use inflation-adjusted coverage amounts (default: False)
    """
    # Set up the plot with common styling
    fig, ax1, font_props = setup_enhanced_plot()

    # Choose coverage column based on inflation adjustment preference
    if use_inflation_adjusted and 'total_coverage_amount_adj_2020' in df.columns:
        coverage_col = 'total_coverage_amount_adj_2020'
    else:
        coverage_col = 'total_coverage_amount'

    # Check if coverage data is available
    if coverage_col not in df.columns:
        print(f"Warning: Coverage data ({coverage_col}) not available, creating simple policy count chart")
        return create_nfip_line_chart(df, 'total_policies', title, subtitle, data_note, unit='count', save_path=save_path)

    # Create bars for coverage amount (secondary axis - left side)
    ax2 = ax1.twinx()

    # Convert coverage to billions for display
    coverage_billions = df[coverage_col] / 1e9

    bars = ax2.bar(df['year'], coverage_billions,
                  color=COLORS['comparison'],
                  alpha=0.6,
                  width=0.6,
                  label='Total Coverage Amount')

    # Plot policy count line on primary axis (right side)
    line = ax1.plot(df['year'], df['total_policies'],
                   color=COLORS['primary'],
                   linewidth=3,
                   marker='o',
                   markersize=8,
                   markerfacecolor=COLORS['primary'],
                   markeredgecolor='white',
                   markeredgewidth=2,
                   label='Number of Policies')

    # Format the plot title
    format_plot_title(ax1, title, subtitle, font_props)

    # Format axes
    font_prop = font_props.get('regular') if font_props else None

    # X-axis formatting
    ax1.set_xticks(df['year'])
    ax1.set_xticklabels(df['year'], fontproperties=font_prop, fontsize=14)

    # Primary y-axis (right side) - Policy count
    ax1.yaxis.set_major_formatter('{x:,.0f}')
    ax1.tick_params(axis='y', labelsize=14, colors=COLORS['primary'])
    ax1.set_ylabel('Number of Policies', color=COLORS['primary'], fontproperties=font_prop, fontsize=14)
    ax1.yaxis.set_label_position('right')
    ax1.yaxis.tick_right()

    # Secondary y-axis (left side) - Coverage amount
    ax2.yaxis.set_major_formatter('${x:.1f}B')
    ax2.tick_params(axis='y', labelsize=14, colors=COLORS['comparison'])

    # Adjust label based on inflation adjustment
    if use_inflation_adjusted and 'total_coverage_amount_adj_2020' in df.columns:
        ylabel = 'Total Coverage Amount (Billions 2020 USD)'
        legend_label = 'Total Coverage Amount (2020 USD)'
    else:
        ylabel = 'Total Coverage Amount (Billions USD)'
        legend_label = 'Total Coverage Amount'

    ax2.set_ylabel(ylabel, color=COLORS['comparison'], fontproperties=font_prop, fontsize=14)
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.tick_left()

    # Create legend
    lines = line
    bars_artist = bars[0] if bars else None

    if bars_artist:
        legend_elements = [bars_artist, lines[0]]
        legend_labels = [legend_label, 'Number of Policies']

        ax1.legend(legend_elements, legend_labels,
                  fontsize=14,
                  frameon=True,
                  facecolor=COLORS['background'],
                  edgecolor='#DDDDDD',
                  loc='upper left',
                  prop=font_prop)

    # Add branding
    add_deep_sky_branding(ax1, font_props, data_note=data_note)

    # Save the plot if a path is provided
    save_plot(fig, save_path)

    return fig

def analyze_hurricane_impacts(claims_df, damage_mode=False):
    """
    Analyze claims by hurricane to get payout amounts or damage amounts for major storms.

    Parameters:
    -----------
    damage_mode : bool
        If True, calculates damage amounts instead of payouts

    Returns:
    --------
    dict
        Dictionary with hurricane names as keys and total amounts as values
    """
    hurricane_impacts = {}

    # Major hurricanes to analyze
    hurricanes_to_check = [
        'Hurricane Ian', 'Hurricane Helene'
    ]

    amount_col = 'total_damage' if damage_mode else 'total_paid'
    amount_type = 'damage' if damage_mode else 'payouts'

    for hurricane in hurricanes_to_check:
        if 'floodEvent' in claims_df.columns:
            hurricane_claims = claims_df[claims_df['floodEvent'] == hurricane]
            if len(hurricane_claims) > 0:
                total_amount = hurricane_claims[amount_col].sum()
                hurricane_impacts[hurricane] = total_amount
                print(f"{hurricane}: ${total_amount:,.0f} in total {amount_type} ({len(hurricane_claims):,} claims)")

    return hurricane_impacts

def create_nfip_bar_chart_with_annotations(df, y_col, title, subtitle, data_note, claims_df, unit='dollar_billions', save_path=None, damage_mode=False):
    """
    Create a bar chart for NFIP claims data with hurricane annotations for 2022 and 2024.

    Parameters:
    -----------
    damage_mode : bool
        If True, uses damage amounts instead of payouts for hurricane calculations
    """
    # Set up the plot with common styling
    fig, ax, font_props = setup_enhanced_plot()

    # For large dollar amounts, convert to billions for better display
    plot_data = df.copy()
    y_values = plot_data[y_col]

    if unit == 'dollar_billions':
        # Convert to billions
        plot_data[y_col] = y_values / 1e9
        y_values = plot_data[y_col]

    # Create bar chart
    plt.bar(plot_data['year'], plot_data[y_col],
             color=COLORS['primary'],
             width=0.7,
             alpha=0.8)

    # Analyze hurricane impacts
    hurricane_impacts = analyze_hurricane_impacts(claims_df, damage_mode=damage_mode)

    # Add annotations for 2022 and 2024 with hurricane information
    font_prop = font_props.get('regular') if font_props else None

    # 2022 annotation (Hurricane Ian)
    if 2022 in plot_data['year'].values:
        bar_2022 = plot_data[plot_data['year'] == 2022]
        if len(bar_2022) > 0:
            bar_height = bar_2022[y_col].iloc[0]

            # Get Hurricane Ian impact
            ian_payout = hurricane_impacts.get('Hurricane Ian', 0) / 1e9  # Convert to billions
            annotation_text = f"Hurricane Ian: ${ian_payout:.1f}B"

            ax.annotate(annotation_text,
                       xy=(2022, bar_height),
                       xytext=(2022, bar_height + bar_height * 0.15),
                       ha='center', va='bottom',
                       fontsize=10, fontproperties=font_prop,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec=COLORS['primary']),
                       arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))

    # 2024 annotation (Hurricane Milton and Helene)
    if 2024 in plot_data['year'].values:
        bar_2024 = plot_data[plot_data['year'] == 2024]
        if len(bar_2024) > 0:
            bar_height = bar_2024[y_col].iloc[0]

            # Get Hurricane Milton and Helene impacts
            helene_payout = hurricane_impacts.get('Hurricane Helene', 0) / 1e9

            annotation_text = f"Hurricane Helene: ${helene_payout:.1f}B"

            ax.annotate(annotation_text,
                       xy=(2024, bar_height),
                       xytext=(2024, bar_height + bar_height * 0.15),
                       ha='center', va='bottom',
                       fontsize=10, fontproperties=font_prop,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec=COLORS['primary']),
                       arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))

    # Format the plot title
    format_plot_title(ax, title, subtitle, font_props)

    # Format axes
    ax.set_xticks(plot_data['year'])
    ax.set_xticklabels(plot_data['year'], fontproperties=font_prop, fontsize=14)

    # Format y-axis
    if unit == 'dollar_billions':
        ax.yaxis.set_major_formatter('${x:.1f}B')
    elif unit == 'dollar':
        ax.yaxis.set_major_formatter('${x:,.0f}')
    elif unit == 'count':
        ax.yaxis.set_major_formatter('{x:,.0f}')

    ax.tick_params(axis='y', labelsize=14)

    # Add branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)

    # Save the plot if a path is provided
    save_plot(fig, save_path)

    return fig

def main():
    """
    Main function to process NFIP data and create charts.
    """

    # Find the most recent data files
    policies_path = find_latest_nfip_file('policies')
    claims_path = find_latest_nfip_file('claims')

    # Create output directory
    output_dir = 'figures/nfip'
    os.makedirs(output_dir, exist_ok=True)

    # Process policies data
    print("Processing NFIP policies data...")
    policies_df = load_and_process_nfip_policies(policies_path)
    policies_yearly = aggregate_policies_by_year(policies_df)

    # Process claims data
    print("Processing NFIP claims data...")
    claims_df = load_and_process_nfip_claims(claims_path)
    claims_yearly = aggregate_claims_by_year(claims_df)
    claims_yearly.to_csv('nfip/florida_nfip_claims_yearly.csv')
    print("Creating NFIP charts...")

    # Policies Analysis Charts

    # 1. Number of NFIP policies over time with coverage amounts (inflation-adjusted)
    create_policies_with_coverage_chart(
        policies_yearly,
        "FLORIDA NFIP FLOOD INSURANCE POLICIES (2020 DOLLARS)",
        "NUMBER OF ACTIVE NFIP POLICIES AND TOTAL COVERAGE AMOUNTS (INFLATION-ADJUSTED)",
        "DATA: FEMA NFIP | RESIDENTIAL BUILDINGS EXCL. CONDO & MOBILE | ADJUSTED TO 2020 DOLLARS",
        save_path=f"{output_dir}/fl_nfip_policies_count_adj.png",
        use_inflation_adjusted=True
    )

    # 2. Average premium cost over time (inflation-adjusted)
    premium_col = 'avg_premium_adj_2020' if 'avg_premium_adj_2020' in policies_yearly.columns else 'avg_premium'
    create_nfip_line_chart(
        policies_yearly,
        premium_col,
        "FLORIDA NFIP AVERAGE PREMIUM COSTS (2020 DOLLARS)",
        "AVERAGE ANNUAL INSURANCE PREMIUM FOR NFIP POLICIES (INFLATION-ADJUSTED)",
        "DATA: FEMA NFIP | RESIDENTIAL BUILDINGS EXCL. CONDO & MOBILE | ADJUSTED TO 2020 DOLLARS",
        unit='dollar',
        save_path=f"{output_dir}/fl_nfip_avg_premium_adj.png"
    )

    # 3. Average policy cost over time (inflation-adjusted)
    policy_cost_col = 'avg_policy_cost_adj_2020' if 'avg_policy_cost_adj_2020' in policies_yearly.columns else 'avg_policy_cost'
    create_nfip_line_chart(
        policies_yearly,
        policy_cost_col,
        "FLORIDA NFIP AVERAGE POLICY COSTS (2020 DOLLARS)",
        "AVERAGE TOTAL POLICY COST FOR NFIP POLICIES (INFLATION-ADJUSTED)",
        "DATA: FEMA NFIP | RESIDENTIAL BUILDINGS EXCL. CONDO & MOBILE | ADJUSTED TO 2020 DOLLARS",
        unit='dollar',
        save_path=f"{output_dir}/fl_nfip_avg_policy_cost_adj.png"
    )

    # Claims Analysis Charts

    # 4. Total amount paid on claims over time (inflation-adjusted)
    total_paid_col = 'total_paid_adj_2020' if 'total_paid_adj_2020' in claims_yearly.columns else 'total_paid'
    create_nfip_bar_chart_with_annotations(
        claims_yearly,
        total_paid_col,
        "FLORIDA SEEING RECORD FLOOD INSURANCE PAYOUTS (2020 DOLLARS)",
        "TOTAL AMOUNT PAID ON NFIP FLOOD CLAIMS (INFLATION-ADJUSTED)",
        "DATA: FEMA NFIP | RESIDENTIAL BUILDINGS EXCL. CONDO & MOBILE | ADJUSTED TO 2020 DOLLARS",
        claims_df,  # Pass the original claims dataframe for hurricane analysis
        unit='dollar_billions',
        save_path=f"{output_dir}/fl_nfip_total_paid_adj.png"
    )

    # 5. Total damage amount over time (inflation-adjusted)
    total_damage_col = 'total_damage_adj_2020' if 'total_damage_adj_2020' in claims_yearly.columns else 'total_damage'
    create_nfip_bar_chart_with_annotations(
        claims_yearly,
        total_damage_col,
        "FLORIDA SEEING RECORD FLOOD DAMAGES (2020 DOLLARS)",
        "TOTAL REPORTED DAMAGE FROM NFIP FLOOD CLAIMS (INFLATION-ADJUSTED)",
        "DATA: FEMA NFIP | RESIDENTIAL BUILDINGS EXCL. CONDO & MOBILE | ADJUSTED TO 2020 DOLLARS",
        claims_df,  # Pass the original claims dataframe for hurricane analysis
        unit='dollar_billions',
        save_path=f"{output_dir}/fl_nfip_total_damage_adj.png",
        damage_mode=True  # Use damage amounts instead of payouts
    )

    # 6. Building vs Contents Claims Payouts (inflation-adjusted)
    building_paid_col = 'building_paid_adj_2020' if 'building_paid_adj_2020' in claims_yearly.columns else 'building_paid'
    create_nfip_line_chart(
        claims_yearly,
        building_paid_col,
        "FLORIDA NFIP BUILDING CLAIMS PAYOUTS (2020 DOLLARS)",
        "AMOUNT PAID ON BUILDING DAMAGE CLAIMS (INFLATION-ADJUSTED)",
        "DATA: FEMA NFIP | RESIDENTIAL BUILDINGS EXCL. CONDO & MOBILE | ADJUSTED TO 2020 DOLLARS",
        unit='dollar_billions',
        save_path=f"{output_dir}/fl_nfip_building_paid_adj.png"
    )

    contents_paid_col = 'contents_paid_adj_2020' if 'contents_paid_adj_2020' in claims_yearly.columns else 'contents_paid'
    create_nfip_line_chart(
        claims_yearly,
        contents_paid_col,
        "FLORIDA NFIP CONTENTS CLAIMS PAYOUTS (2020 DOLLARS)",
        "AMOUNT PAID ON CONTENTS DAMAGE CLAIMS (INFLATION-ADJUSTED)",
        "DATA: FEMA NFIP | RESIDENTIAL BUILDINGS EXCL. CONDO & MOBILE | ADJUSTED TO 2020 DOLLARS",
        unit='dollar_billions',
        save_path=f"{output_dir}/fl_nfip_contents_paid_adj.png"
    )

    print("All NFIP charts created successfully!")

    # Print summary statistics
    print("\n=== NFIP SUMMARY STATISTICS ===\n")

    print("POLICIES SUMMARY:")
    if not policies_yearly.empty:
        latest_year = policies_yearly['year'].max()
        latest_policies = policies_yearly[policies_yearly['year'] == latest_year]
        print(f"Latest year: {latest_year}")
        print(f"Total policies: {latest_policies['total_policies'].iloc[0]:,.0f}")
        print(f"Average premium: ${latest_policies['avg_premium'].iloc[0]:,.0f}")
        print(f"Average policy cost: ${latest_policies['avg_policy_cost'].iloc[0]:,.0f}")

        # Calculate growth from first to last year
        first_year = policies_yearly['year'].min()
        first_policies = policies_yearly[policies_yearly['year'] == first_year]
        if len(first_policies) > 0:
            policy_growth = ((latest_policies['total_policies'].iloc[0] / first_policies['total_policies'].iloc[0]) - 1) * 100
            premium_growth = ((latest_policies['avg_premium'].iloc[0] / first_policies['avg_premium'].iloc[0]) - 1) * 100
            print(f"Policy count change ({first_year}-{latest_year}): {policy_growth:.1f}%")
            print(f"Premium change ({first_year}-{latest_year}): {premium_growth:.1f}%")

    print("\nCLAIMS SUMMARY:")
    if not claims_yearly.empty:
        latest_year = claims_yearly['year'].max()
        latest_claims = claims_yearly[claims_yearly['year'] == latest_year]
        print(f"Latest year: {latest_year}")
        print(f"Total claims: {latest_claims['claim_count'].iloc[0]:,.0f}")
        print(f"Total paid out: ${latest_claims['total_paid'].iloc[0]:,.0f}")
        print(f"Total damage: ${latest_claims['total_damage'].iloc[0]:,.0f}")

        # Show totals across all years
        total_claims = claims_yearly['claim_count'].sum()
        total_paid_all = claims_yearly['total_paid'].sum()
        total_damage_all = claims_yearly['total_damage'].sum()
        print(f"\nALL YEARS TOTAL:")
        print(f"Total claims: {total_claims:,.0f}")
        print(f"Total paid out: ${total_paid_all:,.0f}")
        print(f"Total damage: ${total_damage_all:,.0f}")

if __name__ == "__main__":
    main()
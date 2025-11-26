import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

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

def load_and_process_florida_data(file_path):
    """
    Load and process Florida homeowners insurance data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing Florida insurance data
    
    Returns:
    --------
    pandas DataFrame
        Processed DataFrame with cleaned column names and data types
    """
    print(f"Loading Florida insurance data from: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Clean up the first column name (remove BOM)
    df.columns = [col.replace('\ufeff', '') for col in df.columns]
    
    # Extract year from index column
    df['year'] = df.iloc[:, 0].astype(int)
    
    # Clean numeric columns by removing commas and converting to numeric
    numeric_columns = [
        'Policies in force', 'Number of policies canceled', 'Number of policies nonrenewed',
        'Number of policies canceled due to hurricane risk', 'Number of policies nonrenewed due to hurricane risk',
        'Number of new policies written', 'Total premiums written', 'Number of policies transferred to other insurers'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    # Calculate average premium if not present
    if 'average_premium' not in df.columns:
        if 'Total premiums written' in df.columns and 'Policies in force' in df.columns:
            df['average_premium'] = df['Total premiums written'] / df['Policies in force']

    # Apply inflation adjustment to dollar amounts
    dollar_columns = ['Total premiums written', 'average_premium']
    available_dollar_columns = [col for col in dollar_columns if col in df.columns]

    if available_dollar_columns:
        df = adjust_for_inflation(df, available_dollar_columns)

    print(f"Processed {len(df)} years of data from {df['year'].min()} to {df['year'].max()}")

    return df

def create_policies_chart(df, title_suffix, save_path, custom_title=None, custom_subtitle=None):
    """Create chart for total policies in force over time.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the insurance data
    title_suffix : str
        Suffix to add to default subtitle (e.g., "across all Florida insurers")
    save_path : str
        Path to save the chart
    custom_title : str, optional
        Custom title to override the default
    custom_subtitle : str, optional
        Custom subtitle to override the default
    """
    chart_data = df[['year', 'Policies in force']].copy()
    
    # Use custom title/subtitle if provided, otherwise use defaults
    title = custom_title if custom_title else "FLORIDA HOMEOWNERS INSURANCE POLICIES IN FORCE"
    subtitle = custom_subtitle if custom_subtitle else f"TOTAL NUMBER OF ACTIVE HOMEOWNERS INSURANCE POLICIES {title_suffix.upper()}"
    data_note = "DATA: FLORIDA OFFICE OF INSURANCE REGULATION | OWNER-OCCUPIED RESIDENTIAL HOMEOWNERS (EXCL. TENANT AND CONDO)"
    
    return plot_line_custom(
        chart_data.rename(columns={'Policies in force': 'policies_in_force'}),
        'policies_in_force',
        title,
        subtitle,
        data_note,
        unit='count',
        save_path=save_path
    )

def create_cancelled_chart(df, title_suffix, save_path, custom_title=None, custom_subtitle=None):
    """Create chart for total policies cancelled over time.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the insurance data
    title_suffix : str
        Suffix to add to default subtitle (e.g., "across all Florida insurers")
    save_path : str
        Path to save the chart
    custom_title : str, optional
        Custom title to override the default
    custom_subtitle : str, optional
        Custom subtitle to override the default
    """
    chart_data = df[['year', 'Number of policies canceled']].copy()
    
    # Use custom title/subtitle if provided, otherwise use defaults
    title = custom_title if custom_title else "FLORIDA HOMEOWNERS INSURANCE POLICIES CANCELLED"
    subtitle = custom_subtitle if custom_subtitle else f"TOTAL NUMBER OF HOMEOWNERS INSURANCE POLICIES CANCELLED {title_suffix.upper()}"
    data_note = "DATA: FLORIDA OFFICE OF INSURANCE REGULATION | OWNER-OCCUPIED RESIDENTIAL HOMEOWNERS (EXCL. TENANT AND CONDO)"
    
    return plot_line_custom(
        chart_data.rename(columns={'Number of policies canceled': 'policies_cancelled'}),
        'policies_cancelled',
        title,
        subtitle,
        data_note,
        unit='count',
        save_path=save_path
    )

def create_non_renewed_chart(df, title_suffix, save_path, custom_title=None, custom_subtitle=None):
    """Create chart for total policies non-renewed over time.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the insurance data
    title_suffix : str
        Suffix to add to default subtitle (e.g., "across all Florida insurers")
    save_path : str
        Path to save the chart
    custom_title : str, optional
        Custom title to override the default
    custom_subtitle : str, optional
        Custom subtitle to override the default
    """
    chart_data = df[['year', 'Number of policies nonrenewed']].copy()
    
    # Use custom title/subtitle if provided, otherwise use defaults
    title = custom_title if custom_title else "FLORIDA HOMEOWNERS INSURANCE POLICIES NON-RENEWED"
    subtitle = custom_subtitle if custom_subtitle else f"TOTAL NUMBER OF HOMEOWNERS INSURANCE POLICIES NON-RENEWED {title_suffix.upper()}"
    data_note = "DATA: FLORIDA OFFICE OF INSURANCE REGULATION | OWNER-OCCUPIED RESIDENTIAL HOMEOWNERS (EXCL. TENANT AND CONDO)"
    
    return plot_line_custom(
        chart_data.rename(columns={'Number of policies nonrenewed': 'policies_non_renewed'}),
        'policies_non_renewed',
        title,
        subtitle,
        data_note,
        unit='count',
        save_path=save_path
    )

def create_hurricane_non_renewed_chart(df, title_suffix, save_path, custom_title=None, custom_subtitle=None):
    """Create chart for policies non-renewed due to hurricane risk over time.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the insurance data
    title_suffix : str
        Suffix to add to default subtitle (e.g., "across all Florida insurers")
    save_path : str
        Path to save the chart
    custom_title : str, optional
        Custom title to override the default
    custom_subtitle : str, optional
        Custom subtitle to override the default
    """
    chart_data = df[['year', 'Number of policies nonrenewed due to hurricane risk']].copy()
    
    # Use custom title/subtitle if provided, otherwise use defaults
    title = custom_title if custom_title else "FLORIDA HOMEOWNERS POLICIES NON-RENEWED DUE TO HURRICANE RISK"
    subtitle = custom_subtitle if custom_subtitle else f"TOTAL NUMBER OF POLICIES NON-RENEWED DUE TO HURRICANE RISK {title_suffix.upper()}"
    data_note = "DATA: FLORIDA OFFICE OF INSURANCE REGULATION | OWNER-OCCUPIED RESIDENTIAL HOMEOWNERS (EXCL. TENANT AND CONDO)"
    
    return plot_line_custom(
        chart_data.rename(columns={'Number of policies nonrenewed due to hurricane risk': 'hurricane_non_renewed'}),
        'hurricane_non_renewed',
        title,
        subtitle,
        data_note,
        unit='count',
        save_path=save_path
    )

def create_average_premium_chart(df, title_suffix, save_path, custom_title=None, custom_subtitle=None, use_inflation_adjusted=True):
    """Create chart for average premium over time.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the insurance data
    title_suffix : str
        Suffix to add to default subtitle (e.g., "across all Florida insurers")
    save_path : str
        Path to save the chart
    custom_title : str, optional
        Custom title to override the default
    custom_subtitle : str, optional
        Custom subtitle to override the default
    use_inflation_adjusted : bool, optional
        Whether to use inflation-adjusted values (default: True)
    """
    # Choose column based on inflation adjustment preference
    if use_inflation_adjusted and 'average_premium_adj_2020' in df.columns:
        premium_col = 'average_premium_adj_2020'
        title_suffix_adj = f"{title_suffix} (2020 DOLLARS)"
        data_note_suffix = " | ADJUSTED TO 2020 DOLLARS"
    else:
        premium_col = 'average_premium'
        title_suffix_adj = title_suffix
        data_note_suffix = ""

    chart_data = df[['year', premium_col]].copy()

    # Use custom title/subtitle if provided, otherwise use defaults
    title = custom_title if custom_title else f"FLORIDA HOMEOWNERS INSURANCE AVERAGE PREMIUM{' (2020 DOLLARS)' if use_inflation_adjusted and 'average_premium_adj_2020' in df.columns else ''}"
    subtitle = custom_subtitle if custom_subtitle else f"AVERAGE ANNUAL PREMIUM PER POLICY {title_suffix_adj.upper()}"
    data_note = f"DATA: FLORIDA OFFICE OF INSURANCE REGULATION | OWNER-OCCUPIED RESIDENTIAL HOMEOWNERS (EXCL. TENANT AND CONDO){data_note_suffix}"

    return plot_line_custom(
        chart_data,
        premium_col,
        title,
        subtitle,
        data_note,
        unit='dollar',
        save_path=save_path
    )

def create_new_policies_chart(df, title_suffix, save_path, custom_title=None, custom_subtitle=None):
    """Create chart for new policies written over time.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the insurance data
    title_suffix : str
        Suffix to add to default subtitle (e.g., "across all Florida insurers")
    save_path : str
        Path to save the chart
    custom_title : str, optional
        Custom title to override the default
    custom_subtitle : str, optional
        Custom subtitle to override the default
    """
    chart_data = df[['year', 'Number of new policies written']].copy()
    
    # Use custom title/subtitle if provided, otherwise use defaults
    title = custom_title if custom_title else "FLORIDA HOMEOWNERS INSURANCE NEW POLICIES WRITTEN"
    subtitle = custom_subtitle if custom_subtitle else f"TOTAL NUMBER OF NEW HOMEOWNERS INSURANCE POLICIES WRITTEN {title_suffix.upper()}"
    data_note = "DATA: FLORIDA OFFICE OF INSURANCE REGULATION | OWNER-OCCUPIED RESIDENTIAL HOMEOWNERS (EXCL. TENANT AND CONDO)"
    
    return plot_line_custom(
        chart_data.rename(columns={'Number of new policies written': 'new_policies_written'}),
        'new_policies_written',
        title,
        subtitle,
        data_note,
        unit='count',
        save_path=save_path
    )

def create_citizens_premium_transferred_chart(df, save_path, custom_title=None, custom_subtitle=None, use_inflation_adjusted=True):
    """Create chart showing Citizens average premium with transferred policies on second y-axis.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing Citizens data
    save_path : str
        Path to save the chart
    custom_title : str, optional
        Custom title to override the default
    custom_subtitle : str, optional
        Custom subtitle to override the default
    use_inflation_adjusted : bool, optional
        Whether to use inflation-adjusted values (default: True)
    """
    # Set up the plot with common styling
    fig, ax1, font_props = setup_enhanced_plot()

    # Choose premium column based on inflation adjustment preference
    if use_inflation_adjusted and 'average_premium_adj_2020' in df.columns:
        premium_col = 'average_premium_adj_2020'
        data_note_suffix = " | ADJUSTED TO 2020 DOLLARS"
        title_suffix = " (2020 DOLLARS)"
    else:
        premium_col = 'average_premium'
        data_note_suffix = ""
        title_suffix = ""

    # Plot average premium on primary y-axis
    color1 = COLORS['primary']
    line1 = ax1.plot(df['year'], df[premium_col],
                     color=color1,
                     linewidth=3,
                     marker='o',
                     markersize=8,
                     markerfacecolor=color1,
                     markeredgecolor='white',
                     markeredgewidth=2,
                     label='Average Premium')
    
    # Format primary y-axis (premium)
    font_prop = font_props.get('regular') if font_props else None
    ax1.yaxis.set_major_formatter('${x:,.0f}')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=14)
    ax1.set_ylabel('Average Premium ($)', color=color1, fontproperties=font_prop, fontsize=14)
    
    # Create secondary y-axis for policies in force
    ax2 = ax1.twinx()
    color2 = 'black'
    line2 = ax2.plot(df['year'], df['Policies in force'], 
                     color=color2, 
                     linewidth=3, 
                     marker='s', 
                     markersize=8,
                     markerfacecolor=color2,
                     markeredgecolor='white',
                     markeredgewidth=2,
                     label='Policies in Force')
    
    # Format secondary y-axis (policies in force)
    ax2.yaxis.set_major_formatter('{x:,.0f}')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=14)
    ax2.set_ylabel('Policies in Force', color=color2, fontproperties=font_prop, fontsize=14)
    
    # Format x-axis
    ax1.set_xticks(df['year'])
    ax1.set_xticklabels(df['year'], fontproperties=font_prop, fontsize=14)
    
    # Use custom title/subtitle if provided, otherwise use defaults
    title = custom_title if custom_title else f"CITIZENS PROPERTY INSURANCE PREMIUMS AND POLICIES IN FORCE{title_suffix}"
    subtitle = custom_subtitle if custom_subtitle else f"AVERAGE ANNUAL PREMIUM AND NUMBER OF POLICIES IN FORCE{title_suffix}"
    data_note = f"DATA: FLORIDA OFFICE OF INSURANCE REGULATION | OWNER-OCCUPIED RESIDENTIAL HOMEOWNERS (EXCL. TENANT AND CONDO){data_note_suffix}"
    
    # Format the plot title
    format_plot_title(ax1, title, subtitle, font_props)
    
    # Add branding
    add_deep_sky_branding(ax1, font_props, data_note=data_note)
    
    # Save the plot if a path is provided
    save_plot(fig, save_path)
    
    return fig

def create_citizens_market_share_chart(all_insurers_df, citizens_df, save_path, custom_title=None, custom_subtitle=None):
    """Create chart showing Citizens' market share over time.
    
    Parameters:
    -----------
    all_insurers_df : pandas DataFrame
        DataFrame containing all insurers data
    citizens_df : pandas DataFrame
        DataFrame containing Citizens-only data
    save_path : str
        Path to save the chart
    custom_title : str, optional
        Custom title to override the default
    custom_subtitle : str, optional
        Custom subtitle to override the default
    """
    # Calculate market share for each year
    market_share_data = []
    for year in all_insurers_df['year']:
        total_policies = all_insurers_df[all_insurers_df['year'] == year]['Policies in force'].iloc[0]
        citizens_policies = citizens_df[citizens_df['year'] == year]['Policies in force'].iloc[0]
        market_share = (citizens_policies / total_policies) * 100
        market_share_data.append({'year': year, 'citizens_market_share': market_share})
    
    chart_data = pd.DataFrame(market_share_data)
    
    # Use custom title/subtitle if provided, otherwise use defaults
    title = custom_title if custom_title else "CITIZENS PROPERTY INSURANCE MARKET SHARE"
    subtitle = custom_subtitle if custom_subtitle else "CITIZENS' SHARE OF TOTAL FLORIDA HOMEOWNERS INSURANCE POLICIES"
    data_note = "DATA: FLORIDA OFFICE OF INSURANCE REGULATION | OWNER-OCCUPIED RESIDENTIAL HOMEOWNERS (EXCL. TENANT AND CONDO)"
    
    return plot_line_custom(
        chart_data,
        'citizens_market_share',
        title,
        subtitle,
        data_note,
        unit='percent',
        save_path=save_path
    )

def plot_line_custom(df, y_val, title, subtitle, data_note, unit='count', save_path=None):
    """
    Create a line chart with consistent styling for Florida insurance data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the policy data
    y_val : str
        Column name for the y-axis values
    title : str
        Title for the plot
    subtitle : str
        Subtitle for the plot
    data_note : str
        Data attribution note
    unit : str, optional
        Unit for y-axis values ('dollar', 'count', or 'percent')
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Set up the plot with common styling
    fig, ax, font_props = setup_enhanced_plot()
    
    # Create line plot with markers
    plt.plot(df['year'], df[y_val], 
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
    ax.set_xticks(df['year'])
    ax.set_xticklabels(df['year'], fontproperties=font_prop, fontsize=14)
    
    # Format y-axis
    if unit == 'dollar':
        ax.yaxis.set_major_formatter('${x:,.0f}')
    elif unit == 'count':
        ax.yaxis.set_major_formatter('{x:,.0f}')
    elif unit == 'percent':
        ax.yaxis.set_major_formatter('{x:.0f}%')
    
    ax.tick_params(axis='y', labelsize=14)
    
    # Add branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    # Save the plot if a path is provided
    save_plot(fig, save_path)
    
    return fig

def main():
    """Main function to process Florida insurance data and create charts."""
    
    # Define data paths
    all_insurers_path = 'florida_insurance/florida_homeowners_insurance.csv'
    citizens_path = 'florida_insurance/florida_homeowners_insurance_citizens.csv'
    
    # Create output directory
    output_dir = 'figures/florida'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all insurers data
    print("Processing all Florida insurers data...")
    all_insurers_df = load_and_process_florida_data(all_insurers_path)
    
    # Process Citizens data
    print("Processing Citizens Property Insurance data...")
    citizens_df = load_and_process_florida_data(citizens_path)
    
    # Create charts for all insurers
    print("Creating charts for all Florida insurers...")
    
    # 1. Total policies in force
    create_policies_chart(
        all_insurers_df, 
        "across all Florida insurers",
        f"{output_dir}/fl_all_insurers_policies_in_force.png",
        # Example of how to use custom title/subtitle:
        custom_title="INSURERS ARE ABANDONING FLORIDA HOMEOWNERS",
        custom_subtitle="NUMBER OF ACTIVE POLICIES IS DOWN 76% IN 10 YEARS ACROSS ALL FLORIDA INSURERS"
    )
    
    # 2. Policies cancelled
    create_cancelled_chart(
        all_insurers_df,
        "across all Florida insurers", 
        f"{output_dir}/fl_all_insurers_policies_cancelled.png"
    )
    
    # 3. Policies non-renewed
    create_non_renewed_chart(
        all_insurers_df,
        "across all Florida insurers",
        f"{output_dir}/fl_all_insurers_policies_non_renewed.png"
    )
    
    # 4. Hurricane non-renewals
    create_hurricane_non_renewed_chart(
        all_insurers_df,
        "across all Florida insurers",
        f"{output_dir}/fl_all_insurers_hurricane_non_renewed.png"
    )
    
    # 5. Average premium (inflation-adjusted)
    create_average_premium_chart(
        all_insurers_df,
        "across all Florida insurers",
        f"{output_dir}/fl_all_insurers_average_premium_adj.png",
        custom_title="FLORIDA HOMEOWNERS ARE PAYING MORE THAN EVER FOR INSURANCE (2020 DOLLARS)",
        use_inflation_adjusted=True
    )
    
    # 6. New policies written
    create_new_policies_chart(
        all_insurers_df,
        "across all Florida insurers",
        f"{output_dir}/fl_all_insurers_new_policies_written.png",
        custom_title="INSURERS ARE ABANDONING FLORIDA HOMEOWNERS"
    )
    
    # Create charts for Citizens only
    print("Creating charts for Citizens Property Insurance...")
    
    # 1. Total policies in force
    create_policies_chart(
        citizens_df,
        "for Citizens Property Insurance Corporation",
        f"{output_dir}/fl_citizens_policies_in_force.png"
    )
    
    # 2. Policies cancelled
    create_cancelled_chart(
        citizens_df,
        "for Citizens Property Insurance Corporation",
        f"{output_dir}/fl_citizens_policies_cancelled.png"
    )
    
    # 3. Policies non-renewed
    create_non_renewed_chart(
        citizens_df,
        "for Citizens Property Insurance Corporation", 
        f"{output_dir}/fl_citizens_policies_non_renewed.png"
    )
    
    # 4. Hurricane non-renewals
    create_hurricane_non_renewed_chart(
        citizens_df,
        "for Citizens Property Insurance Corporation",
        f"{output_dir}/fl_citizens_hurricane_non_renewed.png"
    )
    
    # 5. Average premium (inflation-adjusted)
    create_average_premium_chart(
        citizens_df,
        "for Citizens Property Insurance Corporation",
        f"{output_dir}/fl_citizens_average_premium_adj.png",
        custom_title="EVEN THE STATE INSURER-OF-LAST-RESORT IS NOW UNAFFORDABLE (2020 DOLLARS)",
        use_inflation_adjusted=True
    )
    
    # 5b. Average premium with transferred policies (inflation-adjusted)
    create_citizens_premium_transferred_chart(
        citizens_df,
        f"{output_dir}/fl_citizens_premium_transferred_adj.png",
        use_inflation_adjusted=True
    )
    
    # 6. New policies written
    create_new_policies_chart(
        citizens_df,
        "for Citizens Property Insurance Corporation",
        f"{output_dir}/fl_citizens_new_policies_written.png"
    )
    
    # Create Citizens market share chart
    print("Creating Citizens market share chart...")
    create_citizens_market_share_chart(
        all_insurers_df,
        citizens_df,
        f"{output_dir}/fl_citizens_market_share.png",
        custom_title="FLORIDA HOMEOWNERS ARE BEING FORCED OUT OF THE PRIVATE MARKET",
        custom_subtitle="PERCENTAGE OF FLORIDA HOME INSURANCE POLICIES HELD BY THE STATE INSURER-OF-LAST-RESORT"
    )
    
    print("All Florida insurance charts created successfully!")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("\nAll Florida Insurers:")
    print(f"Policies in force: {all_insurers_df['Policies in force'].iloc[-1]:,.0f} (2024)")
    print(f"Average premium: ${all_insurers_df['average_premium'].iloc[-1]:,.0f} (2024)")
    
    print("\nCitizens Property Insurance:")
    print(f"Policies in force: {citizens_df['Policies in force'].iloc[-1]:,.0f} (2024)")
    print(f"Average premium: ${citizens_df['average_premium'].iloc[-1]:,.0f} (2024)")
    
    # Calculate Citizens market share
    citizens_share = (citizens_df['Policies in force'].iloc[-1] / all_insurers_df['Policies in force'].iloc[-1]) * 100
    print(f"Citizens market share: {citizens_share:.1f}% (2024)")

if __name__ == "__main__":
    main()
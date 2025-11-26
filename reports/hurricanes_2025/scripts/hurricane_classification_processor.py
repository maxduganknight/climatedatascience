import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def load_hurricane_data(file_path, sheet_name='HistoricalHurricanes'):
    """
    Load and process hurricane classification data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing hurricane data
    sheet_name : str, optional
        Name of the sheet to load (default: 'HistoricalHurricanes')
    
    Returns:
    --------
    pandas DataFrame
        Processed DataFrame with cleaned data
    """
    print(f"Loading hurricane data from: {file_path}")
    
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Clean column names by removing any trailing spaces
    df.columns = df.columns.str.strip()
    
    # Ensure Year is integer
    df['Year'] = df['Year'].astype(int)
    
    # Remove any rows with missing critical data
    initial_count = len(df)
    df = df.dropna(subset=['Final TCSS category', 'SSHWS', 'Year'])

    # remove peak rainfall rows
    # df = df[~df['County'].str.contains('peak rainfall')]

    cleaned_count = len(df)
    
    if initial_count != cleaned_count:
        print(f"Removed {initial_count - cleaned_count} rows with missing data")
    
    print(f"Loaded {len(df)} hurricane records from {df['Year'].min()} to {df['Year'].max()}")
    
    return df

def analyze_classification_distribution(df):
    """
    Analyze the distribution of hurricane classifications.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Hurricane data
    
    Returns:
    --------
    dict
        Dictionary containing distribution analysis
    """
    analysis = {}
    
    # Overall distribution
    analysis['tcss_distribution'] = df['Final TCSS category'].value_counts().sort_index()
    analysis['sshws_distribution'] = df['SSHWS'].value_counts().sort_index()
    
    # Yearly counts
    analysis['yearly_counts'] = df.groupby('Year').size()
    
    # Cross-tabulation
    analysis['cross_tab'] = pd.crosstab(df['SSHWS'], df['Final TCSS category'], margins=True)
    
    return analysis

def create_classification_comparison_over_time(df, save_path=None):
    """
    Create line charts comparing count of hurricanes above category 4 for TCSS and SSHWS over time.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Hurricane data
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Calculate yearly counts of hurricanes above category 4
    yearly_tcss_high = df[df['Final TCSS category'] >= 4].groupby('Year').size()
    yearly_sshws_high = df[df['SSHWS'] >= 4].groupby('Year').size()
    
    # Create year range based on actual data years, excluding 2025 if present
    actual_years = sorted(df['Year'].unique())
    if 2025 in actual_years:
        actual_years.remove(2025)
    
    # Reindex to include all years in range, filling missing with 0
    yearly_tcss_high = yearly_tcss_high.reindex(actual_years, fill_value=0)
    yearly_sshws_high = yearly_sshws_high.reindex(actual_years, fill_value=0)
    
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot()
    
    # Create bar chart with TCSS bars behind SSHWS bars
    years = yearly_tcss_high.index
    # Plot TCSS bars first (behind)
    plt.bar(years, yearly_tcss_high.values, 
             color=COLORS['primary'], 
             alpha=0.8,
             width=0.8,
             label='Tropical Cycline Severity Scale')
    
    # Plot SSHWS bars on top (in front)
    plt.bar(years, yearly_sshws_high.values, 
             color=COLORS['secondary'], 
             alpha=0.9,
             width=0.6,
             label='Saffir-Simpson Scale')
    
    # Format the plot
    title = "EXTREME HURRICANES ARE BECOMING MORE COMMON"
    subtitle = "CATEGORY 4, 5, 6 HURRICANES PER YEAR: TCSS VS. SAFFIR-SIMPSON SCALE"
    data_note = "DATA: BLOEMENDAAL ET AL. TROPICAL CYCLONE SEVERITY SCALE | TCSS DATA INCLUDES STORM PEAK RAINFALL"
    
    format_plot_title(ax, title, subtitle, font_props)
    
    # Format axes
    font_prop = font_props.get('regular') if font_props else None
    ax.set_xlabel('', fontproperties=font_prop, fontsize=14)
    ax.set_ylabel('', fontproperties=font_prop, fontsize=14)
    
    # Set both x-axis and y-axis limits explicitly
    ax.set_xlim(actual_years[0] - 0.5, actual_years[-1] + 0.5)
    ax.set_ylim(0, max(yearly_tcss_high.max(), yearly_sshws_high.max()) + 1)
    
    # Set x-axis ticks to show every few years to avoid crowding
    tick_years = [year for year in actual_years if year % 5 == 0 or year == actual_years[-1]]
    ax.set_xticks(tick_years)
    ax.set_xticklabels(tick_years, fontproperties=font_prop, fontsize=12)
    
    # Add legend
    ax.legend(fontsize=14, frameon=True, facecolor=COLORS['background'], 
              edgecolor='#DDDDDD', loc='upper left', prop=font_prop)
    
    # Add branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    # Save the plot
    save_plot(fig, save_path)
    
    return fig

def create_classification_distribution_comparison(df, save_path=None):
    """
    Create bar chart comparing distribution of TCSS vs SSHWS classifications.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Hurricane data
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Calculate distributions
    tcss_dist = df['Final TCSS category'].value_counts().sort_index()
    sshws_dist = df['SSHWS'].value_counts().sort_index()
    
    # Normalize to percentages
    tcss_pct = (tcss_dist / len(df)) * 100
    sshws_pct = (sshws_dist / len(df)) * 100
    
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot()
    
    # Create bar positions
    categories = range(max(max(tcss_dist.index), max(sshws_dist.index)) + 1)
    x_pos = np.arange(len(categories))
    width = 0.35
    
    # Align data to categories (fill missing with 0)
    tcss_values = [tcss_pct.get(i, 0) for i in categories]
    sshws_values = [sshws_pct.get(i, 0) for i in categories]
    
    # Create bars
    bars1 = ax.bar(x_pos - width/2, tcss_values, width, 
                   label='TCSS Classification', color=COLORS['primary'], alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, sshws_values, width,
                   label='Saffir-Simpson Scale', color=COLORS['secondary'], alpha=0.8)
    
    # Add value labels on bars
    font_prop = font_props.get('regular') if font_props else None
    for bar, value in zip(bars1, tcss_values):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', 
                   fontproperties=font_prop, fontsize=10)
    
    for bar, value in zip(bars2, sshws_values):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', 
                   fontproperties=font_prop, fontsize=10)
    
    # Format the plot
    title = "TCSS SYSTEM CAPTURES MORE STORM SEVERITY VARIATION"
    subtitle = "DISTRIBUTION OF HURRICANE CATEGORIES: TCSS VS. SAFFIR-SIMPSON SCALE"
    data_note = "DATA: TCSS HURRICANE CLASSIFICATION SYSTEM ANALYSIS | HURRICANES 1995-2024"
    
    format_plot_title(ax, title, subtitle, font_props)
    
    # Format axes
    ax.set_xlabel('Hurricane Category', fontproperties=font_prop, fontsize=14)
    ax.set_ylabel('Percentage of Storms (%)', fontproperties=font_prop, fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontproperties=font_prop, fontsize=14)
    
    # Add legend
    ax.legend(fontsize=14, frameon=True, facecolor=COLORS['background'], 
              edgecolor='#DDDDDD', loc='upper right', prop=font_prop)
    
    # Add branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    # Save the plot
    save_plot(fig, save_path)
    
    return fig

def create_time_series_analysis(df, variable, save_path=None, custom_title=None, custom_subtitle=None):
    """
    Create a flexible time series analysis for any variable.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Hurricane data
    variable : str
        Column name to analyze over time
    save_path : str, optional
        Path to save the figure
    custom_title : str, optional
        Custom title for the plot
    custom_subtitle : str, optional
        Custom subtitle for the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Calculate yearly statistics
    yearly_stats = df.groupby('Year')[variable].agg(['mean', 'count', 'std']).reset_index()
    
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot()
    
    # Plot the trend line
    plt.plot(yearly_stats['Year'], yearly_stats['mean'], 
             color=COLORS['primary'], 
             linewidth=3, 
             marker='o', 
             markersize=8,
             markerfacecolor=COLORS['primary'],
             markeredgecolor='white',
             markeredgewidth=2)
    
    # Add error bars if there's variation
    if yearly_stats['std'].notna().any():
        plt.errorbar(yearly_stats['Year'], yearly_stats['mean'], 
                    yerr=yearly_stats['std'], 
                    color=COLORS['primary'], 
                    alpha=0.3, 
                    capsize=3, 
                    capthick=1)
    
    # Format the plot
    title = custom_title or f"HURRICANE {variable.upper()} TRENDS OVER TIME"
    subtitle = custom_subtitle or f"AVERAGE {variable.upper()} PER YEAR WITH STANDARD DEVIATION"
    data_note = "DATA: TCSS HURRICANE CLASSIFICATION SYSTEM ANALYSIS | HURRICANES 1995-2024"
    
    format_plot_title(ax, title, subtitle, font_props)
    
    # Format axes
    font_prop = font_props.get('regular') if font_props else None
    ax.set_xlabel('Year', fontproperties=font_prop, fontsize=14)
    ax.set_ylabel(f'Average {variable}', fontproperties=font_prop, fontsize=14)
    
    # Add branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    # Save the plot
    save_plot(fig, save_path)
    
    return fig

def create_hurricane_count_by_category(df, classification_system='Final TCSS category', save_path=None):
    """
    Create a time series showing hurricane counts by category.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Hurricane data
    classification_system : str, optional
        Which classification system to use ('Final TCSS category' or 'SSHWS')
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Create yearly counts by category
    yearly_counts = df.groupby(['Year', classification_system]).size().unstack(fill_value=0)
    
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot()
    
    # Create color palette for categories
    n_categories = len(yearly_counts.columns)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, n_categories))
    
    # Create stacked area plot
    ax.stackplot(yearly_counts.index, *[yearly_counts[col] for col in yearly_counts.columns],
                labels=[f'Category {cat}' for cat in yearly_counts.columns],
                colors=colors, alpha=0.8)
    
    # Format the plot
    system_name = "TCSS" if "TCSS" in classification_system else "Saffir-Simpson"
    title = f"HURRICANE FREQUENCY BY {system_name.upper()} CATEGORY"
    subtitle = f"NUMBER OF HURRICANES PER YEAR BY {system_name.upper()} CLASSIFICATION"
    data_note = "DATA: TCSS HURRICANE CLASSIFICATION SYSTEM ANALYSIS | HURRICANES 1995-2024"
    
    format_plot_title(ax, title, subtitle, font_props)
    
    # Format axes
    font_prop = font_props.get('regular') if font_props else None
    ax.set_xlabel('Year', fontproperties=font_prop, fontsize=14)
    ax.set_ylabel('Number of Hurricanes', fontproperties=font_prop, fontsize=14)
    
    # Add legend
    ax.legend(fontsize=12, frameon=True, facecolor=COLORS['background'], 
              edgecolor='#DDDDDD', loc='upper left', prop=font_prop)
    
    # Add branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    # Save the plot
    save_plot(fig, save_path)
    
    return fig

def create_scatter_plot(df, variable, save_path=None, custom_title=None, custom_subtitle=None, color=None):
    """
    Create scatter plot of any variable over time.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Hurricane data
    variable : str
        Column name to plot on y-axis
    save_path : str, optional
        Path to save the figure
    custom_title : str, optional
        Custom title for the plot
    custom_subtitle : str, optional
        Custom subtitle for the plot
    color : str, optional
        Color for the scatter points (defaults to primary color)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Filter out missing data for the specified variable
    df_filtered = df.dropna(subset=[variable])
    
    if len(df_filtered) == 0:
        print(f"Warning: No data available for variable '{variable}'")
        return None
    
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot()
    
    # Use provided color or default to primary
    plot_color = color if color else COLORS['primary']
    
    # Create scatter plot
    plt.scatter(df_filtered['Year'], df_filtered[variable], 
                color=plot_color, 
                alpha=0.7, 
                s=60,
                edgecolors='white',
                linewidth=1)
    
    # Generate default titles if not provided
    if not custom_title:
        var_clean = variable.replace('(', '').replace(')', '').replace('Max ', '').title()
        custom_title = f"HURRICANE {var_clean.upper()} INTENSITY OVER TIME"
    
    if not custom_subtitle:
        custom_subtitle = f"{variable.upper()} FOR EACH HURRICANE BY YEAR"
    
    # Format the plot
    data_note = "DATA: TCSS HURRICANE CLASSIFICATION SYSTEM ANALYSIS | HURRICANES 1995-2024"
    
    format_plot_title(ax, custom_title, custom_subtitle, font_props)
    
    # Format axes
    font_prop = font_props.get('regular') if font_props else None
    ax.set_xlabel('Year', fontproperties=font_prop, fontsize=14)
    ax.set_ylabel(variable, fontproperties=font_prop, fontsize=14)
    
    # Add branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    # Save the plot
    save_plot(fig, save_path)
    
    return fig

def main():
    """Main function to process hurricane classification data and create analysis charts."""
    
    # Define data path
    data_path = 'hurricane_classification/TCSS_Historical_Hurricanes.xlsx'
    
    # Create output directory
    output_dir = 'figures/tcss/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load hurricane data
    print("Loading hurricane classification data...")
    df = load_hurricane_data(data_path)
    
    print(df.head())

    # Perform basic analysis
    print("Analyzing classification distributions...")
    analysis = analyze_classification_distribution(df)
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total hurricanes analyzed: {len(df)}")
    print(f"Years covered: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Average hurricanes per year: {len(df) / (df['Year'].max() - df['Year'].min() + 1):.1f}")
    
    print("\nTCSS Category Distribution:")
    for cat, count in analysis['tcss_distribution'].items():
        pct = (count / len(df)) * 100
        print(f"  Category {cat}: {count} storms ({pct:.1f}%)")
    
    print("\nSaffir-Simpson Scale Distribution:")
    for cat, count in analysis['sshws_distribution'].items():
        pct = (count / len(df)) * 100
        print(f"  Category {cat}: {count} storms ({pct:.1f}%)")
    
    # Create visualization outputs
    print("\nCreating visualization outputs...")
    
    # 1. Classification comparison over time
    print("Creating classification comparison chart...")
    create_classification_comparison_over_time(
        df, 
        save_path=f"{output_dir}hurricane_classification_comparison_time.png"
    )
    
    # 2. Distribution comparison
    print("Creating distribution comparison chart...")
    create_classification_distribution_comparison(
        df, 
        save_path=f"{output_dir}hurricane_classification_distribution.png"
    )
    
    # 3. TCSS category time series analysis
    print("Creating TCSS category trends...")
    create_time_series_analysis(
        df, 
        'Final TCSS category',
        save_path=f"{output_dir}tcss_category_trends.png",
        custom_title="HURRICANE SEVERITY INCREASING UNDER NEW CLASSIFICATION",
        custom_subtitle="AVERAGE TCSS CATEGORY PER YEAR SHOWS UPWARD TREND"
    )
    
    # 4. SSHWS category time series analysis  
    print("Creating Saffir-Simpson category trends...")
    create_time_series_analysis(
        df, 
        'SSHWS',
        save_path=f"{output_dir}sshws_category_trends.png",
        custom_title="TRADITIONAL HURRICANE SCALE SHOWS MODEST INCREASE",
        custom_subtitle="AVERAGE SAFFIR-SIMPSON CATEGORY PER YEAR"
    )
    
    # 5. Hurricane counts by TCSS category
    print("Creating TCSS hurricane count stacks...")
    create_hurricane_count_by_category(
        df, 
        'Final TCSS category',
        save_path=f"{output_dir}hurricane_counts_tcss.png"
    )
    
    # 6. Hurricane counts by SSHWS category
    print("Creating Saffir-Simpson hurricane count stacks...")
    create_hurricane_count_by_category(
        df, 
        'SSHWS',
        save_path=f"{output_dir}hurricane_counts_sshws.png"
    )
    
    # # 7. Rainfall scatter plot
    # print("Creating rainfall scatter plot...")
    # create_scatter_plot(
    #     df,
    #     'Max rainfall (mm)',
    #     save_path=f"{output_dir}hurricane_rainfall_scatter.png",
    #     custom_title="HURRICANE RAINFALL INTENSITY SHOWS HIGH VARIABILITY",
    #     custom_subtitle="MAXIMUM RAINFALL (MM) FOR EACH HURRICANE BY YEAR",
    #     color=COLORS['primary']
    # )
    
    # # 8. Storm surge scatter plot
    # print("Creating storm surge scatter plot...")
    # create_scatter_plot(
    #     df,
    #     'Max storm surge (m)',
    #     save_path=f"{output_dir}hurricane_storm_surge_scatter.png",
    #     custom_title="HURRICANE STORM SURGE INTENSITY OVER TIME",
    #     custom_subtitle="MAXIMUM STORM SURGE (METERS) FOR EACH HURRICANE BY YEAR",
    #     color=COLORS['secondary']
    # )
    
    print("\nAll hurricane classification analysis charts created successfully!")
    
    # Calculate and display correlation
    correlation = df['Final TCSS category'].corr(df['SSHWS'])
    print(f"\nCorrelation between TCSS and Saffir-Simpson: {correlation:.3f}")
    
    # Show cases where classifications differ significantly
    df['classification_diff'] = df['Final TCSS category'] - df['SSHWS']
    large_differences = df[abs(df['classification_diff']) >= 2]
    
    if len(large_differences) > 0:
        print(f"\nHurricanes with large classification differences (≥2 categories): {len(large_differences)}")
        print("Examples:")
        for _, row in large_differences.head(5).iterrows():
            print(f"  {row['Name']} ({row['Year']}): TCSS={row['Final TCSS category']}, SSHWS={row['SSHWS']}")

if __name__ == "__main__":
    main()
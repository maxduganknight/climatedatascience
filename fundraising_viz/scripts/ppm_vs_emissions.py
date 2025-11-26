import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import datetime
import numpy as np
from scipy import stats
from matplotlib.offsetbox import AnnotationBbox

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot, COLORS


def load_ppm_emissions_data(excel_path, ppm_csv_path):
    """
    Load CO2 PPM and emissions data from separate files.

    Parameters:
    -----------
    excel_path : str
        Path to the Global_Carbon_Budget Excel file
    ppm_csv_path : str
        Path to the co2_annmean_gl.csv file

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: year, ppm, emissions_co2
    """
    # Load emissions data from Excel file
    # Row 21 contains headers, data starts at row 22
    emissions_df = pd.read_excel(excel_path,
                                  sheet_name='Global Carbon Budget',
                                  skiprows=22,
                                  header=None,
                                  names=['year', 'fossil_emissions_c', 'land_use', 'atm_growth',
                                         'ocean_sink', 'land_sink', 'cement_carb', 'budget_imb'])

    # Select only year and fossil emissions columns
    emissions_df = emissions_df[['year', 'fossil_emissions_c']].copy()

    # Add 2024 value (hard-coded)
    emissions_2024 = pd.DataFrame({'year': [2024], 'fossil_emissions_c': [10.21015284]})
    emissions_df = pd.concat([emissions_df, emissions_2024], ignore_index=True)

    # Convert from C to CO2 emissions by multiplying by 3.664
    emissions_df['emissions_co2'] = emissions_df['fossil_emissions_c'] * 3.664

    # Drop the carbon column
    emissions_df = emissions_df[['year', 'emissions_co2']]

    # Load PPM data from CSV
    # Skip the first 37 rows (rows 0-36), keeping row 37 as header, data starts at row 38
    ppm_df = pd.read_csv(ppm_csv_path, skiprows=37)

    # Select only year and mean columns
    ppm_df = ppm_df[['year', 'mean']].copy()
    ppm_df.rename(columns={'mean': 'ppm'}, inplace=True)

    # Merge the two datasets on year
    df = pd.merge(emissions_df, ppm_df, on='year', how='inner')

    # Drop any rows with missing values
    df = df.dropna()

    # Ensure year is integer
    df['year'] = df['year'].astype(int)

    return df


def create_ppm_emissions_plot(df):
    """
    Create a dual-axis visualization showing CO2 PPM as a line chart
    and CO2 emissions as a bar chart.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: year, ppm, emissions_co2

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """

    df = df[df['year'] >= 1983]

    fig, ax1, font_props = setup_enhanced_plot(figsize=(15, 10))

    # Create second y-axis for emissions
    ax2 = ax1.twinx()

    # Plot emissions as bars on the primary axis (ax1) - gray
    ax1.bar(df['year'], df['emissions_co2'],
            color=COLORS['comparison'], width=0.7, alpha=0.7,
            label='CO\N{SUBSCRIPT TWO} Emissions')

    # Add horizontal reference line at 400 PPM on secondary axis
    # ax2.axhline(y=400, color='#CCCCCC', linestyle='--', linewidth=2, alpha=0.6, zorder=1)

    # Add shaded area between PPM line and 400 PPM threshold - red
    ax2.fill_between(df['year'], 403, df['ppm'],
                     where=(df['ppm'] >= 403),
                     color=COLORS['primary'], alpha=0.2, zorder=2)
    
    ax2.text(2013, 418, 'CO\N{SUBSCRIPT TWO} CONCENTRATION ACCELERATES\nEVEN AS EMISSIONS PLATEAU',
            color = COLORS['primary'], fontsize=14, ha='center', va='top', fontweight='bold')

    # Plot PPM as a line on the secondary axis (ax2) - red
    ax2.plot(df['year'], df['ppm'],
             color=COLORS['primary'], linewidth=4,
             marker='', markersize=6, solid_capstyle='round',
             label='CO\N{SUBSCRIPT TWO} PPM', zorder=3)

    # Format primary y-axis (emissions) - gray
    ax1.set_ylabel('Industrial CO\N{SUBSCRIPT TWO} Emissions (Gt)', fontsize=16,
                   fontproperties=font_props.get('regular') if font_props else None,
                   labelpad=15, color=COLORS['comparison'])
    ax1.tick_params(axis='y', labelsize=14, labelcolor=COLORS['comparison'])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax1.set_ylim(17, 45)

    # Format secondary y-axis (ppm) - red
    ax2.set_ylabel('CO\N{SUBSCRIPT TWO} Concentration (PPM)', fontsize=16,
                   fontproperties=font_props.get('regular') if font_props else None,
                   labelpad=15, color=COLORS['primary'])
    ax2.tick_params(axis='y', labelsize=14, labelcolor=COLORS['primary'])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

    # Format x-axis
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', labelsize=14)

    # Set x-axis limits with some padding
    ax1.set_xlim(df['year'].min() - 1, df['year'].max() + 1)

    # Update axis spine colors to match the data
    ax2.spines['right'].set_color(COLORS['primary'])  # Red for PPM
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_color(COLORS['comparison'])  # Gray for emissions
    ax2.spines['left'].set_linewidth(2)

    # Add a legend
    # Combine handles from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    font_prop = font_props.get('regular') if font_props else None
    ax1.legend(handles1 + handles2, labels1 + labels2,
              fontsize=14, frameon=True,
              facecolor=COLORS['background'],
              edgecolor='#DDDDDD',
              loc='upper left',
              prop=font_prop)

    return fig


def create_dual_panel_plot(df):
    """
    Create a two-panel stacked visualization:
    - Top panel: CO2 emissions (bar chart)
    - Bottom panel: CO2 PPM concentration (line chart)

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: year, ppm, emissions_co2

    Returns:
    --------
    tuple
        (matplotlib.figure.Figure, font_props dict)
    """
    from utils import setup_space_mono_font, COLORS

    # Setup font
    font_props = setup_space_mono_font()

    # Create figure with two subplots stacked vertically
    fig = plt.figure(figsize=(15, 12), facecolor=COLORS['background'])

    # Create two subplots with shared x-axis
    ax1 = plt.subplot(2, 1, 1)  # Top panel - Emissions
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)  # Bottom panel - PPM

    # --- TOP PANEL: EMISSIONS ---
    ax1.set_facecolor(COLORS['background'])

    # Plot emissions as bars - gray
    ax1.bar(df['year'], df['emissions_co2'],
            color=COLORS['comparison'], width=0.7, alpha=0.7,
            label='CO\N{SUBSCRIPT TWO} Emissions')

    # Format top panel
    ax1.set_ylabel('Industrial CO\N{SUBSCRIPT TWO} Emissions (Gt)', fontsize=16,
                   fontproperties=font_props.get('regular') if font_props else None,
                   labelpad=15, color=COLORS['comparison'])
    ax1.tick_params(axis='y', labelsize=14, labelcolor=COLORS['comparison'], length=0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    # Remove x-axis labels from top panel since it shares x-axis with bottom panel
    ax1.tick_params(axis='x', labelbottom=False, length=0)
    ax1.set_ylim(17, 45)

    # Remove spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_color('#DDDDDD')

    ax1.text(2015, 41, 'INDUSTRIAL EMISSIONS PLATEAUING',
        color=COLORS['comparison'], fontsize=14, ha='center', va='top', fontweight='bold')

    # Add grid
    # ax1.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.8)
    # ax1.set_axisbelow(True)

    # --- BOTTOM PANEL: PPM CONCENTRATION ---
    ax2.set_facecolor(COLORS['background'])

    ax2.text(2015, 422, 'CO\N{SUBSCRIPT TWO} CONCENTRATION ACCELERATING',
            color=COLORS['primary'], fontsize=14, ha='center', va='top', fontweight='bold')

    # Plot PPM as a line - red
    ax2.plot(df['year'], df['ppm'],
             color=COLORS['primary'], linewidth=4,
             marker='', markersize=6, solid_capstyle='round',
             label='CO\N{SUBSCRIPT TWO} Concentration', zorder=3)

    # Format bottom panel
    ax2.set_ylabel('CO\N{SUBSCRIPT TWO} Concentration (PPM)', fontsize=16,
                   fontproperties=font_props.get('regular') if font_props else None,
                   labelpad=15, color=COLORS['primary'])
    ax2.tick_params(axis='y', labelsize=14, labelcolor=COLORS['primary'], length=0)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax2.set_xlabel(None)
    ax2.tick_params(axis='x', labelsize=14, length=0)

    # Remove spines
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_color('#DDDDDD')

    # Add grid
    ax2.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.8)
    ax2.set_axisbelow(True)

    # Set x-axis limits with some padding
    ax2.set_xlim(df['year'].min() - 1, df['year'].max() + 1)

    # Adjust spacing between subplots - reduce hspace to bring panels closer
    plt.subplots_adjust(hspace=0.1)

    return fig, font_props


def create_ppm_growth_rate_plot(growth_csv_path, trend_type='decadal_mean'):
    """
    Create a bar chart showing year-over-year CO2 PPM growth rate.

    Parameters:
    -----------
    growth_csv_path : str
        Path to the NOAA PPM growth rate CSV file
    trend_type : str
        Type of trend to display: 'linear', 'decadal_mean', or 'rolling_average'

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Load growth rate data from CSV
    df_growth = pd.read_csv(growth_csv_path)

    # Rename columns for consistency
    df_growth.columns = ['year', 'ppm_growth', 'uncertainty']

    # Filter to 1960 onwards
    df_growth = df_growth[df_growth['year'] >= 1960].copy()

    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))

    # Plot growth rate as bars
    ax.bar(df_growth['year'], df_growth['ppm_growth'],
           color=COLORS['primary'], width=0.7, alpha=0.7)

    # Add trend based on selected type
    if trend_type == 'decadal_mean':
        # Add decadal mean bars
        decades = [(1960, 1969), (1970, 1979), (1980, 1989), (1990, 1999),
                   (2000, 2009), (2010, 2019), (2020, 2024)]

        for start_year, end_year in decades:
            decade_data = df_growth[(df_growth['year'] >= start_year) &
                                    (df_growth['year'] <= end_year)]
            if len(decade_data) > 0:
                decade_mean = decade_data['ppm_growth'].mean()
                ax.hlines(y=decade_mean, xmin=start_year, xmax=end_year,
                         color='#000000', linewidth=2.5, zorder=5)

    elif trend_type == 'linear':
        # Add linear regression trend line
        x = df_growth['year'].values
        y = df_growth['ppm_growth'].values
        slope, intercept, _, _, _ = stats.linregress(x, y)
        trend_line = slope * x + intercept

        ax.plot(x, trend_line, color='#000000', linewidth=3, linestyle='-',
                solid_capstyle='round', label='Linear trend', zorder=10)

    elif trend_type == 'rolling_average':
        # Add rolling average trend line
        window_size = 10
        df_growth['rolling_mean'] = df_growth['ppm_growth'].rolling(
            window=window_size, center=True).mean()

        ax.plot(df_growth['year'], df_growth['rolling_mean'],
                color='#000000', linewidth=3, linestyle='-',
                solid_capstyle='round', label='10-year average', zorder=10)

    # Format y-axis
    ax.set_ylabel('', fontsize=16,
                  fontproperties=font_props.get('regular') if font_props else None,
                  labelpad=15, color=COLORS['primary'])
    ax.tick_params(axis='y', labelsize=14, labelcolor=COLORS['primary'], length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

    # Format x-axis
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=14, length=0)
    ax.set_xlim(df_growth['year'].min() - 1, df_growth['year'].max() + 1)

    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    # Add grid
    ax.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    return fig


def create_emissions_growth_rate_plot(df, trend_type='decadal_mean'):
    """
    Create a bar chart showing year-over-year emissions growth rate.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: year, ppm, emissions_co2
    trend_type : str
        Type of trend to display: 'linear', 'decadal_mean', or 'rolling_average'

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Calculate year-over-year growth rate
    df_growth = df.copy()
    df_growth['emissions_growth'] = df_growth['emissions_co2'].diff()

    # Filter to 1960 onwards and remove NaN from diff calculation
    df_growth = df_growth[df_growth['year'] >= 1960].dropna()

    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))

    # Plot growth rate as bars
    ax.bar(df_growth['year'], df_growth['emissions_growth'],
           color=COLORS['comparison'], width=0.7, alpha=0.7)

    # Add trend based on selected type
    if trend_type == 'decadal_mean':
        # Add decadal mean bars
        decades = [(1960, 1969), (1970, 1979), (1980, 1989), (1990, 1999),
                   (2000, 2009), (2010, 2019), (2020, 2024)]

        for start_year, end_year in decades:
            decade_data = df_growth[(df_growth['year'] >= start_year) &
                                    (df_growth['year'] <= end_year)]
            if len(decade_data) > 0:
                decade_mean = decade_data['emissions_growth'].mean()
                ax.hlines(y=decade_mean, xmin=start_year, xmax=end_year,
                         color='#000000', linewidth=2.5, zorder=5)

    elif trend_type == 'linear':
        # Add linear regression trend line
        x = df_growth['year'].values
        y = df_growth['emissions_growth'].values
        slope, intercept, _, _, _ = stats.linregress(x, y)
        trend_line = slope * x + intercept

        ax.plot(x, trend_line, color='#000000', linewidth=3, linestyle='-',
                solid_capstyle='round', label='Linear trend', zorder=10)

    elif trend_type == 'rolling_average':
        # Add rolling average trend line
        window_size = 10
        df_growth['rolling_mean'] = df_growth['emissions_growth'].rolling(
            window=window_size, center=True).mean()

        ax.plot(df_growth['year'], df_growth['rolling_mean'],
                color='#000000', linewidth=3, linestyle='-',
                solid_capstyle='round', label='10-year average', zorder=10)

    # Format y-axis
    ax.set_ylabel('Annual Emissions Increase (Gt/year)', fontsize=16,
                  fontproperties=font_props.get('regular') if font_props else None,
                  labelpad=15, color=COLORS['comparison'])
    ax.tick_params(axis='y', labelsize=14, labelcolor=COLORS['comparison'], length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

    # Format x-axis
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=14, length=0)
    ax.set_xlim(df_growth['year'].min() - 1, df_growth['year'].max() + 1)

    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    # Add grid
    ax.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    return fig


def main():
    """
    Main execution function for PPM vs Emissions visualization.
    """
    # Configuration
    excel_path = 'data/Global_Carbon_Budget_2024_v1.0-1.xlsx'
    ppm_csv_path = 'data/carbon_cycle/co2_annmean_gl.csv'

    # Load data
    print("Loading PPM and emissions data...")
    df = load_ppm_emissions_data(excel_path, ppm_csv_path)

    print(f"Data loaded: {len(df)} years ({df['year'].min():.0f} - {df['year'].max():.0f})")
    print(f"PPM range: {df['ppm'].min():.1f} - {df['ppm'].max():.1f}")
    print(f"Emissions range: {df['emissions_co2'].min():.2f} - {df['emissions_co2'].max():.2f} Gt")

    # Create visualization
    print("\nCreating visualization...")
    fig = create_ppm_emissions_plot(df)

    # Format plot
    format_plot_title(plt.gca(),
                     "",
                     "ATMOSPHERIC CO\N{SUBSCRIPT TWO} CONCENTRATION AND GLOBAL EMISSIONS",
                     None)

    # Custom right margin value to accommodate right y-axis label
    right_margin = 0.9

    add_deep_sky_branding(plt.gca(), None,
                         "DATA: GLOBAL CARBON PROJECT (2024) | NOAA/GML \n2024 emissions projection from Tiseo (2025)",
                         analysis_date=datetime.datetime.now())

    # Update favicon position to align with custom right margin
    # The favicon is added by add_deep_sky_branding at (0.95, 0.01)
    # We need to move it to match our new right margin
    for artist in plt.gca().get_children():
        if isinstance(artist, AnnotationBbox):
            # Update the position by modifying the xy attribute
            artist.xy = (right_margin, 0.01)
            artist.xybox = (right_margin, 0.01)

    # Adjust margins for dual-axis chart with custom right margin
    plt.subplots_adjust(bottom=0.15, top=0.85, left=0.08, right=right_margin)

    # Save the plot (bypassing save_plot to preserve our custom margins)
    save_path = 'figures/ppm_vs_emissions.png'
    os.makedirs('figures', exist_ok=True)
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', facecolor=fig.get_facecolor())
    print(f"Figure saved to: {save_path} and {svg_path}")

    print(f"\nPPM vs Emissions plot saved to {save_path}")

    # Create second visualization: Dual-panel stacked plot
    print("\n" + "="*60)
    print("Creating dual-panel stacked visualization...")
    print("="*60)

    # Create dual-panel visualization
    fig_dual, font_props_dual = create_dual_panel_plot(df)

    # Add title and branding
    format_plot_title(plt.gcf().axes[0],  # Use top subplot for title
                     "",
                     "",
                     font_props_dual)

    # Add branding to the bottom subplot
    add_deep_sky_branding(plt.gcf().axes[1],  # Bottom subplot
                         font_props_dual,
                         "DATA: GLOBAL CARBON PROJECT (2024) | NOAA/GML \n2024 emissions projection from Tiseo (2025)",
                         analysis_date=datetime.datetime.now())

    # Save the dual-panel plot
    dual_save_path = 'figures/dual_panel_ppm_emissions.png'
    save_plot(fig_dual, dual_save_path)

    print(f"\nDual-panel plot saved to {dual_save_path}")

    # Create third visualization: CO2 PPM growth rate
    print("\n" + "="*60)
    print("Creating CO2 PPM growth rate visualization...")
    print("="*60)

    ppm_growth_csv_path = 'data/carbon_cycle/noaa_ppm_growth_rate.csv'
    ppm_trend_type = 'decadal_mean'  # Options: 'linear', 'decadal_mean', 'rolling_average'
    fig_ppm_growth = create_ppm_growth_rate_plot(ppm_growth_csv_path, trend_type=ppm_trend_type)

    # Format plot
    format_plot_title(plt.gca(),
                     "",
                     "CO\N{SUBSCRIPT TWO} PPM GROWTH RATE (PPM/YEAR)",
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         "DATA: NOAA/GML",
                         analysis_date=datetime.datetime.now())

    # Save the plot
    ppm_growth_save_path = 'figures/ppm_growth_rate.png'
    save_plot(fig_ppm_growth, ppm_growth_save_path)

    print(f"\nCO2 PPM growth rate plot saved to {ppm_growth_save_path}")

    # Create fourth visualization: Emissions growth rate
    print("\n" + "="*60)
    print("Creating emissions growth rate visualization...")
    print("="*60)

    emissions_trend_type = 'decadal_mean'  # Options: 'linear', 'decadal_mean', 'rolling_average'
    fig_emissions_growth = create_emissions_growth_rate_plot(df, trend_type=emissions_trend_type)

    # Format plot
    format_plot_title(plt.gca(),
                     "",
                     "ANNUAL GLOBAL INCREASE OF CO\N{SUBSCRIPT TWO} EMISSIONS",
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         "DATA: GLOBAL CARBON PROJECT (2024) \n2024 emissions projection from Tiseo (2025)",
                         analysis_date=datetime.datetime.now())

    # Save the plot
    emissions_growth_save_path = 'figures/emissions_growth_rate.png'
    save_plot(fig_emissions_growth, emissions_growth_save_path)

    print(f"\nEmissions growth rate plot saved to {emissions_growth_save_path}")


if __name__ == "__main__":
    main()

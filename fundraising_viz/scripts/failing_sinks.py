import pandas as pd
import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot, COLORS


def load_forest_sink_data(csv_path):
    """
    Load forest carbon sink data from CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to the forest_sink.csv file

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: year, forest_sink_gt
    """
    df = pd.read_csv(csv_path)

    # Remove BOM if present and strip whitespace from column names
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()

    # Ensure year is integer
    df['year'] = df['year'].astype(int)

    # Convert sink values to negative (sinks remove carbon from atmosphere)
    df['forest_sink_gt'] = -df['forest_sink_gt']

    return df


def load_canada_forest_data(excel_path):
    """
    Load Canadian managed forest net flux data from Excel file.

    Parameters:
    -----------
    excel_path : str
        Path to the EN_Ch6_Tables_FullTimeSeries.xlsx file

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: year, net_flux_mt (in megatons CO2e)
    """
    # Read the Excel file without headers
    df = pd.read_excel(excel_path, sheet_name='Table 6–5 All Years', header=None)

    # Row 6 (index 6) contains "Net flux – reported and not reported (kt CO2 eq)"
    # Years start at column 4 (index 4) onwards in row 2 (index 2)
    net_flux_row = df.iloc[6, 4:].values  # Get values from row 6, starting at column 4
    years_row = df.iloc[2, 4:].values     # Get years from row 2, starting at column 4

    # Create DataFrame
    result_df = pd.DataFrame({
        'year': years_row,
        'net_flux_kt': net_flux_row
    })

    # Clean data - remove any NaN values
    result_df = result_df.dropna()

    # Convert year to integer
    result_df['year'] = result_df['year'].astype(int)

    # Convert from kilotons to megatons for easier reading
    result_df['net_flux_mt'] = result_df['net_flux_kt'] / 1000.0

    # Drop the kilotons column
    result_df = result_df[['year', 'net_flux_mt']]

    return result_df


def create_forest_sink_plot(df):
    """
    Create a bar chart showing the declining global net forest carbon sink.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: year, forest_sink_gt

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))

    # Plot forest sink as bars - using secondary green color
    ax.bar(df['year'], df['forest_sink_gt'],
           color=COLORS['secondary'], width=0.7, alpha=0.7)

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1.5, alpha=0.8, zorder=5)

    # Format y-axis
    ax.tick_params(axis='y', labelsize=14, labelcolor=COLORS['secondary'], length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

    # Format x-axis
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=14, length=0)
    ax.set_xlim(df['year'].min() - 0.5, df['year'].max() + 0.5)

    # Update axis spine colors
    ax.spines['left'].set_color(COLORS['secondary'])
    ax.spines['left'].set_linewidth(2)

    return fig


def create_canada_forest_plot(df):
    """
    Create a line chart showing Canadian managed forests' net GHG balance.
    Negative values indicate carbon sinks (absorbing CO2), positive values
    indicate carbon sources (releasing CO2).

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: year, net_flux_mt

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))

    # Plot net flux as a line - using primary color for emphasis
    ax.plot(df['year'], df['net_flux_mt'],
            color='#000000', linewidth=3,
            marker='', solid_capstyle='round', zorder=3)

    # Add a horizontal line at y=0 (carbon sink/source threshold)
    ax.axhline(y=0, color='#666666', linestyle='--', linewidth=2, alpha=0.8, zorder=2)

    # Add text annotation near the threshold line
    ax.text(2008, 20, 'Carbon sink/carbon source threshold',
            color='#666666', fontsize=12, ha='center', va='bottom',
            fontweight='normal')

    # Format y-axis
    ax.tick_params(axis='y', labelsize=14, length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

    # Format x-axis
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=14, length=0)
    ax.set_xlim(df['year'].min() - 1, df['year'].max() + 1)

    # Update axis spine colors
    ax.spines['left'].set_color('#333333')
    ax.spines['left'].set_linewidth(2)

    return fig


def main():
    """
    Main execution function for forest sink visualization.
    """
    # Configuration
    csv_path = 'data/carbon_cycle/forest_sink.csv'
    canada_excel_path = 'data/carbon_cycle/EN_Ch6_Tables_FullTimeSeries.xlsx'

    # Load global forest sink data
    print("Loading forest sink data...")
    df = load_forest_sink_data(csv_path)

    print(f"Data loaded: {len(df)} years ({df['year'].min():.0f} - {df['year'].max():.0f})")
    print(f"Forest sink range: {df['forest_sink_gt'].min():.2f} - {df['forest_sink_gt'].max():.2f} Gt CO2e/yr")

    # Create visualization
    print("\nCreating global forest sink visualization...")
    fig = create_forest_sink_plot(df)

    # Format plot
    format_plot_title(plt.gca(),
                     "",
                     "GLOBAL FOREST CARBON FLUX (GIGATONNES CO2e)",
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         "DATA: WRI, GIBBS ET AL. (2025) | UPDATED WITH 2024 TREE COVER LOSS",
                         analysis_date=datetime.datetime.now())

    # Save the plot
    save_path = 'figures/global_forest_flux.png'
    save_plot(fig, save_path)

    print(f"\nGlobal forest sink plot saved to {save_path}")

    # Load Canadian managed forest data
    print("\n" + "="*60)
    print("Loading Canadian managed forest data...")
    print("="*60)
    df_canada = load_canada_forest_data(canada_excel_path)

    print(f"Data loaded: {len(df_canada)} years ({df_canada['year'].min():.0f} - {df_canada['year'].max():.0f})")
    print(f"Net flux range: {df_canada['net_flux_mt'].min():.2f} - {df_canada['net_flux_mt'].max():.2f} Mt CO2e/yr")

    # Create Canadian forest visualization
    print("\nCreating Canadian managed forest visualization...")
    fig_canada = create_canada_forest_plot(df_canada)

    # Format plot
    format_plot_title(plt.gca(),
                     "",
                     "CANADIAN MANAGED FORESTS' NET GHG BALANCE",
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         "DATA: NATIONAL INVENTORY REPORT 2023, NATURAL RESOURCES CANADA",
                         analysis_date=datetime.datetime.now())

    # Save the plot
    canada_save_path = 'figures/canada_forest_carbon_balance.png'
    save_plot(fig_canada, canada_save_path)

    print(f"\nCanadian forest carbon balance plot saved to {canada_save_path}")


if __name__ == "__main__":
    main()

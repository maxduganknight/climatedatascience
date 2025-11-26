"""
Peatland Emissions Visualization

This script visualizes global peatland emissions by country from UNEP data.
Creates a bar chart comparing countries' emissions from degraded peatlands.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add parent directory to path for utils import
sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot, COLORS


def load_peatland_data(filepath):
    """
    Load peatland emissions data from CSV.

    Parameters:
    -----------
    filepath : str
        Path to the UNEP peatland emissions CSV file

    Returns:
    --------
    pandas.DataFrame
        DataFrame with country and emissions_mt_co2e columns
    """
    df = pd.read_csv(filepath)

    # Sort by emissions in descending order
    df = df.sort_values('emissions_mt_co2e', ascending=False)

    print(f"Loaded peatland emissions data for {len(df)} countries")
    print(f"Total global peatland emissions: {df['emissions_mt_co2e'].sum():.1f} Mt CO2e")
    print(f"Top emitter: {df.iloc[0]['country']} ({df.iloc[0]['emissions_mt_co2e']:.1f} Mt CO2e)")

    return df


def create_peatland_emissions_chart(df, top_n=None):
    """
    Create a horizontal bar chart of peatland emissions by country.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with country and emissions_mt_co2e columns (sorted descending)
    top_n : int, optional
        If specified, only show top N countries. If None, show all.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Filter to top N if specified
    if top_n is not None:
        plot_df = df.head(top_n).copy()
        title_suffix = f" (Top {top_n})"
    else:
        plot_df = df.copy()
        title_suffix = ""

    # Reverse order for horizontal bar chart (highest at top)
    plot_df = plot_df.iloc[::-1]

    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 10))

    # Create horizontal bar chart
    bars = ax.barh(plot_df['country'], plot_df['emissions_mt_co2e'],
                   color=COLORS['primary'], alpha=0.85, height=0.7)

    # Format axes
    # ax.set_xlabel('Emissions (Mt CO₂e/year)', fontsize=12,
    #               fontproperties=font_props.get('regular') if font_props else None)
    ax.set_ylabel('', fontsize=12)

    # Add grid for readability
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Format ticks
    ax.tick_params(axis='both', labelsize=10)

    # Add value labels on bars for top 5 countries
    top_5_indices = list(range(len(plot_df)))[-5:] if len(plot_df) >= 5 else list(range(len(plot_df)))
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        if i in top_5_indices:
            # Place label at end of bar
            ax.text(row['emissions_mt_co2e'] + 10, i, f"{row['emissions_mt_co2e']:.1f}",
                   va='center', ha='left', fontsize=9, fontweight='bold')

    # Set x-axis limits with some padding
    max_emissions = plot_df['emissions_mt_co2e'].max()
    ax.set_xlim(0, max_emissions * 1.15)

    return fig


def main():
    """
    Main execution function.
    """
    # File paths
    data_file = 'data/carbon_cycle/unep_peatland_emissions.csv'
    figures_dir = 'figures'

    # Create output directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)

    # Load peatland emissions data
    print('\n=== Loading Peatland Emissions Data ===\n')
    peatland_df = load_peatland_data(data_file)

    # Create full visualization (all countries)
    print('\n=== Creating Peatland Emissions Visualization ===\n')
    fig = create_peatland_emissions_chart(peatland_df)

    # Add title and branding
    format_plot_title(plt.gca(),
                     '',
                     'EMISSIONS FROM PEATLANDS (Mt CO\N{SUBSCRIPT TWO}e / yr)',
                     None)

    data_note = 'DATA: UNEP GLOBAL PEATLANDS ASSESSMENT.'

    add_deep_sky_branding(plt.gca(), None,
                         data_note=data_note,
                         analysis_date=datetime.now())

    # Save the plot
    save_path = os.path.join(figures_dir, 'peatland_emissions_by_country.png')
    save_plot(fig, save_path)

    print(f'\nVisualization saved to {save_path}')

    # Create top 15 version for cleaner presentation
    print('\n=== Creating Top 15 Visualization ===\n')
    fig_top15 = create_peatland_emissions_chart(peatland_df, top_n=15)

    # Add title and branding
    format_plot_title(plt.gca(),
                     '',
                     'EMISSIONS FROM PEATLANDS (Mt CO\N{SUBSCRIPT TWO}e / yr)',
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         data_note=data_note,
                         analysis_date=datetime.now())

    # Save the top 15 plot
    save_path_top15 = os.path.join(figures_dir, 'peatland_emissions_top15.png')
    save_plot(fig_top15, save_path_top15)

    print(f'Top 15 visualization saved to {save_path_top15}')

    # Print summary statistics
    print('\n=== Summary Statistics ===\n')
    print(f'Total countries: {len(peatland_df)}')
    print(f'Total global peatland emissions: {peatland_df["emissions_mt_co2e"].sum():.1f} Mt CO₂e/year')
    print(f'\nTop 5 emitters:')
    for idx, row in peatland_df.head(5).iterrows():
        pct = (row['emissions_mt_co2e'] / peatland_df['emissions_mt_co2e'].sum()) * 100
        print(f'  {row["country"]}: {row["emissions_mt_co2e"]:.1f} Mt CO₂e ({pct:.1f}% of total)')

    print('\n=== Script Complete ===\n')


if __name__ == '__main__':
    main()

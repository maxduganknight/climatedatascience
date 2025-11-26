import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import datetime
from scipy.interpolate import CubicSpline

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot

def load_venmans_pathways_data(path):
    
    cols = [
        'Year', 'IEA Actual Emissions',
        'RCP 4.5', 'RCP 2.6', 'RCP 1.9', 
        'CAT Policies & Action', 'IEA STEPS', 'Morris Growing Pressures', 'Ou Current Policies'
        ]
    
    df = pd.read_csv(path)
    df = df[cols]
    return df

def plot_pathways(df, simple):
    """
    Plot pathways against one another from 2020-2100
    """
    # Calculate emissions relative to 2021 baseline
    baseline_year = 2021
    baseline_row = df[df['Year'] == baseline_year]

    if simple:
        cols_to_plot = [
        'RCP 4.5',
        'RCP 2.6',
        'RCP 1.9',
        'IEA Actual Emissions'
        ]
    else:
        # Get all columns except Year, then ensure IEA Actual Emissions is last
        cols_to_plot = df.columns[1:].tolist()  # All columns except Year
        if 'IEA Actual Emissions' in cols_to_plot:
            cols_to_plot.remove('IEA Actual Emissions')
            cols_to_plot.append('IEA Actual Emissions')  # Add to end

    # Get 2021 values for normalization
    baseline_values = {}
    for col in df.columns[1:]:  # Skip 'Year' column
        if col in baseline_row.columns and not baseline_row[col].isna().all():
            baseline_values[col] = baseline_row[col].iloc[0]

    # Filter data to 2020-2100 and calculate relative emissions
    df_filtered = df[(df['Year'] >= 2020) & (df['Year'] <= 2100)].copy()

    # Create relative emissions for each pathway
    for col in baseline_values.keys():
        if baseline_values[col] > 0:  # Avoid division by zero
            df_filtered[f'{col}_relative'] = (df_filtered[col] / baseline_values[col]) * 100

    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 8))

    # Define colors and line styles for different pathway types
    pathway_styles = {
        'IEA Actual Emissions' : {'color': 'black', 'linestyle': '-', 'linewidth': 5},
        #'IEA STEPS': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 3},
        'IEA Announced Pledges': {'color': 'gray', 'linestyle': '-', 'linewidth': 3},
        'IPCC AR6 Implemented Policies': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 3},
        'RCP 4.5': {'color': '#d62728', 'linestyle': '--', 'linewidth': 3},
        'RCP 2.6': {'color': '#9467bd', 'linestyle': '--', 'linewidth': 3},
        'RCP 1.9': {'color': '#e377c2', 'linestyle': '--', 'linewidth': 3}
    }

    # Plot each pathway
    for col in cols_to_plot:
        if col in baseline_values and f'{col}_relative' in df_filtered.columns:
            style = pathway_styles.get(col, {'color': 'gray', 'linestyle': '-', 'linewidth': 1})

            # Remove NaN values for plotting
            plot_data = df_filtered[['Year', f'{col}_relative']].dropna()

            if len(plot_data) > 0:
                ax.plot(plot_data['Year'], plot_data[f'{col}_relative'],
                       label=col, **style)

    # Formatting
    ax.set_xlim(2020, 2100)
    ax.set_ylim(0, 150)
    ax.set_xlabel('YEAR', fontproperties=font_props.get('regular') if font_props else None)

    ax.text(2027, 111, 'ACTUAL EMISSIONS',
        fontsize=12, ha='center', va='center',
        fontweight='bold', color='black')

    ax.text(2065, 110, 'PATHWAY to 2.7°C',
        fontsize=12, ha='center', va='center',
        fontweight='bold', color='#d62728')
    
    ax.text(2057, 60, 'PATHWAY to 1.8°C',
        fontsize=12, ha='center', va='center',
        fontweight='bold', color='#9467bd')
    
    ax.text(2035, 25, 'PATHWAY to 1.6°C',
        fontsize=12, ha='center', va='center',
        fontweight='bold', color='#e377c2')
    
    if simple == False:
        ax.text(2092, 85, '4 PROJECTIONS BASED\nON CURRENT POLICIES',
            fontsize=12, ha='center', va='center', color='black')
    
    return fig


def main():
    df = load_venmans_pathways_data('data/needed_removal_capacity/venmans_pathways.csv')
    fig = plot_pathways(df, simple = True)
    # Add titles and branding for v2
    format_plot_title(plt.gca(),
                        "",
                        "EMISSIONS RELATIVE TO 2021",
                        None)

    add_deep_sky_branding(plt.gca(), None,
                            "DATA: VENMANS & CARR (2024) | IEA (2025)",
                            analysis_date=datetime.datetime.now())

    save_path = 'figures/venmans_emissions_pathways_simple.png'
    os.makedirs('figures', exist_ok=True)
    save_plot(fig, save_path)

    print(f"Net emissions plot saved to {save_path}")

    fig_complex = plot_pathways(df, simple = False)
    # Add titles and branding for v2
    format_plot_title(plt.gca(),
                        "",
                        "EMISSIONS RELATIVE TO 2021",
                        None)

    add_deep_sky_branding(plt.gca(), None,
                            "DATA: VENMANS & CARR (2024) Literature-informed likelihoods of future emissions and temperatures.",
                            analysis_date=datetime.datetime.now())

    save_path_complex = 'figures/venmans_emissions_pathways_complex.png'
    os.makedirs('figures', exist_ok=True)
    save_plot(fig_complex, save_path_complex)

    print(f"Net emissions plot saved to {save_path_complex}")

if __name__ == "__main__":
    main()

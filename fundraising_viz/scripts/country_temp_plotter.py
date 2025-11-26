import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add utils directory to path for importing styling functions
sys.path.append('/Users/max/Deep_Sky/GitHub/datascience-platform/reports')
from utils import (
    setup_enhanced_plot, format_plot_title, add_deep_sky_branding,
    save_plot, COLORS
)

def load_country_data(data_dir):
    """Load all country temperature anomaly data files."""
    country_data = {}
    data_path = Path(data_dir)
    print("Output path:")
    print(data_path)
    
    for csv_file in data_path.glob('era5_*_t2m_annual_anom.csv'):
        # Extract country name from filename
        country_name = csv_file.stem.replace('era5_', '').replace('_t2m_annual_anom', '').replace('_', ' ').title()
        
        # Load the data
        df = pd.read_csv(csv_file)
        country_data[country_name] = df
        print(f"Loaded {len(df)} years of data for {country_name}")
    
    return country_data

def plot_country_temperature_anomaly(df, country_name, save_dir=None):
    """Create temperature anomaly plot for a single country."""
    
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 8))
    
    # Create bar chart with blue/red coloring based on pre-industrial anomaly sign
    colors = ['#3498db' if anom < 0 else '#e74c3c' for anom in df['anom_preindustrial']]
    
    bars = ax.bar(df['year'], df['anom_preindustrial'], color=colors, alpha=0.8, width=0.8)
    
    # Add 10-year running mean line (similar to reference plot)
    if len(df) >= 10:
        running_mean = df['anom_preindustrial'].rolling(window=10, center=True).mean()
        ax.plot(df['year'], running_mean, color='black', linewidth=2, label='10-year running mean')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    
    # Format axes
    ax.set_xlabel('')
    ax.set_ylabel('Temperature Anomaly (°C)', fontsize=14, 
                  fontproperties=font_props.get('regular') if font_props else None)
    
    # Set y-axis limits with some padding
    y_min, y_max = df['anom_preindustrial'].min(), df['anom_preindustrial'].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Format x-axis to show every 10 years
    years = df['year']
    decade_ticks = [year for year in years if year % 10 == 0]
    ax.set_xticks(decade_ticks)
    ax.set_xticklabels(decade_ticks, fontproperties=font_props.get('regular') if font_props else None)
    
    # Format title and branding
    title = f"{country_name.upper()} ANNUAL TEMPERATURE ANOMALIES"
    subtitle = f"Temperature anomalies relative to 1850-1900 pre-industrial baseline (°C)"
    
    format_plot_title(ax, title, subtitle, font_props)
    add_deep_sky_branding(ax, font_props, 
                         data_note="DATA: ERA5 REANALYSIS FROM COPERNICUS CLIMATE CHANGE SERVICE")
    
    # Save the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{country_name.lower().replace(' ', '_')}_temp_anomalies.png"
        save_path = os.path.join(save_dir, filename)
        save_plot(fig, save_path)
    
    return fig

def plot_all_countries(data_dir='data/country_temp_anomalies', output_dir='figures/country_temp_anomalies'):
    """Plot temperature anomalies for all countries in the data directory."""
    
    # Load all country data
    country_data = load_country_data(data_dir)
    
    if not country_data:
        print("No country data files found in the specified directory.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for each country
    for country_name, df in country_data.items():
        print(f"Generating plot for {country_name}...")
        fig = plot_country_temperature_anomaly(df, country_name, output_dir)
        plt.close(fig)  # Close figure to free memory
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    # Run the plotting function for all countries
    plot_all_countries()
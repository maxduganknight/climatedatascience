import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot

def load_provincial_wildfire_data(csv_path):
    """
    Load provincial wildfire emissions data from CSV.
    Returns a DataFrame with columns: year, province, wildfire_emissions
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Provincial wildfire data not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded provincial wildfire data: {len(df)} records for {df['province'].nunique()} provinces")
    return df

def create_province_emissions_plot(province_data, province_name):
    """
    Create a line chart of annual wildfire emissions for a single province.

    Parameters:
    -----------
    province_data : pandas DataFrame
        Data for a single province with year and wildfire_emissions columns
    province_name : str
        Name of the province for plot title

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 8))

    # Plot the line
    ax.plot(province_data['year'], province_data['wildfire_emissions'],
            color='#E74C3C', linewidth=3, marker='o', markersize=6,
            solid_capstyle='round')

    # Format axes
    ax.set_xlim(province_data['year'].min() - 1, province_data['year'].max() + 1)
    ax.set_ylim(0, province_data['wildfire_emissions'].max() * 1.1)

    # X-axis formatting
    ax.set_xticks(range(int(province_data['year'].min()), int(province_data['year'].max()) + 1, 2))
    ax.tick_params(axis='both', labelsize=12)

    # Y-axis formatting
    ax.set_ylabel('', fontsize=14,
                  fontproperties=font_props.get('regular') if font_props else None)

    return fig

def generate_all_province_plots(csv_path, output_dir='figures/provinces'):
    """
    Generate wildfire emissions plots for all provinces.

    Parameters:
    -----------
    csv_path : str
        Path to the provincial wildfire emissions CSV file
    output_dir : str
        Directory to save the generated plots
    """
    # Load the data
    df = load_provincial_wildfire_data(csv_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate a plot for each province
    provinces = df['province'].unique()
    print(f"Generating plots for {len(provinces)} provinces...")

    for province in provinces:
        province_data = df[df['province'] == province].copy()
        province_data = province_data.sort_values('year').reset_index(drop=True)

        if len(province_data) == 0:
            print(f"  Skipping {province}: no data")
            continue

        # Create the plot
        fig = create_province_emissions_plot(province_data, province)

        # Add titles and branding
        format_plot_title(plt.gca(),
                         f"{province.upper()}'S WILDFIRE EMISSIONS",
                         f"Annual CO\N{SUBSCRIPT TWO} Emissions from Wildfires (Mt)",
                         None)

        add_deep_sky_branding(plt.gca(), None,
                             "DATA: COPERNICUS ATMOSPHERE MONITORING SERVICE: GFAS")

        # Save the plot
        safe_province_name = province.replace(' ', '_').replace("'", "").lower()
        save_path = os.path.join(output_dir, f'{safe_province_name}_wildfire_emissions.png')
        save_plot(fig, save_path)

        plt.close(fig)
        print(f"  Generated plot for {province}")

def main():
    """
    Main function to generate all provincial wildfire emissions plots.
    """
    csv_path = 'data/wildfire_emissions/canada_cams_wildfire_emissions_provincial.csv'

    try:
        generate_all_province_plots(csv_path)
        print("All provincial wildfire emissions plots generated successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
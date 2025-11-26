import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import zipfile
import re

# Add reports directory to Python path to import shared utilities
import sys
reports_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, reports_dir)

from utils import (
    setup_enhanced_plot, format_plot_title, add_deep_sky_branding,
    save_plot
)

def extract_shoreline_projections(kmz_path):
    """
    Extract shoreline projection data from KMZ file.
    Focus only on the data needed for Figure 4 reproduction.

    Parameters:
    -----------
    kmz_path : str
        Path to the KMZ file

    Returns:
    --------
    dict
        Dictionary containing transect data and projections for different SLR scenarios
    """
    print(f"Extracting shoreline projections from: {kmz_path}")

    # Extract KMZ
    with zipfile.ZipFile(kmz_path, 'r') as kmz_file:
        kml_content = kmz_file.read('doc.kml').decode('utf-8')

    # Parse KML
    root = ET.fromstring(kml_content)
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    # Initialize data containers
    data = {
        'initial_shorelines': {},
        'projected_shorelines': {},
        'slr_scenarios': []
    }

    # Find all folders
    folders = root.findall('.//kml:Folder', ns)
    print(f"Found {len(folders)} folders")

    for folder in folders:
        folder_name_elem = folder.find('kml:name', ns)
        if folder_name_elem is None:
            continue

        folder_name = folder_name_elem.text

        # Process initial shoreline data
        if folder_name == 'initial_shoreline':
            placemarks = folder.findall('.//kml:Placemark', ns)
            for placemark in placemarks:
                name_elem = placemark.find('kml:name', ns)
                if name_elem is None:
                    continue

                name = name_elem.text

                # Extract transect ID
                transect_match = re.search(r'#(\d+)', name)
                if not transect_match:
                    continue

                transect_id = int(transect_match.group(1))

                # Extract coordinates
                coords_elem = placemark.find('.//kml:coordinates', ns)
                if coords_elem is not None:
                    coords_text = coords_elem.text.strip()
                    if coords_text:
                        parts = coords_text.split(',')
                        if len(parts) >= 2:
                            try:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                data['initial_shorelines'][transect_id] = (lon, lat)
                            except ValueError:
                                continue

        # Process SLR scenario folders
        elif folder_name.startswith('model_FINAL_position_SLR_'):
            # Extract SLR value from folder name
            slr_match = re.search(r'SLR_(\d+)cm', folder_name)
            if not slr_match:
                continue

            slr_cm = int(slr_match.group(1))
            if slr_cm not in data['slr_scenarios']:
                data['slr_scenarios'].append(slr_cm)

            if slr_cm not in data['projected_shorelines']:
                data['projected_shorelines'][slr_cm] = {}

            # Find the modeled_shoreline subfolder within this SLR folder
            subfolders = folder.findall('kml:Folder', ns)  # Direct children only
            for subfolder in subfolders:
                subfolder_name_elem = subfolder.find('kml:name', ns)
                if subfolder_name_elem is None:
                    continue

                if subfolder_name_elem.text == 'modeled_shoreline':
                    placemarks = subfolder.findall('.//kml:Placemark', ns)
                    for placemark in placemarks:
                        name_elem = placemark.find('kml:name', ns)
                        if name_elem is None:
                            continue

                        name = name_elem.text

                        # Extract transect ID
                        transect_match = re.search(r'#(\d+)', name)
                        if not transect_match:
                            continue

                        transect_id = int(transect_match.group(1))

                        # Extract coordinates
                        coords_elem = placemark.find('.//kml:coordinates', ns)
                        if coords_elem is not None:
                            coords_text = coords_elem.text.strip()
                            if coords_text:
                                parts = coords_text.split(',')
                                if len(parts) >= 2:
                                    try:
                                        lon = float(parts[0])
                                        lat = float(parts[1])
                                        data['projected_shorelines'][slr_cm][transect_id] = (lon, lat)
                                    except ValueError:
                                        continue

    # Sort SLR scenarios
    data['slr_scenarios'].sort()

    print(f"Extracted data for {len(data['initial_shorelines'])} transects")
    print(f"Found SLR scenarios: {data['slr_scenarios']} cm")
    print(f"Projected shorelines for each scenario: {[len(data['projected_shorelines'].get(slr, {})) for slr in data['slr_scenarios']]}")

    return data

def calculate_shoreline_changes(data):
    """
    Calculate shoreline changes from initial to projected positions.

    Parameters:
    -----------
    data : dict
        Data from extract_shoreline_projections

    Returns:
    --------
    pandas.DataFrame
        DataFrame with transect_id, latitude, and shoreline changes for each SLR scenario
    """
    transect_data = []

    # Get all transect IDs that have both initial and projected data
    initial_transects = set(data['initial_shorelines'].keys())

    for slr_cm in data['slr_scenarios']:
        projected_transects = set(data['projected_shorelines'].get(slr_cm, {}).keys())
        common_transects = initial_transects.intersection(projected_transects)

        print(f"SLR {slr_cm}cm: {len(common_transects)} transects with both initial and projected data")

    # Process transects that have initial shoreline data
    for transect_id in sorted(initial_transects):
        initial_coord = data['initial_shorelines'][transect_id]
        if initial_coord is None:
            continue

        lon_init, lat_init = initial_coord

        # Create base record
        record = {
            'transect_id': transect_id,
            'latitude': lat_init,
            'longitude': lon_init
        }

        # Calculate changes for each SLR scenario
        for slr_cm in data['slr_scenarios']:
            if (slr_cm in data['projected_shorelines'] and
                transect_id in data['projected_shorelines'][slr_cm]):

                projected_coord = data['projected_shorelines'][slr_cm][transect_id]
                if projected_coord is None:
                    record[f'change_slr_{slr_cm}cm'] = np.nan
                    continue

                lon_proj, lat_proj = projected_coord

                # Calculate shoreline change in meters
                # Use approximate conversion: 1 degree longitude ≈ 111320 * cos(latitude) meters
                lat_avg = (lat_init + lat_proj) / 2
                lon_factor = 111320 * np.cos(np.radians(lat_avg))

                # Positive change = seaward movement, negative = landward movement
                change_m = (lon_proj - lon_init) * lon_factor
                record[f'change_slr_{slr_cm}cm'] = change_m
            else:
                record[f'change_slr_{slr_cm}cm'] = np.nan

        transect_data.append(record)

    df = pd.DataFrame(transect_data)

    # Sort by latitude (south to north) to match Figure 4's transect numbering
    df = df.sort_values('latitude').reset_index(drop=True)
    df['transect_number'] = range(len(df))

    print(f"Created DataFrame with {len(df)} transects")

    return df

def create_figure_4_reproduction(df, save_path=None):
    """
    Create Figure 4: Model-projected shoreline positions showing shoreline change
    vs transect number for different SLR scenarios.

    Based on paper description:
    - Shows shoreline change (meters) vs transect number (south to north)
    - Three main SLR scenarios: 1.0m, 1.5m, 2.0m (yellow, orange, red)

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame from calculate_shoreline_changes
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 8))

    # Define the three main scenarios to match Figure 4
    # Convert cm to m for labeling
    target_scenarios = {
        100: {'label': 'SLR=1.0m', 'color': '#FFD700'},  # Yellow
        150: {'label': 'SLR=1.5m', 'color': '#FF8C00'},  # Orange
        200: {'label': 'SLR=2.0m', 'color': '#DC143C'}   # Red
    }

    # Plot each scenario
    for slr_cm, scenario_info in target_scenarios.items():
        column_name = f'change_slr_{slr_cm}cm'

        if column_name in df.columns:
            # Filter out NaN values
            valid_data = df.dropna(subset=[column_name])

            if len(valid_data) > 0:
                plt.plot(valid_data['transect_number'],
                        valid_data[column_name],
                        color=scenario_info['color'],
                        linewidth=2,
                        alpha=0.8,
                        label=scenario_info['label'])

                print(f"{scenario_info['label']}: {len(valid_data)} transects, "
                      f"Mean change: {valid_data[column_name].mean():.1f}m, "
                      f"Range: {valid_data[column_name].min():.1f} to {valid_data[column_name].max():.1f}m")
            else:
                print(f"No valid data for {scenario_info['label']}")
        else:
            print(f"Column {column_name} not found in DataFrame")

    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

    # Format the plot to match Figure 4
    plt.xlabel('Transect Number (South to North)', fontproperties=font_props.get('regular'), fontsize=14)
    plt.ylabel('Projected Shoreline Change [m]', fontproperties=font_props.get('regular'), fontsize=14)

    # Set axis limits
    plt.xlim(0, len(df))

    # Add title
    title = "FLORIDA SHORELINE CHANGE PROJECTIONS"
    subtitle = f"PROJECTED SHORELINE CHANGE BY 2100 - {len(df)} TRANSECTS"
    format_plot_title(ax, title, subtitle, font_props)

    # Add legend
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)

    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Data attribution
    data_note = f"DATA: U.S. GEOLOGICAL SURVEY | COSMOS-COAST MODEL | FLORIDA ATLANTIC COAST"
    add_deep_sky_branding(ax, font_props, data_note=data_note)

    # Save the plot
    save_plot(fig, save_path)

    return fig

def main():
    """Main function to process shoreline data and create Figure 4."""

    # Define paths
    kmz_path = 'beaches/ShorelineChange_projctn_FL/ShorelineChange_projctn_FL_Case4_TrgSlopeB.kmz'
    output_dir = 'figures/shoreline/'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=== FLORIDA SHORELINE DATA PROCESSOR (SIMPLIFIED) ===\n")

    # Check if file exists
    if not os.path.exists(kmz_path):
        print(f"Error: KMZ file not found at {kmz_path}")
        return

    print(f"Processing KMZ file: {kmz_path}")

    # Extract shoreline projection data
    data = extract_shoreline_projections(kmz_path)

    # Calculate shoreline changes
    df = calculate_shoreline_changes(data)

    # Create Figure 4 reproduction
    print(f"\n=== CREATING FIGURE 4 REPRODUCTION ===")

    create_figure_4_reproduction(
        df,
        save_path=f"{output_dir}/fl_figure_4_reproduction.png"
    )

    # Save transect data to CSV
    csv_path = f"{output_dir}/fl_shoreline_changes.csv"
    df.to_csv(csv_path, index=False)
    print(f"Transect data saved to: {csv_path}")

    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Figure saved to: {output_dir}")

if __name__ == "__main__":
    main()
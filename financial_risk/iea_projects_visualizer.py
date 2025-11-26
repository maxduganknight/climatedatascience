#!/usr/bin/env python3
"""
IEA CCUS Projects Visualizer

Creates stacked bar charts for Direct Air Capture (DAC) projects from IEA CCUS database.
Generates visualizations similar to the IEA website charts showing operational and planned 
capture capacity by status and by region.

Author: Deep Sky Research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Import styling utilities from wildfire reports
sys.path.append(str(Path(__file__).parent / '../reports/wildfires_2025/scripts'))
try:
    from utils import (
        setup_space_mono_font, setup_enhanced_plot, format_plot_title, 
        add_deep_sky_branding, save_plot, COLORS
    )
    print("Successfully imported styling utilities from wildfire scripts")
except ImportError as e:
    print(f"Could not import utilities: {e}")
    print("Using fallback styling functions")
    
    # Fallback color scheme
    COLORS = {
        'primary': '#EC6252',       # Red
        'primary_dark': '#B73229',  # Darker red
        'secondary': '#5D9E7E',     # Green
        'tertiary': '#F4B942',      # Yellow/gold
        'comparison': '#929190',    # Grey
        'background': '#f8f8f8'     # Light grey background
    }
    
    def setup_space_mono_font():
        plt.rcParams['font.family'] = 'monospace'
        return None
    
    def setup_enhanced_plot(figsize=(15, 10)):
        font_props = setup_space_mono_font()
        fig = plt.figure(figsize=figsize, facecolor=COLORS['background'])
        ax = plt.gca()
        ax.set_facecolor(COLORS['background'])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('#DDDDDD')
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        return fig, ax, font_props
    
    def format_plot_title(ax, title, subtitle, font_props=None):
        ax.set_title('')
        plt.figtext(0.5, 0.93, title.upper(), fontsize=22, fontweight='bold', ha='center', va='center')
        plt.figtext(0.1, 0.87, subtitle, fontsize=16, ha='left', va='center', color='#444444')
    
    def add_deep_sky_branding(ax, font_props=None, data_note='DATA: IEA CCUS PROJECTS DATABASE. 2025e includes announced capacity which could come online by the end of 2025.'):
        plt.figtext(0.1, 0.01, f'ANALYSIS: DEEP SKY RESEARCH\n{data_note}', 
                   fontsize=12, color='#505050', ha='left', va='bottom')
    
    def save_plot(fig, save_path):
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Use consistent layout adjustments for both charts now that they both have external legends
            plt.subplots_adjust(bottom=0.15, top=0.85, left=0.08, right=0.75)
            plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
            svg_path = save_path.replace('.png', '.svg')
            plt.savefig(svg_path, format='svg', facecolor=fig.get_facecolor(), bbox_inches='tight')
            print(f"Figure saved to: {save_path} and {svg_path}")


def load_and_filter_data(csv_path):
    """
    Load IEA CCUS projects data and filter for DAC projects only.
    
    Parameters:
    -----------
    csv_path : str
        Path to the IEA CCUS projects CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame containing only DAC projects
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} total projects from IEA CCUS database")
    
    # Filter for DAC projects only
    dac_df = df[df['Sector'] == 'DAC'].copy()
    
    print(f"Found {len(dac_df)} DAC projects")
    
    # Clean and process the data
    # Convert capacity columns to numeric, handling ranges
    for col in ['Announced capacity (Mt CO2/yr)', 'Estimated capacity by IEA (Mt CO2/yr)']:
        if col in dac_df.columns:
            dac_df[col] = pd.to_numeric(dac_df[col], errors='coerce')
    
    # Use IEA estimated capacity as primary capacity measure
    dac_df['capacity'] = dac_df['Estimated capacity by IEA (Mt CO2/yr)'].fillna(
        dac_df['Announced capacity (Mt CO2/yr)']
    )
    
    dac_df['status'] = dac_df['Project Status'].fillna('Other')
    
    dac_df['region'] = dac_df['Region'].fillna(dac_df['Region'])
    
    return dac_df


def create_capacity_by_status_chart(dac_df, output_dir):
    """
    Create stacked bar chart showing DAC capacity by project status over time.
    Mimics the left chart from the IEA screenshot.
    
    Parameters:
    -----------
    dac_df : pandas.DataFrame
        DAC projects data
    output_dir : str
        Directory to save output files
    """
    # Create data for each timeline year (2025, 2030, 2035)
    timeline_years = [2025, 2030, 2035]
    timeline_data = []
    
    for year in timeline_years:
        year_data = {'timeline': f'{year}e' if year == 2025 else str(year)}
        
        # Initialize capacity counters
        operational_capacity = 0
        under_construction_planned_capacity = 0
        
        for _, project in dac_df.iterrows():
            project_capacity = project.get('capacity', 0)
            if pd.isna(project_capacity) or project_capacity == 0:
                continue
                
            operation_year = project.get('Operation')
            project_status = project.get('status', '')
            
            # Skip cancelled projects
            if project_status == 'Cancelled':
                continue
            
            # Determine project status for this timeline year
            if pd.notna(operation_year):
                try:
                    op_year = int(float(operation_year))
                    if op_year <= year:
                        # Project is operational by this year
                        operational_capacity += project_capacity
                    else:
                        # Project not yet operational - combine under construction and planned
                        under_construction_planned_capacity += project_capacity
                except (ValueError, TypeError):
                    # No valid operation year, use current status
                    if project_status == 'Operational':
                        operational_capacity += project_capacity
                    else:
                        under_construction_planned_capacity += project_capacity
            else:
                # No operation year specified, use current status
                if project_status == 'Operational':
                    operational_capacity += project_capacity
                else:
                    under_construction_planned_capacity += project_capacity
        
        # Add data for this year
        timeline_data.append({
            'timeline': year_data['timeline'],
            'Operational': operational_capacity,
            'Under construction / Planned': under_construction_planned_capacity
        })
    
    # Convert to DataFrame and pivot
    timeline_df = pd.DataFrame(timeline_data)
    pivot_data = timeline_df.set_index('timeline')
    
    # Keep original chart focused on current projections (2025-2035)
    
    # Define colors for different statuses (matching IEA style)
    status_colors = {
        'Operational': '#4472C4',      # Blue
        'Under construction / Planned': '#FFC000'  # Yellow/Orange
    }
    
    # Original chart colors only
    
    # Set up the plot with same dimensions as region chart
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))
    
    # Create stacked bar chart with narrower bars
    bottom = np.zeros(len(pivot_data.index))
    
    for status in ['Operational', 'Under construction / Planned']:
        if status in pivot_data.columns:
            values = pivot_data[status].values
            bars = ax.bar(range(len(pivot_data.index)), values, bottom=bottom, 
                         color=status_colors.get(status, COLORS['comparison']), 
                         label=status, width=0.5)
            
            # Add numeric labels for non-zero values
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value > 0:
                    label_y = bottom[i] + value/2
                    ax.text(bar.get_x() + bar.get_width()/2, label_y, f'{value:.2f}', 
                           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            
            bottom += values
    
    # Customize the plot
    ax.set_xticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.index, fontsize=14)
    # ax.set_ylabel('Mt CO2 per year', fontsize=14)
    
    # Extend y-axis to 100 MTpa to show trajectory scale
    ax.set_ylim(0, 100)
    
    # Add trajectory curve showing needed growth path to 2050 target
    current_capacity = pivot_data.loc['2025e', 'Operational']  # Starting point (operational only)
    target_capacity = 10000.0  # 10 Gt target
    
    # Create smooth trajectory curve from 2025 to 2050
    smooth_x = np.linspace(0, 2, 100)  # Only to 2035 (index 2) to show how it shoots off
    # Gentler exponential growth curve: more gradual acceleration
    progress_ratio = smooth_x / 3  # Progress from 2025 to 2050 (3 units = 25 years)
    smooth_capacity = current_capacity * ((target_capacity/current_capacity) ** (progress_ratio ** 0.7))  # Gentler curve with power 0.7
    
    # Plot trajectory curve (will shoot off the chart)
    ax.plot(smooth_x, smooth_capacity, 'r-', linewidth=3, alpha=0.8, 
           label='Required trajectory to meet 2050 target', zorder=10)
    
    # Add legend with trajectory line
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4472C4', label='Operational'),
        Patch(facecolor='#FFC000', label='Under construction / Planned'),
        plt.Line2D([0], [0], color='red', linewidth=3, alpha=0.8, label='Required trajectory to 2050')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(.82, 1.1), fontsize=9, 
             frameon=True, facecolor=COLORS['background'], edgecolor='#DDDDDD')
    
    # Format titles
    format_plot_title(ax, 
                     'Operational and planned DAC capacity by status', 
                     'Mt CO2 per year')
    
    # Add branding
    add_deep_sky_branding(ax, data_note='DATA: IEA CCUS PROJECTS DATABASE')
    
    # Save the plot
    output_path = os.path.join(output_dir, 'dac_capacity_by_status.png')
    save_plot(fig, output_path)
    
    plt.close(fig)
    return fig


def create_dac_capacity_2050_chart(dac_df, output_dir):
    """
    Create chart showing DAC capacity projections against 2050 target and required trajectory.
    Shows the gap between current projections and what's needed to meet climate goals.
    """
    # Create data for each timeline year (2025, 2030, 2035)
    timeline_years = [2025, 2030, 2035]
    timeline_data = []
    
    for year in timeline_years:
        year_data = {'timeline': f'{year}e' if year == 2025 else str(year)}
        
        # Initialize capacity counters
        operational_capacity = 0
        under_construction_planned_capacity = 0
        
        for _, project in dac_df.iterrows():
            project_capacity = project.get('capacity', 0)
            if pd.isna(project_capacity) or project_capacity == 0:
                continue
                
            operation_year = project.get('Operation')
            project_status = project.get('status', '')
            
            # Skip cancelled projects
            if project_status == 'Cancelled':
                continue
            
            # Determine project status for this timeline year
            if pd.notna(operation_year):
                try:
                    op_year = int(float(operation_year))
                    if op_year <= year:
                        # Project is operational by this year
                        operational_capacity += project_capacity
                    else:
                        # Project not yet operational - combine under construction and planned
                        under_construction_planned_capacity += project_capacity
                except (ValueError, TypeError):
                    # No valid operation year, use current status
                    if project_status == 'Operational':
                        operational_capacity += project_capacity
                    else:
                        under_construction_planned_capacity += project_capacity
            else:
                # No operation year specified, use current status
                if project_status == 'Operational':
                    operational_capacity += project_capacity
                else:
                    under_construction_planned_capacity += project_capacity
        
        # Add data for this year
        timeline_data.append({
            'timeline': year_data['timeline'],
            'Operational': operational_capacity,
            'Under construction / Planned': under_construction_planned_capacity
        })
    
    # Convert to DataFrame and pivot
    timeline_df = pd.DataFrame(timeline_data)
    pivot_data = timeline_df.set_index('timeline')
    
    # Add 2050 target for context (10 Gt = 10,000 Mt CO2/yr needed by 2050)
    target_2050 = pd.DataFrame({
        'timeline': ['2050 Target'],
        'Operational': [10000.0],  # 10 Gt target
        'Under construction': [0.0],
        'Planned': [0.0]
    }).set_index('timeline')
    
    # Combine current data with 2050 target
    pivot_data = pd.concat([pivot_data, target_2050])
    
    # Define colors for different statuses (matching IEA style)
    status_colors = {
        'Operational': '#4472C4',      # Blue
        'Under construction': '#70AD47', # Green  
        'Planned': '#FFC000'           # Yellow/Orange
    }
    
    # Special styling for 2050 target bar
    target_color = '#E74C3C'  # Red for target
    
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))
    
    # Create simplified chart showing only operational capacity for 2025, 2030, 2035 and 2050 target
    operational_values = []
    
    for idx in pivot_data.index:
        if idx == '2050 Target':
            operational_values.append(10000.0)  # 2050 target
        else:
            # Show only operational capacity (blue bars)
            operational_values.append(pivot_data.loc[idx, 'Operational'])
    
    # Use colors: blue for current projections, red for 2050 target
    colors = []
    for i, idx in enumerate(pivot_data.index):
        if idx == '2050 Target':
            colors.append(target_color)
        else:
            colors.append('#4472C4')  # Blue for operational capacity
    
    # Create bars with minimum height for visibility of small values
    display_values = []
    for val in operational_values:
        if val > 0 and val < 50:  # Make very small bars more visible
            display_values.append(max(val, 50))  # Minimum visual height
        else:
            display_values.append(val)
    
    bars = ax.bar(range(len(pivot_data.index)), display_values, 
                 color=colors, width=0.5, alpha=0.8)
    
    # Add small visual indicators for the actual small values (but no text labels)
    for i, (actual_val, display_val) in enumerate(zip(operational_values, display_values)):
        if 0 < actual_val < 50:
            # Add a small line to show the actual value
            ax.plot([i-0.2, i+0.2], [actual_val, actual_val], 'k-', linewidth=2, alpha=0.6)
    
    # Customize the plot
    ax.set_xticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.index, fontsize=14)
    
    # Add trajectory curve showing needed growth path to 2050 target
    current_capacity = pivot_data.loc['2025e', 'Operational']  # Starting point (operational only)
    target_capacity = 10000.0  # 10 Gt target
    
    # Create smooth trajectory curve from 2025 to 2050 only (ignoring actual 2030/2035 values)
    smooth_x = np.linspace(0, 3, 100)
    # Gentler exponential growth curve: more gradual acceleration
    progress_ratio = smooth_x / 3  # Progress from 2025 to 2050 (3 units = 25 years)
    smooth_capacity = current_capacity * ((target_capacity/current_capacity) ** (progress_ratio ** 0.7))  # Gentler curve with power 0.7
    
    # Plot smooth trajectory curve
    ax.plot(smooth_x, smooth_capacity, 'r-', linewidth=4, alpha=0.7, 
           label='Required trajectory to meet 2050 target', zorder=10)
    
    # Add custom legend for simplified chart
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4472C4', alpha=0.8, label='Projected operational capacity'),
        plt.Line2D([0], [0], color='red', linewidth=4, alpha=0.7, label='Required trajectory to 2050'),
        Patch(facecolor=target_color, label='2050 target (10 Gt/year)')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(.5, .9), 
             fontsize=9, frameon=True, facecolor=COLORS['background'], edgecolor='#DDDDDD')
    
    # Format titles
    format_plot_title(ax, 
                     'DAC capacity gap: projected vs required for 2050 target', 
                     'Mt CO2 per year')
    
    # Add branding
    add_deep_sky_branding(ax, data_note='DATA: IEA CCUS PROJECTS DATABASE')
    
    # Save the plot
    output_path = os.path.join(output_dir, 'dac_capacity_2050.png')
    save_plot(fig, output_path)
    
    plt.close(fig)
    return fig


def create_capacity_by_region_chart(dac_df, output_dir):
    """
    Create stacked bar chart showing DAC capacity by region.
    Mimics the right chart from the IEA screenshot.
    
    Parameters:
    -----------
    dac_df : pandas.DataFrame
        DAC projects data
    output_dir : str
        Directory to save output files
    """
    # Create cumulative capacity data for each timeline year
    timeline_years = [2025, 2030, 2035]
    all_timeline_data = []
    
    for year in timeline_years:
        year_label = f'{"Operational as of Feb. " if year == 2025 else "Planned by "}{year}'
        
        # For each project, determine if it's operational by this year
        for _, project in dac_df.iterrows():
            if project['status'] == 'Cancelled':
                continue
                
            project_capacity = project.get('capacity', 0)
            if pd.isna(project_capacity) or project_capacity == 0:
                continue
            
            operation_year = project.get('Operation')
            project_status = project.get('status', '')
            
            # Determine if project is operational by this timeline year
            is_operational_by_year = False
            
            if project_status == 'Operational':
                is_operational_by_year = True
            elif pd.notna(operation_year):
                try:
                    op_year = int(float(operation_year))
                    if op_year <= year:
                        is_operational_by_year = True
                except (ValueError, TypeError):
                    pass
            
            if is_operational_by_year:
                all_timeline_data.append({
                    'operational_category': year_label,
                    'region': project['region'],
                    'capacity': project_capacity
                })
    
    # Convert to DataFrame and group by timeline and region
    region_capacity = pd.DataFrame(all_timeline_data).groupby(['operational_category', 'region'])['capacity'].sum().reset_index()
    
    # Pivot for stacked bar chart
    pivot_data = region_capacity.pivot(index='operational_category', columns='region', values='capacity').fillna(0)
    
    # Calculate percentages for each category
    pivot_pct = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
    
    # Define colors for different regions (using a diverse palette)
    region_colors = {
        'North America': '#5B9BD5',    # Light blue
        'Europe': '#70AD47',           # Green
        'Middle East': '#FFC000',      # Orange
        'Asia Pacific': '#FF6B6B',     # Coral
        'Australia & New Zealand': '#4ECDC4', # Teal
        'Africa': '#95A5A6',           # Gray
        'Central & South America': '#E67E22'  # Dark orange
    }
    
    # Set up the plot with extra wide figure to accommodate legend
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))
    
    # Create stacked bar chart (100% stacked) with narrower bars
    bottom = np.zeros(len(pivot_pct.index))
    
    # Sort regions by total capacity for consistent ordering
    region_order = pivot_data.sum().sort_values(ascending=False).index
    
    for region in region_order:
        if region in pivot_pct.columns:
            values = pivot_pct[region].values
            bars = ax.bar(range(len(pivot_pct.index)), values, bottom=bottom,
                         color=region_colors.get(region, COLORS['comparison']),
                         label=region, width=0.4)
            
            # Add percentage labels for segments > 5%
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value > 5:  # Only show labels for segments > 5%
                    label_y = bottom[i] + value/2
                    ax.text(bar.get_x() + bar.get_width()/2, label_y, f'{value:.0f}%', 
                           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            
            bottom += values
    
    # Customize the plot
    ax.set_xticks(range(len(pivot_pct.index)))
    ax.set_xticklabels(pivot_pct.index, fontsize=14)
    ax.set_ylabel('%', fontsize=14)
    ax.set_ylim(0, 100)
    
    # Add legend with positioning inside the figure bounds
    ax.legend(loc='center left', bbox_to_anchor=(.82, 1.1), fontsize=9, 
             frameon=True, facecolor=COLORS['background'], edgecolor='#DDDDDD')
    
    # Format titles
    format_plot_title(ax, 
                     'Operational and planned DAC capacity by region', 
                     '%')
    
    # Add branding
    add_deep_sky_branding(ax, data_note='DATA: IEA CCUS PROJECTS DATABASE')
    
    # Save the plot
    output_path = os.path.join(output_dir, 'dac_capacity_by_region.png')
    save_plot(fig, output_path)
    
    plt.close(fig)
    return fig


def print_summary_statistics(dac_df):
    """
    Print summary statistics about the DAC projects.
    
    Parameters:
    -----------
    dac_df : pandas.DataFrame
        DAC projects data
    """
    print("\n" + "="*50)
    print("DAC PROJECTS SUMMARY STATISTICS")
    print("="*50)
    
    # Total projects by status
    print("\nProjects by Status:")
    status_counts = dac_df['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Total capacity by status
    print("\nCapacity by Status (Mt CO2/yr):")
    capacity_by_status = dac_df.groupby('status')['capacity'].sum().sort_values(ascending=False)
    for status, capacity in capacity_by_status.items():
        if pd.notna(capacity) and capacity > 0:
            print(f"  {status}: {capacity:.2f}")
    
    # Projects by region
    print("\nProjects by Region:")
    region_counts = dac_df['region'].value_counts()
    for region, count in region_counts.items():
        print(f"  {region}: {count}")
    
    # Capacity by region
    print("\nCapacity by Region (Mt CO2/yr):")
    capacity_by_region = dac_df.groupby('region')['capacity'].sum().sort_values(ascending=False)
    for region, capacity in capacity_by_region.items():
        if pd.notna(capacity) and capacity > 0:
            print(f"  {region}: {capacity:.2f}")
    
    print("\n" + "="*50)


def main():
    """
    Main function to execute the DAC projects visualization pipeline.
    """
    print("Starting IEA DAC Projects Visualization Pipeline...")
    
    # Set up paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'iea' / 'iea_ccus_projects.csv'
    output_dir = script_dir / 'iea'
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Load and filter data
    try:
        dac_df = load_and_filter_data(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if len(dac_df) == 0:
        print("No DAC projects found in the dataset.")
        return
    
    # Print summary statistics
    print_summary_statistics(dac_df)
    
    # Create visualizations
    print("\nCreating capacity by status chart...")
    fig1 = create_capacity_by_status_chart(dac_df, output_dir)
    
    print("\nCreating 2050 capacity gap analysis chart...")
    fig2 = create_dac_capacity_2050_chart(dac_df, output_dir)
    
    print("\nCreating capacity by region chart...")
    fig3 = create_capacity_by_region_chart(dac_df, output_dir)
    
    print(f"\nVisualization pipeline complete! Charts saved to: {output_dir}")
    print("Generated files:")
    print("  - dac_capacity_by_status.png/svg")
    print("  - dac_capacity_2050.png/svg") 
    print("  - dac_capacity_by_region.png/svg")


if __name__ == "__main__":
    main()
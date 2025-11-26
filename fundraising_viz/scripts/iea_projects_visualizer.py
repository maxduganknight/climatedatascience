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
from scipy.interpolate import interp1d

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot, COLORS


def setup_broken_axis(figsize=(15, 10), bottom_limit=20, scale_multiplier=1000, top_data_limit=None):
    """
    Create a broken axis plot with proportional sizing based on data ranges.

    Parameters:
    -----------
    figsize : tuple
        Figure size
    bottom_limit : float
        Upper limit of the bottom axis
    scale_multiplier : float
        Scale increase between bottom and top axes
    top_data_limit : float
        Upper limit of the top axis (auto-calculated if None)

    Returns:
    --------
    fig, ax_top, ax_bottom : matplotlib objects
        Figure and axis objects for the broken axis plot
    """
    fig = plt.figure(figsize=figsize)

    # Calculate proportional subplot sizes
    top_bottom = bottom_limit * scale_multiplier

    if top_data_limit is None:
        top_data_limit = top_bottom * 2

    bottom_range = bottom_limit
    top_range = top_data_limit - top_bottom
    total_range = bottom_range + top_range

    # Calculate grid proportions (ensure readability)
    bottom_proportion = max(0.25, bottom_range / total_range)
    top_proportion = 1 - bottom_proportion

    # Convert to grid rows (use 10 rows for fine control)
    grid_rows = 10
    bottom_rows = max(2, int(bottom_proportion * grid_rows))
    top_rows = grid_rows - bottom_rows

    # Create subplots with proportional sizing
    ax_top = plt.subplot2grid((grid_rows, 1), (0, 0), rowspan=top_rows)
    ax_bottom = plt.subplot2grid((grid_rows, 1), (top_rows, 0), rowspan=bottom_rows)

    # Remove space between subplots
    plt.subplots_adjust(hspace=0.05)

    # Add scale change annotation
    fig.text(0.17, 0.31, f'Axis scale increases:\n{scale_multiplier:,}X', ha='center', va='center',
            fontsize=10)

    return fig, ax_top, ax_bottom, top_bottom


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


def load_cdr_trajectory_data(csv_path='data/needed_removal_capacity/cdr_scaleup_output.csv'):
    """
    Load CDR trajectory data for 2050 target visualization.

    Parameters:
    -----------
    csv_path : str
        Path to the CDR scaleup output CSV file

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing engineered CDR trajectory data
    """
    try:
        df = pd.read_csv(csv_path)
        # Convert engineered_cdr_tonnes to Mt (from tonnes)
        df['engineered_cdr_mt'] = df['engineered_cdr_tonnes'] / 1e6
        return df
    except FileNotFoundError:
        print(f"Warning: Could not find CDR trajectory data at {csv_path}")
        return None


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
    timeline_years = [2020, 2025, 2030]
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

    # Load CDR trajectory data for 2050 target
    cdr_trajectory = load_cdr_trajectory_data()

    # Add transparent 2050 trajectory bars in background
    if cdr_trajectory is not None:
        # Get 2050 target from CDR scaleup data
        target_2050_data = cdr_trajectory[cdr_trajectory['year'] == 2050]
        if len(target_2050_data) > 0:
            target_2050 = target_2050_data['engineered_cdr_mt'].iloc[0]

            # Linear interpolation from 2020=0 to 2050=target
            trajectory_years = [2020, 2025, 2030]
            trajectory_values = []
            for year in trajectory_years:
                # Linear interpolation from 2020 (0 Mt) to 2050 (target)
                progress = (year - 2020) / (2050 - 2020)
                value = 0 + (target_2050 - 0) * progress
                trajectory_values.append(value)

            # Plot trajectory bars (transparent, behind main bars)
            ax.bar(range(len(pivot_data.index)), trajectory_values,
                   color='red', alpha=0.15, width=0.5, zorder=1,
                   label='Required trajectory to 2050')

    # Create stacked bar chart with narrower bars
    bottom = np.zeros(len(pivot_data.index))

    for status in ['Operational', 'Under construction / Planned']:
        if status in pivot_data.columns:
            values = pivot_data[status].values
            bars = ax.bar(range(len(pivot_data.index)), values, bottom=bottom,
                         color=status_colors.get(status, COLORS['comparison']),
                         label=status, width=0.5, zorder=2)

            # Add numeric labels for non-zero values
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value > 0:
                    label_y = bottom[i] + value/2
                    ax.text(bar.get_x() + bar.get_width()/2, label_y, f'{value:.1f}',
                           ha='center', va='center', fontsize=10, fontweight='bold', color='white')

            bottom += values
    
    # Customize the plot
    ax.set_xticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.index, fontsize=14)
    # ax.set_ylabel('Mt CO2 per year', fontsize=14)

    # Set y-axis limit to 100
    ax.set_ylim(0, 20)

    # Add legend with trajectory bars
    # from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor='#4472C4', label='Operational'),
    #     Patch(facecolor='#FFC000', label='Under construction / Planned'),
    #     Patch(facecolor='red', alpha=0.15, label='Required trajectory to 2050')
    # ]
    # ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(.82, 1.1), fontsize=9,
    #          frameon=True, facecolor=COLORS['background'], edgecolor='#DDDDDD')

    # Format titles
    format_plot_title(ax, 
                     'RAPID DAC SCALE-UP REQUIRED', 
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
    timeline_years = [2020, 2030, 2040, 2050]
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

    # Add 2050 target for context (20 Gt = 10,000 Mt CO2/yr needed by 2050)
    # target_2050 = pd.DataFrame({
    #     # 'timeline': ['2050 Target'],
    #     'Operational': [20000.0],  # 20 Gt target
    #     'Under construction': [0.0],
    #     'Planned': [0.0]
    # }) # .set_index('timeline')

    # Combine current data with 2050 target
    # pivot_data = pd.concat([pivot_data, target_2050])

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

    # Load CDR trajectory data for 2050 target
    cdr_trajectory = load_cdr_trajectory_data()

    # Add transparent 2050 trajectory bars in background
    if cdr_trajectory is not None:
        # Get 2050 target from CDR scaleup data
        target_2050_data = cdr_trajectory[cdr_trajectory['year'] == 2050]
        if len(target_2050_data) > 0:
            target_2050 = target_2050_data['engineered_cdr_mt'].iloc[0]

            # Linear interpolation from 2020=0 to 2050=target
            trajectory_values = []
            for year in timeline_years:
                # Linear interpolation from 2020 (0 Mt) to 2050 (target)
                progress = (year - 2020) / (2050 - 2020)
                value = 0 + (target_2050 - 0) * progress
                trajectory_values.append(value)

            # Plot trajectory bars (transparent, behind main bars)
            ax.bar(range(len(pivot_data.index)), trajectory_values,
                   color='red', alpha=0.15, width=0.5, zorder=1,
                   label='Required trajectory to 2050')

    # Create simplified chart showing only operational capacity for 2025, 2030, 2035 and 2050 target
    operational_values = []

    for idx in pivot_data.index:
        # if idx == '2050 Target':
        #     operational_values.append(20000.0)  # 2050 target
        # else:
            # Show only operational capacity (blue bars)
        operational_values.append(pivot_data.loc[idx, 'Operational'])

    # Use colors: blue for current projections, red for 2050 target
    colors = []
    for i, idx in enumerate(pivot_data.index):
        # if idx == '2050 Target':
        #     colors.append(target_color)
        # else:
        colors.append('#4472C4')  # Blue for operational capacity

    # Create bars with minimum height for visibility of small values
    display_values = []
    for val in operational_values:
        if val > 0 and val < 50:  # Make very small bars more visible
            display_values.append(max(val, 50))  # Minimum visual height
        else:
            display_values.append(val)

    bars = ax.bar(range(len(pivot_data.index)), display_values,
                 color=colors, width=0.5, alpha=0.8, zorder=2)

    # Add small visual indicators for the actual small values (but no text labels)
    for i, (actual_val, display_val) in enumerate(zip(operational_values, display_values)):
        if 0 < actual_val < 50:
            # Add a small line to show the actual value
            ax.plot([i-0.2, i+0.2], [actual_val, actual_val], 'k-', linewidth=2, alpha=0.6)
    
    # Customize the plot
    ax.set_xticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.index, fontsize=14)
    
    ax.annotate('TRAJECTORY NEEDED TO\nMEET 2050 TARGET',
        xy=(1.5, 17000),  
        fontsize=12, ha='center', va='center', color='#E74C3C',
        fontweight='bold')
    
    # ax.plot([1, 2.5], [11000, 20000],
    #     color='#E74C3C')
    
    # Format titles
    format_plot_title(ax, 
                     'RAPID DAC SCALE-UP REQUIRED', 
                     'Mt CO2 per year')
    
    # Add branding
    add_deep_sky_branding(ax, data_note='DATA: IEA CCUS PROJECTS DATABASE')
    
    # Save the plot
    output_path = os.path.join(output_dir, 'dac_capacity_2050.png')
    save_plot(fig, output_path)
    
    plt.close(fig)
    return fig


def create_dac_capacity_2050_with_inset(dac_df, output_dir):
    """
    Create combined chart showing DAC capacity with inset zoom of 2020-2030 period.
    This shows the relationship between the full 2050 trajectory and the near-term zoom.
    """
    # Create data for main chart (2020, 2030, 2040, 2050)
    main_timeline_years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
    main_timeline_data = []

    for year in main_timeline_years:
        year_data = {'timeline': f'{year}e' if year == 2025 else str(year)}
        operational_capacity = 0

        for _, project in dac_df.iterrows():
            project_capacity = project.get('capacity', 0)
            if pd.isna(project_capacity) or project_capacity == 0:
                continue
            operation_year = project.get('Operation')
            project_status = project.get('status', '')
            if project_status == 'Cancelled':
                continue

            if pd.notna(operation_year):
                try:
                    op_year = int(float(operation_year))
                    if op_year <= year:
                        operational_capacity += project_capacity
                except (ValueError, TypeError):
                    if project_status == 'Operational':
                        operational_capacity += project_capacity
            else:
                if project_status == 'Operational':
                    operational_capacity += project_capacity

        main_timeline_data.append({
            'timeline': year_data['timeline'],
            'Operational': operational_capacity
        })

    main_pivot_data = pd.DataFrame(main_timeline_data).set_index('timeline')

    # Create data for inset chart (2020, 2025e, 2030)
    inset_timeline_years = [2020, 2025, 2030]
    inset_timeline_data = []

    for year in inset_timeline_years:
        year_data = {'timeline': f'{year}e' if year == 2025 else str(year)}
        operational_capacity = 0
        under_construction_planned_capacity = 0

        for _, project in dac_df.iterrows():
            project_capacity = project.get('capacity', 0)
            if pd.isna(project_capacity) or project_capacity == 0:
                continue
            operation_year = project.get('Operation')
            project_status = project.get('status', '')
            if project_status == 'Cancelled':
                continue

            if pd.notna(operation_year):
                try:
                    op_year = int(float(operation_year))
                    if op_year <= year:
                        operational_capacity += project_capacity
                    else:
                        under_construction_planned_capacity += project_capacity
                except (ValueError, TypeError):
                    if project_status == 'Operational':
                        operational_capacity += project_capacity
                    else:
                        under_construction_planned_capacity += project_capacity
            else:
                if project_status == 'Operational':
                    operational_capacity += project_capacity
                else:
                    under_construction_planned_capacity += project_capacity

        inset_timeline_data.append({
            'timeline': year_data['timeline'],
            'Operational': operational_capacity,
            'Under construction / Planned': under_construction_planned_capacity
        })

    inset_pivot_data = pd.DataFrame(inset_timeline_data).set_index('timeline')

    # Set up the main plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))

    # Load CDR trajectory data
    cdr_trajectory = load_cdr_trajectory_data()

    # Add transparent trajectory bars to main chart
    if cdr_trajectory is not None:
        target_2050_data = cdr_trajectory[cdr_trajectory['year'] == 2050]
        if len(target_2050_data) > 0:
            target_2050 = target_2050_data['engineered_cdr_mt'].iloc[0]
            trajectory_values = []
            for year in main_timeline_years:
                progress = (year - 2020) / (2050 - 2020)
                value = 0 + (target_2050 - 0) * progress
                trajectory_values.append(value)

            ax.bar(range(len(main_pivot_data.index)), trajectory_values,
                   color='red', alpha=0.15, width=0.5, zorder=1,
                   label='Required trajectory to 2050')

    # Add operational capacity bars to main chart
    operational_values = [main_pivot_data.loc[idx, 'Operational'] for idx in main_pivot_data.index]
    colors = ['#4472C4'] * len(operational_values)

    display_values = []
    for val in operational_values:
        if val > 0 and val < 50:
            display_values.append(max(val, 50))
        else:
            display_values.append(val)

    ax.bar(range(len(main_pivot_data.index)), display_values,
           color=colors, width=0.5, alpha=0.8, zorder=2)

    for i, (actual_val, display_val) in enumerate(zip(operational_values, display_values)):
        if 0 < actual_val < 50:
            ax.plot([i-0.2, i+0.2], [actual_val, actual_val], 'k-', linewidth=2, alpha=0.6)

    # Customize main plot
    ax.set_xticks(range(len(main_pivot_data.index)))
    ax.set_xticklabels(main_pivot_data.index, fontsize=14)

    # Add zoom region highlight on main chart
    # from matplotlib.patches import Rectangle
    # zoom_box = Rectangle((-.25, 0), 2.5, 100, linewidth=2, edgecolor='#666666',
    #                      facecolor='none', linestyle='--', alpha=0.6, zorder=3)
    # ax.add_patch(zoom_box)

    # Create inset axes in bottom-left corner
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax, width="35%", height="35%", loc='lower left',
                          bbox_to_anchor=(0.05, 0.4, 1.4, 1.5), bbox_transform=ax.transAxes)

    # Plot trajectory bars in inset
    if cdr_trajectory is not None:
        inset_trajectory_values = []
        for year in inset_timeline_years:
            progress = (year - 2020) / (2050 - 2020)
            value = 0 + (target_2050 - 0) * progress
            inset_trajectory_values.append(value)

        ax_inset.bar(range(len(inset_pivot_data.index)), inset_trajectory_values,
                     color='red', alpha=0.15, width=0.5, zorder=1)

    # Plot stacked bars in inset
    status_colors = {
        'Operational': '#4472C4',
        'Under construction / Planned': '#FFC000'
    }

    bottom_inset = np.zeros(len(inset_pivot_data.index))
    for status in ['Operational', 'Under construction / Planned']:
        if status in inset_pivot_data.columns:
            values = inset_pivot_data[status].values
            ax_inset.bar(range(len(inset_pivot_data.index)), values, bottom=bottom_inset,
                        color=status_colors.get(status, COLORS['comparison']),
                        width=0.5, zorder=2)
            bottom_inset += values

    # Customize inset
    ax_inset.set_xticks(range(len(inset_pivot_data.index)))
    ax_inset.set_xticklabels(inset_pivot_data.index, fontsize=8)
    ax_inset.set_ylim(0, 20)
    ax_inset.tick_params(axis='y', labelsize=8)

    # Format y-axis labels to whole numbers
    from matplotlib.ticker import FuncFormatter
    ax_inset.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))

    inset_bbox = ax_inset.get_position()
    print(inset_bbox)

    ax.plot([-.14, -.18], [0, 9500],
        color='black', linestyle = ':')
    
    ax.plot([2.3, 3.3], [0, 9500],
        color='black', linestyle = ':')


    # Add legend with trajectory bars
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4472C4', label='Operational'),
        Patch(facecolor='#FFC000', label='Under construction / Planned'),
        Patch(facecolor='red', alpha=0.15, label='Required trajectory to 2050')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(.8, 1.1), fontsize=9,
             frameon=True, facecolor=COLORS['background'], edgecolor='#DDDDDD')

    # Format titles
    format_plot_title(ax,
                     'RAPID DAC SCALE-UP REQUIRED',
                     'Mt CO2 per year')

    # Add branding
    add_deep_sky_branding(ax, data_note='DATA: IEA CCUS PROJECTS DATABASE')

    # Save the plot
    output_path = os.path.join(output_dir, 'dac_capacity_2050_with_inset.png')
    save_plot(fig, output_path)

    plt.close(fig)
    return fig


def create_dac_capacity_2050_with_inset_lines(dac_df, output_dir):
    """
    Create combined chart showing DAC capacity as lines with inset zoom of 2020-2030 period.
    Similar to bar version but uses line plots instead.
    """
    # Create data for main chart (2020, 2025, 2030, 2035, 2040, 2045, 2050)
    main_timeline_years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
    main_timeline_data = []

    for year in main_timeline_years:
        operational_capacity = 0
        under_construction_planned_capacity = 0

        for _, project in dac_df.iterrows():
            project_capacity = project.get('capacity', 0)
            if pd.isna(project_capacity) or project_capacity == 0:
                continue
            operation_year = project.get('Operation')
            project_status = project.get('status', '')
            if project_status == 'Cancelled':
                continue

            if pd.notna(operation_year):
                try:
                    op_year = int(float(operation_year))
                    if op_year <= year:
                        operational_capacity += project_capacity
                    else:
                        under_construction_planned_capacity += project_capacity
                except (ValueError, TypeError):
                    if project_status == 'Operational':
                        operational_capacity += project_capacity
                    else:
                        under_construction_planned_capacity += project_capacity
            else:
                if project_status == 'Operational':
                    operational_capacity += project_capacity
                else:
                    under_construction_planned_capacity += project_capacity

        main_timeline_data.append({
            'year': year,
            'Operational': operational_capacity,
            'Under construction / Planned': under_construction_planned_capacity
        })

    main_df = pd.DataFrame(main_timeline_data)

    # Create data for inset chart (2020, 2025e, 2030)
    inset_timeline_years = [2020, 2025, 2030]
    inset_timeline_data = []

    for year in inset_timeline_years:
        operational_capacity = 0
        under_construction_planned_capacity = 0

        for _, project in dac_df.iterrows():
            project_capacity = project.get('capacity', 0)
            if pd.isna(project_capacity) or project_capacity == 0:
                continue
            operation_year = project.get('Operation')
            project_status = project.get('status', '')
            if project_status == 'Cancelled':
                continue

            if pd.notna(operation_year):
                try:
                    op_year = int(float(operation_year))
                    if op_year <= year:
                        operational_capacity += project_capacity
                    else:
                        under_construction_planned_capacity += project_capacity
                except (ValueError, TypeError):
                    if project_status == 'Operational':
                        operational_capacity += project_capacity
                    else:
                        under_construction_planned_capacity += project_capacity
            else:
                if project_status == 'Operational':
                    operational_capacity += project_capacity
                else:
                    under_construction_planned_capacity += project_capacity

        inset_timeline_data.append({
            'year': year,
            'Operational': operational_capacity,
            'Under construction / Planned': under_construction_planned_capacity
        })

    inset_df = pd.DataFrame(inset_timeline_data)

    # Set up the main plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))

    # Load CDR trajectory data
    cdr_trajectory = load_cdr_trajectory_data()

    # Plot trajectory line on main chart
    if cdr_trajectory is not None:
        target_2050_data = cdr_trajectory[cdr_trajectory['year'] == 2050]
        if len(target_2050_data) > 0:
            target_2050 = target_2050_data['engineered_cdr_mt'].iloc[0]
            trajectory_values = []
            for year in main_timeline_years:
                progress = (year - 2020) / (2050 - 2020)
                value = 0 + (target_2050 - 0) * progress
                trajectory_values.append(value)

            ax.plot(main_df['year'], trajectory_values,
                   color='red', linewidth=2.5, alpha=0.6, zorder=1,
                   label='Required trajectory to 2050', linestyle='--')

    # Plot operational and under construction/planned as lines
    main_df['Total'] = main_df['Operational'] + main_df['Under construction / Planned']

    ax.plot(main_df['year'], main_df['Under construction / Planned'],
           color='#FFC000', linewidth=2.5, zorder=2,
           label='Under construction / Planned', marker='o', markersize=6)
    ax.plot(main_df['year'], main_df['Operational'],
           color='#4472C4', linewidth=2.5, zorder=3,
           label='Operational', marker='o', markersize=6)

    # Customize main plot
    ax.set_xlabel('')
    ax.set_xlim(2020, 2050)
    ax.set_xticks([2020, 2030, 2040, 2050])

    # Create inset axes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax, width="35%", height="35%", loc='lower left',
                          bbox_to_anchor=(0.05, 0.4, 1.4, 1.5), bbox_transform=ax.transAxes)

    # Plot trajectory in inset
    if cdr_trajectory is not None:
        inset_trajectory_values = []
        for year in inset_timeline_years:
            progress = (year - 2020) / (2050 - 2020)
            value = 0 + (target_2050 - 0) * progress
            inset_trajectory_values.append(value)

        ax_inset.plot(inset_df['year'], inset_trajectory_values,
                     color='red', linewidth=2.5, alpha=0.6, zorder=1, linestyle='--')

    # Plot lines in inset
    inset_df['Total'] = inset_df['Operational'] + inset_df['Under construction / Planned']

    ax_inset.plot(inset_df['year'], inset_df['Under construction / Planned'],
                 color='#FFC000', linewidth=2.5, zorder=2, marker='o', markersize=6)
    ax_inset.plot(inset_df['year'], inset_df['Operational'],
                 color='#4472C4', linewidth=2.5, zorder=3, marker='o', markersize=6)

    # Customize inset
    ax_inset.set_xlim(2020, 2030)
    ax_inset.set_ylim(0, 20)
    ax_inset.set_xticks([2020, 2025, 2030])
    # ax_inset.set_xticklabels(['2020', '2025e', '2030'], fontsize=8)
    ax_inset.tick_params(axis='y', labelsize=8)

    # Format y-axis labels to whole numbers
    from matplotlib.ticker import FuncFormatter
    ax_inset.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))

    # Add connecting lines from inset to main chart
    ax.plot([2020, 2022], [0, 9200],
        color='black', linestyle = ':')
    
    ax.plot([2030, 2036.3], [0, 9000],
        color='black', linestyle = ':')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4472C4', alpha=0.8, label='Operational'),
        Patch(facecolor='#FFC000', alpha=0.8, label='Under construction / Planned'),
        Patch(facecolor='red', alpha=0.15, label='Required trajectory to 2050')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(.8, 1.1), fontsize=9,
             frameon=True, facecolor=COLORS['background'], edgecolor='#DDDDDD')

    # Format titles
    format_plot_title(ax,
                     'RAPID DAC SCALE-UP REQUIRED',
                     'Mt CO2 per year')

    # Add branding
    add_deep_sky_branding(ax, data_note='DATA: IEA CCUS PROJECTS DATABASE')

    # Save the plot
    output_path = os.path.join(output_dir, 'dac_capacity_2050_with_inset_lines.png')
    save_plot(fig, output_path)

    plt.close(fig)
    return fig


def create_capacity_trajectory_axis_break(dac_df, output_dir):
    """
    Create bar chart with broken y-axis showing DAC capacity trajectory.
    Uses the modular axis break utility for cleaner code.
    """
    # Calculate capacity data for 2025 and 2030
    timeline_data = []

    for year in [2025, 2030]:
        operational_capacity = 0
        under_construction_planned_capacity = 0

        for _, project in dac_df.iterrows():
            project_capacity = project.get('capacity', 0)
            if pd.isna(project_capacity) or project_capacity == 0:
                continue
            operation_year = project.get('Operation')
            project_status = project.get('status', '')
            if project_status == 'Cancelled':
                continue

            if pd.notna(operation_year):
                try:
                    op_year = int(float(operation_year))
                    if op_year <= year:
                        operational_capacity += project_capacity
                    else:
                        under_construction_planned_capacity += project_capacity
                except (ValueError, TypeError):
                    if project_status == 'Operational':
                        operational_capacity += project_capacity
                    else:
                        under_construction_planned_capacity += project_capacity
            else:
                if project_status == 'Operational':
                    operational_capacity += project_capacity
                else:
                    under_construction_planned_capacity += project_capacity

        timeline_data.append({
            'year': year,
            'year_label': f'{year}e' if year == 2025 else str(year),
            'operational': operational_capacity,
            'planned': under_construction_planned_capacity
        })

    # Load 2050 target
    cdr_trajectory = load_cdr_trajectory_data()
    target_2050 = None
    if cdr_trajectory is not None:
        target_2050_data = cdr_trajectory[cdr_trajectory['year'] == 2050]
        if len(target_2050_data) > 0:
            target_2050 = target_2050_data['engineered_cdr_mt'].iloc[0]

    # Calculate axis limits
    max_bottom_value = max([d['operational'] + d['planned'] for d in timeline_data])
    bottom_limit = max(max_bottom_value * 1.2, 20)  # At least 20 Mt, or 20% above max data
    top_data_limit = target_2050 * 1.1 if target_2050 else bottom_limit * 2000

    # Set up broken axis using the modular utility
    fig, ax_top, ax_bottom, top_bottom = setup_broken_axis(
        figsize=(15, 10),
        bottom_limit=bottom_limit,
        scale_multiplier=1000,
        top_data_limit=top_data_limit
    )

    # Define time-based positions and labels
    all_years = [2025, 2030, 2035, 2040, 2045, 2050]
    x_positions = list(range(len(all_years)))  # [0, 1, 2, 3, 4, 5]
    x_labels = ['2025', '2030', '2035', '2040', '2045', '2050 TARGET']

    # Map data years to positions
    data_positions = [0, 1, 5]  # 2025, 2030, 2050

    operational_color = COLORS['secondary']  # Green
    planned_color = COLORS['tertiary']       # Yellow/gold
    target_color = COLORS['primary']         # Red

    # Plot 2025e and 2030 bars on bottom axis (only at positions 0 and 1)
    for i, data in enumerate(timeline_data):
        pos = data_positions[i]  # Map to correct position
        total = data['operational'] + data['planned']

        # Operational capacity (bottom)
        if data['operational'] > 0:
            ax_bottom.bar(pos, data['operational'],
                         color=operational_color, width=0.6,
                         label='Operational' if i == 0 else "")

        # Planned capacity (top)
        if data['planned'] > 0:
            ax_bottom.bar(pos, data['planned'], bottom=data['operational'],
                         color=planned_color, width=0.6,
                         label='Under construction / Planned' if i == 0 else "")

        # Add value labels
        if total > 0:
            ax_bottom.text(pos, total + 1, f'{total:.2f}',
                          ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Calculate total capacities for curve
    total_2025 = timeline_data[0]['operational'] + timeline_data[0]['planned']
    total_2030 = timeline_data[1]['operational'] + timeline_data[1]['planned']

    # Plot 2050 target bar spanning both axes (at position 5)
    if target_2050 is not None:
        # Bottom part of 2050 bar (fill the bottom axis)
        ax_bottom.bar(5, bottom_limit, color=target_color, width=0.6, alpha=0.8)

        # Top part of 2050 bar (on top axis, scaled properly)
        top_bar_height = target_2050 - top_bottom
        ax_top.bar(5, top_bar_height, bottom=0, color=target_color, width=0.6, alpha=0.8)

        # Add value label on top bar
        ax_top.text(5, top_bar_height/2, f'{target_2050:.0f}',
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        # Calculate and annotate scale-up ratio
        if total_2030 > 0:
            scale_up_ratio = target_2050 / total_2030

            # Add scale-up annotation
            rounded_scale_up = round(scale_up_ratio, -2)  # Round to nearest thousand
            ax_top.annotate(f'{rounded_scale_up:.0f}X GROWTH NEEDED\nFROM 2030 TO 2050',
                           xy=(3, top_bar_height - 1700), xytext=(2.5, top_bar_height * 0.8),
                           fontsize=12, ha='center', va='center', fontweight='bold',
                           color=COLORS['primary'])

    # Customize axes
    for ax in [ax_top, ax_bottom]:
        ax.set_xlim(-0.5, 5.5)
        ax.set_xticks(x_positions)

    ax_bottom.set_ylim(0, bottom_limit)
    ax_bottom.set_xticklabels(x_labels, fontsize=14)
    ax_bottom.set_ylabel('', fontsize=14)

    # Set bottom axis y-ticks to every 10 Mt
    bottom_ticks = [0, 10, 20]
    ax_bottom.set_yticks(bottom_ticks)
    ax_bottom.set_yticklabels([f'{int(tick)}' for tick in bottom_ticks])

    ax_top.set_ylim(0, top_bar_height * 1.1 if target_2050 else 100)
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Set top axis y-ticks to every 1000 Mt (showing actual values)
    if target_2050:
        # Calculate how many 1000s fit in the top bar height
        max_top_value = top_bar_height * 1.1
        num_ticks = int(max_top_value / 1000) + 1
        top_ticks_actual = [top_bottom + (i * 1000) for i in range(num_ticks)]
        top_ticks_display = [(val - top_bottom) for val in top_ticks_actual]

        # Filter to only show ticks that fit within the axis
        valid_ticks = [(actual, display) for actual, display in zip(top_ticks_actual, top_ticks_display)
                      if display <= max_top_value]

        if valid_ticks:
            actual_vals, display_vals = zip(*valid_ticks)
            ax_top.set_yticks(display_vals)
            ax_top.set_yticklabels([f'{int(val)}' for val in actual_vals])

    # Add legend
    from matplotlib.patches import Patch
    import matplotlib.lines as mlines
    legend_elements = [
        Patch(facecolor=operational_color, label='OPERATIONAL'),
        Patch(facecolor=planned_color, label='UNDER CONSTRUCTION / PLANNED'),
        Patch(facecolor=target_color, label='DEMAND IN 2050')
    ]
    ax_top.legend(handles=legend_elements, loc='upper left', fontsize=11,
                 frameon=True, facecolor=COLORS['background'], edgecolor='#DDDDDD')

    # Add title and branding using utils
    format_plot_title(ax_top,
                     'RAPID SCALEUP OF DAC NEEDED TO MEET DEMAND',
                     'Mt CO2 per year')

    # Add branding using utils
    add_deep_sky_branding(ax_bottom, data_note='DATA: IEA CCUS PROJECTS DATABASE')

    # Save the plot using utils
    output_path = os.path.join(output_dir, 'dac_capacity_trajectory.png')
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
        year_label = f'{"Operational in " if year == 2025 else "Planned by "}{year}'
        
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
    csv_path = 'data/iea/iea_ccus_projects.csv'
    output_dir = 'figures/'
    
    # Load and filter data
    try:
        dac_df = load_and_filter_data(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Print summary statistics
    print_summary_statistics(dac_df)
    
    # Create visualizations
    print("\nCreating capacity by status chart...")
    fig1 = create_capacity_by_status_chart(dac_df, output_dir)
    
    print("\nCreating 2050 capacity gap analysis chart...")
    fig2 = create_dac_capacity_2050_chart(dac_df, output_dir)

    print("\nCreating capacity trajectory chart with axis break...")
    fig3 = create_capacity_trajectory_axis_break(dac_df, output_dir)

    # print("\nCreating 2050 capacity chart with inset zoom...")
    # fig3 = create_dac_capacity_2050_with_inset(dac_df, output_dir)

    # print("\nCreating 2050 capacity chart with inset zoom (lines)...")
    # fig4 = create_dac_capacity_2050_with_inset_lines(dac_df, output_dir)

    # print("\nCreating capacity by region chart...")
    # fig5 = create_capacity_by_region_chart(dac_df, output_dir)

if __name__ == "__main__":
    main()
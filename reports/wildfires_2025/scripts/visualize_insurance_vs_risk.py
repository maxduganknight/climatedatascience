import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.font_manager as fm

# Default colors - matching visualize_fwi.py
COLORS = {
    'primary': '#EC6252',       # Red for primary (high risk)
    'primary_dark': '#B73229',  # Darker red for very high risk
    'secondary': '#5D9E7E',     # Green for secondary (low risk)
    'tertiary': '#F4B942',      # Yellow/gold for tertiary (moderate risk)
    'comparison': '#929190',    # Grey for comparison
    'background': '#f8f8f8'     # Light grey background
}

# Risk category specific colors
RISK_COLORS = {
    'Very High Risk': COLORS['primary_dark'],
    'High Risk': COLORS['primary'],
    'Moderate Risk': COLORS['tertiary'],
    'Low Risk': COLORS['secondary']
}

def setup_space_mono_font():
    """Configure Space Mono font in the most direct way possible."""
    import os
    import matplotlib.font_manager as fm
    
    # Initialize a property that we can use for all text elements
    font_props = {}
    
    try:
        # Define font paths for different styles
        font_dir = '/Users/max/Deep_Sky/design/IBM_Plex_Sans,Space_Mono/Space_Mono'
        font_paths = {
            'regular': os.path.join(font_dir, 'SpaceMono-Regular.ttf'),
            'bold': os.path.join(font_dir, 'SpaceMono-Bold.ttf'),
            'italic': os.path.join(font_dir, 'SpaceMono-Italic.ttf'),
            'bolditalic': os.path.join(font_dir, 'SpaceMono-BoldItalic.ttf')
        }
        
        # Check if regular font file exists
        if not os.path.exists(font_paths['regular']):
            print(f"Warning: Font file {font_paths['regular']} not found")
            plt.rcParams['font.family'] = 'monospace'
            return None
        
        # Create font properties for each style
        for style, path in font_paths.items():
            if os.path.exists(path):
                font_props[style] = fm.FontProperties(fname=path)
        
        # Set default family to monospace as fallback
        plt.rcParams['font.family'] = 'monospace'
        
        print("Space Mono font properties created directly from files.")
        return font_props
        
    except Exception as e:
        print(f"Error creating Space Mono font properties: {e}")
        print("Falling back to system monospace font")
        plt.rcParams['font.family'] = 'monospace'
        return None

def plot_nonrenewal_trends_by_risk(data, title=None, save_path=None):
    """
    Create an enhanced visualization of non-renewal trends by risk category.
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing the non-renewal data with columns 'year', 'risk_category', 'non_renewed_pct'
    title : str, optional
        Custom title for the plot
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Setup font
    font_props = setup_space_mono_font()
    if font_props is None:
        # Fall back to a single font property
        font_prop = None
        # Try to use system monospace
        plt.rcParams['font.family'] = 'monospace'
    else:
        # Use the regular font as default
        font_prop = font_props['regular']
        # Use bold for title if available
        title_font = font_props.get('bold', font_props['regular'])
    
    # Prepare data
    yearly_by_risk = data.groupby(['year', 'risk_category'])['non_renewed_pct'].mean().reset_index()
    yearly_by_risk_pivot = yearly_by_risk.pivot(index='year', columns='risk_category', values='non_renewed_pct')
    
    # Calculate growth rates
    start_year = yearly_by_risk_pivot.index.min()
    end_year = yearly_by_risk_pivot.index.max()
    growth_by_risk = (yearly_by_risk_pivot.loc[end_year] / yearly_by_risk_pivot.loc[start_year] - 1) * 100
    
    # Create figure with background color
    fig = plt.figure(figsize=(15, 10), facecolor=COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(COLORS['background'])
    
    # Plot each risk category with enhanced styling
    # Create a list to store handles for ordered legend
    legend_handles = []
    legend_labels = []
    
    # Define the desired order
    desired_order = ['Very High Risk', 'High Risk', 'Moderate Risk', 'Low Risk']
    
    # First create all lines but store handles for legend ordering
    lines = {}
    for category in yearly_by_risk_pivot.columns:
        # Make the Very High Risk line thickest, then High Risk slightly less thick
        if category == 'Very High Risk':
            line_width = 3.5
        elif category == 'High Risk':
            line_width = 3
        else:
            line_width = 2.5
            
        line_style = '-'
        
        line = plt.plot(yearly_by_risk_pivot.index, yearly_by_risk_pivot[category], 
                marker='o', markersize=8,
                color=RISK_COLORS.get(category, COLORS['comparison']),
                linewidth=line_width, linestyle=line_style)
        
        # Store the line handle with its category
        lines[category] = line[0]
    
    # Now create the legend handles and labels in the desired order
    for category in desired_order:
        if category in lines:
            legend_handles.append(lines[category])
            legend_labels.append(f"{category} ({growth_by_risk[category]:.1f}% growth)")
    
    # Add any remaining categories not in the desired order list
    for category in yearly_by_risk_pivot.columns:
        if category not in desired_order and category in lines:
            legend_handles.append(lines[category])
            legend_labels.append(f"{category} ({growth_by_risk[category]:.1f}% growth)")
            
    # Add data points with annotations for the last year
    for category in yearly_by_risk_pivot.columns:
        last_year_value = yearly_by_risk_pivot[category].iloc[-1]
        plt.scatter(end_year, last_year_value, 
                   s=100, color=RISK_COLORS.get(category, COLORS['comparison']), 
                   zorder=5, edgecolor='white', linewidth=1.5)
        
        # Add annotation for final value
        plt.annotate(f"{last_year_value:.1f}%", 
                    xy=(end_year, last_year_value),
                    xytext=(10, 0), textcoords="offset points",
                    va='center', fontsize=12, fontproperties=font_prop,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none'))
    
    # X-axis formatting with font property
    plt.xticks(yearly_by_risk_pivot.index, yearly_by_risk_pivot.index, 
              fontsize=14, fontproperties=font_prop)
    plt.yticks(fontsize=14, fontproperties=font_prop)
    
    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    
    # Add subtle grid
    ax.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    
    # IMPORTANT: First adjust subplot to make room for the title
    plt.subplots_adjust(top=0.80)  # Give the title more space
        
    # First clear any existing title
    ax.set_title('')
    
    # Then create a custom text title with larger font
    plt.figtext(0.5, 0.93, title.upper(),
               fontsize=22,  # Much larger font size
               fontweight='bold',
               fontproperties=title_font if font_props else font_prop,
               ha='center',
               va='center')
    
    # Axis labels with better positioning
    plt.xlabel('')
    plt.ylabel('PERCENT OF POLICIES', fontsize=16, fontproperties=font_prop, labelpad=15)
    
    # Add explanatory subtitle
    subtitle_text = f"INSURANCE NON-RENEWAL RATE BY FIRE RISK CATEGORY ({start_year}-{end_year})"
    plt.figtext(0.5, 0.87, subtitle_text,
               fontsize=16,
               fontproperties=font_prop,
               ha='center',
               va='center',
               color='#444444')
    
    # Add legend with enhanced styling using our ordered handles and labels
    legend = plt.legend(
        handles=legend_handles,
        labels=legend_labels,
        fontsize=14, 
        frameon=True, 
        facecolor=COLORS['background'], 
        edgecolor='#DDDDDD', 
        loc='upper left', 
        prop=font_prop
    )
    
    # Add citation with font property
    plt.figtext(0.02, 0.01, 'VISUALIZATION: DEEP SKY RESEARCH\nDATA: CALIFORNIA DEPARTMENT OF INSURANCE (CDI) AND US FOREST SERVICE', 
               fontsize=12, color='#505050', 
               ha='left', va='bottom',
               fontproperties=font_prop)
    
    # Add Deep Sky icon
    icon_path = '/Users/max/Deep_Sky/design/Favicon/favicon_for_charts.png'
    if os.path.exists(icon_path):
        icon = mpimg.imread(icon_path)
        imagebox = OffsetImage(icon, zoom=0.03)
        ab = AnnotationBbox(imagebox, (0.985, 0.97),
                           xycoords='figure fraction',
                           frameon=False)
        ax.add_artist(ab)
    
    # Adjust y-axis to fit the data rather than starting at zero
    # Get the min and max values in the data
    y_min = yearly_by_risk_pivot.min().min() 
    y_max = yearly_by_risk_pivot.max().max()
    
    # Add padding (5% of the range) to avoid data touching the axes
    y_range = y_max - y_min
    padding = y_range * 0.05
    
    # Set y-axis limits with padding
    y_lower = max(0, y_min - padding)  # Don't go below 0 unless data does
    y_upper = y_max + padding
    
    # Set the limits
    ax.set_ylim(bottom=y_lower, top=y_upper)
    
        # We're NOT using tight_layout, which can constrain the title
    plt.subplots_adjust(bottom=0.15, top=0.85, left=0.1, right=0.95)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Figure saved to: {save_path} and {svg_path}")
    
    return fig

def plot_average_nonrenewal_by_risk(data, title=None, save_path=None):
    """
    Create a bar chart visualization showing non-renewal rates by risk category for 2023.
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing the non-renewal data with columns 'risk_category', 'non_renewed_pct'
    title : str, optional
        Custom title for the plot
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Setup font
    font_props = setup_space_mono_font()
    if font_props is None:
        # Fall back to a single font property
        font_prop = None
        # Try to use system monospace
        plt.rcParams['font.family'] = 'monospace'
    else:
        # Use the regular font as default
        font_prop = font_props['regular']
        # Use bold for title if available
        title_font = font_props.get('bold', font_props['regular'])
    
    # Filter to only 2023 data and calculate average non-renewal rate by risk category
    data_2023 = data[data['year'] == 2023]
    avg_by_risk = data_2023.groupby('risk_category')['non_renewed_pct'].mean().reset_index()
    
    # Order the categories in reverse (Low Risk to Very High Risk)
    desired_order = ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
    avg_by_risk['order'] = avg_by_risk['risk_category'].map({cat: i for i, cat in enumerate(desired_order)})
    avg_by_risk = avg_by_risk.sort_values('order')
    
    # Create figure with background color
    fig = plt.figure(figsize=(15, 10), facecolor=COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(COLORS['background'])
    
    # Create the bar chart
    bars = plt.bar(
        avg_by_risk['risk_category'], 
        avg_by_risk['non_renewed_pct'],
        color=[RISK_COLORS.get(cat, COLORS['comparison']) for cat in avg_by_risk['risk_category']],
        width=0.6,
        edgecolor='white',
        linewidth=1
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', 
            va='bottom',
            fontsize=14,
            fontproperties=font_prop,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none')
        )
    
    # X and Y axis formatting
    plt.xticks(fontsize=14, fontproperties=font_prop)
    plt.yticks(fontsize=14, fontproperties=font_prop)
    
    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    
    # Add subtle grid
    ax.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    
    # IMPORTANT: First adjust subplot to make room for the title
    plt.subplots_adjust(top=0.80)  # Give the title more space
    
    # First clear any existing title
    ax.set_title('')
    
    # Then create a custom text title with larger font
    plt.figtext(0.5, 0.93, title.upper() if title else 'INSURERS ARE LEAVING WILDFIRE PRONE AREAS OF CALIFORNIA',
               fontsize=22,  # Much larger font size
               fontweight='bold',
               fontproperties=title_font if font_props else font_prop,
               ha='center',
               va='center')
    
    # Axis labels with better positioning
    plt.xlabel('')
    plt.ylabel('NON-RENEWAL RATE IN 2023 (%)', fontsize=16, fontproperties=font_prop, labelpad=15)
    
    # Add explanatory subtitle
    subtitle_text = "INSURANCE NON-RENEWAL RATE BY FIRE RISK CATEGORY (2023)"
    plt.figtext(0.5, 0.87, subtitle_text,
               fontsize=16,
               fontproperties=font_prop,
               ha='center',
               va='center',
               color='#444444')
    
    # Add citation with font property
    plt.figtext(0.02, 0.01, 'VISUALIZATION: DEEP SKY RESEARCH\nDATA: CALIFORNIA DEPARTMENT OF INSURANCE (CDI) AND US FOREST SERVICE', 
               fontsize=12, color='#505050', 
               ha='left', va='bottom',
               fontproperties=font_prop)
    
    # Add Deep Sky icon
    icon_path = '/Users/max/Deep_Sky/design/Favicon/favicon_for_charts.png'
    if os.path.exists(icon_path):
        icon = mpimg.imread(icon_path)
        imagebox = OffsetImage(icon, zoom=0.03)
        ab = AnnotationBbox(imagebox, (0.985, 0.97),
                           xycoords='figure fraction',
                           frameon=False)
        ax.add_artist(ab)
    
    # Set y-axis to start from zero for bar charts
    ax.set_ylim(bottom=0)
    
    # Add some padding at the top of the y-axis
    y_max = max(avg_by_risk['non_renewed_pct']) * 1.15
    ax.set_ylim(top=y_max)
    
    # We're NOT using tight_layout, which can constrain the title
    plt.subplots_adjust(bottom=0.15, top=0.85, left=0.1, right=0.95)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Figure saved to: {save_path} and {svg_path}")
    
    return fig

if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('insurance_vs_risk.csv')
    
    # Create output directory if it doesn't exist
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames with today's date
    today = datetime.now().strftime('%Y%m%d')
    
    # 1. Create the time series line chart
    output_file_trends = f'{output_dir}/nonrenewal_trends_by_risk_{today}.png'
    fig_trends = plot_nonrenewal_trends_by_risk(
        data=data,
        title='INSURERS ARE LEAVING WILDFIRE PRONE AREAS OF CALIFORNIA',
        save_path=output_file_trends
    )
    
    # 2. Create the bar chart of average rates for 2023
    output_file_avg = f'{output_dir}/nonrenewal_avg_by_risk_2023_{today}.png'
    fig_avg = plot_average_nonrenewal_by_risk(
        data=data,
        title='INSURERS ARE LEAVING WILDFIRE PRONE AREAS OF CALIFORNIA',
        save_path=output_file_avg
    )
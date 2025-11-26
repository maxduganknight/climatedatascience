import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.font_manager as fm

# Default colors - shared across all scripts
COLORS = {
    'primary': '#EC6252',       # Red for primary (high risk)
    'primary_dark': '#B73229',  # Darker red for very high risk
    'secondary': '#5D9E7E',     # Green for secondary (low risk)
    'tertiary': '#F4B942',      # Yellow/gold for tertiary (moderate risk)
    'comparison': '#929190',    # Grey for comparison
    'background': '#f8f8f8'     # Light grey background
}

RISK_COLORS = {
    'Extreme Fire Risk': COLORS['primary_dark'],  # Red
    'High Fire Risk': COLORS['primary'],       # Orange
    'Low Fire Risk': COLORS['comparison']         # Grey
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
        return font_props
        
    except Exception as e:
        print(f"Error creating Space Mono font properties: {e}")
        print("Falling back to system monospace font")
        plt.rcParams['font.family'] = 'monospace'
        return None

def filter_by_state_zip_codes(df, state_code='CA', zip_column='zip', 
                             zcta_file_path='../data/shapefiles/us/zip_codes/zcta_county_rel_10.txt'):
    """
    Filter a DataFrame to include only ZIP codes that belong to a specific state.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing ZIP codes to filter
    state_code : str, default 'CA'
        Two-letter state code (e.g., 'CA' for California, 'TX' for Texas)
    zip_column : str, default 'zip'
        Name of the column containing ZIP codes in the DataFrame
    zcta_file_path : str, default path to Census file
        Path to the downloaded Census ZCTA to County Relationship File
        
    Returns:
    --------
    pandas DataFrame
        Filtered DataFrame containing only rows with ZIP codes in the specified state
    """
    
    # Mapping of state postal codes to FIPS codes
    state_fips = {
        'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09', 
        'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 
        'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24', 'MA': '25', 
        'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 
        'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39', 
        'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47', 
        'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 
        'WY': '56', 'DC': '11', 'PR': '72'
    }
    
    # Convert state code to FIPS code
    if state_code in state_fips:
        state_fips_code = state_fips[state_code]
        print(f"Using FIPS code '{state_fips_code}' for state '{state_code}'")
    else:
        print(f"Warning: Unknown state code '{state_code}'. Attempting to use directly as FIPS.")
        state_fips_code = state_code
    
    # Ensure ZIP codes are standardized as 5-digit strings
    df = df.copy()
    df[zip_column] = df[zip_column].astype(str).str.zfill(5)
    
    try:
        print(f"Fetching ZIP codes for {state_code} from ZCTA relationship file...")
        
        # Use Census ZIP code relationship files
        zcta_county_rel = pd.read_csv(zcta_file_path, dtype={'ZCTA5': str, 'STATE': str})
        
        # Filter by state FIPS code (not postal code)
        state_zips_df = zcta_county_rel[zcta_county_rel['STATE'] == state_fips_code]
        
        if len(state_zips_df) == 0:
            print(f"No ZIP codes found for state FIPS code '{state_fips_code}' in the ZCTA file")
            # Fall back to pattern matching
            state_zips = set()
        else:
            state_zips = set(state_zips_df['ZCTA5'].astype(str).str.zfill(5))
            print(f"Found {len(state_zips)} unique ZIP codes for {state_code}")
            
    except Exception as e:
        print(f"Error fetching ZIP codes: {e}")
        state_zips = set()
    
    # Fall back to simple pattern matching if we couldn't get the zip codes
    if not state_zips:
        print("Using simple ZIP code pattern matching as fallback")
        if state_code == 'CA':
            # California ZIP codes generally start with 9
            state_zips = set(df[df[zip_column].str.startswith('9')][zip_column])
            print(f"Found {len(state_zips)} California ZIP codes starting with '9'")
        elif state_code == 'TX':
            # Texas ZIP codes generally start with 7 or 88
            state_zips = set(df[(df[zip_column].str.startswith('7')) | 
                              (df[zip_column].str.startswith('88'))][zip_column])
            print(f"Found {len(state_zips)} Texas ZIP codes starting with '7' or '88'")
        else:
            print(f"No pattern matching available for state {state_code}, returning original dataframe")
            return df
    
    # Filter the DataFrame to keep only rows with ZIP codes in the state
    if state_zips:
        initial_count = len(df)
        filtered_df = df[df[zip_column].isin(state_zips)]
        kept_count = len(filtered_df)
        
        if initial_count > 0:
            percent_kept = (kept_count / initial_count) * 100
        else:
            percent_kept = 0
            
        print(f"Filtered to {kept_count} of {initial_count} rows ({percent_kept:.1f}%) containing {state_code} ZIP codes")
        
        return filtered_df
    else:
        print(f"Warning: Could not determine {state_code} ZIP codes, returning original dataframe")
        return df

def process_whp_risk_data(raw_whp_df, state_code):
    """
    Process the WHP (Wildfire Hazard Potential) data.
    """
    print(f"Processing WHP data for state: {state_code}")
    clean_whp_df = raw_whp_df.copy()

    # Remove commas from the 'h_vh_pct' column and convert to numeric
    clean_whp_df['h_vh_pct'] = clean_whp_df['h_vh_pct'].str.replace(',', '')
    clean_whp_df['h_vh_pct'] = pd.to_numeric(clean_whp_df['h_vh_pct'], errors='coerce').fillna(0)
    
    # Convert zip to string for consistent matching
    clean_whp_df['zip'] = clean_whp_df['zip'].astype(str)
    
    # Add leading zeros to ZIP codes that are less than 5 digits
    clean_whp_df['zip'] = clean_whp_df['zip'].str.zfill(5)

    clean_whp_df = clean_whp_df[clean_whp_df['STATE'] == state_code]

    return clean_whp_df

def add_risk_categories(whp_data):
    """
    Create risk categories based on h_vh_pct percentiles.
    - Extreme Fire Risk: Top 10% of h_vh_pct values
    - High Fire Risk: 50th-90th percentile
    - Low Fire Risk: 0-50th percentile
    """
    whp_data_risk_categorized = whp_data.copy()

    # Calculate percentiles for h_vh_pct
    whp_data_risk_categorized['h_vh_pct_percentile'] = whp_data_risk_categorized['h_vh_pct'].rank(pct=True) * 100
    
    # Assign risk categories based on percentiles
    conditions = [
        (whp_data_risk_categorized['h_vh_pct_percentile'] > 90),
        (whp_data_risk_categorized['h_vh_pct_percentile'] > 70),
        (whp_data_risk_categorized['h_vh_pct_percentile'] <= 50)
    ]

    choices = [
        'Extreme Fire Risk',
        'High Fire Risk', 
        'Low Fire Risk'
        ]
    
    whp_data_risk_categorized['risk_category'] = np.select(conditions, choices, default='Low Fire Risk')
    
    return whp_data_risk_categorized

def calculate_average_premium_by_risk_category(
    data, group_by_cols = None, premium_col = 'EARNED_PREMIUM_2020', denom = 'EARNED_EXPOSURE'
    ):

    if group_by_cols is None:
        group_by_cols = ['year', 'risk_category']
    yearly_premiums_by_risk = data.groupby(group_by_cols).agg(
        premium=(premium_col, 'sum'),
        denom=(denom, 'sum')
    ).reset_index()
    yearly_premiums_by_risk['average_premium_adj'] = yearly_premiums_by_risk['premium'] / yearly_premiums_by_risk['denom']

    return yearly_premiums_by_risk

def format_axes(ax, y_label, font_props=None):
    """
    Format the axes with consistent styling.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to format
    y_label : str
        Y-axis label
    font_props : dict, optional
        Font properties from setup_space_mono_font()
    """
    font_prop = font_props.get('regular') if font_props else None
    
    # X-axis formatting with font property
    plt.xticks(fontsize=14, fontproperties=font_prop)
    plt.yticks(fontsize=14, fontproperties=font_prop)
    
    # Axis labels with better positioning
    plt.xlabel('')
    plt.ylabel(y_label, fontsize=16, fontproperties=font_prop, labelpad=15)
    
    # Adjust y-axis to fit the data rather than starting at zero
    y_min = ax.get_ylim()[0]
    y_max = ax.get_ylim()[1]
    
    # Add padding (5% of the range) to avoid data touching the axes
    y_range = y_max - y_min
    padding = y_range * 0.05
    
    # Set y-axis limits with padding
    y_lower = max(0, y_min - padding)  # Don't go below 0 unless data does
    y_upper = y_max + padding
    
    # Set the limits
    ax.set_ylim(bottom=y_lower, top=y_upper)

def plot_lines_by_risk_category(df, y_val, title, subtitle, data_note, unit='dollar', 
                               save_path=None, legend_placement='upper right', 
                               show_change_labels=False, figsize=(15, 10),
                               risk_category_translations=None):
    # Set up the plot with common styling, now with custom figsize
    fig, ax, font_props = setup_enhanced_plot(figsize=figsize)
    
    # Prepare data
    yearly_by_risk_pivot = df.pivot(index='year', columns='risk_category', values=y_val)
    yearly_by_risk_pivot.index = yearly_by_risk_pivot.index.astype(int)
    
    # Plot lines for each risk category
    lines, legend_handles, legend_labels = plot_risk_lines(
        ax, yearly_by_risk_pivot, unit=unit, font_props=font_props, 
        show_change_labels=show_change_labels,
        risk_category_translations=risk_category_translations
    )
    
    # Format the plot
    format_plot_title(ax, title, subtitle, font_props)
    add_legend(ax, legend_handles, legend_labels, font_props, legend_placement)
    add_deep_sky_branding(
        ax, font_props, 
        data_note=data_note
    )
    
    # Save the plot if a path is provided
    save_plot(fig, save_path)
    
    return fig

def plot_line(df, y_val, title, subtitle, data_note, unit='dollar', save_path=None, legend_placement='upper right'):
    """
    Create a bar chart with consistent styling.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the policy data
    y_val : str
        Column name for the y-axis values
    title : str
        Title for the plot
    subtitle : str
        Subtitle for the plot
    data_note : str
        Data attribution note
    unit : str, optional
        Unit for y-axis values ('dollar' or 'percent')
    save_path : str, optional
        Path to save the figure
    legend_placement : str, optional
        Legend placement - 'upper left', 'upper right', etc.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Set up the plot with common styling
    fig, ax, font_props = setup_enhanced_plot()
    
    # Create bar chart instead of line plot
    bars = plt.bar(df['year'], df[y_val], color=COLORS['primary'], width=0.7)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        value_text = f"${height:,.0f}" if unit == 'dollar' else f"{height:.1f}%"
        ax.text(bar.get_x() + bar.get_width()/2, height + (df[y_val].max() * 0.02),
               value_text, ha='center', va='bottom', 
               fontproperties=font_props.get('regular'),
               fontsize=10, color=COLORS['primary_dark'])
    
    # Format the plot title
    format_plot_title(ax, title, subtitle, font_props)
    
    # Custom formatting for axes (without y-label)
    # Set x-axis to show every year
    ax.set_xticks(df['year'])
    ax.set_xticklabels(df['year'], fontproperties=font_props.get('regular'), fontsize=14)
    
    # Format y-axis but skip setting the label
    if unit == 'dollar':
        ax.yaxis.set_major_formatter('${x:,.0f}')
    elif unit == 'percent':
        ax.yaxis.set_major_formatter('{x:.0f}%')
    
    ax.tick_params(axis='y', labelsize=14)
    
    # Add branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    # Save the plot if a path is provided
    save_plot(fig, save_path)
    
    return fig

def setup_enhanced_plot(figsize=(15, 10)):
    """
    Set up a figure with consistent styling for Deep Sky visualizations.
    """
    # Setup font
    font_props = setup_space_mono_font()
    if (font_props is None) or ('regular' not in font_props):
        # Fall back to a single font property
        font_prop = None
        # Try to use system monospace
        plt.rcParams['font.family'] = 'monospace'
    else:
        # Use the regular font as default
        font_prop = font_props['regular']
    
    # Create figure with background color
    fig = plt.figure(figsize=figsize, facecolor=COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(COLORS['background'])
    
    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')

    # Remove tick marks on all axes (keep labels but remove the marks themselves)
    ax.tick_params(axis='both', which='both', length=3, color = COLORS['background'])

    # Add subtle grid
    ax.grid(axis='y', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    return fig, ax, font_props

def format_plot_title(ax, title, subtitle, font_props=None):
    """
    Apply consistent title and subtitle formatting to a plot.
    """
    # Clear any existing title
    ax.set_title('')
    
    # Use the bold font if available
    title_font = font_props.get('bold', font_props.get('regular')) if font_props else None
    font_prop = font_props.get('regular') if font_props else None
    
    # Create a custom centered title
    plt.figtext(0.5, 0.93, title.upper(),
               fontsize=22,
               fontweight='bold',
               fontproperties=title_font,
               ha='center',
               va='center')
    
    # Add left-aligned subtitle with more precise positioning
    plt.figtext(0.1, 0.87, subtitle,
               fontsize=16,
               fontproperties=font_prop,
               ha='left',
               va='center',
               color='#444444')

def add_deep_sky_branding(ax, font_props=None, data_note='', analysis_date=None):
    """
    Add Deep Sky branding elements to the plot.

    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add branding to
    font_props : dict, optional
        Font properties from setup_space_mono_font()
    data_note : str, optional
        Data attribution note
    analysis_date : datetime or str, optional
        Date when the analysis was performed. If datetime, will be formatted as "Jan 20, 2025"
    """
    import datetime

    font_prop = font_props.get('regular') if font_props else None

    # Format the analysis date if provided
    if analysis_date is not None:
        if isinstance(analysis_date, datetime.datetime):
            formatted_date = analysis_date.strftime("%b %d, %Y").upper()
        else:
            # If it's already a string, use it as-is
            formatted_date = str(analysis_date)

        analysis_text = f'ANALYSIS: DEEP SKY RESEARCH | CURRENT AS OF: {formatted_date}\n{data_note}'
    else:
        analysis_text = f'ANALYSIS: DEEP SKY RESEARCH\n{data_note}'

    # Add left-aligned citation at a fixed position
    plt.figtext(0.1, 0.01, analysis_text,
               fontsize=12, color='#505050',
               ha='left', va='bottom',
               fontproperties=font_prop)

    # Add Deep Sky icon at the bottom-right, aligned with chart area
    icon_path = '/Users/max/Deep_Sky/design/Favicon/favicon_for_charts.png'
    if os.path.exists(icon_path):
        icon = mpimg.imread(icon_path)
        imagebox = OffsetImage(icon, zoom=0.03)

        # Position to align with right edge of chart area (matches save_plot right margin of 0.95)
        ab = AnnotationBbox(imagebox, (0.95, 0.01),
                           xycoords='figure fraction',
                           frameon=False,
                           box_alignment=(1.0, 0.0))
        ax.add_artist(ab)

def plot_risk_lines(ax, data_pivot, unit='dollar', desired_order=None, font_props=None, 
                    show_change_labels=False, risk_category_translations=None):
    """
    Plot lines for each risk category with consistent styling.
    
    Parameters:
    -----------
    risk_category_translations : dict, optional
        Dictionary mapping English risk categories to translated versions
        e.g., {'Extreme Fire Risk': 'Risque d\'Incendie Extrême'}
    """
    if desired_order is None:
        desired_order = ['Extreme Fire Risk', 'High Fire Risk', 'Low Fire Risk']
    
    # Set up translation function
    def translate_category(category):
        if risk_category_translations and category in risk_category_translations:
            return risk_category_translations[category]
        return category
        
    # Get the actual end year values from the data instead of recalculating growth
    start_year = data_pivot.index.min()
    end_year = data_pivot.index.max()
    end_year_values = data_pivot.loc[end_year]
    
    # First create all lines but store handles for legend ordering
    lines = {}
    for category in data_pivot.columns:
        # Make the Very High Risk line thickest, then High Risk slightly less thick
        if category == 'Extreme Fire Risk':
            line_width = 3.5
        elif category == 'High Fire Risk':
            line_width = 3
        else:
            line_width = 2.5
            
        line = plt.plot(data_pivot.index, data_pivot[category], 
                marker='o', markersize=8,
                color=RISK_COLORS.get(category, COLORS['comparison']),
                linewidth=line_width)
        
        # Store the line handle with its category
        lines[category] = line[0]
    
    # Add horizontal line at y=0 for percent charts
    # if unit == 'percent':
    #     ax.axhline(y=0, color='#CCCCCC', linestyle='--', alpha=0.7)
    
    # Format y-axis based on unit
    if unit == 'percent':
        ax.yaxis.set_major_formatter('{x:.0f}%')
    elif unit == 'dollar':
        ax.yaxis.set_major_formatter('${x:,.0f}')
    
    # Now create the legend handles and labels in the desired order
    legend_handles = []
    legend_labels = []
    for category in desired_order:
        if category in lines:
            legend_handles.append(lines[category])
            translated_category = translate_category(category)
            
            if show_change_labels:
                legend_labels.append(f"{translated_category}")
            else:
                # For growth data, show the end value; for other data, calculate change
                if unit == 'percent' and 'growth' in str(data_pivot.columns).lower():
                    legend_labels.append(f"{translated_category} ({end_year_values[category]:.1f}% change)")
                else:
                    start_value = data_pivot.loc[start_year, category]
                    end_value = data_pivot.loc[end_year, category]
                    if start_value != 0:
                        change_pct = ((end_value / start_value) - 1) * 100
                        legend_labels.append(f"{translated_category} ({change_pct:.1f}% change)")
                    else:
                        legend_labels.append(f"{translated_category}")
    
    # Add any remaining categories not in the desired order list
    for category in data_pivot.columns:
        if category not in desired_order and category in lines:
            legend_handles.append(lines[category])
            if show_change_labels:
                legend_labels.append(f"{category}")  # Simplified legend when showing change labels
            else:
                if unit == 'percent' and 'growth' in str(data_pivot.columns).lower():
                    legend_labels.append(f"{category} ({end_year_values[category]:.1f}% change)")
                else:
                    start_value = data_pivot.loc[start_year, category]
                    end_value = data_pivot.loc[end_year, category]
                    if start_value != 0:
                        change_pct = ((end_value / start_value) - 1) * 100
                        legend_labels.append(f"{category}")
                    else:
                        legend_labels.append(f"{category}")
            
    # Add data points with annotations for the last year
    font_prop = font_props.get('regular') if font_props else None
    for category in data_pivot.columns:
        last_year_value = data_pivot[category].iloc[-1]
        plt.scatter(end_year, last_year_value, 
                   s=100, color=RISK_COLORS.get(category, COLORS['comparison']), 
                   zorder=5, edgecolor='white', linewidth=1.5)
        
        # Add annotation with appropriate text based on parameter
        if show_change_labels:
            # For growth data, show the actual value from the data
            display_text = f"{last_year_value:.1f}%"
        else:
            if unit == 'dollar':
                display_text = f"${last_year_value:.0f}"
            elif unit == 'percent':
                display_text = f"{last_year_value:.1f}%"
            
        plt.annotate(display_text, 
                    xy=(end_year, last_year_value),
                    xytext=(10, 0), textcoords="offset points",
                    va='center', fontsize=12, fontproperties=font_prop,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none'))
    
    return lines, legend_handles, legend_labels

def add_legend(ax, handles, labels, font_props=None, legend_placement='upper left'):
    """
    Add a consistently styled legend to the plot with flexible positioning.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add the legend to
    handles : list
        List of line objects for the legend
    labels : list
        List of labels for the legend
    font_props : dict, optional
        Font properties from setup_space_mono_font()
    legend_placement : str, optional
        Legend placement - 'upper left', 'upper right', etc.
    """
    font_prop = font_props.get('regular') if font_props else None
    
    legend = ax.legend(
        handles=handles,
        labels=labels,
        fontsize=14, 
        frameon=True, 
        facecolor=COLORS['background'], 
        edgecolor='#DDDDDD', 
        loc=legend_placement, 
        prop=font_prop
    )

def save_plot(fig, save_path):
    """
    Save the figure in multiple formats with fixed margins.
    """
    if save_path:
        # Set fixed figure margins
        plt.subplots_adjust(bottom=0.15, top=0.85, left=0.1, right=0.95)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save with explicit padding
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
        
        # Also save as SVG
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', facecolor=fig.get_facecolor())
        print(f"Figure saved to: {save_path} and {svg_path}")

def calculate_policy_count_growth(data, base_year=2015):
    """
    Calculate policy count growth relative to a base year.
    """
    # For CA data format
    if 'policies_in_force' not in data.columns and all(col in data.columns for col in ['new', 'renewed']):
        data['policies_in_force'] = data['new'] + data['renewed']
        
    # Group by year and risk category
    yearly_by_risk = data.groupby(['year', 'risk_category'], observed=False)['policies_in_force'].sum().reset_index()

    # Calculate policies in force for the base year
    base_year_policies = yearly_by_risk[yearly_by_risk['year'] == base_year].copy()
    base_year_policies = base_year_policies[['risk_category', 'policies_in_force']]
    base_year_policies.rename(columns={'policies_in_force': 'base_year_policies'}, inplace=True)
    
    # Merge with the original dataframe
    merged = pd.merge(
        yearly_by_risk,
        base_year_policies,
        on=['risk_category'],
        how='left'
    )

    merged['policies_growth'] = (merged['policies_in_force'] / merged['base_year_policies'] * 100)
    yearly_by_risk_pivot = merged.pivot(index='year', columns='risk_category', values='policies_growth')
    return yearly_by_risk_pivot
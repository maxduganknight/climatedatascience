"""
Energy Cost Learning Curves Analysis

This script creates a learning curve visualization showing how electricity prices
from different energy sources have changed as capacity increased. It mirrors the
Our World in Data learning curves chart but adds geothermal data from Cascade Institute.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import datetime

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot

# Color scheme matching the reference image
ENERGY_COLORS = {
    'solar': '#24FD8C',           # Red/coral
    'offshore_wind': '#95A5A6',   # Blue
    'onshore_wind': '#95A5A6',    # Dark blue
    'nuclear': '#95A5A6',         # Green
    'coal': '#95A5A6',
    'geothermal': '#DC143C',
    'gas_peaker': '#95A5A6',      # Gray
    'gas_combined': '#95A5A6'    
}

# Energy type labels
ENERGY_LABELS = {
    'solar': 'Solar PV',
    'offshore_wind': 'Offshore wind',
    'onshore_wind': 'Onshore wind',
    'nuclear': 'Nuclear energy',
    'coal': 'Coal',
    'geothermal': 'Geothermal',
    'gas_peaker': "Gas Peaker",
    'gas_combined': "Gas Combined Cycle"
}

# Learning rates (% price decrease per doubling of capacity)
LEARNING_RATES = {      
    'solar': '36%',
    'offshore_wind': '10%',
    'onshore_wind': '23%',
    'nuclear': 'Nuclear has become\nmore expensive',
    'coal': 'Coal has not\nbecome cheaper',
    'geothermal': '~15%'  # Estimated from Cascade Institute data
}


def load_cost_data():
    """
    Load the cost curves data from CSV.
    Returns DataFrame with energy_type, year, lcoe_mwh, and capacity_gw columns.
    """
    print('Loading energy cost data...')

    data_file = 'data/energy/cost_curves.csv'
    df = pd.read_csv(data_file)

    # Remove BOM if present
    df.columns = df.columns.str.replace('\ufeff', '')

    # Convert capacity from GW to MW for consistency with existing code
    df['capacity_mw'] = df['capacity_gw'] * 1000

    print(f'  Loaded {len(df)} data points')
    print(f'  Energy types: {df["energy_type"].unique()}')

    return df


def load_lazard_cost_data():
    """
    Load Lazard LCOE data from CSV.
    Returns DataFrame with year as index and energy types as columns.
    """
    print('Loading Lazard cost data...')

    data_file = 'data/energy/cost_curves_lazard.csv'
    df = pd.read_csv(data_file)

    # Remove BOM if present
    df.columns = df.columns.str.replace('\ufeff', '')

    print(f'  Loaded data for years: {df["year"].unique()}')
    print(f'  Energy types: {[col for col in df.columns if col != "year"]}')
    return df


def prepare_plot_data(cost_df):
    """
    Prepare data for plotting by combining cost and capacity information.
    Note: Excludes geothermal 2019 data to show a clean line from 2010 to 2025-2030.
    """
    plot_data = []

    for _, row in cost_df.iterrows():
        energy_type = row['energy_type']
        year_display = row['year']  # Keep original for display
        lcoe = row['lcoe_mwh']
        capacity = row['capacity_mw']

        # Skip geothermal 2019 data point for learning curve visualization
        if energy_type == 'geothermal' and str(year_display) == '2019':
            continue

        # Map year to lookup value for numeric sorting
        # Handle special case for "2025-2030" -> use 2030 for sorting
        if isinstance(year_display, str) and '-' in str(year_display):
            # Extract the second year from range (e.g., "2025-2030" -> 2030)
            year_lookup = int(str(year_display).split('-')[1])
        else:
            year_lookup = int(year_display) if not pd.isna(year_display) else None

        plot_data.append({
            'energy_type': energy_type,
            'year_display': str(year_display),  # For label display
            'year_numeric': year_lookup,  # For sorting
            'lcoe_mwh': lcoe,
            'capacity_mw': capacity,
            'color': ENERGY_COLORS.get(energy_type, '#888888'),
            'label': ENERGY_LABELS.get(energy_type, energy_type),
            'learning_rate': LEARNING_RATES.get(energy_type, '')
        })

    return pd.DataFrame(plot_data)


def create_cost_reduction_plot(lazard_df):
    """
    Create time-series plot showing cost reductions from 2009-2025.
    Similar to Our World in Data visualization style.
    """
    from utils import COLORS

    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 12))

    font_prop = font_props.get('regular') if font_props else None
    font_bold = font_props.get('bold') if font_props else None

    # Define colors for each energy type (matching Our World in Data style)
    # colors = {
    #     'solar': '#E8664C',           # Red/coral (solar PV)
    #     'nuclear': '#18A08D',         # Teal/green (nuclear)
    #     'onshore_wind': '#3B6B90',    # Dark blue (onshore wind)
    #     'coal': '#7C4585',            # Purple (coal)
    #     'geothermal': '#F39C12',      # Orange (geothermal)
    #     'gas_peaker': '#95A5A6',      # Gray
    #     'gas_combined': '#8B4789'     # Purple
    # }

    # Energy types to plot
    energy_types = ['solar', 'nuclear', 'onshore_wind', 'coal', 'geothermal', 'gas_peaker', 'gas_combined']

    # Get only 2009 and 2025 data
    df_2009 = lazard_df[lazard_df['year'] == 2009].copy()
    df_2025 = lazard_df[lazard_df['year'] == 2025].copy()
    df_2028 = lazard_df[lazard_df['year'] == 2028].copy()

    # Plot lines for each energy type (straight line from 2009 to 2025)
    for energy_type in energy_types:
        if energy_type in df_2009.columns and energy_type in df_2025.columns:
            # Get start value
            start_value = df_2009[energy_type].values[0]

            # For geothermal, use 2025-2030 value and plot at x=2026
            if energy_type == 'geothermal' and len(df_2028) > 0:
                end_value = df_2028[energy_type].values[0]
                end_year = 2027
            else:
                end_value = df_2025[energy_type].values[0]
                end_year = 2025

            # Skip if either value is NaN
            if pd.isna(start_value) or pd.isna(end_value):
                continue

            # Plot straight line from 2009 to end_year
            ax.plot([2009, end_year], [start_value, end_value],
                   color=ENERGY_COLORS.get(energy_type, '#888888'),
                   linewidth=3, marker='o', markersize=8,
                   solid_capstyle='round', zorder=2)

            # Calculate percentage reduction
            if start_value > 0:
                pct_reduction = ((start_value - end_value) / start_value) * 100
            else:
                pct_reduction = 0

            # Label at start (2009)
            ax.text(2009 - 0.3, start_value,
                   f'${int(start_value)}',
                   fontsize=11, ha='right', va='center',
                   color=ENERGY_COLORS.get(energy_type, '#888888'),
                   fontweight='bold')

            # Label at end (2025)
            end_label = ENERGY_LABELS.get(energy_type)

            # Position end labels
            x_offset = 0.5

            if energy_type == 'solar':
                # Add annotation for solar with percentage
                ax.annotate(f'The cost of solar electricity\ndeclined by {int(pct_reduction)}% in 16 years.',
                           xy=(2014, 300),
                           fontsize=11, ha='left', va='center', # fontweight = 'bold',
                           color=ENERGY_COLORS['solar'],
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='#95A5A6',
                                     edgecolor='#95A5A6', linewidth=1.5))
                ax.text(end_year + x_offset, end_value - 3,
                       f'${int(end_value)} {end_label}',
                       fontsize=11, ha='left', va='center',
                       color=ENERGY_COLORS.get(energy_type, '#888888'),
                       fontweight='bold')
            elif energy_type == 'onshore_wind':
                ax.text(end_year + x_offset, end_value + 3,
                       f'${int(end_value)} {end_label}',
                       fontsize=11, ha='left', va='center',
                       color=ENERGY_COLORS.get(energy_type, '#888888'),
                       fontweight='bold')
            elif energy_type == 'coal':
                ax.text(end_year + x_offset, end_value,
                       f'${int(end_value)} {end_label}',
                       fontsize=11, ha='left', va='center',
                       color=ENERGY_COLORS.get(energy_type, '#888888'),
                       fontweight='bold')
            elif energy_type == 'gas_combined':
                ax.text(end_year + x_offset, end_value - 3,
                       f'${int(end_value)} {end_label}',
                       fontsize=11, ha='left', va='center',
                       color=ENERGY_COLORS.get(energy_type, '#888888'),
                       fontweight='bold')
            elif energy_type == 'gas_peaker':
                ax.text(end_year + x_offset, end_value,
                       f'${int(end_value)} {end_label}',
                       fontsize=11, ha='left', va='center',
                       color=ENERGY_COLORS.get(energy_type, '#888888'),
                       fontweight='bold')
            elif energy_type == 'nuclear':
                ax.text(end_year + x_offset, end_value,
                       f'${int(end_value)} {end_label}',
                       fontsize=11, ha='left', va='center',
                       color=ENERGY_COLORS.get(energy_type, '#888888'),
                       fontweight='bold')
            elif energy_type == 'geothermal':
                ax.text(end_year + .2, end_value,
                       f'${int(end_value)} {end_label}',
                       fontsize=11, ha='left', va='center',
                       color=ENERGY_COLORS.get(energy_type, '#888888'),
                       fontweight='bold')
                ax.annotate('Both the learning rate and deployment of geothermal\nhas been minimal in the last decade because of limitations\nin drilling. But the cost basis is very low to start.\nAnd new breakthroughs in drilling and direct heat use have\n the potential to drop the price below $50 per MWh for DAC.',
                        xy=(2009, 25),
                        fontsize=11, ha='left', va='center', # fontweight = 'bold',
                        color=ENERGY_COLORS['geothermal'],
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#95A5A6',
                                  edgecolor='#95A5A6', linewidth=1.5))


    # Formatting
    ax.set_xlim(2008, 2029)
    ax.set_ylim(0, 365)
    ax.set_xticks([2009, 2025, 2027])
    ax.set_xticklabels(['2009', '2025', '2025-2030*'], fontsize=14)
    ax.set_ylabel('')

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig


def create_capacity_bar_chart(cost_df):
    """
    Create bar chart showing total installed capacity in 2019 for each energy type.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 8))

    font_prop = font_props.get('regular') if font_props else None
    font_bold = font_props.get('bold') if font_props else None

    # Filter for 2019 data only (year is stored as string in CSV)
    df_2019 = cost_df[cost_df['year'] == '2019'].copy()

    # Sort by capacity for better visualization
    df_2019 = df_2019.sort_values('capacity_gw', ascending=False)

    # Create bar chart
    bars = ax.bar(range(len(df_2019)), df_2019['capacity_gw'],
                   color=[ENERGY_COLORS.get(et, '#888888') for et in df_2019['energy_type']],
                   edgecolor='white', linewidth=1.5)

    # Customize x-axis
    ax.set_xticks(range(len(df_2019)))
    ax.set_xticklabels([ENERGY_LABELS.get(et, et) for et in df_2019['energy_type']],
                        rotation=0, ha='center', fontproperties=font_prop)

    # Add value labels on top of bars
    for i, (idx, row) in enumerate(df_2019.iterrows()):
        ax.text(i, row['capacity_gw'] + 30, f"{int(row['capacity_gw'])} GW",
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=ENERGY_COLORS.get(row['energy_type'], '#888888'))

    # Formatting
    ax.set_ylabel('')
    ax.set_ylim(0, df_2019['capacity_gw'].max() * 1.15)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig


def create_learning_curve_plot(plot_df):
    """
    Create the learning curve visualization showing price vs cumulative capacity.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(16, 12))

    font_prop = font_props.get('regular') if font_props else None
    font_bold = font_props.get('bold') if font_props else None

    # Set log scale for x-axis to better show the learning curve effect
    ax.set_xscale('log')

    # Plot each energy type
    for energy_type in plot_df['energy_type'].unique():
        df_energy = plot_df[plot_df['energy_type'] == energy_type].sort_values('capacity_mw')

        ax.plot(df_energy['capacity_mw'], df_energy['lcoe_mwh'],
                color=df_energy.iloc[0]['color'], linewidth=3,
                marker='o', markersize=10, markeredgecolor='white',
                markeredgewidth=2, zorder=5)

        # Add start and end labels
        first_point = df_energy.iloc[0]
        last_point = df_energy.iloc[-1]

        # Label for energy type and learning rate
        label_text = last_point['label']
        learning_rate = last_point['learning_rate']

        # Position labels based on energy type
        if energy_type == 'solar':
            # Solar - position at top
            ax.annotate(label_text,
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(8, -3), textcoords="offset fontsize",
                        fontsize=12, fontproperties=font_bold,
                        color=first_point['color'], weight='bold')
            ax.annotate(f"The cost of solar electricity\ndeclined by 82% in 9 years.",
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(10.5, -7), textcoords="offset fontsize",
                        fontsize=9, fontproperties=font_prop,
                        color=first_point['color'])

            # Add price labels at start and end
            ax.annotate(f'${int(first_point["lcoe_mwh"])}/MWh\n{first_point["year_display"]}',
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(0, 12), textcoords="offset points",
                        fontsize=10, fontproperties=font_prop,
                        color=first_point['color'], ha='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor=first_point['color'], linewidth=1.5))

            ax.annotate(f'${int(last_point["lcoe_mwh"])}/MWh\n{last_point["year_display"]}',
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(15, -5), textcoords="offset points",
                        fontsize=10, fontproperties=font_prop,
                        color=last_point['color'], ha='left',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor=last_point['color'], linewidth=1.5))

        elif energy_type == 'offshore_wind':
            # Offshore wind
            ax.annotate(label_text,
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(-1, 2), textcoords="offset fontsize",
                        fontsize=11, fontproperties=font_bold,
                        color=first_point['color'], weight='bold')
            ax.annotate(f"Learning rate: {learning_rate}",
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(-1, 1.3), textcoords="offset fontsize",
                        fontsize=9, fontproperties=font_prop,
                        color=first_point['color'])

            # Price labels
            ax.annotate(f'${int(first_point["lcoe_mwh"])}\n{first_point["year_display"]}',
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(-25, -8), textcoords="offset points",
                        fontsize=9, fontproperties=font_prop,
                        color=first_point['color'], ha='right')

            ax.annotate(f'${int(last_point["lcoe_mwh"])}\n{last_point["year_display"]}',
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(8, -8), textcoords="offset points",
                        fontsize=9, fontproperties=font_prop,
                        color=last_point['color'], ha='left')

        elif energy_type == 'onshore_wind':
            # Onshore wind
            ax.annotate(label_text,
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(-15, 2), textcoords="offset fontsize",
                        fontsize=11, fontproperties=font_bold,
                        color=last_point['color'], weight='bold')
            ax.annotate(f"Learning rate: {learning_rate}",
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(-18.5, 1), textcoords="offset fontsize",
                        fontsize=9, fontproperties=font_prop,
                        color=last_point['color'])

            # Price labels
            ax.annotate(f'${int(first_point["lcoe_mwh"])}\n{first_point["year_display"]}',
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(-10, 8), textcoords="offset points",
                        fontsize=9, fontproperties=font_prop,
                        color=first_point['color'], ha='center')

            ax.annotate(f'${int(last_point["lcoe_mwh"])}\n{last_point["year_display"]}',
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=9, fontproperties=font_prop,
                        color=last_point['color'], ha='left')

        elif energy_type == 'geothermal':
            # Geothermal - new addition
            ax.annotate(label_text,
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(-8, -6), textcoords="offset fontsize",
                        fontsize=11, fontproperties=font_bold,
                        color=last_point['color'], weight='bold')
            ax.annotate(f"Both the learning rate and the deployment\nof geothermal has been minimal in the last decade,\nbecause of the limitations in drilling.\nBut the cost basis is very low to start",
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(-23, -13), textcoords="offset fontsize",
                        fontsize=9, fontproperties=font_prop,
                        color=last_point['color'])
            
            
            ax.annotate("But new breakthroughs in drilling\nand direct heat use have the potential\nto drop the price below $50 per MWh for DAC",
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(5, 2), textcoords="offset fontsize",
                        fontsize=9, fontproperties=font_prop,
                        color=last_point['color'])
            

            # Price labels
            ax.annotate(f'${int(first_point["lcoe_mwh"])}\n{first_point["year_display"]}',
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(-8, 10), textcoords="offset points",
                        fontsize=9, fontproperties=font_prop,
                        color=first_point['color'], ha='center')

            ax.annotate(f'${int(last_point["lcoe_mwh"])}\n{last_point["year_display"]}',
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(8, -8), textcoords="offset points",
                        fontsize=9, fontproperties=font_prop,
                        color=last_point['color'], ha='left')

        elif energy_type == 'nuclear':
            # Nuclear - no learning, became more expensive
            ax.annotate(label_text,
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(4, 4), textcoords="offset fontsize",
                        fontsize=11, fontproperties=font_bold,
                        color=last_point['color'], weight='bold')
            ax.annotate(f"{learning_rate}",
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(5, 2), textcoords="offset fontsize",
                        fontsize=9, fontproperties=font_prop,
                        color=last_point['color'])

            # Price labels
            ax.annotate(f'${int(first_point["lcoe_mwh"])}\n{first_point["year_display"]}',
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(8, -8), textcoords="offset points",
                        fontsize=9, fontproperties=font_prop,
                        color=first_point['color'], ha='left')

            ax.annotate(f'${int(last_point["lcoe_mwh"])}\n{last_point["year_display"]}',
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=9, fontproperties=font_prop,
                        color=last_point['color'], ha='left')

        elif energy_type == 'coal':
            # Coal - no learning, stayed similar
            ax.annotate(label_text,
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(0, -2), textcoords="offset fontsize",
                        fontsize=11, fontproperties=font_bold,
                        color=last_point['color'], weight='bold')
            ax.annotate(f"{learning_rate}",
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(0, -5), textcoords="offset fontsize",
                        fontsize=9, fontproperties=font_prop,
                        color=last_point['color'])

            # Price labels
            ax.annotate(f'${int(first_point["lcoe_mwh"])}\n{first_point["year_display"]}',
                        xy=(first_point['capacity_mw'], first_point['lcoe_mwh']),
                        xytext=(-25, 0), textcoords="offset points",
                        fontsize=9, fontproperties=font_prop,
                        color=first_point['color'], ha='right')

            ax.annotate(f'${int(last_point["lcoe_mwh"])}\n{last_point["year_display"]}',
                        xy=(last_point['capacity_mw'], last_point['lcoe_mwh']),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=9, fontproperties=font_prop,
                        color=last_point['color'], ha='left')

    # Formatting
    ax.set_xlabel('GLOBAL CUMULATIVE INSTALLED CAPACITY (MW)',
                  fontsize=13, fontproperties=font_prop, labelpad=10)
    # ax.set_ylabel('Price per megawatt hour of electricity\n\nThis is the global weighted-average of the\nlevelized costs of energy (LCOE), without subsidies\n(logarithmic axis and adjusted for inflation)',
    #               fontsize=12, fontproperties=font_prop, labelpad=10)

    # Set axis limits similar to reference
    ax.set_xlim(2000, 3000000)
    ax.set_ylim(30, 400)
    ax.set_yscale('log')

    # Custom y-axis ticks
    ax.set_yticks([50, 100, 150, 200, 250, 300])
    ax.set_yticklabels(['$50', '$100', '$150', '$200', '$250', '$300'],
                       fontproperties=font_prop)

    # Custom x-axis ticks
    ax.set_xticks([5000, 10000, 20000, 50000, 100000, 500000, 1000000, 2000000])
    ax.set_xticklabels(['5,000 MW', '10,000 MW', '20,000 MW', '50,000 MW',
                       '100,000 MW', '500,000 MW', '1,000,000 MW', '2,000,000 MW'],
                       fontproperties=font_prop, rotation=0, ha='center')

    # Grid
    #ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)

    return fig


def main():
    """
    Main execution function.
    """
    print('='*60)
    print('Energy Cost Learning Curves Analysis')
    print('='*60)

    # Load data
    cost_df = load_cost_data()

    # Prepare plot data
    plot_df = prepare_plot_data(cost_df)

    print(f'\nPrepared {len(plot_df)} data points for plotting')

    # Create learning curve visualization
    print('\nCreating learning curve visualization...')
    fig = create_learning_curve_plot(plot_df)

    # Add title and branding
    format_plot_title(plt.gca(),
                     "",
                     "LEVELIZED COST OF ENERGY ($/MWh)",
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         "Amended from OurWorldInData.org with Canadian geothermal projection from Cascade Institute (2025). Logarithmic axes.",
                         analysis_date=datetime.datetime.now())

    # Save plot
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    save_plot(fig, 'figures/energy_cost_learning_curves.png')

    print('  ✓ Saved to figures/energy_cost_learning_curves.png')

    # Create cost reduction visualization
    print('\nCreating cost reduction time-series visualization...')
    lazard_df = load_lazard_cost_data()
    fig2 = create_cost_reduction_plot(lazard_df)

    # Add title and branding
    format_plot_title(plt.gca(),
                     "",
                     "LEVELIZED COST OF ENERGY ($/MWh)",
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         "DATA: LAZARD (2025). *GEOTHERMAL FUTURE LCOE FROM CASCADE INSTITUTE (2025).",
                         analysis_date=datetime.datetime.now())

    save_plot(fig2, 'figures/energy_cost_reductions.png')

    print('  ✓ Saved to figures/energy_cost_reductions.png')

    # Create capacity bar chart for 2019
    print('\nCreating 2019 installed capacity bar chart...')
    fig3 = create_capacity_bar_chart(cost_df)

    # Add title and branding
    format_plot_title(plt.gca(),
                     "",
                     "TOTAL INSTALLED CAPACITY (GW)",
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         "DATA: Our World in Data, IRENA 2020, LAZARD. 2019 VALUES.")

    save_plot(fig3, 'figures/energy_capacity_2019.png')

    print('  ✓ Saved to figures/energy_capacity_2019.png')
    print('\n' + '='*60)
    print('✓ Complete!')
    print('='*60)


if __name__ == '__main__':
    main()

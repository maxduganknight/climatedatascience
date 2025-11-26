import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import (
    setup_space_mono_font, setup_enhanced_plot, format_plot_title, 
    add_deep_sky_branding, save_plot, COLORS, RISK_COLORS
)

# Import state-specific processing functions
from wa_insurance_visualizer import process_wa_insurance_data
from ca_insurance_processor import process_roa_premiums_data
from tx_insurance_visualizer import process_tx_insurance_data

def calculate_state_premium_trends(state_data_dict, base_year=2018, end_year=2023):
    """
    Calculate premium growth trends for each state, normalized to 100 in the base year
    
    Parameters:
    -----------
    state_data_dict : dict
        Dictionary with state codes as keys and DataFrames as values
    base_year : int, default 2018
        Year to use as the base (100%) for calculating percentage changes
    end_year : int, default 2023
        Last year to include in the analysis
        
    Returns:
    --------
    DataFrame with yearly premium indices by state
    """
    result_df = pd.DataFrame()
    
    for state_code, data in state_data_dict.items():
        print(f"Processing data for {state_code}...")
        print(data.head())
        # Filter for years between base_year and end_year
        filtered_data = data[data['year'].between(base_year, end_year)]
        
        # Skip if no data available for the requested years
        if filtered_data.empty:
            print(f"No data for {state_code} between {base_year}-{end_year}")
            continue
        
        # Calculate percent change from base year
        if base_year not in filtered_data['year'].values:
            print(f"Base year {base_year} not available for {state_code}, skipping")
            continue
            
        base_premium = filtered_data.loc[filtered_data['year'] == base_year, 'average_premium'].iloc[0]
        filtered_data['premium_index'] = (filtered_data['average_premium'] / base_premium) * 100
        
        # Add state column and keep only relevant fields
        filtered_data['state'] = state_code
        result_df = pd.concat([result_df, filtered_data[['year', 'state', 'premium_index', 'average_premium']]])
    
    return result_df

def plot_state_premium_trends(trends_df, output_path=None):
    """
    Create a visualization comparing premium trends across states
    """
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))
    
    # Get unique states and assign colors
    states = trends_df['state'].unique()
    state_colors = {
        'CA': COLORS['primary'],
        'TX': COLORS['tertiary'],
        'WA': COLORS['secondary'],
    }
    
    # Plot a line for each state
    for state in states:
        state_data = trends_df[trends_df['state'] == state]
        
        # Sort by year to ensure line is drawn correctly
        state_data = state_data.sort_values('year')
        
        # Get state-specific color or use gray as fallback
        color = state_colors.get(state, COLORS['comparison'])
        
        # Plot the line - use premium_index (percentage) instead of raw average_premium
        line = ax.plot(state_data['year'], state_data['premium_index'], 
                      color=color, linewidth=2.5, marker='o', markersize=8, 
                      label=state)
        
        # Add label at the end of each line with percent increase
        last_point = state_data.iloc[-1]
        final_value = last_point['premium_index']
        percent_change = final_value - 100  # Subtract 100 to get percentage change from base
        
        # Add annotation for final value
        ax.annotate(f"{state}: +{percent_change:.1f}%", 
                  xy=(last_point['year'], final_value),
                  xytext=(10, 0), textcoords="offset points",
                  va='center', fontsize=12, fontproperties=font_props.get('regular'),
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none'))
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter('{x:.0f}%')
    
    # Add horizontal line at 100% (base year)
    ax.axhline(y=100, color='#CCCCCC', linestyle='--', alpha=0.7)
    
    # Set y-axis limits to show enough space above the highest point
    y_max = trends_df['premium_index'].max()
    ax.set_ylim([90, max(150, y_max * 1.1)])  # Start at 90% to show context, extend to at least 150%
    
    # Format the plot
    format_plot_title(
        ax, 
        "HOME INSURANCE PREMIUMS RISING ACROSS WILDFIRE-PRONE STATES",
        "PERCENT INCREASE IN AVERAGE HOME INSURANCE PREMIUMS (2020 DOLLARS)",
        font_props
    )
    
    # Add legend
    ax.legend(
        fontsize=14, 
        frameon=True, 
        facecolor=COLORS['background'], 
        edgecolor='#DDDDDD', 
        loc='upper left', 
        prop=font_props.get('regular')
    )
    
    # Add branding
    add_deep_sky_branding(
        ax, 
        font_props, 
        data_note="DATA: CALIFORNIA, TEXAS, WASHINGTON STATE INSURANCE DEPARTMENTS"
    )
    
    # Save the figure
    save_plot(fig, output_path)
    
    return fig

def plot_fair_plan_growth(fair_plan_df, output_path=None):
    """
    Create a visualization comparing Fair Plan growth across states
    
    Parameters:
    -----------
    fair_plan_df : pandas DataFrame
        DataFrame with columns: state, year, fair_policies_in_force
    output_path : str, optional
        Path to save the output figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object for further customization if needed
    """
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))
    
    # Get unique states and assign colors
    states = fair_plan_df['state'].unique()
    state_colors = {
        'CA': COLORS['primary'],
        'TX': COLORS['tertiary'],
        'WA': COLORS['secondary'],
    }
    
    # Track handles for legend
    handles = []
    labels = []
    
    # Plot a line for each state
    for state in states:
        state_data = fair_plan_df[fair_plan_df['state'] == state]
        
        # Sort by year to ensure line is drawn correctly
        state_data = state_data.sort_values('year')

        fair_policies_baseline = state_data.loc[state_data['year'] == 2020, 'fair_policies_in_force'].iloc[0]
        state_data['fair_policies_growth'] = ((state_data['fair_policies_in_force'] - fair_policies_baseline) / fair_policies_baseline) * 100
        
        # Get state-specific color or use gray as fallback
        color = state_colors.get(state, COLORS['comparison'])
        
        # Plot the line - use premium_index (percentage) instead of raw average_premium
        line = ax.plot(state_data['year'], state_data['fair_policies_growth'], 
                      color=color, linewidth=2.5, marker='o', markersize=8, 
                      label=state)
        
        # Store handle for legend
        handles.append(line[0])
        labels.append(state)
        
        # Add label at the end of each line with percent increase
        last_point = state_data.iloc[-1]
        final_value = last_point['fair_policies_growth']
        percent_change = final_value
        
        # Add annotation for final value
        ax.annotate(f"{state}: +{percent_change:.0f}%", 
                  xy=(last_point['year'], final_value),
                  xytext=(10, 0), textcoords="offset points",
                  va='center', fontsize=12, fontproperties=font_props.get('regular'),
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none'))
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter('{x:.0f}%')
    
    # Add horizontal line at 100% (base year)
    ax.axhline(y=0, color='#CCCCCC', linestyle='--', alpha=0.7)

    unique_years = sorted(fair_plan_df['year'].unique())
    ax.set_xticks(unique_years)
    ax.set_xticklabels([str(int(year)) for year in unique_years], fontproperties=font_props.get('regular'))
    
    
    # Set y-axis limits to show enough space above the highest point
    y_max = fair_plan_df['fair_policies_growth'].max() if 'fair_policies_growth' in fair_plan_df.columns else \
            max([state_data['fair_policies_growth'].max() for state in states])
    # ax.set_ylim([90, max(150, y_max * 1.1)])  # Start at 90% to show context
    
    # Format the plot
    format_plot_title(
        ax, 
        "HOMEOWNERS FORCED OUT OF PRIVATE INSURANCE",
        "PERCENT GROWTH IN NUMBER OF FAIR PLAN POLICIES SINCE 2020",
        font_props
    )
    
    # Add legend
    ax.legend(
        handles=handles,
        labels=labels,
        fontsize=14, 
        frameon=True, 
        facecolor=COLORS['background'], 
        edgecolor='#DDDDDD', 
        loc='upper left', 
        prop=font_props.get('regular')
    )
    
    # Add branding
    add_deep_sky_branding(
        ax, 
        font_props, 
        data_note="DATA: CALIFORNIA & TEXAS DEPARTMENTS OF INSURANCE, OREGON DEPARTMENT OF CONSUMER AND BUSINESS SERVICES"
    )
    plt.subplots_adjust(bottom=0.15, top=0.88, left=0.10, right=0.95)


    # Save the figure
    save_plot(fig, output_path)
    
    return fig

if __name__ == "__main__":
    print("Creating multi-state premium trend chart...")
    
    # Load the CPI data for inflation adjustment
    try:
        cpi = pd.read_csv('../data/cpi/cpi.csv')
        print(f"CPI data loaded for years: {cpi['year'].min()}-{cpi['year'].max()}")
    except Exception as e:
        print(f"Error loading CPI data: {e}")
        cpi = pd.DataFrame({'year': [2020], 'cpi': [1.0]})
    
    # Load and process state data
    # Washington (WA)
    try:
        wa_file_path = 'insurance/WA_Request/Supporting_Underlying_Metrics_and_Disclaimer_for_Analyses_of_US_Homeowners_Insurance_Markets_2018-2022.xlsx'
        wa_insurance_df = pd.read_excel(wa_file_path, sheet_name='Supporting Underlying Metrics')
        processed_wa_df = process_wa_insurance_data(wa_insurance_df, cpi)
        grouped_wa_df = processed_wa_df.groupby('year').agg(
            average_premium=('PREMIUM_2020', 'mean')
        ).reset_index()
        grouped_wa_df = grouped_wa_df[['year', 'average_premium']]
        print(f"Washington data processed: {len(processed_wa_df)} records")

    except Exception as e:
        print(f"Error processing Washington data: {e}")
        grouped_wa_df = pd.DataFrame()
    
    # California (CA)
    try:
        ca_file_path = 'insurance/CA_Request/PRA-2025-00261 - Premiums and Exposures by Company.csv'
        ca_insurance_df = pd.read_csv(ca_file_path)
        processed_ca_df = process_roa_premiums_data(ca_insurance_df, cpi)
        # Rename columns to match expected structure
        processed_ca_df = processed_ca_df.rename(columns={
            'EARNED_PREMIUM_2020': 'PREMIUM_2020'
        })
        grouped_ca_df = processed_ca_df.groupby('year').agg(
            total_premium=('PREMIUM_2020', 'sum'),
            exposure_years=('EARNED_EXPOSURE', 'sum')
        ).reset_index()
        grouped_ca_df['average_premium'] = grouped_ca_df['total_premium'] / grouped_ca_df['exposure_years']
        grouped_ca_df = grouped_ca_df[['year', 'average_premium']]
        print(f"California data processed: {len(processed_ca_df)} records")

    except Exception as e:
        print(f"Error processing California data: {e}")
        grouped_ca_df = pd.DataFrame()
    
    # Texas (TX)
    try:
        tx_file_path = 'insurance/TX_Request/combined_insurance_data.csv'
        tx_insurance_df = pd.read_csv(tx_file_path)
        processed_tx_df = process_tx_insurance_data(tx_insurance_df, cpi)
        grouped_tx_df = processed_tx_df.groupby('year').agg(
            total_premium=('PREMIUM_2020', 'sum'),
            policy_count=('POLICIES IN FORCE AT END OF QTR', 'sum')
        ).reset_index()
        grouped_tx_df['average_premium'] = grouped_tx_df['total_premium'] / grouped_tx_df['policy_count']
        grouped_tx_df = grouped_tx_df[['year', 'average_premium']]
        print(f"Texas data processed: {len(processed_tx_df)} records")
    except Exception as e:
        print(f"Error processing Texas data: {e}")
        grouped_tx_df = pd.DataFrame()

    # Oregon
    try:
        or_file_path = 'insurance/OR_Request/or_raw.csv'
        grouped_or_df = pd.read_csv(or_file_path)
        print(f"Oregon data processed: {len(grouped_or_df)} records")
    except Exception as e:
        print(f"Error processing Oregon data: {e}")
        grouped_or_df = pd.DataFrame()
    
    grouped_tx_df['state'] = 'TX'
    grouped_ca_df['state'] = 'CA'
    grouped_wa_df['state'] = 'WA'
    grouped_or_df['state'] = 'OR'

    state_data_dict = {
        'TX': grouped_tx_df,
        'CA': grouped_ca_df,
        'WA': grouped_wa_df,
        'OR': grouped_or_df
    }

    # Filter for only 2018-2023 and normalize to 100 in 2018
    premium_trends = calculate_state_premium_trends(state_data_dict, base_year=2018, end_year=2023)
    
    # Create output directory
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    output_file = f'{output_dir}/multi_state_premium_trends.png'
    fig = plot_state_premium_trends(premium_trends, output_file)
    
    print(f"Multi-state premium trend chart created at: {output_file}")

    # Read in and plot FAIR plan trends across states
    fair_plan_df = pd.read_csv('insurance/fair_state_comparison.csv')
    fair_plan_df = fair_plan_df[fair_plan_df['year'].between(2020, 2024)]
    fair_plan_df = fair_plan_df[fair_plan_df['state'].isin(['CA', 'TX', 'OR'])]

    fair_fig = plot_fair_plan_growth(fair_plan_df, output_path=f'{output_dir}/fair_plan_growth.png')
    print(f"FAIR plan growth chart created at: {output_dir}/fair_plan_growth.png")

    # =====================
    # FRENCH LANGUAGE CHARTS
    # =====================
    print("\nGenerating French language state comparison charts...")

    # Create French versions of state comparison charts
    # 1. French version of premium trends
    def plot_state_premium_trends_fr(trends_df, output_path=None):
        """
        Create a French visualization comparing premium trends across states
        """
        # Set up the plot
        fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))
        
        # Get unique states and assign colors
        states = trends_df['state'].unique()
        state_colors = {
            'CA': COLORS['primary'],
            'TX': COLORS['tertiary'],
            'WA': COLORS['secondary'],
        }
        
        # Plot a line for each state
        for state in states:
            state_data = trends_df[trends_df['state'] == state]
            
            # Sort by year to ensure line is drawn correctly
            state_data = state_data.sort_values('year')
            
            # Get state-specific color or use gray as fallback
            color = state_colors.get(state, COLORS['comparison'])
            
            # Plot the line - use premium_index (percentage) instead of raw average_premium
            line = ax.plot(state_data['year'], state_data['premium_index'], 
                          color=color, linewidth=2.5, marker='o', markersize=8, 
                          label=state)
            
            # Add label at the end of each line with percent increase
            last_point = state_data.iloc[-1]
            final_value = last_point['premium_index']
            percent_change = final_value - 100  # Subtract 100 to get percentage change from base
            
            # Add annotation for final value
            ax.annotate(f"{state}: +{percent_change:.1f}%", 
                      xy=(last_point['year'], final_value),
                      xytext=(10, 0), textcoords="offset points",
                      va='center', fontsize=12, fontproperties=font_props.get('regular'),
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none'))
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter('{x:.0f}%')
        
        # Add horizontal line at 100% (base year)
        ax.axhline(y=100, color='#CCCCCC', linestyle='--', alpha=0.7)
        
        # Set y-axis limits to show enough space above the highest point
        y_max = trends_df['premium_index'].max()
        ax.set_ylim([90, max(150, y_max * 1.1)])  # Start at 90% to show context, extend to at least 150%
        
        # Format the plot
        format_plot_title(
            ax, 
            "LES PRIMES D'ASSURANCE HABITATION AUGMENTENT DANS LES ÉTATS À RISQUE D'INCENDIE",
            "AUGMENTATION EN POURCENTAGE DES PRIMES MOYENNES D'ASSURANCE HABITATION (DOLLARS 2020)",
            font_props
        )
        
        # Add legend
        ax.legend(
            fontsize=14, 
            frameon=True, 
            facecolor=COLORS['background'], 
            edgecolor='#DDDDDD', 
            loc='upper left', 
            prop=font_props.get('regular')
        )
        
        # Add branding
        add_deep_sky_branding(
            ax, 
            font_props, 
            data_note="DONNÉES : DÉPARTEMENTS D'ASSURANCE DE CALIFORNIE, TEXAS, WASHINGTON"
        )
        
        # Save the figure
        save_plot(fig, output_path)
        
        return fig

    # 2. French version of FAIR plan growth
    def plot_fair_plan_growth_fr(fair_plan_df, output_path=None):
        """
        Create a French visualization comparing Fair Plan growth across states
        """
        # Set up the plot
        fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))
        
        # Get unique states and assign colors
        states = fair_plan_df['state'].unique()
        state_colors = {
            'CA': COLORS['primary'],
            'TX': COLORS['tertiary'],
            'WA': COLORS['secondary'],
        }
        
        # Track handles for legend
        handles = []
        labels = []
        
        # Plot a line for each state
        for state in states:
            state_data = fair_plan_df[fair_plan_df['state'] == state]
            
            # Sort by year to ensure line is drawn correctly
            state_data = state_data.sort_values('year')

            fair_policies_baseline = state_data.loc[state_data['year'] == 2020, 'fair_policies_in_force'].iloc[0]
            state_data['fair_policies_growth'] = ((state_data['fair_policies_in_force'] - fair_policies_baseline) / fair_policies_baseline) * 100
            
            # Get state-specific color or use gray as fallback
            color = state_colors.get(state, COLORS['comparison'])
            
            # Plot the line - use premium_index (percentage) instead of raw average_premium
            line = ax.plot(state_data['year'], state_data['fair_policies_growth'], 
                          color=color, linewidth=2.5, marker='o', markersize=8, 
                          label=state)
            
            # Store handle for legend
            handles.append(line[0])
            labels.append(state)
            
            # Add label at the end of each line with percent increase
            last_point = state_data.iloc[-1]
            final_value = last_point['fair_policies_growth']
            percent_change = final_value
            
            # Add annotation for final value
            ax.annotate(f"{state}: +{percent_change:.0f}%", 
                      xy=(last_point['year'], final_value),
                      xytext=(10, 0), textcoords="offset points",
                      va='center', fontsize=12, fontproperties=font_props.get('regular'),
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none'))
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter('{x:.0f}%')
        
        # Add horizontal line at 0% (base year)
        ax.axhline(y=0, color='#CCCCCC', linestyle='--', alpha=0.7)

        unique_years = sorted(fair_plan_df['year'].unique())
        ax.set_xticks(unique_years)
        ax.set_xticklabels([str(int(year)) for year in unique_years], fontproperties=font_props.get('regular'))
        
        # Format the plot
        format_plot_title(
            ax, 
            "PROPRIÉTAIRES FORCÉS DE QUITTER L'ASSURANCE PRIVÉE",
            "CROISSANCE EN POURCENTAGE DU NOMBRE DE POLICES FAIR PLAN DEPUIS 2020",
            font_props
        )
        
        # Add legend
        ax.legend(
            handles=handles,
            labels=labels,
            fontsize=14, 
            frameon=True, 
            facecolor=COLORS['background'], 
            edgecolor='#DDDDDD', 
            loc='upper left', 
            prop=font_props.get('regular')
        )
        
        # Add branding
        add_deep_sky_branding(
            ax, 
            font_props, 
            data_note="DONNÉES: CALIFORNIA & TEXAS DEPARTMENTS OF INSURANCE, OREGON DEPARTMENT OF CONSUMER AND BUSINESS SERVICES"
        )
        plt.subplots_adjust(bottom=0.15, top=0.88, left=0.10, right=0.95)

        # Save the figure
        save_plot(fig, output_path)
        
        return fig

    # Generate French versions
    output_file_fr = f'{output_dir}/multi_state_premium_trends_fr.png'
    fig_fr = plot_state_premium_trends_fr(premium_trends, output_file_fr)
    print(f"French multi-state premium trend chart created at: {output_file_fr}")

    fair_fig_fr = plot_fair_plan_growth_fr(fair_plan_df, output_path=f'{output_dir}/fair_plan_growth_fr.png')
    print(f"French FAIR plan growth chart created at: {output_dir}/fair_plan_growth_fr.png")

    print("French language state comparison charts generated successfully!")

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import datetime
from scipy.interpolate import CubicSpline

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot

def load_gcb_global_historical_emissions(path, tab):
    """
    Load historical emissions data from Global Carbon Budget Excel file.

    Args:
        path: Path to the Excel file
        tab: Sheet index (2 for Historical Budget)

    Returns:
        DataFrame with year, fossil_emissions_excluding_carbonation, land_use_change_emissions
        and total_emissions_gt_co2 (in Gt CO2)
    """
    # Read the Historical Budget tab, skipping headers
    df = pd.read_excel(path, sheet_name=tab, skiprows=15)

    # Select relevant columns and rename for clarity
    df = df[['Year', 'fossil emissions excluding carbonation', 'land-use change emissions']].copy()
    df.rename(columns={
        'Year': 'year',
        'fossil emissions excluding carbonation': 'fossil_emissions_gt_c',
        'land-use change emissions': 'land_use_change_emissions_gt_c'
    }, inplace=True)

    # Convert from GtC to GtCO2 using 3.664 conversion factor (from cell B3 in the Excel)
    conversion_factor = 3.664
    df['fossil_emissions_gt_co2'] = df['fossil_emissions_gt_c'] * conversion_factor
    df['land_use_change_emissions_gt_co2'] = df['land_use_change_emissions_gt_c'] * conversion_factor

    # Calculate total annual emissions in Gt CO2 (treat NaN as 0)
    df['total_emissions_gt_co2'] = df['fossil_emissions_gt_co2'].fillna(0) + df['land_use_change_emissions_gt_co2'].fillna(0)

    # Filter to remove any invalid years
    df = df[df['year'].notna() & (df['year'] >= 1750)].copy()
    df['year'] = df['year'].astype(int)
    return df

def load_processed_pgr_emissions_data(path):
    df = pd.read_csv(path)

    # Convert to numeric, handling missing values
    numeric_cols = ['global_emissions', '2_degree_pathway', '1_5_degree_pathway',
                   'engineered_cdr_tonnes', 'nature_cdr_tonnes', 'global_emissions_counterfactual',
                   'global_emissions_gas', 'global_emissions_oil', 'global_emissions_coal']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def project_fred_cdr_capacity(historical_emissions_df, projections_df, cutoff_date=2100, ramp_down=True):
    """
    Fred believes we need to remove every tonne of CO2 that we have ever emitted to properly return the earth to its safe and natural balance.
    This means CDR needs to be scaled beyond just getting us to net-zero or following the IPCC's 2°C pathway. We need to continue past that point
    and remove all historical emissions. This function will take in historical emissions data from Global Carbon Budget and projections from
    the Production Gap Report and calculate how CDR needs to scale from 2025-2100 in order to get to net zero as quickly as possible and then continue
    to remove all historical emissions.
    For 1800 until 2023 it will rely on Global Carbon Budget data loaded by load_gcb_global_historical_emissions().
    For 2024-2050 it will rely on projected emissions from the Production Gap Report loaded by load_processed_pgr_emissions_data().
    For 2050 until 2100 we will assume that globally we can get emissions to 10 Gt/year by 2100.
    The cutoff_date argument will be used so that we can tweak how urgently we need to have removed all historical emissions.
    It will return a dataframe with colums: historical_emissions, projected_emissions, cdr which will show our vision of how this will play out.
    """

    # Calculate total historical emissions that need to be removed
    # Sum all emissions from 1800 to 2023 (or latest available year)
    historical_start_year = 1850
    historical_end_year = 2023

    # Filter historical data to the period we care about
    historical_filtered = historical_emissions_df[
        (historical_emissions_df['year'] >= historical_start_year) &
        (historical_emissions_df['year'] <= historical_end_year)
    ].copy()

    total_historical_emissions = historical_filtered['total_emissions_gt_co2'].sum()

    # Get future emissions from projections (2024-2050)
    future_emissions = projections_df[projections_df['year'] >= 2024].copy()

    # Extend beyond 2050 with exponential decay towards zero emissions asymptote
    years_beyond_2050 = list(range(2051, cutoff_date + 1))
    if len(years_beyond_2050) > 0:
        # Get 2050 emissions as starting point
        emissions_2050 = future_emissions[future_emissions['year'] == 2050]['global_emissions'].iloc[0] / 1e9 if not future_emissions[future_emissions['year'] == 2050].empty else 30

        # Use exponential decay: emissions(t) = start * exp(-decay_rate * (t-2050))
        # Choose decay rate so emissions approach ~1 Gt/year after many decades
        decay_rate = 0.025  # Adjust this to control how fast emissions decline

        for year in years_beyond_2050:
            years_since_2050 = year - 2050
            # Exponential decay towards zero (but never quite reaching it)
            year_emissions = emissions_2050 * np.exp(-decay_rate * years_since_2050)
            # Add small floor to prevent true zero
            year_emissions = max(year_emissions, 0.1)  # Minimum 0.1 Gt/year

            future_emissions = pd.concat([future_emissions, pd.DataFrame({
                'year': [year],
                'global_emissions': [year_emissions * 1e9]  # Convert back to tonnes
            })], ignore_index=True)

    # Calculate total future emissions that will be added
    total_future_emissions = (future_emissions['global_emissions'] / 1e9).sum()

    # Total CO2 that needs to be removed = historical + future emissions
    total_co2_to_remove = total_historical_emissions + total_future_emissions

    # Create the CDR scaleup trajectory using logistic function (true S-curve with two asymptotes)
    # Start with almost zero CDR in 2025 (0.001 Gt as specified)
    cdr_start_year = 2025
    cdr_start_capacity = 0.001  # Gt CO2/year

    # Create year range for CDR trajectory
    years = list(range(cdr_start_year, cutoff_date + 1))
    timeline_years = cutoff_date - cdr_start_year

    # Logistic function: y = L / (1 + e^(-k(x - x₀)))
    # where L = upper asymptote, k = steepness, x₀ = inflection point

    # Set inflection point (where growth rate is maximum)
    # Earlier inflection for longer timelines = more gradual ramp-up
    if timeline_years <= 75:  # 2100 or earlier
        inflection_year = cdr_start_year + timeline_years * 0.4  # 40% through timeline
        steepness = 0.15  # Steeper for shorter timeline
    elif timeline_years <= 100:  # 2125 or earlier
        inflection_year = cdr_start_year + timeline_years * 0.35  # 35% through timeline
        steepness = 0.10  # Medium steepness
    else:  # 2126 or later (like 2200)
        inflection_year = cdr_start_year + timeline_years * 0.30  # 30% through timeline
        steepness = 0.08  # Gentler for longer timeline

    # Use numerical integration to find L (upper asymptote) that gives correct total
    def logistic_cdr(year, L):
        """Logistic function for annual CDR rate"""
        return L / (1 + np.exp(-steepness * (year - inflection_year)))

    # Iteratively find the upper asymptote L that gives us the target total
    target_total = total_co2_to_remove
    L_estimate = target_total / (timeline_years * 0.6)  # Initial guess

    for _ in range(20):
        total_simulated = 0
        for year in years:
            annual_rate = logistic_cdr(year, L_estimate)
            # Ensure it starts near our minimum in 2025
            if year == cdr_start_year:
                annual_rate = max(annual_rate, cdr_start_capacity)
            total_simulated += annual_rate

        # Adjust estimate
        error_ratio = target_total / total_simulated
        if abs(error_ratio - 1.0) < 0.001:
            break
        L_estimate *= error_ratio

    # Define CDR curve based on ramp_down parameter
    if ramp_down:
        # Compound logistic curve: ramp up, then ramp down (bell curve)
        peak_year = cdr_start_year + timeline_years * 0.6  # Peak at 60% through timeline

        def cdr_curve(year, L):
            """Compound logistic function: ramp up then ramp down"""
            if year <= peak_year:
                # First logistic: ramp up to peak
                return L / (1 + np.exp(-steepness * (year - inflection_year)))
            else:
                # Second logistic: ramp down from peak (mirrored)
                mirror_year = 2 * peak_year - year  # Mirror point around peak
                return L / (1 + np.exp(-steepness * (mirror_year - inflection_year)))
    else:
        # Single logistic curve: smooth ramp up that naturally approaches asymptote
        def cdr_curve(year, L):
            """Single logistic function: smooth ramp up to asymptote"""
            return L / (1 + np.exp(-steepness * (year - inflection_year)))

    # Re-calculate L for the chosen curve
    for _ in range(20):
        total_simulated = 0
        for year in years:
            annual_rate = cdr_curve(year, L_estimate)
            # Ensure it starts very small in 2025
            if year == cdr_start_year:
                annual_rate = cdr_start_capacity
            total_simulated += annual_rate

        # Adjust estimate
        error_ratio = target_total / total_simulated
        if abs(error_ratio - 1.0) < 0.001:
            break
        L_estimate *= error_ratio

    # Generate the actual CDR trajectory using the chosen curve
    cdr_data = []
    cumulative_cdr = 0

    for year in years:
        # Calculate annual CDR using the chosen curve function
        annual_cdr = cdr_curve(year, L_estimate)

        # Ensure it starts very small in 2025
        if year == cdr_start_year:
            annual_cdr = cdr_start_capacity

        # Ensure we don't remove more CO2 than exists
        if cumulative_cdr + annual_cdr > total_co2_to_remove:
            annual_cdr = max(0, total_co2_to_remove - cumulative_cdr)

        cumulative_cdr += annual_cdr

        cdr_data.append({
            'year': year,
            'annual_cdr_gt_co2': annual_cdr,
            'cumulative_cdr_gt_co2': cumulative_cdr
        })

    # Create comprehensive results dataframe
    # Start with historical emissions (1800-2023)
    result_df = historical_emissions_df[['year', 'total_emissions_gt_co2']].copy()
    result_df['emissions_type'] = 'historical'
    result_df['annual_cdr_gt_co2'] = 0  # No CDR in historical period
    result_df['cumulative_cdr_gt_co2'] = 0

    # Add future emissions (2024-2100)
    for _, row in future_emissions.iterrows():
        year = row['year']
        emissions = row['global_emissions'] / 1e9  # Convert to Gt

        # Get CDR data for this year if it exists
        cdr_row = next((x for x in cdr_data if x['year'] == year), None)
        annual_cdr = cdr_row['annual_cdr_gt_co2'] if cdr_row else 0
        cumulative_cdr = cdr_row['cumulative_cdr_gt_co2'] if cdr_row else 0

        result_df = pd.concat([result_df, pd.DataFrame({
            'year': [year],
            'total_emissions_gt_co2': [emissions],
            'emissions_type': ['projected' if year <= 2050 else 'assumption'],
            'annual_cdr_gt_co2': [annual_cdr],
            'cumulative_cdr_gt_co2': [cumulative_cdr]
        })], ignore_index=True)

    # Calculate net annual balance (emissions - CDR)
    result_df['net_annual_balance'] = result_df['total_emissions_gt_co2'] - result_df['annual_cdr_gt_co2']

    # Calculate cumulative net balance
    result_df['cumulative_net_balance'] = result_df['net_annual_balance'].cumsum()

    return result_df

def build_remove_all_emissions_plot(df):
    """
    This function will plot total global emissions and total CDR from 1850 until the end of the data. It will take in data provided by
    project_fred_cdr_capacity() and show historical emissions vs. CDR reversing those historical emissions.
    """

    fig, ax, font_props = setup_enhanced_plot(figsize=(16, 12))

    # Filter data for better visualization (show from 1850 onwards for clarity)
    df_plot = df[df['year'] >= 1850].copy()
    years = df_plot['year']
    emissions = df_plot['total_emissions_gt_co2']
    annual_cdr = df_plot['annual_cdr_gt_co2']
    cumulative_cdr = df_plot['cumulative_cdr_gt_co2']

    # Create color coding based on different periods
    historical_mask = df_plot['emissions_type'] == 'historical'
    projected_mask = df_plot['emissions_type'] == 'projected'
    assumption_mask = df_plot['emissions_type'] == 'assumption'

    # Plot emissions as positive bars
    bar_width = 1.0

    # Historical emissions (1850-2023)
    if historical_mask.any():
        ax.bar(years[historical_mask], emissions[historical_mask],
               color='#C0392B', alpha=0.8, width=bar_width)

    # Projected emissions (2024-2050)
    if projected_mask.any():
        ax.bar(years[projected_mask], emissions[projected_mask],
               color='#E74C3C', alpha=0.7, width=bar_width)

    # Assumption emissions (2051-2100)
    if assumption_mask.any():
        ax.bar(years[assumption_mask], emissions[assumption_mask],
               color='#E74C3C', alpha=0.7, width=bar_width)

    # Plot CDR as negative bars (removals)
    cdr_mask = annual_cdr > 0
    if cdr_mask.any():
        ax.bar(years[cdr_mask], -annual_cdr[cdr_mask],
               color='#27AE60', alpha=0.8, width=bar_width,
               label='Annual CDR')

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Add vertical lines to mark key periods
    ax.axvline(x=2025, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    # ax.axvline(x=2050, color='gray', linestyle='--', alpha=0.7, linewidth=2)

    # Calculate some key statistics for annotations
    total_historical = df[df['emissions_type'] == 'historical']['total_emissions_gt_co2'].sum()
    total_future = df[df['emissions_type'].isin(['projected', 'assumption'])]['total_emissions_gt_co2'].sum()
    total_all = total_historical + total_future
    max_annual_cdr = annual_cdr.max()

    # Formatting - determine range from data
    data_start_year = df_plot['year'].min()
    data_end_year = df_plot['year'].max()
    ax.set_xlim(data_start_year - 5, data_end_year + 5)
    ax.set_ylim(-max_annual_cdr * 1.2, emissions.max() * 1.1)

    # Axis labels
    ax.set_xlabel('YEAR', fontsize=14, fontproperties=font_props.get('regular') if font_props else None)
    ax.set_ylabel('', fontsize=14, fontproperties=font_props.get('regular') if font_props else None)

    # Tick formatting - adjust interval based on data range
    timeline_span = data_end_year - data_start_year
    tick_interval = 25 if timeline_span <= 300 else 50  # Wider intervals for longer timelines
    ax.set_xticks(range(int(data_start_year), int(data_end_year) + 1, tick_interval))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Annotations - make positions relative to data timeline
    ax.text(1925, emissions.max() * 0.4, 'HISTORICAL\nEMISSIONS',
            fontsize=14, ha='center', va='center',
            fontweight='bold', alpha=0.9, color='#C0392B',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#C0392B', alpha=0.9))

    # ax.text(data_start_year + timeline_span * 0.5, emissions.max() * 0.6, 'PROJECTED\nEMISSIONS',
    #         fontsize=14, ha='center', va='center',
    #         fontweight='bold', alpha=0.9, color='#E74C3C',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
    #                  edgecolor='#E74C3C', alpha=0.9))

    ax.text(data_start_year + timeline_span * 0.75, emissions.max() * 0.4, 'PROJECTED\nEMISSIONS',
            fontsize=14, ha='center', va='center',
            fontweight='bold', color='#E74C3C',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#E74C3C', alpha=0.9))

    ax.text(data_start_year + timeline_span * 0.8, -max_annual_cdr * 0.7, f'CDR REMOVES\nALL {total_all:.0f} Gt CO₂',
            fontsize=14, ha='center', va='center',
            fontweight='bold', alpha=0.9, color='#27AE60',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#27AE60', alpha=0.9))

    # Add period labels
    ax.text(2025, emissions.max() * 1.1, '2025',
            fontsize=14, ha='center', va='bottom',
            fontweight='bold', color='gray')

    return fig

def main():
    processed_pgr_csv = 'data/needed_removal_capacity/cdr_scaleup_output.csv'
    gcb_excel_file = 'data/Global_Carbon_Budget_2024_v1.0-1.xlsx'

    historical_emissions_df = load_gcb_global_historical_emissions(gcb_excel_file, tab=2)
    projections_df = load_processed_pgr_emissions_data(processed_pgr_csv)

    # Choose CDR curve type: True for bell curve (ramp down), False for plateau
    ramp_down = False  # Change this to switch between curve types

    fred_vision_df = project_fred_cdr_capacity(historical_emissions_df, projections_df, cutoff_date=2200, ramp_down=ramp_down)

    fred_vision_plot = build_remove_all_emissions_plot(fred_vision_df)

    # Add titles and branding for v2
    format_plot_title(plt.gca(),
                        "",
                        "GLOBAL CO\N{SUBSCRIPT TWO} EMISSION & REMOVAL (GIGATONNES)",
                        None)

    add_deep_sky_branding(plt.gca(), None,
                            "DATA: GLOBAL CARBON PROJECT (2024); SEI, CLIMATE ANALYTICS, IISD (2025) THE PRODUCTION GAP REPORT; CDR.FYI",
                            analysis_date=datetime.datetime.now())

    save_path = 'figures/total_removal_scenario.png'
    os.makedirs('figures', exist_ok=True)
    save_plot(fred_vision_plot, save_path)

    print(f"Net emissions plot saved to {save_path}")

if __name__ == "__main__":
    main()

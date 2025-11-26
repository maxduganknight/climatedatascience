"""
CDR Scale-Up Analysis V2 - Direct from Source Data

This script builds the CDR scale-up analysis directly from source data files:
- Global Carbon Budget 2024 (historical emissions)
- Production Gap Report 2025 (fossil fuel projections)
- CDR.fyi (historical CDR deliveries)

It applies documented assumptions and generates all intermediate data and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import datetime
from scipy.interpolate import CubicSpline

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot

# =======================
# HARD-CODED ASSUMPTIONS
# =======================

# Emission factors from PGR 2025 Methodology (GtCO2/EJ)
EMISSION_FACTORS = {
    'coal': 0.098,   # GtCO2eq per EJ
    'oil': 0.069,    # GtCO2eq per EJ
    'gas': 0.058     # GtCO2eq per EJ
}

# Convert GtC to GtCO2: multiply by 3.664 (molecular weight ratio)
GTC_TO_GTCO2 = 3.664

# 2°C pathway will be loaded from PGR Excel file
# (median of "Pathways consistent with 2C" from Figure ES-1 and 2.1, row 10)

# CDR transition parameters (from Oxford Principles)
CDR_START_YEAR = 2025
CDR_END_YEAR = 2050
INITIAL_NATURE_RATIO = 0.95  # 95% nature-based in 2025
FINAL_ENGINEERED_RATIO = 1.00  # 100% engineered by 2050

# Exponential growth parameter for CDR scale-up
CDR_GROWTH_K = 9  # Higher = faster flattening

# Lower bound

# Prof Dan Macelroy writing for Neg8 assumed that hard-to-abate emissions are 20%
# of total emissions and that 50% of that will need to be removed with DAC getting him to 4.16 GT 
# https://neg8carbon.com/costing-direct-air-capture-deployment-towards-zero-and-negative-carbon-emissions/
# HARD_TO_ABATE_TONNES = 4.16e9  # tonnes CO2 per year from 2050-2100

# The CDR primer estimates that the range of hard-to-abate ("hard-to-avoid") emissions is 1.5-3.1 Gt. 
HARD_TO_ABATE_TONNES = (1.5e9 + 3.1e9)/2
LEGACY_TOTAL_TONNES = 65e9  # total tonnes CO2 to remove by 2100


def load_historical_emissions():
    """
    Load historical emissions from Global Carbon Budget 2024.
    Returns DataFrame with year and global_emissions (in tonnes CO2).
    """
    print('Loading historical emissions from Global Carbon Budget 2024...')

    gcb_file = 'data/Global_Carbon_Budget_2024_v1.0-1.xlsx'

    # Read Historical Budget sheet (header is at row 14, but becomes data, so we skip 15 rows)
    df = pd.read_excel(gcb_file, sheet_name='Historical Budget', skiprows=15)

    # First column is year, second is fossil emissions excluding carbonation (GtC/yr)
    df_clean = pd.DataFrame({
        'year': pd.to_numeric(df.iloc[:, 0], errors='coerce'),
        'fossil_emissions_gtc': pd.to_numeric(df.iloc[:, 1], errors='coerce')
    })

    # Remove rows with NaN
    df_clean = df_clean.dropna()

    # Convert GtC to GtCO2 (tonnes)
    df_clean['global_emissions'] = df_clean['fossil_emissions_gtc'] * GTC_TO_GTCO2 * 1e9

    # Filter to 2000-2023
    df_clean = df_clean[(df_clean['year'] >= 2000) & (df_clean['year'] <= 2023)]

    print(f'  Loaded {len(df_clean)} years of historical emissions (2000-2023)')
    print(f'  2023 emissions: {df_clean[df_clean["year"] == 2023]["global_emissions"].values[0]/1e9:.2f} GtCO2')

    return df_clean[['year', 'global_emissions']]


def load_2c_pathway_from_pgr():
    """
    Load 2°C pathway from Production Gap Report 2025.
    Returns dictionary of year -> emissions (tonnes CO2).
    """
    print('\nLoading 2°C pathway from PGR 2025...')

    pgr_file = 'data/needed_removal_capacity/PGR2025_data.xlsx'

    # Read Figure ES-1 and 2.1 sheet
    df = pd.read_excel(pgr_file, sheet_name='Figure ES-1 and 2.1', header=None)

    # Row 4 (index 4) has years in columns 3-8
    # Row 10 (index 10) has 2C median pathway values
    years_row = df.iloc[4, 3:9]
    pathway_row = df.iloc[10, 3:9]

    # Create dictionary
    pathway_dict = {}
    for i, year in enumerate(years_row.values):
        if not pd.isna(year) and not pd.isna(pathway_row.values[i]):
            pathway_dict[int(year)] = float(pathway_row.values[i]) * 1e9  # Convert GtCO2 to tonnes

    print(f'  Loaded {len(pathway_dict)} pathway points: {sorted(pathway_dict.keys())}')
    print(f'  2050 target: {pathway_dict[2050]/1e9:.2f} GtCO2')

    return pathway_dict


def load_pgr_total_emissions():
    """
    Load total projected emissions from Production Gap Report 2025.
    This is used for quality checking against our calculated totals.
    Returns dictionary of year -> total emissions (tonnes CO2).
    """
    print('\nLoading PGR total projected emissions for quality check...')

    pgr_file = 'data/needed_removal_capacity/PGR2025_data.xlsx'

    # Read Figure ES-1 and 2.1 sheet
    df = pd.read_excel(pgr_file, sheet_name='Figure ES-1 and 2.1', header=None)

    # Row 4 (index 4) has years in columns 2-8 (2023-2050)
    # Row 5 (index 5) has total GPP (Government Plans and Projections) from PGR2025
    # Note: Row 6 is GPP from PGR2023 (older data), so we use row 5
    years_row = df.iloc[4, 2:9]
    total_emissions_row = df.iloc[5, 2:9]

    # Create dictionary
    total_emissions_dict = {}
    for i, year in enumerate(years_row.values):
        if not pd.isna(year) and not pd.isna(total_emissions_row.values[i]):
            total_emissions_dict[int(year)] = float(total_emissions_row.values[i]) * 1e9  # Convert GtCO2 to tonnes

    print(f'  Loaded {len(total_emissions_dict)} total emission points: {sorted(total_emissions_dict.keys())}')
    if total_emissions_dict:
        first_year = sorted(total_emissions_dict.keys())[0]
        print(f'  {first_year} total emissions: {total_emissions_dict[first_year]/1e9:.2f} GtCO2')

    return total_emissions_dict


def load_pgr_fossil_fuel_data():
    """
    Load fossil fuel production projections from Production Gap Report 2025.
    Reads data directly from Excel file.
    Returns DataFrame with year and production in EJ for coal, oil, gas.
    """
    print('\nLoading fossil fuel projections from PGR 2025...')

    pgr_file = 'data/needed_removal_capacity/PGR2025_data.xlsx'

    # Read Figure ES-2 and 2.2 sheet which has fossil fuel production data
    df = pd.read_excel(pgr_file, sheet_name='Figure ES-2 and 2.2', header=None)

    # Extract years from row 5 (index 4), columns D-J (index 3-9)
    # Note: Row numbers in Excel are 1-indexed, but pandas uses 0-indexing
    # So Excel row 5 = pandas index 4
    years_row_idx = 4
    years = df.iloc[years_row_idx, 3:10].tolist()  # columns D-J (index 3-9)

    # Extract fossil fuel data from "Government plans and projections (GPP) from PGR2025"
    # Coal: Excel row 6 = pandas index 5
    coal_row_idx = 5
    coal_values = df.iloc[coal_row_idx, 3:10].tolist()

    # Oil: Excel row 21 = pandas index 20
    oil_row_idx = 20
    oil_values = df.iloc[oil_row_idx, 3:10].tolist()

    # Gas: Excel row 36 = pandas index 35
    gas_row_idx = 35
    gas_values = df.iloc[gas_row_idx, 3:10].tolist()

    # Create DataFrame
    fossil_fuel_ej = pd.DataFrame({
        'year': [int(y) for y in years],
        'coal_ej': coal_values,
        'oil_ej': oil_values,
        'gas_ej': gas_values
    })

    # Convert from EJ to GtCO2 using emission factors
    fossil_fuel_ej['coal_gtco2'] = fossil_fuel_ej['coal_ej'] * EMISSION_FACTORS['coal'] * 1e9
    fossil_fuel_ej['oil_gtco2'] = fossil_fuel_ej['oil_ej'] * EMISSION_FACTORS['oil'] * 1e9
    fossil_fuel_ej['gas_gtco2'] = fossil_fuel_ej['gas_ej'] * EMISSION_FACTORS['gas'] * 1e9

    print(f'  Loaded {len(fossil_fuel_ej)} projection years: {fossil_fuel_ej["year"].tolist()}')
    print(f'  2023 coal: {fossil_fuel_ej[fossil_fuel_ej["year"] == 2023]["coal_ej"].values[0]:.2f} EJ')
    print(f'  2023 oil: {fossil_fuel_ej[fossil_fuel_ej["year"] == 2023]["oil_ej"].values[0]:.2f} EJ')
    print(f'  2023 gas: {fossil_fuel_ej[fossil_fuel_ej["year"] == 2023]["gas_ej"].values[0]:.2f} EJ')

    return fossil_fuel_ej


def load_historical_gas_emissions():
    """
    Load historical gas emissions from Global Carbon Budget 2024.
    Returns DataFrame with year and historical_gas_emissions (in tonnes CO2).
    """
    print('\nLoading historical gas emissions from Global Carbon Budget 2024...')

    gcb_file = 'data/Global_Carbon_Budget_2024_v1.0-1.xlsx'

    # Read Fossil Emissions by Category sheet (header is at row 8)
    df = pd.read_excel(gcb_file, sheet_name='Fossil Emissions by Category', skiprows=8)
    df.columns = ['Year', 'Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other', 'PerCapita']

    # Filter to 2000-2023 and select year and gas
    df_clean = df[(df['Year'] >= 2000) & (df['Year'] <= 2023)][['Year', 'Gas']].copy()
    df_clean = df_clean.rename(columns={'Year': 'year', 'Gas': 'gas_mtc'})

    # Convert MtC to tonnes CO2 (MtC * 3.664 * 1e6)
    df_clean['historical_gas_emissions'] = df_clean['gas_mtc'] * GTC_TO_GTCO2 * 1e6

    print(f'  Loaded {len(df_clean)} years of historical gas emissions (2000-2023)')
    print(f'  2023 gas emissions: {df_clean[df_clean["year"] == 2023]["historical_gas_emissions"].values[0]/1e9:.2f} GtCO2')

    return df_clean[['year', 'historical_gas_emissions']]


def load_historical_cdr():
    """
    Load historical CDR deliveries from CDR.fyi data.
    Returns DataFrame with year, engineered_cdr_tonnes, nature_cdr_tonnes.
    """
    print('\nLoading historical CDR deliveries...')

    cdr_file = 'data/needed_removal_capacity/tons_delivered_by_type.csv'
    df = pd.read_csv(cdr_file)

    # Pivot to get engineered and nature-based in separate columns
    df_pivot = df.pivot(index='year', columns='type', values='tons_delivered').reset_index()
    df_pivot.columns.name = None
    df_pivot = df_pivot.rename(columns={
        'Engineered': 'engineered_cdr_tonnes',
        'Nature-based': 'nature_cdr_tonnes'
    })

    print(f'  Loaded {len(df_pivot)} years of CDR data (2022-2025)')

    return df_pivot


def interpolate_with_cubic_spline(df, column_name, year_column='year'):
    """
    Interpolate missing values in a dataframe column using cubic spline.

    Args:
        df: DataFrame containing the data
        column_name: Name of the column to interpolate
        year_column: Name of the year column (default: 'year')

    Returns:
        DataFrame with interpolated values
    """
    # Get non-NaN values for spline fitting
    mask = df[column_name].notna()
    if mask.sum() > 1:  # Need at least 2 points for interpolation
        known_years = df.loc[mask, year_column].values
        known_values = df.loc[mask, column_name].values

        # Create cubic spline with natural boundary conditions
        cs = CubicSpline(known_years, known_values, bc_type='natural')

        # Apply to all years within the range
        interpolate_mask = (df[year_column] >= known_years.min()) & (df[year_column] <= known_years.max())
        df.loc[interpolate_mask, column_name] = cs(df.loc[interpolate_mask, year_column].values)

    return df


def create_2c_pathway_series(years, pathway_dict):
    """
    Create interpolated 2°C pathway for all years.
    Uses cubic spline interpolation between key points.

    Args:
        years: Array of years to interpolate for
        pathway_dict: Dictionary of year -> emissions (tonnes CO2)
    """
    pathway_years = sorted(pathway_dict.keys())
    pathway_values = [pathway_dict[y] for y in pathway_years]

    # Create cubic spline
    cs = CubicSpline(pathway_years, pathway_values, bc_type='natural')

    # Interpolate for years within pathway range
    result = []
    for year in years:
        if pathway_years[0] <= year <= pathway_years[-1]:
            # Convert to scalar if it's an array
            val = cs(year)
            result.append(float(val) if isinstance(val, np.ndarray) else val)
        else:
            result.append(np.nan)

    return result  # Return list directly instead of Series


def join_historical_to_projections(historical_emissions, fossil_fuel_projections):
    """
    Join historical emissions (2000-2023) to fossil fuel projections (2023-2050)
    using linear interpolation for 2020-2023 to smooth the transition.
    """
    print('\nJoining historical and projected emissions...')

    # Get 2023 value from historical
    hist_2023 = historical_emissions[historical_emissions['year'] == 2023]['global_emissions'].values[0]

    # Get 2023 value from projections (sum of fossil fuels)
    proj_2023_row = fossil_fuel_projections[fossil_fuel_projections['year'] == 2023]
    proj_2023 = (proj_2023_row['coal_gtco2'].values[0] +
                 proj_2023_row['oil_gtco2'].values[0] +
                 proj_2023_row['gas_gtco2'].values[0])

    print(f'  Historical 2023: {hist_2023/1e9:.2f} GtCO2')
    print(f'  Projected 2023: {proj_2023/1e9:.2f} GtCO2')
    print(f'  Discrepancy: {(proj_2023 - hist_2023)/1e9:.2f} GtCO2')

    # Linear interpolation from 2020 historical to 2023 projected
    hist_2020 = historical_emissions[historical_emissions['year'] == 2020]['global_emissions'].values[0]

    # Create adjusted historical emissions for 2020-2023
    years_2020_2023 = np.array([2020, 2021, 2022, 2023])
    emissions_2020_2023 = np.interp(years_2020_2023, [2020, 2023], [hist_2020, proj_2023])

    # Replace 2020-2023 in historical with interpolated values
    historical_adjusted = historical_emissions.copy()
    for i, year in enumerate(years_2020_2023):
        historical_adjusted.loc[historical_adjusted['year'] == year, 'global_emissions'] = emissions_2020_2023[i]

    return historical_adjusted


def build_complete_dataset():
    """
    Build the complete dataset combining all sources.
    """
    # Load all source data
    historical_emissions = load_historical_emissions()
    pathway_2c_dict = load_2c_pathway_from_pgr()
    fossil_fuel_proj = load_pgr_fossil_fuel_data()
    historical_gas = load_historical_gas_emissions()
    historical_cdr = load_historical_cdr()

    # Join historical to projections
    historical_adjusted = join_historical_to_projections(historical_emissions, fossil_fuel_proj)

    # Create complete year range (2000-2050)
    all_years = list(range(2000, 2051))
    df = pd.DataFrame({'year': all_years})

    # Add historical emissions (2000-2023)
    df = df.merge(historical_adjusted, on='year', how='left')

    # Add historical gas emissions (2000-2023) from GCB
    df = df.merge(historical_gas, on='year', how='left')

    # Add fossil fuel breakdown for projection years
    df = df.merge(
        fossil_fuel_proj[['year', 'coal_gtco2', 'oil_gtco2', 'gas_gtco2']].rename(columns={
            'coal_gtco2': 'global_emissions_coal',
            'oil_gtco2': 'global_emissions_oil',
            'gas_gtco2': 'global_emissions_gas'
        }),
        on='year',
        how='left'
    )

    # For gas emissions: use historical GCB data (2000-2023), then PGR projections (2024-2050)
    # Blend at 2024 to smooth the transition
    for idx, row in df.iterrows():
        year = row['year']
        if year <= 2020 and not pd.isna(row['historical_gas_emissions']):
            # Use GCB historical data
            df.loc[idx, 'global_emissions_gas'] = row['historical_gas_emissions']
        elif year == 2023:
            # Blend 2023 GCB with 2023 PGR
            gcb_2023 = df[df['year'] == 2023]['historical_gas_emissions'].values[0]
            pgr_2023 = df[df['year'] == 2023]['global_emissions_gas'].values[0]
            df.loc[idx, 'global_emissions_gas'] = (gcb_2023 + pgr_2023) / 2
        elif year >= 2030 and not pd.isna(row['global_emissions_gas']):
            # Use PGR projected data
            df.loc[idx, 'global_emissions_gas'] = row['global_emissions_gas']

    # Interpolate fossil fuel breakdown for missing years using cubic spline
    for col in ['global_emissions_coal', 'global_emissions_oil', 'global_emissions_gas']:
        if col in df.columns:
            # Get non-NaN values for spline fitting
            mask = df[col].notna()
            if mask.sum() > 1:  # Need at least 2 points for interpolation
                known_years = df.loc[mask, 'year'].values
                known_values = df.loc[mask, col].values

                # Create cubic spline with natural boundary conditions
                cs = CubicSpline(known_years, known_values, bc_type='natural')

                # Apply to all years within the range
                interpolate_mask = (df['year'] >= known_years.min()) & (df['year'] <= known_years.max())
                df.loc[interpolate_mask, col] = cs(df.loc[interpolate_mask, 'year'].values)

    # For projection years (2024-2050), calculate total emissions from fossil fuel sum
    # Include 2024 since we have interpolated fossil fuel data for it
    proj_mask = df['year'] >= 2024
    df.loc[proj_mask, 'global_emissions'] = (
        df.loc[proj_mask, 'global_emissions_coal'].fillna(0) +
        df.loc[proj_mask, 'global_emissions_oil'].fillna(0) +
        df.loc[proj_mask, 'global_emissions_gas'].fillna(0)
    )

    # Add 2°C pathway
    df['2_degree_pathway'] = create_2c_pathway_series(df['year'].values, pathway_2c_dict)

    # Add historical CDR
    df = df.merge(historical_cdr, on='year', how='left')

    return df


def pgr_data_loading_quality_check(df_complete):
    """
    Quality check: Compare our calculated total emissions against PGR's reported totals.

    Asserts that our sum of fossil fuel emissions matches PGR's total emissions
    for years 2023, 2025, 2030, 2035, 2040, 2045, 2050.

    Args:
        df_complete: DataFrame with calculated emissions

    Raises:
        AssertionError: If emissions don't match within tolerance
    """
    print('\n' + '='*60)
    print('DATA QUALITY CHECK: PGR Total Emissions')
    print('='*60)

    # Load PGR's reported total emissions
    pgr_totals = load_pgr_total_emissions()

    # Check years present in both datasets
    check_years = [2023, 2025, 2030, 2035, 2040, 2045, 2050]
    available_years = [y for y in check_years if y in pgr_totals]

    print(f'\nComparing calculated vs. PGR reported totals for years: {available_years}')
    print('-'*60)

    tolerance_pct = 1.0  # 1% tolerance for rounding/calculation differences
    all_passed = True

    for year in available_years:
        # Get calculated total from our dataframe
        calculated = df_complete[df_complete['year'] == year]['global_emissions'].values[0]

        # Get PGR reported total
        pgr_reported = pgr_totals[year]

        # Calculate difference
        diff_gt = (calculated - pgr_reported) / 1e9
        diff_pct = (calculated - pgr_reported) / pgr_reported * 100

        # Check if within tolerance
        passed = abs(diff_pct) <= tolerance_pct
        all_passed = all_passed and passed

        status = '✓' if passed else '✗ FAILED'
        print(f'{year}: Calculated={calculated/1e9:.2f} Gt, PGR={pgr_reported/1e9:.2f} Gt, '
              f'Diff={diff_gt:+.2f} Gt ({diff_pct:+.2f}%) {status}')

    print('-'*60)

    if all_passed:
        print('✓ All checks passed! Calculated emissions match PGR totals.')
    else:
        error_msg = (f'\n✗ Quality check FAILED: Calculated emissions differ from PGR reported '
                    f'totals by more than {tolerance_pct}% for one or more years.\n'
                    f'This suggests an error in fossil fuel emission calculations.')
        raise AssertionError(error_msg)

    print('='*60)


def project_cdr_scale_up(df):
    """
    Project CDR capacity scale-up from 2025 to 2050.
    Uses exponential growth curve and transition from nature-based to engineered.
    """
    print('\nProjecting CDR scale-up...')

    # Calculate target CDR for 2050
    emissions_2050 = df[df['year'] == 2050]['global_emissions'].values[0]
    pathway_2050 = df[df['year'] == 2050]['2_degree_pathway'].values[0]
    target_cdr_2050 = emissions_2050 - pathway_2050

    print(f'  2050 emissions: {emissions_2050/1e9:.1f} GtCO2')
    pathway_2050_str = f'{pathway_2050/1e9:.1f}' if not pd.isna(pathway_2050) else 'NaN'
    print(f'  2050 pathway target: {pathway_2050_str} GtCO2')
    target_2050_str = f'{target_cdr_2050/1e9:.1f}' if not pd.isna(target_cdr_2050) else 'NaN'
    print(f'  Required CDR by 2050: {target_2050_str} GtCO2/year')

    # Get starting CDR from 2025 historical
    start_eng = df[df['year'] == 2025]['engineered_cdr_tonnes'].values[0] if not pd.isna(df[df['year'] == 2025]['engineered_cdr_tonnes'].values[0]) else 1e6
    start_nat = df[df['year'] == 2025]['nature_cdr_tonnes'].values[0] if not pd.isna(df[df['year'] == 2025]['nature_cdr_tonnes'].values[0]) else 1e6

    # Project for each year 2026-2050
    for year in range(2026, 2051):
        progress = (year - CDR_START_YEAR) / (CDR_END_YEAR - CDR_START_YEAR)

        # Exponential growth factor (fast initially, slowing down)
        growth_factor = 1 - np.exp(-CDR_GROWTH_K * progress)

        # Total CDR capacity for this year
        total_cdr = start_eng + start_nat + (target_cdr_2050 - (start_eng + start_nat)) * growth_factor

        # Transition ratio (sigmoid curve)
        ratio_transition = 1 / (1 + np.exp(-6 * (progress - 0.5)))
        eng_ratio = (1 - INITIAL_NATURE_RATIO) + (FINAL_ENGINEERED_RATIO - (1 - INITIAL_NATURE_RATIO)) * ratio_transition
        nat_ratio = 1 - eng_ratio

        # Assign to dataframe
        df.loc[df['year'] == year, 'engineered_cdr_tonnes'] = total_cdr * eng_ratio
        df.loc[df['year'] == year, 'nature_cdr_tonnes'] = total_cdr * nat_ratio

    # Fill any remaining NaNs in CDR columns with 0 (for years before 2022)
    df['engineered_cdr_tonnes'] = df['engineered_cdr_tonnes'].fillna(0)
    df['nature_cdr_tonnes'] = df['nature_cdr_tonnes'].fillna(0)

    total_cdr_2050 = df[df['year'] == 2050]['engineered_cdr_tonnes'].values[0] + df[df['year'] == 2050]['nature_cdr_tonnes'].values[0]
    print(f'  Projected total CDR in 2050: {total_cdr_2050/1e9:.1f} GtCO2/year')

    return df


def calculate_total_cdr_needed(df, emissions_column='global_emissions'):
    """
    Calculate total CDR needed by summing (emissions - 2_degree_pathway) for all years.

    Args:
        df: DataFrame with emissions and pathway data
        emissions_column: Which emissions column to use

    Returns:
        float: Total CDR needed in tonnes
    """
    # Filter to future years only (2025-2050)
    future_data = df[df['year'] >= 2025].copy()

    # Calculate annual CDR needs
    annual_cdr_needed = future_data[emissions_column] - future_data['2_degree_pathway']
    annual_cdr_needed = annual_cdr_needed.fillna(0).clip(lower=0)

    # Sum total over all years
    total_cdr_needed = annual_cdr_needed.sum()

    return total_cdr_needed


def create_net_emissions_chart(df_complete, emissions_pathway='actual', ax=None, total_cdr_needed_gt=None):
    """
    Create stacked bar chart showing net global emissions over time.

    Args:
        df_complete: DataFrame with emissions and CDR data
        emissions_pathway: 'actual' to use global_emissions, 'counterfactual' to use global_emissions_counterfactual
        ax: Optional matplotlib axis to plot on. If None, creates new figure.
        total_cdr_needed_gt: Total CDR needed in Gt for annotation. If None, calculates automatically.
    """
    if ax is None:
        fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))
    else:
        fig = ax.figure
        font_props = None

    # Select emissions column based on pathway
    emissions_col = 'global_emissions' if emissions_pathway == 'actual' else 'global_emissions_counterfactual'

    # Convert to Gt for better readability
    emissions_gt = df_complete[emissions_col] / 1e9
    eng_cdr_gt = df_complete['engineered_cdr_tonnes'] / 1e9
    nat_cdr_gt = df_complete['nature_cdr_tonnes'] / 1e9
    pathway_2c_gt = df_complete['2_degree_pathway'] / 1e9

    # Calculate total CDR needed if not provided
    if total_cdr_needed_gt is None:
        total_cdr_needed_tonnes = calculate_total_cdr_needed(df_complete, emissions_col)
        total_cdr_needed_gt = total_cdr_needed_tonnes / 1e9

    # Chart values for relative annotation position
    total_cdr_rounded = round(total_cdr_needed_gt, -2)
    cdr_annual = total_cdr_needed_gt/24
    max_emissions = max(emissions_gt)

    # Calculate net emissions
    net_emissions = emissions_gt - eng_cdr_gt - nat_cdr_gt
    # Filter to years with data
    years = df_complete['year']
    mask = pd.notna(emissions_gt)

    years_plot = years[mask]
    emissions_plot = emissions_gt[mask]
    eng_cdr_plot = eng_cdr_gt[mask]
    nat_cdr_plot = nat_cdr_gt[mask]
    net_plot = net_emissions[mask]
    pathway_2c_plot = pathway_2c_gt[mask]

    # Split data into past and projected
    past_mask = years_plot < 2025
    projected_mask = years_plot >= 2025
    pathway_2c_mask = years_plot >=2020

    # Create stacked bars - emissions above zero, removals below zero
    bar_width = 0.8

    # Past emissions (solid color)
    if past_mask.any():
        bars_past_emissions = ax.bar(years_plot[past_mask], emissions_plot[past_mask],
                                   color='#C0392B', alpha=0.9, width=bar_width,
                                   label='Past Emissions')

    # Projected emissions (different color)
    if projected_mask.any():
        bars_projected_emissions = ax.bar(years_plot[projected_mask], emissions_plot[projected_mask],
                                        color='#E74C3C', alpha=0.7, width=bar_width,
                                        label='Projected Emissions')

    # CDR removals (negative, below zero)
    bars_eng_cdr = ax.bar(years_plot, -eng_cdr_plot,
                         color='#3498DB', alpha=0.8, width=bar_width)

    bars_nat_cdr = ax.bar(years_plot, -nat_cdr_plot,
                         bottom=-eng_cdr_plot,
                         color='#27AE60', alpha=0.8, width=bar_width)

    # 2°C pathway line (independent of CDR calculations)
    ax.plot(years_plot[pathway_2c_mask], pathway_2c_plot[pathway_2c_mask], color='#2C3E50', linewidth=3,
            marker='o', markersize=4, label='2°C Pathway')

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xlim(2014, 2051)
    ax.set_ylim(-27,49)

    # Y-axis formatting
    ax.set_ylabel('', fontsize=14,
                  fontproperties=font_props.get('regular') if font_props else None)

    # X-axis formatting
    ax.set_xticks(range(2020, 2051, 10))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add annotations
    ax.text(2019, max_emissions, 'PAST\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#C0392B')

    ax.text(2040, max_emissions, 'PROJECTED\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#E74C3C')

    ax.text(2033, -cdr_annual+3, 'LOW DURABILITY CDR',
        fontsize=12, ha='center', va='center',
        fontweight='bold', alpha=0.9, color='#27AE60',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
            edgecolor='#27AE60', alpha=0.9))

    ax.text(2043, 21.5, '2°C PATHWAY',
        fontsize=12, ha='center', va='center',
        fontweight='bold', alpha=1, color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#2C3E50', alpha=0.9))

    ax.text(2047, -3.5, 'HIGH DURABILITY CDR',
        fontsize=12, ha='center', va='center',
        fontweight='bold', alpha=0.8, color='#3498DB',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#3498DB', alpha=1))
    
    # How many  CDR tonnes removed by 2050 cumulatively
    ax.text(2020, -cdr_annual/2, f'{total_cdr_rounded:.0f} Gt CO\N{SUBSCRIPT TWO}\nREMOVED BY 2050',
        fontsize=12, ha='center', va='center',
        fontweight='bold', alpha=0.9, color='#2C3E50')
    # Total tonnes removed cumulatively lines
    ax.plot([2023, 2023], [-1, -cdr_annual],
        color='#E74C3C', linewidth=3, alpha=0.8)

    ax.plot([2023, 2023.5], [-1, -1],
        color='#E74C3C', linewidth=3, alpha=0.8)

    ax.plot([2023, 2023.5], [-cdr_annual, -cdr_annual],
        color='#E74C3C', linewidth=3, alpha=0.8)
    
    # 2025 label
    ax.text(2025, 46.5, '2025',
        fontsize=12, ha='center', va='center',
        fontweight='bold', alpha=0.7, color='#2C3E50')
    
    ax.plot([2025, 2025], [45,42],
        color='#2C3E50', linewidth=2, alpha=0.7)

    return fig


def create_cdr_only_chart(df_complete):
    """
    Create chart showing only CDR removals (positive values) for 2025-2050.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))

    # Filter to 2025-2050 and convert to Gt
    df_filtered = df_complete[df_complete['year'] >= 2025].copy()
    years = df_filtered['year']
    eng_cdr_gt = df_filtered['engineered_cdr_tonnes'] / 1e9
    nat_cdr_gt = df_filtered['nature_cdr_tonnes'] / 1e9

    # Create stacked bars
    bar_width = 0.8

    # Nature-based CDR (bottom)
    bars_nat_cdr = ax.bar(years, nat_cdr_gt,
                         color='#27AE60', alpha=0.8, width=bar_width)

    # Engineered CDR (on top)
    bars_eng_cdr = ax.bar(years, eng_cdr_gt,
                         bottom=nat_cdr_gt,
                         color='#3498DB', alpha=0.8, width=bar_width)

    # Formatting
    ax.set_xlim(2024, 2051)

    # Calculate max for y-axis
    max_total = (eng_cdr_gt + nat_cdr_gt).max()
    ax.set_ylim(0, max_total * 1.1)

    # Y-axis formatting
    ax.set_ylabel('', fontsize=14,
                  fontproperties=font_props.get('regular') if font_props else None)

    # X-axis formatting
    ax.set_xticks(range(2025, 2051, 5))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add text annotations
    nature_2030 = df_filtered[df_filtered['year'] == 2030]['nature_cdr_tonnes'].iloc[0] / 1e9
    eng_2050 = df_filtered[df_filtered['year'] == 2050]['engineered_cdr_tonnes'].iloc[0] / 1e9
    nature_2050 = df_filtered[df_filtered['year'] == 2050]['nature_cdr_tonnes'].iloc[0] / 1e9

    ax.text(2027, 19, f"MAJORITY LOW DURABILITY\nCDR IN 2030",
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=1, color='#27AE60')

    ax.plot([2029.6, 2030.4], [0.1, 0.1],
        color='#E74C3C', alpha=0.8, linewidth = 4)

    ax.plot([2029.6, 2030.4], [nature_2030, nature_2030],
        color="#E74C3C", alpha=0.8, linewidth = 4)

    ax.plot([2029.6, 2029.6], [0, nature_2030],
        color='#E74C3C', alpha=0.8, linewidth = 4)

    ax.plot([2030.4, 2030.4], [0, nature_2030],
        color='#E74C3C', alpha=0.8, linewidth = 4)

    ax.text(2048, 24, f"APPROACHING 100% HIGH\nDURABILITY CDR IN 2050",
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=1, color='#3498DB')

    ax.plot([2049.6, 2050.4], [eng_2050 + nature_2050, eng_2050 + nature_2050],
    color='#E74C3C', alpha=0.8, linewidth = 4)

    ax.plot([2049.6, 2050.4], [nature_2050, nature_2050],
        color="#E74C3C", alpha=0.8, linewidth = 4)

    ax.plot([2049.6, 2049.6], [nature_2050, eng_2050 + nature_2050],
        color='#E74C3C', alpha=0.8, linewidth = 4)

    ax.plot([2050.4, 2050.4], [nature_2050, eng_2050 + nature_2050],
        color='#E74C3C', alpha=0.8, linewidth = 4)

    ax.text(2045, 12, f"HIGH DURABILITY CDR",
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=1, color='#3498DB',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
            edgecolor='#2C3E50', alpha=1))

    ax.text(2037, 7, "LOW DURABILITY CDR",
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=1, color='#27AE60',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
            edgecolor='#2C3E50', alpha=0.9))

    return fig


def create_fossil_fuel_emissions_chart(df_complete, ax=None):
    """
    Create stacked bar chart showing fossil fuel emissions by type from 2015-2050.
    """
    if ax is None:
        fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))
    else:
        fig = ax.figure
        font_props = None

    # Use full date range like net emissions chart (2015-2050)
    df_filtered = df_complete[df_complete['year'] >= 2015].copy()

    # For historical years (2015-2024), use total emissions
    # For future years (2025+), use fossil fuel breakdown
    historical_mask = df_filtered['year'] < 2025
    future_mask = df_filtered['year'] >= 2025

    # Historical total emissions (single bar)
    historical_emissions_gt = df_filtered.loc[historical_mask, 'global_emissions'] / 1e9
    historical_years = df_filtered.loc[historical_mask, 'year']

    # Future fossil fuel breakdown
    emissions_oil_gt = df_filtered.loc[future_mask, 'global_emissions_oil'] / 1e9
    emissions_gas_gt = df_filtered.loc[future_mask, 'global_emissions_gas'] / 1e9
    emissions_coal_gt = df_filtered.loc[future_mask, 'global_emissions_coal'] / 1e9
    future_years = df_filtered.loc[future_mask, 'year']

    # Group coal and oil into "Other" category
    emissions_other_gt = emissions_coal_gt + emissions_oil_gt

    # Filter out any NaN values for future data
    future_mask_clean = pd.notna(emissions_other_gt) & pd.notna(emissions_gas_gt)
    other_plot = emissions_other_gt[future_mask_clean]
    gas_plot = emissions_gas_gt[future_mask_clean]
    future_years_plot = future_years[future_mask_clean]

    # Create stacked bars
    bar_width = 0.8

    # Historical emissions (single bars for 2015-2024)
    if len(historical_years) > 0 and len(historical_emissions_gt) > 0:
        historical_mask_clean = pd.notna(historical_emissions_gt)
        ax.bar(historical_years[historical_mask_clean], historical_emissions_gt[historical_mask_clean],
               color='#C0392B', alpha=0.9, width=bar_width)

    # Future fossil fuel breakdown (2025-2050)
    if len(future_years_plot) > 0:
        # Other (coal + oil) - bottom layer in gray
        bars_other = ax.bar(future_years_plot, other_plot,
                          color='#808080', alpha=0.7, width=bar_width)

        # Gas - top layer
        bars_gas = ax.bar(future_years_plot, gas_plot,
                         bottom=other_plot,
                         color='#FF6347', alpha=0.8, width=bar_width)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Formatting - same as net emissions chart
    ax.set_xlim(2014, 2051)
    ax.set_ylim(0, 49)

    # Y-axis formatting
    ax.set_ylabel('', fontsize=14,
                  fontproperties=font_props.get('regular') if font_props else None)

    # X-axis formatting - same as net emissions chart
    ax.set_xticks(range(2020, 2051, 10))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add annotations for different periods and fossil fuel types
    ax.text(2019, 40, 'HISTORICAL\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#C0392B')

    ax.text(2040, 42.5, 'PROJECTED\nEMISSIONS',
        fontsize=12, ha='center', va='center',
        fontweight='bold', color='gray')

    # Future fossil fuel breakdown annotations
    if len(other_plot) > 0 and len(gas_plot) > 0:
        mid_year = 2035
        # Find closest year in our data
        closest_idx = (future_years_plot - mid_year).abs().idxmin() if len(future_years_plot) > 0 else 0

        if closest_idx in other_plot.index:
            other_mid = other_plot.loc[closest_idx]
            gas_mid = gas_plot.loc[closest_idx]

            # Other (coal + oil) annotation
            ax.text(mid_year, other_mid / 2, 'OTHER\n(COAL + OIL)',
                    fontsize=12, ha='center', va='center',
                    fontweight='bold', alpha=0.9, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#808080',
                    edgecolor='#2C3E50', alpha=0.9))

            # Gas annotation
            ax.text(mid_year, other_mid + gas_mid / 2, 'GAS',
                    fontsize=12, ha='center', va='center',
                    fontweight='bold', alpha=0.9, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF6347',
                    edgecolor='#2C3E50', alpha=0.9))

    return fig


def create_gas_emissions_production_chart(df_complete):
    """
    Create bar chart showing gas emissions (Gt) from 2015-2050.
    Uses different colors for past (darker) vs projected (lighter) emissions.
    """
    # Filter to 2015-2050 range (same as fossil fuel chart)
    df_filtered = df_complete[df_complete['year'] >= 2015].copy()

    # Convert gas emissions from tonnes to Gt
    gas_emissions_gt = df_filtered['global_emissions_gas'] / 1e9
    years = df_filtered['year']

    # Create figure
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))

    # Create bar chart
    bar_width = 0.8

    # Split into past and projected
    past_mask = (years < 2025) & pd.notna(gas_emissions_gt)
    projected_mask = (years >= 2025) & pd.notna(gas_emissions_gt)

    # Past emissions (darker red/coral)
    if past_mask.any():
        ax.bar(years[past_mask], gas_emissions_gt[past_mask],
               color='#C0392B', alpha=0.9, width=bar_width)

    # Projected emissions (lighter red/coral - same as net emissions chart)
    if projected_mask.any():
        ax.bar(years[projected_mask], gas_emissions_gt[projected_mask],
               color='#FF6347', alpha=0.8, width=bar_width)

    # Formatting - match fossil fuel chart
    ax.set_xlim(2014, 2051)
    max_emissions = max(gas_emissions_gt[pd.notna(gas_emissions_gt)])
    ax.set_ylim(0, max_emissions * 1.1)

    # Y-axis formatting
    ax.set_ylabel('', fontsize=14,
                  fontproperties=font_props.get('regular') if font_props else None)

    # X-axis formatting - same as fossil fuel chart
    ax.set_xticks(range(2020, 2051, 10))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Add annotations for past and projected periods (matching net emissions chart style)
    ax.text(2019, max_emissions * 0.85, 'PAST\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#C0392B')

    ax.text(2040, max_emissions * 1.05, 'PROJECTED\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#FF6347')

    return fig


def project_lower_bound_cdr_scenario(df):
    """
    Project lower bound CDR needs from 2025 to 2100.
    This scenario assumes rapid emissions cuts, with CDR needed for:
    1. Emissions smoothly decline from 2024 actual to 4.16 Gt in 2050
    2. Hard-to-abate emissions: 4.16 Gt CO2 annually from 2050-2100
    3. Legacy emissions: 65 Gt CO2 total to remove by 2100

    CDR must offset annual emissions AND gradually remove legacy emissions.
    Uses same transition from nature-based to engineered as upper bound.
    """
    print('\nProjecting lower bound CDR scenario (hard-to-abate + legacy)...')

    # Create extended dataframe to 2100
    years_extended = list(range(2000, 2101))
    df_extended = pd.DataFrame({'year': years_extended})

    # Merge with existing data (contains historical emissions through 2023)
    df_extended = df_extended.merge(df[['year', 'global_emissions']], on='year', how='left')

    # Initialize columns for lower bound scenario
    df_extended['lower_bound_emissions'] = 0.0
    df_extended['lower_bound_legacy_removal'] = 0.0
    df_extended['lower_bound_total_cdr'] = 0.0
    df_extended['lower_bound_eng_cdr'] = 0.0
    df_extended['lower_bound_nat_cdr'] = 0.0

    # Copy historical emissions (2000-2023) from Global Carbon Project
    historical_mask = df_extended['year'] <= 2023
    df_extended.loc[historical_mask, 'lower_bound_emissions'] = df_extended.loc[historical_mask, 'global_emissions']

    # Get 2023 emissions value for smooth transition
    emissions_2023 = df_extended[df_extended['year'] == 2023]['global_emissions'].values[0]

    # Project emissions 2024-2100: smooth parabolic curve asymptotically approaching 4.16 GT
    # Use exponential decay towards the hard-to-abate floor
    # emissions(t) = hard_to_abate + (start - hard_to_abate) * exp(-decay_rate * t)
    decay_rate = 0.09  # Controls how fast we approach hard-to-abate (lower = slower approach)

    for year in range(2024, 2101):
        years_since_2023 = year - 2023
        # Exponential approach to hard-to-abate asymptote
        emissions = HARD_TO_ABATE_TONNES + (emissions_2023 - HARD_TO_ABATE_TONNES) * np.exp(-decay_rate * years_since_2023)
        df_extended.loc[df_extended['year'] == year, 'lower_bound_emissions'] = emissions

    # CDR capacity scale-up: Single exponential growth curve from 2025 to 2100
    # Area under curve must equal cumulative_cdr_target_2100
    # CDR must offset ALL emissions from 2025-2100 PLUS remove legacy emissions
    # cumulative_emissions_2025_2100 = df_extended[(df_extended['year'] >= 2025) & (df_extended['year'] <= 2100)]['lower_bound_emissions'].sum()
    # cumulative_cdr_target_2100 = cumulative_emissions_2025_2100 + LEGACY_TOTAL
    cumulative_cdr_target_2100 = (HARD_TO_ABATE_TONNES*50) + LEGACY_TOTAL_TONNES

    # Create saturating exponential growth curve: CDR(t) = a * (1 - exp(-k * t))
    # This grows rapidly at first, then slows down (opposite of regular exponential)
    years_2025_2100 = np.arange(2025, 2101)
    n_years = len(years_2025_2100)

    # Generate saturating exponential growth curve
    growth_rate = 0.08  # Controls how fast it saturates (higher = faster initial growth)
    trial_cdr = []
    for year in years_2025_2100:
        years_since_2024 = year - 2024
        # Saturating exponential: rapid growth initially, slowing over time
        capacity = 1 - np.exp(-growth_rate * years_since_2024)
        trial_cdr.append(capacity)

    trial_cdr = np.array(trial_cdr)

    # Scale the curve so that sum (area under curve) = cumulative_cdr_target_2100
    scaling_factor = cumulative_cdr_target_2100 / trial_cdr.sum()

    # Apply scaled CDR capacity and calculate legacy removal
    for i, year in enumerate(years_2025_2100):
        total_cdr_capacity = trial_cdr[i] * scaling_factor
        df_extended.loc[df_extended['year'] == year, 'lower_bound_total_cdr'] = total_cdr_capacity

        # Legacy removal = total CDR - emissions being offset
        emissions_year = df_extended.loc[df_extended['year'] == year, 'lower_bound_emissions'].values[0]
        legacy_removal = max(0, total_cdr_capacity - emissions_year)
        df_extended.loc[df_extended['year'] == year, 'lower_bound_legacy_removal'] = legacy_removal

    # Split into engineered vs nature-based using same transition logic
    for year in range(2025, 2101):
        # Progress through transition (complete by 2050, maintain after)
        if year <= 2050:
            progress = (year - 2025) / (2050 - 2025)
        else:
            progress = 1.0

        # Transition ratio (sigmoid curve)
        ratio_transition = 1 / (1 + np.exp(-6 * (progress - 0.5)))
        eng_ratio = (1 - INITIAL_NATURE_RATIO) + (FINAL_ENGINEERED_RATIO - (1 - INITIAL_NATURE_RATIO)) * ratio_transition
        nat_ratio = 1 - eng_ratio

        total_cdr = df_extended.loc[df_extended['year'] == year, 'lower_bound_total_cdr'].values[0]

        df_extended.loc[df_extended['year'] == year, 'lower_bound_eng_cdr'] = total_cdr * eng_ratio
        df_extended.loc[df_extended['year'] == year, 'lower_bound_nat_cdr'] = total_cdr * nat_ratio

    print(f'  2023 emissions: {emissions_2023/1e9:.2f} Gt CO2')
    print(f'  Hard-to-abate floor: {HARD_TO_ABATE_TONNES/1e9:.2f} Gt CO2/year')
    print(f'  Legacy total to remove: {LEGACY_TOTAL_TONNES/1e9:.0f} Gt CO2')
    print(f'  Cumulative CDR target (2025-2100): {cumulative_cdr_target_2100/1e9:.0f} Gt CO2')
    total_cdr_2100 = df_extended[df_extended['year'] == 2100]['lower_bound_total_cdr'].values[0]
    print(f'  CDR capacity in 2100: {total_cdr_2100/1e9:.1f} Gt CO2/year')
    return df_extended


def calculate_legacy_emissions_per_year(emissions_gt, hard_to_abate_gt, legacy_total_gt, years):
    """
    Calculate how much legacy emissions to show in each year's bar.

    Args:
        emissions_gt: Array of total emissions in Gt for each year
        hard_to_abate_gt: Hard-to-abate floor in Gt
        legacy_total_gt: Total legacy emissions to remove (65 Gt)
        years: Array of years

    Returns:
        Array of legacy emissions in Gt for each year
    """
    n_years = len(years)

    # Calculate available space above hard-to-abate for each year
    available_space = np.maximum(emissions_gt - hard_to_abate_gt, 0)

    # Start with equal distribution
    target_per_year = legacy_total_gt / n_years

    # Allocate legacy emissions, capped by available space
    legacy_per_year = np.zeros(n_years)
    remaining_legacy = legacy_total_gt

    # First pass: allocate up to target or available space, whichever is smaller
    for i in range(n_years):
        allocated = min(target_per_year, available_space[i])
        legacy_per_year[i] = allocated
        remaining_legacy -= allocated

    # Second pass: distribute remaining legacy evenly to years with extra space
    if remaining_legacy > 0:
        # Find years with extra space
        extra_space = available_space - legacy_per_year
        years_with_space = extra_space > 0
        n_years_with_space = years_with_space.sum()

        if n_years_with_space > 0:
            # Distribute evenly across years with space
            additional_per_year = remaining_legacy / n_years_with_space

            for i in range(n_years):
                if years_with_space[i]:
                    # Add evenly, but cap at available space
                    additional = min(additional_per_year, extra_space[i])
                    legacy_per_year[i] += additional
                    remaining_legacy -= additional

            # If still remaining (because we hit caps), do another pass
            # if remaining_legacy > 1e-6:  # Small tolerance for floating point
            #     for i in range(n_years):
            #         if remaining_legacy <= 0:
            #             break
            #         extra = available_space[i] - legacy_per_year[i]
            #         if extra > 0:
            #             additional = min(remaining_legacy, extra)
            #             legacy_per_year[i] += additional
            #             remaining_legacy -= additional

    return legacy_per_year


def create_lower_bound_cdr_chart(df_lower_bound):
    """
    Create chart showing lower bound CDR scenario.
    Shows historical + projected emissions declining to hard-to-abate levels,
    and CDR ramping up to offset emissions plus remove legacy CO2.
    Matches style of net_emissions_cdr_scaleup chart.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))

    # Filter to 2015-2100 for visualization
    df_filtered = df_lower_bound[df_lower_bound['year'] >= 2000].copy()

    # Convert to Gt
    years = df_filtered['year']
    emissions_gt = df_filtered['lower_bound_emissions'] / 1e9
    total_cdr_gt = df_filtered['lower_bound_total_cdr'] / 1e9

    # Hard-to-abate constant in Gt
    hard_to_abate_gt = HARD_TO_ABATE_TONNES / 1e9
    legacy_total_gt = LEGACY_TOTAL_TONNES / 1e9

    bar_width = 0.8

    # Split into past (2015-2024) and projected (2025+)
    past_mask = years < 2025
    projected_mask = years >= 2025

    # Calculate legacy emissions portion for each year (for CDR stacking)
    legacy_per_year = calculate_legacy_emissions_per_year(emissions_gt, hard_to_abate_gt, legacy_total_gt, years)

    # Past emissions - solid red bars (no stacking)
    if past_mask.any():
        ax.bar(years[past_mask], emissions_gt[past_mask],
               color='#C0392B', alpha=0.9, width=bar_width)

    # Projected emissions - solid red bars (no stacking)
    if projected_mask.any():
        ax.bar(years[projected_mask], emissions_gt[projected_mask],
               color='#E74C3C', alpha=0.7, width=bar_width)

    # CDR removals (negative, below zero) - stacked to show what it's addressing
    # Total CDR = hard-to-abate offset + legacy removal (no additional capacity)
    if projected_mask.any():
        # Brown: offsetting hard-to-abate emissions (bottom layer)
        hard_to_abate_removal = np.minimum(total_cdr_gt[projected_mask], hard_to_abate_gt)

        # Orange: removing legacy emissions (everything above hard-to-abate)
        legacy_removal = total_cdr_gt[projected_mask] - hard_to_abate_removal

        # Plot stacked CDR bars (negative values, below zero)
        # Brown: offsetting hard-to-abate
        ax.bar(years[projected_mask], -hard_to_abate_removal,
               color='#8B4513', alpha=0.8, width=bar_width)

        # Orange: removing legacy emissions
        ax.bar(years[projected_mask], -legacy_removal,
               bottom=-hard_to_abate_removal,
               color='#FF8C00', alpha=0.8, width=bar_width)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xlim(1999, 2101)

    # Calculate max values for y-axis
    max_removal = -total_cdr_gt.max()
    max_emission = emissions_gt.max()
    y_max = max(max_emission * 1.1, 10)
    y_min = max(max_removal * 1.2, -15)
    ax.set_ylim(y_min, y_max)

    # Y-axis formatting
    ax.set_ylabel('', fontsize=14,
                  fontproperties=font_props.get('regular') if font_props else None)

    # X-axis formatting
    ax.set_xticks(range(2000, 2101, 50))
    ax.tick_params(axis='both', labelsize=12)
    # ax.grid(axis='y', alpha=0.3)

    # Remove axis spines (edge lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add annotations
    ax.text(2019, max_emission * 1.05, 'PAST\nEMISSIONS',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#C0392B')

    ax.text(2045, max_emission * 0.4, 'AGGRESSIVE EMISSIONS\nREDUCTION',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#E74C3C')

    ax.text(2035, -4.5, 'HARD-TO-ABATE\nEMISSIONS',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#8B4513')

    ax.text(2075, -4.5, 'LEGACY\nEMISSIONS',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.9, color='#FF8C00'
                )

    # ax.text(2080, max_removal * 0.5, 'CDR',
    #         fontsize=12, ha='center', va='center',
    #         fontweight='bold', alpha=0.8, color='#3498DB',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
    #             edgecolor='#3498DB', alpha=1)
    #             )

    # Calculate total CDR cumulatively (sum over 2025-2100)
    total_cdr = total_cdr_gt[years >= 2025].sum()
    ax.text(2017, max_removal * .6, f'{total_cdr:.0f} Gt CO\N{SUBSCRIPT TWO}\nREMOVED BY 2100',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.9, color='#2C3E50')

    # Cumulative removal bracket
    bracket_x = 2024
    bracket_top = max_removal * 0.1
    bracket_bottom = max_removal
    ax.plot([bracket_x, bracket_x], [bracket_top, bracket_bottom],
            color='#E74C3C', linewidth=2, alpha=0.8)
    ax.plot([bracket_x, bracket_x + 0.25], [bracket_top, bracket_top],
            color='#E74C3C', linewidth=2, alpha=0.8)
    ax.plot([bracket_x, bracket_x + 0.25], [bracket_bottom, bracket_bottom],
            color='#E74C3C', linewidth=2, alpha=0.8)

    # 2025 label
    # ax.text(2025, y_max * 0.95, '2025',
    #         fontsize=12, ha='center', va='center',
    #         fontweight='bold', alpha=0.7, color='#2C3E50')

    # ax.plot([2025, 2025], [y_max * 0.90, y_max * 0.77],
    #         color='#2C3E50', linewidth=2, alpha=0.7)

    return fig

def project_emissions_to_2100(df):
    '''
    Project emissions from 2050 to 2100 using cubic spline fit to historical trend.
    Also projects CDR capacity to 2100.
    '''
    # Create extended dataframe
    years_extended = list(range(df['year'].min(), 2101))
    df_extended = pd.DataFrame({'year': years_extended})

    # Merge with existing data
    df_extended = df_extended.merge(df, on='year', how='left')

    # Get emissions data up to 2050 for fitting
    emissions_data = df[df['year'] <= 2050][['year', 'global_emissions']].dropna()

    if len(emissions_data) > 3:  # Need at least 4 points for cubic spline
        # Use last 30 years of data for better trend fitting
        recent_data = emissions_data[emissions_data['year'] >= 2010]

        if len(recent_data) >= 4:
            # Fit cubic spline to recent emissions trend
            cs_emissions = CubicSpline(recent_data['year'].values,
                                      recent_data['global_emissions'].values,
                                      bc_type='natural')

            # Project emissions 2051-2100
            for year in range(2051, 2101):
                projected = cs_emissions(year)
                # Ensure emissions don't go negative
                projected = max(projected, HARD_TO_ABATE_TONNES)
                df_extended.loc[df_extended['year'] == year, 'global_emissions'] = projected

    # Project CDR capacity to 2100
    # Get CDR data up to 2050
    cdr_eng_data = df[df['year'] <= 2050][['year', 'engineered_cdr_tonnes']].dropna()
    cdr_nat_data = df[df['year'] <= 2050][['year', 'nature_cdr_tonnes']].dropna()

    if len(cdr_eng_data) > 3:
        # Get 2050 values
        eng_2050 = df[df['year'] == 2050]['engineered_cdr_tonnes'].values[0]
        nat_2050 = df[df['year'] == 2050]['nature_cdr_tonnes'].values[0]

        # Continue growth from 2050 to 2100, approaching a plateau
        for year in range(2051, 2101):
            progress = (year - 2050) / (2100 - 2050)
            # Slower growth factor for later years
            growth_factor = 1 + (1.5 * (1 - np.exp(-2 * progress)))  # Approaches 2.5x by 2100

            df_extended.loc[df_extended['year'] == year, 'engineered_cdr_tonnes'] = eng_2050 * growth_factor
            df_extended.loc[df_extended['year'] == year, 'nature_cdr_tonnes'] = nat_2050 * (1 + 0.2 * progress)  # Slower growth

    return df_extended

def create_upper_bound_cdr_chart(df_complete):
    """
    Create upper bound CDR scenario chart showing high emissions pathway
    with CDR needed to reach 2°C target.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))

    # Project emissions to 2100
    df_extended = project_emissions_to_2100(df_complete)

    # Filter to 2000-2100 for visualization
    df_filtered = df_extended[df_extended['year'] >= 2000].copy()

    # Convert to Gt
    years = df_filtered['year']
    emissions_gt = df_filtered['global_emissions'] / 1e9

    # Combine engineered and nature-based CDR into total
    eng_cdr = df_filtered['engineered_cdr_tonnes'].fillna(0)
    nat_cdr = df_filtered['nature_cdr_tonnes'].fillna(0)
    total_cdr_gt = (eng_cdr + nat_cdr) / 1e9

    # Hard-to-abate constant in Gt
    hard_to_abate_gt = HARD_TO_ABATE_TONNES / 1e9
    legacy_total_gt = LEGACY_TOTAL_TONNES / 1e9

    bar_width = 0.8

    # Split into past (2000-2024) and projected (2025+)
    past_mask = years < 2025
    projected_mask = years >= 2025

    # Past emissions - solid red bars
    if past_mask.any():
        ax.bar(years[past_mask], emissions_gt[past_mask],
               color='#C0392B', alpha=0.9, width=bar_width)

    # Projected emissions - solid red bars
    if projected_mask.any():
        ax.bar(years[projected_mask], emissions_gt[projected_mask],
               color='#E74C3C', alpha=0.7, width=bar_width)

    # CDR removals (negative, below zero) - stacked brown/orange
    if projected_mask.any():
        # Brown: offsetting hard-to-abate emissions (bottom layer)
        hard_to_abate_removal = np.minimum(total_cdr_gt[projected_mask], hard_to_abate_gt)

        # Orange: removing legacy emissions (everything above hard-to-abate)
        legacy_removal = np.maximum(total_cdr_gt[projected_mask] - hard_to_abate_gt, 0)

        # Plot stacked CDR bars (negative values, below zero)
        ax.bar(years[projected_mask], -hard_to_abate_removal,
               color='#8B4513', alpha=0.8, width=bar_width)

        ax.bar(years[projected_mask], -legacy_removal,
               bottom=-hard_to_abate_removal,
               color='#FF8C00', alpha=0.8, width=bar_width)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xlim(1999, 2101)

    # Calculate max values for y-axis
    max_removal = -total_cdr_gt[total_cdr_gt > 0].max() if (total_cdr_gt > 0).any() else -5
    max_emission = emissions_gt.max()
    y_max = max(max_emission * 1.1, 10)
    y_min = max(max_removal * 1.2, -25)
    ax.set_ylim(y_min, y_max)

    # Y-axis formatting
    ax.set_ylabel('', fontsize=14,
                  fontproperties=font_props.get('regular') if font_props else None)

    # X-axis formatting
    ax.set_xticks(range(2000, 2101, 50))
    ax.tick_params(axis='both', labelsize=12)

    # Remove axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add annotations
    ax.text(2019, max_emission * 1.05, 'PAST\nEMISSIONS',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#C0392B')

    ax.text(2045, max_emission * 0.65, 'HIGH EMISSIONS\nPATHWAY (PGR)',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#E74C3C')

    ax.text(2035, max_removal * 0.3, 'HARD-TO-ABATE\nEMISSIONS',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#8B4513')

    ax.text(2070, max_removal * 0.6, 'LEGACY\nEMISSIONS',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.9, color='#FF8C00')

    # Calculate total CDR cumulatively (sum over 2025-2100)
    total_cdr = total_cdr_gt[years >= 2025].sum()
    ax.text(2017, max_removal * 0.5, f'{total_cdr:.0f} Gt CO\N{SUBSCRIPT TWO}\nREMOVED BY 2100',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.9, color='#2C3E50')

    # Cumulative removal bracket
    bracket_x = 2024
    bracket_top = max_removal * 0.1
    bracket_bottom = max_removal * 0.95
    ax.plot([bracket_x, bracket_x], [bracket_top, bracket_bottom],
            color='#E74C3C', linewidth=2, alpha=0.8)
    ax.plot([bracket_x, bracket_x + 0.25], [bracket_top, bracket_top],
            color='#E74C3C', linewidth=2, alpha=0.8)
    ax.plot([bracket_x, bracket_x + 0.25], [bracket_bottom, bracket_bottom],
            color='#E74C3C', linewidth=2, alpha=0.8)

    return fig



def main():
    """
    Main execution function.
    """
    print('='*60)
    print('CDR Scale-Up Analysis V2')
    print('Building from source data files')
    print('='*60)

    # Build complete dataset
    df_complete = build_complete_dataset()

    # Run quality check on PGR data
    pgr_data_loading_quality_check(df_complete)

    # Project CDR scale-up
    df_complete = project_cdr_scale_up(df_complete)

    # Save output CSV
    output_dir = 'data/needed_removal_capacity'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'cdr_scaleup_output.csv')
    df_complete.to_csv(output_csv, index=False)
    print(f'\n✓ Saved complete dataset to: {output_csv}')

    # Create visualizations
    print('\n' + '='*60)
    print('Creating Visualizations')
    print('='*60)

    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Net emissions chart
    print('\n1. Creating net emissions chart...')
    net_emissions_fig = create_net_emissions_chart(df_complete)
    format_plot_title(plt.gca(), "", "GLOBAL CO\N{SUBSCRIPT TWO} EMISSIONS & REMOVALS (GIGATONNES)", None)
    add_deep_sky_branding(plt.gca(), None,
                         "DATA: GLOBAL CARBON PROJECT (2024); SEI, CLIMATE ANALYTICS, IISD (2025) THE PRODUCTION GAP REPORT; CDR.FYI",
                         analysis_date=datetime.datetime.now())
    save_plot(net_emissions_fig, 'figures/net_emissions_cdr_scaleup.png')
    print('  ✓ Saved to figures/net_emissions_cdr_scaleup.png')

    # 2. CDR only chart
    print('\n2. Creating CDR scale-up chart...')
    cdr_fig = create_cdr_only_chart(df_complete)
    format_plot_title(plt.gca(), "", "GLOBAL CARBON REMOVAL CAPACITY (GIGATONNES)", None)
    add_deep_sky_branding(plt.gca(), None,
                         "DATA: The Oxford Principles for Net Zero Aligned Carbon Offsetting (revised 2024).",
                         analysis_date=datetime.datetime.now())
    save_plot(cdr_fig, 'figures/cdr_scaleup.png')
    print('  ✓ Saved to figures/cdr_scaleup.png')

    # 3. Fossil fuel emissions chart
    print('\n3. Creating fossil fuel emissions chart...')
    fossil_fig = create_fossil_fuel_emissions_chart(df_complete)
    format_plot_title(plt.gca(), "", "GLOBAL CO\N{SUBSCRIPT TWO} EMISSIONS BY FOSSIL FUEL (GIGATONNES)", None)
    add_deep_sky_branding(plt.gca(), None,
                         "DATA: THE PRODUCTION GAP REPORT (2025)",
                         analysis_date=datetime.datetime.now())
    save_plot(fossil_fig, 'figures/fossil_fuel_emissions.png')
    print('  ✓ Saved to figures/fossil_fuel_emissions.png')

    # 4. Gas emissions chart
    print('\n4. Creating gas emissions chart...')
    gas_fig = create_gas_emissions_production_chart(df_complete)
    format_plot_title(plt.gca(), "", "GLOBAL NATURAL GAS EMISSIONS (GIGATONNES)", None)
    add_deep_sky_branding(plt.gca(), None,
                         "DATA: GLOBAL CARBON PROJECT (2024); SEI, CLIMATE ANALYTICS, IISD (2025) THE PRODUCTION GAP REPORT",
                         analysis_date=datetime.datetime.now())
    save_plot(gas_fig, 'figures/gas_emissions_production.png')
    print('  ✓ Saved to figures/gas_emissions_production.png')

    # # 5. Lower bound CDR scenario chart
    # print('\n5. Creating lower bound CDR scenario chart...')

    # df_lower_bound = project_lower_bound_cdr_scenario(df_complete)

    # # Save lower bound data
    # output_lower_bound_csv = os.path.join(output_dir, 'cdr_lower_bound_output.csv')
    # df_lower_bound.to_csv(output_lower_bound_csv, index=False)
    # print(f'  ✓ Saved lower bound dataset to: {output_lower_bound_csv}')

    # lower_bound_fig = create_lower_bound_cdr_chart(df_lower_bound)
    # format_plot_title(plt.gca(), "", "GLOBAL CO\N{SUBSCRIPT TWO} EMISSIONS & REMOVALS (GIGATONNES)", None)
    # add_deep_sky_branding(plt.gca(), None,
    #                      f"CDR PRIMER (2021), MACELROY (2025). HARD-TO-ABATE EMISSIONS = {HARD_TO_ABATE_TONNES/1e9} GT/YR, LEGACY EMISSIONS = 65 GT TOTAL")
    # save_plot(lower_bound_fig, 'figures/lower_bound_cdr_scenario.png')
    # print('  ✓ Saved to figures/lower_bound_cdr_scenario.png')

    # upper_bound_fig = create_upper_bound_cdr_chart(df_complete)
    # format_plot_title(plt.gca(), "", "UPPER BOUND: HIGH EMISSIONS PATHWAY", None)
    # add_deep_sky_branding(plt.gca(), None,
    #                      "DATA: GLOBAL CARBON PROJECT (2024); SEI, CLIMATE ANALYTICS, IISD (2025) THE PRODUCTION GAP REPORT")
    # save_plot(upper_bound_fig, 'figures/upper_bound_cdr_scenario.png')
    # print('  ✓ Saved to figures/upper_bound_cdr_scenario.png')

    print('\n' + '='*60)
    print('✓ Complete!')
    print('='*60)


if __name__ == '__main__':
    main()

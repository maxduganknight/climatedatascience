"""
CDR Bounds Analysis - Lower and Upper Bound Scenarios

This script creates two CDR scenarios:
- Lower Bound: Aggressive emissions reduction to hard-to-abate floor + legacy removal
- Upper Bound: High emissions pathway (PGR) requiring large CDR to reach 1.5°C target

Author: Deep Sky
"""

import pandas as pd
import numpy as np
import sys
from scipy.interpolate import CubicSpline

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot

# =======================
# GLOBAL CONSTANTS
# =======================

# Hard-to-abate emissions (tonnes CO2 per year)
# The CDR primer estimates 1.5-3.1 Gt for hard-to-avoid emissions
HARD_TO_ABATE_TONNES = (1.5e9 + 3.1e9) / 2

# Legacy emissions to remove (total tonnes CO2)
LEGACY_TOTAL_TONNES = 65e9

# Molecular weight conversion
GTC_TO_GTCO2 = 3.664


# =======================
# DATA LOADING FUNCTIONS
# =======================

def load_historical_emissions():
    """
    Load historical emissions from Global Carbon Budget 2024.
    Returns DataFrame with year and global_emissions (in tonnes CO2).
    """
    print('Loading historical emissions from Global Carbon Budget 2024...')

    gcb_file = 'data/Global_Carbon_Budget_2024_v1.0-1.xlsx'

    # Read Historical Budget sheet
    df = pd.read_excel(gcb_file, sheet_name='Historical Budget', skiprows=15)

    # First column is year, second is fossil emissions (GtC/yr)
    df_clean = pd.DataFrame({
        'year': pd.to_numeric(df.iloc[:, 0], errors='coerce'),
        'fossil_emissions_gtc': pd.to_numeric(df.iloc[:, 1], errors='coerce')
    })

    df_clean = df_clean.dropna()

    # Convert GtC to tonnes CO2
    df_clean['past_emissions'] = df_clean['fossil_emissions_gtc'] * GTC_TO_GTCO2 * 1e9

    # Filter to 2000-2023
    df_clean = df_clean[(df_clean['year'] >= 2000) & (df_clean['year'] <= 2023)]

    print(f'  Loaded {len(df_clean)} years of historical emissions (2000-2023)')
    print(f'  2023 emissions: {df_clean[df_clean["year"] == 2023]["past_emissions"].values[0]/1e9:.2f} GtCO2')

    return df_clean[['year', 'past_emissions']]


def load_pgr_projections():
    """
    Load Production Gap Report 2025 projections for emissions and climate pathways.
    Returns DataFrame with year, upper_bound_emissions, pathway_1_5c, and pathway_2c.
    """
    print('\nLoading Production Gap Report 2025 projections...')

    pgr_file = 'data/needed_removal_capacity/PGR2025_data.xlsx'

    # Read Figure ES-1 and 2.1 sheet
    df = pd.read_excel(pgr_file, sheet_name='Figure ES-1 and 2.1', header=None)

    # Row 4 has years, Row 5 has total emissions, Row 10 has 2C pathway, Row 14 has 1.5C pathway
    years_row = df.iloc[4, 2:9]
    total_emissions_row = df.iloc[5, 2:9]
    pathway_2c_row = df.iloc[10, 2:9]
    pathway_1_5c_row = df.iloc[14, 2:9]

    # Create dataframe
    pgr_data = []
    for i, year in enumerate(years_row.values):
        if not pd.isna(year):
            pgr_data.append({
                'year': int(year),
                'upper_bound_emissions': float(total_emissions_row.values[i]) * 1e9,  # GtCO2 to tonnes
                'pathway_2c': float(pathway_2c_row.values[i]) * 1e9 if not pd.isna(pathway_2c_row.values[i]) else np.nan,
                'pathway_1_5c': float(pathway_1_5c_row.values[i]) * 1e9 if not pd.isna(pathway_1_5c_row.values[i]) else np.nan
            })

    df_pgr = pd.DataFrame(pgr_data)

    print(f'  Loaded {len(df_pgr)} projection years: {df_pgr["year"].tolist()}')

    return df_pgr


# =======================
# PROJECTION FUNCTIONS
# =======================

def interpolate_pgr_emissions(df_pgr):
    """
    Interpolate PGR emissions to fill in missing years using cubic spline.

    Args:
        df_pgr: DataFrame with PGR emissions projections (sparse years)

    Returns:
        DataFrame with emissions interpolated for all years 2023-2050
    """
    # Create full year range from 2023 to 2050
    all_years = np.arange(2023, 2051)

    # Get known years and values for emissions
    known_years = df_pgr['year'].values
    known_emissions = df_pgr['upper_bound_emissions'].values

    # Use cubic spline to interpolate
    cs_emissions = CubicSpline(known_years, known_emissions, bc_type='natural')
    interpolated_emissions = cs_emissions(all_years)

    # Do the same for 1.5C pathway
    pathway_1_5c_data = df_pgr[['year', 'pathway_1_5c']].dropna()
    if len(pathway_1_5c_data) > 2:
        cs_pathway_1_5c = CubicSpline(pathway_1_5c_data['year'].values,
                                      pathway_1_5c_data['pathway_1_5c'].values,
                                      bc_type='natural')
        interpolated_pathway_1_5c = cs_pathway_1_5c(all_years)
    else:
        interpolated_pathway_1_5c = [np.nan] * len(all_years)

    # Do the same for 2C pathway
    pathway_2c_data = df_pgr[['year', 'pathway_2c']].dropna()
    if len(pathway_2c_data) > 2:
        cs_pathway_2c = CubicSpline(pathway_2c_data['year'].values,
                                    pathway_2c_data['pathway_2c'].values,
                                    bc_type='natural')
        interpolated_pathway_2c = cs_pathway_2c(all_years)
    else:
        interpolated_pathway_2c = [np.nan] * len(all_years)

    return pd.DataFrame({
        'year': all_years,
        'upper_bound_emissions': interpolated_emissions,
        'pathway_1_5c': interpolated_pathway_1_5c,
        'pathway_2c': interpolated_pathway_2c
    })


def project_upper_bound_to_2100(df_pgr_interpolated):
    """
    Project upper bound emissions from 2050 to 2100 using linear extrapolation.

    Args:
        df_pgr_interpolated: DataFrame with interpolated PGR emissions (2023-2050)

    Returns:
        DataFrame with projections for 2051-2100
    """
    # Get 2040 and 2050 to establish linear trend
    emissions_2030 = df_pgr_interpolated[df_pgr_interpolated['year'] == 2030]['upper_bound_emissions'].values[0]
    emissions_2040 = df_pgr_interpolated[df_pgr_interpolated['year'] == 2040]['upper_bound_emissions'].values[0]
    emissions_2050 = df_pgr_interpolated[df_pgr_interpolated['year'] == 2050]['upper_bound_emissions'].values[0]

    # Linear projection (declining)
    annual_change = (emissions_2050 - emissions_2030) / 20

    # Project 2051-2100
    projections = []
    for year in range(2051, 2101):
        years_since_2050 = year - 2050
        projected = emissions_2050 + (annual_change * years_since_2050)
        # Floor at hard-to-abate
        projected = max(projected, HARD_TO_ABATE_TONNES)
        projections.append({'year': year, 'upper_bound_emissions': projected})

    return pd.DataFrame(projections)


def project_pathways_to_2100(df_pgr_interpolated):
    """
    Project climate pathways to 2100 (approach net zero by 2050, maintain after).

    Args:
        df_pgr_interpolated: DataFrame with interpolated pathway data (2023-2050)

    Returns:
        DataFrame with pathway projections for 2051-2100
    """
    projections = []

    # Get 2050 values from interpolated data
    pathway_1_5c_data = df_pgr_interpolated[['year', 'pathway_1_5c']].dropna()
    pathway_2c_data = df_pgr_interpolated[['year', 'pathway_2c']].dropna()

    value_1_5c_2050 = pathway_1_5c_data[pathway_1_5c_data['year'] == 2050]['pathway_1_5c'].values[0] if len(pathway_1_5c_data) > 0 else np.nan
    value_2c_2050 = pathway_2c_data[pathway_2c_data['year'] == 2050]['pathway_2c'].values[0] if len(pathway_2c_data) > 0 else np.nan

    # Project to 2100 (approaches net zero)
    for year in range(2051, 2101):
        progress = (year - 2050) / 50

        # 1.5C decays faster to net zero
        projected_1_5c = value_1_5c_2050 * np.exp(-3 * progress) if not np.isnan(value_1_5c_2050) else np.nan

        # 2C decays slower
        projected_2c = value_2c_2050 * np.exp(-2 * progress) if not np.isnan(value_2c_2050) else np.nan

        projections.append({
            'year': year,
            'pathway_1_5c': projected_1_5c,
            'pathway_2c': projected_2c
        })

    return pd.DataFrame(projections)


def calculate_lower_bound_emissions(years, emissions_2023):
    """
    Calculate lower bound emissions: exponential decay from 2023 to hard-to-abate floor.
    """
    decay_rate = 0.09
    lower_bound = []

    for year in years:
        if year <= 2023:
            lower_bound.append(np.nan)  # Will use historical
        else:
            years_since_2023 = year - 2023
            emissions = HARD_TO_ABATE_TONNES + (emissions_2023 - HARD_TO_ABATE_TONNES) * np.exp(-decay_rate * years_since_2023)
            lower_bound.append(emissions)

    return lower_bound


def calculate_lower_bound_cdr(years):
    """
    Calculate lower bound CDR: hard-to-abate offset + legacy removal.
    Total = (HARD_TO_ABATE * 50 years) + LEGACY_TOTAL
    Distributed with exponential growth curve.
    """
    # Target total CDR over 2025-2100
    cumulative_target = (HARD_TO_ABATE_TONNES * 50) + LEGACY_TOTAL_TONNES

    # Generate exponential growth curve
    years_2025_2100 = np.array([y for y in years if 2025 <= y <= 2100])

    growth_rate = 0.08
    trial_cdr = []
    for year in years_2025_2100:
        years_since_2024 = year - 2024
        capacity = 1 - np.exp(-growth_rate * years_since_2024)
        trial_cdr.append(capacity)

    trial_cdr = np.array(trial_cdr)

    # Scale to hit cumulative target
    scaling_factor = cumulative_target / trial_cdr.sum()
    lower_bound_cdr = trial_cdr * scaling_factor

    # Create full array
    result = []
    cdr_idx = 0
    for year in years:
        if 2025 <= year <= 2100:
            result.append(lower_bound_cdr[cdr_idx])
            cdr_idx += 1
        else:
            result.append(0)

    return result


def calculate_upper_bound_cdr(df):
    """
    Calculate upper bound CDR: difference between emissions and 1.5°C pathway.
    """
    upper_cdr = []
    for _, row in df.iterrows():
        if row['year'] >= 2025 and not pd.isna(row['upper_bound_emissions']) and not pd.isna(row['pathway_1_5c']):
            cdr = row['upper_bound_emissions'] - row['pathway_1_5c']
            upper_cdr.append(max(cdr, 0))
        else:
            upper_cdr.append(0)

    return upper_cdr


# =======================
# VISUALIZATION FUNCTIONS
# =======================

def create_scenario_chart(df, emissions_col, cdr_col, title_text, emissions_label,
                          cdr_stacked=False, branding_text='', show_pathways=False):
    """
    Create visualization for CDR scenario.

    Args:
        df: DataFrame with emissions and CDR data
        emissions_col: Column name for projected emissions (e.g., 'lower_bound_emissions')
        cdr_col: Column name for CDR data (e.g., 'lower_bound_cdr')
        title_text: Title for the chart
        emissions_label: Label for emissions annotation
        cdr_stacked: If True, stack CDR as hard-to-abate + legacy. If False, single CDR bar.
        branding_text: Text for data source branding
        show_pathways: If True, plot 1.5°C and 2°C climate pathways

    Returns:
        Matplotlib figure
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))

    # Filter to 2000-2100
    df_plot = df[df['year'] >= 2000].copy()

    years = df_plot['year']
    bar_width = 0.8

    # Masks
    past_mask = years < 2025
    projected_mask = years >= 2025

    # Combine past and projected emissions
    emissions_gt = df_plot['past_emissions'].fillna(df_plot[emissions_col]) / 1e9
    cdr_gt = df_plot[cdr_col] / 1e9

    # Plot emissions bars
    if past_mask.any():
        ax.bar(years[past_mask], emissions_gt[past_mask],
               color='#C0392B', alpha=0.9, width=bar_width)

    if projected_mask.any():
        ax.bar(years[projected_mask], emissions_gt[projected_mask],
               color='#E74C3C', alpha=0.7, width=bar_width)

    # Plot CDR bars
    if projected_mask.any():
        if cdr_stacked:
            # Stacked: hard-to-abate + legacy
            hard_to_abate_gt = HARD_TO_ABATE_TONNES / 1e9
            hard_to_abate_removal = np.minimum(cdr_gt[projected_mask], hard_to_abate_gt)
            legacy_removal = cdr_gt[projected_mask] - hard_to_abate_removal

            ax.bar(years[projected_mask], -hard_to_abate_removal,
                   color='#8B4513', alpha=0.8, width=bar_width)
            ax.bar(years[projected_mask], -legacy_removal,
                   bottom=-hard_to_abate_removal,
                   color='#FF8C00', alpha=0.8, width=bar_width)
        else:
            # Single CDR bar
            ax.bar(years[projected_mask], -cdr_gt[projected_mask],
                   color='#3498DB', alpha=0.8, width=bar_width)

    # Plot climate pathways if requested
    if show_pathways:
        # Filter to projection years
        pathway_mask = years >= 2023

        # Plot 1.5°C pathway
        if 'pathway_1_5c' in df_plot.columns:
            pathway_1_5c_gt = df_plot['pathway_1_5c'] / 1e9
            ax.plot(years[pathway_mask], pathway_1_5c_gt[pathway_mask],
                   color='#27AE60', linewidth=3, linestyle='--',
                   label='1.5°C Pathway', alpha=0.8)

        # Plot 2°C pathway
        if 'pathway_2c' in df_plot.columns:
            pathway_2c_gt = df_plot['pathway_2c'] / 1e9
            ax.plot(years[pathway_mask], pathway_2c_gt[pathway_mask],
                   color='#F39C12', linewidth=3, linestyle='--',
                   label='2°C Pathway', alpha=0.8)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xlim(1999, 2101)

    max_emission = emissions_gt.max()
    max_removal = -cdr_gt.max() if cdr_gt.max() > 0 else -5
    y_max = max(max_emission * 1.1, 10)
    y_min = max(max_removal * 1.2, -25)
    ax.set_ylim(-30, 45)

    ax.set_ylabel('', fontsize=14, fontproperties=font_props.get('regular') if font_props else None)
    ax.set_xticks(range(2000, 2101, 50))
    ax.tick_params(axis='both', labelsize=12)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotations
    ax.text(2010, 40, 'PAST\nEMISSIONS',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#C0392B')

    # Get emissions value for 2045 (using the year mask)
    emissions_2060 = emissions_gt[years == 2060].iloc[0]
    ax.text(2060, emissions_2060 + 5, emissions_label,
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.8, color='#E74C3C')

    # CDR annotations
    if cdr_stacked:
        ax.text(2035, max_removal * 0.3, 'HARD-TO-ABATE\nEMISSIONS',
                fontsize=11, ha='center', va='center',
                fontweight='bold', alpha=0.8, color='#8B4513')
        ax.text(2070, max_removal * 0.6, 'LEGACY\nEMISSIONS',
                fontsize=11, ha='center', va='center',
                fontweight='bold', alpha=0.9, color='#FF8C00')
    else:
        ax.text(2075, max_removal - 3, 'CDR CAPACITY\nNEEDED',
                fontsize=12, ha='center', va='center',
                fontweight='bold', alpha=0.8, color='#3498DB',
                # bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                #          edgecolor='#3498DB', alpha=1)
                         )

    # Total CDR annotation
    total_cdr = cdr_gt[years >= 2025].sum()
    ax.text(2017, min(max_removal * 0.5, -2.5), f'{total_cdr:.0f} Gt CO\N{SUBSCRIPT TWO}\nREMOVED BY 2100',
            fontsize=11, ha='center', va='center',
            fontweight='bold', alpha=0.9, color='#2C3E50')

    # Bracket
    bracket_x = 2024
    bracket_top = -.5
    bracket_bottom = max_removal * .95
    ax.plot([bracket_x, bracket_x], [bracket_top, bracket_bottom],
            color='#E74C3C', linewidth=2, alpha=0.8)
    ax.plot([bracket_x, bracket_x + 0.25], [bracket_top, bracket_top],
            color='#E74C3C', linewidth=2, alpha=0.8)
    ax.plot([bracket_x, bracket_x + 0.25], [bracket_bottom, bracket_bottom],
            color='#E74C3C', linewidth=2, alpha=0.8)

    # Add title and branding
    format_plot_title(ax, "", title_text, None)
    add_deep_sky_branding(ax, None, branding_text)

    return fig


# =======================
# MAIN FUNCTION
# =======================

def main():
    print('='*60)
    print('CDR Bounds Analysis')
    print('='*60)

    # Load data
    df_hist = load_historical_emissions()
    df_pgr = load_pgr_projections()

    # Interpolate PGR data to fill missing years (2023-2050)
    df_pgr_interpolated = interpolate_pgr_emissions(df_pgr)

    # Project to 2100
    df_upper_2100 = project_upper_bound_to_2100(df_pgr_interpolated)
    df_pathways_2100 = project_pathways_to_2100(df_pgr_interpolated)

    # Combine all data
    years_all = list(range(2000, 2101))
    df_combined = pd.DataFrame({'year': years_all})

    # Merge historical
    df_combined = df_combined.merge(df_hist, on='year', how='left')

    # Merge interpolated PGR projections (2023-2050)
    df_combined = df_combined.merge(df_pgr_interpolated, on='year', how='left')

    # Merge 2051-2100 projections
    df_combined = df_combined.merge(df_upper_2100, on='year', how='left')
    df_combined = df_combined.merge(df_pathways_2100, on='year', how='left')

    # Fill upper_bound_emissions and pathway columns
    df_combined['upper_bound_emissions'] = df_combined['upper_bound_emissions_x'].fillna(df_combined['upper_bound_emissions_y'])
    df_combined['pathway_1_5c'] = df_combined['pathway_1_5c_x'].fillna(df_combined['pathway_1_5c_y'])
    df_combined['pathway_2c'] = df_combined['pathway_2c_x'].fillna(df_combined['pathway_2c_y'])
    df_combined = df_combined.drop(columns=['upper_bound_emissions_x', 'upper_bound_emissions_y',
                                            'pathway_1_5c_x', 'pathway_1_5c_y',
                                            'pathway_2c_x', 'pathway_2c_y'])

    # Calculate lower bound emissions
    emissions_2023 = df_hist[df_hist['year'] == 2023]['past_emissions'].values[0]
    df_combined['lower_bound_emissions'] = calculate_lower_bound_emissions(years_all, emissions_2023)

    # Calculate CDR capacities
    df_combined['lower_bound_cdr'] = calculate_lower_bound_cdr(years_all)
    df_combined['upper_bound_cdr'] = calculate_upper_bound_cdr(df_combined)

    # Save combined dataset
    output_csv = 'data/needed_removal_capacity/cdr_bounds_output.csv'
    df_combined.to_csv(output_csv, index=False)
    print(f'\n✓ Saved dataset to: {output_csv}')

    # Create visualizations
    print('\nCreating visualizations...')

    # Lower bound
    lower_fig = create_scenario_chart(
        df=df_combined,
        emissions_col='lower_bound_emissions',
        cdr_col='lower_bound_cdr',
        title_text="GLOBAL CO\N{SUBSCRIPT TWO} EMISSIONS & REMOVALS (GIGATONNES)",
        emissions_label="AGGRESSIVE EMISSIONS\nREDUCTION",
        cdr_stacked=False,
        branding_text=f"DATA: CDR PRIMER (2021). HARD-TO-ABATE = {HARD_TO_ABATE_TONNES/1e9:.1f} GT/YR, LEGACY = 65 GT TOTAL",
        show_pathways=False  # Set to True to show pathways, False to hide
    )
    save_plot(lower_fig, 'figures/lower_bound_cdr_scenario.png')
    print('  ✓ Saved lower_bound_cdr_scenario.png')

    # Upper bound
    upper_fig = create_scenario_chart(
        df=df_combined,
        emissions_col='upper_bound_emissions',
        cdr_col='upper_bound_cdr',
        title_text="GLOBAL CO\N{SUBSCRIPT TWO} EMISSIONS & REMOVALS (GIGATONNES)",
        emissions_label="CURRENT\nPOLICIES",
        cdr_stacked=False,
        branding_text="DATA: GLOBAL CARBON PROJECT (2024); SEI, CLIMATE ANALYTICS, IISD (2025) PGR",
        show_pathways=False  # Set to True to show pathways, False to hide
    )
    save_plot(upper_fig, 'figures/upper_bound_cdr_scenario.png')
    print('  ✓ Saved upper_bound_cdr_scenario.png')

    print('\n' + '='*60)
    print('✓ Complete!')
    print('='*60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
FaIR Climate Model - Hausfather's CO2 Pulse Approach

Replicates Zeke Hausfather's methodology using:
1. Calibrated constrained ensemble (841 members from fair-calibrate v1.4.1)
2. SSP2-4.5 baseline scenario
3. Perturbation method: difference between perturbed and unperturbed runs

Reference: https://www.linkedin.com/posts/zeke-hausfather-7327699_im-not-sure-many-folks-realize-just-how-activity-7369501799839395843-XoyY/
"""

import numpy as np
import pandas as pd
import sys
import os
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
import warnings
warnings.filterwarnings('ignore')

# Add path for visualization utilities
sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot, COLORS


def download_calibrated_data():
    """
    Download calibrated constrained ensemble data if not present.
    """
    import urllib.request

    base_url = "https://raw.githubusercontent.com/OMS-NetZero/FAIR/master/examples/data/calibrated_constrained_ensemble/"

    files = [
        "calibrated_constrained_parameters_calibration1.4.1.csv",
        "species_configs_properties_calibration1.4.1.csv"
    ]

    data_dir = "data/fair"
    os.makedirs(data_dir, exist_ok=True)

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                url = base_url + filename
                urllib.request.urlretrieve(url, filepath)
                print(f"  Downloaded to {filepath}")
            except Exception as e:
                print(f"  Error downloading {filename}: {e}")
                print(f"  You may need to download manually from: {url}")
                return False

    return True


def setup_hausfather_model(start_year=1750, end_year=3000, use_full_ensemble=False):
    """
    Initialize FaIR model with Hausfather's approach.

    Args:
        start_year: Start year (1750 for full calibration)
        end_year: End year for simulation
        use_full_ensemble: If True, use all 841 ensemble members (memory intensive)

    Returns:
        Initialized FAIR instance
    """
    # Initialize model
    f = FAIR(ch4_method="Thornhill2021")  # Required for calibrated ensemble

    # Define time period
    f.define_time(start_year, end_year, 1)

    # Define single ssp245 scenario
    # We'll run this model twice: once as baseline, once with pulse
    f.define_scenarios(['ssp245'])

    # Load calibrated configs
    try:
        params_file = 'data/fair/calibrated_constrained_parameters_calibration1.4.1.csv'
        df_configs = pd.read_csv(params_file, index_col=0)

        if use_full_ensemble:
            configs = df_configs.index.tolist()
            print(f"Using full ensemble: {len(configs)} members")
        else:
            # Use subset for memory efficiency (every 10th member for ~84 configs)
            configs = df_configs.index[::10].tolist()
            print(f"Using subset ensemble: {len(configs)} members")

        f.define_configs(configs)

    except FileNotFoundError:
        print("Calibrated ensemble files not found. Attempting to download...")
        if download_calibrated_data():
            # Retry after download
            params_file = 'data/fair/calibrated_constrained_parameters_calibration1.4.1.csv'
            df_configs = pd.read_csv(params_file, index_col=0)
            configs = df_configs.index[::10].tolist() if not use_full_ensemble else df_configs.index.tolist()
            f.define_configs(configs)
        else:
            raise FileNotFoundError("Could not load calibrated ensemble data")

    # Define species from calibrated properties
    species_file = 'data/fair/species_configs_properties_calibration1.4.1.csv'
    species, properties = read_properties(filename=species_file)
    f.define_species(species, properties)

    # Allocate arrays
    f.allocate()

    return f, params_file, species_file


def load_ssp_emissions(f):
    """
    Load SSP2-4.5 emissions data.
    """
    print("Loading SSP2-4.5 emissions data...")

    try:
        # Load SSP2-4.5 using RCMIP into 'ssp245' scenario
        f.fill_from_rcmip()
        print("Successfully loaded RCMIP emissions data for ssp245")

        # Clean NaNs in the loaded ssp245 data
        print("Cleaning NaN values...")
        nan_species = set()
        for specie in f.species:
            for config in f.configs:
                emissions = f.emissions.loc[dict(
                    scenario='ssp245',
                    specie=specie,
                    config=config
                )].values

                nan_count = np.isnan(emissions).sum()
                if nan_count > 0:
                    nan_species.add((specie, nan_count))
                    emissions[np.isnan(emissions)] = 0
                    fill(f.emissions, emissions,
                         scenario='ssp245', specie=specie, config=config)

        for specie, count in nan_species:
            print(f"  Filled {count} NaN values in {specie} with zeros")

        # Clean NaNs in forcing
        nan_forcing = set()
        for specie in f.species:
            for config in f.configs:
                forcing = f.forcing.loc[dict(
                    scenario='ssp245',
                    specie=specie,
                    config=config
                )].values

                nan_count = np.isnan(forcing).sum()
                if nan_count > 0:
                    nan_forcing.add((specie, nan_count))
                    forcing[np.isnan(forcing)] = 0
                    fill(f.forcing, forcing,
                         scenario='ssp245', specie=specie, config=config)

        for specie, count in nan_forcing:
            print(f"  Filled {count} NaN values in {specie} forcing with zeros")

        return True

    except Exception as e:
        print(f"Error loading RCMIP data: {e}")
        import traceback
        traceback.print_exc()
        return False


def add_pulse_to_emissions(f, pulse_year=2020, pulse_gt=40):
    """
    Add CO2 pulse to emissions.

    Args:
        f: FAIR instance with loaded SSP emissions
        pulse_year: Year to add pulse
        pulse_gt: Amount of CO2 in gigatonnes
    """
    print(f"Adding {pulse_gt} Gt CO2 pulse at year {pulse_year}...")

    # Find the time index for pulse year
    pulse_idx = np.where(f.timebounds[:-1] == pulse_year)[0]

    if len(pulse_idx) == 0:
        print(f"Warning: Pulse year {pulse_year} not found in timebounds")
        return False

    pulse_idx = pulse_idx[0]

    # Add pulse to CO2 FFI emissions
    co2_specie = 'CO2 FFI'

    for config in f.configs:
        current = f.emissions.loc[dict(
            scenario='ssp245',
            specie=co2_specie,
            config=config
        )].values.copy()

        current[pulse_idx] += pulse_gt

        fill(f.emissions, current,
             scenario='ssp245', specie=co2_specie, config=config)

    print(f"Successfully added pulse at index {pulse_idx}")
    return True


def configure_model(f, params_file, species_file):
    """
    Configure model with calibrated parameters.
    """
    print("Configuring model with calibrated parameters...")

    # Fill species configurations
    f.fill_species_configs(filename=species_file)

    # Override with calibrated climate parameters
    f.override_defaults(params_file)

    # Initialize forcing and other boundary conditions
    # Volcanic and solar forcing
    try:
        volcanic_solar_file = 'data/fair/volcanic_solar.csv'
        if os.path.exists(volcanic_solar_file):
            f.fill_from_csv(forcing_file=volcanic_solar_file)
    except:
        print("Note: Volcanic/solar forcing file not found, using defaults")


def run_model(f):
    """
    Run the FaIR model.
    """
    print("Initializing model state...")

    # Initialize concentrations at pre-industrial levels
    initialise(f.concentration, 278, specie='CO2')
    initialise(f.concentration, 722, specie='CH4')
    initialise(f.concentration, 270, specie='N2O')

    # Initialize other state variables
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    print("Running model...")
    f.run(progress=True)
    print("Simulation complete!")


def extract_pulse_effect(f_baseline, f_pulse, pulse_year=2020):
    """
    Extract PULSE EFFECT by differencing baseline and pulse model runs.

    Args:
        f_baseline: FAIR instance with baseline run (no pulse)
        f_pulse: FAIR instance with pulse run
        pulse_year: Year to start results from

    Returns:
        DataFrame with pulse-only temperature effect and uncertainty bounds
    """
    print("Extracting pulse effect (differencing model runs)...")

    # Get temperature data from both models
    temp_baseline = f_baseline.temperature.loc[dict(layer=0, scenario='ssp245')].values
    temp_pulse = f_pulse.temperature.loc[dict(layer=0, scenario='ssp245')].values

    # Calculate difference (this is the pulse-only effect!)
    temp_diff = temp_pulse - temp_baseline

    # Calculate percentiles across configs
    temp_median = np.percentile(temp_diff, 50, axis=1)
    temp_5th = np.percentile(temp_diff, 5, axis=1)
    temp_95th = np.percentile(temp_diff, 95, axis=1)

    # Get years
    years = f_baseline.timebounds[:-1] if len(f_baseline.timebounds) == len(temp_median) + 1 else f_baseline.timebounds[:len(temp_median)]

    # Create dataframe
    df = pd.DataFrame({
        'year': years.astype(int),
        'temp_median': temp_median,
        'temp_5th': temp_5th,
        'temp_95th': temp_95th
    })

    # Filter to years >= pulse_year for cleaner visualization
    df = df[df['year'] >= pulse_year].reset_index(drop=True)

    return df


def plot_hausfather_results(df_results, pulse_year=2020, pulse_gt=40, save_path=None):
    """
    Create Deep Sky visualization of Hausfather-style temperature response.
    """
    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))

    # Plot uncertainty range
    ax.fill_between(df_results['year'],
                     df_results['temp_5th'],
                     df_results['temp_95th'],
                     alpha=0.3,
                     color=COLORS['comparison'],
                     label='5th-95th percentile')

    # Plot median temperature response
    ax.plot(df_results['year'],
            df_results['temp_median'],
            color=COLORS['primary'],
            linewidth=3,
            label='Median response')

    # Add vertical line at pulse year
    ax.axvline(x=pulse_year, color='#666666', linestyle='--',
               linewidth=1.5, alpha=0.7)

    # Format axes
    font_prop = font_props.get('regular') if font_props else None
    ax.set_xlabel('YEAR', fontsize=16, fontproperties=font_prop)
    ax.set_ylabel('', fontsize=16, fontproperties=font_prop)

    # Set axis limits
    ax.set_ylim(bottom=0, top=0.032)
    ax.set_xlim(2000, 3000)

    # Format y-axis
    ax.yaxis.set_major_formatter('{x:.3f}')

    # Annotations
    ax.annotate(f'{pulse_gt} Gts OF CO₂\nEMITTED IN {pulse_year}',
                xy=(pulse_year + 60, 0.028),
                fontsize=11, ha='center', va='center',
                color='#666666', fontweight="bold")

    ax.annotate('MEDIAN WARMING',
                xy=(pulse_year + 580, 0.016),
                fontsize=11, ha='center', va='center',
                color=COLORS['primary'], fontweight="bold")

    ax.annotate('5TH-95TH PERCENTILE\nWARMING',
                xy=(pulse_year + 230, 0.0092),
                fontsize=11, ha='center', va='center',
                color=COLORS['comparison'], fontweight="bold")

    # Add title and subtitle
    title = ""
    subtitle = "GLOBAL TEMPERATURE CHANGE (°C)"
    format_plot_title(ax, title, subtitle, font_props)

    # Add branding
    data_note = "MODEL: FAIR V2.1 CALIBRATED ENSEMBLE | REPLICATED FROM ZEKE HAUSFATHER'S ANALYSIS"
    add_deep_sky_branding(ax, font_props, data_note=data_note)

    # Save
    save_plot(fig, save_path)

    return fig


def main():
    """Main execution function."""
    # Configuration
    START_YEAR = 1750  # Need full historical period for SSP scenarios
    END_YEAR = 3000
    PULSE_YEAR = 2020
    PULSE_GT = 40
    USE_FULL_ENSEMBLE = True  # Set to True for all 841 members (high memory)

    print("="*60)
    print("FaIR Climate Model - Hausfather Replication")
    print("="*60)
    print("Running TWO models: baseline (no pulse) and pulse")
    print("="*60)

    # ===== BASELINE MODEL (NO PULSE) =====
    print("\n### BASELINE MODEL (no pulse) ###")
    print(f"\n1. Setting up baseline model ({START_YEAR}-{END_YEAR})...")
    f_baseline, params_file, species_file = setup_hausfather_model(
        START_YEAR, END_YEAR, USE_FULL_ENSEMBLE
    )

    print("\n2. Configuring baseline model...")
    configure_model(f_baseline, params_file, species_file)

    print("\n3. Loading SSP2-4.5 emissions for baseline...")
    if not load_ssp_emissions(f_baseline):
        print("ERROR: Could not load SSP emissions data for baseline")
        return None

    print("\n4. Running baseline simulation...")
    run_model(f_baseline)

    # ===== PULSE MODEL (WITH PULSE) =====
    print("\n### PULSE MODEL (with pulse) ###")
    print(f"\n5. Setting up pulse model ({START_YEAR}-{END_YEAR})...")
    f_pulse, _, _ = setup_hausfather_model(
        START_YEAR, END_YEAR, USE_FULL_ENSEMBLE
    )

    print("\n6. Configuring pulse model...")
    configure_model(f_pulse, params_file, species_file)

    print("\n7. Loading SSP2-4.5 emissions for pulse model...")
    if not load_ssp_emissions(f_pulse):
        print("ERROR: Could not load SSP emissions data for pulse model")
        return None

    print(f"\n8. Adding {PULSE_GT} Gt pulse to {PULSE_YEAR}...")
    if not add_pulse_to_emissions(f_pulse, PULSE_YEAR, PULSE_GT):
        print("ERROR: Could not add pulse")
        return None

    print("\n9. Running pulse simulation...")
    run_model(f_pulse)

    # ===== DIFFERENCE THE TWO MODELS =====
    print("\n### EXTRACTING PULSE EFFECT ###")
    print("\n10. Extracting pulse effect (differencing models)...")
    df_results = extract_pulse_effect(f_baseline, f_pulse, PULSE_YEAR)

    # Save results
    output_file = 'data/fair/fair_pulse_results_hausfather.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("Pulse Effect Summary (Hausfather Method)")
    print("="*60)
    print(f"Isolated warming from {PULSE_GT} Gt CO₂ pulse in {PULSE_YEAR}")
    print("-"*60)

    # Show stats
    peak_temp = df_results['temp_median'].max()
    final_temp = df_results['temp_median'].iloc[-1]
    final_year = int(df_results['year'].iloc[-1])
    persistence = (final_temp / peak_temp * 100) if peak_temp > 0 else 0

    print(f"Peak warming (median):          {peak_temp:.4f}°C")
    print(f"Warming at year {final_year} (median): {final_temp:.4f}°C")
    print(f"Warming persistence:            {persistence:.1f}% of peak")
    print("="*60)

    # Create visualization
    print("\n11. Creating visualization...")
    figure_path = 'figures/fair_co2_pulse_temperature_hausfather.png'
    plot_hausfather_results(df_results, PULSE_YEAR, PULSE_GT, save_path=figure_path)
    print(f"Visualization saved to {figure_path}")

    return df_results


if __name__ == '__main__':
    df = main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import datetime

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot


# def load_co2_data(csv_path):
#     """
#     Load atmospheric CO2 data from Mauna Loa Observatory.
#     Returns annual average CO2 concentrations.
#     """
#     df = pd.read_csv(csv_path, comment='#')

#     # Group by year and take annual average
#     annual_df = df.groupby('year')['average'].mean().reset_index()
#     annual_df.columns = ['year', 'co2_ppm']

#     return annual_df


def model_cdr_scenarios(base_year_range=(0, 100)):
    """
    Model different CDR durability scenarios on CO2 trajectory.

    Scenarios:
    1. Baseline: Natural CO2 trajectory
    2. Nature-based CDR: 10-year permanence, applied in 2010
    3. Engineered CDR: 1000-year permanence, applied in 2005
    """
    # Filter to desired year range and extend for projections
    years = list(range(base_year_range[0], base_year_range[1] + 1))

    # Create extended dataframe
    result_df = pd.DataFrame({'year': years, 'co2_ppm': 0})

    last_year = result_df['year'].max()

    # Create baseline scenario (no CDR)
    result_df['baseline'] = result_df['co2_ppm'].copy()

    # CDR parameters
    cdr_reduction_ppm = 1.289  # Amount of CO2 PPM reduction from CDR intervention

    # Scenario 1: Nature-based CDR (starts at year 0, immediate vertical reduction then gradual release over 40 years)
    result_df['nature_cdr'] = result_df['baseline'].copy()

    # Apply nature-based CDR starting from year 0
    nature_start_year = 10
    nature_duration = 40
    # Apply permanent engineered CDR reduction from year 10
    engineered_start_year = 10

    for i, year in enumerate(years):
        if year >= nature_start_year and year <= nature_start_year + nature_duration:
            years_elapsed = year - nature_start_year

            # Immediate full reduction at start, then gradual release over 40 years
            reduction = cdr_reduction_ppm * (1 - years_elapsed / nature_duration)
            result_df.loc[result_df['year'] == year, 'nature_cdr'] -= reduction

    # Scenario 2: Engineered CDR (applied at year 10, permanent for 1000+ years)
    result_df['engineered_cdr'] = result_df['baseline'].copy()

    for i, year in enumerate(years):
        if year >= engineered_start_year:
            result_df.loc[result_df['year'] == year, 'engineered_cdr'] -= cdr_reduction_ppm
    return result_df


def model_temperature_scenarios(base_year_range=(0, 100)):
    """
    Model temperature response to CDR and emissions interventions.

    All interventions applied in 2005:
    - DAC: -1°C within 20 years, then constant
    - Emissions: +1°C within 20 years, then constant
    - Nature-based CDR: -1°C within 20 years, then gradual return to baseline over 20 years
    """
    years = list(range(base_year_range[0], base_year_range[1] + 1))
    result_df = pd.DataFrame({'year': years})

    # Initialize all scenarios at baseline (0°C impact)
    result_df['dac_temp'] = 0.0
    result_df['nature_temp'] = 0.0
    result_df['emissions_temp'] = 0.0

    dac_intervention_year = 10
    nature_intervention_year = 10
    response_duration = 20  # Years to reach full impact
    max_temp_change = 1.0   # °C
    decay_rate = 0.25  # Controls how quickly we approach the asymptote
    nature_duration = 40

    for i, year in enumerate(years):
        # DAC temperature impact (permanent cooling)
        if year >= dac_intervention_year:
            years_since_dac = year - dac_intervention_year
            result_df.loc[result_df['year'] == year, 'dac_temp'] = -max_temp_change * (1 - np.exp(-decay_rate * years_since_dac))

        # Emissions temperature impact (permanent warming - mirror of DAC)
        if year >= dac_intervention_year:
            years_since_emissions = year - dac_intervention_year
            result_df.loc[result_df['year'] == year, 'emissions_temp'] = max_temp_change * (1 - np.exp(-decay_rate * years_since_emissions))

        # Nature-based CDR temperature impact (follows DAC curve initially, then returns to 0 after 40 years)
        if year >= nature_intervention_year and year <= nature_intervention_year + nature_duration:
            years_since_nature = year - nature_intervention_year

            # Calculate the DAC-like cooling curve
            dac_like_cooling = -max_temp_change * (1 - np.exp(-decay_rate * years_since_nature))

            # After carbon starts being re-released, temperature gradually returns to baseline
            # Linear return to zero over the 40 year period
            retention_factor = 1 - (years_since_nature / nature_duration)

            temp_effect = dac_like_cooling * retention_factor

            result_df.loc[result_df['year'] == year, 'nature_temp'] = temp_effect
    return result_df


def create_ppm_impact_chart(df):
    """
    Create chart showing relative impact of CDR and emissions on atmospheric CO2 (difference from baseline).
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))

    # Calculate impacts relative to baseline
    nature_impact = df['nature_cdr'] - df['baseline']
    engineered_impact = df['engineered_cdr'] - df['baseline']

    # Filter to only plot from intervention year onwards (include one year before for vertical line)
    nature_start_year = 10
    engineered_start_year = 20

    # For nature-based: explicitly add point at (9, 0) for vertical line
    nature_mask = df['year'] >= nature_start_year
    nature_years = [nature_start_year - 1] + df.loc[nature_mask, 'year'].tolist()
    nature_values = [0] + nature_impact[nature_mask].tolist()

    # For engineered: explicitly add point at (19, 0) for vertical line
    engineered_mask = df['year'] >= engineered_start_year
    engineered_years = [engineered_start_year - 1] + df.loc[engineered_mask, 'year'].tolist()
    engineered_values = [0] + engineered_impact[engineered_mask].tolist()

    # Plot the impact scenarios
    ax.plot(nature_years, nature_values,
            color='#27AE60', linewidth=3, linestyle='--', alpha=0.8)

    ax.plot(engineered_years, engineered_values,
            color='#3498DB', linewidth=3, linestyle='-', alpha=0.8) 

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.5, .3)

    ax.set_xlabel('YEAR', fontsize=14,
                  fontproperties=font_props.get('regular') if font_props else None)

    # Y-axis formatting
    ax.set_ylabel('', fontsize=14,
                  fontproperties=font_props.get('regular') if font_props else None)

    # X-axis formatting
    ax.set_xticks(range(0, 101, 25))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add annotations to highlight key differences
    # Nature-based CDR annotation
    ax.annotate('CO₂ REMOVAL:\nLOW DURABILITY',
                xy=(10, -1.375),
                fontsize=11, ha='center', va='center', color='#27AE60',
                fontweight='bold')

    # Carbon release annotation for nature-based
    ax.annotate('CO₂ GRADUALLY\nRE-RELEASED',
                xy=(45, -.45),
                fontsize=11, ha='center', va='center', color='#27AE60',
                fontweight='bold')

    ax.annotate('40 YEAR\nPERMANENCE',
                xy=(30, 0.1),
                fontsize=11, ha='center', va='center', color='#27AE60',
                fontweight='bold')

    ax.plot([10, 50], [.05, .05],
        color='#27AE60', alpha=0.8)

    ax.plot([10, 10], [.07, 0.03],
        color='#27AE60', alpha=0.8)

    ax.plot([50, 50], [.07, 0.03],
        color='#27AE60', alpha=0.8)

    # Engineered CDR annotation
    ax.annotate('CO₂ REMOVAL:\nHIGH DURABILITY',
                xy=(25.3, -1.375),
                fontsize=11, ha='center', va='center', color='#3498DB',
                fontweight='bold')

    ax.annotate('1000+ YEAR\nPERMANENCE',
            xy=(70, -1.2),
            fontsize=11, ha='center', va='center', color='#3498DB',
            fontweight='bold')

    return fig


# def create_temperature_chart(temp_df):
#     """
#     Create chart showing temperature response to CDR and emissions interventions.
#     """
#     fig, ax, font_props = setup_enhanced_plot(figsize=(14, 10))

#     # Plot the temperature scenarios
#     ax.plot(temp_df['year'], temp_df['nature_temp'],
#             color='#27AE60', linewidth=3, linestyle='--',
#             label='Nature-based CDR', alpha=0.8)

#     ax.plot(temp_df['year'], temp_df['dac_temp'],
#             color='#3498DB', linewidth=3, linestyle='-',
#             label='Direct Air Capture', alpha=0.8)

#     ax.plot(temp_df['year'], temp_df['emissions_temp'],
#             color='#E74C3C', linewidth=3, linestyle='-',
#             label='CO₂ Emissions', alpha=0.8)

#     # Add zero line
#     ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

#     # Formatting
#     ax.set_xlim(0, 100.5)
#     ax.set_ylim(-1.2, 1.2)

#     # Y-axis formatting
#     ax.set_ylabel('', fontsize=14,
#                   fontproperties=font_props.get('regular') if font_props else None)

#     # X-axis formatting
#     ax.set_xticks(range(0, 101, 25))
#     ax.tick_params(axis='both', labelsize=12)
#     ax.grid(axis='y', alpha=0.3)

#     # Add annotations to highlight key differences
#     # Nature-based CDR annotation
#     ax.annotate('CO₂ REMOVAL:\nNATURE-BASED',
#                 xy=(28, -0.4),
#                 fontsize=11, ha='center', va='center', color='#27AE60',
#                 fontweight='bold')

#     # Carbon release annotation for nature-based
#     ax.annotate('COOLING REVERSED\nAS CO₂ GRADUALLY\nRE-RELEASED',
#                 xy=(50, -0.4),
#                 fontsize=11, ha='center', va='center', color='#27AE60',
#                 fontweight='bold')

#     ax.annotate('40 YEAR PERMANENCE',
#                 xy=(40, 0.1),
#                 fontsize=11, ha='center', va='center', color='#27AE60',
#                 fontweight='bold')
    
#     ax.plot([20, 60], [.05, .05],
#         color='#27AE60', alpha=0.8)
    
#     ax.plot([20, 20], [.025, 0.075],
#         color='#27AE60', alpha=0.8)

#     ax.plot([60, 60], [.025, 0.075],
#         color='#27AE60', alpha=0.8)


#     # DAC annotation
#     ax.annotate('CO₂ REMOVAL:\nENGINEERED',
#                 xy=(6, -0.4),
#                 fontsize=11, ha='center', va='center', color='#3498DB',
#                 fontweight='bold')

#     ax.annotate('1000+ YEAR\nCOOLING',
#                 xy=(70, -1.1),
#                 fontsize=11, ha='center', va='center', color='#3498DB',
#                 fontweight='bold')

#     # Emissions annotation
#     ax.annotate('CO₂ EMISSIONS',
#                 xy=(6, 0.4),
#                 fontsize=11, ha='center', va='center', color='#E74C3C',
#                 fontweight='bold')

#     ax.annotate('1000+ YEAR\nWARMING',
#                 xy=(70, 1.1),
#                 fontsize=11, ha='center', va='center', color='#E74C3C',
#                 fontweight='bold')

#     return fig


def create_mathieu_temp_chart(temp_df, scenario_df):
    """
    Create dual-panel chart showing temperature response (line) and carbon removed (bars).
    Top panel: Nature-based CDR
    Bottom panel: Engineered CDR
    Each panel has temperature on left y-axis and carbon removed on right y-axis.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Load font properties for consistency
    from utils import setup_enhanced_plot
    _, _, font_props = setup_enhanced_plot(figsize=(14, 10))
    plt.close()  # Close the temp figure from setup

    # Recalculate carbon removed (negative of impact)
    nature_carbon = -(scenario_df['nature_cdr'] - scenario_df['baseline'])
    engineered_carbon = -(scenario_df['engineered_cdr'] - scenario_df['baseline'])

    # Color scheme
    nature_color = '#27AE60'
    engineered_color = '#8B4513'
    line_color = '#2E4057'

    # ===== TOP PANEL: Nature-based CDR =====
    ax1_right = ax1.twinx()

    # Plot bars (carbon removed)
    bar_width = 0.8
    ax1_right.bar(scenario_df['year'], nature_carbon,
                  width=bar_width, color=nature_color, alpha=0.7, zorder=1)

    # Plot line (temperature response)
    ax1.plot(temp_df['year'], temp_df['nature_temp'],
             color=line_color, linewidth=3, linestyle='-', zorder=2)

    # Formatting for top panel
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax1.set_xlim(0, 25)
    ax1.set_ylim(-1.2, 1.2)
    ax1_right.set_ylim(0, 1.6)

    # Add vertical line at carbon removal event and label
    ax1.axvline(x=9.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax2.axvline(x=9.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(10, 1.25, 'CO₂ REMOVED', fontsize=11, ha='center', va='bottom',
             fontweight='bold', color='black')

    # Add y-axis labels and titles
    ax1.set_yticks([0])
    ax1_right.set_yticks([0])
    ax1.tick_params(axis='x', labelsize=11, length=0)  # Remove tick marks
    ax1.tick_params(axis='y', labelsize=10)
    ax1_right.tick_params(axis='y', labelsize=10)

    ax1.set_ylabel('TEMPERATURE RESPONSE', fontsize=11, fontweight='bold',
                   color='black')
    ax1_right.set_ylabel('CO₂ REMOVED AND STORED', fontsize=11, fontweight='bold',
                         color=nature_color)

    # Add + and - symbols on temperature response axis
    ax1.text(-0.01, 0.75, '+', transform=ax1.get_yaxis_transform(),
             fontsize=12, ha='right', va='center', fontweight='bold')
    ax1.text(-0.01, -0.75, '-', transform=ax1.get_yaxis_transform(),
             fontsize=12, ha='right', va='center', fontweight='bold')

    # Add label
    ax1.text(0.5, 0.95, 'LOW DURABILITY', transform=ax1.transAxes,
             fontsize=14, ha='center', va='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=line_color, linewidth=2))
    
    ax1.annotate('40-YEAR PERMANENCE',
            xy=(30, .6),
            fontsize=11, ha='center', va='center', color=nature_color,
            fontweight='bold')
    
    # ax1.plot([10, 50], [.83, .83],
    #     color=nature_color, alpha=0.8)
    
    # ax1.plot([10, 10], [.83, 1],
    #     color=nature_color, alpha=0.8)

    # ax1.plot([50, 50], [.83, 1],
    #     color=nature_color, alpha=0.8)
    
    ax1.annotate('COOLING LOST',
            xy=(50, -.3),
            fontsize=11, ha='center', va='center', color='black',
            fontweight='bold')
    
    ax1.annotate('AS CO₂ RE-RELEASED',
            xy=(50, -.75),
            fontsize=11, ha='center', va='center', color=nature_color,
            fontweight='bold')


    # ===== BOTTOM PANEL: Engineered CDR =====
    ax2_right = ax2.twinx()

    # Plot bars (carbon removed)
    ax2_right.bar(scenario_df['year'], engineered_carbon,
                  width=bar_width, color=engineered_color, alpha=0.7, zorder=1)

    # Plot line (temperature response)
    ax2.plot(temp_df['year'], temp_df['dac_temp'],
             color=line_color, linewidth=3, linestyle='-', zorder=2)

    # Formatting for bottom panel
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-1.2, 1.2)
    ax2_right.set_ylim(0, 1.6)

    ax2.set_xlabel('YEAR', fontsize=12,
                   fontproperties=font_props.get('regular') if font_props else None)

    ax2.set_xticks(range(0, 101, 25))
    ax2.tick_params(axis='x', labelsize=11, length=0)  # Remove tick marks

    # Add y-axis labels and titles
    ax2.set_yticks([0])
    ax2_right.set_yticks([0])
    ax2.tick_params(axis='y', labelsize=10)
    ax2_right.tick_params(axis='y', labelsize=10)

    ax2.set_ylabel('TEMPERATURE RESPONSE', fontsize=11, fontweight='bold',
                   color='black')
    ax2_right.set_ylabel('CO₂ REMOVED AND STORED', fontsize=11, fontweight='bold',
                         color=engineered_color)

    # Add + and - symbols on temperature response axis
    ax2.text(-0.01, 0.75, '+', transform=ax2.get_yaxis_transform(),
             fontsize=12, ha='right', va='center', fontweight='bold')
    ax2.text(-0.01, -0.75, '-', transform=ax2.get_yaxis_transform(),
             fontsize=12, ha='right', va='center', fontweight='bold')

    # Add label
    ax2.text(0.5, 0.95, 'HIGH DURABILITY', transform=ax2.transAxes,
             fontsize=14, ha='center', va='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=line_color, linewidth=2))
    
    ax2.annotate('1000+ YEAR PERMANENCE',
            xy=(75, .9),
            fontsize=11, ha='center', va='center', color=engineered_color,
            fontweight='bold',
            zorder=10)

    # Use ax2_right for annotation so it appears above the bars
    ax2_right.annotate('COOLING MAINTAINED',
        xy=(75, 2),  # Adjusted y-coordinate for right axis scale (0-12)
        fontsize=11, ha='center', va='center', color='black',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
            edgecolor='#2C3E50'),
        zorder=10)

    # Add legend in top right corner of the overall figure
    # from matplotlib.patches import Patch
    # from matplotlib.lines import Line2D

    # legend_elements = [
    #     Patch(facecolor=engineered_color, alpha=0.7, label='Durable CDR Tonnes Removed'),
    #     Patch(facecolor=nature_color, alpha=0.7, label='Nature-based CDR Tonnes Removed'),
    #     Line2D([0], [0], color=line_color, linewidth=3, label='Global Temperature Response')
    # ]

    # # Place legend on the top panel in upper right
    # ax1.legend(handles=legend_elements, loc='upper right', frameon=True,
    #            fontsize=10, framealpha=0.9)

    plt.tight_layout()

    return fig


def main():
    """
    Main function to create CDR durability visualization.
    """
    try:

        # Model CDR scenarios
        scenario_df = model_cdr_scenarios()
        print(f"Modeled CDR scenarios for {len(scenario_df)} years")

        # Create CO2 impact visualization
        fig1 = create_ppm_impact_chart(scenario_df)

        # Add titles and branding for CO2 plot
        format_plot_title(plt.gca(),
                         "",
                         "IMPACT ON ATMOSPERIC CO₂ PPM OF REMOVING 20 GIGATONNES CO₂",
                         None)

        add_deep_sky_branding(plt.gca(), None,
                             "ISOMETRIC'S 40-YEAR DURABILITY THRESHOLD AS LOW DURABILITY BASELINE")

        # Save CO2 plot
        save_path1 = 'figures/cdr_durability_ppm.png'
        os.makedirs('figures', exist_ok=True)
        save_plot(fig1, save_path1)

        print(f"CDR durability plot saved to {save_path1}")

        # Create temperature response visualization
        temp_df = model_temperature_scenarios()

        # fig2 = create_temperature_chart(temp_df)

        # # Add titles and branding for temperature plot
        # format_plot_title(plt.gca(),
        #                  "M",
        #                  "Global Temperature Response",
        #                  None)

        # add_deep_sky_branding(plt.gca(), None,
        #                      "Global temperature response to equal amounts of CO₂ emitted, removed with nature-based or engineered CDR.",
        #                      analysis_date=datetime.datetime.now())

        # # Save temperature plot
        # save_path2 = 'figures/cdr_durability_temp.png'
        # save_plot(fig2, save_path2)

        # print(f"CDR temperature plot saved to {save_path2}")

        # Create Mathieu temperature chart with dual axes
        fig3 = create_mathieu_temp_chart(temp_df, scenario_df)

        # Add titles and branding for Mathieu chart
        format_plot_title(fig3.axes[0],
                         "",
                         None,
                         None)

        add_deep_sky_branding(fig3.axes[1], None,
                             "ISOMETRIC'S 40-YEAR DURABILITY THRESHOLD AS LOW DURABILITY BASELINE")

        # Save Mathieu chart
        save_path3 = 'figures/cdr_durability_mathieu_temp.png'
        save_plot(fig3, save_path3)

        print(f"Mathieu temperature chart saved to {save_path3}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
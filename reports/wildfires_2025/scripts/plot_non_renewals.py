import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  
from matplotlib.offsetbox import OffsetImage, AnnotationBbox 
import os

# Import shared utilities
from utils import (
    setup_space_mono_font, process_whp_risk_data, add_risk_categories,
    setup_enhanced_plot, format_plot_title, add_deep_sky_branding,
    plot_risk_lines, add_legend, save_plot, calculate_policy_count_growth,
    plot_lines_by_risk_category, plot_line, calculate_average_premium_by_risk_category,
    filter_by_state_zip_codes,
    COLORS, RISK_COLORS
)

def process_ca_non_renewal_data(raw_ca_non_renewal_df):
    print("Processing non-renewal data...")
    clean_non_renewal_df = raw_ca_non_renewal_df.copy()

    numeric_cols = ['new', 'renewed', 'non_renewed']
    for col in numeric_cols:
        # Check if the column is already numeric
        if pd.api.types.is_numeric_dtype(clean_non_renewal_df[col]):
            # Already numeric, no conversion needed
            continue
        else:
            # It's a string, so filter and convert
            clean_non_renewal_df = clean_non_renewal_df[clean_non_renewal_df[col].str.contains(r'^[\d,]+$', regex=True, na=False)]
            clean_non_renewal_df[col] = clean_non_renewal_df[col].str.replace(',', '').astype(int)

    # Filter out zip codes with less than 5 policies in force
    clean_non_renewal_df['policies_in_force'] = clean_non_renewal_df['new'] + clean_non_renewal_df['renewed']
    clean_non_renewal_df = clean_non_renewal_df[clean_non_renewal_df['policies_in_force'] > 5]

    # Calculate non-renewal rate
    clean_non_renewal_df['non_renewal_rate'] = clean_non_renewal_df['non_renewed'] / (clean_non_renewal_df['new'] + clean_non_renewal_df['renewed']) * 100

    clean_non_renewal_df = clean_non_renewal_df[['zip', 'year', 'state', 'new', 'renewed', 'non_renewed', 'non_renewal_rate']]

    clean_non_renewal_df = filter_by_state_zip_codes(
        clean_non_renewal_df,
        state_code='CA',
        zip_column='zip'
    )

    return clean_non_renewal_df

if __name__ == "__main__":
    ca_df_1 = pd.read_excel('insurance/Residential-Property-Voluntary-Market-New-Renew-NonRenew-by-ZIP-2015-2021.xlsx')
    ca_df_1 = ca_df_1[ca_df_1['Year'] < 2020]
    ca_df_1['Non-Renewed'] = ca_df_1['Insured-Initiated Nonrenewed'] + ca_df_1['Insurer-Initiated Nonrenewed']
    ca_df_1 = ca_df_1[['ZIP Code', 'Year', 'New', 'Renewed', 'Non-Renewed']]

    ca_df_2 = pd.read_excel('insurance/Residential-Insurance-Voluntary-Market-New-Renew-NonRenew-by-ZIP-2020-2023.xlsx')
    ca_df_2 = ca_df_2[ca_df_2['Year'] >= 2020]
    ca_df_2 = ca_df_2[['ZIP Code', 'Year', 'New', 'Renewed', 'Non-Renewed']]

    ca_df = pd.concat([ca_df_1, ca_df_2], ignore_index=True)
    ca_df = ca_df.rename(columns={'ZIP Code': 'zip', 'Year': 'year', 'New': 'new', 'Renewed': 'renewed', 'Non-Renewed': 'non_renewed'})
    ca_df['zip'] = ca_df['zip'].astype(str).str.zfill(5)
    ca_df['state'] = 'CA'
    processed_ca_df = process_ca_non_renewal_data(ca_df)

    wa_df = pd.read_excel('insurance/WA_Request/Supporting_Underlying_Metrics_and_Disclaimer_for_Analyses_of_US_Homeowners_Insurance_Markets_2018-2022.xlsx', sheet_name='Supporting Underlying Metrics')
    wa_df['zip'] = wa_df['ZIP Code'].astype(str).str.zfill(5)
    wa_df['non_renewal_rate'] = wa_df['Nonrenewal Rate'] * 100
    wa_df['state'] = 'WA'
    wa_df['year'] = wa_df['Year'].astype(int)
    wa_df = wa_df[['zip', 'year', 'state', 'non_renewal_rate']]
    
    processed_wa_df = filter_by_state_zip_codes(
        wa_df,
        state_code='WA',
        zip_column='zip'
    )

    print(f"Processed California non-renewal data: {processed_ca_df.head()}")
    print(f"Processed Washington non-renewal data: {processed_wa_df.head()}")

    combined_non_renewal_df = pd.concat([processed_ca_df, processed_wa_df], ignore_index=True)

    whp_df = pd.read_csv('whp_fs/whp_clean.csv')
    ca_processed_whp_df = process_whp_risk_data(whp_df, state_code='CA')
    wa_processed_whp_df = process_whp_risk_data(whp_df, state_code='WA')
    combined_whp_df = pd.concat([ca_processed_whp_df, wa_processed_whp_df], ignore_index=True)
    
    non_renewal_whp_merged = pd.merge(combined_non_renewal_df, combined_whp_df, left_on='zip', right_on='zip', how='inner')
    non_renewal_whp_merged = add_risk_categories(non_renewal_whp_merged)

    non_renewals_grouped = non_renewal_whp_merged.groupby(
        ['year', 'risk_category', 'state'], 
        observed=False
        ).agg({
            'non_renewal_rate': 'mean',
            'new': 'sum',
            'renewed': 'sum',
            'non_renewed': 'sum'
        }).reset_index()
    non_renewals_grouped['year'] = non_renewals_grouped['year'].astype(int)
    non_renewals_grouped.to_csv('insurance/ca_wa_non_renewals_by_risk_category.csv', index=False)

    # Set up the plots
    ca_non_renewals_path = f'figures/ca_non_renewals.png'
    ca_non_renewals = plot_lines_by_risk_category(
        non_renewals_grouped[non_renewals_grouped['state'] == 'CA'],
        y_val='non_renewal_rate',
        title='INSURERS ARE LEAVING WILDFIRE PRONE AREAS OF CALIFORNIA',
        subtitle='NON RENEWAL RATE',
        unit='percent',
        data_note='DATA: CALIFORNIA DEPARTMENT OF INSURANCE, US FOREST SERVICE',
        save_path=ca_non_renewals_path,
        legend_placement='upper left',
        show_change_labels=False
    )

    wa_non_renewals_path = f'figures/wa_non_renewals.png'
    wa_non_renewals = plot_lines_by_risk_category(
        non_renewals_grouped[non_renewals_grouped['state'] == 'WA'],
        y_val='non_renewal_rate',
        title='INSURERS ARE LEAVING WILDFIRE PRONE AREAS OF WASHINGTON',
        subtitle='NON RENEWAL RATE',
        unit='percent',
        data_note='DATA: WASHINGTON STATE OFFICE OF THE INSURANCE COMMISSIONER, US FOREST SERVICE',
        save_path=wa_non_renewals_path,
        legend_placement='upper left',
        show_change_labels=False
    )

    ax = wa_non_renewals.axes[0]
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    save_plot(wa_non_renewals, wa_non_renewals_path)

    # =====================
    # FRENCH LANGUAGE CHARTS
    # =====================
    print("\nGenerating French language non-renewal charts...")

    french_risk_translations = {
        'Extreme Fire Risk': 'Risque d\'Incendie Extrême',
        'High Fire Risk': 'Risque d\'Incendie Élevé', 
        'Low Fire Risk': 'Risque d\'Incendie Faible'
    }

    # French version of CA non-renewals chart
    ca_non_renewals_path_fr = f'figures/ca_non_renewals_fr.png'
    ca_non_renewals_fr = plot_lines_by_risk_category(
        non_renewals_grouped[non_renewals_grouped['state'] == 'CA'],
        y_val='non_renewal_rate',
        title='LES ASSUREURS QUITTENT LES ZONES À RISQUE D\'INCENDIE DE CALIFORNIE',
        subtitle='TAUX DE NON-RENOUVELLEMENT',
        unit='percent',
        data_note='DONNÉES: CALIFORNIA DEPARTMENT OF INSURANCE, US FOREST SERVICE',
        save_path=ca_non_renewals_path_fr,
        legend_placement='upper left',
        show_change_labels=False,
        risk_category_translations=french_risk_translations
    )

    # French version of WA non-renewals chart
    wa_non_renewals_path_fr = f'figures/wa_non_renewals_fr.png'
    wa_non_renewals_fr = plot_lines_by_risk_category(
        non_renewals_grouped[non_renewals_grouped['state'] == 'WA'],
        y_val='non_renewal_rate',
        title='LES ASSUREURS QUITTENT LES ZONES À RISQUE D\'INCENDIE DE WASHINGTON',
        subtitle='TAUX DE NON-RENOUVELLEMENT',
        unit='percent',
        data_note='DONNÉES: WASHINGTON STATE OFFICE OF THE INSURANCE COMMISSIONER, US FOREST SERVICE',
        save_path=wa_non_renewals_path_fr,
        legend_placement='upper left',
        show_change_labels=False,
        risk_category_translations=french_risk_translations
    )

    ax_fr = wa_non_renewals_fr.axes[0]
    ax_fr.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    save_plot(wa_non_renewals_fr, wa_non_renewals_path_fr)

    print("French language non-renewal charts generated successfully!")



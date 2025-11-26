import sys
from pathlib import Path
dashboard_dir = Path(__file__).parent.parent
sys.path.append(str(dashboard_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter
import xarray as xr
import json
import datetime
import warnings
import logging

from utils.logging_utils import setup_logging
from utils.paths import DATA_DIR, RAW_DIR, PROCESSED_DIR

WORKING_DIR = RAW_DIR / 'home_insurance_exploration'

# Set up plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def load_data(path):
    """Load raw CA home insurance data"""
    raw_ca_data = pd.read_csv(path)
    return raw_ca_data

def process_data(raw_ca_data):
    """Process the insurance data for analysis"""
    # Drop rows with NaN year values
    df = raw_ca_data.copy()
    df = df[~df['Year'].isna()]
    
    # Convert Year to integer for proper sorting
    df['Year'] = df['Year'].astype(int)
    
    # Convert monetary values to float
    numeric_columns = ['Written Premiums', 'Written Exposures', 'Average Written Premiums']
    for col in numeric_columns:
        df[col] = df[col].astype(float)
    
    # Create a clean dataframe excluding totals
    clean_df = df[~df['Policy Type'].str.contains('Total', na=False)]
    
    # Load CPI data for inflation adjustment
    cpi_data_path = WORKING_DIR / 'cpi.csv'
    cpi_df = pd.read_csv(cpi_data_path)
    
    # Ensure year is treated as int
    cpi_df['year'] = cpi_df['year'].astype(int)
    
    # Set latest year for normalization (use 2021 or the most recent year in the insurance data)
    latest_year = max(clean_df['Year'].max(), 2021)
    reference_cpi = cpi_df[cpi_df['year'] == latest_year]['cpi'].values[0]
    
    # Merge CPI data with insurance data
    clean_df = clean_df.merge(cpi_df, left_on='Year', right_on='year', how='left')
    
    # Drop any rows where CPI data is missing
    clean_df = clean_df.dropna(subset=['cpi'])
    
    # Calculate inflation adjustment factor (normalize to latest year)
    clean_df['Inflation_Factor'] = reference_cpi / clean_df['cpi']
    
    # Calculate inflation-adjusted premiums
    clean_df['Adjusted_Avg_Premium'] = clean_df['Average Written Premiums'] * clean_df['Inflation_Factor']
    
    # Clean up merged dataframe by dropping the redundant 'year' column
    clean_df = clean_df.drop(columns=['year'])
    
    return clean_df

def generate_summary_stats(clean_df):
    """Generate summary statistics for the insurance data"""
    # Annual growth rates by policy type
    policy_types = clean_df['Policy Type'].unique()
    summary_stats = []
    
    for policy in policy_types:
        policy_data = clean_df[clean_df['Policy Type'] == policy].sort_values('Year')
        
        # Skip if not enough data points
        if len(policy_data) < 2:
            continue
            
        # Calculate annual growth rates
        policy_data['Growth_Rate'] = policy_data['Average Written Premiums'].pct_change() * 100
        policy_data['Adjusted_Growth_Rate'] = policy_data['Adjusted_Avg_Premium'].pct_change() * 100
        
        # Calculate statistics
        stats = {
            'Policy Type': policy,
            'Earliest Year': policy_data['Year'].min(),
            'Latest Year': policy_data['Year'].max(),
            'Start Premium': policy_data.loc[policy_data['Year'].idxmin(), 'Average Written Premiums'],
            'End Premium': policy_data.loc[policy_data['Year'].idxmax(), 'Average Written Premiums'],
            'Total Growth %': ((policy_data.loc[policy_data['Year'].idxmax(), 'Average Written Premiums'] / 
                              policy_data.loc[policy_data['Year'].idxmin(), 'Average Written Premiums']) - 1) * 100,
            'Inflation Adjusted Total Growth %': ((policy_data.loc[policy_data['Year'].idxmax(), 'Adjusted_Avg_Premium'] / 
                              policy_data.loc[policy_data['Year'].idxmin(), 'Adjusted_Avg_Premium']) - 1) * 100,
            'Avg Annual Growth Rate %': policy_data['Growth_Rate'].mean(),
            'Max Annual Growth Rate %': policy_data['Growth_Rate'].max(),
            'Std Dev Growth Rate': policy_data['Growth_Rate'].std()
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Additional overall statistics
    overall_stats = clean_df.groupby('Year').agg({
        'Average Written Premiums': 'mean',
        'Written Premiums': 'sum',
        'Written Exposures': 'sum'
    }).reset_index()
    
    overall_stats['Avg_Premium_Weighted'] = overall_stats['Written Premiums'] / overall_stats['Written Exposures']
    
    return summary_df, overall_stats

def create_exploratory_plots(clean_df, overall_stats, outdir):
    """Create exploratory plots for the insurance data"""
    # Make sure output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    # 1. Time series of average premiums by policy type
    plt.figure(figsize=(14, 8))
    for policy in clean_df['Policy Type'].unique():
        policy_data = clean_df[clean_df['Policy Type'] == policy].sort_values('Year')
        plt.plot(policy_data['Year'], policy_data['Average Written Premiums'], marker='o', linewidth=2, label=policy)
    
    plt.title('Average Written Premium by Policy Type Over Time', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Average Premium ($)')
    plt.xticks(clean_df['Year'].unique())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Policy Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outdir / 'avg_premium_by_policy_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Inflation-adjusted premiums
    plt.figure(figsize=(14, 8))
    for policy in clean_df['Policy Type'].unique():
        policy_data = clean_df[clean_df['Policy Type'] == policy].sort_values('Year')
        plt.plot(policy_data['Year'], policy_data['Adjusted_Avg_Premium'], marker='o', linewidth=2, label=policy)
    
    plt.title('Inflation-Adjusted Average Premium by Policy Type (2021 Dollars)', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Adjusted Average Premium ($)')
    plt.xticks(clean_df['Year'].unique())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Policy Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outdir / 'inflation_adjusted_premium.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Year-over-year growth rates
    growth_data = clean_df.copy()
    growth_data['Growth_Rate'] = growth_data.groupby('Policy Type')['Average Written Premiums'].pct_change() * 100
    
    plt.figure(figsize=(14, 8))
    for policy in growth_data['Policy Type'].unique():
        policy_data = growth_data[(growth_data['Policy Type'] == policy) & (~pd.isna(growth_data['Growth_Rate']))].sort_values('Year')
        if len(policy_data) > 0:
            plt.plot(policy_data['Year'], policy_data['Growth_Rate'], marker='o', linewidth=2, label=policy)
    
    plt.title('Year-over-Year Growth Rate in Average Premiums', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Annual Growth Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.legend(title='Policy Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outdir / 'annual_growth_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Total premiums over time (all policy types)
    plt.figure(figsize=(14, 8))
    plt.plot(overall_stats['Year'], overall_stats['Written Premiums'] / 1e9, marker='o', linewidth=3, color='darkblue')
    plt.title('Total Written Premiums Over Time (All Policy Types)', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Total Written Premiums (Billion $)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(outdir / 'total_written_premiums.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Heatmap of premium growth by policy type and year
    try:
        # Create a pivot table for the heatmap
        heatmap_data = growth_data.pivot_table(
            index='Policy Type', 
            columns='Year', 
            values='Growth_Rate',
            aggfunc='mean'  # Use mean for growth rates
        )
        
        if not heatmap_data.empty:
            plt.figure(figsize=(14, 8))
            sns.heatmap(heatmap_data, cmap='RdYlGn_r', center=0, annot=True, fmt='.1f', 
                       cbar_kws={'label': 'Growth Rate (%)'})
            plt.title('Annual Premium Growth Rate by Policy Type (%)', fontsize=16)
            plt.tight_layout()
            plt.savefig(outdir / 'premium_growth_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logger.warning("No data available for growth rate heatmap")
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        
    # 7. Compare inflation-adjusted vs nominal premium growth (NEW)
    try:
        # Calculate average premiums across all policy types by year
        yearly_avg = clean_df.groupby('Year').agg({
            'Average Written Premiums': 'mean',
            'Adjusted_Avg_Premium': 'mean'
        }).reset_index()
        
        # Create comparison plot
        plt.figure(figsize=(12, 7))
        plt.plot(yearly_avg['Year'], yearly_avg['Average Written Premiums'], 
                marker='o', linewidth=2, label='Nominal Premiums', color='blue')
        plt.plot(yearly_avg['Year'], yearly_avg['Adjusted_Avg_Premium'], 
                marker='s', linewidth=2, label='Inflation-Adjusted Premiums (2021$)', 
                color='red', linestyle='--')
        
        plt.title('Nominal vs. Inflation-Adjusted Average Premiums', fontsize=16)
        plt.xlabel('Year')
        plt.ylabel('Average Premium ($)')
        plt.xticks(yearly_avg['Year'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / 'nominal_vs_adjusted_premiums.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error creating premium comparison plot: {e}")

if __name__ == '__main__':    
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting home insurance exploration analysis")
    
    # Load and process data
    raw_ca_data_path = WORKING_DIR / 'ca_home_insurance_raw.csv'
    logger.info(f"Loading data from {raw_ca_data_path}")
    raw_df = load_data(raw_ca_data_path)
    
    logger.info("Processing insurance data")
    clean_df = process_data(raw_df)
    
    logger.info("Generating summary statistics")
    summary_stats, overall_stats = generate_summary_stats(clean_df)
    summary_stats.to_csv(WORKING_DIR / 'summary_stats.csv', index=False)
    overall_stats.to_csv(WORKING_DIR / 'overall_stats.csv', index=False)
    
    logger.info("Creating exploratory plots")
    create_exploratory_plots(clean_df, overall_stats, WORKING_DIR)
    
    logger.info("Analysis complete. Results saved to: %s", WORKING_DIR)


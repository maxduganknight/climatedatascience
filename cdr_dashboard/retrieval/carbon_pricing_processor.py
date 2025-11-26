import sys
from pathlib import Path
# Add dashboard directory to Python path
cdr_dashboard_dir = Path(__file__).parent.parent
if str(cdr_dashboard_dir) not in sys.path:
    sys.path.append(str(cdr_dashboard_dir))

import pandas as pd
import os
import json 
from datetime import datetime
from utils.logging_utils import setup_logging
from utils.paths import RAW_DIR, PROCESSED_DIR, PROJECT_ROOT
from utils.retrieval_utils import save_dataset, load_dataset_config

# Setup logger
logger = setup_logging()

now = datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
today = now.strftime("%Y%m%d")

def process_carbon_pricing_data(df):
    # Skip the first row (row 0) which contains the "Data last updated..." text
    # Use the second row (row 1) as the header
    # df = pd.read_excel(world_bank_excel_page, sheet_name='Compliance_Gen Info', skiprows=1)
    
    # Now filter to the columns we want
    relevant_cols = ['Unique ID', 'Instrument name', 'Type', 'Status', 'Share of global emissions covered']
    df = df[relevant_cols].copy()

    df.loc[:, 'Share of global emissions covered'] = df['Share of global emissions covered'] * 100 * 100  # Convert to percentage

    # Filter for national policies - make a copy to avoid the warning
    national_policies = df[~df['Type'].str.contains('Subnational', na=False)].copy()
    
    # Filter for in-force policies - make a copy to avoid the warning
    in_force_policies = national_policies[
        (national_policies['Status'].str.contains('Implemented', na=False)) &
        (~national_policies['Status'].str.contains('Abolished', na=False))
    ].copy()
    
    # Now add the new column using .loc to avoid the warning
    in_force_policies.loc[:, 'implemented_year'] = in_force_policies['Status'].str.extract(r'(\d{4})').astype(int)
    
    # Clean up emissions data: convert to numeric and handle percentage format
    in_force_policies.loc[:, 'Share of global emissions covered'] = pd.to_numeric(
        in_force_policies['Share of global emissions covered'].astype(str).str.replace('%', '').str.strip(),
        errors='coerce'
    ) / 100  # Convert percentage to decimal
    
    # Group by country and get the earliest implementation year
    countries_df = in_force_policies.groupby('Unique ID').agg({
        'implemented_year': 'min',
        'Share of global emissions covered': 'first'  # Take the first emissions value for each country
    }).reset_index()
    
    # Fixed year range from 2010 to 2024, regardless of when earliest implementation occurred
    years = list(range(2010, 2025))
    
    # Create empty dataframe for cumulative counts
    cumulative_countries_df = pd.DataFrame({'year': years})
    
    # Calculate cumulative count by year and emissions coverage
    cumulative_counts = []
    cumulative_emissions = []
    
    for year in years:
        # Filter countries with carbon pricing by this year
        countries_by_year = countries_df[countries_df['implemented_year'] <= year]
        
        # Count countries
        count = len(countries_by_year)
        cumulative_counts.append(count)
        
        # Sum emissions percentages
        emissions_pct = countries_by_year['Share of global emissions covered'].sum()
        cumulative_emissions.append(emissions_pct)
    
    cumulative_countries_df['countries_count'] = cumulative_counts
    cumulative_countries_df['global_emissions_pct'] = cumulative_emissions
    
    # Optional: Round the emissions percentage to 3 decimal places for readability
    cumulative_countries_df['global_emissions_pct'] = cumulative_countries_df['global_emissions_pct'].round(2)
    
    return cumulative_countries_df

if __name__ == "__main__":
    logger.info("Starting local data processing using configuration...")

    # Ensure the PROCESSED_DIR exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        datasets_config = load_dataset_config()
        # Find the carbon_pricing config in the list
        carbon_pricing_config = next((config for config in datasets_config if config['name'] == 'carbon_pricing'), None)
        
        if not carbon_pricing_config:
            raise ValueError("Carbon pricing configuration not found in dataset config")
            
        raw_file_path = RAW_DIR / carbon_pricing_config['raw_filename']
        df = pd.read_excel(raw_file_path, sheet_name='Compliance_Gen Info', skiprows=1)
        processed_data = process_carbon_pricing_data(df)
        
        # Save the main carbon pricing data
        output_base = carbon_pricing_config['output_filename_base']
        output_path = save_dataset(processed_data, output_base, is_local=True)
        logger.success(f"Saved processed carbon pricing data to {output_path}")        
    except Exception as e:
        logger.error(f"Failed to load dataset configuration. Exiting. Error: {e}")
        sys.exit(1) # Exit if config cannot be loaded
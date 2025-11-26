import requests
import pandas as pd
import time
import math
import sys
from typing import Optional, Dict, List
from pathlib import Path

sys.path.append('/Users/max/Deep_Sky')
from creds import CENSUS_API_KEY

'''
Script for pulling US Census ACS data at the county level for multiple years.
Currently pulls median home value for 2023, 2018, and 2013.
https://www.census.gov/programs-surveys/acs
'''

# B25077_001E is the ACS variable for "Median Value (Dollars) of Owner-Occupied Housing Units"
# B19013_001E is the ACS variable for "Median Household Income in the Past 12 Months (in 2019 Inflation-Adjusted Dollars)"
# B19301_001E is the ACS variable for "Per Capita Income in the Past 12 Months (in 2019 Inflation-Adjusted Dollars)"
# Constants

CENSUS_VARS = {
    'B25077_001E': 'median_home_value',
    # 'B19013_001E': 'median_household_income',
    # 'B19301_001E': 'per_capita_income',
    # 'B01003_001E': 'population'
}

YEARS = [2023, 2018, 2013]
MISSING_VALUE = '-666666666'

def query_census_api(query: str, key: str = CENSUS_API_KEY, retries: int = 3, delay: int = 5) -> Optional[Dict]:
    query = f"{query}&key={key}"
    for attempt in range(retries):
        try:
            response = requests.get(query)
            response.raise_for_status()
            if response.status_code == 204:
                return None
            data = response.json()
            if len(data) <= 1:
                return None
            return data
        except requests.RequestException as e:
            print(f"Warning: Request failed for query {query} (attempt {attempt + 1}/{retries})")
            time.sleep(delay)
    return None

def parse_census_value(value: Optional[str]) -> float:
    """Parse Census value, converting missing values to NaN"""
    if value is None or value == MISSING_VALUE:
        return math.nan
    return float(value)

def build_county_df() -> pd.DataFrame:
    """Build DataFrame with county-level Census data across multiple years"""
    # Initialize empty DataFrame
    county_data = []
    
    for year in YEARS:
        print(f"\nProcessing year {year}")
        base_url = f"https://api.census.gov/data/{year}/acs/acs5"
        vars_str = ','.join(CENSUS_VARS.keys())
        
        # Query county-level data
        county_query = f"{base_url}?get=NAME,{vars_str}&for=county:*"
        county_response = query_census_api(county_query)
        
        if county_response is None:
            print(f"Warning: No data returned for {year}")
            continue
            
        # Process each county
        for entry in county_response[1:]:  # Skip header row
            county_name = entry[0].split(",")[0].strip()
            state_name = entry[0].split(",")[1].strip()
            home_value = parse_census_value(entry[1])
            state_fips = entry[-2]
            county_fips = entry[-1]
            
            county_data.append({
                'county_fips': county_fips,
                'county_name': county_name,
                'state_fips': state_fips,
                'state_name': state_name,
                f'home_value_{year}': home_value
            })
            
        time.sleep(0.25)  # Rate limiting
    
    # Convert to DataFrame
    df = pd.DataFrame(county_data)
    
    # Deduplicate counties by grouping
    df = df.groupby(['county_fips', 'county_name', 'state_fips', 'state_name']).first().reset_index()
    df = df.drop(['county_clean', 'state_clean'], axis=1)

    # Print a sample of county names and their last words
    print("Sample of county names and their last words:")
    print(df['county_name'].head(10).apply(lambda x: f"{x} -> last word: '{x.split()[-1]}'"))

    # Check for exact string matches
    print("\nRows with 'city' in county name:")
    print(df[df['county_name'].str.contains('city', case=False)])

    # Try different filtering approaches
    df_filtered = df[~df['county_name'].str.contains('city', case=False)]

    # Print before and after counts
    print(f"\nBefore filtering: {len(df)} rows")
    print(f"After filtering: {len(df_filtered)} rows")

    # Look at the structure of a problematic row
    print("\nExample of a 'city' row:")
    print(df[df['county_name'].str.contains('city', case=False)].iloc[0])

    df = df[~df['county_name'].str.split().str[-1].str.lower().eq('city')]

    df['home_value_change_2018_2023'] = df['home_value_2023'] - df['home_value_2018']
    df['home_value_pct_change_2018_2023'] = df['home_value_change_2018_2023'] / df['home_value_2018']

    return df

def save_county_data(df: pd.DataFrame) -> None:
    """Save county DataFrame to CSV file"""
    if not df.empty:
        output_file = 'county_home_values.csv'
        
        df.to_csv(output_file, index=False)
        print(f"\nCounty data saved to '{output_file}'")
        print(f"Shape: {df.shape}")
        print("\nPreview:")
        print(df.head())

if __name__ == "__main__":

    df = build_county_df()
    save_county_data(df)

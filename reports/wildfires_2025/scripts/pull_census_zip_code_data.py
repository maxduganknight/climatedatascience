import requests
import json
import argparse
import time
import math
import sys
import pandas as pd
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # For progress bars

sys.path.append('/Users/max/Deep_Sky')
from creds import CENSUS_API_KEY

'''
Script for pulling US Census ACS data at the ZIP Code Tabulation Area (ZCTA) level
for Texas and California. Retrieves population, households, median income, and home values.
https://www.census.gov/programs-surveys/geography/guidance/geo-areas/zctas.html
'''

# Census API variable codes and their friendly names
CENSUS_VARS = {
    'B01003_001E': 'population',        # Total population
    'B11001_001E': 'households',        # Total households
    'B19013_001E': 'median_household_income',  # Median household income
    'B25077_001E': 'median_home_value'  # Median home value
}

# State FIPS codes
STATE_FIPS = {
    'TX': '48',
    'CA': '06'
}

# ZIP code prefixes by state for filtering
STATE_ZIP_PREFIXES = {
    # Texas ZIP codes (primarily start with 7 and 8)
    '48': ['75', '76', '77', '78', '79'],
    
    # California ZIP codes (start with 9)
    '06': ['90', '91', '92', '93', '94', '95', '96']
}

# Constants
MISSING_VALUE = '-666666666'
DATA_DIR = 'census_data'
TEST_DIR = 'testing_data'

# Available ACS years (2009-2023)
AVAILABLE_YEARS = list(map(str, range(2009, 2024)))

def query_census_api(query: str, key: str = CENSUS_API_KEY, retries: int = 3, delay: int = 5) -> Optional[Dict]:
    """
    Query the Census API with retries and error handling
    
    Args:
        query: Base API URL without the key
        key: Census API key
        retries: Number of retry attempts
        delay: Delay between retries in seconds
    
    Returns:
        API response data as dictionary, or None if request failed
    """
    full_query = f"{query}&key={key}"
    
    for attempt in range(retries):
        try:
            response = requests.get(full_query)
            
            # If success, return the data
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    print(f"Successfully retrieved data with {len(data)-1} records")
                    return data
                else:
                    print("Warning: API returned empty dataset")
                    return None
            
            # Handle specific error codes
            elif response.status_code == 400:
                print(f"Error 400: Bad Request - The API request format is invalid")
            elif response.status_code == 404:
                print(f"Error 404: Not Found - The requested endpoint doesn't exist")
            elif response.status_code == 429:
                print(f"Error 429: Too Many Requests - Rate limit exceeded")
                # For rate limiting, increase the delay
                time.sleep(delay * 2)
            else:
                print(f"Error {response.status_code}: {response.reason}")
                
            # Show response text for debugging if available
            if response.text:
                print(f"Response text: {response.text[:200]}...")  # Show first 200 chars
                
            # Wait before retry
            time.sleep(delay)
            
        except requests.RequestException as e:
            print(f"Request error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    
    # If all attempts failed
    print("Error: All API request attempts failed")
    return None

def parse_census_value(value: Optional[str]) -> float:
    """Parse Census value, converting missing values to NaN"""
    if value is None or value == MISSING_VALUE or value == '':
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan

def get_zcta_data_from_census(variables: List[str], year: str = "2022", verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Fetch all ZCTA data from the Census API
    
    Args:
        variables: List of Census API variable codes
        year: Census ACS year
        verbose: Whether to print detailed progress
    
    Returns:
        DataFrame containing all ZCTA data or None if request failed
    """
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    vars_str = ",".join(variables)
    
    # Use the format that worked: zip code tabulation area
    api_url = f"{base_url}?get=NAME,{vars_str}&for=zip%20code%20tabulation%20area:*"
    
    if verbose:
        print(f"Fetching nationwide ZIP code data from {year} ACS...")
    
    # Get data from Census API
    zcta_data = query_census_api(api_url)
    if zcta_data is None:
        return None
        
    # Convert to DataFrame
    columns = zcta_data[0]
    data = zcta_data[1:]
    
    df = pd.DataFrame(data, columns=columns)
    
    if verbose:
        print(f"Retrieved {len(df)} total ZCTAs.")
    
    # Verify we have the ZIP code column
    zip_column = 'zip code tabulation area'
    if zip_column not in df.columns:
        print(f"Error: Could not find ZIP code column. Available columns: {df.columns.tolist()}")
        return None
        
    # Create a clean dataframe
    result_df = pd.DataFrame()
    result_df['zip_code'] = df[zip_column]
    result_df['name'] = df['NAME'].apply(lambda x: x.replace('ZCTA5 ', '') if 'ZCTA5 ' in x else x)
    
    # Add year column
    result_df['year'] = year
    
    # Parse all requested variables
    for var in variables:
        var_name = CENSUS_VARS.get(var, var)
        if var in df.columns:
            result_df[var_name] = df[var].apply(parse_census_value)
        else:
            if verbose:
                print(f"Warning: Variable {var} not found in response")
            result_df[var_name] = float('nan')
    
    return result_df

def filter_zctas_by_state(zcta_df: pd.DataFrame, state_codes: List[str], verbose: bool = True) -> pd.DataFrame:
    """
    Filter ZCTAs to only include those in specified states
    
    Args:
        zcta_df: DataFrame with all ZCTA data
        state_codes: List of state FIPS codes to include
        verbose: Whether to print detailed progress
    
    Returns:
        DataFrame filtered to include only the specified states
    """
    all_state_data = []
    
    for state_code in state_codes:
        # Get the prefixes for this state
        if state_code not in STATE_ZIP_PREFIXES:
            print(f"Warning: No ZIP prefixes defined for state FIPS {state_code}")
            continue
            
        prefixes = STATE_ZIP_PREFIXES[state_code]
        
        # Filter by prefix
        mask = zcta_df['zip_code'].apply(lambda x: any(str(x).startswith(p) for p in prefixes))
        state_df = zcta_df[mask].copy()
        
        # Add state information
        state_df['state_fips'] = state_code
        state_abbr = {v: k for k, v in STATE_FIPS.items()}.get(state_code, "Unknown")
        state_df['state'] = state_abbr
        
        if verbose:
            print(f"Found {len(state_df)} ZIP codes in state {state_abbr}")
        all_state_data.append(state_df)
    
    if not all_state_data:
        return pd.DataFrame()
        
    # Combine all state data
    combined_df = pd.concat(all_state_data, ignore_index=True)
    return combined_df

def save_zip_data(df: pd.DataFrame, output_path: str, test_mode: bool = False) -> Tuple[str, str]:
    """
    Save ZIP code data to CSV and JSON files
    
    Args:
        df: DataFrame to save
        output_path: Base path to save files (without extension)
        test_mode: If True, save to testing directory
    
    Returns:
        Tuple of (csv_path, json_path) where files were saved
    """
    if df.empty:
        print("No data to save")
        return "", ""
    
    # Create the base directory
    base_dir = Path(TEST_DIR if test_mode else DATA_DIR)
    base_dir.mkdir(exist_ok=True)
    
    # Set file path
    file_base = Path(output_path)
    file_base.parent.mkdir(exist_ok=True)
    
    # Save as CSV
    csv_path = f"{file_base}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data saved to CSV: {csv_path}")
    
    # Save as JSON (nested format with ZIP code as key)
    json_path = f"{file_base}.json"
    
    # Convert to dictionary with ZIP code and year as nested keys
    zip_dict = {}
    for _, row in df.iterrows():
        zip_code = row['zip_code']
        year = row['year'] if 'year' in row else 'unknown'
        
        if zip_code not in zip_dict:
            zip_dict[zip_code] = {}
        
        # Drop zip_code before storing
        row_data = row.drop('zip_code').to_dict()
        
        # Store by year if we're doing multi-year
        if 'year' in row:
            zip_dict[zip_code][year] = row_data
        else:
            zip_dict[zip_code] = row_data
    
    with open(json_path, 'w') as f:
        json.dump(zip_dict, f, indent=2)
    
    print(f"Data saved to JSON: {json_path}")
    return csv_path, json_path

def get_census_zcta_data_for_year(
    states: List[str], 
    variables: List[str], 
    year: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Get Census ZCTA data for a specific year
    
    Args:
        states: List of state abbreviations
        variables: List of Census variable codes
        year: Census ACS year
        verbose: Whether to print detailed progress
    
    Returns:
        DataFrame with data for the specified year
    """
    # Convert state abbreviations to FIPS codes
    state_codes = [STATE_FIPS.get(state, "") for state in states]
    state_codes = [code for code in state_codes if code]  # Remove empty strings
    
    if not state_codes:
        print("Error: No valid states specified")
        return pd.DataFrame()
    
    # Try to get data for the specified year
    zcta_df = get_zcta_data_from_census(variables, year, verbose)
    
    # If we have no data, return empty DataFrame
    if zcta_df is None or zcta_df.empty:
        if verbose:
            print(f"Could not retrieve ZCTA data for {year}")
        return pd.DataFrame()
    
    # Filter by state
    result_df = filter_zctas_by_state(zcta_df, state_codes, verbose)
    
    return result_df

def get_multi_year_census_data(
    states: List[str],
    variables: List[str] = None,
    years: List[str] = None,
    output_path: str = None,
    test_mode: bool = False,
    max_workers: int = 1  # Be careful with concurrent API calls
) -> pd.DataFrame:
    """
    Get Census ZCTA data for multiple years and combine into a single DataFrame
    
    Args:
        states: List of state abbreviations
        variables: List of Census variable codes (defaults to CENSUS_VARS)
        years: List of years to query (defaults to latest available)
        output_path: Path to save output files (without extension)
        test_mode: If True, save to testing directory
        max_workers: Maximum number of concurrent API requests (1 = sequential)
    
    Returns:
        Combined DataFrame with data for all years
    """
    # Set default variables if not provided
    if variables is None:
        variables = list(CENSUS_VARS.keys())
        
    # Set default years if not provided
    if years is None:
        years = [AVAILABLE_YEARS[-1]]  # Just the most recent year
    
    print(f"Retrieving data for {len(years)} years: {', '.join(years)}")
    
    all_year_data = []
    failed_years = []
    
    if max_workers > 1:
        # Parallel processing for multiple years
        print(f"Using parallel processing with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all year queries
            future_to_year = {
                executor.submit(
                    get_census_zcta_data_for_year, 
                    states, variables, year, False  # Less verbose output in parallel
                ): year for year in years
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_year), total=len(years), desc="Processing years"):
                year = future_to_year[future]
                try:
                    year_data = future.result()
                    if not year_data.empty:
                        all_year_data.append(year_data)
                        print(f"✓ Year {year}: Retrieved {len(year_data)} records")
                    else:
                        failed_years.append(year)
                        print(f"✗ Year {year}: Failed to retrieve data")
                except Exception as e:
                    failed_years.append(year)
                    print(f"✗ Year {year}: Error - {str(e)}")
    else:
        # Sequential processing
        for year in tqdm(years, desc="Processing years"):
            try:
                year_data = get_census_zcta_data_for_year(states, variables, year)
                if not year_data.empty:
                    all_year_data.append(year_data)
                    print(f"✓ Year {year}: Retrieved {len(year_data)} records")
                else:
                    failed_years.append(year)
                    print(f"✗ Year {year}: Failed to retrieve data")
            except Exception as e:
                failed_years.append(year)
                print(f"✗ Year {year}: Error - {str(e)}")
    
    # Report on failed years
    if failed_years:
        print(f"\nWarning: Failed to retrieve data for {len(failed_years)} years: {', '.join(failed_years)}")
    
    # Combine all year data
    if not all_year_data:
        print("Error: No data retrieved for any year")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_year_data, ignore_index=True)
    print(f"Combined data for {len(all_year_data)} years: {combined_df.shape[0]} total records")
    
    # Save the data if an output path is provided
    if output_path is not None:
        save_zip_data(combined_df, output_path, test_mode)
    
    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pull Census ZIP code data for specified states')
    parser.add_argument('-t', '--test', action='store_true', 
                      help='Save to test directory')
    parser.add_argument('-s', '--states', nargs='+', default=['TX', 'CA'], 
                      help='List of state abbreviations to include (default: TX, CA)')
    parser.add_argument('-v', '--variables', nargs='+', 
                      help='Specific Census variables to retrieve (default: population, households)')
    parser.add_argument('-o', '--output', type=str, 
                      help='Output file path (without extension)')
    parser.add_argument('-y', '--year', type=str, default="2022",
                      help='Census ACS year to query (default: 2022)')
    parser.add_argument('-a', '--all-years', action='store_true',
                      help='Retrieve data for all available years (2009-2023)')
    parser.add_argument('--years', nargs='+',
                      help='Specific years to retrieve (e.g., 2019 2020 2021)')
    parser.add_argument('-w', '--workers', type=int, default=1,
                      help='Number of parallel workers for multi-year retrieval (default: 1)')
    
    args = parser.parse_args()
    
    # Get variables specified by user or use defaults
    variables = args.variables if args.variables else list(CENSUS_VARS.keys())
    
    # Determine which years to query
    if args.all_years:
        years_to_query = AVAILABLE_YEARS
    elif args.years:
        years_to_query = args.years
    else:
        years_to_query = [args.year]  # Just use the single year
    
    print(f"Running with settings:")
    print(f"- States: {args.states}")
    print(f"- Variables: {variables}")
    print(f"- Years: {years_to_query}")
    print(f"- Parallel workers: {args.workers}")
    
    # Set default output path if not specified
    output_path = args.output
    if output_path is None:
        states_str = "_".join(args.states)
        years_str = "multi_year" if len(years_to_query) > 1 else years_to_query[0]
        output_path = f"{DATA_DIR}/zip_census_data_{states_str}_{years_str}"
    
    # Get the data for multiple years
    zip_data = get_multi_year_census_data(
        states=args.states,
        variables=variables,
        years=years_to_query,
        output_path=output_path,
        test_mode=args.test,
        max_workers=args.workers
    )
    
    # Print summary information
    if not zip_data.empty:
        print("\nData retrieval complete!")
        print(f"Total records retrieved: {len(zip_data)}")
        
        # Count unique ZIP codes and years
        unique_zips = zip_data['zip_code'].nunique()
        unique_years = zip_data['year'].nunique()
        print(f"Unique ZIP codes: {unique_zips}")
        print(f"Years covered: {unique_years}")
        
        # Print sample data
        print("\nSample data (first 5 rows):")
        print(zip_data.head())
        
        # Print data quality summary
        print("\nData quality summary:")
        for col in zip_data.columns:
            if col not in ['zip_code', 'name', 'state', 'state_fips', 'year']:
                non_null = zip_data[col].notnull().sum()
                pct = non_null / len(zip_data) * 100
                print(f"- {col}: {non_null}/{len(zip_data)} values ({pct:.1f}% complete)")
                
        # Print year summary
        print("\nRecords by year:")
        year_counts = zip_data['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"- {year}: {count} records")
    else:
        print("ERROR: Could not retrieve any data after multiple attempts.")
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

logger = setup_logging()

now = datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
today = now.strftime("%Y%m%d")

# Exclude Holocene prepay per Charlie's advice and Red Trail Energy deal per Kevin at CDR.fyi
ORDER_IDS_TO_EXCLUDE = [
    '45afd8e0-8be9-4598-8437-e30dc620a4c8',
    '8af105d5-fb61-5882-b2ea-98332d5ae20b'
]

DAC_METHODS = ['DAC', 'Direct Air Carbon Capture and Storage (DACCS)', 'Direct Air Carbon Capture and Sequestration (DACCS)']

def process_orders_data(orders_df):
    """
    Process orders data to extract relevant columns, compute cumulative sums,
    and fill in missing dates with previous cumulative values.
    
    Args:
        orders_df (pd.DataFrame): Raw orders data from CDR.FYI
    
    Returns:
        pd.DataFrame: Processed dataframe with date, cumulative purchased and delivered tons
    """
    logger.info("Processing orders data")
    
    # Filter for relevant columns only
    relevant_cols = ['order_id', 'announcement_date', 'tons_purchased', 'tons_delivered', 'price_usd', 'type_of_agreement', 'method']
    df = orders_df[relevant_cols].copy()
    
    # Ensure date is in datetime format and sort chronologically
    df['date'] = pd.to_datetime(df['announcement_date'])
    df = df.sort_values('date')

    # Replace any NaN values with 0 for numerical columns
    df['tons_purchased'] = df['tons_purchased'].fillna(0)
    df['tons_delivered'] = df['tons_delivered'].fillna(0)

    # Filter for only Credit Sale orders
    df = df[df['type_of_agreement'] == 'Credit Sale']

    # Exclude Holocene prepay per Charlie's advice and Red Trail Energy deal per Kevin at CDR.fyi
    df = df[~df['order_id'].isin(ORDER_IDS_TO_EXCLUDE)]  

    # Create a filter for DAC methods
    df['dac_tons_purchased'] = df['tons_purchased'].where(df['method'].isin(DAC_METHODS), 0)
    df['dac_tons_delivered'] = df['tons_delivered'].where(df['method'].isin(DAC_METHODS), 0)

    df['price_usd'] = df['price_usd'].fillna(0)
    df['dac_price_usd'] = df['price_usd'].where(df['method'].isin(DAC_METHODS), 0)

    # Calculate weighted average DAC price per ton and fill in this value for orders without price_usd values
    df_dac_price_public = df[df['method'].isin(DAC_METHODS) & (df['dac_price_usd'] > 0)].copy()  # Add explicit copy here
    df_dac_price_public['dac_price_per_ton'] = df_dac_price_public['dac_price_usd'] / df_dac_price_public['dac_tons_purchased'].replace(0, pd.NA)

    # Filter out rows where dac_price_per_ton is NA (could happen if dac_tons_purchased is 0 or NA)
    df_dac_price_valid = df_dac_price_public.dropna(subset=['dac_price_per_ton'])

    # Calculate weighted average price per ton
    # weighted_avg = sum(price_i * weight_i) / sum(weight_i)
    weighted_avg_price = (
        (df_dac_price_valid['dac_price_per_ton'] * df_dac_price_valid['dac_tons_purchased']).sum() / 
        df_dac_price_valid['dac_tons_purchased'].sum()
    )

    # Fill missing values with the weighted average * tons purchased
    mask = df['method'].isin(DAC_METHODS) & (df['dac_price_usd'] == 0)

    # Can either use weighted average or just plug in $550 per ton as Phil suggested
    df.loc[mask, 'dac_price_usd'] = df.loc[mask, 'dac_tons_purchased'] * weighted_avg_price
    # df.loc[mask, 'dac_price_usd'] = df.loc[mask, 'dac_tons_purchased'] * 500

    # Compute cumulative sums
    df['tons_purchased_cum'] = df['tons_purchased'].cumsum()
    df['tons_delivered_cum'] = df['tons_delivered'].cumsum()
    df['dac_tons_purchased_cum'] = df['dac_tons_purchased'].cumsum()
    df['dac_tons_delivered_cum'] = df['dac_tons_delivered'].cumsum()
    df['price_usd_cum'] = df['price_usd'].cumsum()
    df['dac_price_usd_cum'] = df['dac_price_usd'].cumsum()

    # Remove duplicate dates by keeping the maximum cumulative values for numeric columns
    numeric_cols = [
        'tons_purchased_cum', 'tons_delivered_cum', 'dac_tons_purchased_cum', 'dac_tons_delivered_cum', 'price_usd_cum', 'dac_price_usd_cum'
        ]
    df = df.groupby('date', as_index=False)[numeric_cols].max()

    # Generate a complete date range
    start_date = df['date'].min()
    end_date = df['date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date)

    # Reindex the dataframe to include all dates
    df = df.set_index('date')
    df = df.reindex(all_dates, method='pad')  # Fill missing dates with previous values
    df.index.name = 'date'

    # Reset the index and format the date as string (YYYY-MM-DD)
    df = df.reset_index()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    logger.success(f"Processed orders data: {len(df)} rows")
    
    # Return the processed dataframe with required columns
    return df[[
        'date', 'tons_purchased_cum', 'tons_delivered_cum', 'dac_tons_purchased_cum', 'dac_tons_delivered_cum', 'price_usd_cum', 'dac_price_usd_cum'
        ]]

def process_purchasers_data(purchasers_df, orders_df):
    """
    Process purchasers data to extract relevant columns and calculate 
    cumulative count of unique purchasers by date.

    Args:
        purchasers_df (pd.DataFrame): Raw purchasers data from CDR.FYI
        orders_df (pd.DataFrame): Orders data containing method information

    Returns:
        pd.DataFrame: Processed dataframe with date and cumulative purchaser counts
    """
    logger.info("Processing purchasers data")

    # Select and prepare purchasers data
    purchasers_df = purchasers_df[['purchaser_id', 'created_at']].copy()
    purchasers_df['date'] = pd.to_datetime(purchasers_df['created_at'])
    purchasers_df = purchasers_df.sort_values('date')
    purchasers_df['date'] = purchasers_df['date'].dt.date  # Convert to date only

    # Prepare orders data with DAC flag
    orders_df = orders_df[['announcement_date', 'purchaser_id', 'method']].copy()
    orders_df['is_dac'] = orders_df['method'].isin(DAC_METHODS).astype(int)
    
    # Merge purchasers with orders to get DAC information
    df_merged = purchasers_df.merge(orders_df, on='purchaser_id', how='left')
    df_merged['is_dac'] = df_merged['is_dac'].fillna(0).astype(int)  # Fill NaN with 0
    
    # Count all unique purchasers by date
    all_purchasers = df_merged.groupby('date')['purchaser_id'].nunique().reset_index()
    all_purchasers.columns = ['date', 'new_purchasers']
    all_purchasers['purchasers_count_cum'] = all_purchasers['new_purchasers'].cumsum()
    
    # Count DAC purchasers by date
    dac_purchasers = df_merged[df_merged['is_dac'] == 1].groupby('date')['purchaser_id'].nunique().reset_index()
    dac_purchasers.columns = ['date', 'new_dac_purchasers']

    # Merge the two counts
    result_df = all_purchasers.merge(dac_purchasers[['date', 'new_dac_purchasers']],
                                     on='date', how='left')
    result_df['new_dac_purchasers'] = result_df['new_dac_purchasers'].fillna(0).astype(int)

    # dac cum sum
    result_df['dac_purchasers_count_cum'] = result_df['new_dac_purchasers'].cumsum()
    
    # Generate a complete date range
    start_date = result_df['date'].min()
    end_date = result_df['date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date)

    # Reindex the dataframe to include all dates
    result_df = result_df.set_index('date')
    result_df = result_df.reindex(all_dates, method='pad')  # Fill missing dates with previous values
    result_df.index.name = 'date'

    # Reset the index and format the date as string (YYYY-MM-DD)
    result_df = result_df.reset_index()
    result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')

    # Select and format final columns
    result_df = result_df[['date', 'purchasers_count_cum', 'dac_purchasers_count_cum']]
    result_df['date'] = result_df['date'].astype(str) 
    logger.success(f"Processed purchasers data: {len(result_df)} rows")
    return result_df

def process_suppliers_data(suppliers_df, orders_df):
    """
    Process suppliers data to build stacked bars of suppliers and tons sold.

    Args:
        suppliers_df (pd.DataFrame): Raw suppliers data from CDR.FYI
        orders_df (pd.DataFrame): Raw orders data from CDR.FYI

    Returns:
        pd.DataFrame: Processed dataframe with year, supplier, and tons sold 
    """
    logger.info("Processing suppliers data")
    
    # Filter for relevant columns only
    suppliers_df = suppliers_df[['supplier_id', 'name']].copy()
    orders_df = orders_df[['announcement_date', 'supplier_id', 'tons_purchased', 'method']].copy()

    orders_df['is_dac'] = orders_df['method'].isin(DAC_METHODS).astype(int)
    
    merged = orders_df.merge(suppliers_df, on='supplier_id', how='left')
    merged['announcement_date'] = pd.to_datetime(merged['announcement_date'])
    merged['year'] = merged['announcement_date'].dt.year
    merged['tons_purchased'] = merged['tons_purchased'].fillna(0)
    merged['tons_purchased'] = merged['tons_purchased'].astype(float)

    merged_grouped = merged.groupby(['year', 'name', 'is_dac'], as_index=False).agg(
        tons_sold=('tons_purchased', 'sum')
    )
    logger.success(f"Processed suppliers data: {len(merged_grouped)} rows")
    
    return merged_grouped


def process_latest_deals(orders, purchasers, suppliers):
    """
    Process the latest deals by joining orders with purchasers and suppliers
    and extracting the most recent 10 deals with required fields.

    Args:
        orders (pd.DataFrame): Orders data from CDR.FYI
        purchasers (pd.DataFrame): Purchasers data from CDR.FYI
        suppliers (pd.DataFrame): Suppliers data from CDR.FYI

    Returns:
        pd.DataFrame: DataFrame with the most recent 10 deals
    """
    logger.info("Processing latest deals data")
    
    # Merge orders with purchasers and suppliers to get names
    orders = orders.copy()
    purchasers = purchasers[['purchaser_id', 'name']].rename(columns={'name': 'purchaser_name'})
    suppliers = suppliers[['supplier_id', 'name']].rename(columns={'name': 'supplier_name'})

    # Join orders with purchasers and suppliers
    merged = orders.merge(purchasers, on='purchaser_id', how='left')
    merged = merged.merge(suppliers, on='supplier_id', how='left')

    # Normalize the 'method' column
    merged['method'] = merged['method'].replace('Direct Air Carbon Capture and Storage (DACCS)', 'DAC')

    # Filter and sort by announcement_date
    merged['announcement_date'] = pd.to_datetime(merged['announcement_date'])
    latest_deals = merged.sort_values(by='announcement_date', ascending=False).head(10)

    # Select required columns
    latest_deals = latest_deals[['announcement_date', 'supplier_name', 'purchaser_name', 'tons_purchased', 'method']]

    # Fill missing values with placeholders
    latest_deals['supplier_name'] = latest_deals['supplier_name'].fillna('Unknown Supplier')
    latest_deals['purchaser_name'] = latest_deals['purchaser_name'].fillna('Unknown Purchaser')
    latest_deals['tons_purchased'] = latest_deals['tons_purchased'].fillna(0)
    latest_deals['method'] = latest_deals['method'].fillna('Unknown Method')
    logger.success(f"Processed latest deals: {len(latest_deals)} rows")
    return latest_deals


def scrape_dollars_spent(url="https://cdr.fyi/"):
    """
    Scrapes the total dollars spent on carbon removal from cdr.fyi.
    MDK temporary function to pull their dollars spent value straight from front-end until they give me calculation.
    
    Returns:
        float: The total dollars spent value as a float, or None if scraping fails
    """
    logger.info("Scraping total dollars spent on carbon removal from cdr.fyi")
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Request the webpage
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the element by specific class - the h3 with text-primary class
        dollars_element = soup.find('h3', class_='text-primary')
        
        if dollars_element and "spent on CO2 removal" in dollars_element.find_next('p').text:
            # Extract and clean the text
            dollars_text = dollars_element.text.strip()
            # Remove $ and commas, then convert to float
            dollars_value = float(dollars_text.replace('$', '').replace(',', ''))
            
            logger.success(f"Successfully scraped dollars spent: ${dollars_text}")
            return dollars_value
        else:
            # Look for all text-primary h3 elements and check context
            all_h3_elements = soup.find_all('h3', class_='text-primary')
            for h3 in all_h3_elements:
                if h3.find_next('p') and "spent on CO2" in h3.find_next('p').text:
                    dollars_text = h3.text.strip()
                    dollars_value = float(dollars_text.replace('$', '').replace(',', ''))
                    logger.success(f"Successfully scraped dollars spent: ${dollars_value}")
                    return dollars_value
            
            logger.error("Could not find the dollars spent element on the page")
            return None
            
    except Exception as e:
        logger.error(f"Error scraping dollars spent: {e}", exc_info=True)
        return None

def create_dollars_df(value, today=None):
    """
    Create a DataFrame with dollars spent value.
    
    Args:
        value (float): The dollars spent value to save
        today (str): Date string in YYYYMMDD format, defaults to current date
    
    Returns:
        pd.DataFrame: DataFrame containing the dollars spent value
    """
    if today is None:
        today = datetime.now().strftime("%Y%m%d")
        
    # Create a small dataframe with the value
    df = pd.DataFrame({
        'metric': ['dollars_spent'],
        'value': [value]
    })
    
    logger.success(f"Created dollars spent DataFrame with value: ${value:,.2f}")
    
    return df

# --- Add mapping and config loading (similar to updater) ---
PROCESSOR_FUNCTIONS = {
    "process_orders_data": process_orders_data,
    "process_latest_deals": process_latest_deals,
    "process_purchasers_data": process_purchasers_data,
    "process_suppliers_data": process_suppliers_data
}


if __name__== "__main__":
    logger.info("Starting local data processing using configuration...")

    # Ensure the PROCESSED_DIR exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load configuration
    try:
        datasets_config = load_dataset_config()
    except Exception as e:
        logger.error(f"Failed to load dataset configuration. Exiting. Error: {e}")
        sys.exit(1) # Exit if config cannot be loaded

    # --- Load required raw dataframes ---
    raw_dataframes = {}
    logger.info("Loading required raw data files...")
    required_files = set()
    # Determine all unique raw files needed by processors
    for config in datasets_config:
        if config.get("processor_func_name"): # Only consider datasets that will be processed
             required_files.update(config.get("requires_raw", []))

    # Find the corresponding filenames for the required raw datasets
    files_to_load = {}
    for req_name in required_files:
        found = False
        for config in datasets_config:
            if config["name"] == req_name and config.get("raw_filename"):
                files_to_load[req_name] = config["raw_filename"]
                found = True
                break
        if not found:
             logger.warning(f"Configuration for required raw dataset '{req_name}' or its raw_filename not found.")


    # Load the necessary raw files
    for name, filename in files_to_load.items():
        file_path = RAW_DIR / filename
        if file_path.exists():
            try:
                logger.info(f"Loading raw file: {file_path}")
                # Use consistent read_csv options
                raw_dataframes[name] = pd.read_csv(
                    file_path,
                    low_memory=False
                )
                # Special case for older orders file that might have index column
                if name == 'orders' and 'Unnamed: 0' in raw_dataframes[name].columns:
                     raw_dataframes[name] = pd.read_csv(file_path, index_col=0, low_memory=False)

            except Exception as e:
                logger.error(f"Error loading raw file {file_path}: {e}")
        else:
            logger.error(f"Required raw file not found: {file_path}. Cannot proceed with processing dependent datasets.")
            # We don't exit here, but dependent processing steps will fail later

    # --- Process datasets based on config ---
    logger.info("Processing datasets...")
    for config in datasets_config:
        processor_func_name = config.get('processor_func_name')
        if not processor_func_name:
            continue # Skip datasets without a processor

        processor_func = PROCESSOR_FUNCTIONS.get(processor_func_name)
        if not processor_func:
            logger.error(f"Processor function '{processor_func_name}' not found in mapping for dataset '{config['name']}'. Skipping.")
            continue

        output_base = config['output_filename_base']
        required_raw_names = config['requires_raw']
        dataset_name = config['name']

        logger.info(f"Attempting to process dataset: {dataset_name} using {processor_func_name}...")

        # Gather required input dataframes
        input_dfs = []
        missing_raw = False
        for raw_name in required_raw_names:
            if raw_name in raw_dataframes:
                input_dfs.append(raw_dataframes[raw_name])
            else:
                # Log error if a required raw file wasn't loaded successfully earlier
                logger.error(f"Missing required raw dataframe '{raw_name}' (file: {files_to_load.get(raw_name, 'N/A')}) for processing '{dataset_name}'. Skipping.")
                missing_raw = True
                break

        if missing_raw:
            continue # Skip processing this dataset

        # Call the processor function
        try:
            processed_df = processor_func(*input_dfs)

            # Use save_dataset for consistent file handling
            output_path = save_dataset(processed_df, output_base, is_local=True)
            logger.success(f"Saved processed {dataset_name} data to {output_path}")

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}", exc_info=True)

    # Scrape and save dollars spent
    try:
        logger.info("Scraping dollars spent from cdr.fyi...")
        dollars_spent = scrape_dollars_spent()
        
        if dollars_spent is not None:
            # Create dollars spent DataFrame
            dollars_df = create_dollars_df(dollars_spent)
            
            # Save using the consistent save_dataset function
            output_path = save_dataset(dollars_df, "dollars_spent", is_local=True)
            logger.success(f"Saved dollars spent value (${dollars_spent:,.2f}) to {output_path}")
    except Exception as e:
        logger.error(f"Error scraping or saving dollars spent: {e}", exc_info=True)
        
    logger.info("Local data processing finished.")


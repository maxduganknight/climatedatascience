import sys
from pathlib import Path
cdr_dashboard_dir = Path(__file__).parent.parent
if str(cdr_dashboard_dir) not in sys.path:
    sys.path.append(str(cdr_dashboard_dir))

import pandas as pd
import os
import json
from datetime import datetime, timezone
from utils.logging_utils import setup_logging
from utils.paths import RAW_DIR, PROCESSED_DIR, PROJECT_ROOT, SALES_DIR, IS_LAMBDA
from utils.retrieval_utils import save_dataset
import requests
import re
from fuzzywuzzy import fuzz
from google.oauth2 import service_account
from googleapiclient.discovery import build

logger = setup_logging()

# Only import from creds.py when not running in Lambda
if not IS_LAMBDA:
    try:
        sys.path.append('/Users/max/Deep_Sky/')
        from creds import ATTIO_API_TOKEN, GOOGLE_SERVICE_ACCOUNT
        logger.info("Using Attio API token from creds.py")
    except ImportError:
        logger.warning("Could not import credentials from creds.py")
        ATTIO_API_TOKEN = None
        GOOGLE_SERVICE_ACCOUNT = None
else:
    # Initialize these as None when running in Lambda; they'll be set by missing_leads_handler.py
    ATTIO_API_TOKEN = None
    GOOGLE_SERVICE_ACCOUNT = None

now = datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
today = now.strftime("%Y%m%d")

def pull_attio_data(base_url="https://api.attio.com/v2", token=None):
    """
    Pulls data from Attio using the provided URL and credentials.
    """
    if not token:
        logger.error("No API token provided")
        return pd.DataFrame()
    
    # Set up the request headers with the authentication token
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        # List available objects
        logger.info("Retrieving objects information from Attio")
        objects_response = requests.get(f"{base_url}/objects", headers=headers)
        
        if objects_response.status_code != 200:
            logger.error(f"API request failed with status code: {objects_response.status_code}")
            logger.error(f"Response content: {objects_response.text}")
            return pd.DataFrame()
        
        objects_data = objects_response.json()
        
        # Summarize available objects - more concise logging
        objects_summary = []
        for obj in objects_data.get("data", []):
            objects_summary.append({
                "api_slug": obj.get("api_slug"),
                "object_id": obj.get("id", {}).get("object_id")
            })
        
        objects_df = pd.DataFrame(objects_summary)
        logger.info(f"Available objects in Attio: {', '.join(objects_df['api_slug'].tolist())}")
        
        # Get the company object
        company_object = next((obj for obj in objects_data["data"] if obj["api_slug"] == "companies"), None)
        if company_object:
            # Get object attributes
            attributes_response = requests.get(
                f"{base_url}/objects/{company_object['api_slug']}/attributes",
                headers=headers
            )
            
            if attributes_response.status_code == 200:
                attributes_data = attributes_response.json()
                
                attributes_summary = []
                for attr in attributes_data.get("data", []):
                    attributes_summary.append({
                        "api_slug": attr.get("api_slug"),
                        "type": attr.get("type")
                    })
                
                attributes_df = pd.DataFrame(attributes_summary)
                logger.info(f"Retrieved {len(attributes_df)} company attributes")
                all_attribute_slugs = attributes_df['api_slug'].tolist() # Get all attribute slugs
                
                logger.info("Retrieving company records with all attributes from Attio")
                
                all_records_data = [] # Renamed from all_records to avoid confusion, will store processed data
                page = 1
                per_page = 100
                
                while True:
                    logger.info(f"Fetching page {page} of company records")
                    
                    records_response = requests.post(
                        f"{base_url}/objects/companies/records/query",  
                        headers=headers,
                        json={
                            "limit": per_page,
                            "offset": (page - 1) * per_page,
                            "attributes": all_attribute_slugs # Request all attributes
                        }
                    )
                    
                    if records_response.status_code != 200:
                        logger.error(f"Failed to retrieve records page {page}: {records_response.status_code}")
                        logger.error(f"Response content: {records_response.text}")
                        break
                    
                    records_page_data = records_response.json()
                    current_page_records = records_page_data.get("data", [])
                    
                    if not current_page_records:
                        logger.info(f"No more records found after page {page-1}")
                        break
                    
                    # Process records from the current page
                    for record in current_page_records:
                        record_data = {"id": record.get("id", {}).get("record_id")}
                        values = record.get("values", {})
                        for attr_key, attr_values_list in values.items():
                            if attr_values_list and isinstance(attr_values_list, list) and len(attr_values_list) > 0:
                                latest_value_entry = attr_values_list[0]
                                if "value" in latest_value_entry:
                                    record_data[attr_key] = latest_value_entry["value"]
                                elif "values" in latest_value_entry: # For multi-select or similar
                                    record_data[attr_key] = latest_value_entry["values"]
                                elif "option" in latest_value_entry: # For select attributes
                                    record_data[attr_key] = latest_value_entry["option"]["title"] if latest_value_entry.get("option") else None
                                elif "domain" in latest_value_entry: # For domain attributes
                                    record_data[attr_key] = latest_value_entry["domain"]
                                # Add other specific type handlers if necessary, or a generic fallback
                                else:
                                    complex_data = {k: v for k, v in latest_value_entry.items()
                                                    if k not in ["active_from", "active_until", "created_by_actor", "attribute_type"]}
                                    if complex_data:
                                        record_data[attr_key] = json.dumps(complex_data)
                        all_records_data.append(record_data)
                    
                    logger.info(f"Retrieved and processed {len(current_page_records)} companies on page {page}")
                    
                    if len(current_page_records) < per_page:
                        break
                        
                    page += 1
                
                logger.info(f"Successfully retrieved and processed data for {len(all_records_data)} total company records")
                
                records_df = pd.DataFrame(all_records_data)
                
                if not records_df.empty:
                    logger.info(f"Successfully created DataFrame for {len(records_df)} companies")
                    non_empty_cols = [col for col in records_df.columns if records_df[col].notna().any()]
                    logger.info(f"DataFrame has {len(non_empty_cols)} populated columns for companies")
                    
                    key_cols = ['id', 'name', 'domains', 'categories', 'created_at']
                    available_keys = [col for col in key_cols if col in records_df.columns]
                
                return records_df
            else:
                logger.error(f"Failed to retrieve attributes: {attributes_response.status_code}")
                logger.error(f"Response content: {attributes_response.text}")
        else:
            logger.warning("Company object not found in Attio")
        
        return pd.DataFrame()
 
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing Attio data: {str(e)}")
        return pd.DataFrame()

def create_frequency_dict(list_col):
    """Convert a list of values to a dictionary with counts of each unique value"""
    if not list_col or not isinstance(list_col, list):
        return {}
    
    result = {}
    for item in list_col:
        result[item] = result.get(item, 0) + 1
    return result

def retrieve_cdr_fyi_purchasers(purchasers_path, orders_path):
    """
    Retrieves the CDR FYI purchasers from a CSV file.
    
    Args:
        path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The CDR FYI purchasers as a DataFrame.
    """

    purchasers_df = pd.read_csv(purchasers_path)
    purchasers_filtered = purchasers_df[purchasers_df['methods'] != 'Private Person']
    purchasers_filtered = purchasers_filtered[purchasers_filtered['name'] != 'Not Disclosed']

    orders_df = pd.read_csv(orders_path)

    # Merge the two DataFrames on the 'purchaser_id' column
    merged_df = pd.merge(purchasers_filtered, orders_df, on='purchaser_id', how='inner')

    relevant_cols = [
        'purchaser_id', 'order_id', 'name', 'method', 'status', 'tons_purchased',
        'price_usd', 'type_of_agreement', 'tons_delivered', 'website'
        ]

    merged_df = merged_df[relevant_cols]
    merged_grouped = merged_df.groupby('purchaser_id').agg({
        'order_id': 'count',
        'name': 'first',
        'method': list,
        'status': list,
        'tons_purchased': 'sum',
        'price_usd': 'sum',
        'type_of_agreement': list,
        'tons_delivered': 'sum',
        'website': 'first'
    }).reset_index()
    
    # Convert list columns to frequency dictionaries
    list_columns = ['method', 'status', 'type_of_agreement']
    for col in list_columns:
        merged_grouped[f'{col}_freq'] = merged_grouped[col].apply(create_frequency_dict)
    
    merged_grouped = merged_grouped.rename(columns={
        'purchaser_id': 'purchaser_id',
        'order_id': 'num_orders',
        'name': 'purchaser_name',
        'price_usd': 'total_spent_usd',
        'method_freq': 'cdr_methods',
        'status_freq': 'order_statuses',
        'type_of_agreement_freq': 'agreement_types',
        'website': 'website'
    })

    output_cols = [
        'purchaser_id', 'purchaser_name', 'num_orders', 'tons_purchased', 'total_spent_usd', 'tons_delivered',
        'cdr_methods', 'order_statuses', 'agreement_types', 'website'
    ]

    merged_grouped = merged_grouped[output_cols]

    return merged_grouped

def clean_attio_data(records_df):
    """
    Clean and process the Attio data to create a more usable DataFrame
    """
    # Copy the dataframe to avoid modifying the original
    df = records_df.copy()
    
    # Define columns to flatten from JSON strings
    json_columns = [col for col in df.columns if df[col].dtype == 'object' and 
                    df[col].notna().any() and 
                    isinstance(df[col].iloc[0], str) and 
                    df[col].iloc[0].startswith('{')]
    
    # Handle complex JSON columns
    for col in df.columns:
        # Skip id column and already processed columns
        if col == 'id' or col == 'record_id':
            continue
            
        # Extract specific fields from complex JSON objects
        if col == 'domains':
            # Keep domains as is - it's already a simple value
            pass
        elif col == 'team':
            # Extract target_record_id from team JSON
            df[f'{col}_id'] = df[col].apply(lambda x: 
                json.loads(x)['target_record_id'] if isinstance(x, str) and x.startswith('{') else None)
        elif col == 'categories':
            # Keep categories as is - it's already extracted as the option title
            pass
        elif col == 'primary_location':
            # Extract location components
            def extract_location(loc_str):
                if not isinstance(loc_str, str) or not loc_str.startswith('{'):
                    return pd.Series([None, None, None, None])
                loc = json.loads(loc_str)
                return pd.Series([
                    loc.get('line_1'),
                    loc.get('locality'),
                    loc.get('region'),
                    loc.get('country_code')
                ])
            
            location_df = df[col].apply(extract_location)
            df['address_line1'] = location_df[0]
            df['city'] = location_df[1]
            df['state_region'] = location_df[2]
            df['country'] = location_df[3]
        elif col.endswith('_interaction'):
            # Extract interaction date and type
            def extract_interaction(interaction_str):
                if not isinstance(interaction_str, str) or not interaction_str.startswith('{'):
                    return pd.Series([None, None])
                interaction = json.loads(interaction_str)
                return pd.Series([
                    interaction.get('interacted_at'),
                    interaction.get('interaction_type')
                ])
            
            interaction_df = df[col].apply(extract_interaction)
            df[f'{col}_date'] = interaction_df[0]
            df[f'{col}_type'] = interaction_df[1]
            
    # Drop original complex columns
    complex_cols_to_drop = ['team', 'primary_location', 
                           'first_email_interaction', 'last_email_interaction',
                           'first_interaction', 'last_interaction', 
                           'next_interaction', 'strongest_connection_user',
                           'created_by']
    
    # Only drop columns that exist
    cols_to_drop = [col for col in complex_cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Convert dates to proper datetime format
    date_columns = [col for col in df.columns if 'date' in col.lower() or col.endswith('_at')]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], utc=True)  # Always parse with UTC timezone
        except:
            pass
            
    return df

def process_attio_data(attio_data):
    """
    Process the Attio data to extract relevant information.
    
    Args:
        attio_data (pd.DataFrame): The Attio data to process.
        
    Returns:
        pd.DataFrame: The processed Attio data.
    """
    # Clean and process the data
    processed_df = clean_attio_data(attio_data)

    # Calculate 'days_since_last_interaction'
    # Assuming 'last_interaction_date' is created by clean_attio_data and is timezone-aware (UTC)
    if 'last_interaction_date' in processed_df.columns and pd.api.types.is_datetime64_any_dtype(processed_df['last_interaction_date']):
        # Ensure current time is also UTC for correct comparison
        now_utc = datetime.now(timezone.utc)
        processed_df['days_since_last_interaction'] = (now_utc - processed_df['last_interaction_date']).dt.days
    else:
        logger.warning("'last_interaction_date' column not found or not datetime, 'days_since_last_interaction' will be missing or NaN.")
        processed_df['days_since_last_interaction'] = pd.NA # Or a default large number like 9999
    
    relevant_columns = [
        'id', 'name', 'domains', 'categories', 'created_at',
        'city', 'state_region', 'country',
        'estimated_arr_usd', 'days_since_last_interaction'
    ]

    # Filter for existing relevant columns to avoid errors
    existing_relevant_columns = [col for col in relevant_columns if col in processed_df.columns]
    processed_df = processed_df[existing_relevant_columns].copy()

    # The following line seems to be unused as last_6_mos_interactions is not returned or used later in this function
    last_6_mos_interactions = processed_df[
        processed_df['days_since_last_interaction'].notna() & (processed_df['days_since_last_interaction'] <= 180)
    ]
    
    return last_6_mos_interactions

def normalize_company_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower()
    # Remove common legal entity types
    name = re.sub(r'\b(inc|llc|ltd|gmbh|corporation|corp|co|ag|limited|management|asset|group|technologies|communications|partners|)\b\.?', '', name)
    # Remove special characters and extra whitespace
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def extract_domain(url):
    """Extract domain from URL"""
    if pd.isna(url):
        return ""
    url = str(url).lower().strip()
    
    # Remove http://, https://, www.
    domain = re.sub(r'^https?://', '', url)
    domain = re.sub(r'^www\.', '', domain)
    
    # Extract domain name (before path or query parameters)
    domain = domain.split('/')[0]
    domain = domain.split('?')[0]
    
    return domain


def exact_match_companies(attio, cdr):
    """
    Find matching companies between Attio and CDR dataframes.
    Matches companies if EITHER company name OR domain matches exactly.
    
    Returns:
        DataFrame: Merged dataframe with match indicators
    """
    
    # Create an empty result dataframe with all CDR companies
    result = cdr.copy()
    # Add empty columns for match indicators
    result['attio_id'] = None
    result['attio_name'] = None
    result['attio_clean_name'] = None
    result['attio_clean_website'] = None
    result['domain_match'] = 0
    result['name_match'] = 0
    result['match'] = 0
    
    # First pass: Match by domain
    if 'cdr_clean_website' in result.columns and 'attio_clean_website' in attio.columns:
        domain_matches = pd.merge(
            cdr[['cdr_id', 'cdr_clean_website']],
            attio[['attio_id', 'attio_name', 'attio_clean_name', 'attio_clean_website']],
            how='inner',
            left_on='cdr_clean_website',
            right_on='attio_clean_website'
        )
        
        # Update result with domain matches
        if not domain_matches.empty:
            for _, row in domain_matches.iterrows():
                match_mask = result['cdr_id'] == row['cdr_id']
                if match_mask.any():
                    result.loc[match_mask, 'attio_id'] = row['attio_id']
                    result.loc[match_mask, 'attio_name'] = row['attio_name']
                    result.loc[match_mask, 'attio_clean_name'] = row['attio_clean_name']
                    result.loc[match_mask, 'attio_clean_website'] = row['attio_clean_website']
                    result.loc[match_mask, 'domain_match'] = 1
                    result.loc[match_mask, 'match'] = 1
    
    # Second pass: Match by name
    name_matches = pd.merge(
        cdr[['cdr_id', 'cdr_clean_name']],
        attio[['attio_id', 'attio_clean_name', 'attio_name', 'attio_clean_website']],
        how='inner',
        left_on='cdr_clean_name',
        right_on='attio_clean_name'
    )
    
    # Update result with name matches
    if not name_matches.empty:
        for _, row in name_matches.iterrows():
            match_mask = result['cdr_id'] == row['cdr_id']
            if match_mask.any():
                # Only update attio_id if not already set by domain match
                if pd.isna(result.loc[match_mask, 'attio_id'].iloc[0]):
                    result.loc[match_mask, 'attio_id'] = row['attio_id']
                    result.loc[match_mask, 'attio_name'] = row['attio_name']
                    result.loc[match_mask, 'attio_clean_name'] = row['attio_clean_name']
                    result.loc[match_mask, 'attio_clean_website'] = row['attio_clean_website']
                
                result.loc[match_mask, 'name_match'] = 1
                result.loc[match_mask, 'match'] = 1
    
    # Add the unmatched Attio companies to the result
    matched_attio_ids = result['attio_id'].dropna().unique()
    unmatched_attio = attio[~attio['attio_id'].isin(matched_attio_ids)]
    
    # Create empty rows for unmatched Attio companies
    attio_rows = []
    for _, row in unmatched_attio.iterrows():
        new_row = pd.Series({
            'attio_id': row['attio_id'],
            'attio_name': row['attio_name'],
            'attio_clean_name': row['attio_clean_name'],
            'attio_clean_website': row['attio_clean_website'],
            'domain_match': 0,
            'name_match': 0,
            'match': 0
        })
        # Add all other Attio columns
        for col in row.index:
            if col not in new_row:
                new_row[col] = row[col]
        
        attio_rows.append(new_row)
    
    # Append unmatched Attio companies to the result
    if attio_rows:
        result = pd.concat([result, pd.DataFrame(attio_rows)], ignore_index=True)
    
    return result
def fuzzy_match_companies(attio_data, cdr_data, threshold=70, output_file="fuzzy_matches.csv"):
    """
    Perform enhanced fuzzy matching between missing leads and Attio companies.
    
    Args:
        attio_data: DataFrame containing Attio companies
        cdr_data: DataFrame containing CDR.fyi purchasers
        threshold: Minimum total similarity score to consider a match (0-100)
        output_file: Path to save the fuzzy matches CSV
        
    Returns:
        DataFrame with fuzzy matches and similarity scores
    """
    # Create empty lists to store the results
    matches = []
    
    # Create clean copies of the input data
    attio = attio_data.copy()
    cdr = cdr_data.copy()
        
    # Filter out very short company names that cause false positives
    MIN_NAME_LENGTH = 3
    attio = attio[attio['attio_clean_name'].str.len() >= MIN_NAME_LENGTH]
    
    print(f"Processing {len(cdr)} CDR companies against {len(attio)} Attio companies")
        
    # Iterative matching with weighted scoring
    for i, cdr_row in cdr.iterrows():
        best_match = None
        best_match_row = None
        best_id = None
        best_total_score = 0
        best_score_components = {}
        
        cdr_name = cdr_row['cdr_clean_name']
        cdr_clean_website = cdr_row['cdr_clean_website']
        
        # Skip entries with empty names
        if not cdr_name or len(cdr_name) < MIN_NAME_LENGTH:
            continue
            
        for j, attio_row in attio.iterrows():
            attio_id = attio_row['attio_id']
            attio_name = attio_row['attio_clean_name']
            attio_clean_website = attio_row['attio_clean_website']
            
            # Skip comparisons where one name is a substring of the other 
            # AND there's a large length difference
            if (attio_name in cdr_name or cdr_name in attio_name) and abs(len(attio_name) - len(cdr_name)) > 10:
                continue
            
            # Initialize score components
            score_components = {
                'name_score': 0,
                'domain_score': 0,
                'length_penalty': 0
            }
            
            # 1. Company Name Similarity (Primary component, max 70 points)
            if cdr_name and attio_name:
                # Try different fuzzy match methods
                token_sort = fuzz.token_sort_ratio(cdr_name, attio_name)
                token_set = fuzz.token_set_ratio(cdr_name, attio_name)
                partial = fuzz.partial_ratio(cdr_name, attio_name)
                
                # Weight the scores differently based on name characteristics
                name_lengths = (len(cdr_name), len(attio_name))
                length_diff = abs(name_lengths[0] - name_lengths[1])
                
                # Apply length difference penalty
                length_penalty = min(20, length_diff * 2)
                score_components['length_penalty'] = -length_penalty
                
                # Combine scores with emphasis on token_set for different word orders
                name_score = 0.5 * token_set + 0.3 * token_sort + 0.2 * partial
                
                # Apply word count bonus for multi-word matches
                cdr_word_count = len(cdr_name.split())
                attio_word_count = len(attio_name.split())
                if cdr_word_count > 1 and attio_word_count > 1:
                    common_words = len(set(cdr_name.split()) & set(attio_name.split()))
                    if common_words >= 2:
                        name_score = min(100, name_score + 10)
                
                # Scale to max 70 points
                score_components['name_score'] = (name_score / 100) * 70
            
            # 2. Domain Similarity (max 30 points)
            if cdr_clean_website and attio_clean_website:
                # Domain similarity score
                domain_score = 0
                
                # Exact domain match (highest priority)
                if cdr_clean_website == attio_clean_website:
                    domain_score = 30
                # Root domain match (e.g., apple.com and apple.co.uk)
                elif (cdr_clean_website.split('.')[0] == attio_clean_website.split('.')[0] and
                      len(cdr_clean_website.split('.')[0]) > 3):
                    domain_score = 20
                # One domain contains the other (but not as a tiny substring)
                elif ((cdr_clean_website in attio_clean_website or 
                       attio_clean_website in cdr_clean_website) and
                      min(len(cdr_clean_website), len(attio_clean_website)) > 5):
                    domain_score = 10
                
                score_components['domain_score'] = domain_score
            
            # Calculate total score (max 100 points)
            total_score = (score_components['name_score'] + 
                          score_components['domain_score'] + 
                          score_components['length_penalty'])
            
            # Update best match if better score found
            if total_score > best_total_score and total_score >= threshold:
                best_total_score = total_score
                best_match = attio_name
                best_match_row = attio_row
                best_id = attio_id
                best_score_components = score_components
        
        # Store the result if a match was found
        if best_match:
            matches.append({
                'cdr_id': cdr_row.get('cdr_id'),
                'cdr_name': cdr_row.get('cdr_name'),
                'cdr_clean_name': cdr_name,
                'cdr_clean_website': cdr_clean_website,
                'attio_id': best_id,
                'attio_name': best_match_row.get('attio_name'),
                'attio_clean_name': best_match,
                'attio_clean_website': best_match_row.get('attio_clean_website'),
                'fuzzy_score': round(best_total_score, 1),  # Overall score
                'name_score': round(best_score_components['name_score'], 1), 
                'domain_score': round(best_score_components['domain_score'], 1),
                'length_penalty': round(best_score_components['length_penalty'], 1)
            })
    
    # Create DataFrame from results
    matches_df = pd.DataFrame(matches)
    
    # Sort by score
    if not matches_df.empty:
        matches_df = matches_df.sort_values(by='fuzzy_score', ascending=False)
    
    return matches_df

def save_fuzzy_matches(fuzzy_matches, output_path=SALES_DIR / 'fuzzy_matches.csv'):
    """
    Save fuzzy matches to a CSV file for analysis
    
    Args:
        fuzzy_matches: DataFrame of fuzzy matches with scores
        output_path: Path to save the CSV
    """
    if fuzzy_matches.empty:
        print("No fuzzy matches to save")
        return
        
    # Sort by total score in descending order
    sorted_matches = fuzzy_matches.sort_values('fuzzy_score', ascending=False)
    
    # Add confidence level column
    def get_confidence(score):
        if score >= 85:
            return "High"
        elif score >= 75:
            return "Medium"
        else:
            return "Low"
    
    if 'fuzzy_score' in sorted_matches.columns:
        sorted_matches['confidence'] = sorted_matches['fuzzy_score'].apply(get_confidence)
    
    # Save to CSV
    sorted_matches.to_csv(output_path, index=False)
    print(f"Saved {len(sorted_matches)} fuzzy matches to {output_path}")

def add_fuzzy_match_to_results(result_df, fuzzy_row, match_threshold=75):
    # Make a copy to avoid modifying the original
    result = result_df.copy()
    
    attio_id = fuzzy_row['attio_id']
    cdr_id = fuzzy_row['cdr_id']
    fuzzy_score = fuzzy_row['fuzzy_score']
    
    # Look for existing rows with either this Attio ID or CDR ID
    attio_mask = result['attio_id'] == attio_id
    cdr_mask = result['cdr_id'] == cdr_id
    
    # Check if the row already has an exact match
    if cdr_mask.any() and result.loc[cdr_mask, 'match'].iloc[0] == 1:
        # Skip updating this row since it already has an exact match
        return result
    
    # Case 1: Row exists with the Attio ID
    if attio_mask.any():
        # Skip if this Attio row already has an exact match
        if result.loc[attio_mask, 'match'].iloc[0] == 1:
            return result
            
        # Update the existing row with Attio ID
        result.loc[attio_mask, 'cdr_id'] = cdr_id
        result.loc[attio_mask, 'fuzzy_score'] = fuzzy_score
        
        # Only mark as match if score is high enough
        if fuzzy_score >= match_threshold:
            result.loc[attio_mask, 'match'] = 1
    
    # Case 2: Row exists with the CDR ID
    elif cdr_mask.any():
        # Update the existing row with CDR ID
        result.loc[cdr_mask, 'attio_id'] = attio_id
        result.loc[cdr_mask, 'fuzzy_score'] = fuzzy_score
        
        # Only mark as match if score is high enough
        if fuzzy_score >= match_threshold:
            result.loc[cdr_mask, 'match'] = 1
    
    # Case 3: Neither ID exists in the results, create a new row
    else:
        # Create a new row with the fuzzy match
        new_row = pd.Series({
            'attio_id': attio_id,
            'attio_name': fuzzy_row['attio_name'],
            'cdr_id': cdr_id, 
            'cdr_name': fuzzy_row['cdr_name'],
            'fuzzy_score': fuzzy_score,
            'domain_match': 0,
            'name_match': 0,
            'match': 1 if fuzzy_score >= match_threshold else 0
        })
        
        # Add to result DataFrame
        result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
    
    return result

def extract_missing_leads(result_df):
    """
    Extract records that exist in CDR.fyi but not in Attio.
    
    Args:
        result_df (pd.DataFrame): The merged dataframe with match indicators
        
    Returns:
        pd.DataFrame: DataFrame containing only the missing leads
    """
    # Find rows where:
    # 1. There's a CDR ID (meaning it's a CDR record)
    # 2. There's no matching Attio ID or match is 0 (meaning it wasn't found in Attio)
    missing_leads = result_df[
        (result_df['cdr_id'].notna()) & 
        ((result_df['attio_id'].isna()) | (result_df['match'] == 0))
    ].copy()
    
    # Sort by relevant columns (e.g., company name)
    if 'cdr_name' in missing_leads.columns:
        missing_leads = missing_leads.sort_values('cdr_name')
    
    # Select only the relevant columns for the missing leads report
    missing_leads_cols = [
            'cdr_name', 'cdr_website',
            'cdr_tons_purchased', 'cdr_total_spent_usd', 'cdr_methods', 
            'cdr_order_statuses', 'cdr_agreement_types'
        ]
    
    missing_leads = missing_leads[missing_leads_cols]
    missing_leads = missing_leads.rename(columns={
        'cdr_name': 'purchaser_name',
        'cdr_website': 'website',
        'cdr_tons_purchased': 'tons_purchased',
        'cdr_total_spent_usd': 'total_spent_usd',
        'cdr_methods': 'methods',
        'cdr_order_statuses': 'order_statuses',
        'cdr_agreement_types': 'agreement_types'
    })
    
    missing_leads = missing_leads.sort_values('tons_purchased', ascending=False)

    return missing_leads

def print_summary_statistics(cdr_fyi_df, attio_df, missing_leads_df):
    # Print number of CDR FYI companues
    print(f"Number of CDR FYI companies: {len(cdr_fyi_df)}")
    # Print number of Attio companies
    print(f"Number of Attio companies: {len(attio_df)}")
    # Print number of missing leads
    print(f"Number of missing leads: {len(missing_leads_df)}")

def missing_leads_checker(attio_df, cdr_df, output_path, fuzzy_path, threshold=25):

    """
    Main function to check for missing leads between Attio and CDR FYI
    
    Args:
        attio_df: Attio companies dataframe
        cdr_df: CDR FYI companies dataframe
        output_path: Path to save the merged results
        fuzzy_path: Path to save the fuzzy matches
        
    Returns:
        DataFrame: Merged dataframe with match indicators
    """
    print(f"Processing {len(attio_df)} Attio companies and {len(cdr_df)} CDR FYI companies")

    # Create copies to avoid modifying originals
    attio = attio_df.copy()
    cdr = cdr_df.copy()

    # Clean company names
    attio['clean_name'] = attio['name'].apply(normalize_company_name)
    cdr['clean_name'] = cdr['purchaser_name'].apply(normalize_company_name)
    
    # Extract domains
    attio['clean_website'] = attio['domains'].apply(extract_domain)
    cdr['clean_website'] = cdr['website'].apply(extract_domain)
    
    # Select relevant columns
    attio_cols = ['id', 'name', 'clean_name', 'clean_website']
    cdr_cols = ['purchaser_id', 'purchaser_name', 'clean_name', 'website', 'clean_website', 'tons_purchased', 'total_spent_usd', 'cdr_methods', 'order_statuses', 'agreement_types']

    attio = attio[attio_cols]
    cdr = cdr[cdr_cols]

    # Add prefixes to column names to avoid conflicts during merge
    attio = attio.add_prefix('attio_')
    
    cdr = cdr.add_prefix('cdr_')
    cdr = cdr.rename(columns={
        'cdr_purchaser_id': 'cdr_id',
        'cdr_purchaser_name': 'cdr_name',
        'cdr_cdr_methods': 'cdr_methods'
    })
    
    # STEP 1: Exact Matching
    print("Step 1: Performing exact matching...")
    result = exact_match_companies(attio, cdr)
    result.to_csv( SALES_DIR / 'result_before_fuzzy.csv', index=False)
    exact_matches = result[result['match'] == 1]
    print(f"Found {len(exact_matches)} exact matches (name or domain)")

    # STEP 2: Fuzzy Matching for remaining CDR companies
    print("Step 2: Performing fuzzy matching for unmatched CDR companies...")
    matched_cdr_ids = exact_matches['cdr_id'].unique()
    unmatched_cdr = cdr[~cdr_df['purchaser_id'].isin(matched_cdr_ids)]
    print(f"Looking for fuzzy matches for {len(unmatched_cdr)} unmatched CDR companies")
    
    fuzzy_matches = fuzzy_match_companies(attio, unmatched_cdr, threshold)
    print(f"Found {len(fuzzy_matches)} potential fuzzy matches")

    print(f"Fuzzy match columns: {fuzzy_matches.columns.tolist()}")
    print(f"Result df columns: {result.columns.tolist()}")

    # STEP 3: Update matches with fuzzy results
    if not fuzzy_matches.empty:
        print("Step 3: Integrating fuzzy matches...")
        save_fuzzy_matches(fuzzy_matches, fuzzy_path)
        
        # Add fuzzy matches to results
        for _, row in fuzzy_matches.iterrows():
            # Update related rows or add new rows
            result = add_fuzzy_match_to_results(result, row, match_threshold=threshold)
    
    result.to_csv(SALES_DIR / "result_after_fuzzy.csv", index=False)

    # STEP 4: Generate missing leads report
    print("Step 4: Generating missing leads report...")
    missing_leads = extract_missing_leads(result)

    # Output summary statistics
    print_summary_statistics(cdr, attio, missing_leads)

    # Save to CSV
    if not missing_leads.empty:
        missing_leads.to_csv(output_path, index=False)
        print(f"Missing leads saved to {output_path}")
    else:
        print("No missing leads found!")
    
    return missing_leads

def write_leads_to_google_sheet(missing_leads_df, sheet_id):
    try:
        # Format the date for sheet name and timestamp
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        sheet_name = "raw"
        
        # Make a copy of the DataFrame to avoid modifying the original
        df_to_write = missing_leads_df.copy()
        
        # Convert complex data types to strings
        for column in df_to_write.columns:
            if df_to_write[column].apply(type).eq(dict).any() or df_to_write[column].apply(type).eq(list).any():
                df_to_write[column] = df_to_write[column].apply(lambda x: str(x) if x is not None else "")
        
        # Use the credentials
        credentials = service_account.Credentials.from_service_account_info(
            GOOGLE_SERVICE_ACCOUNT, 
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        
        # Build the service
        service = build('sheets', 'v4', credentials=credentials)
        
        metadata_values = [
            ["Last Updated:", timestamp],
            [""]  # Empty row as separator
        ]
        
        # Convert DataFrame to values list, adding metadata at the top
        data_values = [df_to_write.columns.tolist()]  # Header row
        data_values.extend(df_to_write.values.tolist())  # Data rows
        
        # Combine metadata and data values
        values = metadata_values + data_values
        
        # Create the body data
        body = {
            'values': values
        }
        
        # Create the body data
        body = {
            'values': values
        }
        
        # Check if sheet exists and create if needed
        sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
        sheets = sheet_metadata.get('sheets', '')
        sheet_exists = False
        
        for sheet in sheets:
            if sheet['properties']['title'] == sheet_name:
                sheet_exists = True
                break
        
        if not sheet_exists:
            request_body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': sheet_name
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=sheet_id,
                body=request_body
            ).execute()
        
        # Clear existing content and write new data
        range_name = f"{sheet_name}!A1:Z10000"
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range=range_name
        ).execute()
        
        result = service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A1",
            valueInputOption='RAW',
            body=body
        ).execute()
            
        print(f"Successfully wrote {result.get('updatedCells')} cells to Google Sheet.")
        print(f"Sheet URL: https://docs.google.com/spreadsheets/d/{sheet_id}")
        return True
        
    except Exception as e:
        print(f"Error writing to Google Sheet: {e}")
        if hasattr(e, 'content'):
            print(f"Detailed error: {e.content}")
        return False

def main():
    # Define the URL and credentials for Attio
    url = "https://api.attio.com/v2"

    # Pull data from Attio
    attio_data = pull_attio_data(url, ATTIO_API_TOKEN)

    # Process the data
    processed_attio_data = process_attio_data(attio_data)
    
    # Save to CSV
    output_path = os.path.join(SALES_DIR, "attio_companies.csv")
    processed_attio_data.to_csv(output_path, index=False)
    logger.info(f"Saved processed Attio data to {output_path}")

    # MDK uncomment to skip attio api part
    #processed_attio_data = pd.read_csv(os.path.join(SALES_DIR, "attio_companies_20250512.csv"))

    # Retrieve CDR.fyi purchasers
    purchasers_path = os.path.join(RAW_DIR, "cdr_fyi_purchasers.csv")
    orders_path = os.path.join(RAW_DIR, "cdr_fyi_orders.csv")
    cdr_fyi_purchasers = retrieve_cdr_fyi_purchasers(purchasers_path, orders_path)
    
    # Create paths for output files
    missing_leads_path = os.path.join(SALES_DIR, f"missing_leads.csv")
    fuzzy_matches_path = os.path.join(SALES_DIR, f"fuzzy_matches.csv")
    
    # Find missing leads and save fuzzy matches
    missing_leads = missing_leads_checker(
        processed_attio_data,  # Use processed data instead of raw data
        cdr_fyi_purchasers,
        output_path=missing_leads_path,
        fuzzy_path=fuzzy_matches_path,
        threshold=60
    )

    # Write missing leads to Google Sheets
    write_leads_to_google_sheet(missing_leads, sheet_id="1t2iux4ZjivaRAGaWaIi1dnCcVGIbkcn_0rQtnFrdKmQ")

if __name__ == "__main__":
    # Call this function before running the main matching process
    # debug_specific_match(cdr_id = 'd6994d5e-c05f-47eb-9b84-c0f21c03c5d6', attio_id = '48397f33-03f1-4b9d-9ae4-0c15db3e477b')
    main()
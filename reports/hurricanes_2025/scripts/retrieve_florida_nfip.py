import pandas as pd
import requests
import logging
from datetime import datetime, timezone
import os
import time

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_nfip_data(base_url, data_key, fields, state_filter, data_type, date_field=None, cutoff_date=None):
    """
    Generic function to fetch NFIP data from FEMA API with optimizations.
    """
    params = {
        '$filter': state_filter,
        '$format': 'json',
        '$select': ','.join(fields)
    }

    logging.info(f"Fetching Florida NFIP {data_type} data from FEMA API...")
    logging.info(f"API URL: {base_url}")
    logging.info(f"Filter: {state_filter}")
    logging.info(f"Fields: {', '.join(fields)}")

    all_data = []
    skip = 0
    top = 1000  # API pagination limit
    seen_ids = set()
    max_retries = 3
    # duplicate_threshold = 0.99  # Stop if >80% duplicates in a batch

    while True:
        params['$skip'] = skip
        params['$top'] = top

        # Retry logic for API failures
        for attempt in range(max_retries):
            try:
                logging.debug(f"API Request (attempt {attempt + 1}): {base_url} with params: {params}")
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logging.error(f"API request failed after {max_retries} attempts: {e}")
                    return pd.DataFrame(all_data)
                else:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logging.warning(f"API request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)

        data = response.json()

        if not data.get(data_key):
            logging.info(f"No more {data_type} data available from API")
            break

        records = data[data_key]
        if not records:
            logging.info(f"Empty batch received, stopping")
            break

        # Process records and check for duplicates
        duplicate_count = 0
        new_records = []

        for record in records:
            record_id = record.get('id')
            if record_id in seen_ids:
                duplicate_count += 1
            else:
                seen_ids.add(record_id)

                # Check date cutoff if specified
                if date_field and cutoff_date and record.get(date_field):
                    try:
                        record_date = pd.to_datetime(record[date_field])
                        if record_date < cutoff_date:
                            logging.info(f"Reached records before cutoff date {cutoff_date}, stopping")
                            all_data.extend(new_records)
                            logging.info(f"Total {data_type} records retrieved: {len(all_data)}")
                            return pd.DataFrame(all_data)
                    except:
                        pass  # Continue if date parsing fails

                new_records.append(record)

        all_data.extend(new_records)

        # Log progress with reduced verbosity for duplicates
        if duplicate_count > 0:
            duplicate_rate = duplicate_count / len(records)
            logging.info(f"Retrieved {len(records)} {data_type} records, {len(new_records)} unique, {duplicate_count} duplicates ({duplicate_rate:.1%}) (total unique: {len(all_data)})")

            # # Stop if too many duplicates
            # if duplicate_rate > duplicate_threshold:
            #     logging.warning(f"High duplicate rate ({duplicate_rate:.1%}), likely reached end of useful data")
            #     break
        else:
            logging.info(f"Retrieved {len(records)} {data_type} records, {len(new_records)} unique (total unique: {len(all_data)})")

        # Show sample record on first batch for validation
        if skip == 0 and records:
            logging.info(f"Sample {data_type} record: {records[0]}")

        # If we got fewer records than requested, we've reached the end
        if len(records) < top:
            logging.info(f"Retrieved final batch of {len(records)} records")
            break

        skip += top

    logging.info(f"Total {data_type} records retrieved: {len(all_data)}")
    logging.info(f"Total unique IDs: {len(seen_ids)}")
    return pd.DataFrame(all_data)

def get_florida_nfip_claims():
    """
    Retrieve Florida NFIP flood claims data from FEMA API.
    Returns a pandas DataFrame with specified columns.
    """
    base_url = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
    fields = [
        'id',
        'dateOfLoss',
        'occupancyType',
        'amountPaidOnBuildingClaim',
        'amountPaidOnContentsClaim',
        'yearOfLoss',
        'causeOfDamage',
        'floodEvent',
        'state',
        'latitude',
        'longitude',
        'contentsDamageAmount',
        'buildingDamageAmount',
        'primaryResidenceIndicator'
    ]

    # Filter for Florida claims after 2009 with non-zero damage amounts
    filter_query = ("state eq 'FL' and yearOfLoss gt 2009 and "
                   "(contentsDamageAmount gt 0 or buildingDamageAmount gt 0 or "
                   "amountPaidOnBuildingClaim gt 0 or amountPaidOnContentsClaim gt 0)")

    cutoff_date = pd.to_datetime('2020-01-01')
    return fetch_nfip_data(base_url, 'FimaNfipClaims', fields, filter_query, "claims",
                          date_field='dateOfLoss', cutoff_date=cutoff_date)

def get_florida_nfip_policies():
    """
    Retrieve Florida NFIP policies data from FEMA API.
    Returns a pandas DataFrame with specified columns.
    """
    base_url = "https://www.fema.gov/api/open/v2/FimaNfipPolicies"
    fields = [
        'id',
        'cancellationDateOfFloodPolicy',
        'occupancyType',
        'originalNBDate',
        'policyCost',
        'policyCount',
        'policyEffectiveDate',
        'policyTerminationDate',
        'primaryResidenceIndicator',
        'totalInsurancePremiumOfThePolicy',
        'propertyState',
        'primaryResidenceIndicator',
        'totalBuildingInsuranceCoverage',
        'totalContentsInsuranceCoverage'
    ]

    # Filter for Florida policies effective after 2009
    filter_query = "propertyState eq 'FL' and policyEffectiveDate gt 2009-12-31T23:59:59.999Z"

    cutoff_date = pd.to_datetime('2010-01-01')
    return fetch_nfip_data(base_url, 'FimaNfipPolicies', fields, filter_query, "policies",
                          date_field='policyEffectiveDate', cutoff_date=cutoff_date)

def save_data(df, output_dir, data_type):
    """Save the DataFrame to CSV file with timestamp."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"florida_nfip_{data_type}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    df.to_csv(filepath, index=False)
    logging.info(f"Data saved to {filepath}")

    return filepath

def inspect_data(df, data_type):
    """Inspect and validate the downloaded data."""
    logging.info(f"=== {data_type.upper()} DATA INSPECTION ===")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")

    # Check for Florida-specific data
    if 'propertyState' in df.columns:
        state_counts = df['propertyState'].value_counts()
        logging.info(f"State distribution: {state_counts.to_dict()}")
        non_fl_count = len(df[df['propertyState'] != 'FL'])
        if non_fl_count > 0:
            logging.warning(f"Found {non_fl_count} non-Florida records!")
    elif 'state' in df.columns:
        state_counts = df['state'].value_counts()
        logging.info(f"State distribution: {state_counts.to_dict()}")
        non_fl_count = len(df[df['state'] != 'FL'])
        if non_fl_count > 0:
            logging.warning(f"Found {non_fl_count} non-Florida records!")

    # Show sample data
    logging.info(f"Sample records:\n{df.head(2).to_string()}")

    # Check for missing data
    missing_data = df.isnull().sum()
    logging.info(f"Missing data counts:\n{missing_data[missing_data > 0]}")

def main():
    setup_logging()
    output_dir = "nfip"

    # Retrieve claims data
    logging.info("=== Retrieving NFIP Claims Data ===")
    claims_df = get_florida_nfip_claims()

    if not claims_df.empty:
        save_data(claims_df, output_dir, "claims")
        logging.info(f"Claims dataset shape: {claims_df.shape}")
        logging.info(f"Claims year range: {claims_df['yearOfLoss'].min()} - {claims_df['yearOfLoss'].max()}")
    else:
        logging.warning("No claims data retrieved")

    # Retrieve policies data
    logging.info("=== Retrieving NFIP Policies Data ===")
    policies_df = get_florida_nfip_policies()

    if not policies_df.empty:
        inspect_data(policies_df, "policies")
        save_data(policies_df, output_dir, "policies")
    else:
        logging.warning("No policies data retrieved")

if __name__ == "__main__":
    main()
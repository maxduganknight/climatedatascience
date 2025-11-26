"""
Analyze Franzmann et al. (2025) EGS placement data to extract national capacity estimates.

This script processes the 04_allPlacements.csv data from:
https://data.mendeley.com/datasets/sbs8k66bwf/2

Outputs:
- franzmann_national_capacities.csv: One row per country with aggregated capacities
- franzmann_regional_capacities.csv: Detailed sub-national breakdowns with region names
"""

import pandas as pd
import requests
import json
from pathlib import Path

# Load the data (this may take a moment - 631 MB file)
print("Loading Franzmann placement data...")
df = pd.read_csv('/Users/max/Downloads/04_allPlacements.csv')

print(f"Loaded {len(df):,} plant placements")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Extract country code from gid1 (e.g., "USA.44_1" -> "USA", "CHN" -> "CHN", "ABW" -> "ABW")
df['country_code'] = df['gid1'].str.split('.').str[0]

print(f"\nExtracted country codes from gid1")
print(f"Sample mappings:")
print(df[['gid1', 'country_code']].drop_duplicates().head(20))

# Download GADM region name lookup
print(f"\n{'='*70}")
print("DOWNLOADING GADM REGION NAME LOOKUP")
print(f"{'='*70}")

def download_gadm_names():
    """
    Download GADM level 1 administrative region names.
    GADM provides comprehensive global administrative boundaries.
    """
    gadm_lookup = {}

    # Get unique country codes
    countries = df['country_code'].unique()
    print(f"Found {len(countries)} unique countries to lookup")

    # GADM 4.1 base URL for JSON data
    base_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{}_1.json"

    # Try to download for each country
    successful = 0
    failed_countries = []

    for i, country in enumerate(sorted(countries)):
        if len(country) != 3:  # Skip if not a 3-letter ISO code
            continue

        url = base_url.format(country)

        try:
            print(f"  [{i+1}/{len(countries)}] Fetching {country}...", end=" ")
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Extract region names and codes
                for feature in data['features']:
                    props = feature['properties']
                    gid = props.get('GID_1', '')
                    name = props.get('NAME_1', '')

                    if gid and name:
                        gadm_lookup[gid] = name

                print(f"✓ ({len([k for k in gadm_lookup.keys() if k.startswith(country)])} regions)")
                successful += 1
            else:
                print(f"✗ (HTTP {response.status_code})")
                failed_countries.append(country)

        except Exception as e:
            print(f"✗ ({str(e)[:40]})")
            failed_countries.append(country)

    print(f"\nSuccessfully downloaded: {successful}/{len(countries)} countries")
    print(f"Total regions mapped: {len(gadm_lookup)}")

    if failed_countries:
        print(f"Failed countries ({len(failed_countries)}): {', '.join(failed_countries[:20])}")

    return gadm_lookup

gadm_lookup = download_gadm_names()

# Save the GADM lookup for future use
gadm_lookup_file = '/Users/max/Deep_Sky/GitHub/datascience-platform/fundraising_viz/data/energy/geothermal/gadm_level1_lookup.csv'
Path(gadm_lookup_file).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(list(gadm_lookup.items()), columns=['GADM_Code', 'Region_Name']).to_csv(gadm_lookup_file, index=False)
print(f"\nSaved GADM lookup to: {gadm_lookup_file}")

# The 'gid1' column contains country/region codes
# Let's look at what countries are represented
print(f"\n{'='*70}")
print("UNIQUE COUNTRIES/REGIONS")
print(f"{'='*70}")
unique_regions = df['gid1'].unique()
print(f"Number of unique regions: {len(unique_regions)}")
print(f"Sample regions: {list(unique_regions[:20])}")

# Aggregate by country using the Gringarten model (the economically viable one per paper)
print(f"\n{'='*70}")
print("REGIONAL CAPACITY ESTIMATES (Gringarten Model)")
print(f"{'='*70}")

regional_capacity = df.groupby('gid1').agg({
    'Pout_Gringarten_MW': 'sum',
    'LCOE_Gringarten_Eurct_per_kWh': 'mean',
    'lon': 'count'  # Count number of plants
}).reset_index()

regional_capacity.columns = ['Region_Code', 'Total_Capacity_MW', 'Avg_LCOE_EUR_kWh', 'Num_Plants']

# Convert MW to GW for easier reading
regional_capacity['Total_Capacity_GW'] = regional_capacity['Total_Capacity_MW'] / 1000

# Sort by capacity
regional_capacity = regional_capacity.sort_values('Total_Capacity_GW', ascending=False)

print("\nTop 30 regions by capacity:")
print(regional_capacity.head(30).to_string(index=False))

# Try to map some common country codes
# gid1 appears to use GADM codes (e.g., USA for United States, CHN for China, etc.)
print(f"\n{'='*70}")
print("SPECIFIC COUNTRY ESTIMATES (if identifiable)")
print(f"{'='*70}")

# Look for specific countries mentioned in the paper
countries_of_interest = {
    'USA': ['USA'],
    'China': ['CHN'],
    'Japan': ['JPN'],
    'Germany': ['DEU'],
    'Iceland': ['ISL'],
    'Brazil': ['BRA'],
    'Russia': ['RUS'],
    'Canada': ['CAN'],
    'Australia': ['AUS'],
}

for country_name, possible_codes in countries_of_interest.items():
    matching_regions = regional_capacity[regional_capacity['Region_Code'].str.startswith(tuple(possible_codes), na=False)]
    if len(matching_regions) > 0:
        total_capacity = matching_regions['Total_Capacity_GW'].sum()
        avg_lcoe = matching_regions['Avg_LCOE_EUR_kWh'].mean()
        num_plants = matching_regions['Num_Plants'].sum()
        print(f"\n{country_name}:")
        print(f"  Total Capacity: {total_capacity:.1f} GW")
        print(f"  Average LCOE: €{avg_lcoe:.3f}/kWh")
        print(f"  Number of plants: {num_plants:,}")
        print(f"  Regional breakdown:")
        for _, row in matching_regions.head(10).iterrows():
            print(f"    {row['Region_Code']}: {row['Total_Capacity_GW']:.2f} GW")

# Calculate global total
global_total_gw = regional_capacity['Total_Capacity_GW'].sum()
global_avg_lcoe = regional_capacity['Avg_LCOE_EUR_kWh'].mean()
global_num_plants = regional_capacity['Num_Plants'].sum()

print(f"\n{'='*70}")
print("GLOBAL TOTALS")
print(f"{'='*70}")
print(f"Total Global Capacity: {global_total_gw:.1f} GW ({global_total_gw/1000:.2f} TW)")
print(f"Average Global LCOE: €{global_avg_lcoe:.3f}/kWh")
print(f"Total Number of Plants: {global_num_plants:,}")

# Add region names to regional capacity data
print("\nAdding region names from GADM lookup...")
regional_capacity['Region_Name'] = regional_capacity['Region_Code'].map(gadm_lookup)

# Reorder columns to have name after code
cols = list(regional_capacity.columns)
cols.remove('Region_Name')
cols.insert(1, 'Region_Name')
regional_capacity = regional_capacity[cols]

# Show how many regions were successfully mapped
mapped_count = regional_capacity['Region_Name'].notna().sum()
total_count = len(regional_capacity)
print(f"Successfully mapped {mapped_count}/{total_count} regions ({mapped_count/total_count*100:.1f}%)")

# Save regional (sub-national) results to CSV with region names
regional_output_file = '/Users/max/Deep_Sky/GitHub/datascience-platform/fundraising_viz/data/energy/geothermal/franzmann_regional_capacities.csv'
regional_capacity.to_csv(regional_output_file, index=False)
print(f"\nSaved regional capacities with names to: {regional_output_file}")

# Now create NATIONAL aggregation (one row per country)
print(f"\n{'='*70}")
print("NATIONAL CAPACITY ESTIMATES (Aggregated by Country)")
print(f"{'='*70}")

national_capacity = df.groupby('country_code').agg({
    'Pout_Gringarten_MW': 'sum',
    'LCOE_Gringarten_Eurct_per_kWh': 'mean',
    'lon': 'count'  # Count number of plants
}).reset_index()

national_capacity.columns = ['Country_Code', 'Total_Capacity_MW', 'Avg_LCOE_EUR_kWh', 'Num_Plants']

# Convert MW to GW
national_capacity['Total_Capacity_GW'] = national_capacity['Total_Capacity_MW'] / 1000
national_capacity['Total_Capacity_TW'] = national_capacity['Total_Capacity_MW'] / 1_000_000

# Sort by capacity
national_capacity = national_capacity.sort_values('Total_Capacity_GW', ascending=False)

print("\nTop 50 countries by capacity:")
print(national_capacity.head(50).to_string(index=False))

# Save national results to CSV
national_output_file = '/Users/max/Deep_Sky/GitHub/datascience-platform/fundraising_viz/data/energy/geothermal/franzmann_national_capacities.csv'
national_capacity.to_csv(national_output_file, index=False)
print(f"\nSaved national capacities to: {national_output_file}")

# Also check the distribution of LCOE
print(f"\n{'='*70}")
print("LCOE DISTRIBUTION")
print(f"{'='*70}")
print(f"Min LCOE: €{df['LCOE_Gringarten_Eurct_per_kWh'].min():.3f}/kWh")
print(f"Max LCOE: €{df['LCOE_Gringarten_Eurct_per_kWh'].max():.3f}/kWh")
print(f"Median LCOE: €{df['LCOE_Gringarten_Eurct_per_kWh'].median():.3f}/kWh")
print(f"Mean LCOE: €{df['LCOE_Gringarten_Eurct_per_kWh'].mean():.3f}/kWh")

# Count plants below economic thresholds
below_5ct = (df['LCOE_Gringarten_Eurct_per_kWh'] < 0.05).sum()
below_10ct = (df['LCOE_Gringarten_Eurct_per_kWh'] < 0.10).sum()
capacity_below_10ct = df[df['LCOE_Gringarten_Eurct_per_kWh'] < 0.10]['Pout_Gringarten_MW'].sum() / 1000

print(f"\nPlants with LCOE < €0.05/kWh: {below_5ct:,} ({below_5ct/len(df)*100:.1f}%)")
print(f"Plants with LCOE < €0.10/kWh: {below_10ct:,} ({below_10ct/len(df)*100:.1f}%)")
print(f"Capacity with LCOE < €0.10/kWh: {capacity_below_10ct:.1f} GW")

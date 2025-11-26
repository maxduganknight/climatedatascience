#!/usr/bin/env python3
"""
Data Aggregator for Instacart Thanksgiving Analysis
Combines retailer-specific CSV files into a single standardized dataset.
"""

import re
from pathlib import Path

import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
RAW_DATA_DIR = Path("instacart_raw_data")
OUTPUT_DIR = Path("processed")
OUTPUT_FILE = OUTPUT_DIR / "combined.csv"

# ============================================================================
# PRICE STANDARDIZATION FUNCTIONS
# ============================================================================


def parse_price_to_numeric(price_str):
    """
    Parse various price formats into a numeric value.

    Examples:
        "$0.47/lb" -> 0.47
        "$2.88" -> 2.88
        "2 for $5" -> 2.50
        "FREE" -> 0.0
        "$0.00" -> 0.0
        "2.5" -> 2.50
    """
    if pd.isna(price_str) or price_str == "":
        return None

    price_str = str(price_str).strip()

    # Handle FREE
    if price_str.upper() == "FREE":
        return 0.0

    # Handle "X for $Y" (e.g., "2 for $5")
    multibuy_match = re.search(r"(\d+)\s*for\s*\$?([\d.]+)", price_str, re.IGNORECASE)
    if multibuy_match:
        quantity = float(multibuy_match.group(1))
        total_price = float(multibuy_match.group(2))
        return round(total_price / quantity, 2)

    # Handle "$X/lb" or "$X per lb" (return per-unit price)
    per_lb_match = re.search(
        r"\$?([\d.]+)\s*[/]?\s*(per\s+)?lb", price_str, re.IGNORECASE
    )
    if per_lb_match:
        return float(per_lb_match.group(1))

    # Handle "$X/oz" or "$X per oz"
    per_oz_match = re.search(
        r"\$?([\d.]+)\s*[/]?\s*(per\s+)?oz", price_str, re.IGNORECASE
    )
    if per_oz_match:
        return float(per_oz_match.group(1))

    # Handle standard dollar amounts "$X.XX" or "X.XX"
    dollar_match = re.search(r"\$?([\d.]+)", price_str)
    if dollar_match:
        return float(dollar_match.group(1))

    return None


def standardize_promo_type(promo_type):
    """Standardize promo type values."""
    if pd.isna(promo_type) or promo_type == "":
        return "UNKNOWN"
    return str(promo_type).strip().upper()


# ============================================================================
# MAIN AGGREGATION LOGIC
# ============================================================================


def combine_data():
    """Combine all retailer CSV files into a single dataset."""

    all_data = []

    # Find all CSV files matching pattern
    csv_files = sorted(RAW_DATA_DIR.glob("*_Thanksgiving_*.csv"))

    print(f"Found {len(csv_files)} CSV files to process:")

    for csv_file in csv_files:
        # Extract retailer and year from filename
        # Format: {Retailer}_Thanksgiving_{Year}.csv
        filename = csv_file.stem
        parts = filename.split("_Thanksgiving_")

        if len(parts) != 2:
            print(f"  ⚠ Skipping {csv_file.name} - unexpected filename format")
            continue

        retailer = parts[0]
        year = parts[1]

        print(f"  Processing {retailer} ({year})...")

        # Read CSV (handle potential quoting issues)
        try:
            df = pd.read_csv(csv_file, quoting=1, on_bad_lines="skip")

            # Drop any unnamed columns from trailing commas
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

            # Add retailer column
            df["Retailer"] = retailer

            # Verify year column exists and matches filename
            if "Year" in df.columns:
                # Use year from data, not filename (in case of mixed years)
                pass
            else:
                print(f"    ⚠ No Year column found, using {year} from filename")
                df["Year"] = int(year)

            # Add to collection
            all_data.append(df)
            print(f"    ✓ Loaded {len(df)} rows")

        except Exception as e:
            print(f"    ✗ Error reading file: {e}")
            continue

    if not all_data:
        raise ValueError("No data files were successfully loaded!")

    # Combine all dataframes
    print("\nCombining data...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Standardize promo types
    print("Standardizing promo types...")
    combined_df["Promo_Type_Std"] = combined_df["Promo_Type"].apply(
        standardize_promo_type
    )

    # Parse and standardize prices
    print("Parsing and standardizing prices...")
    combined_df["Price_Per_Unit"] = combined_df["Promo_Price"].apply(
        parse_price_to_numeric
    )

    # Also parse Net_Price if it exists
    if "Net_Price" in combined_df.columns:
        combined_df["Net_Price_Numeric"] = combined_df["Net_Price"].apply(
            parse_price_to_numeric
        )

    # Remove any trailing commas from headers (fix for FoodLion)
    combined_df.columns = combined_df.columns.str.rstrip(",")

    # Convert Year to int
    combined_df["Year"] = pd.to_numeric(combined_df["Year"], errors="coerce").astype(
        "Int64"
    )

    # Sort by Retailer, Year, Page
    combined_df = combined_df.sort_values(["Retailer", "Year", "Page"]).reset_index(
        drop=True
    )

    return combined_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("INSTACART THANKSGIVING DATA AGGREGATOR")
    print("=" * 70)
    print()

    # Combine data
    combined_df = combine_data()

    # Save to CSV
    print(f"\nSaving combined data to {OUTPUT_FILE}...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(combined_df):,}")
    print(f"Total retailers: {combined_df['Retailer'].nunique()}")
    print(f"Years covered: {sorted(combined_df['Year'].dropna().unique())}")
    print()
    print("Rows by Retailer and Year:")
    print(combined_df.groupby(["Retailer", "Year"]).size().to_string())
    print()
    print("Promo Type Distribution:")
    print(combined_df["Promo_Type_Std"].value_counts().to_string())
    print()
    print(
        f"Rows with standardized price: {combined_df['Price_Per_Unit'].notna().sum():,} ({combined_df['Price_Per_Unit'].notna().mean() * 100:.1f}%)"
    )
    print(
        f"Rows with standardized price: {combined_df['Price_Per_Unit'].notna().sum():,} ({combined_df['Price_Per_Unit'].notna().mean() * 100:.1f}%)"
    )
    print()
    print("✓ Data aggregation complete!")

#!/usr/bin/env python3
"""
Utility functions and constants for Instacart Thanksgiving analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
DATA_DIR = Path("processed")
FIGURES_DIR = Path("figures")
COMBINED_DATA_FILE = DATA_DIR / "combined.csv"

# Years
PRE_COVID_YEARS = [2019, 2020]
POST_COVID_YEAR = 2025

# Promo type categorization (better to worse for consumers)
PROMO_HIERARCHY = {
    "BOGO_FREE": 1,  # Best: Buy one get one free (100% off second item)
    "BUY 1 GET 1 FREE": 1,
    "BUY 2 GET 2 FREE": 1,
    "BOGO_PERCENT": 2,  # Good: BOGO with percentage off
    "PERCENT_OFF": 3,  # Good: Percentage discount
    "DOLLAR_OFF": 4,  # Good: Fixed dollar discount
    "MULTIBUY": 5,  # Moderate: Buy multiple for discount
    "MIX_MATCH": 6,  # Moderate: Mix and match deals
    "STRAIGHT_PRICE": 7,  # Basic: Straight reduced price
    "DIGITAL_ONLY": 8,  # Conditional: Requires digital coupon
    "BUNDLE": 9,  # Conditional: Must buy bundle
    "SPEND_GET": 10,  # Conditional: Spend threshold required
    "MEMBER_PRICE": 11,  # Conditional: Membership required
    "CLUB PRICE": 11,
    "POINTS_MULTIPLIER": 12,  # Worst: Only points, no immediate savings
    "INSTANT_REBATE": 13,  # Worst: Delayed savings
    "UNKNOWN": 99,  # Unknown
}

# Deal quality tiers
DEAL_QUALITY_TIERS = {
    "Premium": ["BOGO_FREE", "BUY 1 GET 1 FREE", "BUY 2 GET 2 FREE"],
    "Strong": ["BOGO_PERCENT", "PERCENT_OFF", "DOLLAR_OFF"],
    "Moderate": ["MULTIBUY", "MIX_MATCH", "STRAIGHT_PRICE"],
    "Conditional": [
        "DIGITAL_ONLY",
        "BUNDLE",
        "SPEND_GET",
        "MEMBER_PRICE",
        "CLUB PRICE",
    ],
    "Weak": ["POINTS_MULTIPLIER", "INSTANT_REBATE"],
}

# Product categories for convergence analysis
PRODUCT_CATEGORIES = {
    "Turkey": ["turkey", "whole turkey", "frozen turkey", "fresh turkey"],
    "Ham": ["ham", "spiral ham", "shank", "butt portion"],
    "Potatoes": ["potato", "potatoes", "russet", "sweet potato", "yams"],
    "Cranberries": ["cranberry", "cranberries", "cranberry sauce"],
    "Stuffing": ["stuffing", "stove top"],
    "Rolls": ["roll", "rolls", "hawaiian rolls", "dinner rolls"],
    "Pie": ["pie", "pumpkin pie"],
    "Green Beans": ["green beans", "green bean"],
    "Gravy": ["gravy"],
    "Butter": ["butter"],
    "Cream": ["heavy cream", "whipping cream", "half and half"],
    "Wine": ["wine"],
    "Beer": ["beer"],
    "Soda": ["soda", "cola", "pepsi", "coke", "sprite", "7-up"],
}

# ============================================================================
# DATA LOADING
# ============================================================================


def load_combined_data():
    """Load the combined dataset."""
    df = pd.read_csv(COMBINED_DATA_FILE)
    return df


def filter_by_year_period(df, period="pre"):
    """
    Filter data by time period.

    Args:
        df: DataFrame to filter
        period: 'pre' for pre-COVID (2019/2020) or 'post' for post-COVID (2025)

    Returns:
        Filtered DataFrame
    """
    if period == "pre":
        return df[df["Year"].isin(PRE_COVID_YEARS)]
    elif period == "post":
        return df[df["Year"] == POST_COVID_YEAR]
    else:
        raise ValueError("period must be 'pre' or 'post'")


# ============================================================================
# PROMO TYPE ANALYSIS
# ============================================================================


def get_deal_quality_tier(promo_type):
    """Get the quality tier for a given promo type."""
    for tier, promos in DEAL_QUALITY_TIERS.items():
        if promo_type in promos:
            return tier
    return "Other"


def add_deal_quality_tier(df):
    """Add deal quality tier column to dataframe."""
    df = df.copy()
    df["Deal_Quality"] = df["Promo_Type_Std"].apply(get_deal_quality_tier)
    return df


def get_promo_hierarchy_score(promo_type):
    """Get hierarchy score for a promo type (lower is better for consumer)."""
    return PROMO_HIERARCHY.get(promo_type, 99)


# ============================================================================
# PRODUCT CATEGORIZATION
# ============================================================================


def categorize_product(product_name):
    """
    Categorize a product based on its name.

    Args:
        product_name: Product name string

    Returns:
        Category name or 'Other'
    """
    if pd.isna(product_name):
        return "Other"

    product_lower = str(product_name).lower()

    for category, keywords in PRODUCT_CATEGORIES.items():
        if any(keyword in product_lower for keyword in keywords):
            return category

    return "Other"


def add_product_category(df):
    """Add product category column to dataframe."""
    df = df.copy()
    df["Product_Category"] = df["Product"].apply(categorize_product)
    return df


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================


def setup_plot_style():
    """Set up consistent matplotlib style for all plots."""
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def save_figure(filename, dpi=300):
    """
    Save figure to the figures directory.

    Args:
        filename: Name of file to save (without path)
        dpi: Resolution for saved figure
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    filepath = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {filepath}")


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================


def calculate_percent_change(old_value, new_value):
    """Calculate percent change from old to new value."""
    if old_value == 0:
        return None
    return ((new_value - old_value) / old_value) * 100


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

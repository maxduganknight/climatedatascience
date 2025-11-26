#!/usr/bin/env python3
"""
Worse Deals Analysis
Analyzes whether promotional deals have gotten worse for consumers between
pre-COVID (2019/2020) and post-COVID (2025) periods.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    DEAL_QUALITY_TIERS,
    add_deal_quality_tier,
    calculate_percent_change,
    filter_by_year_period,
    get_promo_hierarchy_score,
    load_combined_data,
    print_section_header,
    save_figure,
    setup_plot_style,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set to True to exclude digital-only and other conditional deals
EXCLUDE_CONDITIONAL = False

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def analyze_deal_type_distribution(df_pre, df_post):
    """Analyze the distribution of deal types between periods."""

    print_section_header("DEAL TYPE DISTRIBUTION ANALYSIS")

    # Get top deal types
    all_promo_types = pd.concat(
        [df_pre["Promo_Type_Std"], df_post["Promo_Type_Std"]]
    ).value_counts()
    top_promo_types = all_promo_types.head(10).index.tolist()

    # Calculate distribution for each period
    pre_dist = df_pre["Promo_Type_Std"].value_counts(normalize=True) * 100
    post_dist = df_post["Promo_Type_Std"].value_counts(normalize=True) * 100

    # Compare distributions
    comparison = pd.DataFrame(
        {"Pre-COVID %": pre_dist, "Post-COVID %": post_dist}
    ).fillna(0)

    comparison["Change (pp)"] = comparison["Post-COVID %"] - comparison["Pre-COVID %"]
    comparison = comparison.loc[top_promo_types].round(2)

    print("\nTop 10 Promo Type Distribution Changes:")
    print(comparison.to_string())

    # Key findings
    print("\nKEY FINDINGS:")
    bogo_free_change = (
        comparison.loc["BOGO_FREE", "Change (pp)"]
        if "BOGO_FREE" in comparison.index
        else 0
    )
    bogo_pct_change = (
        comparison.loc["BOGO_PERCENT", "Change (pp)"]
        if "BOGO_PERCENT" in comparison.index
        else 0
    )

    if bogo_free_change < 0:
        print(
            f"  • BOGO FREE deals decreased by {abs(bogo_free_change):.1f} percentage points"
        )
    else:
        print(
            f"  • BOGO FREE deals increased by {bogo_free_change:.1f} percentage points"
        )

    if "BOGO_PERCENT" in comparison.index:
        if bogo_pct_change > 0:
            print(
                f"  • BOGO PERCENT deals increased by {bogo_pct_change:.1f} percentage points"
            )
        else:
            print(
                f"  • BOGO PERCENT deals decreased by {abs(bogo_pct_change):.1f} percentage points"
            )

    return comparison


def analyze_deal_quality_tiers(df_pre, df_post):
    """Analyze deal quality by tier."""

    print_section_header("DEAL QUALITY TIER ANALYSIS")

    # Add quality tiers
    df_pre = add_deal_quality_tier(df_pre)
    df_post = add_deal_quality_tier(df_post)

    # Calculate distribution by tier
    pre_tiers = df_pre["Deal_Quality"].value_counts(normalize=True) * 100
    post_tiers = df_post["Deal_Quality"].value_counts(normalize=True) * 100

    tier_comparison = pd.DataFrame(
        {"Pre-COVID %": pre_tiers, "Post-COVID %": post_tiers}
    ).fillna(0)

    tier_comparison["Change (pp)"] = (
        tier_comparison["Post-COVID %"] - tier_comparison["Pre-COVID %"]
    )

    # Sort by quality (Premium first)
    tier_order = ["Premium", "Strong", "Moderate", "Conditional", "Weak", "Other"]
    tier_comparison = tier_comparison.reindex(
        [t for t in tier_order if t in tier_comparison.index]
    )

    print("\nDeal Quality Tier Distribution:")
    print(tier_comparison.round(2).to_string())

    # Key findings
    print("\nKEY FINDINGS:")
    premium_change = (
        tier_comparison.loc["Premium", "Change (pp)"]
        if "Premium" in tier_comparison.index
        else 0
    )
    conditional_change = (
        tier_comparison.loc["Conditional", "Change (pp)"]
        if "Conditional" in tier_comparison.index
        else 0
    )

    if premium_change < 0:
        print(
            f"  • Premium deals (BOGO FREE) decreased by {abs(premium_change):.1f} percentage points"
        )
    else:
        print(
            f"  • Premium deals (BOGO FREE) increased by {premium_change:.1f} percentage points"
        )

    if conditional_change > 0:
        print(
            f"  • Conditional deals (digital/membership required) increased by {conditional_change:.1f} percentage points"
        )
    else:
        print(
            f"  • Conditional deals (digital/membership required) decreased by {abs(conditional_change):.1f} percentage points"
        )

    return tier_comparison


def analyze_by_retailer(df_pre, df_post):
    """Analyze deal quality changes by retailer."""

    print_section_header("RETAILER-LEVEL DEAL QUALITY ANALYSIS")

    # Add quality tiers
    df_pre = add_deal_quality_tier(df_pre)
    df_post = add_deal_quality_tier(df_post)

    # Calculate premium deal percentage by retailer
    retailers = sorted(df_pre["Retailer"].unique())

    retailer_analysis = []
    for retailer in retailers:
        pre_ret = df_pre[df_pre["Retailer"] == retailer]
        post_ret = df_post[df_post["Retailer"] == retailer]

        if len(post_ret) == 0:
            continue

        pre_premium_pct = (pre_ret["Deal_Quality"] == "Premium").mean() * 100
        post_premium_pct = (post_ret["Deal_Quality"] == "Premium").mean() * 100

        pre_conditional_pct = (pre_ret["Deal_Quality"] == "Conditional").mean() * 100
        post_conditional_pct = (post_ret["Deal_Quality"] == "Conditional").mean() * 100

        retailer_analysis.append(
            {
                "Retailer": retailer,
                "Pre Premium %": pre_premium_pct,
                "Post Premium %": post_premium_pct,
                "Premium Change": post_premium_pct - pre_premium_pct,
                "Pre Conditional %": pre_conditional_pct,
                "Post Conditional %": post_conditional_pct,
                "Conditional Change": post_conditional_pct - pre_conditional_pct,
            }
        )

    retailer_df = pd.DataFrame(retailer_analysis).sort_values("Premium Change")

    print("\nRetailer Premium Deal Changes:")
    print(
        retailer_df[["Retailer", "Pre Premium %", "Post Premium %", "Premium Change"]]
        .round(2)
        .to_string(index=False)
    )

    print("\nRetailer Conditional Deal Changes:")
    print(
        retailer_df[
            [
                "Retailer",
                "Pre Conditional %",
                "Post Conditional %",
                "Conditional Change",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    return retailer_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_deal_type_comparison(comparison):
    """Create bar chart comparing deal type distributions."""

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(comparison))
    width = 0.35

    ax.bar(
        x - width / 2,
        comparison["Pre-COVID %"],
        width,
        label="Pre-COVID (2019/2020)",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        comparison["Post-COVID %"],
        width,
        label="Post-COVID (2025)",
        alpha=0.8,
    )

    ax.set_xlabel("Promotion Type")
    ax.set_ylabel("Percentage of Deals")
    ax.set_title("Promotional Deal Type Distribution: Pre-COVID vs Post-COVID")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure("deal_type_distribution.png")
    plt.close()


def plot_deal_quality_tiers(tier_comparison):
    """Create stacked bar chart showing deal quality tiers."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie charts for each period
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#95a5a6"]

    ax1.pie(
        tier_comparison["Pre-COVID %"],
        labels=tier_comparison.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax1.set_title("Pre-COVID Deal Quality Distribution")

    ax2.pie(
        tier_comparison["Post-COVID %"],
        labels=tier_comparison.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax2.set_title("Post-COVID Deal Quality Distribution")

    save_figure("deal_quality_tiers.png")
    plt.close()


def plot_retailer_premium_changes(retailer_df):
    """Plot premium deal changes by retailer."""

    fig, ax = plt.subplots(figsize=(12, 6))

    retailer_df_sorted = retailer_df.sort_values("Premium Change")
    colors = [
        "#e74c3c" if x < 0 else "#2ecc71" for x in retailer_df_sorted["Premium Change"]
    ]

    ax.barh(
        retailer_df_sorted["Retailer"],
        retailer_df_sorted["Premium Change"],
        color=colors,
        alpha=0.7,
    )
    ax.set_xlabel("Change in Premium Deals (percentage points)")
    ax.set_ylabel("Retailer")
    ax.set_title("Change in Premium Deals (BOGO FREE) by Retailer")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    save_figure("retailer_premium_changes.png")
    plt.close()


def plot_retailer_conditional_changes(retailer_df):
    """Plot conditional deal changes by retailer."""

    fig, ax = plt.subplots(figsize=(12, 6))

    retailer_df_sorted = retailer_df.sort_values("Conditional Change")
    colors = [
        "#2ecc71" if x < 0 else "#e74c3c"
        for x in retailer_df_sorted["Conditional Change"]
    ]

    ax.barh(
        retailer_df_sorted["Retailer"],
        retailer_df_sorted["Conditional Change"],
        color=colors,
        alpha=0.7,
    )
    ax.set_xlabel("Change in Conditional Deals (percentage points)")
    ax.set_ylabel("Retailer")
    ax.set_title(
        "Change in Conditional Deals (Digital/Membership Required) by Retailer"
    )
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    save_figure("retailer_conditional_changes.png")
    plt.close()


def plot_deal_type_changes_waterfall(comparison):
    """Create waterfall chart showing changes in deal types."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by magnitude of change
    comparison_sorted = comparison.sort_values("Change (pp)", ascending=False).head(10)

    colors = [
        "#2ecc71" if x > 0 else "#e74c3c" for x in comparison_sorted["Change (pp)"]
    ]

    ax.bar(
        range(len(comparison_sorted)),
        comparison_sorted["Change (pp)"],
        color=colors,
        alpha=0.7,
    )
    ax.set_xticks(range(len(comparison_sorted)))
    ax.set_xticklabels(comparison_sorted.index, rotation=45, ha="right")
    ax.set_ylabel("Change in Deal Prevalence (percentage points)")
    ax.set_title("Top Changes in Deal Types: Pre-COVID to Post-COVID")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    save_figure("deal_type_changes.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WORSE DEALS ANALYSIS")
    print("Analyzing changes in promotional deal quality")
    print("=" * 70)

    # Setup
    setup_plot_style()

    # Load data
    print("\nLoading data...")
    df = load_combined_data()

    df_pre = filter_by_year_period(df, "pre")
    df_post = filter_by_year_period(df, "post")
    df_post = filter_by_year_period(df, "post")

    print(f"  Pre-COVID period: {len(df_pre):,} rows (2019-2020)")
    print(f"  Post-COVID period: {len(df_post):,} rows (2025)")

    # Run analyses
    deal_type_comparison = analyze_deal_type_distribution(df_pre, df_post)
    tier_comparison = analyze_deal_quality_tiers(df_pre, df_post)
    retailer_analysis = analyze_by_retailer(df_pre, df_post)

    # Generate visualizations
    print_section_header("GENERATING VISUALIZATIONS")
    plot_deal_type_comparison(deal_type_comparison)
    plot_deal_quality_tiers(tier_comparison)
    plot_retailer_premium_changes(retailer_analysis)
    plot_retailer_conditional_changes(retailer_analysis)
    plot_deal_type_changes_waterfall(deal_type_comparison)

    # Summary
    print_section_header("SUMMARY")
    print("Analysis complete! Check the figures/ directory for visualizations.")
    print("\nFiles generated:")
    print("  - deal_type_distribution.png")
    print("  - deal_quality_tiers.png")
    print("  - retailer_premium_changes.png")
    print("  - retailer_conditional_changes.png")
    print("  - deal_type_changes.png")

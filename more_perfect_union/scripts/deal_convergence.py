#!/usr/bin/env python3
"""
Deal Convergence Analysis
Analyzes convergence in prices and promotional strategies across retailers and brands.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils import (
    PRODUCT_CATEGORIES,
    add_deal_quality_tier,
    add_product_category,
    filter_by_year_period,
    load_combined_data,
    print_section_header,
    save_figure,
    setup_plot_style,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Minimum sample size for category analysis
MIN_SAMPLE_SIZE = 10

# Product categories to analyze in detail
FOCUS_CATEGORIES = [
    "Turkey",
    "Ham",
    "Potatoes",
    "Cranberries",
    "Rolls",
    "Wine",
    "Beer",
    "Soda",
]

# ============================================================================
# PRICE CONVERGENCE ANALYSIS
# ============================================================================


def analyze_price_convergence_by_category(df_pre, df_post):
    """Analyze price convergence across retailers for each product category."""

    print_section_header("PRICE CONVERGENCE ANALYSIS BY CATEGORY")

    # Add categories
    df_pre = add_product_category(df_pre)
    df_post = add_product_category(df_post)

    # Filter to rows with valid prices
    df_pre = df_pre[df_pre["Price_Per_Unit"].notna()]
    df_post = df_post[df_post["Price_Per_Unit"].notna()]

    results = []

    for category in FOCUS_CATEGORIES:
        pre_cat = df_pre[df_pre["Product_Category"] == category]
        post_cat = df_post[df_post["Product_Category"] == category]

        if len(pre_cat) < MIN_SAMPLE_SIZE or len(post_cat) < MIN_SAMPLE_SIZE:
            continue

        # Calculate coefficient of variation (CV) as measure of dispersion
        pre_prices = pre_cat["Price_Per_Unit"]
        post_prices = post_cat["Price_Per_Unit"]

        pre_cv = (
            (pre_prices.std() / pre_prices.mean()) * 100
            if pre_prices.mean() > 0
            else np.nan
        )
        post_cv = (
            (post_prices.std() / post_prices.mean()) * 100
            if post_prices.mean() > 0
            else np.nan
        )

        # Calculate price variance across retailers
        pre_retailer_means = pre_cat.groupby("Retailer")["Price_Per_Unit"].mean()
        post_retailer_means = post_cat.groupby("Retailer")["Price_Per_Unit"].mean()

        pre_retailer_cv = (
            (pre_retailer_means.std() / pre_retailer_means.mean()) * 100
            if pre_retailer_means.mean() > 0
            else np.nan
        )
        post_retailer_cv = (
            (post_retailer_means.std() / post_retailer_means.mean()) * 100
            if post_retailer_means.mean() > 0
            else np.nan
        )

        results.append(
            {
                "Category": category,
                "Pre_Mean_Price": pre_prices.mean(),
                "Post_Mean_Price": post_prices.mean(),
                "Pre_CV": pre_cv,
                "Post_CV": post_cv,
                "CV_Change": post_cv - pre_cv,
                "Pre_Retailer_CV": pre_retailer_cv,
                "Post_Retailer_CV": post_retailer_cv,
                "Retailer_CV_Change": post_retailer_cv - pre_retailer_cv,
                "Pre_Count": len(pre_cat),
                "Post_Count": len(post_cat),
                "Retailers_Pre": pre_cat["Retailer"].nunique(),
                "Retailers_Post": post_cat["Retailer"].nunique(),
            }
        )

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        print("\nPrice Variability Across All Products:")
        print(
            results_df[
                [
                    "Category",
                    "Pre_Mean_Price",
                    "Post_Mean_Price",
                    "Pre_CV",
                    "Post_CV",
                    "CV_Change",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )

        print("\nPrice Variability Across Retailers (by category mean):")
        print(
            results_df[
                [
                    "Category",
                    "Pre_Retailer_CV",
                    "Post_Retailer_CV",
                    "Retailer_CV_Change",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )

        print("\nKEY FINDINGS:")
        convergence_categories = results_df[results_df["Retailer_CV_Change"] < 0]
        divergence_categories = results_df[results_df["Retailer_CV_Change"] > 0]

        if len(convergence_categories) > 0:
            print(
                f"  • {len(convergence_categories)} categories show price CONVERGENCE across retailers:"
            )
            for _, row in convergence_categories.iterrows():
                print(
                    f"    - {row['Category']}: CV decreased by {abs(row['Retailer_CV_Change']):.1f}%"
                )

        if len(divergence_categories) > 0:
            print(
                f"  • {len(divergence_categories)} categories show price DIVERGENCE across retailers:"
            )
            for _, row in divergence_categories.iterrows():
                print(
                    f"    - {row['Category']}: CV increased by {row['Retailer_CV_Change']:.1f}%"
                )

    return results_df


def analyze_price_convergence_by_brand(df_pre, df_post):
    """Analyze price convergence across retailers for major brands."""

    print_section_header("PRICE CONVERGENCE ANALYSIS BY BRAND")

    # Filter to rows with brands and valid prices
    df_pre = df_pre[
        (df_pre["Brand"].notna())
        & (df_pre["Brand"] != "")
        & (df_pre["Price_Per_Unit"].notna())
    ]
    df_post = df_post[
        (df_post["Brand"].notna())
        & (df_post["Brand"] != "")
        & (df_post["Price_Per_Unit"].notna())
    ]

    # Get top brands that appear in both periods
    pre_brands = df_pre["Brand"].value_counts()
    post_brands = df_post["Brand"].value_counts()
    common_brands = set(pre_brands[pre_brands >= MIN_SAMPLE_SIZE].index) & set(
        post_brands[post_brands >= MIN_SAMPLE_SIZE].index
    )

    results = []

    for brand in sorted(common_brands):
        pre_brand = df_pre[df_pre["Brand"] == brand]
        post_brand = df_post[df_post["Brand"] == brand]

        # Calculate price variance across retailers for this brand
        pre_retailer_means = pre_brand.groupby("Retailer")["Price_Per_Unit"].mean()
        post_retailer_means = post_brand.groupby("Retailer")["Price_Per_Unit"].mean()

        # Need at least 2 retailers for variance
        if len(pre_retailer_means) < 2 or len(post_retailer_means) < 2:
            continue

        pre_cv = (
            (pre_retailer_means.std() / pre_retailer_means.mean()) * 100
            if pre_retailer_means.mean() > 0
            else np.nan
        )
        post_cv = (
            (post_retailer_means.std() / post_retailer_means.mean()) * 100
            if post_retailer_means.mean() > 0
            else np.nan
        )

        results.append(
            {
                "Brand": brand,
                "Pre_Mean_Price": pre_brand["Price_Per_Unit"].mean(),
                "Post_Mean_Price": post_brand["Price_Per_Unit"].mean(),
                "Pre_Retailer_CV": pre_cv,
                "Post_Retailer_CV": post_cv,
                "CV_Change": post_cv - pre_cv,
                "Retailers_Pre": len(pre_retailer_means),
                "Retailers_Post": len(post_retailer_means),
            }
        )

    results_df = pd.DataFrame(results).sort_values("CV_Change")

    if len(results_df) > 0:
        print(f"\nAnalyzed {len(results_df)} brands that appear in multiple retailers:")
        print("\nTop 10 Brands with Strongest Price Convergence:")
        print(
            results_df.head(10)[
                ["Brand", "Pre_Retailer_CV", "Post_Retailer_CV", "CV_Change"]
            ]
            .round(2)
            .to_string(index=False)
        )

        print("\nTop 10 Brands with Strongest Price Divergence:")
        print(
            results_df.tail(10)[
                ["Brand", "Pre_Retailer_CV", "Post_Retailer_CV", "CV_Change"]
            ]
            .round(2)
            .to_string(index=False)
        )

        print("\nKEY FINDINGS:")
        convergence_pct = (results_df["CV_Change"] < 0).mean() * 100
        print(
            f"  • {convergence_pct:.1f}% of brands show price convergence across retailers"
        )
        print(f"  • {100 - convergence_pct:.1f}% of brands show price divergence")

        avg_change = results_df["CV_Change"].mean()
        if avg_change < 0:
            print(f"  • Average CV change: {avg_change:.1f}% (convergence)")
        else:
            print(f"  • Average CV change: +{avg_change:.1f}% (divergence)")

    return results_df


# ============================================================================
# PROMO TYPE CONVERGENCE ANALYSIS
# ============================================================================


def analyze_promo_convergence_by_category(df_pre, df_post):
    """Analyze convergence in promotional strategies across retailers."""

    print_section_header("PROMOTIONAL STRATEGY CONVERGENCE BY CATEGORY")

    # Add categories and quality tiers
    df_pre = add_product_category(df_pre)
    df_post = add_product_category(df_post)
    df_pre = add_deal_quality_tier(df_pre)
    df_post = add_deal_quality_tier(df_post)

    results = []

    for category in FOCUS_CATEGORIES:
        pre_cat = df_pre[df_pre["Product_Category"] == category]
        post_cat = df_post[df_post["Product_Category"] == category]

        if len(pre_cat) < MIN_SAMPLE_SIZE or len(post_cat) < MIN_SAMPLE_SIZE:
            continue

        # Calculate diversity of promo types (Shannon entropy)
        pre_promo_dist = pre_cat["Promo_Type_Std"].value_counts(normalize=True)
        post_promo_dist = post_cat["Promo_Type_Std"].value_counts(normalize=True)

        pre_entropy = stats.entropy(pre_promo_dist)
        post_entropy = stats.entropy(post_promo_dist)

        # Calculate % Premium deals
        pre_premium_pct = (pre_cat["Deal_Quality"] == "Premium").mean() * 100
        post_premium_pct = (post_cat["Deal_Quality"] == "Premium").mean() * 100

        # Calculate variance in premium deal % across retailers
        pre_retailer_premium = pre_cat.groupby("Retailer", group_keys=False).apply(
            lambda x: (x["Deal_Quality"] == "Premium").mean() * 100,
            include_groups=False,
        )
        post_retailer_premium = post_cat.groupby("Retailer", group_keys=False).apply(
            lambda x: (x["Deal_Quality"] == "Premium").mean() * 100,
            include_groups=False,
        )

        results.append(
            {
                "Category": category,
                "Pre_Promo_Diversity": pre_entropy,
                "Post_Promo_Diversity": post_entropy,
                "Diversity_Change": post_entropy - pre_entropy,
                "Pre_Premium_%": pre_premium_pct,
                "Post_Premium_%": post_premium_pct,
                "Premium_Change": post_premium_pct - pre_premium_pct,
                "Pre_Retailer_Premium_Std": pre_retailer_premium.std(),
                "Post_Retailer_Premium_Std": post_retailer_premium.std(),
                "Retailer_Std_Change": post_retailer_premium.std()
                - pre_retailer_premium.std(),
            }
        )

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        print("\nPromotional Strategy Diversity (Shannon Entropy):")
        print(
            results_df[
                [
                    "Category",
                    "Pre_Promo_Diversity",
                    "Post_Promo_Diversity",
                    "Diversity_Change",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )

        print("\nPremium Deal Prevalence:")
        print(
            results_df[
                ["Category", "Pre_Premium_%", "Post_Premium_%", "Premium_Change"]
            ]
            .round(2)
            .to_string(index=False)
        )

        print("\nRetailer Convergence in Premium Deal Strategy:")
        print(
            results_df[
                [
                    "Category",
                    "Pre_Retailer_Premium_Std",
                    "Post_Retailer_Premium_Std",
                    "Retailer_Std_Change",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )

        print("\nKEY FINDINGS:")
        strategy_convergence = results_df[results_df["Retailer_Std_Change"] < 0]
        if len(strategy_convergence) > 0:
            print(
                f"  • {len(strategy_convergence)} categories show CONVERGENCE in promotional strategies:"
            )
            for _, row in strategy_convergence.iterrows():
                print(
                    f"    - {row['Category']}: Retailer std decreased by {abs(row['Retailer_Std_Change']):.1f}%"
                )

    return results_df


def analyze_overall_retailer_convergence(df_pre, df_post):
    """Analyze overall promotional strategy similarity across retailers."""

    print_section_header("OVERALL RETAILER PROMOTIONAL CONVERGENCE")

    df_pre = add_deal_quality_tier(df_pre)
    df_post = add_deal_quality_tier(df_post)

    # Calculate deal quality distribution for each retailer
    retailers = sorted(
        set(df_pre["Retailer"].unique()) & set(df_post["Retailer"].unique())
    )

    print(f"\nAnalyzing {len(retailers)} retailers that appear in both periods...")

    # Calculate average distance between retailer promo distributions
    def get_promo_distribution(df, retailer):
        ret_df = df[df["Retailer"] == retailer]
        return ret_df["Deal_Quality"].value_counts(normalize=True)

    # Calculate pairwise Jensen-Shannon divergence
    from scipy.spatial.distance import jensenshannon

    pre_divergences = []
    post_divergences = []

    quality_tiers = ["Premium", "Strong", "Moderate", "Conditional", "Weak", "Other"]

    for i, ret1 in enumerate(retailers):
        for ret2 in retailers[i + 1 :]:
            pre_dist1 = get_promo_distribution(df_pre, ret1).reindex(
                quality_tiers, fill_value=0
            )
            pre_dist2 = get_promo_distribution(df_pre, ret2).reindex(
                quality_tiers, fill_value=0
            )

            post_dist1 = get_promo_distribution(df_post, ret1).reindex(
                quality_tiers, fill_value=0
            )
            post_dist2 = get_promo_distribution(df_post, ret2).reindex(
                quality_tiers, fill_value=0
            )

            pre_div = jensenshannon(pre_dist1, pre_dist2)
            post_div = jensenshannon(post_dist1, post_dist2)

            if not np.isnan(pre_div) and not np.isnan(post_div):
                pre_divergences.append(pre_div)
                post_divergences.append(post_div)

    if len(pre_divergences) > 0:
        pre_mean_div = np.mean(pre_divergences)
        post_mean_div = np.mean(post_divergences)

        print(f"\nAverage pairwise promotional strategy divergence:")
        print(f"  Pre-COVID: {pre_mean_div:.4f}")
        print(f"  Post-COVID: {post_mean_div:.4f}")
        print(f"  Change: {post_mean_div - pre_mean_div:.4f}")

        if post_mean_div < pre_mean_div:
            pct_change = ((pre_mean_div - post_mean_div) / pre_mean_div) * 100
            print(
                f"\nKEY FINDING: Promotional strategies have CONVERGED by {pct_change:.1f}%"
            )
        else:
            pct_change = ((post_mean_div - pre_mean_div) / pre_mean_div) * 100
            print(
                f"\nKEY FINDING: Promotional strategies have DIVERGED by {pct_change:.1f}%"
            )

    return {
        "pre_mean_divergence": pre_mean_div if len(pre_divergences) > 0 else None,
        "post_mean_divergence": post_mean_div if len(post_divergences) > 0 else None,
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_price_convergence_by_category(results_df):
    """Plot price convergence across categories."""

    if len(results_df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: CV comparison
    categories = results_df["Category"]
    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(
        x - width / 2,
        results_df["Pre_Retailer_CV"],
        width,
        label="Pre-COVID",
        alpha=0.8,
    )
    ax1.bar(
        x + width / 2,
        results_df["Post_Retailer_CV"],
        width,
        label="Post-COVID",
        alpha=0.8,
    )
    ax1.set_xlabel("Product Category")
    ax1.set_ylabel("Coefficient of Variation (%)")
    ax1.set_title("Price Variability Across Retailers by Category")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Change in CV
    colors = [
        "#2ecc71" if x < 0 else "#e74c3c" for x in results_df["Retailer_CV_Change"]
    ]
    ax2.barh(categories, results_df["Retailer_CV_Change"], color=colors, alpha=0.7)
    ax2.set_xlabel("Change in CV (percentage points)")
    ax2.set_ylabel("Product Category")
    ax2.set_title("Price Convergence/Divergence by Category")
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)

    save_figure("price_convergence_by_category.png")
    plt.close()


def plot_brand_price_convergence(results_df, top_n=15):
    """Plot price convergence for top brands."""

    if len(results_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top converging and diverging brands
    top_converging = results_df.nsmallest(top_n, "CV_Change")
    top_diverging = results_df.nlargest(top_n, "CV_Change")

    plot_df = pd.concat([top_converging, top_diverging]).drop_duplicates()
    plot_df = plot_df.sort_values("CV_Change")

    colors = ["#2ecc71" if x < 0 else "#e74c3c" for x in plot_df["CV_Change"]]

    ax.barh(plot_df["Brand"], plot_df["CV_Change"], color=colors, alpha=0.7)
    ax.set_xlabel("Change in CV (percentage points)")
    ax.set_ylabel("Brand")
    ax.set_title(f"Top {top_n} Brands: Price Convergence/Divergence Across Retailers")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    save_figure("brand_price_convergence.png")
    plt.close()


def plot_promo_convergence_by_category(results_df):
    """Plot promotional strategy convergence."""

    if len(results_df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Premium deal prevalence
    categories = results_df["Category"]
    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(
        x - width / 2, results_df["Pre_Premium_%"], width, label="Pre-COVID", alpha=0.8
    )
    ax1.bar(
        x + width / 2,
        results_df["Post_Premium_%"],
        width,
        label="Post-COVID",
        alpha=0.8,
    )
    ax1.set_xlabel("Product Category")
    ax1.set_ylabel("Premium Deals (%)")
    ax1.set_title("Premium Deal Prevalence by Category")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Retailer strategy convergence
    colors = [
        "#2ecc71" if x < 0 else "#e74c3c" for x in results_df["Retailer_Std_Change"]
    ]
    ax2.barh(categories, results_df["Retailer_Std_Change"], color=colors, alpha=0.7)
    ax2.set_xlabel("Change in Retailer Strategy Variance")
    ax2.set_ylabel("Product Category")
    ax2.set_title(
        "Promotional Strategy Convergence by Category\n(Negative = More Similar)"
    )
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)

    save_figure("promo_convergence_by_category.png")
    plt.close()


def plot_price_distributions_by_category(df_pre, df_post, category="Turkey"):
    """Plot price distributions for a specific category."""

    df_pre = add_product_category(df_pre)
    df_post = add_product_category(df_post)

    pre_cat = df_pre[
        (df_pre["Product_Category"] == category) & (df_pre["Price_Per_Unit"].notna())
    ]
    post_cat = df_post[
        (df_post["Product_Category"] == category) & (df_post["Price_Per_Unit"].notna())
    ]

    if len(pre_cat) < 5 or len(post_cat) < 5:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot by retailer
    retailers = sorted(
        set(pre_cat["Retailer"].unique()) | set(post_cat["Retailer"].unique())
    )

    pre_data = [
        pre_cat[pre_cat["Retailer"] == r]["Price_Per_Unit"].values for r in retailers
    ]
    post_data = [
        post_cat[post_cat["Retailer"] == r]["Price_Per_Unit"].values for r in retailers
    ]

    positions_pre = np.arange(len(retailers)) * 2
    positions_post = positions_pre + 0.8

    bp1 = ax1.boxplot(
        pre_data,
        positions=positions_pre,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        label="Pre-COVID",
    )
    bp2 = ax1.boxplot(
        post_data,
        positions=positions_post,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral", alpha=0.7),
        label="Post-COVID",
    )

    ax1.set_xticks(positions_pre + 0.4)
    ax1.set_xticklabels(retailers, rotation=45, ha="right")
    ax1.set_ylabel("Price per Unit ($)")
    ax1.set_title(f"{category} Price Distribution by Retailer")
    ax1.legend(
        handles=[bp1["boxes"][0], bp2["boxes"][0]], labels=["Pre-COVID", "Post-COVID"]
    )
    ax1.grid(axis="y", alpha=0.3)

    # Histogram overlay
    ax2.hist(
        pre_cat["Price_Per_Unit"],
        bins=20,
        alpha=0.5,
        label="Pre-COVID",
        color="blue",
        density=True,
    )
    ax2.hist(
        post_cat["Price_Per_Unit"],
        bins=20,
        alpha=0.5,
        label="Post-COVID",
        color="red",
        density=True,
    )
    ax2.set_xlabel("Price per Unit ($)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"{category} Price Distribution: Pre vs Post COVID")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    save_figure(f"price_distribution_{category.lower()}.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DEAL CONVERGENCE ANALYSIS")
    print("Analyzing price and promotional strategy convergence")
    print("=" * 70)

    # Setup
    setup_plot_style()

    # Load data
    print("\nLoading data...")
    df = load_combined_data()

    df_pre = filter_by_year_period(df, "pre")
    df_post = filter_by_year_period(df, "post")

    print(f"  Pre-COVID period: {len(df_pre):,} rows (2019-2020)")
    print(f"  Post-COVID period: {len(df_post):,} rows (2025)")

    # Price convergence analyses
    price_by_category = analyze_price_convergence_by_category(df_pre, df_post)
    price_by_brand = analyze_price_convergence_by_brand(df_pre, df_post)

    # Promo convergence analyses
    promo_by_category = analyze_promo_convergence_by_category(df_pre, df_post)
    overall_convergence = analyze_overall_retailer_convergence(df_pre, df_post)

    # Generate visualizations
    print_section_header("GENERATING VISUALIZATIONS")
    plot_price_convergence_by_category(price_by_category)
    plot_brand_price_convergence(price_by_brand)
    plot_promo_convergence_by_category(promo_by_category)

    # Category-specific deep dives
    for category in ["Turkey", "Ham", "Potatoes"]:
        plot_price_distributions_by_category(df_pre, df_post, category)

    # Summary
    print_section_header("SUMMARY")
    print("Analysis complete! Check the figures/ directory for visualizations.")
    print("\nFiles generated:")
    print("  - price_convergence_by_category.png")
    print("  - brand_price_convergence.png")
    print("  - promo_convergence_by_category.png")
    print("  - price_distribution_turkey.png")
    print("  - price_distribution_ham.png")
    print("  - price_distribution_potatoes.png")

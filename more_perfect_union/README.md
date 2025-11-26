# Instacart Thanksgiving Analysis

This analysis examines grocery promotional data around Thanksgiving to answer key questions about deal quality and pricing convergence.

## Project Structure

```
more_perfect_union/
├── instacart_raw_data/          # Raw CSV files by retailer and year
├── processed/                    # Processed/combined data
│   └── combined.csv             # All data combined with standardization
├── figures/                      # Generated visualizations
├── data_aggregator.py           # Combines and standardizes raw data
├── worse_deals.py               # Analyzes if deals have gotten worse
├── deal_convergence.py          # Analyzes price/promo convergence
└── utils.py                     # Shared functions and configurations
```

## Scripts

### 1. data_aggregator.py
Combines retailer-specific CSV files into a single standardized dataset.

**Features:**
- Parses various price formats ($/lb, "2 for $5", BOGO, etc.)
- Creates standardized `Price_Per_Unit` field
- Adds retailer column
- Handles data quality issues (trailing commas, etc.)

**Usage:**
```bash
python data_aggregator.py
```

**Output:** `processed/combined.csv`

### 2. worse_deals.py
Analyzes whether promotional deals have gotten worse for consumers.

**Analysis:**
- Deal type distribution changes (BOGO FREE vs BOGO PERCENT, etc.)
- Deal quality tier analysis (Premium, Strong, Moderate, Conditional, Weak)
- Retailer-level changes in deal quality
- Shift toward conditional deals (digital/membership required)

**Usage:**
```bash
python worse_deals.py
```

**Key Findings:**
- BOGO FREE deals increased by 4.1 percentage points
- Premium deals overall increased by 3.9 percentage points
- Conditional deals increased by 1.7 percentage points
- Significant variation by retailer (Publix +10.5%, HEB -17.5%)

**Visualizations Generated:**
- `deal_type_distribution.png` - Pre/post comparison of deal types
- `deal_quality_tiers.png` - Pie charts showing quality distribution
- `retailer_premium_changes.png` - Premium deal changes by retailer
- `retailer_conditional_changes.png` - Conditional deal changes by retailer
- `deal_type_changes.png` - Waterfall chart of changes

### 3. deal_convergence.py
Analyzes convergence in prices and promotional strategies across retailers and brands.

**Analysis:**
- Price convergence by product category
- Price convergence by brand across retailers
- Promotional strategy convergence
- Overall retailer promotional similarity

**Usage:**
```bash
python deal_convergence.py
```

**Key Findings:**

**Price Convergence:**
- 3 categories show convergence: Turkey (-7.3%), Wine (-8.8%), Soda (-44.3%)
- 5 categories show divergence: Ham (+14.2%), Potatoes (+188%), Cranberries (+30.6%)
- 57.1% of brands show price convergence across retailers

**Promo Convergence:**
- Overall promotional strategies have DIVERGED by 15.1%
- Only Ham shows strategy convergence across retailers
- Most categories show increased diversity in promotional tactics

**Visualizations Generated:**
- `price_convergence_by_category.png` - Price variability changes
- `brand_price_convergence.png` - Brand-level convergence/divergence
- `promo_convergence_by_category.png` - Promotional strategy changes
- `price_distribution_turkey.png` - Turkey price distributions
- `price_distribution_ham.png` - Ham price distributions
- `price_distribution_potatoes.png` - Potato price distributions

## Data

**Time Periods:**
- Pre-COVID: 2019-2020 (2,882 rows)
- Post-COVID: 2025 (2,337 rows)

**Retailers Covered:**
Albertsons, FoodLion, Giant, HEB, Publix, Safeway, Schnucks, Shoprite, Sprouts, StopShop, WinnDixie

**Deal Types:**
- BOGO_FREE: Buy one get one free
- BOGO_PERCENT: Buy one get one X% off
- MULTIBUY: Buy multiple for discount
- STRAIGHT_PRICE: Reduced price
- DIGITAL_ONLY: Digital coupon required
- PERCENT_OFF: Percentage discount
- DOLLAR_OFF: Dollar amount off
- MIX_MATCH: Mix and match deals
- BUNDLE: Bundle pricing
- SPEND_GET: Spend threshold required
- And others...

## Configuration

Global configurations are defined at the top of each script and in `utils.py`:

**In utils.py:**
- `PRODUCT_CATEGORIES`: Keywords for categorizing products
- `DEAL_QUALITY_TIERS`: Tiered groupings of deal types
- `PROMO_HIERARCHY`: Ranking of deals (1=best for consumer)

**Modifiable Settings:**
- Minimum sample size for analysis
- Product categories to analyze
- Visualization styles
- Output directories

## Requirements

- Python 3.x
- pandas
- matplotlib
- numpy
- scipy

## Running the Full Analysis

```bash
# 1. Combine and standardize data
python data_aggregator.py

# 2. Analyze deal quality changes
python worse_deals.py

# 3. Analyze convergence
python deal_convergence.py
```

All visualizations will be saved to `figures/` directory.

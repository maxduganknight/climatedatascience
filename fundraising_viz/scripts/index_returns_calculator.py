from pathlib import Path

import pandas as pd

# Read the CSV file
data_path = (
    Path(__file__).parent.parent
    / "data"
    / "adaptation_market_growth"
    / "ai_saas_index_returns.csv"
)
df = pd.read_csv(data_path)

# Convert month to datetime and sort by date
df["month"] = pd.to_datetime(df["month"], format="%d-%b-%y")
df = df.sort_values("month")

# Get the most recent price (latest date)
latest_date = df["month"].max()
latest_igv = df[df["month"] == latest_date]["igv"].values[0]
latest_aiq = df[df["month"] == latest_date]["aiq"].values[0]


# Calculate returns for different time periods
def calculate_return(current_price, past_price):
    """Calculate percentage return"""
    return ((current_price - past_price) / past_price) * 100


# Find prices at different time periods
months_back = [6, 12, 24, 36, 60]
results = []

for months in months_back:
    target_date = latest_date - pd.DateOffset(months=months)
    # Find the closest date to the target (in case exact date doesn't exist)
    closest_idx = (df["month"] - target_date).abs().idxmin()
    past_row = df.loc[closest_idx]

    igv_return = calculate_return(latest_igv, past_row["igv"])
    aiq_return = calculate_return(latest_aiq, past_row["aiq"])

    period_name = f"{months // 12}-year" if months >= 12 else f"{months}-month"

    results.append(
        {
            "Period": period_name,
            "IGV Return (%)": f"{igv_return:.2f}",
            "AIQ Return (%)": f"{aiq_return:.2f}",
        }
    )

# Create and print results table
results_df = pd.DataFrame(results)
print(f"\nReturns as of {latest_date.strftime('%B %Y')}")
print(f"Current prices - IGV: ${latest_igv:.2f}, AIQ: ${latest_aiq:.2f}\n")
print(results_df.to_string(index=False))
